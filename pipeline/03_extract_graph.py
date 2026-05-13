"""
03_extract_graph.py — Extract drug relationships -> Neo4j graph

Reads:  extracted_data/chunks/*.json
Uses:   DeepSeek V4 Pro API to extract structured relationships per drug
Builds: Neo4j graph database (bolt://localhost:7687)

Graph schema:
  (Drug)-[:TREATS]->(Condition)
  (Drug)-[:CONTRAINDICATED_FOR]->(Contraindication)
  (Drug)-[:CAUSES]->(SideEffect)
  (Drug)-[:BELONGS_TO]->(DrugClass)
  (Drug)-[:INTERACTS_WITH]->(Interaction)
"""

import os
import re
import json
import glob
import time
from pathlib import Path
from tqdm import tqdm
from neo4j import GraphDatabase
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEEPSEEK_GRAPH_API_KEY, DEEPSEEK_GRAPH_BASE_URL, DEEPSEEK_GRAPH_MODEL_ID,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CHUNKS_DIR, STRIP_THINKING
)

LOG_FILE        = r".\extracted_data\graph_log.txt"
CHECKPOINT_FILE = r".\extracted_data\graph_checkpoint.json"

# ---------------------------------------------------------------------------
# Context window budget (32K total):
#   - System prompt : ~300 tokens
#   - Drug text     : ~9,600 tokens (24,000 chars ÷ 2.5 chars/token)
#   - Output JSON   : 1,024 tokens
#   - Safety buffer : ~21,844 tokens remaining
# ---------------------------------------------------------------------------
GRAPH_N_CTX      = 32768
GRAPH_MAX_TOKENS = 4096       # JSON output can be large for DeepSeek
DRUG_TEXT_LIMIT  = 24_000     # characters sent to model per drug
MAX_RETRIES      = 4          # retry count on JSON parse failure

EXTRACT_PROMPT = """\
You are a pharmaceutical expert. Extract the following drug information as JSON.
Return ONLY valid JSON. No other text. No markdown code blocks.

⚠️ CRITICAL: Your response must be valid JSON parseable by json.loads().
Do NOT add any text before or after the JSON. No markdown. No explanations.

กฎสำคัญสำหรับ indications, contraindications, และ side_effects:
- แต่ละรายการต้องเป็นคีย์เวิร์ดสั้น 1-4 คำ เช่น "ปวดหัว", "headache", "ไข้", "fever"
- ห้ามใช้ประโยคยาว ห้ามใช้ "เช่น", "ได้แก่", หรือคำอธิบาย
- ให้มีทั้งภาษาไทยและภาษาอังกฤษ และชื่อพ้องที่ใช้บ่อย เช่น "ปวดหัว" และ "ปวดศีรษะ" อยู่ในรายการเดียวกัน
- ห้ามใส่วงเล็บ () หรือหมายเหตุใดๆ ให้เป็นคีย์เวิร์ดสะอาดเท่านั้น

Example:
{
  "drug_name": "Paracetamol",
  "drug_class": "Analgesics",
  "indications": ["ปวดหัว", "ปวดศีรษะ", "headache", "ไข้", "fever", "ปวดฟัน", "toothache"],
  "contraindications": ["โรคตับ", "liver disease", "แพ้ยา", "drug allergy"],
  "side_effects": ["คลื่นไส้", "nausea", "ปวดท้อง", "stomach pain"],
  "interactions": ["warfarin", "alcohol"]
}
"""


# ---------------------------------------------------------------------------
# Model + DB init
# ---------------------------------------------------------------------------

def get_deepseek_client():
    """Initialize DeepSeek API client for graph extraction."""
    print(f"Initializing DeepSeek client: {DEEPSEEK_GRAPH_MODEL_ID}")
    client = OpenAI(
        api_key=DEEPSEEK_GRAPH_API_KEY,
        base_url=DEEPSEEK_GRAPH_BASE_URL
    )
    print("DeepSeek client initialized.")
    return client


def get_neo4j_driver():
    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Neo4j connection OK.")
    return driver


def create_indexes(driver):
    """Create indexes for faster Cypher queries."""
    indexes = [
        "CREATE INDEX drug_pill_name IF NOT EXISTS FOR (d:Drug) ON (d.pill_name)",
        "CREATE INDEX condition_name  IF NOT EXISTS FOR (c:Condition) ON (c.name)",
        "CREATE INDEX contraind_name  IF NOT EXISTS FOR (c:Contraindication) ON (c.name)",
    ]
    with driver.session() as session:
        for idx in indexes:
            session.run(idx)
    print("Neo4j indexes created.")


def clear_neo4j(driver):
    """Drop all nodes and relationships for a clean rebuild."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Neo4j database cleared.")


# ---------------------------------------------------------------------------
# Cleaning helper
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    # Strip complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip unclosed <think>... to end of string
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # Strip "Thinking Process:" or "Reasoning:" header blocks
    text = re.sub(r"^(?:Thinking Process|Reasoning):.*?\n\n", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(client: OpenAI, drug_text: str) -> dict | None:
    """Call DeepSeek API to extract structured entities, with retry on JSON parse failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_GRAPH_MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": EXTRACT_PROMPT + "\nIMPORTANT: Return ONLY the JSON object. Do not explain. Do not think out loud.",
                    },
                    {"role": "user", "content": f"ข้อมูลยา:\n\n{drug_text[:DRUG_TEXT_LIMIT]}"},
                ],
                max_tokens=GRAPH_MAX_TOKENS,
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            if not raw:
                continue
            if STRIP_THINKING:
                raw = strip_thinking(raw)
            raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"```$",          "", raw.strip(), flags=re.MULTILINE)
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            print(f"\n  ⚠️ JSON parse failed (attempt {attempt}/{MAX_RETRIES})")
            print(f"     Raw response (first 200 chars): {raw[:200]}...")
            print(f"     Error: {e}")
            time.sleep(0.5)
        except Exception as e:
            print(f"\n  ⚠️ API/Other Error (attempt {attempt}/{MAX_RETRIES})")
            print(f"     Error type: {type(e).__name__}")
            print(f"     Error message: {e}")
            return None
    return None


# ---------------------------------------------------------------------------
# Neo4j write helpers
# ---------------------------------------------------------------------------

def merge_drug_to_neo4j(session, entities: dict, pill_name: str):
    """Write a drug and all its relationships into Neo4j using MERGE."""
    drug_name = entities.get("drug_name", pill_name)

    # Create or update the Drug node
    session.run(
        "MERGE (d:Drug {pill_name: $pill_name}) SET d.name = $name",
        pill_name=pill_name, name=drug_name,
    )

    if entities.get("drug_class"):
        session.run(
            """
            MERGE (d:Drug {pill_name: $pill_name})
            MERGE (c:DrugClass {name: $cls})
            MERGE (d)-[:BELONGS_TO]->(c)
            """,
            pill_name=pill_name, cls=entities["drug_class"],
        )

    for item in entities.get("indications", []):
        session.run(
            """
            MERGE (d:Drug {pill_name: $pill_name})
            MERGE (c:Condition {name: $name})
            MERGE (d)-[:TREATS]->(c)
            """,
            pill_name=pill_name, name=item,
        )

    for item in entities.get("contraindications", []):
        session.run(
            """
            MERGE (d:Drug {pill_name: $pill_name})
            MERGE (ci:Contraindication {name: $name})
            MERGE (d)-[:CONTRAINDICATED_FOR]->(ci)
            """,
            pill_name=pill_name, name=item,
        )

    for item in entities.get("side_effects", []):
        session.run(
            """
            MERGE (d:Drug {pill_name: $pill_name})
            MERGE (s:SideEffect {name: $name})
            MERGE (d)-[:CAUSES]->(s)
            """,
            pill_name=pill_name, name=item,
        )

    for item in entities.get("interactions", []):
        session.run(
            """
            MERGE (d:Drug {pill_name: $pill_name})
            MERGE (i:Interaction {name: $name})
            MERGE (d)-[:INTERACTS_WITH]->(i)
            """,
            pill_name=pill_name, name=item,
        )


def merge_drug_node_only(session, pill_name: str):
    """Fallback: create a bare Drug node with no relationships."""
    session.run(
        "MERGE (d:Drug {pill_name: $pill_name}) SET d.name = $pill_name",
        pill_name=pill_name,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    """Load set of already-processed pill names from checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_checkpoint(done: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done), f)


def build_graph(recreate: bool = True):
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "*.json")))
    if not chunk_files:
        print(f"No chunk files found in {CHUNKS_DIR}")
        return

    client = get_deepseek_client()
    driver = get_neo4j_driver()

    if recreate:
        clear_neo4j(driver)

    create_indexes(driver)

    # Load checkpoint — skip already-processed drugs if resuming after a crash
    done = set() if recreate else load_checkpoint()
    skipped = len(done)
    if skipped:
        print(f"Resuming from checkpoint — skipping {skipped} already-processed drugs.")

    with open(LOG_FILE, "a" if done else "w", encoding="utf-8") as log:
        if not done:
            log.write("status   | drug\n" + "-" * 40 + "\n")

    ok_count       = 0
    fallback_count = 0

    dry_run_count = 0

    for chunk_file in tqdm(chunk_files, desc="Building graph"):
        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            continue

        pill_name = chunks[0]["metadata"].get("pill_name", Path(chunk_file).stem)

        # Dry-run mode: stop after processing N drugs
        if DRY_RUN_LIMIT is not None and dry_run_count >= DRY_RUN_LIMIT:
            print(f"\n[Dry-run] Stopped after {DRY_RUN_LIMIT} drugs.")
            break
        dry_run_count += 1

        # Skip if already processed in a previous run
        if pill_name in done:
            continue

        drug_text = "\n\n".join(c.get("text", "") for c in chunks)

        entities = extract_entities(client, drug_text)

        with driver.session() as session:
            if entities:
                merge_drug_to_neo4j(session, entities, pill_name)
                status = "OK"
                ok_count += 1
            else:
                merge_drug_node_only(session, pill_name)
                status = "FALLBACK"
                fallback_count += 1

        done.add(pill_name)
        save_checkpoint(done)

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"{status:<8} | {pill_name}\n")

    driver.close()

    print(f"\n--- Done ---")
    print(f"OK: {ok_count}  |  Fallback (no relationships): {fallback_count}")
    print(f"Neo4j: {NEO4J_URI}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-recreate", action="store_true",
                        help="Skip clearing the Neo4j database before building (additive mode)")
    parser.add_argument("--dry-run", nargs="?", type=int, const=5, default=None,
                        help="Test mode: process only N drugs (default: 5). Shows detailed logs.")
    args = parser.parse_args()
    DRY_RUN_LIMIT = args.dry_run
    build_graph(recreate=not args.no_recreate)
