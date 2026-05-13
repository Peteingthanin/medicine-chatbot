"""
06_repair_conditions.py — Atomize long-sentence Condition nodes into atomic keywords

ONE-TIME script. Does NOT re-read any markdown files.
Steps:
  1. Query all Condition nodes from Neo4j
  2. For each long-sentence condition, use the 9B LLM to split into atomic bilingual keywords
  3. Delete the old Condition node
  4. Create new atomic Condition nodes + re-link all TREATS edges
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llama_cpp import Llama
from neo4j import GraphDatabase

from config import GRAPH_MODEL_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ATOMIZE_PROMPT = """\
คุณคือผู้เชี่ยวชาญด้านเภสัชกรรม
แยกอาการ/โรคต่อไปนี้ออกเป็นคีย์เวิร์ดสั้นๆ แต่ละอาการ 1-4 คำ
ให้มีทั้งภาษาไทยและภาษาอังกฤษ รวมชื่อพ้องที่ใช้บ่อย เช่น "ปวดหัว" และ "ปวดศีรษะ"
ส่งคืนเป็น JSON เท่านั้น ห้ามมีข้อความอื่น

รูปแบบ: {"keywords": ["ปวดหัว", "ปวดศีรษะ", "headache", "ไข้", "fever"]}
"""

# Conditions shorter than this are considered already atomic — skip LLM call
SHORT_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_sentence(text: str) -> bool:
    """Return True if this looks like a long sentence, not an atomic keyword."""
    return len(text) > SHORT_THRESHOLD or "เช่น" in text or text.count(",") >= 1


def atomize_condition(model: Llama, condition: str) -> list[str]:
    """Use 9B LLM to split a condition sentence into atomic bilingual keywords."""
    messages = [
        {"role": "system", "content": ATOMIZE_PROMPT},
        {"role": "user",   "content": f"อาการ: {condition}"},
    ]
    try:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response["choices"][0]["message"]["content"]
        data = json.loads(raw.strip())
        keywords = [k.strip() for k in data.get("keywords", []) if k.strip()]
        return keywords if keywords else [condition]
    except Exception as e:
        print(f"    ⚠️  LLM failed for '{condition[:40]}': {e}")
        return [condition]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("06_repair_conditions.py -- Graph Condition Atomizer")
    print("=" * 60)

    # Load 9B model (used for graph work)
    print("\n[1/4] Loading 9B model...")
    model = Llama(
        model_path=GRAPH_MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("      OK Model loaded.")

    # Connect to Neo4j
    print("\n[2/4] Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("      OK Connected.")

    # Fetch all Condition nodes
    print("\n[3/4] Fetching Condition nodes...")
    with driver.session() as session:
        result = session.run("MATCH (c:Condition) RETURN c.name AS name ORDER BY name")
        conditions = [r["name"] for r in result]
    print(f"      Found {len(conditions)} condition nodes.")

    # Atomize sentences
    print("\n[4/4] Atomizing long-sentence conditions...")
    repaired = 0
    skipped  = 0

    for old_name in conditions:
        if not is_sentence(old_name):
            skipped += 1
            continue

        # Find all drugs connected to this condition
        with driver.session() as session:
            result = session.run(
                "MATCH (d:Drug)-[:TREATS]->(c:Condition {name: $name}) RETURN d.pill_name AS pill",
                name=old_name,
            )
            connected_drugs = [r["pill"] for r in result]

        if not connected_drugs:
            # Orphan condition -- just delete it
            with driver.session() as session:
                session.run("MATCH (c:Condition {name: $name}) DETACH DELETE c", name=old_name)
            print(f"  [DEL] Deleted orphan: '{old_name[:60]}'")
            continue

        # Ask LLM to atomize
        keywords = atomize_condition(model, old_name)
        safe_name = old_name[:50].encode("ascii", "replace").decode("ascii")
        print(f"  [FIX] '{safe_name}' -> {len(keywords)} keywords")

        # Replace in Neo4j (delete old, create new atomic nodes)
        with driver.session() as session:
            # Delete old condition node + its relationships
            session.run(
                "MATCH (c:Condition {name: $name}) DETACH DELETE c",
                name=old_name,
            )
            # Create new atomic nodes and re-link to all connected drugs
            for kw in keywords:
                for pill in connected_drugs:
                    session.run(
                        """
                        MERGE (d:Drug {pill_name: $pill})
                        MERGE (c:Condition {name: $kw})
                        MERGE (d)-[:TREATS]->(c)
                        """,
                        pill=pill, kw=kw,
                    )
        repaired += 1

    driver.close()

    print("\n" + "=" * 60)
    print(f"DONE!  Atomized: {repaired}  |  Already atomic (skipped): {skipped}")
    print("=" * 60)
    print("\nNext step: Restart uvicorn and test a query like 'paracetamol headache'")


if __name__ == "__main__":
    main()
