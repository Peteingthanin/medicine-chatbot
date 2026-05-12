"""
retrieval/retriever_hybrid.py — Phase 2: Graph-first + Vector retrieval

Flow:
  query
    -> LLM extracts intent / symptoms / drug keywords
    -> Neo4j Cypher: find candidate drugs (TREATS edges matching symptoms)
    -> Neo4j Cypher: filter by user contraindications
    -> Vector retrieval: get detail chunks for each candidate drug
    -> Return candidates + ranked chunks
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from llama_cpp import Llama
from qdrant_client import QdrantClient

from ai.config.db  import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_COLLECTION
from ai.config.llm import DEEPSEEK_MODEL_ID, OPENROUTER_MODEL_ID
from ai.config.app import STRIP_THINKING
from ai.retrieval.retriever_vector import VectorRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HybridRetrievalResult:
    candidate_drugs:    list[str]
    filtered_out:       list[str]    # drugs removed by contraindication filter
    chunks:             list[RetrievedChunk]
    pipeline:           str = "hybrid"
    retrieval_ms:       float = 0.0
    graph_ms:           float = 0.0
    vector_ms:          float = 0.0
    query:              str = ""
    extracted_symptoms: list[str] = field(default_factory=list)
    intent:             str = "general_knowledge"


# ---------------------------------------------------------------------------
# Intent / entity extraction prompt
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """\
คุณคือ AI จำแนกเจตนาของผู้ใช้ (Intent Classifier)
อ่านคำถามและคืนค่า JSON ที่มีโครงสร้างดังนี้:
1. "intent": เลือกจาก ["symptom_search", "drug_inquiry", "general_knowledge"]
   - symptom_search: ผู้ใช้บอกอาการป่วยและต้องการหายา (เช่น "ปวดหัว", "มีไข้")
   - drug_inquiry: ผู้ใช้ถามเกี่ยวกับยาที่ระบุชื่อชัดเจน (เช่น "พาราเซตามอลมีผลข้างเคียงไหม", "กิน ibuprofen ได้ไหม")
   - general_knowledge: คำถามทั่วไปที่ไม่ระบุยาหรืออาการเฉพาะเจาะจง
2. "symptoms": รายการอาการที่พบ (สร้างคำพ้องความหมาย 3-5 คำ)
3. "drugs": รายการชื่อยาที่พบในคำถาม (ถ้ามี)

--- ตัวอย่าง (EXAMPLES) ---
User: "ผมปวดศรีษะ มียาอะไรช่วยได้บ้าง"
Assistant: {"intent": "symptom_search", "symptoms": ["ปวดศรีษะ", "ปวดศีรษะ", "ปวดหัว", "headache", "migraine", "ไมเกรน"], "drugs": []}

User: "Paracetamol มีข้อระวังอะไรบ้าง"
Assistant: {"intent": "drug_inquiry", "symptoms": [], "drugs": ["Paracetamol", "พาราเซตามอล"]}

User: "ควรเก็บรักษายาอย่างไร"
Assistant: {"intent": "general_knowledge", "symptoms": [], "drugs": []}
--- จบตัวอย่าง ---

ส่งคืนเป็น JSON เท่านั้น ห้ามมีข้อความอธิบาย
"""


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        embed_model: Llama,
        chat_model: Optional[Llama],
        qdrant: QdrantClient,
        deepseek_client=None,
        openai_client=None,
    ):
        self.vector_retriever = VectorRetriever(embed_model, qdrant)
        self.chat_model       = chat_model
        self.deepseek_client  = deepseek_client
        self.openai_client    = openai_client
        self.neo4j_driver     = None
        self._connect_neo4j()

    def _connect_neo4j(self):
        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
                )
                self.neo4j_driver.verify_connectivity()
                with self.neo4j_driver.session() as session:
                    nodes = session.run("MATCH (n) RETURN count(n) as nodes").single()["nodes"]
                    rels  = session.run("MATCH ()-[r]->() RETURN count(r) as rels").single()["rels"]
                print(f"Neo4j connected: {nodes} nodes, {rels} relationships")
                return
            except (ServiceUnavailable, OSError) as e:
                if attempt < max_retries:
                    wait = attempt * 5
                    print(f"Neo4j not ready (attempt {attempt}/{max_retries}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Warning: Neo4j connection failed after {max_retries} attempts ({e}). Phase 2 graph queries will fall back to vector-only.")
                    self.neo4j_driver = None
            except Exception as e:
                print(f"Warning: Neo4j connection failed ({e}). Phase 2 graph queries will fall back to vector-only.")
                self.neo4j_driver = None
                return

    async def _extract_intent_and_entities(self, query: str, model_choice: str) -> dict:
        """Use LLM to extract intent, symptoms, and drug keywords from the user query."""
        print("  [Phase 2: Graph] Extracting intent and entities from query...")
        default_res = {"intent": "general_knowledge", "symptoms": [], "drugs": []}
        try:
            messages = [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": query},
            ]

            from ai.llm.generate import strip_thinking
            
            if model_choice in ("deepseek", "cloud") and self.deepseek_client is not None:
                response = await self.deepseek_client.chat.completions.create(
                    model=DEEPSEEK_MODEL_ID, messages=messages,
                    max_tokens=4096, temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
            elif model_choice == "minimax" and self.openai_client is not None:
                response = await self.openai_client.chat.completions.create(
                    model=OPENROUTER_MODEL_ID, messages=messages,
                    max_tokens=4096, temperature=0.1,
                )
                raw = response.choices[0].message.content
            elif self.chat_model is not None:
                response = self.chat_model.create_chat_completion(
                    messages=messages, max_tokens=4096, temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = response["choices"][0]["message"]["content"]
            else:
                return default_res

            if STRIP_THINKING:
                raw = strip_thinking(raw)

            if not raw:
                return default_res

            raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"```$",          "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(raw.strip())

            intent       = data.get("intent", "general_knowledge")
            symptoms_list = data.get("symptoms", [])
            drugs_list    = data.get("drugs", [])

            symptoms_list = [re.sub(r"\s*\([^)]*\)", "", s).lower().strip() for s in symptoms_list]
            symptoms_list = [s for s in dict.fromkeys(symptoms_list) if s]

            drugs_list = [re.sub(r"\s*\([^)]*\)", "", d).lower().strip() for d in drugs_list]
            drugs_list = [d for d in dict.fromkeys(drugs_list) if d]

            if intent not in ["symptom_search", "drug_inquiry", "general_knowledge"]:
                intent = "general_knowledge"

            # If the LLM correctly identified a symptom_search or drug_inquiry
            # but returned empty entities, fall back to the raw query words
            # rather than silently downgrading the intent.
            if intent == "symptom_search" and not symptoms_list:
                # Extract non-trivial words from the query as fallback symptom keywords
                stopwords = {"ผม", "ฉัน", "คุณ", "มี", "เป็น", "อาการ", "ยา", "ที่", "ได้", "จะ", "และ", "หรือ", "ว่า", "ก็", "ใน", "ของ", "ให้", "นี้"}
                fallback = [w.strip() for w in re.split(r"[\s,]+", query) if w.strip() and w.strip() not in stopwords and len(w.strip()) > 1]
                if fallback:
                    symptoms_list = fallback
                    print(f"  [Phase 2: Graph] ⚠️ LLM returned no symptoms for symptom_search — using query words as fallback: {symptoms_list}")

            if intent == "drug_inquiry" and not drugs_list and not symptoms_list:
                # If drug inquiry but nothing extracted, downgrade to general
                intent = "general_knowledge"

            res = {"intent": intent, "symptoms": symptoms_list, "drugs": drugs_list}
            print(f"  [Phase 2: Graph] Extracted: {res}")
            return res

        except Exception as e:
            print(f"  [Phase 2: Graph] ⚠️ Extraction failed ({e}), falling back to general_knowledge.")
            return default_res

    def _find_candidates(self, symptoms: list[str]) -> list[str]:
        """Cypher query: find Drug nodes whose TREATS edges match any symptom keyword."""
        if not self.neo4j_driver or not symptoms:
            return []
        with self.neo4j_driver.session() as session:
            result = session.run(
                """
                UNWIND $symptoms AS symptom
                MATCH (d:Drug)-[:TREATS]->(c:Condition)
                WHERE toLower(c.name) CONTAINS toLower(symptom)
                   OR toLower(symptom) CONTAINS toLower(c.name)
                RETURN d.pill_name AS pill_name, count(*) AS score
                ORDER BY score DESC
                """,
                symptoms=symptoms,
            )
            return [record["pill_name"] for record in result if record["pill_name"]]

    def _filter_contraindications(
        self, candidates: list[str], user_contraindications: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Cypher query: remove drugs that are contraindicated for the user's conditions.
        Returns (safe_drugs, filtered_out_drugs).
        """
        if not user_contraindications or not self.neo4j_driver:
            return candidates, []
        with self.neo4j_driver.session() as session:
            result = session.run(
                """
                UNWIND $candidates AS pill
                UNWIND $contraindications AS user_ci
                MATCH (d:Drug {pill_name: pill})-[:CONTRAINDICATED_FOR]->(ci:Contraindication)
                WHERE toLower(ci.name) CONTAINS toLower(user_ci)
                   OR toLower(user_ci) CONTAINS toLower(ci.name)
                RETURN DISTINCT d.pill_name AS pill_name
                """,
                candidates=candidates,
                contraindications=user_contraindications,
            )
            unsafe = {record["pill_name"] for record in result}
        safe    = [p for p in candidates if p not in unsafe]
        removed = [p for p in candidates if p in unsafe]
        return safe, removed

    async def retrieve(
        self,
        query: str,
        user_contraindications: Optional[list[str]] = None,
        top_k_per_drug: int = 3,
        max_candidates: int = 5,
        model_choice: str = "local",
        pre_extracted_data: Optional[dict] = None,
    ) -> HybridRetrievalResult:
        t_total = time.time()

        extracted = pre_extracted_data or await self._extract_intent_and_entities(query, model_choice)
        intent   = extracted["intent"]
        symptoms = extracted["symptoms"]
        drugs    = extracted["drugs"]

        candidates = []
        all_chunks = []
        safe_candidates = []
        filtered_out = []
        graph_ms = 0.0
        vector_ms = 0.0

        # --- Path C: General Knowledge (no graph) ---
        if intent == "general_knowledge":
            print("  [Phase 2: Vector] General knowledge intent. Skipping graph...")
            t_vec = time.time()
            query_vector = self.vector_retriever.embed_query(query)
            res = self.vector_retriever.retrieve_by_vector(
                vector=query_vector, query=query,
                top_k=top_k_per_drug * 2,
            )
            all_chunks = res.chunks
            vector_ms = (time.time() - t_vec) * 1000

        # --- Path B: Drug Inquiry ---
        elif intent == "drug_inquiry" and drugs:
            print(f"  [Phase 2: Graph] Drug inquiry intent. Using drugs: {drugs}")
            t_graph = time.time()
            all_known_drugs = self.vector_retriever.get_all_drugs()
            for d in drugs:
                for k in all_known_drugs:
                    if d.lower() in k.lower() or k.lower() in d.lower():
                        if k not in candidates:
                            candidates.append(k)
            if not candidates:
                print("  [Phase 2: Graph] ⚠️ Extracted drugs not found in DB. Falling back to symptom search.")
                intent = "symptom_search"
            else:
                safe_candidates, filtered_out = self._filter_contraindications(candidates, user_contraindications or [])
                graph_ms = (time.time() - t_graph) * 1000
                print(f"  [Phase 2: Vector] Searching vector DB for {len(safe_candidates)} matched drugs...")
                t_vec = time.time()
                query_vector = self.vector_retriever.embed_query(query)
                for pill in safe_candidates:
                    res = self.vector_retriever.retrieve_by_vector(
                        vector=query_vector, query=query,
                        top_k=top_k_per_drug, pill_name_filter=pill,
                    )
                    all_chunks.extend(res.chunks)
                all_chunks.sort(key=lambda c: c.score, reverse=True)
                vector_ms = (time.time() - t_vec) * 1000

        # --- Path A: Symptom Search (or fallback from drug inquiry) ---
        if intent == "symptom_search":
            t_graph = time.time()
            print("  [Phase 2: Graph] Symptom search intent. Finding candidate drugs...")
            candidates = self._find_candidates(symptoms)[:max_candidates]
            if not candidates:
                print("  [Phase 2: Graph] ⚠️ No candidates found. Falling back to all drugs.")
                candidates = self.vector_retriever.get_all_drugs()[:max_candidates]
            safe_candidates, filtered_out = self._filter_contraindications(candidates, user_contraindications or [])
            graph_ms = (time.time() - t_graph) * 1000
            print(f"  [Phase 2: Graph] Safe candidates: {safe_candidates}. Filtered out: {filtered_out}.")
            print(f"  [Phase 2: Vector] Searching vector DB for {len(safe_candidates)} candidates...")
            t_vec = time.time()
            query_vector = self.vector_retriever.embed_query(query)
            for pill in safe_candidates:
                res = self.vector_retriever.retrieve_by_vector(
                    vector=query_vector, query=query,
                    top_k=top_k_per_drug, pill_name_filter=pill,
                )
                all_chunks.extend(res.chunks)
            all_chunks.sort(key=lambda c: c.score, reverse=True)
            vector_ms = (time.time() - t_vec) * 1000

        total_ms = (time.time() - t_total) * 1000

        return HybridRetrievalResult(
            candidate_drugs=safe_candidates,
            filtered_out=filtered_out,
            chunks=all_chunks,
            pipeline="hybrid",
            retrieval_ms=round(total_ms, 1),
            graph_ms=round(graph_ms, 1),
            vector_ms=round(vector_ms, 1),
            query=query,
            extracted_symptoms=symptoms,
            intent=intent,
        )
