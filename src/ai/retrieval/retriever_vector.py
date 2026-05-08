"""
retrieval/retriever_vector.py — Phase 1: Pure vector retrieval

Flow:
  query -> embed with instruction prefix -> Qdrant top-k -> group by pill -> return
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from ai.config.llm import EMBED_QUERY_PREFIX
from ai.config.db  import QDRANT_PATH, QDRANT_COLLECTION


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    pill_name:   str
    section:     str
    subsection:  Optional[str]
    topic:       Optional[str]
    subtopic:    Optional[str]
    detail:      list
    text:        str
    score:       float
    source_file: str


@dataclass
class VectorRetrievalResult:
    chunks:       list[RetrievedChunk]
    pipeline:     str = "vector"
    retrieval_ms: float = 0.0
    query:        str = ""


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class VectorRetriever:
    def __init__(self, embed_model: Llama, qdrant: QdrantClient):
        self.embed_model = embed_model
        self.qdrant      = qdrant

    def embed_query(self, query: str) -> list:
        """Embed a user query with the instruction prefix for better retrieval."""
        prefixed = f"{EMBED_QUERY_PREFIX}{query}"
        response = self.embed_model.create_embedding(prefixed)
        return response["data"][0]["embedding"]

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        pill_name_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> VectorRetrievalResult:
        """Embed the query and search. Convenience wrapper for single-call use (Phase 1)."""
        vector = self.embed_query(query)
        return self.retrieve_by_vector(
            vector=vector, query=query, top_k=top_k,
            pill_name_filter=pill_name_filter, section_filter=section_filter,
        )

    def retrieve_by_vector(
        self,
        vector: list,
        query: str = "",
        top_k: int = 8,
        pill_name_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> VectorRetrievalResult:
        """Search Qdrant using a pre-computed query vector (avoids redundant embedding)."""
        t0 = time.time()

        conditions = []
        if pill_name_filter:
            conditions.append(FieldCondition(key="pill_name",
                                             match=MatchValue(value=pill_name_filter)))
        if section_filter:
            conditions.append(FieldCondition(key="section",
                                             match=MatchValue(value=section_filter)))

        qdrant_filter = Filter(must=conditions) if conditions else None

        response = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits = response.points

        chunks = []
        for hit in hits:
            p = hit.payload
            chunks.append(RetrievedChunk(
                pill_name=p.get("pill_name", ""),
                section=p.get("section", ""),
                subsection=p.get("subsection"),
                topic=p.get("topic"),
                subtopic=p.get("subtopic"),
                detail=p.get("detail", []),
                text=p.get("text", ""),
                score=hit.score,
                source_file=p.get("source_file", ""),
            ))

        elapsed = (time.time() - t0) * 1000

        return VectorRetrievalResult(
            chunks=chunks,
            pipeline="vector",
            retrieval_ms=round(elapsed, 1),
            query=query,
        )

    def get_all_drugs(self) -> list[str]:
        """Return distinct pill_names in the collection."""
        result = self.qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=True,
            limit=1000,
        )
        drugs = set()
        for point in result[0]:
            name = point.payload.get("pill_name")
            if name:
                drugs.add(name)
        return sorted(drugs)
