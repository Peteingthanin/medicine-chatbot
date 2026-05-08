"""
config/db.py — Database connection settings (Qdrant vector store + Neo4j graph store)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Vector store — Qdrant
# ---------------------------------------------------------------------------
QDRANT_PATH       = os.getenv("QDRANT_PATH", r".\vector_db")
QDRANT_COLLECTION = "medication_chunks"

# ---------------------------------------------------------------------------
# Graph store — Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Legacy pickle path — kept for backward compat, no longer used at runtime
GRAPH_PATH = r".\graph_db\medication_graph.pkl"
