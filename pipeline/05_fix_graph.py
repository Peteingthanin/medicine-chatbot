"""
05_fix_graph.py — One-time cleanup of the Neo4j graph

Fixes:
  1. Merge case-sensitivity duplicates (e.g., "Bipolar Disorder" vs "Bipolar disorder")
     across all node types (Condition, Contraindication, SideEffect, Interaction, DrugClass)
  2. Trim leading/trailing whitespace from all node names
"""

import sys
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

NODE_TYPES = ["Condition", "Contraindication", "SideEffect", "Interaction", "DrugClass"]


def merge_case_duplicates(session, label: str) -> int:
    """
    For a given node label, find all nodes that have the same name when lowercased.
    Keep the first alphabetically and re-point all relationships to it, then delete duplicates.
    Returns the number of merges performed.
    """
    # Find groups of nodes with same lowercased, trimmed name
    result = session.run(f"""
        MATCH (n:{label})
        WITH toLower(trim(n.name)) AS norm, collect(n) AS nodes
        WHERE size(nodes) > 1
        RETURN norm, nodes
    """)

    total_merged = 0
    for record in result:
        nodes = record["nodes"]
        # Keep the first one (shortest name, or alphabetically first)
        nodes_sorted = sorted(nodes, key=lambda n: n["name"])
        keeper  = nodes_sorted[0]
        to_merge = nodes_sorted[1:]

        for dup in to_merge:
            # Re-point all incoming relationships to keeper
            session.run(f"""
                MATCH (src)-[r]->(dup:{label})
                WHERE elementId(dup) = $dup_id
                MATCH (keeper:{label})
                WHERE elementId(keeper) = $keeper_id
                AND NOT (src)-[:{get_rel_type(label)}]->(keeper)
                MERGE (src)-[:{get_rel_type(label)}]->(keeper)
            """, dup_id=dup.element_id, keeper_id=keeper.element_id)

            # Delete the duplicate node (detach removes any leftover rels)
            session.run("""
                MATCH (n)
                WHERE elementId(n) = $dup_id
                DETACH DELETE n
            """, dup_id=dup.element_id)

            total_merged += 1

    return total_merged


def get_rel_type(label: str) -> str:
    mapping = {
        "Condition":       "TREATS",
        "Contraindication":"CONTRAINDICATED_FOR",
        "SideEffect":      "CAUSES",
        "Interaction":     "INTERACTS_WITH",
        "DrugClass":       "BELONGS_TO",
    }
    return mapping.get(label, "RELATED_TO")


def trim_whitespace(session, label: str) -> int:
    result = session.run(f"""
        MATCH (n:{label})
        WHERE n.name <> trim(n.name)
        SET n.name = trim(n.name)
        RETURN count(n) AS fixed
    """)
    return result.single()["fixed"]


def main():
    print("\n🔧  Neo4j Graph Cleanup")
    print(f"    URI: {NEO4J_URI}\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    total_trimmed = 0
    total_merged  = 0

    with driver.session() as session:
        # Step 1: Trim whitespace
        print("Step 1: Trimming whitespace from node names...")
        for label in NODE_TYPES:
            fixed = trim_whitespace(session, label)
            if fixed:
                print(f"  Trimmed {fixed} {label} nodes")
                total_trimmed += fixed
        print(f"  Total trimmed: {total_trimmed}")

        # Step 2: Merge case-sensitivity duplicates
        print("\nStep 2: Merging case-sensitivity duplicates...")
        for label in NODE_TYPES:
            merged = merge_case_duplicates(session, label)
            if merged:
                print(f"  Merged {merged} duplicate {label} nodes")
                total_merged += merged
        print(f"  Total merged: {total_merged}")

    driver.close()

    print(f"\n✅  Cleanup complete!")
    print(f"    Whitespace fixes : {total_trimmed}")
    print(f"    Duplicate merges : {total_merged}")
    print(f"\n  Run 04_evaluate_graph.py again to verify the results.\n")


if __name__ == "__main__":
    main()
