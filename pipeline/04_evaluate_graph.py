"""
04_evaluate_graph.py — Structural quality report for the Neo4j medication graph

Runs a series of Cypher queries to evaluate:
  1. Overview: node/relationship counts by type
  2. Drug richness: how many relationships each drug has
  3. Orphan drugs: drugs with no TREATS relationships (possible extraction failures)
  4. Coverage: % of drugs with each relationship type
  5. Top conditions, side effects, contraindications (spot-check accuracy)
  6. Possible duplicate nodes: nodes with very similar names
"""

import sys
from pathlib import Path
from collections import defaultdict

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(session, cypher: str, **params):
    return list(session.run(cypher, **params))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_overview(session):
    section("1. OVERVIEW — Node & Relationship Counts")

    node_counts = run(session, """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    print(f"\n  {'Node Type':<22} {'Count':>8}")
    print(f"  {'-'*32}")
    for r in node_counts:
        print(f"  {r['label']:<22} {r['count']:>8,}")

    rel_counts = run(session, """
        MATCH ()-[r]->()
        RETURN type(r) AS rel, count(r) AS count
        ORDER BY count DESC
    """)
    print(f"\n  {'Relationship Type':<28} {'Count':>8}")
    print(f"  {'-'*38}")
    for r in rel_counts:
        print(f"  {r['rel']:<28} {r['count']:>8,}")


def report_richness(session):
    section("2. DRUG RICHNESS — Relationships per Drug")

    stats = run(session, """
        MATCH (d:Drug)
        OPTIONAL MATCH (d)-[r]->()
        RETURN d.pill_name AS drug, count(r) AS total_rels
        ORDER BY total_rels DESC
    """)

    if not stats:
        print("  No drugs found.")
        return

    counts = [r["total_rels"] for r in stats]
    avg    = sum(counts) / len(counts)
    min_c  = min(counts)
    max_c  = max(counts)

    print(f"\n  Total drugs     : {len(stats)}")
    print(f"  Avg connections : {avg:.1f}")
    print(f"  Max connections : {max_c}  ({stats[0]['drug']})")
    print(f"  Min connections : {min_c}")

    # Distribution buckets
    buckets = defaultdict(int)
    for c in counts:
        if c == 0:   buckets["0 (orphan)"] += 1
        elif c <= 5: buckets["1–5"] += 1
        elif c <= 15:buckets["6–15"] += 1
        else:        buckets["16+"] += 1

    print(f"\n  Distribution:")
    for label, cnt in sorted(buckets.items()):
        bar = "█" * min(cnt, 40)
        print(f"    {label:<12} {bar} {cnt}")


def report_orphans(session):
    section("3. ORPHAN DRUGS — No TREATS Relationship (Extraction Failures)")

    orphans = run(session, """
        MATCH (d:Drug)
        WHERE NOT (d)-[:TREATS]->()
        RETURN d.pill_name AS drug
        ORDER BY drug
    """)

    if not orphans:
        print("\n  ✅  All drugs have at least one TREATS relationship!")
    else:
        print(f"\n  ⚠️  {len(orphans)} drugs have NO indication data:")
        for r in orphans:
            print(f"    - {r['drug']}")


def report_coverage(session):
    section("4. COVERAGE — % of Drugs with Each Relationship Type")

    total = run(session, "MATCH (d:Drug) RETURN count(d) AS n")[0]["n"]
    if total == 0:
        print("  No drugs found.")
        return

    rels = [
        ("TREATS",             "Indications (treats)"),
        ("CONTRAINDICATED_FOR","Contraindications"),
        ("CAUSES",             "Side effects"),
        ("BELONGS_TO",         "Drug class"),
        ("INTERACTS_WITH",     "Drug interactions"),
    ]

    print(f"\n  {'Relationship':<28} {'Coverage':>10}")
    print(f"  {'-'*40}")
    for rel, label in rels:
        result = run(session, f"""
            MATCH (d:Drug)-[:{rel}]->()
            RETURN count(DISTINCT d) AS n
        """)
        count = result[0]["n"]
        pct   = count / total * 100
        bar   = "█" * int(pct / 5)
        print(f"  {label:<28} {pct:>6.1f}%  {bar}")


def report_top_nodes(session):
    section("5. SPOT CHECK — Top Nodes by Frequency")

    queries = [
        ("Top 10 Conditions (Indications)",   "Condition",       "TREATS"),
        ("Top 10 Contraindications",          "Contraindication","CONTRAINDICATED_FOR"),
        ("Top 10 Side Effects",               "SideEffect",      "CAUSES"),
    ]

    for title, label, rel in queries:
        print(f"\n  --- {title} ---")
        results = run(session, f"""
            MATCH (d:Drug)-[:{rel}]->(n:{label})
            RETURN n.name AS name, count(d) AS drug_count
            ORDER BY drug_count DESC
            LIMIT 10
        """)
        for i, r in enumerate(results, 1):
            print(f"    {i:>2}. {r['name']:<45} ({r['drug_count']} drugs)")


def report_possible_duplicates(session):
    section("6. POSSIBLE DUPLICATES — Nodes with Similar Names")

    print("\n  Checking Condition nodes with very similar names...")
    results = run(session, """
        MATCH (c:Condition)
        WITH toLower(trim(c.name)) AS norm, collect(c.name) AS names, count(*) AS cnt
        WHERE cnt > 1
        RETURN norm, names, cnt
        ORDER BY cnt DESC
        LIMIT 20
    """)

    if not results:
        print("  ✅  No exact duplicate Condition nodes found.")
    else:
        print(f"  ⚠️  {len(results)} groups of exact duplicates:")
        for r in results:
            print(f"    '{r['norm']}' — {r['cnt']} copies: {r['names']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n🔍  Neo4j Graph Quality Report")
    print(f"    URI: {NEO4J_URI}\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        report_overview(session)
        report_richness(session)
        report_orphans(session)
        report_coverage(session)
        report_top_nodes(session)
        report_possible_duplicates(session)

    driver.close()
    print(f"\n{'='*60}")
    print("  Report complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
