#!/usr/bin/env python3
"""Sweep community-detection parameters to find the best Leiden config.

Runs non-destructively — projects a temporary GDS graph, runs Leiden in
stream mode (no writes), and reports size distributions.

Usage:
    python scripts/calibrate_communities.py
    python scripts/calibrate_communities.py --apply --gamma 2.5
    python scripts/calibrate_communities.py --apply --gamma 2.5 --exclude RELATED_TO
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
from dataclasses import dataclass, field

from neo4j import AsyncGraphDatabase

# Add project root to path
sys.path.insert(0, ".")
from seed_storage.config import settings

log = logging.getLogger("calibrate")

ALL_REL_TYPES = [
    "RELATED_TO", "WORKS_FOR", "FOUNDED", "DISCUSSES", "CITES",
    "CREATED", "USES", "PART_OF", "SUPPORTS", "LOCATED_IN",
]

# Relationship types that are semantically weak / catch-all
WEAK_RELS = {"RELATED_TO", "DISCUSSES"}


@dataclass
class SweepResult:
    gamma: float
    excluded: list[str]
    weighted: bool
    community_count: int
    modularity: float
    sizes: list[int] = field(default_factory=list)
    singletons: int = 0

    @property
    def median_size(self) -> float:
        return statistics.median(self.sizes) if self.sizes else 0

    @property
    def max_size(self) -> int:
        return max(self.sizes) if self.sizes else 0

    @property
    def p90_size(self) -> float:
        return (
            sorted(self.sizes)[int(len(self.sizes) * 0.9)] if self.sizes else 0
        )

    def score(self) -> float:
        """Heuristic quality score — higher is better.

        Penalises: giant communities, too many singletons, low modularity.
        Rewards: moderate community count, tight size distribution.
        """
        if not self.sizes:
            return -999
        entity_count = sum(self.sizes)
        # Fraction in mega-communities (>50 members)
        mega_frac = sum(s for s in self.sizes if s > 50) / entity_count
        # Singleton fraction
        singleton_frac = self.singletons / len(self.sizes)
        return (
            self.modularity * 2
            - mega_frac * 3
            - singleton_frac * 1.5
            + (1.0 if 5 <= self.median_size <= 20 else 0)
        )


async def get_driver():
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


async def sweep_gamma(
    driver,
    gamma: float,
    rel_types: list[str],
    weighted: bool = False,
) -> SweepResult:
    """Run Leiden in stream mode (no writes) with given parameters."""
    graph_name = f"calibrate_{gamma}_{hash(tuple(rel_types)) % 10000}"
    excluded = [r for r in ALL_REL_TYPES if r in ALL_REL_TYPES and r not in rel_types]

    async with driver.session() as session:
        # Build projection config
        rel_config = {}
        for rt in rel_types:
            cfg = {"orientation": "UNDIRECTED"}
            if weighted and rt in WEAK_RELS:
                cfg["properties"] = {"weight": {"defaultValue": 0.25}}
            elif weighted:
                cfg["properties"] = {"weight": {"defaultValue": 1.0}}
            rel_config[rt] = cfg

        # Drop if exists
        try:
            await session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        # Project
        rel_config_cypher = ", ".join(
            f"{rt}: {{{', '.join(f'{k}: {repr(v)}' if isinstance(v, str) else f'{k}: {v}' for k, v in cfg.items())}}}"
            for rt, cfg in rel_config.items()
        )

        # Use a simpler projection approach
        rel_projection = {rt: {"orientation": "UNDIRECTED"} for rt in rel_types}

        # Project with native projection
        try:
            await session.run(
                "CALL gds.graph.project($name, '__Entity__', $rels)",
                name=graph_name,
                rels=rel_projection,
            )
        except Exception as exc:
            log.error("Projection failed for gamma=%.1f: %s", gamma, exc)
            return SweepResult(
                gamma=gamma, excluded=excluded, weighted=weighted,
                community_count=0, modularity=0.0,
            )

        # Run Leiden in stream mode
        try:
            result = await session.run(
                """
                CALL gds.leiden.stream($name, {
                    gamma: $gamma,
                    includeIntermediateCommunities: false
                })
                YIELD nodeId, communityId
                RETURN communityId, count(*) AS members
                ORDER BY members DESC
                """,
                name=graph_name,
                gamma=gamma,
            )
            rows = [
                {"communityId": r["communityId"], "members": r["members"]}
                async for r in result
            ]
        except Exception as exc:
            log.error("Leiden stream failed for gamma=%.1f: %s", gamma, exc)
            rows = []

        # Clean up
        try:
            await session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

    sizes = [r["members"] for r in rows]
    singletons = sum(1 for s in sizes if s == 1)

    return SweepResult(
        gamma=gamma,
        excluded=excluded,
        weighted=weighted,
        community_count=len(rows),
        modularity=0.0,  # stream mode doesn't return modularity; we use size heuristics
        sizes=sizes,
        singletons=singletons,
    )


async def apply_best(driver, gamma: float, rel_types: list[str]) -> dict:
    """Apply the chosen parameters: wipe old communities and run Leiden write mode."""
    graph_name = "entity_graph"

    async with driver.session() as session:
        # Wipe old communities
        log.info("Removing old __Community__ nodes and IN_COMMUNITY rels...")
        await session.run("""
            MATCH (c:__Community__)
            DETACH DELETE c
        """)
        await session.run("""
            MATCH (e:__Entity__) REMOVE e.community_id
        """)

        # Drop existing projection if any
        try:
            await session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        # Project
        rel_projection = {rt: {"orientation": "UNDIRECTED"} for rt in rel_types}
        await session.run(
            "CALL gds.graph.project($name, '__Entity__', $rels)",
            name=graph_name,
            rels=rel_projection,
        )

        # Run Leiden write mode
        result = await session.run(
            """
            CALL gds.leiden.write($name, {
                writeProperty: 'community_id',
                gamma: $gamma,
                includeIntermediateCommunities: false
            })
            YIELD communityCount, modularity
            RETURN communityCount, modularity
            """,
            name=graph_name,
            gamma=gamma,
        )
        record = await result.single()
        community_count = record["communityCount"]
        modularity = record["modularity"]

        # Clean up projection
        await session.run(f"CALL gds.graph.drop('{graph_name}')")

        # Create __Community__ nodes
        await session.run("""
            MATCH (e:__Entity__) WHERE e.community_id IS NOT NULL
            WITH e.community_id AS cid, collect(e) AS members
            MERGE (c:__Community__ {id: 'community_' + toString(cid)})
            SET c.member_count = size(members),
                c.group_id = 'ant-haul'
            WITH c, members
            UNWIND members AS m
            MERGE (m)-[:IN_COMMUNITY]->(c)
        """)

    log.info(
        "Applied: gamma=%.2f, %d communities, modularity=%.4f",
        gamma, community_count, modularity,
    )
    return {"community_count": community_count, "modularity": modularity}


async def generate_summaries(driver, limit: int = 50) -> int:
    """Generate LLM summaries for communities without summaries."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    summarized = 0

    async with driver.session() as session:
        result = await session.run("""
            MATCH (c:__Community__) WHERE c.summary IS NULL
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
            WITH c, collect(e.name + ': ' + coalesce(e.description, '')) AS members,
                 c.member_count AS member_count
            WHERE member_count >= 3
            RETURN c.id AS cid, members, member_count
            ORDER BY member_count DESC
            LIMIT $limit
        """, limit=limit)

        communities = [
            {"cid": r["cid"], "members": r["members"], "count": r["member_count"]}
            async for r in result
        ]

    for comm in communities:
        member_text = "\n".join(comm["members"][:30])
        response = client.chat.completions.create(
            model=settings.EXTRACTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are naming a community (cluster) of related entities from a knowledge graph. "
                        "Give a SHORT, SPECIFIC name (3-8 words) that captures what uniquely ties these "
                        "entities together. Avoid vague words like 'diverse', 'various', 'range of'. "
                        "Then write 1 sentence explaining the connection.\n\n"
                        "Format: NAME: <short name>\nDESCRIPTION: <1 sentence>"
                    ),
                },
                {"role": "user", "content": f"Community ({comm['count']} members):\n{member_text}"},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        # Extract name and description
        name_line = ""
        desc_line = ""
        for line in raw.split("\n"):
            if line.upper().startswith("NAME:"):
                name_line = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DESCRIPTION:"):
                desc_line = line.split(":", 1)[1].strip()
        summary = f"{name_line}: {desc_line}" if name_line else raw

        async with driver.session() as session:
            await session.run(
                "MATCH (c:__Community__ {id: $cid}) SET c.summary = $summary, c.name = $name",
                cid=comm["cid"], summary=summary, name=name_line or summary[:60],
            )
        summarized += 1
        if summarized % 10 == 0:
            log.info("Summarized %d/%d communities...", summarized, len(communities))

    return summarized


def print_result(r: SweepResult, prefix: str = ""):
    top10 = sorted(r.sizes, reverse=True)[:10]
    label = prefix or f"gamma={r.gamma:.1f}"
    excl = f" (excl: {','.join(r.excluded)})" if r.excluded else ""
    print(
        f"  {label}{excl}: "
        f"{r.community_count} communities, "
        f"max={r.max_size}, median={r.median_size:.0f}, p90={r.p90_size:.0f}, "
        f"singletons={r.singletons}, "
        f"score={r.score():.2f}"
    )
    print(f"    top10 sizes: {top10}")


async def run_sweep():
    driver = await get_driver()

    # Check GDS
    try:
        async with driver.session() as s:
            r = await s.run("RETURN gds.version() AS v")
            rec = await r.single()
            print(f"GDS version: {rec['v']}")
    except Exception:
        print("ERROR: Neo4j GDS plugin not available")
        await driver.close()
        return

    gammas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    # Sweep 1: All relationship types
    print("\n=== Sweep 1: All relationship types ===")
    all_rels = [r for r in ALL_REL_TYPES]
    results_all = []
    for g in gammas:
        r = await sweep_gamma(driver, g, all_rels)
        results_all.append(r)
        print_result(r)

    # Sweep 2: Exclude RELATED_TO (catch-all noise)
    print("\n=== Sweep 2: Exclude RELATED_TO ===")
    no_related = [r for r in ALL_REL_TYPES if r != "RELATED_TO"]
    results_no_related = []
    for g in gammas:
        r = await sweep_gamma(driver, g, no_related)
        results_no_related.append(r)
        print_result(r)

    # Sweep 3: Exclude RELATED_TO + DISCUSSES
    print("\n=== Sweep 3: Exclude RELATED_TO + DISCUSSES ===")
    no_weak = [r for r in ALL_REL_TYPES if r not in WEAK_RELS]
    results_no_weak = []
    for g in gammas:
        r = await sweep_gamma(driver, g, no_weak)
        results_no_weak.append(r)
        print_result(r)

    # Best overall
    all_results = results_all + results_no_related + results_no_weak
    best = max(all_results, key=lambda r: r.score())
    print(f"\n=== BEST CONFIG ===")
    print_result(best, prefix="BEST")
    excl_flag = f" --exclude {' '.join(best.excluded)}" if best.excluded else ""
    print(f"\n  To apply: python scripts/calibrate_communities.py --apply --gamma {best.gamma}{excl_flag}")

    await driver.close()


async def run_apply(gamma: float, exclude: list[str], summarize: bool):
    driver = await get_driver()
    rel_types = [r for r in ALL_REL_TYPES if r not in exclude]
    print(f"Applying: gamma={gamma}, rels={rel_types}")
    result = await apply_best(driver, gamma, rel_types)
    print(f"Result: {result['community_count']} communities, modularity={result['modularity']:.4f}")

    if summarize:
        print("Generating community summaries...")
        count = await generate_summaries(driver, limit=100)
        print(f"Summarized {count} communities")

    # Print final distribution
    async with driver.session() as session:
        r = await session.run("""
            MATCH (c:__Community__)
            OPTIONAL MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
            WITH c, count(e) AS members
            RETURN members, count(*) AS num_communities
            ORDER BY members DESC
        """)
        print("\nFinal size distribution:")
        async for rec in r:
            bar = "#" * min(rec["num_communities"], 60)
            print(f"  {rec['members']:>4} members: {rec['num_communities']:>4} communities {bar}")

    await driver.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate community detection")
    parser.add_argument("--apply", action="store_true",
                        help="Apply chosen parameters (wipes old communities)")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Resolution parameter (default: 2.0)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Relationship types to exclude")
    parser.add_argument("--no-summarize", action="store_true",
                        help="Skip LLM summary generation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.apply:
        asyncio.run(run_apply(args.gamma, args.exclude, not args.no_summarize))
    else:
        asyncio.run(run_sweep())


if __name__ == "__main__":
    main()
