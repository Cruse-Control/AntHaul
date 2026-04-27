"""Community detection -- Neo4j GDS Leiden + LLM community summaries.

Requires Neo4j GDS plugin. Falls back gracefully if not available.
Run as: python -m seed_storage.communities [--rebuild]

Calibration (2026-04-27, 2411 entities):
  gamma=10.0 with all 10 entity-entity rel types gave best results:
  172 communities (after filtering <3 members), modularity=0.59
"""
from __future__ import annotations

import asyncio
import logging

from openai import OpenAI

from seed_storage.config import settings
from seed_storage.graph import get_driver

log = logging.getLogger("communities")

# All entity-entity relationship types to include in community detection.
COMMUNITY_REL_TYPES = [
    "RELATED_TO", "WORKS_FOR", "FOUNDED", "DISCUSSES", "CITES",
    "CREATED", "USES", "PART_OF", "SUPPORTS", "LOCATED_IN",
]

# Minimum community size — singletons and pairs are noise, not communities.
MIN_COMMUNITY_SIZE = 3


async def check_gds_available() -> bool:
    """Check if Neo4j GDS plugin is installed."""
    driver = await get_driver()
    try:
        async with driver.session() as session:
            result = await session.run("RETURN gds.version() AS version")
            record = await result.single()
            log.info("GDS version: %s", record["version"])
            return True
    except Exception:
        log.warning("Neo4j GDS plugin not available")
        return False


async def run_leiden(gamma: float = 10.0) -> dict:
    """Run Leiden community detection on __Entity__ graph.

    Projects entity->entity relationships into a GDS graph, runs Leiden,
    writes community IDs back to __Community__ nodes.

    Args:
        gamma: Resolution parameter. Higher = more, smaller communities.
               Calibrated to 10.0 for the current graph density (~1 edge/entity).
    """
    driver = await get_driver()
    async with driver.session() as session:
        # Drop stale projection if exists
        try:
            await session.run("CALL gds.graph.drop('entity_graph', false)")
        except Exception:
            pass

        # Remove old communities
        await session.run("MATCH (c:__Community__) DETACH DELETE c")
        await session.run("MATCH (e:__Entity__) REMOVE e.community_id")

        # Project the graph with all entity-entity rel types
        rel_projection = {rt: {"orientation": "UNDIRECTED"} for rt in COMMUNITY_REL_TYPES}
        await session.run(
            "CALL gds.graph.project($name, '__Entity__', $rels)",
            name="entity_graph",
            rels=rel_projection,
        )

        # Run Leiden
        result = await session.run("""
            CALL gds.leiden.write('entity_graph', {
                writeProperty: 'community_id',
                gamma: $gamma,
                includeIntermediateCommunities: false
            })
            YIELD communityCount, modularity
            RETURN communityCount, modularity
        """, gamma=gamma)

        record = await result.single()
        community_count = record["communityCount"]
        modularity = record["modularity"]

        # Clean up projection
        await session.run("CALL gds.graph.drop('entity_graph')")

        # Create __Community__ nodes only for clusters >= MIN_COMMUNITY_SIZE
        await session.run("""
            MATCH (e:__Entity__) WHERE e.community_id IS NOT NULL
            WITH e.community_id AS cid, collect(e) AS members
            WHERE size(members) >= $min_size
            MERGE (c:__Community__ {id: 'community_' + toString(cid)})
            SET c.member_count = size(members),
                c.group_id = 'ant-haul'
            WITH c, members
            UNWIND members AS m
            MERGE (m)-[:IN_COMMUNITY]->(c)
        """, min_size=MIN_COMMUNITY_SIZE)

        # Count actual communities created (after size filter)
        result = await session.run("MATCH (c:__Community__) RETURN count(c) AS cnt")
        record = await result.single()
        filtered_count = record["cnt"]

    log.info(
        "Leiden: %d raw communities, %d after size filter (>=%d), modularity=%.3f",
        community_count, filtered_count, MIN_COMMUNITY_SIZE, modularity,
    )
    return {
        "community_count": filtered_count,
        "community_count_raw": community_count,
        "modularity": modularity,
    }


async def summarize_communities(limit: int = 200) -> int:
    """Generate LLM summaries for communities without summaries."""
    driver = await get_driver()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    summarized = 0

    async with driver.session() as session:
        result = await session.run("""
            MATCH (c:__Community__) WHERE c.summary IS NULL
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
            WITH c, collect(e.name + ': ' + coalesce(e.description, '')) AS members,
                 c.member_count AS member_count
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
            log.info("Summarized %d/%d communities", summarized, len(communities))

    return summarized


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    async def main():
        if not await check_gds_available():
            print("Neo4j GDS plugin not available. Install it first.")
            sys.exit(1)
        result = await run_leiden()
        print(f"Detected {result['community_count']} communities (modularity={result['modularity']:.3f})")
        count = await summarize_communities()
        print(f"Summarized {count} communities")

    asyncio.run(main())
