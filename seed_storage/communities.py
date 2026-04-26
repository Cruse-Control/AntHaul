"""Community detection -- Neo4j GDS Leiden + LLM community summaries.

Requires Neo4j GDS plugin. Falls back gracefully if not available.
Run as: python -m seed_storage.communities [--rebuild]
"""
from __future__ import annotations

import asyncio
import logging

from openai import OpenAI

from seed_storage.config import settings
from seed_storage.graph import get_driver

log = logging.getLogger("communities")


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


async def run_leiden(gamma: float = 1.0) -> dict:
    """Run Leiden community detection on __Entity__ graph.

    Projects entity->entity relationships into a GDS graph, runs Leiden,
    writes community IDs back to __Community__ nodes.
    """
    driver = await get_driver()
    async with driver.session() as session:
        # Project the graph
        await session.run("""
            CALL gds.graph.project(
                'entity_graph',
                '__Entity__',
                {
                    RELATED_TO: {orientation: 'UNDIRECTED'},
                    WORKS_FOR: {orientation: 'UNDIRECTED'},
                    FOUNDED: {orientation: 'UNDIRECTED'},
                    DISCUSSES: {orientation: 'UNDIRECTED'},
                    CITES: {orientation: 'UNDIRECTED'},
                    CREATED: {orientation: 'UNDIRECTED'},
                    USES: {orientation: 'UNDIRECTED'},
                    PART_OF: {orientation: 'UNDIRECTED'}
                }
            )
        """)

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

        # Create __Community__ nodes from community IDs
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

    log.info("Leiden: %d communities, modularity=%.3f", community_count, modularity)
    return {"community_count": community_count, "modularity": modularity}


async def summarize_communities(limit: int = 20) -> int:
    """Generate LLM summaries for communities without summaries."""
    driver = await get_driver()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    summarized = 0

    async with driver.session() as session:
        result = await session.run("""
            MATCH (c:__Community__) WHERE c.summary IS NULL
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
            WITH c, collect(e.name + ': ' + coalesce(e.description, '')) AS members
            RETURN c.id AS cid, members
            LIMIT $limit
        """, limit=limit)

        communities = [{"cid": r["cid"], "members": r["members"]} async for r in result]

    for comm in communities:
        member_text = "\n".join(comm["members"][:20])
        response = client.chat.completions.create(
            model=settings.EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "Summarize this community of related entities in 2-3 sentences."},
                {"role": "user", "content": f"Community members:\n{member_text}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        summary = response.choices[0].message.content.strip()

        async with driver.session() as session:
            await session.run(
                "MATCH (c:__Community__ {id: $cid}) SET c.summary = $summary",
                cid=comm["cid"], summary=summary,
            )
        summarized += 1
        log.info("Summarized community %s", comm["cid"])

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
        print(f"Detected {result['community_count']} communities")
        count = await summarize_communities()
        print(f"Summarized {count} communities")

    asyncio.run(main())
