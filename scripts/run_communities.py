#!/usr/bin/env python3
"""Community detection - GDS Leiden + LLM cluster labeling.

Projects entity graph -> runs Leiden algorithm -> creates __Community__ nodes
-> LLM labels each cluster.

Usage:
    cd /home/wyler-zahm/Desktop/cruse-control/AntHaul
    source .venv/bin/activate
    # Check GDS availability first:
    python -m scripts.run_communities --check-only
    # Run full detection:
    python -m scripts.run_communities
"""
from __future__ import annotations
import argparse
import asyncio
import json


async def check_gds(session) -> bool:
    """Return True if GDS plugin is available."""
    try:
        result = await session.run(
            "CALL gds.list() YIELD name RETURN name LIMIT 1"
        )
        rows = [r async for r in result]
        return len(rows) > 0
    except Exception as e:
        print(f"GDS not available: {e}")
        return False


async def project_entity_graph(session) -> bool:
    """Project the entity subgraph for Leiden. Returns True on success."""
    try:
        # Drop stale projection if it exists
        await session.run(
            "CALL gds.graph.drop('entity-graph', false) YIELD graphName"
        )
    except Exception:
        pass  # projection may not exist yet

    try:
        await session.run("""
            CALL gds.graph.project(
              'entity-graph',
              '__Entity__',
              {
                RELATED_TO: {orientation: 'UNDIRECTED'},
                USES:        {orientation: 'UNDIRECTED'},
                PART_OF:     {orientation: 'UNDIRECTED'},
                CITES:       {orientation: 'UNDIRECTED'},
                SUPPORTS:    {orientation: 'UNDIRECTED'}
              }
            )
        """)
        print("Entity graph projected successfully")
        return True
    except Exception as e:
        print(f"Failed to project entity graph: {e}")
        return False


async def run_leiden(session, resolution: float = 1.0) -> bool:
    """Run Leiden algorithm and write community_id to entity nodes."""
    try:
        result = await session.run("""
            CALL gds.leiden.write('entity-graph', {
              writeProperty: 'community_id',
              gamma: $resolution,
              maxLevels: 5,
              minCommunitySize: 3
            })
            YIELD communityCount, modularity
            RETURN communityCount, modularity
        """, resolution=resolution)
        record = await result.single()
        if record:
            print(f"Leiden complete: {record['communityCount']} communities, modularity={record['modularity']:.4f}")
        return True
    except Exception as e:
        print(f"Leiden failed: {e}")
        return False


async def create_community_nodes(session, client) -> int:
    """Create __Community__ nodes and LLM-label each cluster."""
    communities = []
    async for r in await session.run("""
        MATCH (e:__Entity__)
        WHERE e.community_id IS NOT NULL
        WITH e.community_id AS cid, collect(e.name)[0..12] AS names, count(e) AS size
        WHERE size >= 3
        ORDER BY size DESC LIMIT 50
        RETURN cid, names, size
    """):
        communities.append(dict(r))

    print(f"Found {len(communities)} communities to label")
    labeled = 0

    for c in communities:
        names_str = ", ".join(str(n) for n in c["names"])
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content":
                    f"Give a 3-5 word thematic label for this cluster of entities: {names_str}\n"
                    "Reply with just the label, no quotes, no punctuation."}],
                max_tokens=20,
            )
            label = resp.choices[0].message.content.strip()
        except Exception:
            label = f"Cluster {c['cid']}"

        try:
            await session.run("""
                MERGE (com:__Community__ {community_id: $cid})
                SET com.id = $cid, com.label = $label, com.size = $size,
                    com.member_names = $names
                WITH com
                MATCH (e:__Entity__ {community_id: $cid})
                MERGE (e)-[:IN_COMMUNITY]->(com)
            """, cid=str(c["cid"]), label=label, size=c["size"], names=c["names"])
            print(f"  [{c['size']} members] {label}")
            labeled += 1
        except Exception as e:
            print(f"  Failed to create community node: {e}")

    return labeled


async def run(check_only: bool = False, resolution: float = 1.0) -> None:
    from seed_storage.graph import get_driver
    from seed_storage.config import settings
    from openai import AsyncOpenAI

    driver = await get_driver()

    async with driver.session() as session:
        gds_available = await check_gds(session)

        if not gds_available:
            print("GDS plugin not available on this Neo4j instance.")
            print("To install: add neo4j-graph-data-science to Neo4j plugins in K8s StatefulSet")
            return

        print("GDS available!")

        if check_only:
            print("--check-only mode: GDS is available, stopping here.")
            return

        projected = await project_entity_graph(session)
        if not projected:
            return

        leiden_ok = await run_leiden(session, resolution=resolution)
        if not leiden_ok:
            print("Leiden failed. Community nodes NOT created.")
            return

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        labeled = await create_community_nodes(session, client)
        print(f"\nDone: {labeled} communities created and labeled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run community detection on entity graph")
    parser.add_argument("--check-only", action="store_true", help="Just check if GDS is available")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden gamma parameter")
    args = parser.parse_args()
    asyncio.run(run(args.check_only, args.resolution))
