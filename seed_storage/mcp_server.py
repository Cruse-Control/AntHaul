"""MCP server exposing the AntHaul knowledge graph to Claude Code sessions.

Run as: uv run python -m seed_storage.mcp_server
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from seed_storage import staging
from seed_storage.embeddings import embed_text
from seed_storage.graph import (
    get_driver, hybrid_search, fulltext_search,
    get_entity_context, get_stats, close,
)

log = logging.getLogger("ant-haul-mcp")

mcp = FastMCP(
    "ant-haul",
    instructions="Query the CruseControl knowledge graph (Neo4j, typed entities)",
)


@mcp.tool()
async def search_graph(query: str, limit: int = 10) -> list[dict]:
    """Search the knowledge graph using hybrid vector + fulltext search.

    Args:
        query: Natural language search query
        limit: Maximum number of results (default 10)

    Returns entities and facts matching the query, ranked by relevance.
    """
    embedding = await embed_text(query)
    results = await hybrid_search(query=query, embedding=embedding, limit=limit)
    return [{"node": r["node"], "score": r["score"]} for r in results]


@mcp.tool()
async def get_context(entity: str) -> dict:
    """Get full context for an entity -- all connected facts, sources, and relationships.

    Args:
        entity: Name of the entity to look up

    Returns the entity's relationships grouped by direction.
    """
    results = await fulltext_search(entity, index_name="entity_name_fulltext", limit=1)
    if not results:
        return {"entity": entity, "found": False, "message": "No entity found"}

    entity_id = results[0]["node"].get("id")
    if not entity_id:
        return {"entity": entity, "found": False}

    return await get_entity_context(entity_id)


@mcp.tool()
async def explore(concept: str, depth: int = 2) -> dict:
    """Explore a concept -- search + expand to related entities via graph traversal.

    Args:
        concept: The concept, theme, or entity to explore
        depth: How many hops to traverse (1-3, default 2)
    """
    depth = max(1, min(3, depth))
    embedding = await embed_text(concept)
    results = await hybrid_search(query=concept, embedding=embedding, limit=5)

    driver = await get_driver()
    async with driver.session() as session:
        related_result = await session.run(
            f"""MATCH (n:__Entity__)
                WHERE toLower(n.name) CONTAINS toLower($concept)
                MATCH path = (n)-[*1..{depth}]-(m:__Entity__)
                RETURN DISTINCT m.name AS name, m.entity_type AS type,
                       m.description AS description
                LIMIT 20""",
            concept=concept,
        )
        related = [
            {"name": r["name"], "type": r["type"], "description": r.get("description", "")}
            async for r in related_result
        ]

    return {
        "concept": concept,
        "search_results": [{"node": r["node"], "score": r["score"]} for r in results],
        "related_entities": related,
    }


@mcp.tool()
async def recent(hours: int = 24, source_type: str = "", limit: int = 10) -> list[dict]:
    """Get recently loaded items from the pipeline.

    Args:
        hours: Look back window in hours (default 24)
        source_type: Filter by source type (optional)
        limit: Number of items (default 10, max 50)
    """
    limit = max(1, min(50, limit))
    items = staging.get_recently_loaded(hours=hours)
    if source_type:
        items = [i for i in items if i.get("source_type") == source_type]
    return [
        {
            "source_type": item["source_type"],
            "source_uri": item["source_uri"],
            "author": item.get("author", ""),
            "channel": item.get("channel", ""),
            "created_at": str(item["created_at"]) if item.get("created_at") else None,
            "word_count": item.get("word_count", 0),
            "tags": (item.get("metadata") or {}).get("tags", []),
        }
        for item in items[:limit]
    ]


@mcp.tool()
async def status() -> dict:
    """Get pipeline status -- item counts by status plus graph stats."""
    pipeline = staging.count_by_status()
    graph = await get_stats()
    return {"pipeline": pipeline, "graph": graph}


@mcp.tool()
async def express_ingest_url(url: str) -> dict:
    """Immediately ingest a URL into the knowledge graph (10-30 seconds).

    Runs the full pipeline: stage -> process -> enrich -> extract -> load.
    """
    from ingestion.express import express_ingest as _express
    return await _express(url=url, author="mcp-express", channel="mcp-express")


@mcp.tool()
async def rush_item(source_uri: str) -> dict:
    """Rush a previously staged item through the pipeline immediately."""
    from ingestion.express import express_ingest as _express
    return await _express(url=source_uri, author="mcp-rush", channel="mcp-rush")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
