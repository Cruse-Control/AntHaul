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
    upsert_entity, delete_entity, merge_entities,
    create_relationship, delete_relationship,
    write_synthesis, persist_query, link_source_tag,
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


@mcp.tool()
async def kg_upsert_entity(
    name: str,
    entity_type: str,
    description: str,
    canonical_name: str = "",
    aliases: list[str] | None = None,
) -> dict:
    """Create or update an entity in the knowledge graph.

    Args:
        name: Display name of the entity
        entity_type: One of: Person | Organization | Product | Concept | Location | Event | Decision | MentalModel
        description: Description of the entity (used for embedding - make it substantive)
        canonical_name: Stable lowercase-hyphen key (auto-derived from name if omitted)
        aliases: Alternative names for this entity

    Returns dict with entity_id and canonical_name.
    """
    cname = canonical_name or name.strip().lower().replace(" ", "-")
    embedding = await embed_text(f"{name}. {description}")
    eid = await upsert_entity(
        canonical_name=cname,
        name=name,
        entity_type=entity_type,
        description=description,
        embedding=embedding,
        aliases=aliases or [],
    )
    return {"entity_id": eid, "canonical_name": cname}


@mcp.tool()
async def kg_delete_entity(entity_id: str) -> dict:
    """Delete an entity and all its relationships from the graph.

    Args:
        entity_id: The entity's id property (get it from search_graph or get_context)

    Irreversible. Verify entity_id first with get_context().
    Returns {deleted: bool}.
    """
    deleted = await delete_entity(entity_id=entity_id)
    return {"deleted": deleted, "entity_id": entity_id}


@mcp.tool()
async def kg_merge_entities(keep_entity_id: str, remove_entity_id: str) -> dict:
    """Merge two entities: transfer all relationships to keep, delete remove.

    Args:
        keep_entity_id: ID of entity to keep
        remove_entity_id: ID of entity to delete (its name is absorbed as an alias)

    Use when deduplicating near-identical entities. Relationships are re-created
    as RELATED_TO (APOC required for type preservation).
    """
    return await merge_entities(keep_id=keep_entity_id, remove_id=remove_entity_id)


@mcp.tool()
async def kg_create_relationship(
    source_entity_id: str,
    target_entity_id: str,
    relationship_type: str,
    description: str = "",
    confidence: float = 0.8,
) -> dict:
    """Create a typed relationship between two entities.

    Args:
        source_entity_id: ID of source entity
        target_entity_id: ID of target entity
        relationship_type: One of: WORKS_FOR | FOUNDED | CITES | CREATED | USES |
                           PART_OF | LOCATED_IN | RELATED_TO | SUPPORTS | MENTIONS
        description: Human-readable explanation of the relationship
        confidence: Float 0-1 (default 0.8)

    Defaults to RELATED_TO if type is invalid.
    """
    await create_relationship(
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=relationship_type,
        description=description,
        confidence=confidence,
    )
    return {
        "created": True,
        "type": relationship_type,
        "source": source_entity_id,
        "target": target_entity_id,
    }


@mcp.tool()
async def kg_delete_relationship(
    source_entity_id: str,
    target_entity_id: str,
    relationship_type: str,
) -> dict:
    """Delete a specific typed relationship between two entities.

    Args:
        source_entity_id: ID of source entity
        target_entity_id: ID of target entity
        relationship_type: The relationship type to delete (must be exact)
    """
    deleted = await delete_relationship(
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=relationship_type,
    )
    return {"deleted": deleted, "type": relationship_type}


@mcp.tool()
async def kg_write_synthesis(entity_id: str, synthesis: str) -> dict:
    """Write a long-form synthesis article (markdown) onto an entity node.

    Args:
        entity_id: The entity's id property
        synthesis: Markdown article, 3-5 paragraphs. Cover: what the entity is,
                   key relationships, why it matters, notable facts from sources.

    Stored as entity.synthesis. Returned by get_context() and search_graph().
    """
    updated = await write_synthesis(entity_id=entity_id, synthesis=synthesis)
    return {"updated": updated, "entity_id": entity_id}


@mcp.tool()
async def kg_persist_query(
    question: str,
    answer_md: str,
    entity_ids: list[str] | None = None,
) -> dict:
    """Save a valuable query result back into the graph as a Query node.

    Args:
        question: The question that was asked
        answer_md: The answer in markdown format
        entity_ids: List of entity IDs referenced in the answer (creates REFERENCES edges)

    Query nodes are searchable via search_graph(). Only persist substantial answers
    (comparisons, deep dives, novel synthesis) - not trivial lookups.
    Answers compound over time - future agents find them via search.
    """
    qid = await persist_query(
        question=question,
        answer_md=answer_md,
        entity_ids=entity_ids or [],
    )
    return {"query_id": qid, "question": question}


@mcp.tool()
async def kg_tag_source(source_id: str, tags: list[str]) -> dict:
    """Apply one or more tags to a source node.

    Args:
        source_id: The source node's id property
        tags: List of tag names (automatically lowercased and trimmed)
    """
    for tag in tags:
        await link_source_tag(source_id=source_id, tag_name=tag.lower().strip())
    return {"tagged": len(tags), "source_id": source_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
