"""Seed Storage API — FastAPI server for the knowledge graph."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, graph, embeddings

_VIZ_DIST = Path(__file__).resolve().parent.parent / "viz" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await graph.init_schema()
    yield
    await graph.close()


app = FastAPI(title="Seed Storage", lifespan=lifespan)

if _VIZ_DIST.is_dir():
    app.mount("/viz", StaticFiles(directory=str(_VIZ_DIST), html=True), name="viz")


# ── Models ─────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    content: str
    source_type: str = "manual"
    source_uri: str = ""
    author: str = "unknown"
    channel: str = ""
    media_type: str = "text"
    metadata: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # "vector" | "fulltext" | "hybrid"
    tiers: list[str] = ["fact"]
    limit: int = 10
    expand: bool = True


class IngestResponse(BaseModel):
    source_id: str
    message: str = "Ingested successfully"


class QueryResult(BaseModel):
    results: list[dict[str, Any]]
    count: int


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/health")
async def health():
    try:
        driver = await graph.get_driver()
        await driver.verify_connectivity()
        return {"status": "ok", "neo4j": "connected"}
    except Exception as e:
        raise HTTPException(503, detail=f"Neo4j unavailable: {e}")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest raw content as a Source node with embedding."""
    embedding = await embeddings.embed_text(req.content)
    source_id = await graph.create_source(
        source_type=req.source_type,
        source_uri=req.source_uri,
        raw_content=req.content,
        media_type=req.media_type,
        embedding=embedding,
        author=req.author,
        channel=req.channel,
        metadata=req.metadata,
    )
    return IngestResponse(source_id=source_id)


@app.post("/api/query", response_model=QueryResult)
async def query(req: QueryRequest):
    """Hybrid search across the knowledge graph."""
    query_embedding = await embeddings.embed_text(req.query)

    results = []

    if req.mode in ("vector", "hybrid"):
        if req.expand and "fact" in req.tiers:
            vector_results = await graph.vector_search_with_expansion(
                query_embedding, limit=req.limit
            )
        else:
            for tier in req.tiers:
                label = tier.capitalize()
                tier_results = await graph.vector_search(
                    query_embedding, label=label, limit=req.limit
                )
                results.extend(tier_results)
            vector_results = results
        results = vector_results

    if req.mode in ("fulltext", "hybrid"):
        for tier in req.tiers:
            label = tier.capitalize()
            ft_results = await graph.fulltext_search(
                req.query, label=label, limit=req.limit
            )
            results.extend(ft_results)

    # Deduplicate by node id
    seen = set()
    unique = []
    for r in results:
        node = r.get("fact") or r.get("node")
        if node and node.get("id") not in seen:
            seen.add(node["id"])
            unique.append(r)

    # Sort by score descending
    unique.sort(key=lambda r: r.get("score", 0), reverse=True)

    return QueryResult(results=unique[: req.limit], count=len(unique))


@app.get("/api/stats")
async def stats():
    """Graph statistics."""
    return await graph.get_stats()


# ── Graph visualization endpoints ─────────────────────────────────


@app.get("/api/graph/full")
async def graph_full(limit: int = Query(50000, le=100000)):
    """Return all entities and entity-to-entity relationships for viz."""
    driver = await graph.get_driver()
    async with driver.session() as session:
        node_result = await session.run(
            """
            MATCH (e:__Entity__)
            RETURN e.id AS id, e.name AS name, e.canonical_name AS canonical_name,
                   e.entity_type AS entity_type, e.description AS description,
                   e.aliases AS aliases, e.created_at AS created_at
            LIMIT $limit
            """,
            limit=limit,
        )
        nodes = [dict(r) async for r in node_result]

        edge_result = await session.run(
            """
            MATCH (a:__Entity__)-[r]->(b:__Entity__)
            RETURN a.id AS source, b.id AS target, type(r) AS type,
                   r.description AS description, r.confidence AS confidence
            LIMIT $limit
            """,
            limit=limit,
        )
        edges = [dict(r) async for r in edge_result]

    return {"nodes": nodes, "edges": edges}


@app.get("/api/graph/neighborhood/{entity_id}")
async def graph_neighborhood(entity_id: str, depth: int = Query(1, ge=1, le=3)):
    """Return the N-hop neighborhood of an entity."""
    driver = await graph.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (start:__Entity__ {id: $id})
            CALL apoc.path.subgraphAll(start, {maxLevel: $depth,
                 labelFilter: '__Entity__'})
            YIELD nodes, relationships
            UNWIND nodes AS n
            WITH collect(DISTINCT {
                id: n.id, name: n.name, canonical_name: n.canonical_name,
                entity_type: n.entity_type, description: n.description,
                aliases: n.aliases, created_at: n.created_at
            }) AS nodeList, relationships
            UNWIND relationships AS r
            WITH nodeList, collect(DISTINCT {
                source: startNode(r).id, target: endNode(r).id,
                type: type(r), description: r.description,
                confidence: r.confidence
            }) AS edgeList
            RETURN nodeList AS nodes, edgeList AS edges
            """,
            id=entity_id,
            depth=depth,
        )
        record = await result.single()
        if not record:
            raise HTTPException(404, detail="Entity not found")
        return {"nodes": record["nodes"], "edges": record["edges"]}


@app.get("/api/graph/search")
async def graph_search(
    q: str = Query(..., min_length=1), limit: int = Query(20, le=100)
):
    """Search entities by name (fulltext)."""
    driver = await graph.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query)
            YIELD node, score
            RETURN node.id AS id, node.name AS name,
                   node.canonical_name AS canonical_name,
                   node.entity_type AS entity_type,
                   node.description AS description, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            query=q,
            limit=limit,
        )
        return {"results": [dict(r) async for r in result]}


# ── Entrypoint ─────────────────────────────────────────────────────


def main():
    uvicorn.run(
        "seed_storage.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
