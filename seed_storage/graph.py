"""Neo4j graph client -- typed dual-label entities, vector indices, semantic relationships.

All entity nodes carry the __Entity__ base label plus a type-specific label:
  (:Person:__Entity__), (:Organization:__Entity__), (:Concept:__Entity__), etc.

Vector indices use text-embedding-3-small (1536 dims).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from . import config

_driver: AsyncDriver | None = None
_driver_loop_id: int | None = None  # id() of the event loop that created the driver


async def get_driver() -> AsyncDriver:
    """Return a cached AsyncDriver bound to the current event loop.

    The driver's internal connection pool binds to the event loop that created it.
    If the current loop differs (e.g. a new ``asyncio.run()`` call in a Celery
    task), the stale driver is discarded and a fresh one is created.  This
    prevents ``got Future attached to a different loop`` errors.
    """
    global _driver, _driver_loop_id
    current_loop_id = id(asyncio.get_running_loop())
    if _driver is not None and _driver_loop_id != current_loop_id:
        # Stale driver from a previous event loop — discard without closing
        # (the old loop is dead, so awaiting close() would itself fail).
        _driver = None
        _driver_loop_id = None
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        _driver_loop_id = current_loop_id
    return _driver


async def close():
    global _driver, _driver_loop_id
    if _driver:
        await _driver.close()
        _driver = None
        _driver_loop_id = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


# -- Schema initialization --

EMBEDDING_DIM = config.settings.EMBEDDING_DIM  # 1536


async def init_schema():
    """Create indexes and constraints. Idempotent."""
    driver = await get_driver()

    constraints = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT entity_canonical IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.canonical_name IS UNIQUE",
        "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT query_id IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE",
        "CREATE CONSTRAINT meta_key IF NOT EXISTS FOR (m:__Meta__) REQUIRE m.key IS UNIQUE",
    ]

    vector_indexes = [
        f"""CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
           FOR (n:__Entity__) ON (n.embedding)
           OPTIONS {{indexConfig: {{
             `vector.dimensions`: {EMBEDDING_DIM},
             `vector.similarity_function`: 'cosine'
           }}}}""",
        f"""CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
           FOR (f:Fact) ON (f.embedding)
           OPTIONS {{indexConfig: {{
             `vector.dimensions`: {EMBEDDING_DIM},
             `vector.similarity_function`: 'cosine'
           }}}}""",
        f"""CREATE VECTOR INDEX source_embedding IF NOT EXISTS
           FOR (s:Source) ON (s.embedding)
           OPTIONS {{indexConfig: {{
             `vector.dimensions`: {EMBEDDING_DIM},
             `vector.similarity_function`: 'cosine'
           }}}}""",
    ]

    lookup_indexes = [
        "CREATE INDEX source_uri IF NOT EXISTS FOR (s:Source) ON (s.source_uri)",
        "CREATE INDEX source_type IF NOT EXISTS FOR (s:Source) ON (s.type)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:__Entity__) ON (e.name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:__Entity__) ON (e.entity_type)",
        "CREATE INDEX tag_name_idx IF NOT EXISTS FOR (t:Tag) ON (t.name)",
        "CREATE INDEX meta_updated_at IF NOT EXISTS FOR (m:__Meta__) ON (m.updated_at)",
    ]

    fulltext_indexes = [
        """CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
           FOR (n:__Entity__) ON EACH [n.name, n.description]""",
        """CREATE FULLTEXT INDEX fact_statement IF NOT EXISTS
           FOR (f:Fact) ON EACH [f.statement]""",
        """CREATE FULLTEXT INDEX source_content IF NOT EXISTS
           FOR (s:Source) ON EACH [s.raw_content]""",
        """CREATE FULLTEXT INDEX query_fulltext IF NOT EXISTS
   FOR (q:Query) ON EACH [q.question, q.answer_md]""",
    ]

    async with driver.session() as session:
        for stmt in constraints + vector_indexes + lookup_indexes + fulltext_indexes:
            try:
                await session.run(stmt)
            except Exception:
                pass  # Index/constraint may already exist with different config


# -- Entity operations (dual-label) --

VALID_ENTITY_TYPES = {
    "Person", "Organization", "Product", "Concept",
    "Location", "Event", "Decision", "MentalModel",
}


async def upsert_entity(
    *,
    canonical_name: str,
    name: str,
    entity_type: str,
    description: str = "",
    embedding: list[float],
    aliases: list[str] | None = None,
    group_id: str = "ant-haul",
) -> str:
    """Create or merge an entity node with dual labels.

    Uses MERGE on canonical_name. On create, sets both __Entity__ and type label.
    On match, updates description (keep longer) and appends aliases.
    Returns the entity node id.
    """
    driver = await get_driver()
    entity_id = _uuid()
    now = _now()

    async with driver.session() as session:
        result = await session.run(
            """
            MERGE (e:__Entity__ {canonical_name: $canonical_name})
            ON CREATE SET
                e.id = $id,
                e.name = $name,
                e.entity_type = $entity_type,
                e.description = $description,
                e.embedding = $embedding,
                e.aliases = $aliases,
                e.group_id = $group_id,
                e.created_at = $now,
                e.updated_at = $now
            ON MATCH SET
                e.description = CASE WHEN size(e.description) < size($description)
                                     THEN $description ELSE e.description END,
                e.aliases = [x IN (coalesce(e.aliases, []) + $aliases) WHERE x IS NOT NULL | x],
                e.embedding = $embedding,
                e.updated_at = $now
            RETURN e.id AS id
            """,
            id=entity_id,
            canonical_name=canonical_name,
            name=name,
            entity_type=entity_type,
            description=description,
            embedding=embedding,
            aliases=aliases or [],
            group_id=group_id,
            now=now,
        )
        record = await result.single()
        node_id = record["id"]

        # Set type-specific label (idempotent)
        if entity_type in VALID_ENTITY_TYPES:
            await session.run(
                f"MATCH (e:__Entity__ {{id: $id}}) SET e:{entity_type}",
                id=node_id,
            )

    return node_id


async def create_source(
    *,
    source_type: str,
    source_uri: str,
    raw_content: str,
    embedding: list[float],
    author: str,
    created_at: str | None = None,
    channel: str = "",
    group_id: str = "ant-haul",
) -> str:
    """Create a Source node (provenance). Returns node id."""
    driver = await get_driver()
    node_id = _uuid()
    now = _now()

    async with driver.session() as session:
        await session.run(
            """
            MERGE (s:Source {source_uri: $source_uri})
            ON CREATE SET
                s.id = $id,
                s.type = $type,
                s.raw_content = $raw_content,
                s.embedding = $embedding,
                s.author = $author,
                s.created_at = $created_at,
                s.ingested_at = $ingested_at,
                s.channel = $channel,
                s.group_id = $group_id
            RETURN s.id AS id
            """,
            id=node_id,
            type=source_type,
            source_uri=source_uri,
            raw_content=raw_content[:5000],
            embedding=embedding,
            author=author,
            created_at=created_at or now,
            ingested_at=now,
            channel=channel,
            group_id=group_id,
        )
    return node_id


async def create_fact(
    *,
    statement: str,
    embedding: list[float],
    source_id: str,
    confidence: float = 0.8,
    group_id: str = "ant-haul",
) -> str:
    """Create a Fact node linked to its Source. Returns fact id."""
    driver = await get_driver()
    fact_id = _uuid()
    now = _now()

    async with driver.session() as session:
        await session.run(
            """
            MATCH (s:Source {id: $source_id})
            CREATE (f:Fact {
                id: $id,
                statement: $statement,
                embedding: $embedding,
                confidence: $confidence,
                extracted_at: $now,
                group_id: $group_id
            })
            CREATE (f)-[:EXTRACTED_FROM]->(s)
            """,
            id=fact_id,
            statement=statement,
            embedding=embedding,
            confidence=confidence,
            source_id=source_id,
            now=now,
            group_id=group_id,
        )
    return fact_id


ALLOWED_RELATIONSHIP_TYPES = {
    "WORKS_FOR", "FOUNDED", "CITES", "CREATED", "USES",
    "PART_OF", "LOCATED_IN", "RELATED_TO", "SUPPORTS", "MENTIONS",
    "HAS_TAG", "IN_COMMUNITY",
}


async def create_relationship(
    *,
    source_entity_id: str,
    target_entity_id: str,
    relationship_type: str,
    description: str = "",
    confidence: float = 0.8,
) -> None:
    """Create a typed relationship between two entity nodes. Uses MERGE to avoid duplicates."""
    driver = await get_driver()
    if relationship_type not in ALLOWED_RELATIONSHIP_TYPES:
        relationship_type = "RELATED_TO"

    async with driver.session() as session:
        await session.run(
            f"""
            MATCH (a:__Entity__ {{id: $source_id}})
            MATCH (b:__Entity__ {{id: $target_id}})
            MERGE (a)-[r:{relationship_type}]->(b)
            ON CREATE SET r.description = $description, r.confidence = $confidence,
                          r.created_at = $now
            """,
            source_id=source_entity_id,
            target_id=target_entity_id,
            description=description,
            confidence=confidence,
            now=_now(),
        )


async def link_fact_entity(fact_id: str, entity_id: str):
    """Create a MENTIONS relationship between Fact and Entity."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (f:Fact {id: $fact_id})
            MATCH (e:__Entity__ {id: $entity_id})
            MERGE (f)-[:MENTIONS]->(e)
            """,
            fact_id=fact_id,
            entity_id=entity_id,
        )


async def link_source_tag(source_id: str, tag_name: str):
    """Create a HAS_TAG relationship between Source and Tag (MERGE both)."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (s:Source {id: $source_id})
            MERGE (t:Tag {name: $tag_name})
            MERGE (s)-[:HAS_TAG]->(t)
            """,
            source_id=source_id,
            tag_name=tag_name,
        )


async def delete_entity(*, entity_id: str) -> bool:
    """Delete an entity and all its relationships. Returns True if node existed."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (e:__Entity__ {id: $id}) WITH e, count(e) AS found "
            "DETACH DELETE e RETURN found",
            id=entity_id,
        )
        record = await result.single()
        return bool(record and record["found"])


async def merge_entities(*, keep_id: str, remove_id: str) -> dict:
    """Merge remove_id into keep_id.
    - Absorbs aliases from remove into keep
    - Re-creates outgoing/incoming rels from remove as RELATED_TO on keep
    - Deletes remove node
    Returns dict with keep_id, removed_id, merged_at.
    """
    driver = await get_driver()
    now = _now()
    async with driver.session() as session:
        await session.run(
            """MATCH (keep:__Entity__ {id: $kid}), (remove:__Entity__ {id: $rid})
               SET keep.aliases = [x IN (
                   coalesce(keep.aliases, []) + [remove.name]
                   + coalesce(remove.aliases, [])
               ) WHERE x IS NOT NULL | x],
               keep.updated_at = $now""",
            kid=keep_id, rid=remove_id, now=now,
        )
        await session.run(
            """MATCH (remove:__Entity__ {id: $rid})-[r]->(target)
               WHERE target.id <> $kid AND NOT target:Source AND NOT target:Tag
               MATCH (keep:__Entity__ {id: $kid})
               MERGE (keep)-[nr:RELATED_TO]->(target)
               ON CREATE SET nr.description = coalesce(r.description, ''),
                             nr.confidence = coalesce(r.confidence, 0.7),
                             nr.created_at = $now,
                             nr.merged_from = $rid""",
            rid=remove_id, kid=keep_id, now=now,
        )
        await session.run(
            """MATCH (source)-[r]->(remove:__Entity__ {id: $rid})
               WHERE source.id <> $kid AND source:__Entity__
               MATCH (keep:__Entity__ {id: $kid})
               MERGE (source)-[nr:RELATED_TO]->(keep)
               ON CREATE SET nr.description = coalesce(r.description, ''),
                             nr.confidence = coalesce(r.confidence, 0.7),
                             nr.created_at = $now,
                             nr.merged_from = $rid""",
            rid=remove_id, kid=keep_id, now=now,
        )
        await session.run(
            "MATCH (e:__Entity__ {id: $rid}) DETACH DELETE e",
            rid=remove_id,
        )
    return {"keep_id": keep_id, "removed_id": remove_id, "merged_at": now}


async def delete_relationship(
    *, source_entity_id: str, target_entity_id: str, relationship_type: str
) -> bool:
    """Delete a specific typed relationship between two entities."""
    driver = await get_driver()
    if relationship_type not in ALLOWED_RELATIONSHIP_TYPES:
        return False
    async with driver.session() as session:
        result = await session.run(
            f"""MATCH (a:__Entity__ {{id: $sid}})-[r:{relationship_type}]->(b:__Entity__ {{id: $tid}})
                DELETE r RETURN count(r) AS deleted""",
            sid=source_entity_id, tid=target_entity_id,
        )
        record = await result.single()
        return bool(record and record["deleted"] > 0)


async def write_synthesis(*, entity_id: str, synthesis: str) -> bool:
    """Write a long-form synthesis article (markdown) onto an entity node."""
    driver = await get_driver()
    now = _now()
    async with driver.session() as session:
        result = await session.run(
            """MATCH (e:__Entity__ {id: $id})
               SET e.synthesis = $synthesis, e.synthesis_updated_at = $now
               RETURN count(e) AS updated""",
            id=entity_id, synthesis=synthesis, now=now,
        )
        record = await result.single()
        return bool(record and record["updated"] > 0)


async def persist_query(
    *,
    question: str,
    answer_md: str,
    entity_ids: list[str] | None = None,
    group_id: str = "ant-haul",
) -> str:
    """Save a query result as a Query node, linked to referenced entities via REFERENCES."""
    driver = await get_driver()
    query_id = _uuid()
    now = _now()
    async with driver.session() as session:
        await session.run(
            """CREATE (q:Query {
                id: $id,
                question: $question,
                answer_md: $answer_md,
                created_at: $now,
                group_id: $group_id
            })""",
            id=query_id, question=question,
            answer_md=answer_md, now=now, group_id=group_id,
        )
        for eid in (entity_ids or []):
            await session.run(
                """MATCH (q:Query {id: $qid}), (e:__Entity__ {id: $eid})
                   MERGE (q)-[:REFERENCES]->(e)""",
                qid=query_id, eid=eid,
            )
    return query_id


async def upsert_meta(
    *,
    key: str,
    content: str,
    content_type: str = "markdown",
) -> None:
    """Create or replace a __Meta__ node - a named document stored in the graph.

    Used to persist operational documents (graph index, lint report) inside Neo4j
    so they are accessible to any agent regardless of runtime environment.

    key:          unique identifier e.g. 'graph_index', 'lint_report'
    content:      document body (markdown text)
    content_type: 'markdown' (default) or 'json'
    """
    driver = await get_driver()
    now = _now()
    async with driver.session() as session:
        await session.run(
            """
            MERGE (m:__Meta__ {key: $key})
            SET m.content = $content,
                m.content_type = $content_type,
                m.updated_at = $now
            """,
            key=key,
            content=content,
            content_type=content_type,
            now=now,
        )


async def get_meta(key: str) -> dict | None:
    """Retrieve a __Meta__ node by key. Returns None if not found.

    Returns dict with keys: key, content, content_type, updated_at.
    """
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (m:__Meta__ {key: $key}) RETURN m {.*} AS m",
            key=key,
        )
        record = await result.single()
        if not record:
            return None
        return dict(record["m"])


# -- Read operations --

async def vector_search(
    embedding: list[float],
    index_name: str = "entity_embedding",
    limit: int = 10,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Search nodes by vector similarity."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            CALL db.index.vector.queryNodes($index, $k, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node {.*, embedding: null} AS node, score
            ORDER BY score DESC
            """,
            index=index_name,
            k=limit,
            embedding=embedding,
            threshold=threshold,
        )
        return [{"node": r["node"], "score": r["score"]} async for r in result]


async def fulltext_search(
    query: str,
    index_name: str = "entity_name_fulltext",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Full-text search on node content."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            CALL db.index.fulltext.queryNodes($index, $search_query)
            YIELD node, score
            RETURN node {.*, embedding: null} AS node, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            index=index_name,
            search_query=query,
            limit=limit,
        )
        return [{"node": r["node"], "score": r["score"]} async for r in result]


async def hybrid_search(
    query: str,
    embedding: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid vector + fulltext search on entities. Deduplicates by node id."""
    vector_results = await vector_search(embedding, limit=limit)
    fulltext_results = await fulltext_search(query, limit=limit)

    seen_ids = set()
    combined = []
    for r in vector_results + fulltext_results:
        node_id = r["node"].get("id")
        if node_id and node_id not in seen_ids:
            seen_ids.add(node_id)
            combined.append(r)
    return sorted(combined, key=lambda x: x["score"], reverse=True)[:limit]


async def get_entity_context(entity_id: str) -> dict:
    """Get full context for an entity -- connected entities and relationships."""
    driver = await get_driver()
    async with driver.session() as session:
        entity_result = await session.run(
            "MATCH (e:__Entity__ {id: $id}) RETURN e {.*, embedding: null} AS entity",
            id=entity_id,
        )
        entity_record = await entity_result.single()
        if not entity_record:
            return {"found": False}

        out_result = await session.run(
            """MATCH (e:__Entity__ {id: $id})-[r]->(m)
               WHERE m:__Entity__ OR m:Fact
               RETURN type(r) AS rel_type, r.description AS description,
                      m {.*, embedding: null} AS target, labels(m) AS labels
               LIMIT 50""",
            id=entity_id,
        )
        outgoing = [
            {"type": r["rel_type"], "description": r["description"],
             "target": r["target"], "labels": r["labels"]}
            async for r in out_result
        ]

        in_result = await session.run(
            """MATCH (m)-[r]->(e:__Entity__ {id: $id})
               WHERE m:__Entity__ OR m:Source OR m:Fact
               RETURN type(r) AS rel_type, r.description AS description,
                      m {.*, embedding: null} AS source, labels(m) AS labels
               LIMIT 50""",
            id=entity_id,
        )
        incoming = [
            {"type": r["rel_type"], "description": r["description"],
             "source": r["source"], "labels": r["labels"]}
            async for r in in_result
        ]

        return {
            "found": True,
            "entity": entity_record["entity"],
            "outgoing": outgoing,
            "incoming": incoming,
        }


async def get_stats() -> dict:
    """Return graph statistics."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (n)
            WITH labels(n) AS lbls, count(*) AS cnt
            UNWIND lbls AS label
            RETURN label, sum(cnt) AS count
            ORDER BY count DESC
            """
        )
        counts = {r["label"]: r["count"] async for r in result}

        rel_result = await session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            """
        )
        rels = {r["type"]: r["count"] async for r in rel_result}

        return {"nodes": counts, "relationships": rels}
