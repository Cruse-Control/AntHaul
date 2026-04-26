"""3-tier entity resolution for entity dedup/merge.

Tier 1: Canonical name normalization (O(1), deterministic)
Tier 2: Embedding cosine similarity via Neo4j vector index (threshold 0.65)
Tier 3: LLM-as-judge for ambiguous 0.65-0.90 band (~5% of entities)
"""
from __future__ import annotations

import logging

from openai import OpenAI

from seed_storage.config import settings
from seed_storage.embeddings import embed_text
from seed_storage.models import ExtractedEntity
from seed_storage.preseed import get_alias_map

log = logging.getLogger("resolution")


# -- Tier 1: Canonical Name Normalization --

def normalize_name(name: str, alias_map: dict[str, str] | None = None) -> str:
    """Normalize an entity name to canonical form.

    Steps: lowercase -> strip @/# -> lookup preseed aliases.
    """
    canonical = name.lower().strip().lstrip("@#")
    if alias_map:
        canonical = alias_map.get(canonical, canonical)
    return canonical


# -- Tier 2: Embedding Similarity --

async def find_similar_entities(
    name: str,
    entity_type: str,
    description: str,
    driver,
    top_k: int = 5,
) -> list[dict]:
    """Query Neo4j vector index for similar __Entity__ nodes.

    Returns list of {id, name, canonical_name, entity_type, score} sorted by score desc.
    """
    embed_text_str = f"{name}: {description}" if description else name
    embedding = await embed_text(embed_text_str)

    async with driver.session() as session:
        result = await session.run(
            """
            CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.id AS id, node.name AS name, node.canonical_name AS canonical_name,
                   node.entity_type AS entity_type, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=embedding,
            threshold=settings.ENTITY_SIMILARITY_THRESHOLD,
        )
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "canonical_name": r["canonical_name"],
                "entity_type": r["entity_type"],
                "score": r["score"],
            }
            async for r in result
        ]


# -- Tier 3: LLM Judge --

def llm_judge_same_entity(entity_a: str, desc_a: str,
                          entity_b: str, desc_b: str,
                          client: OpenAI | None = None) -> bool:
    """Ask LLM whether two entities are the same. Returns True if same.

    Only called for ambiguous cases (score between ENTITY_SIMILARITY_THRESHOLD
    and ENTITY_AMBIGUOUS_THRESHOLD).
    """
    if client is None:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": "Answer YES or NO only."},
            {"role": "user", "content": (
                f"Are these the same entity?\n\n"
                f"Entity A: {entity_a}\nDescription: {desc_a}\n\n"
                f"Entity B: {entity_b}\nDescription: {desc_b}\n\n"
                f"Answer YES or NO."
            )},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")


# -- Full Resolution Pipeline --

async def resolve_entity(
    entity: ExtractedEntity,
    driver,
    alias_map: dict[str, str] | None = None,
    client: OpenAI | None = None,
) -> dict:
    """Resolve an extracted entity against the existing graph.

    Returns:
        {"action": "merge", "existing_id": str, "canonical_name": str} -- merge into existing
        {"action": "create", "canonical_name": str} -- create new node
    """
    if alias_map is None:
        alias_map = get_alias_map()

    # Tier 1: Canonical name normalization
    canonical = normalize_name(entity.name, alias_map)

    # Check if graph has any entities yet (cold start)
    async with driver.session() as session:
        count_result = await session.run(
            "MATCH (n:__Entity__) RETURN count(n) AS cnt"
        )
        record = await count_result.single()
        entity_count = record["cnt"] if record else 0

    if entity_count == 0:
        # Cold start -- skip Tier 2/3, all entities are new
        return {"action": "create", "canonical_name": canonical}

    # Tier 2: Embedding similarity
    try:
        candidates = await find_similar_entities(
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            driver=driver,
        )
    except Exception as exc:
        log.warning("Tier 2 vector search failed for '%s': %s", entity.name, exc)
        return {"action": "create", "canonical_name": canonical}

    if not candidates:
        return {"action": "create", "canonical_name": canonical}

    best = candidates[0]

    # Exact canonical match -- always merge
    if best["canonical_name"] == canonical:
        return {"action": "merge", "existing_id": best["id"], "canonical_name": canonical}

    # High confidence -- merge without LLM
    if best["score"] >= settings.ENTITY_AMBIGUOUS_THRESHOLD:
        log.info("Tier 2 merge (score=%.3f): '%s' -> '%s'",
                 best["score"], entity.name, best["name"])
        return {"action": "merge", "existing_id": best["id"],
                "canonical_name": best["canonical_name"]}

    # Ambiguous band -- Tier 3 LLM judge
    if best["score"] >= settings.ENTITY_SIMILARITY_THRESHOLD:
        try:
            is_same = llm_judge_same_entity(
                entity.name, entity.description,
                best["name"], f"Type: {best['entity_type']}",
                client=client,
            )
            if is_same:
                log.info("Tier 3 merge (LLM confirmed): '%s' -> '%s'",
                         entity.name, best["name"])
                return {"action": "merge", "existing_id": best["id"],
                        "canonical_name": best["canonical_name"]}
        except Exception as exc:
            # Tier 3 failure -- fall back to create (safe default)
            log.warning("Tier 3 LLM judge failed for '%s': %s", entity.name, exc)

    return {"action": "create", "canonical_name": canonical}
