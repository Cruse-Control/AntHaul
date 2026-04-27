"""Batch ingest using OpenAI Batch API for extraction, embeddings, and resolution.

5-phase pipeline:
  Phase 1: Submit extraction calls as a Batch API job (50% cheaper)
  Phase 2: Submit entity+source embedding calls as a Batch API job
  Phase 3: Tier 2 vector search in Neo4j (local, no API)
  Phase 4: Submit Tier 3 LLM judge calls as a Batch API job
  Phase 5: Write resolved entities + relationships to Neo4j

Usage:
  python -m scripts.batch_ingest [--limit N] [--dry-run] [--status enriched]
  python -m scripts.batch_ingest --resume <batch_id>
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from seed_storage import staging
from seed_storage.config import settings, BATCH_SIZE_DEFAULT
from seed_storage.extraction import (
    EXTRACTION_SCHEMA,
    _build_system_prompt,
    _apply_coreference,
    _normalize_entity_type,
    _parse_extraction,
)
from seed_storage.embeddings import embed_text, embed_texts
from seed_storage.graph import get_driver, close as close_driver, init_schema
from seed_storage.models import ExtractedEntity, ExtractedRelationship
from seed_storage.preseed import get_alias_map, init_preseed_table
from seed_storage.resolution import normalize_name, find_similar_entities, llm_judge_same_entity

log = logging.getLogger("batch_ingest")

# How often to poll for batch completion
POLL_INTERVAL_S = 10
# Max time to wait for a single batch before giving up
BATCH_TIMEOUT_S = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _build_extraction_request(item: dict, alias_map: dict, request_id: str) -> dict:
    """Build a single JSONL request for the Batch API extraction call."""
    content = item.get("raw_content", "") or ""
    source_type = item.get("source_type", "unknown")
    meta = item.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)

    processed_content = _apply_coreference(content, alias_map)

    enrichment_header = ""
    tags = meta.get("tags", [])
    summary = meta.get("summary", "")
    if tags and tags != ["uncategorized"]:
        enrichment_header += f"Tags: {', '.join(tags)}\n"
    if summary:
        enrichment_header += f"Summary: {summary}\n"
    discord_ctx = meta.get("discord_context", "")
    if discord_ctx:
        enrichment_header += f"Shared with context: {discord_ctx}\n"

    user_content = f"Source type: {source_type}\n"
    if enrichment_header:
        user_content += enrichment_header + "\n"
    user_content += f"Content:\n{processed_content[:8000]}"

    system_prompt = _build_system_prompt(source_type, alias_map)

    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": settings.EXTRACTION_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_schema", "json_schema": EXTRACTION_SCHEMA},
            "temperature": 0.1,
        },
    }


def _build_embedding_request(text: str, request_id: str) -> dict:
    """Build a single JSONL request for the Batch API embedding call."""
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": settings.EMBEDDING_MODEL,
            "input": text,
        },
    }


def _build_judge_request(
    entity_a: str, desc_a: str, entity_b: str, desc_b: str, request_id: str
) -> dict:
    """Build a single JSONL request for the Batch API LLM judge call."""
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": settings.EXTRACTION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You determine whether two entity references refer to the same "
                        "real-world entity. Answer YES only if they are definitely the same "
                        "entity. Answer NO if they are different entities, even if related. "
                        "Answer YES or NO only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Are these the same entity?\n\n"
                        f"Entity A: {entity_a}\n{desc_a}\n\n"
                        f"Entity B: {entity_b}\n{desc_b}\n\n"
                        f"Answer YES or NO."
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 5,
        },
    }


def _submit_batch(client: OpenAI, requests: list[dict], description: str) -> str:
    """Upload JSONL, create a batch, return the batch ID."""
    jsonl = "\n".join(json.dumps(r) for r in requests)
    file = client.files.create(
        file=io.BytesIO(jsonl.encode()), purpose="batch"
    )
    log.info("Uploaded %d requests as file %s", len(requests), file.id)

    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions" if requests[0]["url"].endswith("completions") else "/v1/embeddings",
        completion_window="24h",
        metadata={"description": description},
    )
    log.info("Created batch %s (%s): %d requests", batch.id, description, len(requests))
    return batch.id


def _wait_for_batch(client: OpenAI, batch_id: str, description: str) -> dict[str, dict]:
    """Poll until batch completes. Returns {custom_id: response_body}."""
    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        elapsed = int(time.monotonic() - start)

        log.info(
            "Batch %s [%s]: %d/%d completed (%ds elapsed)",
            batch_id, status, completed, total, elapsed,
        )

        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} {status}: {batch.errors}")
        if elapsed > BATCH_TIMEOUT_S:
            raise TimeoutError(f"Batch {batch_id} timed out after {elapsed}s")

        time.sleep(POLL_INTERVAL_S)

    # Download results
    output_file_id = batch.output_file_id
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} completed but no output file")

    content = client.files.content(output_file_id)
    results = {}
    for line in content.text.strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        response = entry.get("response", {})
        if response.get("status_code") == 200:
            results[custom_id] = response["body"]
        else:
            log.warning("Batch request %s failed: %s", custom_id, response)
    log.info("Batch %s complete: %d/%d successful results", batch_id, len(results), total)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_batch_ingest(
    limit: int = BATCH_SIZE_DEFAULT,
    status: str = "enriched",
    dry_run: bool = False,
) -> dict:
    """Run the 5-phase batch ingest pipeline."""

    log.info("=== Batch Ingest Start (limit=%d, status=%s, dry_run=%s) ===", limit, status, dry_run)

    # Load shared resources
    init_preseed_table()
    alias_map = get_alias_map()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # Fetch items to process
    items = staging.get_staged(status=status, limit=limit)
    if not items:
        log.info("No items at status=%s", status)
        return {"total": 0}

    # Filter out tiny content (same as extract_one)
    processable = []
    skipped_tiny = 0
    for item in items:
        content = item.get("raw_content", "") or ""
        if len(content.split()) < 50:
            skipped_tiny += 1
            staging.update_status([str(item["id"])], "extracted")
            staging.patch_metadata(str(item["id"]), {
                "extraction": {
                    "entities": [], "relationships": [],
                    "model_used": "skipped", "tokens_input": 0, "tokens_output": 0,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
            })
        else:
            processable.append(item)

    log.info("Items: %d processable, %d skipped (tiny content)", len(processable), skipped_tiny)
    if not processable:
        return {"total": len(items), "skipped": skipped_tiny, "processed": 0}

    if dry_run:
        log.info("DRY RUN: would process %d items via batch API", len(processable))
        return {"total": len(items), "would_process": len(processable), "skipped": skipped_tiny}

    # Mark items as in-progress
    item_ids = [str(i["id"]) for i in processable]
    staging.update_status(item_ids, "extracting")

    # Build item lookup
    item_map = {str(item["id"]): item for item in processable}

    # -----------------------------------------------------------------------
    # Phase 1: Batch extraction
    # -----------------------------------------------------------------------
    log.info("--- Phase 1: Extraction (%d items) ---", len(processable))
    t0 = time.monotonic()

    extraction_requests = []
    for item in processable:
        req_id = f"extract_{item['id']}"
        extraction_requests.append(
            _build_extraction_request(item, alias_map, req_id)
        )

    batch_id = _submit_batch(client, extraction_requests, f"batch_ingest_extraction_{len(processable)}")
    extraction_results = _wait_for_batch(client, batch_id, "extraction")

    # Parse extraction results and store in metadata
    all_entities: dict[str, list[ExtractedEntity]] = {}  # item_id -> entities
    all_relationships: dict[str, list[ExtractedRelationship]] = {}
    extraction_failed = 0

    for item in processable:
        item_id = str(item["id"])
        req_id = f"extract_{item_id}"
        result_body = extraction_results.get(req_id)

        if not result_body:
            log.warning("No extraction result for %s", item_id)
            staging.update_status([item_id], "failed")
            staging.patch_metadata(item_id, {"extraction_error": "batch_no_result"})
            extraction_failed += 1
            continue

        try:
            choices = result_body.get("choices", [])
            raw_content = choices[0]["message"]["content"] if choices else "{}"
            raw = json.loads(raw_content)
            usage = result_body.get("usage", {})

            parsed = _parse_extraction(
                raw,
                model_used=settings.EXTRACTION_MODEL,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

            all_entities[item_id] = parsed.entities
            all_relationships[item_id] = parsed.relationships

            staging.patch_metadata(item_id, {
                "extraction": {
                    **parsed.model_dump(),
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "batch_mode": True,
                },
            })
            staging.update_status([item_id], "extracted")

        except Exception as exc:
            log.warning("Failed to parse extraction for %s: %s", item_id, exc)
            staging.update_status([item_id], "failed")
            staging.patch_metadata(item_id, {"extraction_error": str(exc)[:500]})
            extraction_failed += 1

    extracted_ids = [iid for iid in all_entities if all_entities[iid] or all_relationships[iid]]
    log.info(
        "Phase 1 complete: %d extracted, %d failed, %d empty (%.1fs)",
        len(extracted_ids), extraction_failed, len(all_entities) - len(extracted_ids),
        time.monotonic() - t0,
    )

    if not extracted_ids:
        log.info("No entities to resolve, done.")
        return {
            "total": len(items), "skipped": skipped_tiny,
            "extracted": len(all_entities), "failed": extraction_failed,
        }

    # -----------------------------------------------------------------------
    # Phase 2: Batch embeddings (source content + entity descriptions)
    # -----------------------------------------------------------------------
    log.info("--- Phase 2: Embeddings ---")
    t1 = time.monotonic()

    embedding_requests = []
    embed_id_map: dict[str, tuple[str, str]] = {}  # req_id -> (item_id, "source"|entity_name)

    for item_id in extracted_ids:
        item = item_map[item_id]
        content = (item.get("raw_content", "") or "")[:500]
        if content:
            req_id = f"embed_source_{item_id}"
            embedding_requests.append(_build_embedding_request(content, req_id))
            embed_id_map[req_id] = (item_id, "__source__")

        for entity in all_entities.get(item_id, []):
            embed_text_str = f"{entity.name}: {entity.description}" if entity.description else entity.name
            req_id = f"embed_entity_{item_id}_{entity.canonical_name[:40]}"
            # Deduplicate by req_id (same entity name from different items)
            if req_id not in embed_id_map:
                embedding_requests.append(_build_embedding_request(embed_text_str, req_id))
                embed_id_map[req_id] = (item_id, entity.canonical_name)

    if embedding_requests:
        batch_id = _submit_batch(client, embedding_requests, f"batch_ingest_embeddings_{len(embedding_requests)}")
        embedding_results = _wait_for_batch(client, batch_id, "embeddings")
    else:
        embedding_results = {}

    # Parse embedding results
    source_embeddings: dict[str, list[float]] = {}  # item_id -> embedding
    entity_embeddings: dict[str, list[float]] = {}  # canonical_name -> embedding

    for req_id, result_body in embedding_results.items():
        item_id, key = embed_id_map.get(req_id, (None, None))
        if not item_id:
            continue
        data = result_body.get("data", [])
        if not data:
            continue
        embedding = data[0].get("embedding", [])
        if key == "__source__":
            source_embeddings[item_id] = embedding
        else:
            entity_embeddings[key] = embedding

    log.info(
        "Phase 2 complete: %d source embeds, %d entity embeds (%.1fs)",
        len(source_embeddings), len(entity_embeddings), time.monotonic() - t1,
    )

    # -----------------------------------------------------------------------
    # Phase 3: Tier 2 vector search (local Neo4j queries)
    # -----------------------------------------------------------------------
    log.info("--- Phase 3: Entity resolution (Tier 1 + Tier 2) ---")
    t2 = time.monotonic()

    driver = await get_driver()
    await init_schema()

    # Check if graph has entities (cold start detection)
    async with driver.session() as session:
        result = await session.run("MATCH (n:__Entity__) RETURN count(n) AS cnt")
        record = await result.single()
        entity_count = record["cnt"] if record else 0

    # Collect all unique entities across all items for resolution
    unique_entities: dict[str, tuple[ExtractedEntity, str]] = {}  # canonical -> (entity, item_id)
    for item_id in extracted_ids:
        for entity in all_entities.get(item_id, []):
            canonical = normalize_name(entity.name, alias_map)
            if canonical not in unique_entities:
                unique_entities[canonical] = (entity, item_id)

    # Resolution results: canonical_name -> {"action": "merge"|"create", ...}
    resolution_map: dict[str, dict] = {}
    tier3_candidates: list[tuple[str, ExtractedEntity, dict]] = []  # (canonical, entity, best_candidate)

    _COMPATIBLE_TYPES = {
        "Person": {"Person"}, "Organization": {"Organization"},
        "Product": {"Product"}, "Concept": {"Concept", "Product"},
        "Location": {"Location"}, "Event": {"Event"},
    }

    for canonical, (entity, item_id) in unique_entities.items():
        # Tier 1: canonical name check
        if entity_count == 0:
            resolution_map[canonical] = {"action": "create", "canonical_name": canonical}
            continue

        # Tier 2: embedding similarity via Neo4j vector index
        try:
            embed_text_str = f"{entity.name}: {entity.description}" if entity.description else entity.name
            # Use pre-computed embedding if available
            embedding = entity_embeddings.get(entity.canonical_name)
            if not embedding:
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
                    k=5, embedding=embedding,
                    threshold=settings.ENTITY_SIMILARITY_THRESHOLD,
                )
                candidates = [
                    {
                        "id": r["id"], "name": r["name"],
                        "canonical_name": r["canonical_name"],
                        "entity_type": r["entity_type"], "score": r["score"],
                    }
                    async for r in result
                ]

        except Exception as exc:
            log.warning("Tier 2 failed for '%s': %s", entity.name, exc)
            resolution_map[canonical] = {"action": "create", "canonical_name": canonical}
            continue

        if not candidates:
            resolution_map[canonical] = {"action": "create", "canonical_name": canonical}
            continue

        # Type filtering
        compatible = _COMPATIBLE_TYPES.get(entity.entity_type, {entity.entity_type})
        type_filtered = [c for c in candidates if c["entity_type"] in compatible]
        best_pool = type_filtered if type_filtered else candidates
        best = best_pool[0]

        # Exact canonical match
        for c in candidates:
            if c["canonical_name"] == canonical:
                resolution_map[canonical] = {"action": "merge", "existing_id": c["id"], "canonical_name": canonical}
                break
        else:
            if best in type_filtered and best["score"] >= settings.ENTITY_AMBIGUOUS_THRESHOLD:
                resolution_map[canonical] = {
                    "action": "merge", "existing_id": best["id"],
                    "canonical_name": best["canonical_name"],
                }
            elif best in type_filtered and best["score"] >= settings.ENTITY_SIMILARITY_THRESHOLD:
                # Ambiguous — needs Tier 3 LLM judge
                tier3_candidates.append((canonical, entity, best))
            else:
                resolution_map[canonical] = {"action": "create", "canonical_name": canonical}

    log.info(
        "Phase 3 complete: %d resolved (Tier 1+2), %d need Tier 3 judge (%.1fs)",
        len(resolution_map), len(tier3_candidates), time.monotonic() - t2,
    )

    # -----------------------------------------------------------------------
    # Phase 4: Batch Tier 3 LLM judge calls
    # -----------------------------------------------------------------------
    if tier3_candidates:
        log.info("--- Phase 4: LLM Judge (%d candidates) ---", len(tier3_candidates))
        t3 = time.monotonic()

        judge_requests = []
        judge_meta: dict[str, tuple[str, dict]] = {}  # req_id -> (canonical, best_candidate)

        for canonical, entity, best in tier3_candidates:
            req_id = f"judge_{canonical[:60]}"
            judge_requests.append(_build_judge_request(
                entity.name, entity.description,
                best["name"], f"Type: {best['entity_type']}",
                req_id,
            ))
            judge_meta[req_id] = (canonical, best)

        batch_id = _submit_batch(client, judge_requests, f"batch_ingest_judge_{len(judge_requests)}")
        judge_results = _wait_for_batch(client, batch_id, "judge")

        for req_id, result_body in judge_results.items():
            canonical, best = judge_meta.get(req_id, (None, None))
            if not canonical:
                continue
            choices = result_body.get("choices", [])
            answer = choices[0]["message"]["content"].strip().upper() if choices else "NO"
            if answer.startswith("YES"):
                resolution_map[canonical] = {
                    "action": "merge", "existing_id": best["id"],
                    "canonical_name": best["canonical_name"],
                }
                log.info("Tier 3 merge: '%s' -> '%s'", canonical, best["name"])
            else:
                resolution_map[canonical] = {"action": "create", "canonical_name": canonical}

        # Handle any that didn't get results
        for canonical, entity, best in tier3_candidates:
            if canonical not in resolution_map:
                resolution_map[canonical] = {"action": "create", "canonical_name": canonical}

        log.info("Phase 4 complete (%.1fs)", time.monotonic() - t3)
    else:
        log.info("--- Phase 4: Skipped (no Tier 3 candidates) ---")

    # -----------------------------------------------------------------------
    # Phase 5: Write to Neo4j
    # -----------------------------------------------------------------------
    log.info("--- Phase 5: Neo4j writes (%d items) ---", len(extracted_ids))
    t4 = time.monotonic()

    from seed_storage.graph import upsert_entity, create_source, create_relationship, link_source_tag

    loaded = 0
    load_failed = 0

    for item_id in extracted_ids:
        item = item_map[item_id]
        meta = item.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)

        entities = all_entities.get(item_id, [])
        relationships = all_relationships.get(item_id, [])

        try:
            # Create Source node
            source_embedding = source_embeddings.get(item_id) or await embed_text(
                (item.get("raw_content", "") or "")[:500]
            )
            source_node_id = await create_source(
                source_type=item["source_type"],
                source_uri=item["source_uri"],
                raw_content=item.get("raw_content", "") or "",
                embedding=source_embedding,
                author=item.get("author", ""),
                created_at=str(item.get("created_at", "")),
                channel=item.get("channel", ""),
            )

            # Link tags
            tags = meta.get("tags", [])
            for tag in tags:
                if tag and tag != "uncategorized":
                    await link_source_tag(source_node_id, tag)

            # Upsert entities
            entity_id_map: dict[str, str] = {}
            for entity in entities:
                canonical = normalize_name(entity.name, alias_map)
                resolution = resolution_map.get(canonical, {"action": "create", "canonical_name": canonical})

                embedding = entity_embeddings.get(entity.canonical_name)
                if not embedding:
                    embed_str = f"{entity.name}: {entity.description}" if entity.description else entity.name
                    embedding = await embed_text(embed_str)

                if resolution["action"] == "merge":
                    entity_id_map[canonical] = resolution["existing_id"]
                else:
                    node_id = await upsert_entity(
                        name=entity.name,
                        canonical_name=canonical,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        embedding=embedding,
                        aliases=entity.aliases,
                    )
                    entity_id_map[canonical] = node_id

            # Create relationships
            rels_created = 0
            for rel in relationships:
                src_canonical = normalize_name(rel.source, alias_map)
                tgt_canonical = normalize_name(rel.target, alias_map)
                src_id = entity_id_map.get(src_canonical)
                tgt_id = entity_id_map.get(tgt_canonical)
                if src_id and tgt_id:
                    await create_relationship(
                        source_entity_id=src_id,
                        target_entity_id=tgt_id,
                        relationship_type=rel.relationship_type,
                        description=rel.description,
                    )
                    rels_created += 1
                else:
                    log.warning(
                        "Relationship skipped: %s -[%s]-> %s (missing %s)",
                        rel.source, rel.relationship_type, rel.target,
                        "source" if not src_id else "target",
                    )
            log.info("Item %s: %d entities, %d/%d relationships written",
                     item_id[:8], len(entities), rels_created, len(relationships))

            staging.update_status([item_id], "loaded")
            loaded += 1

        except Exception as exc:
            log.error("Failed to load %s: %s", item_id, exc, exc_info=True)
            staging.update_status([item_id], "failed")
            staging.patch_metadata(item_id, {"load_error": str(exc)[:500]})
            load_failed += 1

    await close_driver()

    log.info(
        "Phase 5 complete: %d loaded, %d failed (%.1fs)",
        loaded, load_failed, time.monotonic() - t4,
    )

    result = {
        "total": len(items),
        "skipped_tiny": skipped_tiny,
        "extracted": len(all_entities) - extraction_failed,
        "extraction_failed": extraction_failed,
        "entities_resolved": len(resolution_map),
        "tier3_judged": len(tier3_candidates),
        "loaded": loaded,
        "load_failed": load_failed,
    }
    log.info("=== Batch Ingest Complete: %s ===", json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Batch ingest via OpenAI Batch API")
    parser.add_argument("--limit", type=int, default=BATCH_SIZE_DEFAULT, help="Max items to process")
    parser.add_argument("--status", default="enriched", help="Status to pull items from (default: enriched)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    result = asyncio.run(run_batch_ingest(
        limit=args.limit,
        status=args.status,
        dry_run=args.dry_run,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
