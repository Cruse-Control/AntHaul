"""Step 3: Load extracted items into Neo4j via custom graph writes.

Reads items at status='extracted', resolves entities, writes typed nodes.
Includes error classification, persistent circuit breaker, batch cost ceiling,
and Discord alerting for failures.

Run as: python -m ingestion.loader [--dry-run] [--concurrency N]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum

from ingestion import discord_touch
from seed_storage import staging
from seed_storage.config import (
    BATCH_COST_CEILING_USD,
    CIRCUIT_BREAKER_THRESHOLD,
    DISCORD_OPS_ALERTS_CHANNEL,
)
from seed_storage.embeddings import embed_text
from seed_storage.graph import (
    upsert_entity,
    create_source,
    create_relationship,
    link_source_tag,
    get_driver,
    close,
    init_schema,
)
from seed_storage.models import ExtractedEntity, ExtractedRelationship
from seed_storage.preseed import get_alias_map, init_preseed_table
from seed_storage.resolution import resolve_entity

log = logging.getLogger("loader")

# Cost estimation constants (gpt-4o-mini pricing).
GPT4O_MINI_INPUT_PER_M = 0.15
GPT4O_MINI_OUTPUT_PER_M = 0.60
EMBED_PER_M = 0.02
ESTIMATED_OUTPUT_RATIO = 0.15


# -- Error Classification --

class ErrorKind(Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    CREDIT_AUTH = "credit_auth"


def classify_error(exc: Exception) -> ErrorKind:
    """Classify an exception for retry/fail/halt decisions."""
    # --- OpenAI SDK errors ---
    try:
        import openai
        if isinstance(exc, openai.AuthenticationError):
            return ErrorKind.CREDIT_AUTH
        if isinstance(exc, openai.PermissionDeniedError):
            return ErrorKind.CREDIT_AUTH
        if isinstance(exc, openai.RateLimitError):
            if "credit balance" in str(exc).lower():
                return ErrorKind.CREDIT_AUTH
            return ErrorKind.RETRYABLE
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return ErrorKind.RETRYABLE
        if isinstance(exc, openai.InternalServerError):
            return ErrorKind.RETRYABLE
        if isinstance(exc, openai.BadRequestError):
            return ErrorKind.NON_RETRYABLE
    except ImportError:
        pass

    # --- Neo4j errors ---
    try:
        from neo4j.exceptions import (
            ServiceUnavailable,
            SessionExpired,
            TransientError,
        )
        if isinstance(exc, (ServiceUnavailable, SessionExpired, TransientError)):
            return ErrorKind.RETRYABLE
    except ImportError:
        pass

    # --- Fallback: network-like errors are retryable ---
    exc_name = type(exc).__name__.lower()
    if any(kw in exc_name for kw in ("timeout", "connect", "network")):
        return ErrorKind.RETRYABLE

    return ErrorKind.NON_RETRYABLE


# -- Cost Estimation --

def _content_hash(text: str) -> str:
    """Hash full content for exact-match dedup across different URLs."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def _estimate_cost(token_count: int) -> float:
    """Rough cost estimate for extraction + embedding."""
    input_cost = (token_count / 1_000_000) * GPT4O_MINI_INPUT_PER_M
    output_tokens = int(token_count * ESTIMATED_OUTPUT_RATIO)
    output_cost = (output_tokens / 1_000_000) * GPT4O_MINI_OUTPUT_PER_M
    embed_cost = (token_count / 1_000_000) * EMBED_PER_M
    return input_cost + output_cost + embed_cost


# -- Content Quality Gate --

_AUTH_WALL_PATTERNS = [
    "sign in to", "log in to", "create an account",
    "please enable javascript", "access denied",
    "403 forbidden", "404 not found",
    "just a moment...", "checking your browser",
    "you need to enable javascript", "verify you are human",
]


def _is_loadable(content: str, source_type: str) -> tuple[bool, str]:
    """Quality gate -- reject garbage content before graph ingestion."""
    stripped = content.strip()
    if not stripped or len(stripped) < 20:
        return False, "content_too_short"

    lower = stripped.lower()

    if len(stripped) < 500:
        for pattern in _AUTH_WALL_PATTERNS:
            if pattern in lower:
                return False, f"auth_wall:{pattern}"

    if lower.count("cookie") > 3 and len(stripped) < 300:
        return False, "cookie_wall"

    if stripped.startswith("[") and stripped.endswith("]") and len(stripped) < 100:
        return False, "stub_content"

    if stripped.startswith("[Tweet by") and "http" in stripped and len(stripped) < 200:
        return False, "tweet_stub"

    if "something went wrong" in lower and "try again" in lower:
        return False, "scrape_error_page"

    return True, "ok"


# -- Single Item Load --

async def _load_one_item(item: dict, alias_map: dict, client, driver, batch_id: str | None) -> str:
    """Load a single extracted item into Neo4j. Returns 'loaded'|'failed'|'skipped'."""
    item_id = str(item["id"])
    meta = item.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)

    extraction = meta.get("extraction", {})
    entities_raw = extraction.get("entities", [])
    relationships_raw = extraction.get("relationships", [])

    if not entities_raw and not relationships_raw:
        staging.update_status([item_id], "loaded", batch_id)
        return "skipped"

    # 1. Create Source node for provenance
    source_embedding = await embed_text((item.get("raw_content", "") or "")[:500])
    source_node_id = await create_source(
        source_type=item["source_type"],
        source_uri=item["source_uri"],
        raw_content=item.get("raw_content", "") or "",
        embedding=source_embedding,
        author=item.get("author", ""),
        created_at=str(item.get("created_at", "")),
        channel=item.get("channel", ""),
    )

    # 2. Link enrichment tags to Source
    tags = meta.get("tags", [])
    for tag in tags:
        if tag and tag != "uncategorized":
            await link_source_tag(source_node_id, tag)

    # 3. Resolve and upsert each entity
    entity_id_map: dict[str, str] = {}  # canonical_name -> neo4j node id
    for e_raw in entities_raw:
        entity = ExtractedEntity(**e_raw) if isinstance(e_raw, dict) else e_raw
        resolution = await resolve_entity(entity, driver, alias_map=alias_map, client=client)

        canonical = resolution["canonical_name"]
        if resolution["action"] == "merge":
            entity_id_map[canonical] = resolution["existing_id"]
        else:
            entity_embedding = await embed_text(f"{entity.name}: {entity.description}")
            node_id = await upsert_entity(
                canonical_name=canonical,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                embedding=entity_embedding,
                aliases=entity.aliases,
            )
            entity_id_map[canonical] = node_id

    # 4. Create relationships
    for r_raw in relationships_raw:
        rel = ExtractedRelationship(**r_raw) if isinstance(r_raw, dict) else r_raw
        source_canonical = rel.source.lower().strip()
        target_canonical = rel.target.lower().strip()
        source_canonical = alias_map.get(source_canonical, source_canonical)
        target_canonical = alias_map.get(target_canonical, target_canonical)

        src_id = entity_id_map.get(source_canonical)
        tgt_id = entity_id_map.get(target_canonical)
        if src_id and tgt_id:
            await create_relationship(
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relationship_type=rel.relationship_type,
                description=rel.description,
                confidence=rel.confidence,
            )

    staging.update_status([item_id], "loaded", batch_id)
    return "loaded"


# -- Batch Loading --

async def load_batch(limit: int = 200, dry_run: bool = False, concurrency: int = 1):
    """Load a batch of extracted items into Neo4j."""

    # Check persistent circuit breaker before doing anything.
    breaker = staging.is_breaker_tripped()
    if breaker:
        log.warning("Circuit breaker tripped (%s) -- skipping batch", breaker["reason"])
        return

    # Reset orphaned items from crashed batches.
    orphans_loading = staging.reset_orphaned_loading()
    orphans_extracting = staging.reset_orphaned_extracting()
    if orphans_loading:
        log.info("Reset %d orphaned 'loading' items", orphans_loading)
    if orphans_extracting:
        log.info("Reset %d orphaned 'extracting' items", orphans_extracting)

    items = staging.get_staged(status="extracted", limit=limit)
    if not items:
        log.info("No extracted items to load")
        return

    total_tokens = sum(i.get("token_estimate", 0) or 0 for i in items)
    estimated_cost = _estimate_cost(total_tokens)
    log.info(
        "Loading %d items (%d tokens, ~$%.4f estimated)",
        len(items), total_tokens, estimated_cost,
    )

    if dry_run:
        log.info("Dry run -- skipping actual load")
        for item in items:
            log.info("  [%s] %s (%d tokens)", item["source_type"], item["source_uri"], item.get("token_estimate", 0))
        return

    batch_id = str(uuid.uuid4())
    item_ids = [str(i["id"]) for i in items]
    staging.update_status(item_ids, "loading", batch_id)

    # Initialize schema and preseed
    await init_schema()
    init_preseed_table()
    alias_map = get_alias_map()
    driver = await get_driver()

    loaded = 0
    failed = 0
    skipped = 0
    consecutive_failures = 0
    running_cost = 0.0
    credit_auth_error: str | None = None

    # Sequential loading (concurrency=1) for consistent resolution
    for item in items:
        if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            item_id = str(item["id"])
            staging.update_status([item_id], "extracted")
            continue

        if credit_auth_error:
            item_id = str(item["id"])
            staging.update_status([item_id], "extracted")
            continue

        item_id = str(item["id"])
        content = item.get("raw_content", "") or ""

        # Quality gate
        loadable, reason = _is_loadable(content, item["source_type"])
        if not loadable:
            log.info("Rejected [%s] %s -- %s", item["source_type"], item["source_uri"], reason)
            staging.update_status([item_id], "rejected")
            continue

        try:
            result = await _load_one_item(item, alias_map, None, driver, batch_id)
            if result == "loaded":
                loaded += 1
                consecutive_failures = 0
                running_cost += _estimate_cost(item.get("token_estimate", 0) or 0)
                if running_cost >= BATCH_COST_CEILING_USD:
                    log.warning("Batch cost ceiling ($%.2f) reached", BATCH_COST_CEILING_USD)
                    break
                log.info("Loaded [%s] %s", item["source_type"], item["source_uri"])
                await discord_touch.react(item, "loaded")
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
        except Exception as exc:
            kind = classify_error(exc)
            if kind == ErrorKind.CREDIT_AUTH:
                log.error("CREDIT/AUTH error loading %s: %s", item["source_uri"], exc)
                staging.update_status([item_id], "failed")
                failed += 1
                credit_auth_error = str(exc)[:500]
                staging.trip_breaker(f"CREDIT_AUTH: {str(exc)[:200]}", cooldown_hours=None)
            elif kind == ErrorKind.NON_RETRYABLE:
                log.warning("Non-retryable error loading %s: %s", item["source_uri"], exc)
                staging.update_status([item_id], "failed")
                failed += 1
                consecutive_failures += 1
            else:
                log.warning("Retryable error loading %s: %s", item["source_uri"], exc)
                staging.update_status([item_id], "extracted")
                consecutive_failures += 1

    await close()

    # Batch alerts
    if credit_auth_error:
        staging.trip_breaker(f"CREDIT_AUTH in batch {batch_id}", cooldown_hours=None)
    elif consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        staging.trip_breaker(
            f"CONSECUTIVE_FAILURES: {consecutive_failures} in batch {batch_id}",
            cooldown_hours=1,
        )

    log.info("Batch %s complete: %d loaded, %d failed, %d skipped (cost ~$%.4f)",
             batch_id, loaded, failed, skipped, running_cost)


async def estimate():
    """Show cost estimate without loading anything."""
    await load_batch(dry_run=True, limit=5000)


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if "--dry-run" in sys.argv or "--estimate" in sys.argv:
        asyncio.run(estimate())
    else:
        asyncio.run(load_batch())
