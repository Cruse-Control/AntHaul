"""Step 3: Load enriched items into Neo4j via Graphiti.

Supports concurrent loading via asyncio.Semaphore for 5-10x speedup.
Includes error classification, persistent circuit breaker, batch cost ceiling,
and Discord alerting for failures.

Run as: python -m ingestion.loader [--dry-run] [--concurrency N]
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum

import json

from ingestion import discord_touch
from seed_storage import staging
from seed_storage.config import (
    BATCH_COST_CEILING_USD,
    CIRCUIT_BREAKER_THRESHOLD,
    DISCORD_OPS_ALERTS_CHANNEL,
)
from seed_storage.graphiti_client import add_episode, close

log = logging.getLogger("loader")

# Cost estimation constants (corrected April 2026 pricing).
HAIKU_INPUT_PER_M = 0.80
HAIKU_OUTPUT_PER_M = 4.00
GEMINI_EMBED_PER_M = 0.02
ESTIMATED_OUTPUT_RATIO = 0.15


# ── Error Classification ─────────────────────────────────────────────

class ErrorKind(Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    CREDIT_AUTH = "credit_auth"


def classify_error(exc: Exception) -> ErrorKind:
    """Classify an exception for retry/fail/halt decisions.

    Order matters — check specific types before their parent classes.
    Graphiti wraps Anthropic RateLimitError into its own type, so check
    Graphiti errors first. If a RateLimitError escapes Graphiti, it means
    Graphiti already exhausted its internal 4-retry budget — don't retry again.
    """
    # --- Graphiti wrapped errors (check FIRST — these come from add_episode) ---
    try:
        from graphiti_core.llm_client.errors import RateLimitError as GraphitiRateLimit
        from graphiti_core.llm_client.errors import RefusalError
        if isinstance(exc, GraphitiRateLimit):
            if "credit balance" in str(exc).lower():
                return ErrorKind.CREDIT_AUTH
            # Graphiti already retried 4x internally — don't retry again
            return ErrorKind.NON_RETRYABLE
        if isinstance(exc, RefusalError):
            return ErrorKind.NON_RETRYABLE
    except ImportError:
        pass

    # --- Anthropic SDK errors ---
    try:
        import anthropic
        if isinstance(exc, anthropic.AuthenticationError):
            return ErrorKind.CREDIT_AUTH
        if isinstance(exc, anthropic.PermissionDeniedError):
            return ErrorKind.CREDIT_AUTH
        if isinstance(exc, anthropic.RateLimitError):
            if "credit balance" in str(exc).lower():
                return ErrorKind.CREDIT_AUTH
            return ErrorKind.RETRYABLE
        if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
            return ErrorKind.RETRYABLE
        if isinstance(exc, (anthropic.InternalServerError,)):
            return ErrorKind.RETRYABLE
        if isinstance(exc, anthropic.BadRequestError):
            return ErrorKind.NON_RETRYABLE
    except ImportError:
        pass

    # --- Google GenAI errors ---
    try:
        from google.genai.errors import ClientError as GeminiClientError
        from google.genai.errors import ServerError as GeminiServerError
        if isinstance(exc, GeminiClientError):
            code = getattr(exc, "code", 0)
            if code in (401, 403):
                return ErrorKind.CREDIT_AUTH
            if code == 429:
                return ErrorKind.RETRYABLE
            return ErrorKind.NON_RETRYABLE
        if isinstance(exc, GeminiServerError):
            return ErrorKind.RETRYABLE
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

    # Unknown errors default to non-retryable (safe for cost).
    return ErrorKind.NON_RETRYABLE


# ── Cost Estimation ───────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """Hash full content for exact-match dedup across different URLs."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def _estimate_cost(token_count: int) -> float:
    """Rough cost estimate for one Graphiti episode."""
    input_cost = (token_count / 1_000_000) * HAIKU_INPUT_PER_M
    output_tokens = int(token_count * ESTIMATED_OUTPUT_RATIO)
    output_cost = (output_tokens / 1_000_000) * HAIKU_OUTPUT_PER_M
    embed_cost = (token_count / 1_000_000) * GEMINI_EMBED_PER_M
    return input_cost + output_cost + embed_cost


# ── Content Quality Gate (#12) ────────────────────────────────────────

_AUTH_WALL_PATTERNS = [
    "sign in to", "log in to", "create an account",
    "please enable javascript", "access denied",
    "403 forbidden", "404 not found",
    "just a moment...", "checking your browser",
    "you need to enable javascript", "verify you are human",
]


def _is_loadable(content: str, source_type: str) -> tuple[bool, str]:
    """Quality gate — reject garbage content before Graphiti ingestion.

    Returns (is_loadable, reason).
    """
    stripped = content.strip()
    if not stripped or len(stripped) < 20:
        return False, "content_too_short"

    lower = stripped.lower()

    # Auth/login walls (only reject if content is short — real articles may mention login)
    if len(stripped) < 500:
        for pattern in _AUTH_WALL_PATTERNS:
            if pattern in lower:
                return False, f"auth_wall:{pattern}"

    # Cookie consent / GDPR walls
    if lower.count("cookie") > 3 and len(stripped) < 300:
        return False, "cookie_wall"

    # Stub content like "[Tweet by @author] URL" from old failed extractors
    if stripped.startswith("[") and stripped.endswith("]") and len(stripped) < 100:
        return False, "stub_content"

    # Stub fallbacks from old X/Twitter resolver
    if stripped.startswith("[Tweet by") and "http" in stripped and len(stripped) < 200:
        return False, "tweet_stub"

    # X/Twitter "Something went wrong" error pages (old scraper failures)
    if "something went wrong" in lower and "try again" in lower:
        return False, "scrape_error_page"

    return True, "ok"


def _build_enriched_content(content: str, metadata: dict) -> str:
    """Prepend enrichment metadata (tags, summary, speakers) to episode body.

    This ensures Graphiti sees the metadata during entity extraction,
    improving entity resolution and graph quality.
    """
    header_parts = []
    tags = metadata.get("tags", [])
    summary = metadata.get("summary", "")
    speakers = metadata.get("speakers", [])

    if tags and tags != ["uncategorized"]:
        header_parts.append(f"Tags: {', '.join(tags)}")
    if summary:
        header_parts.append(f"Summary: {summary}")
    discord_ctx = metadata.get("discord_context", "")
    if discord_ctx:
        header_parts.append(f"Shared with context: {discord_ctx}")
    if speakers:
        speaker_strs = []
        for s in speakers:
            name = s.get("name", "")
            role = s.get("role", "")
            if name:
                speaker_strs.append(f"{name} ({role})" if role else name)
        if speaker_strs:
            header_parts.append(f"Speakers: {', '.join(speaker_strs)}")

    if header_parts:
        return "\n".join(header_parts) + "\n\n" + content
    return content


# ── Batch Loading ─────────────────────────────────────────────────────

async def load_batch(limit: int = 200, dry_run: bool = False, concurrency: int = 5):
    """Load a batch of enriched items into Neo4j with concurrent Graphiti calls."""

    # Check persistent circuit breaker before doing anything.
    breaker = staging.is_breaker_tripped()
    if breaker:
        log.warning("Circuit breaker tripped (%s) — skipping batch", breaker["reason"])
        return

    # Reset orphaned 'loading' items from crashed batches.
    orphans = staging.reset_orphaned_loading()
    if orphans:
        log.info("Reset %d orphaned 'loading' items back to 'enriched'", orphans)

    items = staging.get_staged(status="enriched", limit=limit)
    if not items:
        log.info("No enriched items to load")
        return

    total_tokens = sum(i.get("token_estimate", 0) or 0 for i in items)
    estimated_cost = _estimate_cost(total_tokens)
    log.info(
        "Loading %d items (%d tokens, ~$%.4f estimated, concurrency=%d)",
        len(items), total_tokens, estimated_cost, concurrency,
    )

    if dry_run:
        log.info("Dry run — skipping actual load")
        for item in items:
            log.info("  [%s] %s (%d tokens)", item["source_type"], item["source_uri"], item.get("token_estimate", 0))
        return

    batch_id = str(uuid.uuid4())
    item_ids = [str(i["id"]) for i in items]
    staging.update_status(item_ids, "loading", batch_id)

    loaded = 0
    failed = 0
    deduped = 0
    consecutive_failures = 0
    running_cost = 0.0
    credit_auth_error: str | None = None
    cost_ceiling_hit = False
    seen_hashes: set[str] = set()
    seen_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def _load_one(item: dict):
        nonlocal loaded, failed, deduped, consecutive_failures, running_cost
        nonlocal credit_auth_error, cost_ceiling_hit

        async with sem:
            # Check circuit breaker / cost ceiling before each item.
            async with counter_lock:
                if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD or cost_ceiling_hit:
                    item_id = str(item["id"])
                    staging.update_status([item_id], "enriched")  # Return to queue
                    return
                if credit_auth_error:
                    item_id = str(item["id"])
                    staging.update_status([item_id], "enriched")  # Return to queue
                    return

            item_id = str(item["id"])
            source_type = item["source_type"]
            source_uri = item["source_uri"]
            content = item["raw_content"] or ""

            # Quality gate (#12) — reject garbage before expensive Graphiti call.
            loadable, reason = _is_loadable(content, source_type)
            if not loadable:
                log.info("Rejected [%s] %s — %s", source_type, source_uri, reason)
                staging.update_status([item_id], "rejected")
                return

            # Content-hash dedup.
            h = _content_hash(content)
            async with seen_lock:
                if h in seen_hashes:
                    log.info("Skipping content-duplicate: %s", source_uri)
                    staging.update_status([item_id], "deduped")
                    async with counter_lock:
                        deduped += 1
                    return
                seen_hashes.add(h)

            # Enrich content with metadata (#10) — prepend tags/summary/speakers.
            meta = item.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            enriched_content = _build_enriched_content(content, meta)

            # Single attempt — Graphiti handles retries internally (4x with backoff).
            channel = item.get("channel", "")
            try:
                await add_episode(
                    name=source_uri,
                    content=enriched_content,
                    source="text",
                    source_description=f"{source_type} from #{channel}" if channel else source_type,
                    reference_time=item.get("created_at") or datetime.now(timezone.utc),
                )
                staging.update_status([item_id], "loaded", batch_id)
                async with counter_lock:
                    loaded += 1
                    consecutive_failures = 0
                    running_cost += _estimate_cost(item.get("token_estimate", 0) or 0)
                    if running_cost >= BATCH_COST_CEILING_USD:
                        cost_ceiling_hit = True
                        log.warning("Batch cost ceiling ($%.2f) reached — stopping", BATCH_COST_CEILING_USD)
                log.info("Loaded [%s] %s", source_type, source_uri)
                await discord_touch.react(item, "loaded")

            except Exception as exc:
                kind = classify_error(exc)

                if kind == ErrorKind.CREDIT_AUTH:
                    log.error("CREDIT/AUTH error loading %s: %s", source_uri, exc)
                    staging.update_status([item_id], "failed")
                    async with counter_lock:
                        failed += 1
                        credit_auth_error = str(exc)[:500]
                    # Trip persistent breaker (manual reset only for credit errors).
                    staging.trip_breaker(f"CREDIT_AUTH: {str(exc)[:200]}", cooldown_hours=None)
                    return

                if kind == ErrorKind.NON_RETRYABLE:
                    log.warning("Non-retryable error loading %s: %s", source_uri, exc)
                    staging.update_status([item_id], "failed")
                    async with counter_lock:
                        failed += 1
                        consecutive_failures += 1
                    return

                # RETRYABLE — put back in queue for next batch (don't retry now).
                log.warning("Retryable error loading %s (will retry next batch): %s", source_uri, exc)
                staging.update_status([item_id], "enriched")
                async with counter_lock:
                    consecutive_failures += 1

    try:
        tasks = [asyncio.create_task(_load_one(item)) for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await close()

    # ── Batch alerts ──────────────────────────────────────────────────
    if credit_auth_error:
        await discord_touch.alert(
            DISCORD_OPS_ALERTS_CHANNEL,
            "URGENT: Credit/Auth Error — seed-storage loader",
            f"Loader halted due to credit or authentication error.\n\n"
            f"**Error:** {credit_auth_error}\n"
            f"**Batch:** `{batch_id}`\n"
            f"**Progress:** {loaded} loaded, {failed} failed, {deduped} deduped / {len(items)} items\n"
            f"**Circuit breaker tripped** — loader will not run until manually reset.",
            urgent=True,
            color=0xFF0000,
        )
    elif consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        staging.trip_breaker(f"CONSECUTIVE_FAILURES: {consecutive_failures} in batch {batch_id}", cooldown_hours=1)
        await discord_touch.alert(
            DISCORD_OPS_ALERTS_CHANNEL,
            "Circuit Breaker Tripped — seed-storage loader",
            f"Loader stopped after {consecutive_failures} consecutive failures.\n\n"
            f"**Batch:** `{batch_id}`\n"
            f"**Progress:** {loaded} loaded, {failed} failed, {deduped} deduped / {len(items)} items\n"
            f"**Auto-reset:** 1 hour",
            color=0xFF8C00,
        )
    elif cost_ceiling_hit:
        await discord_touch.alert(
            DISCORD_OPS_ALERTS_CHANNEL,
            "Cost Ceiling Hit — seed-storage loader",
            f"Batch stopped at cost ceiling (${BATCH_COST_CEILING_USD:.2f}).\n\n"
            f"**Batch:** `{batch_id}`\n"
            f"**Progress:** {loaded} loaded, {failed} failed, {deduped} deduped / {len(items)} items\n"
            f"**Running cost:** ${running_cost:.4f}",
            color=0xFF8C00,
        )
    elif failed > 0:
        await discord_touch.alert(
            DISCORD_OPS_ALERTS_CHANNEL,
            "seed-storage loader — batch with failures",
            f"**Batch:** `{batch_id}`\n"
            f"**Results:** {loaded} loaded, {failed} failed, {deduped} deduped / {len(items)} items\n"
            f"**Est. cost:** ${running_cost:.4f}",
            color=0xFF8C00,
        )

    log.info("Batch %s complete: %d loaded, %d failed, %d deduped (cost ~$%.4f)",
             batch_id, loaded, failed, deduped, running_cost)


async def estimate():
    """Show cost estimate without loading anything."""
    await load_batch(dry_run=True, limit=5000)


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    # Load .env for local development (Graphiti needs OPENAI_API_KEY in os.environ).
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    concurrency = 5
    for arg in sys.argv:
        if arg.startswith("--concurrency="):
            concurrency = int(arg.split("=")[1])
        elif arg == "--concurrency" and sys.argv.index(arg) + 1 < len(sys.argv):
            concurrency = int(sys.argv[sys.argv.index(arg) + 1])

    if "--dry-run" in sys.argv or "--estimate" in sys.argv:
        asyncio.run(estimate())
    else:
        asyncio.run(load_batch(concurrency=concurrency))
