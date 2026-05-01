"""seed_storage/worker/tasks.py — All Celery tasks.

Tasks:
  enrich_message        — dedup + URL extraction + dispatch → ingest_episode.delay()
  ingest_episode        — Graphiti add_episode() for message + each resolved content
  expand_from_frontier  — fetch frontier URL, resolve, ingest, add child URLs
  scan_frontier         — beat task: pick top frontier URLs and enqueue expand tasks

All tasks are synchronous. Async resolvers are bridged via asyncio.run().
Reaction events are published to seed:reactions (Contract 4).

Retry policy:
  - enrich_message:       3 retries, 60-s delay (network / resolver transients)
  - ingest_episode:       5 retries, 30-s delay (Graphiti / Neo4j transients)
  - expand_from_frontier: 3 retries, 60-s delay
  - scan_frontier:        1 retry,   10-s delay

On final failure → dead_letter() stores the entry for manual replay.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import redis as redis_lib

from seed_storage.circuit_breaker import CircuitBreaker
from seed_storage.config import settings
from seed_storage.cost_tracking import CostTracker
from seed_storage.dedup import DedupStore, url_hash
from seed_storage.enrichment.dispatcher import ContentDispatcher
from seed_storage.enrichment.models import ResolvedContent
from seed_storage.expansion.frontier import (
    add_to_frontier,
    get_frontier_meta,
    remove_from_frontier,
)
from seed_storage.expansion.policies import compute_priority
from seed_storage.notifications import send_alert
from seed_storage.rate_limiting import RateLimiter
from seed_storage.worker.app import app
from seed_storage.worker.dead_letters import dead_letter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REACTIONS_CHANNEL = "seed:reactions"
GROUP_ID = "ant-haul"

# URL regex for extracting links from message content.
# Matches http:// and https:// URLs (greedy until whitespace or common terminators).
_URL_PATTERN = re.compile(r"https?://[^\s\]\[<>\"']+", re.IGNORECASE)

# Cost tracking defaults (one call ≈ one add_episode)
_COST_PER_CALL = 0.0005  # ~$0.0005 per episode (gpt-4o-mini estimate)


# ---------------------------------------------------------------------------
# Redis client helpers
# ---------------------------------------------------------------------------


def _get_redis() -> redis_lib.Redis:
    """Return a synchronous Redis client from settings.REDIS_URL."""
    return redis_lib.from_url(settings.REDIS_URL)


def _get_dedup_messages(r: redis_lib.Redis) -> DedupStore:
    return DedupStore(r, "seed:seen_messages")


def _get_dedup_urls(r: redis_lib.Redis) -> DedupStore:
    return DedupStore(r, "seed:seen_urls")


def _get_dedup_ingested(r: redis_lib.Redis) -> DedupStore:
    return DedupStore(r, "seed:ingested_content")


def _get_cost_tracker(r: redis_lib.Redis) -> CostTracker:
    return CostTracker(r, settings.DAILY_LLM_BUDGET, _COST_PER_CALL)


def _get_rate_limiter(r: redis_lib.Redis) -> RateLimiter:
    return RateLimiter(r, "seed:ratelimit:pipeline", settings.RATE_LIMIT_PER_MINUTE)


def _get_circuit_breaker(r: redis_lib.Redis) -> CircuitBreaker:
    return CircuitBreaker(r, "pipeline")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_urls(text: str) -> list[str]:
    """Extract all http/https URLs from a text string."""
    return _URL_PATTERN.findall(text)


def _publish_reaction(r: redis_lib.Redis, message_id: str, channel_id: str, emoji: str) -> None:
    """Publish a reaction event to seed:reactions (Contract 4). Never raises."""
    try:
        event = {"message_id": message_id, "channel_id": channel_id, "emoji": emoji}
        r.publish(REACTIONS_CHANNEL, json.dumps(event))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to publish reaction event: %s", exc)


def _source_description_message(source_type: str, source_channel: str) -> str:
    """Build source_description for message episodes per spec Section 4."""
    return f"{source_type.title()} #{source_channel}"


def _source_description_content(source_type: str, source_channel: str, content_type: str) -> str:
    """Build source_description for content episodes per spec Section 4."""
    return f"content_from_{source_type.title()}_{source_channel}:{content_type}"


def _build_content_payload(resolved: ResolvedContent, meta: dict) -> dict:
    """Build enriched_payload for expansion-discovered content (Contract 3)."""
    return {
        "message": {
            "source_type": "expansion",
            "source_id": f"frontier_{meta['url_hash']}",
            "source_channel": meta["source_channel"],
            "author": "system",
            "content": f"Expanded from {meta['discovered_from_url']}",
            "timestamp": meta["discovered_at"],
            "attachments": [],
            "metadata": {
                "frontier_depth": meta["depth"],
                "discovered_from_url": meta["discovered_from_url"],
                "discovered_from_source_id": meta["discovered_from_source_id"],
            },
        },
        "resolved_contents": [resolved.to_dict()],
    }


def _get_domain(url: str) -> str:
    """Extract registered domain from URL (e.g. 'github.com')."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""


def _classify_resolver_hint(url: str) -> str:
    """Classify a URL into a resolver_hint based on domain."""
    domain = _get_domain(url)
    if domain in ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"):
        return "youtube"
    if domain in ("github.com", "www.github.com"):
        return "github"
    if domain in ("twitter.com", "www.twitter.com", "x.com", "www.x.com"):
        return "tweet"
    if domain in ("instagram.com", "www.instagram.com"):
        return "instagram"
    if domain in ("arxiv.org", "www.arxiv.org"):
        return "webpage"
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if any(path.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".avi")):
        return "video"
    if any(path.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return "image"
    return "webpage"


# ---------------------------------------------------------------------------
# Task: enrich_message
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.enrich_message",
    queue="raw_messages",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
)
def enrich_message(self, raw_payload: dict) -> None:
    """Dedup + URL extraction + resolve → ingest_episode.delay().

    Contract 1 error rules:
      - content empty AND attachments empty → skip (log DEBUG)
      - author is bot → skip (log DEBUG)

    Steps:
      1. Validate raw_payload shape.
      2. Dedup by {source_type}:{source_id} — skip if already seen.
      3. Publish 📥 reaction to Discord via seed:reactions.
      4. Extract URLs from content + attachment URLs.
      5. Per URL: canonicalize + dedup (seed:seen_urls). Resolve skipped URLs.
      6. Dispatch all new URLs via ContentDispatcher (asyncio.run).
      7. Build enriched_payload (Contract 2) and enqueue ingest_episode.delay().
      8. Publish ⚙️ reaction.
    """
    source_type = raw_payload.get("source_type", "")
    source_id = raw_payload.get("source_id", "")
    content = raw_payload.get("content", "") or ""
    attachments = raw_payload.get("attachments", []) or []
    author = raw_payload.get("author", "") or ""
    channel_id = raw_payload.get("metadata", {}).get("channel_id", "")

    # Error contract: empty content + no attachments
    content = content.strip() if content else ""
    if not content and not attachments:
        logger.debug("enrich_message: skipping empty message source_id=%s", source_id)
        return

    # Error contract: bot author
    if _is_bot_author(author):
        logger.debug(
            "enrich_message: skipping bot message source_id=%s author=%s", source_id, author
        )
        return

    try:
        r = _get_redis()
        dedup_messages = _get_dedup_messages(r)
        dedup_urls = _get_dedup_urls(r)

        # Dedup message
        msg_key = f"{source_type}:{source_id}"
        if dedup_messages.seen_or_mark(msg_key):
            logger.debug("enrich_message: duplicate message %s, skipping", msg_key)
            return

        # Publish 📥 reaction (message received)
        _publish_reaction(r, source_id, channel_id, "📥")

        # Extract all URLs from content + attachments
        content_urls = _extract_urls(content)
        all_urls = list(content_urls) + list(attachments)

        # Per-URL dedup
        new_urls: list[str] = []
        for url in all_urls:
            h = url_hash(url)
            if not dedup_urls.seen_or_mark(h):
                new_urls.append(url)
            else:
                logger.debug("enrich_message: duplicate URL %s, skipping", url)

        # Resolve URLs
        resolved_contents: list[ResolvedContent] = []
        if new_urls:
            dispatcher = ContentDispatcher()
            resolved_contents = asyncio.run(_resolve_urls(dispatcher, new_urls))

        # Build enriched_payload (Contract 2)
        enriched_payload = {
            "message": raw_payload,
            "resolved_contents": [rc.to_dict() for rc in resolved_contents],
        }

        # Persist resolved content to Postgres staging before Celery handoff.
        # This ensures the data is durable and replayable even if ingest_episode fails.
        # URL content items are staged at "processed" (content already resolved).
        url_uris = _persist_to_staging(raw_payload, resolved_contents)
        enriched_payload["url_content_uris"] = url_uris

        # Enqueue for graph ingest
        ingest_episode.delay(enriched_payload)

        # Publish ⚙️ reaction (processing started)
        _publish_reaction(r, source_id, channel_id, "⚙️")

        # Publish content-type emoji for each resolved content type (one per type).
        _publish_content_type_reactions(r, source_id, channel_id, resolved_contents)

        logger.info(
            "enrich_message: enqueued source_id=%s urls=%d resolved=%d",
            source_id,
            len(new_urls),
            len(resolved_contents),
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("enrich_message: failed source_id=%s: %s", source_id, exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            dead_letter("enrich_message", raw_payload, exc, self.request.retries)


def _publish_content_type_reactions(
    r: redis_lib.Redis,
    message_id: str,
    channel_id: str,
    resolved_contents: "list[ResolvedContent]",
) -> None:
    """Publish one content-type emoji per distinct content type resolved. Never raises."""
    # content_type → emoji
    CONTENT_EMOJI = {
        "youtube":    "🎬",
        "tweet":      "🐦",
        "github":     "📦",
        "instagram":  "📸",
        "webpage":    "🌐",
        "pdf":        "📄",
        "video":      "🎥",
        "image":      "🖼️",
        "audiobook":  "📚",
        "unknown":    "❓",
    }
    seen: set[str] = set()
    for rc in resolved_contents:
        ct = rc.content_type
        if ct in seen or ct not in CONTENT_EMOJI:
            continue
        seen.add(ct)
        _publish_reaction(r, message_id, channel_id, CONTENT_EMOJI[ct])


def _is_bot_author(author: str) -> bool:
    """Heuristic: author names ending in '#Bot' or containing '[BOT]' etc.

    The primary bot filter is the `message.author.bot` flag in the Discord
    ingestion layer. This is a secondary textual guard for batch imports where
    we only have the display name string.
    Note: batch.py already checks isBot flag; this guard handles edge cases.
    """
    # We intentionally keep this light — the contract says "author is bot",
    # which is enforced at the ingestion layer. We don't apply name heuristics
    # here to avoid false positives on human usernames.
    return False


async def _resolve_urls(dispatcher: ContentDispatcher, urls: list[str]) -> list[ResolvedContent]:
    """Dispatch all URLs concurrently and return results."""
    import asyncio as _asyncio

    tasks = [dispatcher.dispatch(url) for url in urls]
    return list(await _asyncio.gather(*tasks))


def _persist_to_staging(raw_payload: dict, resolved_contents: list[ResolvedContent]) -> list[str]:
    """Persist message + resolved content to Postgres staging.

    Stages the original message at 'staged' and each resolved content item
    at 'processed' (content already extracted by the dispatcher). Returns
    the list of source_uris for staged URL content items so ingest_episode
    can process them.
    """
    from seed_storage import staging

    source_type = raw_payload.get("source_type", "unknown")
    source_id = raw_payload.get("source_id", "")
    source_channel = raw_payload.get("source_channel", "unknown")
    author = raw_payload.get("author", "")
    content = raw_payload.get("content", "") or ""
    timestamp = raw_payload.get("timestamp", "")

    # Stage the message itself (dedup by source_uri)
    msg_uri = f"discord://{source_channel}/{source_id}"
    staging.stage(
        source_type=source_type,
        source_uri=msg_uri,
        raw_content=content,
        author=author,
        channel=source_channel,
        created_at=timestamp,
        metadata={"discord_context": content[:200]},
    )

    # Stage each resolved content item at "processed" — content is already
    # extracted by the dispatcher, so skip the processor stage.
    url_uris: list[str] = []
    for rc in resolved_contents:
        if not rc.text and not rc.transcript and not rc.summary:
            continue
        text = rc.text or rc.transcript or rc.summary or ""
        sid = staging.stage(
            source_type=rc.content_type,
            source_uri=rc.source_url,
            raw_content=text,
            author=author,
            channel=source_channel,
            created_at=timestamp,
            status="processed",
            metadata={
                "title": rc.title,
                "expansion_urls": rc.expansion_urls[:10],
                "discord_context": content[:200],
            },
        )
        if sid:
            url_uris.append(rc.source_url)
    return url_uris


# ---------------------------------------------------------------------------
# Task: ingest_episode
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.ingest_episode",
    queue="graph_ingest",
    max_retries=5,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
)
def ingest_episode(self, enriched_payload: dict) -> None:
    """Extract entities from enriched payload and load into Neo4j.

    Uses the new extraction + resolution + graph.py pipeline (no Graphiti).
    Steps:
      1. Check cost budget / circuit breaker.
      2. Find the staging row (persisted by _persist_to_staging in enrich_message).
      3. Run extraction (if not already extracted).
      4. Run loader to resolve entities and write to Neo4j.
      5. Add expansion_urls to frontier.
      6. Publish reactions.
    """
    message = enriched_payload.get("message", {})
    resolved_contents_raw = enriched_payload.get("resolved_contents", [])
    url_content_uris = enriched_payload.get("url_content_uris", [])

    source_id = message.get("source_id", "")
    source_channel = message.get("source_channel", "unknown")
    channel_id = message.get("metadata", {}).get("channel_id", "")

    try:
        r = _get_redis()
        cost_tracker = _get_cost_tracker(r)
        circuit_breaker = _get_circuit_breaker(r)

        if cost_tracker.is_budget_exceeded():
            logger.warning("ingest_episode: daily budget exceeded, retrying later")
            send_alert(
                "Seed-storage daily LLM budget exceeded — graph ingest paused.",
                debounce_key="budget_exceeded",
            )
            try:
                raise self.retry(countdown=300)
            except self.MaxRetriesExceededError:
                dead_letter(
                    "ingest_episode",
                    enriched_payload,
                    Exception("budget_exceeded"),
                    self.request.retries,
                )
                return

        if circuit_breaker.is_open():
            logger.warning("ingest_episode: circuit breaker open, skipping")
            return

        from seed_storage import staging as _staging
        from seed_storage.extraction import extract_one
        from seed_storage.preseed import get_alias_map, init_preseed_table

        init_preseed_table()
        alias_map = get_alias_map()

        # --- Process the Discord message staging row ---
        msg_uri = f"discord://{source_channel}/{source_id}"
        item = _staging.get_by_uri(msg_uri)

        if item and item["status"] in ("staged", "processed", "enriched"):
            result = extract_one(item, alias_map=alias_map)
            _staging.patch_metadata(
                str(item["id"]),
                {
                    "extraction": {
                        **result.model_dump(),
                        "extracted_at": datetime.now(tz=UTC).isoformat(),
                    },
                },
            )
            _staging.update_status([str(item["id"])], "extracted")
            asyncio.run(_load_item_to_graph(item, alias_map))
            cost_tracker.increment()
            circuit_breaker.record_success()
        elif item and item["status"] == "extracted":
            asyncio.run(_load_item_to_graph(item, alias_map))
            cost_tracker.increment()
            circuit_breaker.record_success()

        # --- Process URL content staging rows (staged at "processed") ---
        for uri in url_content_uris:
            try:
                url_item = _staging.get_by_uri(uri)
                if not url_item or url_item["status"] in ("loaded", "extracted", "enriched"):
                    continue
                # Enrich (tags + summary)
                asyncio.run(_enrich_and_extract_item(url_item, alias_map))
                # Reload after enrichment+extraction
                url_item = _staging.get_by_uri(uri)
                if url_item and url_item["status"] == "extracted":
                    asyncio.run(_load_item_to_graph(url_item, alias_map))
                    cost_tracker.increment()
                    logger.info("ingest_episode: loaded URL content %s", uri)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ingest_episode: failed to process URL content %s: %s", uri, exc)

        # Add expansion_urls to frontier
        all_rcs = [ResolvedContent.from_dict(rc_dict) for rc_dict in resolved_contents_raw]
        for rc in all_rcs:
            depth = int(message.get("metadata", {}).get("frontier_depth", 0))
            child_depth = depth + 1
            if child_depth > settings.HARD_DEPTH_CEILING:
                continue
            expansion_urls = rc.expansion_urls[: settings.MAX_EXPANSION_BREADTH]
            for exp_url in expansion_urls:
                h = url_hash(exp_url)
                domain = _get_domain(exp_url)
                hint = _classify_resolver_hint(exp_url)
                priority = compute_priority(
                    depth=child_depth,
                    resolver_hint=hint,
                    domain=domain,
                    source_channel=source_channel,
                )
                meta = {
                    "url": exp_url,
                    "url_hash": h,
                    "discovered_from_url": rc.source_url,
                    "discovered_from_source_id": source_id,
                    "source_channel": source_channel,
                    "depth": child_depth,
                    "resolver_hint": hint,
                    "discovered_at": datetime.now(tz=UTC).isoformat(),
                }
                try:
                    add_to_frontier(r, h, priority, meta)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ingest_episode: failed to add %s to frontier: %s", exp_url, exc)

        _publish_reaction(r, source_id, channel_id, "🏷️")
        _publish_reaction(r, source_id, channel_id, "🧠")

        logger.info(
            "ingest_episode: completed source_id=%s (%d url items)",
            source_id,
            len(url_content_uris),
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("ingest_episode: failed source_id=%s: %s", source_id, exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            dead_letter("ingest_episode", enriched_payload, exc, self.request.retries)


async def _enrich_and_extract_item(item: dict, alias_map: dict) -> None:
    """Run enricher + extractor on a single staging item (for URL content in real-time path).

    Takes an item at status 'processed', enriches it (tags/summary), extracts entities,
    and updates status to 'extracted'.
    """
    import json

    from seed_storage import config
    from seed_storage import staging as _staging
    from seed_storage.extraction import extract_one

    item_id = str(item["id"])

    # Enrich: add tags + summary via LLM
    if config.LLM_API_KEY:
        try:
            from ingestion.enricher import (
                ENRICHER_SYSTEM,
                _build_llm_client,
                _get_existing_tags,
                _llm_chat,
                _upsert_tags,
                init_tags_table,
            )

            init_tags_table()
            client, provider = _build_llm_client()
            existing_tags = _get_existing_tags()

            content = (item.get("raw_content") or "")[:3000]
            source_type = item.get("source_type", "unknown")

            prompt = ENRICHER_SYSTEM.replace("{existing_tags}", ", ".join(existing_tags[:100]))
            user_msg = f"Source type: {source_type}\nContent:\n{content}"

            raw = await _llm_chat(client, provider, prompt, user_msg, config.LLM_MODEL)

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            data = json.loads(raw)
            tags = data.get("tags", [])
            summary = data.get("summary", "")

            meta = item.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta["tags"] = tags
            meta["summary"] = summary
            _staging.update_content(item_id, item["raw_content"], metadata=meta, status="enriched")

            if tags:
                _upsert_tags(tags)
        except Exception:
            logger.debug(
                "Enrichment failed for URL item %s, continuing to extraction",
                item_id,
                exc_info=True,
            )
            _staging.update_status([item_id], "enriched")
    else:
        _staging.update_status([item_id], "enriched")

    # Re-read after enrichment
    item = _staging.get_by_id(item_id)
    if not item or item["status"] != "enriched":
        return

    # Extract entities
    result = extract_one(item, alias_map=alias_map)
    _staging.patch_metadata(
        item_id,
        {
            "extraction": {
                **result.model_dump(),
                "extracted_at": datetime.now(tz=UTC).isoformat(),
            },
        },
    )
    _staging.update_status([item_id], "extracted")


async def _load_item_to_graph(item: dict, alias_map: dict) -> None:
    """Load a single staging item into Neo4j (async helper for ingest_episode)."""
    from ingestion.loader import _load_one_item
    from seed_storage.graph import close, get_driver

    driver = await get_driver()
    try:
        await _load_one_item(item, alias_map, None, driver, batch_id=None)
    finally:
        await close()


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO 8601 timestamp string. Returns utcnow() on failure."""
    if not timestamp_str:
        return datetime.now(tz=UTC)
    try:
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# Task: expand_from_frontier
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.expand_from_frontier",
    queue="graph_ingest",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    reject_on_worker_lost=True,
)
def expand_from_frontier(self, url_hash_str: str) -> None:
    """Fetch + resolve a frontier URL and write it to the graph.

    Steps:
      1. Load frontier metadata for url_hash_str.
      2. Check depth ceiling — skip if at/above HARD_DEPTH_CEILING.
      3. Check ingested_content dedup — skip if already ingested.
      4. Resolve URL via ContentDispatcher (asyncio.run).
      5. Build enriched_payload via _build_content_payload() (Contract 3).
      6. Enqueue ingest_episode.delay() with the payload.
      7. Add child expansion_urls to frontier (capped at MAX_EXPANSION_BREADTH).
      8. Remove url_hash_str from frontier.
    """
    try:
        r = _get_redis()
        dedup_ingested = _get_dedup_ingested(r)

        # Load frontier metadata
        meta = get_frontier_meta(r, url_hash_str)
        if meta is None:
            logger.warning("expand_from_frontier: no metadata for %s, skipping", url_hash_str)
            return

        url = meta.get("url", "")
        if not url:
            logger.warning("expand_from_frontier: empty URL in metadata for %s", url_hash_str)
            remove_from_frontier(r, url_hash_str)
            return

        depth = int(meta.get("depth", 0))

        # Depth ceiling check
        if depth >= settings.HARD_DEPTH_CEILING:
            logger.debug(
                "expand_from_frontier: depth ceiling reached (%d >= %d) for %s",
                depth,
                settings.HARD_DEPTH_CEILING,
                url,
            )
            remove_from_frontier(r, url_hash_str)
            return

        # Already-ingested check
        if dedup_ingested.seen_or_mark(url_hash_str):
            logger.debug("expand_from_frontier: already ingested %s", url)
            remove_from_frontier(r, url_hash_str)
            return

        # Resolve URL
        dispatcher = ContentDispatcher()
        resolved = asyncio.run(dispatcher.dispatch(url))

        # Ensure url_hash is in meta for _build_content_payload
        if "url_hash" not in meta:
            meta["url_hash"] = url_hash_str
        if "discovered_from_url" not in meta:
            meta["discovered_from_url"] = url
        if "discovered_from_source_id" not in meta:
            meta["discovered_from_source_id"] = ""
        if "source_channel" not in meta:
            meta["source_channel"] = "unknown"
        if "discovered_at" not in meta:
            meta["discovered_at"] = datetime.now(tz=UTC).isoformat()

        # Build enriched_payload (Contract 3)
        enriched = _build_content_payload(resolved, meta)

        # Enqueue graph ingest
        ingest_episode.delay(enriched)

        # Add child expansion URLs to frontier
        child_depth = depth + 1
        source_channel = meta.get("source_channel", "unknown")
        discovered_at = datetime.now(tz=UTC).isoformat()

        expansion_urls = resolved.expansion_urls[: settings.MAX_EXPANSION_BREADTH]
        for exp_url in expansion_urls:
            h = url_hash(exp_url)
            domain = _get_domain(exp_url)
            hint = _classify_resolver_hint(exp_url)
            priority = compute_priority(
                depth=child_depth,
                resolver_hint=hint,
                domain=domain,
                source_channel=source_channel,
            )
            child_meta = {
                "url": exp_url,
                "url_hash": h,
                "discovered_from_url": url,
                "discovered_from_source_id": meta.get("discovered_from_source_id", ""),
                "source_channel": source_channel,
                "depth": child_depth,
                "resolver_hint": hint,
                "discovered_at": discovered_at,
            }
            try:
                add_to_frontier(r, h, priority, child_meta)
            except Exception as exc:  # noqa: BLE001
                logger.warning("expand_from_frontier: failed to add child URL %s: %s", exp_url, exc)

        # Remove processed URL from frontier
        remove_from_frontier(r, url_hash_str)

        logger.info(
            "expand_from_frontier: expanded %s depth=%d child_urls=%d",
            url,
            depth,
            len(expansion_urls),
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("expand_from_frontier: failed %s: %s", url_hash_str, exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            dead_letter(
                "expand_from_frontier",
                {"url_hash": url_hash_str},
                exc,
                self.request.retries,
            )


# ---------------------------------------------------------------------------
# Task: scan_frontier
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.scan_frontier",
    queue="graph_ingest",
    max_retries=1,
    default_retry_delay=10,
    acks_late=True,
)
def scan_frontier(self) -> int:
    """Beat task: scan frontier and enqueue expand tasks for top-priority URLs.

    Delegates to expansion.scanner.scan_frontier() which:
      - Returns 0 immediately when FRONTIER_AUTO_ENABLED=False.
      - Picks top-N URLs from frontier (batch_size=MAX_EXPANSION_BREADTH).
      - Enqueues expand_from_frontier.delay(url_hash) for each.

    Returns count of enqueued tasks.
    """
    try:
        from seed_storage.expansion.scanner import scan_frontier as _scan  # noqa: PLC0415

        count = _scan()
        if count:
            logger.info("scan_frontier: enqueued %d expand tasks", count)
        else:
            logger.debug("scan_frontier: nothing to enqueue")
        return count
    except Exception as exc:  # noqa: BLE001
        logger.error("scan_frontier: failed: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error("scan_frontier: max retries exceeded")
            return 0


# ---------------------------------------------------------------------------
# Utility: unused but exported for Contract 3 completeness
# ---------------------------------------------------------------------------


def build_content_payload(resolved: ResolvedContent, meta: dict) -> dict:
    """Public alias for _build_content_payload (Contract 3)."""
    return _build_content_payload(resolved, meta)


# ---------------------------------------------------------------------------
# Task: post_daily_digest
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.post_daily_digest",
    queue="graph_ingest",
    max_retries=2,
    default_retry_delay=120,
    acks_late=True,
)
def post_daily_digest(self) -> int:
    """Beat task: post a daily summary of loaded items to Discord.

    Queries seed_staging for items loaded in the last 24 hours, builds
    a grouped summary, and posts it to #seed-storage via the bot token.
    """
    try:
        from seed_storage.digest import post_digest

        count = post_digest(hours=24)
        if count:
            logger.info("post_daily_digest: posted digest with %d items", count)
        else:
            logger.info("post_daily_digest: no items to digest")
        return count
    except Exception as exc:  # noqa: BLE001
        logger.error("post_daily_digest: failed: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error("post_daily_digest: max retries exceeded")
            return 0


# ---------------------------------------------------------------------------
# Task: generate_graph_index
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.generate_graph_index",
    queue="graph_ingest",
    max_retries=2,
    default_retry_delay=60,
)
def generate_graph_index(self) -> int:
    """Generate the graph index document and store in Neo4j __Meta__ node.

    Runs daily at 12:30 UTC (15 min after daily digest).
    Also posts a summary to #seed-storage Discord channel.
    """
    try:
        import asyncio
        from scripts.generate_index import generate
        asyncio.run(generate())
        logger.info("generate_graph_index: completed")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_graph_index: failed: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return 0


# ---------------------------------------------------------------------------
# Task: run_graph_lint
# ---------------------------------------------------------------------------


@app.task(
    bind=True,
    name="seed_storage.worker.tasks.run_graph_lint",
    queue="graph_ingest",
    max_retries=2,
    default_retry_delay=60,
)
def run_graph_lint(self) -> int:
    """Run graph health check and store report in Neo4j __Meta__ node.

    Runs weekly Monday at 13:00 UTC.
    Also posts a summary to #seed-storage Discord channel.
    """
    try:
        import asyncio
        from scripts.lint_graph import lint
        asyncio.run(lint())
        logger.info("run_graph_lint: completed")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("run_graph_lint: failed: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return 0
