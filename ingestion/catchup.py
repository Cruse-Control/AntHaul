"""
Startup catch-up: fetch recent Discord messages via REST API and
process any that were missed during bot downtime.

Called from watcher.py on_ready. Fetches the last N messages from each
watched channel, checks if each was already staged, and stages new ones.
If after_timestamp is provided (from seed_bot_state), only fetches messages
newer than that time using Discord's 'after' snowflake parameter.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.classifier import Platform, classify, extract_urls
from ingestion.signal_filter import is_noise
from seed_storage import staging

log = logging.getLogger("catchup")

DISCORD_API = "https://discord.com/api/v10"
CATCHUP_LIMIT = 50  # Max messages to backfill per channel on startup


async def run_catchup(
    token: str,
    watched_channel_ids: set[int],
    after_timestamp: Optional[str] = None,
) -> int:
    """Fetch recent messages from watched channels and stage any not yet seen.

    Args:
        token: Discord bot token (without 'Bot ' prefix)
        watched_channel_ids: Set of channel IDs to check
        after_timestamp: ISO 8601 timestamp. If provided, only fetch messages
                        newer than this time. If None, fetches last CATCHUP_LIMIT messages.

    Returns:
        Number of messages staged.
    """
    if after_timestamp is not None:
        ts_ms = int(datetime.fromisoformat(after_timestamp).timestamp() * 1000)
        after_snowflake = (ts_ms - 1420070400000) << 22
        params: dict = {"limit": CATCHUP_LIMIT, "after": str(after_snowflake)}
    else:
        params = {"limit": CATCHUP_LIMIT}

    total_staged = 0

    async with httpx.AsyncClient(timeout=15) as http:
        for channel_id in watched_channel_ids:
            try:
                resp = await http.get(
                    f"{DISCORD_API}/channels/{channel_id}/messages",
                    headers={"Authorization": f"Bot {token}"},
                    params=params,
                )
                if resp.status_code == 403:
                    log.debug("Catchup: no access to channel %s (403) — skipping", channel_id)
                    continue
                if resp.status_code == 404:
                    log.debug("Catchup: channel %s not found (404) — skipping", channel_id)
                    continue
                if resp.status_code != 200:
                    log.warning(
                        "Catchup: unexpected status %s for channel %s — skipping",
                        resp.status_code,
                        channel_id,
                    )
                    continue

                messages = resp.json()
                for msg in messages:
                    already_staged = staging.get_by_discord_msg_id(msg["id"])
                    if already_staged:
                        continue
                    staged = _stage_catchup_message(channel_id, msg)
                    total_staged += staged

            except Exception as exc:
                log.warning("Catchup: error fetching channel %s: %s", channel_id, exc)
                continue

    log.info("Catchup complete — staged %d new messages across %d channels", total_staged, len(watched_channel_ids))
    return total_staged


def _stage_catchup_message(channel_id: int, msg: dict) -> int:
    """Stage a single backfilled Discord message. Returns number of items staged."""
    content = msg.get("content", "").strip()
    attachment_urls = [a["url"] for a in msg.get("attachments", [])]

    if not content and not attachment_urls:
        return 0

    meta = {
        "discord_msg_id": msg["id"],
        "discord_channel_id": str(channel_id),
        "discord_guild_id": msg.get("guild_id"),
        "discord_author_id": msg.get("author", {}).get("id"),
        "discord_timestamp": msg.get("timestamp"),
        "backfilled": True,
    }
    author = msg.get("author", {}).get("username", "unknown")
    created_at = msg.get("timestamp")
    staged_count = 0

    # Attachment-only message: stage each attachment as web source
    if not content and attachment_urls:
        for url in attachment_urls:
            sid = staging.stage(
                source_type=Platform.WEB.value,
                source_uri=url,
                raw_content=url,
                author=author,
                channel=str(channel_id),
                created_at=created_at,
                metadata=meta,
            )
            if sid:
                staged_count += 1
        return staged_count

    # Content with URLs: classify and stage each URL
    urls = extract_urls(content)
    if urls:
        for url in urls:
            category = classify(url)
            sid = staging.stage(
                source_type=category.value,
                source_uri=url,
                raw_content=content,
                author=author,
                channel=str(channel_id),
                created_at=created_at,
                metadata=meta,
            )
            if sid:
                staged_count += 1
        return staged_count

    # Plain text (no URLs): filter noise, then stage
    if is_noise(content):
        return 0

    msg_id = msg["id"]
    guild_id = msg.get("guild_id", "0")
    msg_uri = f"discord://{guild_id}/{channel_id}/{msg_id}"
    sid = staging.stage(
        source_type=Platform.PLAIN_TEXT.value,
        source_uri=msg_uri,
        raw_content=content,
        author=author,
        channel=str(channel_id),
        created_at=created_at,
        metadata=meta,
    )
    if sid:
        staged_count += 1

    return staged_count
