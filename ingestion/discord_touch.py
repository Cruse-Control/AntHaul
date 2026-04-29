"""Discord pipeline status reactions — shared module for all pipeline steps.

Updates emoji reactions on the original Discord message to show pipeline progress.
Fails silently — a missing reaction should never block the pipeline.

Usage from any pipeline step:
    await discord_touch.react(item, "processed")
    await discord_touch.react(item, "failed", error_msg="Details here")
"""

from __future__ import annotations

import logging
import os
from urllib.parse import quote as url_quote

import httpx

log = logging.getLogger("discord_touch")

DISCORD_API = "https://discord.com/api/v10"

# Pipeline stage → emoji mapping.
STAGE_EMOJI = {
    "staged":    "\U0001f4e5",      # 📥
    "processed": "\u2699\ufe0f",    # ⚙️
    "enriched":  "\U0001f3f7\ufe0f",# 🏷️
    "loaded":    "\U0001f9e0",      # 🧠
    "submodule": "\U0001f517",      # 🔗
    "failed":    "\u274c",          # ❌
    "deduped":   "\U0001f501",      # 🔁
}

ALL_STAGE_EMOJI = set(STAGE_EMOJI.values())


def _get_token() -> str:
    """Resolve Discord bot token from file or env, stripping any 'Bot ' prefix."""
    token_file = os.environ.get("DISCORD_BOT_ANT_FARM_FILE_PATH", "")
    if token_file and os.path.exists(token_file):
        token = open(token_file).read().strip()
    else:
        token = os.environ.get("DISCORD_BOT_ANT_FARM_TOKEN", "")
    if token.startswith("Bot "):
        token = token[4:]
    return token


def _encode_emoji(emoji: str) -> str:
    """URL-encode an emoji for the Discord reaction API."""
    return url_quote(emoji)


async def react(item: dict, status: str, error_msg: str = ""):
    """Update the Discord reaction on the original message to reflect pipeline status.

    Args:
        item: A staging row dict with metadata containing discord_msg_id + discord_channel_id.
        status: The new pipeline status (staged, processed, enriched, loaded, failed, etc.).
        error_msg: Optional error message — posted as a thread reply on failure.
    """
    meta = item.get("metadata") or {}
    if isinstance(meta, str):
        import json
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return

    msg_id = meta.get("discord_msg_id")
    channel_id = meta.get("discord_channel_id")
    if not msg_id or not channel_id:
        return

    token = _get_token()
    if not token:
        return

    emoji = STAGE_EMOJI.get(status)
    if not emoji:
        return

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            headers = {"Authorization": f"Bot {token}"}
            base = f"{DISCORD_API}/channels/{channel_id}/messages/{msg_id}"

            # Add new stage emoji (accumulate rather than replace to avoid rate limits).
            resp = await http.put(
                f"{base}/reactions/{_encode_emoji(emoji)}/@me",
                headers=headers,
            )
            if resp.status_code == 429:
                # Respect rate limit — wait and retry once.
                import json as _json
                retry_after = _json.loads(resp.content).get("retry_after", 1)
                import asyncio
                await asyncio.sleep(retry_after)
                await http.put(
                    f"{base}/reactions/{_encode_emoji(emoji)}/@me",
                    headers=headers,
                )

            # On failure, reply in thread with the error message.
            if status == "failed" and error_msg:
                await http.post(
                    f"{DISCORD_API}/channels/{channel_id}/messages",
                    headers=headers,
                    json={
                        "content": f"\u26a0\ufe0f {error_msg[:1900]}",
                        "message_reference": {"message_id": msg_id},
                    },
                )

    except Exception:
        # Never let Discord errors block the pipeline.
        log.debug("Discord touch failed for msg %s (non-blocking)", msg_id, exc_info=True)


async def alert(
    channel_id: str,
    title: str,
    message: str,
    *,
    color: int = 0xFF0000,
    urgent: bool = False,
) -> None:
    """Post an alert embed to a Discord channel. Non-blocking, fails silently.

    Args:
        channel_id: Discord channel ID to post to.
        title: Embed title (short, max 256 chars).
        message: Embed description (multiline, max 4000 chars).
        color: Embed sidebar color. Red=0xFF0000, Orange=0xFF8C00, Green=0x00FF00.
        urgent: If True, prepend @here mention to ping channel members.
    """
    if not channel_id:
        log.debug("discord_touch.alert: channel_id is empty — skipping (DISCORD_OPS_ALERTS_CHANNEL not configured)")
        return

    token = _get_token()
    if not token:
        log.warning("No Discord token available — skipping alert")
        return

    embed = {
        "title": title[:256],
        "description": message[:4000],
        "color": color,
    }
    payload: dict = {"embeds": [embed]}
    if urgent:
        payload["content"] = "@here"

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(
                f"{DISCORD_API}/channels/{channel_id}/messages",
                headers={"Authorization": f"Bot {token}"},
                json=payload,
            )
            if resp.status_code >= 400:
                log.warning("Discord alert failed: %d %s", resp.status_code, resp.text[:200])
    except Exception:
        log.debug("Discord alert failed (non-blocking)", exc_info=True)
