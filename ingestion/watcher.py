"""Discord watcher — Step 1: capture messages, classify URLs, stage for processing.

Applies signal filter to drop noise. Captures discord_msg_id in metadata.
Routes content from #ant-food-router to the correct typed channel.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import discord
import httpx

from ingestion.catchup import run_catchup
from ingestion.classifier import Platform, classify, extract_urls
from ingestion.pipeline_worker import PipelineWorker
from ingestion.signal_filter import is_noise
from seed_storage import staging
from seed_storage.staging import get_bot_last_seen, upsert_bot_last_seen

log = logging.getLogger("watcher")

# Pipeline worker — processes URLs end-to-end as they arrive.
# Initialized in start_watcher() if API keys are available.
_pipeline: PipelineWorker | None = None

# Router channels — content here gets reposted to the correct typed channel.
# Includes ant-food-router + external intel channels (we can't control what they send).
ROUTER_CHANNEL_IDS = {
    1487576638269948167,  # #ant-food-router
    1488083965389836369,  # #scouting-external-colonies
    1488084249188765796,  # #hive-mind-announcements
}

# Platform → destination channel ID for routing.
ROUTE_MAP = {
    Platform.INSTAGRAM: 1487648814280998963,
    Platform.YOUTUBE: 1490082833627218112,
    Platform.X_TWITTER: 1490083051496280145,
    Platform.GITHUB: 1489826423643308203,
    Platform.WEB: 1490083131921797220,
    Platform.INSTAGRAM_IMAGE: 1487648814280998963,
}

# Platform → emoji for routing feedback.
PLATFORM_EMOJI = {
    Platform.INSTAGRAM: "\U0001f4f8",      # 📸
    Platform.INSTAGRAM_IMAGE: "\U0001f4f8", # 📸
    Platform.YOUTUBE: "\U0001f3ac",         # 🎬
    Platform.X_TWITTER: "\U0001f426",       # 🐦
    Platform.GITHUB: "\U0001f4e6",          # 📦
    Platform.WEB: "\U0001f310",             # 🌐
}

DISCORD_API = "https://discord.com/api/v10"


async def start_watcher():
    """Connect to Discord and start watching for URLs and text."""
    # Prefer file-injected credential (real token, needed for WebSocket gateway).
    # Falls back to env var (may be a proxy token — works for REST but not WS).
    token_file = os.environ.get("DISCORD_BOT_ANT_FARM_FILE_PATH", "")
    if token_file and os.path.exists(token_file):
        token = open(token_file).read().strip()
        log.info("Loaded Discord token from file: %s", token_file)
    else:
        token = os.environ.get("DISCORD_BOT_ANT_FARM_TOKEN", "")
    if not token:
        log.error("DISCORD_BOT_ANT_FARM_TOKEN not set")
        return
    # Credential value may include "Bot " prefix (ant-keeper convention).
    # Strip it since discord.py and our HTTP calls add it themselves.
    if token.startswith("Bot "):
        token = token[4:]

    watched_ids = {
        int(ch)
        for ch in os.environ.get("WATCHED_CHANNELS", "").split(",")
        if ch.strip()
    }
    if not watched_ids:
        log.error("WATCHED_CHANNELS not set")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        last_seen = get_bot_last_seen()
        log.info("Connected as %s — watching %d channels. Last seen: %s",
                 client.user, len(watched_ids), last_seen or "never")
        # Update last_seen immediately so we know when this session started
        upsert_bot_last_seen(datetime.now(timezone.utc).isoformat())
        # Run catch-up in background to not delay on_ready
        asyncio.create_task(run_catchup(token, watched_ids, after_timestamp=last_seen))

    @client.event
    async def on_message(message: discord.Message):
        if message.channel.id not in watched_ids:
            return
        # In router channels: skip own messages that were already routed
        # (they have the 🔀 reaction). Fresh bot posts get routed normally.
        if message.author == client.user:
            if message.channel.id in ROUTER_CHANNEL_IDS:
                # Check if this message already has a 🔀 reaction (already routed).
                for r in message.reactions:
                    if str(r.emoji) == "\U0001f500" and r.me:
                        return  # Already routed — skip to prevent loop.
                # No 🔀 yet — this is a fresh post, let it route.
            # In non-router channels: always process own messages (routed content).

        text = message.content.strip()

        # ── Router channel: handle even if no text (may have attachments) ──
        if message.channel.id in ROUTER_CHANNEL_IDS:
            await _handle_router(message, text, token)
            return

        # ── Normal channel: skip if no content and no attachments ──────────
        if not text and not message.attachments:
            return

        # ── Normal channel: capture and stage ────────────────────────
        channel_name = getattr(message.channel, "name", str(message.channel.id))
        author = str(message.author)
        urls = extract_urls(text)
        staged_any = False

        base_meta = {
            "discord_msg_id": str(message.id),
            "discord_channel_id": str(message.channel.id),
            "discord_guild_id": str(message.guild.id) if message.guild else None,
            "discord_author_id": str(message.author.id),
            "discord_timestamp": message.created_at.isoformat() if message.created_at else None,
        }

        if urls:
            for url in urls:
                category = classify(url)
                sid = staging.stage(
                    source_type=category.value,
                    source_uri=url,
                    raw_content=text,
                    author=author,
                    channel=channel_name,
                    created_at=message.created_at.isoformat() if message.created_at else None,
                    metadata=base_meta,
                )
                if sid:
                    log.info("Staged %s [%s]: %s", category.value, channel_name, url)
                    staged_any = True
                    if _pipeline:
                        item = staging.get_by_id(sid)
                        if item:
                            _pipeline.enqueue(item)
        else:
            if is_noise(text):
                log.debug("Filtered noise [%s]: %s...", channel_name, text[:40])
                return

            msg_uri = f"discord://{message.guild.id}/{message.channel.id}/{message.id}"
            sid = staging.stage(
                source_type=Platform.PLAIN_TEXT.value,
                source_uri=msg_uri,
                raw_content=text,
                author=author,
                channel=channel_name,
                created_at=message.created_at.isoformat() if message.created_at else None,
                metadata=base_meta,
            )
            if sid:
                log.info("Staged plain_text [%s]: %s...", channel_name, text[:60])
                staged_any = True
                if _pipeline:
                    item = staging.get_by_id(sid)
                    if item:
                        _pipeline.enqueue(item)

        if staged_any:
            try:
                await message.add_reaction("\U0001f4e5")  # 📥 staged
            except discord.errors.Forbidden:
                pass

    # Start the pipeline worker if API keys are available.
    global _pipeline
    if os.environ.get("ANTHROPIC_API_KEY"):
        _pipeline = PipelineWorker(concurrency=3)
        await _pipeline.start()
        log.info("Pipeline worker enabled (inline processing)")
    else:
        log.info("Pipeline worker disabled (no ANTHROPIC_API_KEY — cron-only mode)")

    await client.start(token)


async def _handle_router(message: discord.Message, text: str, token: str):
    """Route content from #ant-food-router to the correct typed channel."""
    urls = extract_urls(text)
    # Include Discord CDN attachment URLs so attachment-only messages get routed
    attachment_urls = [a.url for a in message.attachments if a.url]
    # Deduplicate: keep attachment URLs not already found in text
    for au in attachment_urls:
        if au not in urls:
            urls.append(au)

    if not urls:
        # Plain text with no URL — stage it here (no better place).
        if is_noise(text):
            return
        channel_name = getattr(message.channel, "name", str(message.channel.id))
        msg_uri = f"discord://{message.guild.id}/{message.channel.id}/{message.id}"
        sid = staging.stage(
            source_type=Platform.PLAIN_TEXT.value,
            source_uri=msg_uri,
            raw_content=text,
            author=str(message.author),
            channel=channel_name,
            created_at=message.created_at.isoformat() if message.created_at else None,
            metadata={
                "discord_msg_id": str(message.id),
                "discord_channel_id": str(message.channel.id),
                "discord_guild_id": str(message.guild.id) if message.guild else None,
                "discord_author_id": str(message.author.id),
                "discord_timestamp": message.created_at.isoformat() if message.created_at else None,
            },
        )
        if sid:
            try:
                await message.add_reaction("\U0001f4e5")  # 📥 staged here
            except discord.errors.Forbidden:
                pass
        return

    # Route each URL to the correct channel.
    routed = False
    for url in urls:
        platform = classify(url)
        dest_channel_id = ROUTE_MAP.get(platform)

        if dest_channel_id:
            # Repost to the destination channel.
            try:
                async with httpx.AsyncClient(timeout=10) as http:
                    resp = await http.post(
                        f"{DISCORD_API}/channels/{dest_channel_id}/messages",
                        headers={"Authorization": f"Bot {token}"},
                        json={"content": text},
                    )
                    resp.raise_for_status()
                    log.info("Routed %s → channel %s", platform.value, dest_channel_id)
                    routed = True
            except Exception:
                log.warning("Failed to route %s to channel %s", url, dest_channel_id)

    # React on the original to show where it was routed.
    try:
        await message.add_reaction("\U0001f500")  # 🔀 routed
        # Add platform-specific emoji.
        for url in urls:
            platform = classify(url)
            emoji = PLATFORM_EMOJI.get(platform)
            if emoji:
                await message.add_reaction(emoji)
                break  # One platform emoji is enough.
    except discord.errors.Forbidden:
        pass

    if not routed:
        # No route found — stage it here as fallback.
        for url in urls:
            category = classify(url)
            staging.stage(
                source_type=category.value,
                source_uri=url,
                raw_content=text,
                author=str(message.author),
                channel=getattr(message.channel, "name", "ant-food-router"),
                metadata={
                    "discord_msg_id": str(message.id),
                    "discord_channel_id": str(message.channel.id),
                },
            )
        try:
            await message.add_reaction("\U0001f4e5")  # 📥 staged (no route)
        except discord.errors.Forbidden:
            pass
