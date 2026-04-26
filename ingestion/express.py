"""Express ingest -- full pipeline for a single URL in one shot.

Bypasses cron waits by running stage -> process -> enrich -> extract -> load synchronously.
Typical latency: 5-15 seconds for a web article, longer for video content.

Run as: python -m ingestion.express <url>
Or call express_ingest() from the MCP server / other code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
from anthropic import AsyncAnthropic

from ingestion import discord_touch
from ingestion.classifier import Platform, classify, clean_url
from ingestion.enricher import _enrich_one, _get_existing_tags, _upsert_tags, init_tags_table
from ingestion.processor import process_one
from seed_storage import staging
from seed_storage.graph import close

log = logging.getLogger("express")


async def express_ingest(
    url: str,
    author: str = "express",
    channel: str = "express",
) -> dict:
    """Full pipeline for a single URL: stage -> process -> enrich -> extract -> load.

    Resumes from the item's current status if it already exists in staging.
    Returns dict with status, timing, and source_uri.
    """
    t0 = time.monotonic()
    url = clean_url(url)
    platform = classify(url)

    # 1. Stage (or find existing)
    sid = staging.stage(
        source_type=platform.value,
        source_uri=url,
        raw_content=url,
        author=author,
        channel=channel,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    if sid is None:
        item = staging.get_by_uri(url)
        if item is None:
            return {"status": "error", "message": "URL exists but could not be retrieved", "source_uri": url}
        if item["status"] == "loaded":
            return {"status": "already_loaded", "source_uri": url, "elapsed_seconds": round(time.monotonic() - t0, 1)}
    else:
        item = staging.get_by_id(sid)

    item_id = str(item["id"])
    current_status = item["status"]

    # Set up shared clients
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic = AsyncAnthropic(api_key=api_key) if api_key else None
    analyzer_url = os.environ.get("ANALYZER_BASE_URL", "http://localhost:8000")

    async with httpx.AsyncClient(timeout=120) as http:
        # 2. Process (if needed)
        if current_status in ("staged",):
            await process_one(item, http, anthropic, analyzer_url)
            item = staging.get_by_id(item_id)
            if item["status"] == "failed":
                return {"status": "failed", "stage": "process", "source_uri": url,
                        "elapsed_seconds": round(time.monotonic() - t0, 1)}
            current_status = item["status"]

        # 3. Enrich (if needed)
        if current_status in ("processed",):
            if anthropic:
                init_tags_table()
                existing_tags = _get_existing_tags()

                enrichment = await _enrich_one(anthropic, item, existing_tags)

                meta = item.get("metadata") or {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                for key, value in enrichment.items():
                    if key in ("published_at", "speakers") and key in meta:
                        continue
                    meta[key] = value

                item_author = item.get("author", "unknown")
                speakers = meta.get("speakers", [])
                curator_names = {s["name"] for s in speakers if s.get("role") == "curator"}
                if item_author and item_author != "unknown" and item_author not in curator_names:
                    speakers.append({"name": item_author, "role": "curator", "platform": "discord"})
                    meta["speakers"] = speakers

                staging.update_content(item_id, item["raw_content"], metadata=meta, status="enriched")

                new_tags = enrichment.get("tags", [])
                if new_tags:
                    _upsert_tags(new_tags)
                log.info("Express enriched [%s] %s -> tags=%s", item["source_type"], url, new_tags)
            else:
                staging.update_status([item_id], "enriched")

            item = staging.get_by_id(item_id)
            current_status = item["status"]
            await discord_touch.react(item, "enriched")

        # 4. Extract (if needed)
        if current_status in ("enriched",):
            from seed_storage.extraction import extract_one
            from seed_storage.preseed import get_alias_map, init_preseed_table
            init_preseed_table()
            result = extract_one(item, alias_map=get_alias_map())
            staging.patch_metadata(item_id, {
                "extraction": {
                    **result.model_dump(),
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
            })
            staging.update_status([item_id], "extracted")
            item = staging.get_by_id(item_id)
            current_status = "extracted"

        # 5. Load (if needed)
        if current_status in ("extracted",):
            from seed_storage.preseed import get_alias_map
            from seed_storage.graph import get_driver
            from ingestion.loader import _load_one_item

            alias_map = get_alias_map()
            driver = await get_driver()
            await _load_one_item(item, alias_map, None, driver, batch_id=None)
            current_status = "loaded"
            await discord_touch.react(item, "loaded")

    elapsed = round(time.monotonic() - t0, 1)
    return {"status": current_status, "source_uri": url, "elapsed_seconds": elapsed}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not url:
        print("Usage: python -m ingestion.express <url>")
        sys.exit(1)

    result = asyncio.run(express_ingest(url))
    print(json.dumps(result, indent=2, default=str))

    # Clean up Neo4j connection
    asyncio.run(close())
