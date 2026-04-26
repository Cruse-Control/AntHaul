"""Step 2: Process staged items by medium.

Reads items with status='staged', performs medium-specific extraction
(transcript, README, readability, etc.), updates with processed content.

Run as: python -m ingestion.processor
Or register as an ant-keeper scheduled task.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from urllib.parse import parse_qs, urlparse

import httpx
from anthropic import AsyncAnthropic

from ingestion import discord_touch
from seed_storage import staging

log = logging.getLogger("processor")

ADJUDICATOR_SYSTEM = """\
You analyze transcripts from video content to decide if the video's visuals \
carry important information beyond what's spoken.

Say YES if the content likely shows: code/terminals, diagrams/flowcharts, \
UI demos, screen recordings, data visualizations, physical tutorials, \
or significant text overlays not in the spoken audio.

Say NO if: talking head, speech conveys everything, visuals are decorative.

Respond ONLY with valid JSON: {"visuals_important": true/false, "reason": "one sentence"}\
"""


async def process_batch(limit: int = 50, concurrency: int = 3):
    """Process a batch of staged items with bounded concurrency."""
    items = staging.get_staged(status="staged", limit=limit)
    if not items:
        log.info("No staged items to process")
        return

    log.info("Processing %d staged items (concurrency=%d)", len(items), concurrency)

    async with httpx.AsyncClient(timeout=120) as http:
        from seed_storage import config

        anthropic = None
        if config.LLM_API_KEY:
            anthropic = AsyncAnthropic(api_key=config.LLM_API_KEY)

        analyzer_url = os.environ.get("ANALYZER_BASE_URL", "http://localhost:8000")

        sem = asyncio.Semaphore(concurrency)

        async def _process_guarded(item):
            async with sem:
                await process_one(item, http, anthropic, analyzer_url)

        tasks = [asyncio.create_task(_process_guarded(item)) for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)


async def process_one(
    item: dict,
    http: httpx.AsyncClient,
    anthropic: AsyncAnthropic | None = None,
    analyzer_url: str = "",
) -> None:
    """Process a single staged item — extract content by medium type.

    Updates the staging table directly (status → 'processed' or 'failed').
    Can be called from process_batch() or express_ingest().
    """
    item_id = str(item["id"])
    source_type = item["source_type"]
    source_uri = item["source_uri"]

    # Preserve Discord commentary for URL entries — the processor replaces
    # raw_content with extracted content, but the original message has context.
    original_text = (item.get("raw_content") or "").strip()
    has_discord_context = (
        source_type not in ("plain_text", "conversation_thread", "discord_link", "media_link")
        and original_text
        and original_text != source_uri
        and len(original_text) > len(source_uri) + 5
    )

    try:
        if source_type == "instagram":
            content, meta = await _process_instagram(http, anthropic, analyzer_url, source_uri)
        elif source_type == "youtube":
            content, meta = await _process_youtube(http, anthropic, analyzer_url, source_uri)
        elif source_type == "github":
            content, meta = await _process_github(http, source_uri)
            # Non-blocking: try to add as submodule to inspirational-materials.
            try:
                from ingestion.submodule_adder import add_submodule
                sub_result = add_submodule(source_uri, description=content[:500], push=True, create_pr=True)
                meta["submodule_status"] = sub_result.get("status", "unknown")
                if sub_result.get("path"):
                    meta["submodule_path"] = sub_result["path"]
                log.info("Submodule add for %s: %s", source_uri, sub_result.get("status"))
            except Exception:
                log.debug("Submodule add failed for %s (non-blocking)", source_uri, exc_info=True)
        elif source_type == "x_twitter":
            content, meta = await _process_x(http, source_uri)
        elif source_type == "web":
            content, meta = await _process_web(http, source_uri)
        elif source_type == "instagram_image":
            content, meta = await _process_instagram_image(http, source_uri)
        elif source_type == "conversation_thread":
            # Threaded conversations already have merged content — passthrough.
            content = item["raw_content"]
            meta = item.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
        elif source_type in ("plain_text", "discord_link", "media_link"):
            content = item["raw_content"]
            meta = {}
        else:
            log.warning("Unknown source_type '%s' for %s", source_type, source_uri)
            content = item["raw_content"]
            meta = {}

        # Stash Discord commentary in metadata so the loader can include it.
        if has_discord_context:
            commentary = original_text.replace(source_uri, "").strip()
            if len(commentary) > 5:
                meta["discord_context"] = commentary

        staging.update_content(item_id, content, metadata=meta, status="processed")
        log.info("Processed [%s] %s", source_type, source_uri)
        await discord_touch.react(item, "processed")

    except Exception:
        # Context-only fallback: store the Discord message as content.
        raw = item.get("raw_content", "")
        if raw and len(raw.strip()) > 10:
            log.warning("HTTP extraction failed for [%s] %s — falling back to Discord context", source_type, source_uri)
            staging.update_content(
                item_id, raw,
                metadata={"fetch_status": "context_only"},
                status="processed",
            )
            await discord_touch.react(item, "processed")
        else:
            log.exception("Failed to process [%s] %s (no context to fall back to)", source_type, source_uri)
            staging.update_status([item_id], "failed")
            await discord_touch.react(item, "failed", error_msg=f"Extraction failed for {source_uri}")


# ── Medium-specific extractors ─────────────────────────────────────


async def _process_instagram(
    http: httpx.AsyncClient, anthropic: AsyncAnthropic | None, base: str, url: str
) -> tuple[str, dict]:
    """Transcript via MCP backend + adjudicator."""
    resp = await http.post(f"{base}/api/video/analyze", json={"instagram_url": url, "analysis_type": "transcription"})
    resp.raise_for_status()
    job_id = resp.json().get("job_id") or resp.json().get("id")
    transcript = await _poll_job(http, base, job_id)

    meta: dict = {}
    visuals_important, reason = await _adjudicate(anthropic, transcript)
    meta["adjudicator_decision"] = visuals_important
    meta["adjudicator_reason"] = reason

    if visuals_important:
        resp = await http.post(f"{base}/api/video/analyze", json={"instagram_url": url, "analysis_type": "comprehensive"})
        resp.raise_for_status()
        rich_id = resp.json().get("job_id") or resp.json().get("id")
        rich = await _poll_job(http, base, rich_id)
        meta["media_analysis"] = rich
        return f"{transcript}\n\n---\nVisual Analysis:\n{rich}", meta

    return transcript, meta


async def _process_instagram_image(
    http: httpx.AsyncClient, url: str
) -> tuple[str, dict]:
    """Extract Instagram image post caption via instaloader (no auth required)."""
    shortcode = _extract_instagram_shortcode(url)
    if not shortcode:
        return f"[Instagram image post — could not parse shortcode] {url}", {}

    try:
        import instaloader

        loader = instaloader.Instaloader()
        post = instaloader.Post.from_shortcode(loader.context, shortcode)

        caption = (post.caption or "").strip()
        owner = post.owner_username or ""
        likes = post.likes
        date = post.date_utc

        if caption:
            content = f"Instagram post by @{owner}:\n\n{caption}"
        else:
            content = f"Instagram post by @{owner} (no caption)"

        meta: dict = {}
        if owner:
            meta["author"] = f"@{owner}"
            meta["speakers"] = [{"name": f"@{owner}", "role": "author", "platform": "instagram"}]
        if date:
            meta["published_at"] = date.isoformat()
        if likes:
            meta["engagement"] = {"likes": likes}
        return content, meta

    except Exception:
        log.warning("Instaloader failed for %s, falling back to context", url, exc_info=True)

    return f"[Instagram image post] {url}", {}


def _extract_instagram_shortcode(url: str) -> str:
    """Extract Instagram shortcode from /p/SHORTCODE/ or /reel/SHORTCODE/ URLs."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    for i, p in enumerate(parts):
        if p in ("p", "reel", "reels") and i + 1 < len(parts):
            return parts[i + 1]
    return ""


async def _process_youtube(
    http: httpx.AsyncClient, anthropic: AsyncAnthropic | None, base: str, url: str
) -> tuple[str, dict]:
    """Free transcript API + adjudicator."""
    video_id = _extract_yt_id(url)
    transcript = _get_yt_transcript(video_id)

    # Extract structured metadata from YouTube Data API.
    channel_title, published_at, video_title, video_desc = await _fetch_yt_metadata(http, video_id)
    meta: dict = {}
    if published_at:
        meta["published_at"] = published_at
    if channel_title:
        meta["speakers"] = [{"name": channel_title, "role": "creator", "platform": "youtube"}]

    # Fallback: if no transcript, use title + description from YouTube API.
    if transcript.startswith("[No transcript") and (video_title or video_desc):
        transcript = f"# {video_title}\n\n{video_desc}" if video_desc else f"# {video_title}"
        meta["fetch_status"] = "metadata_fallback"
        log.info("YouTube transcript unavailable for %s, using title+description fallback", url)

    visuals_important, reason = await _adjudicate(anthropic, transcript)
    meta["adjudicator_decision"] = visuals_important
    meta["adjudicator_reason"] = reason

    if visuals_important:
        try:
            resp = await http.post(f"{base}/api/video/analyze", json={"url": url, "analysis_type": "visual_description"})
            resp.raise_for_status()
            job_id = resp.json().get("job_id") or resp.json().get("id")
            rich = await _poll_job(http, base, job_id)
            meta["media_analysis"] = rich
            return f"{transcript}\n\n---\nVisual Analysis:\n{rich}", meta
        except Exception:
            log.warning("Visual analysis failed for %s, using transcript only", url)

    return transcript, meta


async def _process_github(http: httpx.AsyncClient, url: str) -> tuple[str, dict]:
    """API → README + metadata."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return f"[Could not parse GitHub repo from {url}]", {}

    owner, repo = parts[0], parts[1]
    api = f"https://api.github.com/repos/{owner}/{repo}"

    resp = await http.get(api)
    resp.raise_for_status()
    meta_data = resp.json()

    description = meta_data.get("description", "") or ""
    stars = meta_data.get("stargazers_count", 0)
    language = meta_data.get("language", "") or ""
    topics = meta_data.get("topics", [])

    # Fetch README + high-signal config files (CLAUDE.md, AGENTS.md).
    docs = {}
    for doc_path in ["readme", "contents/CLAUDE.md", "contents/AGENTS.md"]:
        try:
            r = await http.get(f"{api}/{doc_path}")
            if r.status_code == 200:
                data = r.json()
                text = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
                name = data.get("name", doc_path.split("/")[-1])
                docs[name] = text
        except Exception:
            pass

    content = f"# {owner}/{repo}\n\n{description}\n\nStars: {stars} | Language: {language}"
    if topics:
        content += f"\nTopics: {', '.join(topics)}"
    for name, text in docs.items():
        content += f"\n\n---\n## {name}\n\n{text}"

    return content, {
        "stars": stars,
        "language": language,
        "topics": topics,
        "published_at": meta_data.get("created_at", ""),
        "speakers": [{"name": meta_data.get("owner", {}).get("login", owner), "role": "maintainer", "platform": "github"}],
    }


async def _process_x(http: httpx.AsyncClient, url: str) -> tuple[str, dict]:
    """Tweet extraction via FxTwitter API (free, no auth required)."""
    username, tweet_id = _extract_tweet_info(url)
    if not tweet_id:
        return f"[Could not parse tweet ID from {url}]", {}

    # FxTwitter requires username in the path and a User-Agent header.
    fx_path = f"{username}/status/{tweet_id}" if username else f"i/status/{tweet_id}"
    resp = await http.get(
        f"https://api.fxtwitter.com/{fx_path}",
        timeout=30,
        headers={"User-Agent": "SeedStorage/1.0", "Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    tweet = data.get("tweet", {})

    author = tweet.get("author", {})
    author_name = author.get("name", "")
    author_handle = f"@{author.get('screen_name', '')}"
    text = tweet.get("text", "")

    content_parts = [f"Tweet by {author_name} ({author_handle}):", "", text]

    # Include media descriptions if present.
    media = tweet.get("media", {})
    if media:
        for photo in media.get("photos", []):
            alt = photo.get("altText", "")
            if alt:
                content_parts.append(f"\n[Image: {alt}]")
        for video in media.get("videos", []):
            dur = video.get("duration", 0)
            content_parts.append(f"\n[Video: {dur}s]")

    # Include quote tweet if present.
    quote = tweet.get("quote", {})
    if quote:
        q_author = quote.get("author", {})
        q_name = q_author.get("name", "")
        q_handle = f"@{q_author.get('screen_name', '')}"
        content_parts.append(f"\nQuoting {q_name} ({q_handle}):")
        content_parts.append(quote.get("text", ""))

    content = "\n".join(content_parts)

    meta: dict = {
        "author": author_handle,
        "author_name": author_name,
        "tweet_id": tweet_id,
        "speakers": [{"name": author_name or author_handle, "role": "author", "platform": "x.com"}],
    }
    created_at = tweet.get("created_at")
    if created_at:
        meta["published_at"] = created_at
    likes = tweet.get("likes", 0)
    retweets = tweet.get("retweets", 0)
    if likes or retweets:
        meta["engagement"] = {"likes": likes, "retweets": retweets}

    return content, meta


def _extract_tweet_info(url: str) -> tuple[str, str]:
    """Extract (username, tweet_id) from x.com or twitter.com URL.

    Returns ("", "") if the URL can't be parsed.
    """
    parts = url.rstrip("/").split("/")
    username = ""
    tweet_id = ""
    for i, p in enumerate(parts):
        if p in ("x.com", "twitter.com") and i + 1 < len(parts):
            username = parts[i + 1]
        if p == "status" and i + 1 < len(parts):
            tid = parts[i + 1].split("?")[0]
            if tid.isdigit():
                tweet_id = tid
    return username, tweet_id


async def _process_web(http: httpx.AsyncClient, url: str) -> tuple[str, dict]:
    """Readability article extraction."""
    from readability import Document
    from bs4 import BeautifulSoup

    resp = await http.get(url, follow_redirects=True)
    resp.raise_for_status()

    doc = Document(resp.text)
    title = doc.short_title() or ""
    soup = BeautifulSoup(doc.summary(), "lxml")
    text = soup.get_text(separator="\n", strip=True)

    full_soup = BeautifulSoup(resp.text, "lxml")
    author = ""
    for attr in [{"name": "author"}, {"property": "og:author"}, {"name": "twitter:creator"}]:
        tag = full_soup.find("meta", attrs=attr)
        if tag and tag.get("content"):
            author = tag["content"]
            break

    published_at = ""
    for attr in [
        {"property": "article:published_time"},
        {"property": "og:article:published_time"},
        {"name": "date"},
        {"name": "publishdate"},
    ]:
        tag = full_soup.find("meta", attrs=attr)
        if tag and tag.get("content"):
            published_at = tag["content"]
            break

    # Extract outbound links from the full HTML.
    outbound = []
    for a in full_soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and len(href) < 500:
            outbound.append(href)

    content = f"# {title}\n\n{text}" if title else text
    meta = {"title": title, "author": author}
    if published_at:
        meta["published_at"] = published_at
    if author:
        meta["speakers"] = [{"name": author, "role": "author", "platform": "web"}]
    if outbound:
        meta["outbound_links"] = outbound[:50]
    return content, meta


# ── YouTube metadata ───────────────────────────────────────────────


async def _fetch_yt_metadata(http: httpx.AsyncClient, video_id: str) -> tuple[str, str, str, str]:
    """Fetch channel, date, title, description from YouTube Data API v3.

    Returns (channel_title, published_at, title, description).
    Degrades gracefully if no API key.
    """
    key = os.environ.get("YOUTUBE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return "", "", "", ""
    try:
        resp = await http.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "snippet", "id": video_id, "key": key},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return "", "", "", ""
        snippet = items[0]["snippet"]
        return (
            snippet.get("channelTitle", ""),
            snippet.get("publishedAt", ""),
            snippet.get("title", ""),
            snippet.get("description", ""),
        )
    except Exception:
        log.debug("YouTube Data API failed for %s, skipping metadata", video_id)
        return "", "", "", ""


# ── Helpers ────────────────────────────────────────────────────────


async def _adjudicate(anthropic: AsyncAnthropic | None, transcript: str) -> tuple[bool, str]:
    if not anthropic or not transcript or transcript.startswith("[No transcript"):
        return False, "skipped"

    from seed_storage import config

    resp = await anthropic.messages.create(
        model=config.LLM_MODEL,
        max_tokens=150,
        system=ADJUDICATOR_SYSTEM,
        messages=[{"role": "user", "content": transcript[:8000]}],
    )
    try:
        data = json.loads(resp.content[0].text)
        return data.get("visuals_important", False), data.get("reason", "")
    except (json.JSONDecodeError, IndexError):
        return False, "parse error"


async def _poll_job(http: httpx.AsyncClient, base: str, job_id: str, timeout: int = 300) -> str:
    elapsed = 0
    while elapsed < timeout:
        # Try /api/video/status/{id} (Instagram MCP backend) then /api/jobs/{id} (fallback).
        resp = await http.get(f"{base}/api/video/status/{job_id}")
        if resp.status_code == 404 or resp.status_code == 405:
            resp = await http.get(f"{base}/api/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status == "completed":
            # Extract text from nested result structure.
            result = data.get("analysis_result") or data.get("result", {})
            if isinstance(result, dict):
                analysis = result.get("analysis", result)
                if isinstance(analysis, dict):
                    return analysis.get("text", "") or analysis.get("transcription", "") or str(analysis)
                return str(analysis)
            return str(result)
        if status in ("failed", "error", "FAILED"):
            raise RuntimeError(f"Job {job_id} failed: {data.get('error_message') or data.get('error')}")
        await asyncio.sleep(3)
        elapsed += 3
    raise TimeoutError(f"Job {job_id} timed out")


def _extract_yt_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[1].split("/")[0]
        return parse_qs(parsed.query).get("v", [""])[0]
    return ""


def _get_yt_transcript(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(e["text"] for e in entries)
    except Exception:
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(e["text"] for e in entries)
        except Exception:
            return f"[No transcript available for video {video_id}]"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(process_batch(limit=args.limit, concurrency=args.concurrency))
