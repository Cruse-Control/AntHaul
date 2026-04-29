"""YouTube resolver — yt-dlp metadata + transcript extraction.

Extracts video metadata (title, description, channel, duration, view count)
and the best available transcript. Uses youtube_transcript_api for transcripts
(more reliable than yt-dlp subtitle data with skip_download=True), with yt-dlp
subtitle data as fallback. Truncates transcript at 12 000 tokens (~48 000 chars).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

from seed_storage.enrichment.models import ResolvedContent
from seed_storage.enrichment.resolvers.base import BaseResolver

logger = logging.getLogger(__name__)

_MAX_CHARS = 48_000  # ~12 000 tokens
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
_SHORTS_RE = re.compile(r"/shorts/([A-Za-z0-9_-]{11})")


def _extract_video_id(url: str) -> str | None:
    """Extract the 11-char YouTube video ID from various URL forms."""
    parsed = urlparse(url)
    # youtu.be/<id>
    if parsed.hostname in ("youtu.be",):
        path = parsed.path.lstrip("/")
        if len(path) == 11:
            return path
        return path.split("/")[0] if path else None
    # youtube.com/shorts/<id>
    shorts_match = _SHORTS_RE.search(parsed.path)
    if shorts_match:
        return shorts_match.group(1)
    # youtube.com/watch?v=<id>
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    return None


def _truncate(text: str) -> str:
    if len(text) > _MAX_CHARS:
        return text[:_MAX_CHARS]
    return text


def _get_transcript_api(video_id: str) -> str | None:
    """Get transcript via youtube_transcript_api (most reliable method)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Try English first
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return " ".join(e["text"] for e in entries)
        except Exception:
            pass

        # Fall back to any language
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(e["text"] for e in entries)
        except Exception:
            pass
    except ImportError:
        logger.debug("youtube_transcript_api not installed")

    return None


class YouTubeResolver(BaseResolver):
    """Resolves YouTube video URLs using yt-dlp + youtube_transcript_api."""

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in _YOUTUBE_HOSTS

    async def resolve(self, url: str) -> ResolvedContent:
        try:
            return await self._resolve_internal(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YouTubeResolver failed for %s: %s", url, exc)
            return ResolvedContent.error_result(url, str(exc))

    async def _resolve_internal(self, url: str) -> ResolvedContent:
        import asyncio

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._fetch_ydl, url)
        return result

    def _fetch_ydl(self, url: str) -> ResolvedContent:
        import yt_dlp  # type: ignore[import-untyped]

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
            "socket_timeout": 20,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise ValueError("yt-dlp returned no info")

        title = info.get("title") or ""
        description = info.get("description") or ""
        channel = info.get("channel") or info.get("uploader") or ""
        duration = info.get("duration")
        view_count = info.get("view_count")
        upload_date = info.get("upload_date")  # YYYYMMDD string
        video_id = _extract_video_id(url) or info.get("id") or ""

        # --- Transcript extraction ---
        transcript: str | None = None

        # Method 1: youtube_transcript_api (most reliable)
        if video_id:
            transcript = _get_transcript_api(video_id)

        # Method 2: yt-dlp subtitle data (fallback — often empty with skip_download)
        if not transcript:
            subtitles: dict = info.get("subtitles") or {}
            auto_captions: dict = info.get("automatic_captions") or {}

            for sub_dict in (subtitles, auto_captions):
                for lang_key in ("en", "en-US", "en-GB"):
                    if lang_key in sub_dict:
                        entries = sub_dict[lang_key]
                        for entry in entries:
                            if entry.get("ext") in ("vtt", "srv3", "srv2", "srv1", "json3"):
                                raw = entry.get("data")
                                if raw:
                                    transcript = _clean_vtt(raw)
                                break
                        if transcript:
                            break
                if transcript:
                    break

        # Build text from description + transcript
        text_parts = []
        if description:
            text_parts.append(description)
        if transcript:
            text_parts.append(f"[Transcript]\n{transcript}")

        text = "\n\n".join(text_parts)
        text = _truncate(text)
        if transcript:
            transcript = _truncate(transcript)

        # Extract URLs from description for expansion
        expansion_urls = []
        for match in re.finditer(r"https?://[^\s<>\"']+", description):
            expansion_urls.append(match.group(0))

        metadata: dict = {
            "channel": channel,
            "duration_seconds": duration,
            "view_count": view_count,
            "upload_date": upload_date,
            "video_id": video_id,
        }
        if channel:
            metadata["speakers"] = [{"name": channel, "role": "creator", "platform": "youtube"}]

        return ResolvedContent(
            source_url=url,
            content_type="youtube",
            title=title or None,
            text=text,
            transcript=transcript,
            summary=None,
            expansion_urls=expansion_urls[:20],
            metadata=metadata,
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )


def _clean_vtt(raw: str) -> str:
    """Strip VTT/WebVTT timing lines and tags, returning plain transcript text."""
    lines = []
    for line in raw.splitlines():
        # Skip WEBVTT header, timing lines, blank lines
        if re.match(r"^WEBVTT", line):
            continue
        if re.match(r"^\d{2}:\d{2}", line):
            continue
        if re.match(r"^\s*$", line):
            continue
        if re.match(r"^\d+\s*$", line):
            continue
        # Strip inline VTT tags like <c.colorXXX> and timestamps <00:00:00.000>
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return " ".join(lines)
