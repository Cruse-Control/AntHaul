"""Instagram resolver — yt-dlp video download + Whisper transcription for reels,
vision LLM for image posts, oEmbed API fallback.

Instagram reels are video-first: download via yt-dlp, extract audio with ffmpeg,
transcribe with Whisper (same pipeline as VideoResolver). For image posts, use
vision LLM (same as ImageResolver). Always include caption text alongside
transcript/description. Falls back to oEmbed API if yt-dlp fails.
"""

from __future__ import annotations

import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from seed_storage.enrichment.models import ResolvedContent
from seed_storage.enrichment.resolvers.base import BaseResolver

logger = logging.getLogger(__name__)

_INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}
_TIMEOUT = 30.0


class InstagramResolver(BaseResolver):
    """Extracts Instagram reel/post content.

    Reels: yt-dlp download → ffmpeg audio → Whisper transcription + caption.
    Image posts: vision LLM description + caption.
    Fallback: oEmbed API for caption text.
    """

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host not in _INSTAGRAM_HOSTS:
            return False
        path = parsed.path
        return any(seg in path for seg in ("/reel/", "/reels/", "/p/"))

    async def resolve(self, url: str) -> ResolvedContent:
        is_reel = "/reel/" in url or "/reels/" in url

        # Try yt-dlp first (works for both reels and image posts)
        try:
            if is_reel:
                result = await self._resolve_reel(url)
            else:
                result = await self._resolve_image_post(url)
            if result.text and len(result.text.strip()) > 10:
                return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("Instagram primary extraction failed for %s: %s", url, exc)

        # Fallback: oEmbed API
        try:
            return await self._resolve_oembed(url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Instagram oEmbed fallback failed for %s: %s", url, exc)

        # Last resort: og: tags from HTML
        try:
            return await self._resolve_og_tags(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Instagram all methods failed for %s: %s", url, exc)

        return ResolvedContent.error_result(url, "Could not extract Instagram content")

    async def _resolve_reel(self, url: str) -> ResolvedContent:
        """Download reel video → ffmpeg audio → Whisper transcription."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._download_and_transcribe, url)

    def _download_and_transcribe(self, url: str) -> ResolvedContent:
        """Synchronous: yt-dlp download → ffmpeg → Whisper."""
        import subprocess

        import yt_dlp  # type: ignore[import-untyped]

        # Download video to temp file
        video_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = Path(video_tmp.name)
        video_tmp.close()
        audio_path: Path | None = None

        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "outtmpl": str(video_path),
                "socket_timeout": 30,
                "format": "worst[ext=mp4]/worst",  # smallest file for transcription
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            if not info:
                raise ValueError("yt-dlp returned no info")

            caption = (info.get("description") or "").strip()
            uploader = info.get("uploader") or info.get("channel") or ""
            uploader_id = info.get("uploader_id") or ""
            title = (info.get("title") or "").strip()
            timestamp = info.get("timestamp")

            # Extract audio with ffmpeg
            audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = Path(audio_tmp.name)
            audio_tmp.close()

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    "-y",
                    str(audio_path),
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.debug(
                    "ffmpeg failed for Instagram: %s",
                    result.stderr.decode("utf-8", errors="replace")[:200],
                )
                # If audio extraction fails, use caption only
                return self._build_caption_result(
                    url, caption, uploader, uploader_id, title, timestamp
                )

            # Transcribe with Whisper
            transcript = ""
            try:
                import whisper  # type: ignore[import-untyped]

                model = whisper.load_model("base")
                whisper_result = model.transcribe(str(audio_path), fp16=False)
                transcript = whisper_result.get("text", "").strip()
            except Exception as exc:
                logger.debug("Whisper transcription failed for Instagram: %s", exc)

            # Build combined content
            content_parts = []
            if uploader:
                content_parts.append(f"Instagram reel by @{uploader_id or uploader}:")
                content_parts.append("")
            if caption and caption != title:
                content_parts.append(f"Caption: {caption}")
            if transcript:
                content_parts.append(f"\n[Transcript]\n{transcript}")

            text = "\n".join(content_parts)

            metadata = self._build_metadata(uploader, uploader_id, timestamp)
            if transcript:
                metadata["transcription_backend"] = "whisper"

            return ResolvedContent(
                source_url=url,
                content_type="instagram",
                title=title or None,
                text=text,
                transcript=transcript or None,
                summary=None,
                expansion_urls=[],
                metadata=metadata,
                extraction_error=None,
                resolved_at=datetime.now(tz=UTC),
            )

        finally:
            for tmp in (video_path, audio_path):
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass

    async def _resolve_image_post(self, url: str) -> ResolvedContent:
        """For image posts: get caption + vision LLM description."""
        import asyncio

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, self._fetch_ytdlp_metadata, url)

        caption = (info.get("description") or "").strip()
        uploader = info.get("uploader") or info.get("channel") or ""
        uploader_id = info.get("uploader_id") or ""
        title = (info.get("title") or "").strip()
        timestamp = info.get("timestamp")
        thumbnail = info.get("thumbnail") or ""

        # Try vision LLM on thumbnail if available
        vision_text = ""
        if thumbnail:
            try:
                vision_text = await self._describe_image(thumbnail)
            except Exception as exc:
                logger.debug("Vision LLM failed for Instagram image: %s", exc)

        content_parts = []
        if uploader:
            content_parts.append(f"Instagram post by @{uploader_id or uploader}:")
            content_parts.append("")
        if caption and caption != title:
            content_parts.append(caption)
        if vision_text:
            content_parts.append(f"\n[Image description]\n{vision_text}")

        text = "\n".join(content_parts)
        metadata = self._build_metadata(uploader, uploader_id, timestamp)

        return ResolvedContent(
            source_url=url,
            content_type="instagram",
            title=title or None,
            text=text,
            transcript=None,
            summary=vision_text or None,
            expansion_urls=[],
            metadata=metadata,
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )

    def _fetch_ytdlp_metadata(self, url: str) -> dict:
        """Fetch metadata without downloading."""
        import yt_dlp  # type: ignore[import-untyped]

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "socket_timeout": 20,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return info or {}

    async def _describe_image(self, image_url: str) -> str:
        """Use vision LLM to describe an image."""
        import base64

        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            image_data = resp.content

        if not image_data:
            return ""

        b64_image = base64.b64encode(image_data).decode("utf-8")
        content_type = "image/jpeg"

        try:
            from seed_storage.config import settings

            api_key = settings.OPENAI_API_KEY
            model = settings.LLM_MODEL
        except Exception:  # noqa: BLE001
            return ""

        import openai

        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{content_type};base64,{b64_image}"},
                        },
                        {
                            "type": "text",
                            "text": "Describe this Instagram post image. Include any visible text, people, setting, and context.",
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content or ""

    async def _resolve_oembed(self, url: str) -> ResolvedContent:
        """Fallback: oEmbed API for caption text."""
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                "https://api.instagram.com/oembed/",
                params={"url": url, "omitscript": "true"},
            )
            resp.raise_for_status()
            data = resp.json()

        title = data.get("title", "")
        author = data.get("author_name", "")

        content_parts = []
        if author:
            content_parts.append(f"Instagram post by @{author}:")
            content_parts.append("")
        if title:
            content_parts.append(title)

        text = "\n".join(content_parts)
        if not text.strip():
            return ResolvedContent.error_result(url, "oEmbed returned empty content")

        metadata: dict = {}
        if author:
            metadata["author"] = f"@{author}"
            metadata["speakers"] = [{"name": author, "role": "creator", "platform": "instagram"}]

        return ResolvedContent(
            source_url=url,
            content_type="instagram",
            title=title or None,
            text=text,
            transcript=None,
            summary=None,
            expansion_urls=[],
            metadata=metadata,
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )

    async def _resolve_og_tags(self, url: str) -> ResolvedContent:
        """Last resort: fetch og:title and og:description from page HTML."""
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SeedStorage/2.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        title = _extract_meta(html, "og:title")
        description = _extract_meta(html, "og:description")

        if not description and not title:
            return ResolvedContent.error_result(url, "No og: tags found")

        text = (
            f"{title}\n\n{description}".strip()
            if title and description
            else (title or description or "")
        )

        return ResolvedContent(
            source_url=url,
            content_type="instagram",
            title=title or None,
            text=text,
            transcript=None,
            summary=None,
            expansion_urls=[],
            metadata={},
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )

    def _build_caption_result(self, url, caption, uploader, uploader_id, title, timestamp):
        """Build a ResolvedContent from caption only (when video processing fails)."""
        content_parts = []
        if uploader:
            content_parts.append(f"Instagram reel by @{uploader_id or uploader}:")
            content_parts.append("")
        if caption:
            content_parts.append(caption)

        return ResolvedContent(
            source_url=url,
            content_type="instagram",
            title=title or None,
            text="\n".join(content_parts),
            transcript=None,
            summary=None,
            expansion_urls=[],
            metadata=self._build_metadata(uploader, uploader_id, timestamp),
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )

    def _build_metadata(self, uploader, uploader_id, timestamp):
        metadata: dict = {}
        if uploader:
            metadata["author"] = f"@{uploader_id or uploader}"
            metadata["speakers"] = [{"name": uploader, "role": "creator", "platform": "instagram"}]
        if timestamp:
            metadata["published_at"] = datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
        return metadata


def _extract_meta(html: str, prop: str) -> str:
    """Extract content from <meta property="..." content="...">."""
    for pattern in [
        rf'<meta\s+property="{re.escape(prop)}"\s+content="([^"]*)"',
        rf'<meta\s+content="([^"]*)"\s+property="{re.escape(prop)}"',
        rf'<meta\s+name="{re.escape(prop)}"\s+content="([^"]*)"',
        rf'<meta\s+content="([^"]*)"\s+name="{re.escape(prop)}"',
    ]:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""
