"""Webpage resolver — trafilatura with readability-lxml fallback.

Extracts clean article text from web pages. Truncates at 8000 tokens
(approx 32 000 chars). Collects hyperlinks as expansion_urls.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC
from urllib.parse import urljoin, urlparse

import httpx

from seed_storage.enrichment.models import ResolvedContent
from seed_storage.enrichment.resolvers.base import BaseResolver

logger = logging.getLogger(__name__)

# ~4 chars per token × 8000 tokens
_MAX_CHARS = 32_000
_TIMEOUT = 30.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract absolute HTTP(S) links from raw HTML."""
    urls = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
        href = match.group(1).strip()
        if href.startswith(("http://", "https://")):
            urls.append(href)
        elif href.startswith("/"):
            parsed = urlparse(base_url)
            urls.append(f"{parsed.scheme}://{parsed.netloc}{href}")
        elif not href.startswith(("#", "mailto:", "javascript:")):
            urls.append(urljoin(base_url, href))
    # Deduplicate while preserving order
    seen: set[str] = set()
    result = []
    for u in urls:
        if u not in seen and u.startswith("http"):
            seen.add(u)
            result.append(u)
    return result


class WebpageResolver(BaseResolver):
    """Extracts article text from general web pages.

    Resolution order:
    1. trafilatura (preferred — cleaner output)
    2. readability-lxml (fallback)
    3. error_result if both fail
    """

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https")

    async def resolve(self, url: str) -> ResolvedContent:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
                final_url = str(response.url)
        except httpx.TimeoutException:
            return ResolvedContent.error_result(url, "Request timed out")
        except httpx.HTTPStatusError as exc:
            return ResolvedContent.error_result(url, f"HTTP {exc.response.status_code}")
        except Exception as exc:  # noqa: BLE001
            return ResolvedContent.error_result(url, str(exc))

        expansion_urls = _extract_links(html, final_url)

        # --- Attempt 1: trafilatura ---
        text: str | None = None
        title: str | None = None
        try:
            import trafilatura  # type: ignore[import-untyped]

            text = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=True,
                favor_recall=True,
                deduplicate=True,
            )
            if text:
                meta = trafilatura.extract_metadata(html)
                if meta:
                    title = meta.title
        except Exception as exc:  # noqa: BLE001
            logger.debug("trafilatura failed for %s: %s", url, exc)

        # --- Attempt 2: readability-lxml ---
        if not text:
            try:
                from readability import Document  # type: ignore[import-untyped]

                doc = Document(html)
                title = doc.title()
                # readability returns HTML — strip tags
                content_html = doc.summary()
                text = re.sub(r"<[^>]+>", " ", content_html)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) < 50:
                    text = None
            except Exception as exc:  # noqa: BLE001
                logger.debug("readability fallback failed for %s: %s", url, exc)

        if not text:
            return ResolvedContent.error_result(url, "Could not extract text from page")

        # Truncate
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS]

        return ResolvedContent(
            source_url=url,
            content_type="webpage",
            title=title,
            text=text,
            transcript=None,
            summary=None,
            expansion_urls=expansion_urls,
            metadata={"final_url": final_url},
            extraction_error=None,
            resolved_at=_utcnow(),
        )


def _utcnow():
    from datetime import datetime

    return datetime.now(tz=UTC)
