"""Fallback resolver — best-effort HTML extraction.

Used when no other resolver matches. Performs a simple HTTP GET and
extracts text using BeautifulSoup. Never raises — returns a minimal
ResolvedContent on any failure.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse

import httpx

from seed_storage.enrichment.models import ResolvedContent
from seed_storage.enrichment.resolvers.base import BaseResolver

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0


class FallbackResolver(BaseResolver):
    """Last-resort resolver using BeautifulSoup text extraction.

    Intentionally lenient — accepts any HTTP(S) URL and never raises.
    """

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https")

    async def resolve(self, url: str) -> ResolvedContent:
        """Best-effort extraction. Never raises."""
        try:
            return await self._resolve_internal(url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("FallbackResolver failed for %s: %s", url, exc)
            # Return a minimal result — not an error_result so text can be ""
            return ResolvedContent(
                source_url=url,
                content_type="webpage",
                title=None,
                text="",
                transcript=None,
                summary=None,
                expansion_urls=[],
                metadata={"fallback": True},
                extraction_error=str(exc),
                resolved_at=datetime.now(tz=UTC),
            )

    async def _resolve_internal(self, url: str) -> ResolvedContent:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=_TIMEOUT,
                headers={"User-Agent": "seed-storage/2.0"},
            ) as client:
                response = await client.get(url)
                html = response.text
        except httpx.TimeoutException:
            return ResolvedContent(
                source_url=url,
                content_type="webpage",
                title=None,
                text="",
                transcript=None,
                summary=None,
                expansion_urls=[],
                metadata={"fallback": True},
                extraction_error="Request timed out",
                resolved_at=datetime.now(tz=UTC),
            )
        except Exception as exc:  # noqa: BLE001
            return ResolvedContent(
                source_url=url,
                content_type="webpage",
                title=None,
                text="",
                transcript=None,
                summary=None,
                expansion_urls=[],
                metadata={"fallback": True},
                extraction_error=str(exc),
                resolved_at=datetime.now(tz=UTC),
            )

        title: str | None = None
        text = ""
        expansion_urls: list[str] = []

        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(html, "lxml")

            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Extract expansion links from <a href>
            final_url = str(response.url)
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.startswith(("http://", "https://")):
                    expansion_urls.append(href)
                elif href.startswith("/"):
                    parsed = urlparse(final_url)
                    expansion_urls.append(f"{parsed.scheme}://{parsed.netloc}{href}")
                elif not href.startswith(("#", "mailto:", "javascript:")):
                    expansion_urls.append(urljoin(final_url, href))
            # Deduplicate
            seen: set[str] = set()
            expansion_urls = [
                u for u in expansion_urls if u.startswith("http") and not (u in seen or seen.add(u))
            ]  # type: ignore[func-returns-value]

            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Get text
            raw_text = soup.get_text(separator=" ")
            # Collapse whitespace
            text = re.sub(r"\s+", " ", raw_text).strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("BeautifulSoup parsing failed for %s: %s", url, exc)
            # Fall back to simple tag stripping
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()

        return ResolvedContent(
            source_url=url,
            content_type="webpage",
            title=title,
            text=text,
            transcript=None,
            summary=None,
            expansion_urls=expansion_urls[:50],
            metadata={"fallback": True},
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )
