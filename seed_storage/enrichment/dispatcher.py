"""Content dispatcher — routes URLs to resolvers by priority order.

Priority (highest to lowest):
1. TwitterResolver  — twitter.com / x.com (must come before WebpageResolver)
2. YouTubeResolver  — youtube.com / youtu.be
3. GitHubResolver   — github.com repos
4. ImageResolver    — image file extensions
5. PDFResolver      — .pdf extensions
6. VideoResolver    — video file extensions
7. WebpageResolver  — generic HTTP(S) pages
8. FallbackResolver — last resort (always matches HTTP(S))

The dispatcher tries resolvers in order and uses the first one where
``can_handle()`` returns True. If the selected resolver raises (which it
shouldn't — resolvers must catch exceptions internally), the dispatcher
catches it and returns an error_result.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from seed_storage.enrichment.models import ResolvedContent
from seed_storage.enrichment.resolvers.base import BaseResolver
from seed_storage.enrichment.resolvers.fallback import FallbackResolver
from seed_storage.enrichment.resolvers.github import GitHubResolver
from seed_storage.enrichment.resolvers.image import ImageResolver
from seed_storage.enrichment.resolvers.instagram import InstagramResolver
from seed_storage.enrichment.resolvers.pdf import PDFResolver
from seed_storage.enrichment.resolvers.twitter import TwitterResolver
from seed_storage.enrichment.resolvers.video import VideoResolver
from seed_storage.enrichment.resolvers.webpage import WebpageResolver
from seed_storage.enrichment.resolvers.youtube import YouTubeResolver

logger = logging.getLogger(__name__)

# Default resolver priority order
_DEFAULT_RESOLVERS: list[BaseResolver] = [
    TwitterResolver(),
    YouTubeResolver(),
    GitHubResolver(),
    InstagramResolver(),
    ImageResolver(),
    PDFResolver(),
    VideoResolver(),
    WebpageResolver(),
    FallbackResolver(),
]


class ContentDispatcher:
    """Routes URLs to the appropriate resolver and returns ResolvedContent.

    Can be constructed with a custom resolver list for testing.
    """

    def __init__(self, resolvers: list[BaseResolver] | None = None) -> None:
        self._resolvers = resolvers if resolvers is not None else _DEFAULT_RESOLVERS

    def _pick_resolver(self, url: str) -> BaseResolver | None:
        """Return the first resolver that can handle *url*, or None."""
        for resolver in self._resolvers:
            if resolver.can_handle(url):
                return resolver
        return None

    async def dispatch(self, url: str) -> ResolvedContent:
        """Dispatch *url* to the appropriate resolver.

        Always returns a ResolvedContent — never raises.
        Sets ``resolved_at`` on the returned object (UTC).
        """
        resolver = self._pick_resolver(url)

        if resolver is None:
            logger.warning("No resolver found for URL: %s", url)
            result = ResolvedContent.error_result(url, "No resolver matched this URL")
        else:
            resolver_name = type(resolver).__name__
            logger.debug("Dispatching %s → %s", url, resolver_name)
            try:
                result = await resolver.resolve(url)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Resolver %s raised unexpectedly for %s: %s",
                    resolver_name,
                    url,
                    exc,
                    exc_info=True,
                )
                result = ResolvedContent.error_result(url, str(exc))

        # Stamp resolved_at (dispatcher is authoritative per spec)
        object.__setattr__(result, "resolved_at", datetime.now(tz=UTC))
        return result
