"""Twitter/X resolver — FxTwitter API (free, no auth).

Extracts tweet text, author, media descriptions, quote tweets, and URLs
via the public FxTwitter API (api.fxtwitter.com). Also checks the author's
thread replies for additional links (tweets are often teasers with the real
article URL in a reply).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from seed_storage.enrichment.models import ResolvedContent
from seed_storage.enrichment.resolvers.base import BaseResolver

logger = logging.getLogger(__name__)

_TWITTER_HOSTS = {"twitter.com", "www.twitter.com", "x.com", "www.x.com", "mobile.twitter.com"}
_TIMEOUT = 15.0
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _extract_tweet_info(url: str) -> tuple[str, str]:
    """Extract (username, tweet_id) from x.com or twitter.com URL."""
    parts = url.rstrip("/").split("/")
    username = ""
    tweet_id = ""
    for i, p in enumerate(parts):
        if p in (
            "x.com",
            "twitter.com",
            "www.x.com",
            "www.twitter.com",
            "mobile.twitter.com",
        ) and i + 1 < len(parts):
            username = parts[i + 1]
        if p == "status" and i + 1 < len(parts):
            tid = parts[i + 1].split("?")[0]
            if tid.isdigit():
                tweet_id = tid
    return username, tweet_id


def _extract_urls_from_text(text: str) -> list[str]:
    """Extract HTTP(S) URLs from text, filtering out twitter/x.com links."""
    urls = []
    for match in _URL_RE.finditer(text):
        u = match.group(0).rstrip(".,;:)!?]")
        parsed = urlparse(u)
        host = (parsed.hostname or "").lower()
        # Skip self-referential twitter links and t.co shortened
        if host in _TWITTER_HOSTS or host == "t.co":
            continue
        urls.append(u)
    return urls


class TwitterResolver(BaseResolver):
    """Extracts tweet content via FxTwitter API (free, no auth required).

    Extracts: tweet text, author handle/name, media descriptions,
    quote tweet content, engagement metrics, URLs for expansion.
    Also checks the thread (author's replies) for additional links.
    """

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower() in _TWITTER_HOSTS

    async def resolve(self, url: str) -> ResolvedContent:
        username, tweet_id = _extract_tweet_info(url)
        if not tweet_id:
            return ResolvedContent.error_result(url, "Could not parse tweet ID from URL")

        try:
            return await self._fetch_fxtwitter(url, username, tweet_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TwitterResolver failed for %s: %s", url, exc)
            return ResolvedContent.error_result(url, str(exc))

    async def _fetch_fxtwitter(self, url: str, username: str, tweet_id: str) -> ResolvedContent:
        fx_path = f"{username}/status/{tweet_id}" if username else f"i/status/{tweet_id}"

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"https://api.fxtwitter.com/{fx_path}",
                headers={"User-Agent": "SeedStorage/2.0", "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            tweet = data.get("tweet", {})

            author = tweet.get("author", {})
            author_name = author.get("name", "")
            author_handle = f"@{author.get('screen_name', '')}"
            author_screen = author.get("screen_name", "")
            text = tweet.get("text", "")

            content_parts = [f"Tweet by {author_name} ({author_handle}):", "", text]

            # Media descriptions
            media = tweet.get("media", {})
            if media:
                for photo in media.get("photos", []):
                    alt = photo.get("altText", "")
                    if alt:
                        content_parts.append(f"\n[Image: {alt}]")
                for video in media.get("videos", []):
                    dur = video.get("duration", 0)
                    content_parts.append(f"\n[Video: {dur}s]")

            # Quote tweet
            expansion_urls: list[str] = []
            quote = tweet.get("quote", {})
            if quote:
                q_author = quote.get("author", {})
                q_name = q_author.get("name", "")
                q_handle = f"@{q_author.get('screen_name', '')}"
                q_text = quote.get("text", "")
                content_parts.append(f"\nQuoting {q_name} ({q_handle}):")
                content_parts.append(q_text)
                q_url = quote.get("url", "")
                if q_url:
                    expansion_urls.append(q_url)
                # Extract URLs from quoted tweet text
                expansion_urls.extend(_extract_urls_from_text(q_text))

            # Extract URLs from tweet text
            expansion_urls.extend(_extract_urls_from_text(text))

            # Extract URLs from API urls field
            urls_in_tweet = tweet.get("urls", [])
            if isinstance(urls_in_tweet, list):
                for u in urls_in_tweet:
                    if isinstance(u, dict):
                        expanded = u.get("expanded_url") or u.get("url", "")
                        if expanded:
                            expansion_urls.append(expanded)
                    elif isinstance(u, str):
                        expansion_urls.append(u)

            # Check thread: fetch author's reply to this tweet for additional links
            thread_text = await self._fetch_thread_replies(client, author_screen, tweet_id)
            if thread_text:
                content_parts.append(f"\n[Thread reply by {author_handle}]:")
                content_parts.append(thread_text)
                expansion_urls.extend(_extract_urls_from_text(thread_text))

        # Deduplicate expansion_urls
        seen: set[str] = set()
        unique_urls = []
        for u in expansion_urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        content = "\n".join(content_parts)

        metadata: dict = {
            "author": author_handle,
            "author_name": author_name,
            "tweet_id": tweet_id,
            "speakers": [
                {"name": author_name or author_handle, "role": "author", "platform": "x.com"}
            ],
        }
        created_at = tweet.get("created_at")
        if created_at:
            metadata["published_at"] = created_at
        likes = tweet.get("likes", 0)
        retweets = tweet.get("retweets", 0)
        if likes or retweets:
            metadata["engagement"] = {"likes": likes, "retweets": retweets}

        return ResolvedContent(
            source_url=url,
            content_type="tweet",
            title=f"Tweet by {author_name}",
            text=content,
            transcript=None,
            summary=None,
            expansion_urls=unique_urls[:20],
            metadata=metadata,
            extraction_error=None,
            resolved_at=datetime.now(tz=UTC),
        )

    async def _fetch_thread_replies(
        self, client: httpx.AsyncClient, author_screen: str, tweet_id: str
    ) -> str:
        """Try to fetch the author's immediate reply (thread continuation).

        FxTwitter doesn't have a replies endpoint, but if the tweet has a
        conversation_id or reply chain, we can check if the author replied
        to their own tweet (common pattern for thread starters with links).
        We try fetching tweet_id+1 through tweet_id+3 as a heuristic.
        """
        if not author_screen:
            return ""

        # Heuristic: check a few IDs after this tweet for self-replies
        # Twitter IDs are sequential-ish within a short time window
        try:
            base_id = int(tweet_id)
        except ValueError:
            return ""

        reply_texts: list[str] = []
        for offset in range(1, 4):
            try:
                next_id = str(base_id + offset)
                resp = await client.get(
                    f"https://api.fxtwitter.com/{author_screen}/status/{next_id}",
                    headers={"User-Agent": "SeedStorage/2.0", "Accept": "application/json"},
                    timeout=5,
                )
                if resp.status_code != 200:
                    continue
                reply_data = resp.json()
                reply_tweet = reply_data.get("tweet", {})
                reply_author = reply_tweet.get("author", {}).get("screen_name", "")
                # Only include if it's the same author replying to the original tweet
                if reply_author.lower() == author_screen.lower():
                    reply_text = reply_tweet.get("text", "")
                    if reply_text:
                        reply_texts.append(reply_text)
            except Exception:  # noqa: BLE001
                continue

        return "\n".join(reply_texts)
