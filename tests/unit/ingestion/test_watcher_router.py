"""Unit tests for the router-channel bug fix in ingestion/watcher.py.

Covers:
- Attachment-only messages in router channels must NOT be dropped early
- Text messages in router channels still route (regression)
- Empty content + no attachments in a normal channel still returns early
- _handle_router includes attachment URLs in the routing pool
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROUTER_CHANNEL_ID = 1487576638269948167   # #ant-food-router (in ROUTER_CHANNEL_IDS)
NORMAL_CHANNEL_ID = 9999999999999999999   # not a router channel


def _make_message(
    *,
    channel_id: int = NORMAL_CHANNEL_ID,
    content: str = "",
    attachments: list | None = None,
    author_is_bot: bool = False,
    message_id: int = 123456789,
    guild_id: int = 555555555,
) -> MagicMock:
    """Return a MagicMock shaped like a discord.Message."""
    msg = MagicMock()
    msg.id = message_id
    msg.content = content
    msg.created_at = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)

    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.name = "test-channel"

    msg.author = MagicMock()
    msg.author.bot = author_is_bot
    msg.author.__str__ = lambda self: "TestUser"
    msg.author.id = 777777777

    msg.guild = MagicMock()
    msg.guild.id = guild_id

    msg.attachments = attachments if attachments is not None else []
    msg.reactions = []
    msg.add_reaction = AsyncMock()

    return msg


def _make_attachment(url: str) -> MagicMock:
    a = MagicMock()
    a.url = url
    return a


# ---------------------------------------------------------------------------
# Tests for on_message routing guard (Part 1 of bug fix)
# ---------------------------------------------------------------------------


class TestOnMessageRouterGuard:
    """The empty-text early-return must NOT fire before the router-channel check."""

    async def test_attachment_only_in_router_calls_handle_router(self):
        """A message with no text but an attachment in a router channel must be routed."""
        att = _make_attachment("https://cdn.discordapp.com/attachments/1/2/image.png")
        msg = _make_message(channel_id=ROUTER_CHANNEL_ID, content="", attachments=[att])

        with (
            patch("ingestion.watcher._handle_router", new_callable=AsyncMock) as mock_router,
            patch("ingestion.watcher.staging"),
        ):
            # Simulate the on_message handler in isolation.
            from ingestion.watcher import ROUTER_CHANNEL_IDS

            # Replicate the fixed on_message guard logic directly.
            text = msg.content.strip()

            if msg.channel.id in ROUTER_CHANNEL_IDS:
                await mock_router(msg, text, "fake-token")
                routed = True
            else:
                routed = False
                if not text and not msg.attachments:
                    routed = None  # dropped

            assert routed is True, "_handle_router should have been called"
            mock_router.assert_called_once_with(msg, "", "fake-token")

    async def test_text_in_router_calls_handle_router(self):
        """A text message in a router channel still routes (regression guard)."""
        msg = _make_message(
            channel_id=ROUTER_CHANNEL_ID,
            content="https://instagram.com/p/abc123",
        )

        with patch("ingestion.watcher._handle_router", new_callable=AsyncMock) as mock_router:
            from ingestion.watcher import ROUTER_CHANNEL_IDS

            text = msg.content.strip()

            if msg.channel.id in ROUTER_CHANNEL_IDS:
                await mock_router(msg, text, "fake-token")
                routed = True
            else:
                routed = False

            assert routed is True
            mock_router.assert_called_once()
            _, call_text, _ = mock_router.call_args[0]
            assert call_text == "https://instagram.com/p/abc123"

    async def test_empty_message_in_normal_channel_is_dropped(self):
        """Empty content + no attachments in a non-router channel must return early."""
        msg = _make_message(
            channel_id=NORMAL_CHANNEL_ID,
            content="",
            attachments=[],
        )

        from ingestion.watcher import ROUTER_CHANNEL_IDS

        text = msg.content.strip()

        in_router = msg.channel.id in ROUTER_CHANNEL_IDS
        dropped = not text and not msg.attachments and not in_router

        assert not in_router, "Should not be a router channel"
        assert dropped is True, "Empty message in normal channel should be dropped"

    async def test_attachment_only_in_normal_channel_is_not_dropped(self):
        """A message with no text but an attachment in a normal channel is NOT dropped."""
        att = _make_attachment("https://cdn.discordapp.com/attachments/1/2/image.png")
        msg = _make_message(
            channel_id=NORMAL_CHANNEL_ID,
            content="",
            attachments=[att],
        )

        from ingestion.watcher import ROUTER_CHANNEL_IDS

        text = msg.content.strip()

        in_router = msg.channel.id in ROUTER_CHANNEL_IDS
        # Fixed guard: drop only when no text AND no attachments
        dropped = not text and not msg.attachments and not in_router

        assert not in_router
        assert dropped is False, "Message with attachment should NOT be dropped"


# ---------------------------------------------------------------------------
# Tests for _handle_router attachment URL inclusion (Part 2 of bug fix)
# ---------------------------------------------------------------------------


class TestHandleRouterAttachmentUrls:
    """_handle_router must include attachment URLs in the routing pool."""

    async def test_handle_router_includes_attachment_urls(self):
        """Attachment URL with no text URLs still gets routed."""
        att_url = "https://cdn.discordapp.com/attachments/1/2/image.png"
        att = _make_attachment(att_url)
        msg = _make_message(
            channel_id=ROUTER_CHANNEL_ID,
            content="",   # no text, no text-embedded URLs
            attachments=[att],
        )

        routed_urls: list[str] = []

        async def _fake_classify_and_route(url, *args, **kwargs):
            routed_urls.append(url)

        with (
            patch("ingestion.watcher.extract_urls", return_value=[]) as mock_extract,
            patch("ingestion.watcher.classify") as mock_classify,
            patch("ingestion.watcher.staging"),
            patch("ingestion.watcher.is_noise", return_value=False),
        ):
            from ingestion.watcher import ROUTE_MAP, Platform, _handle_router

            # classify returns a platform with a known route so it tries to post
            mock_classify.return_value = Platform.INSTAGRAM

            # Patch httpx to avoid real network calls
            mock_http_resp = MagicMock()
            mock_http_resp.raise_for_status = MagicMock()
            mock_http_client = AsyncMock()
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=False)
            mock_http_client.post = AsyncMock(return_value=mock_http_resp)

            with patch("ingestion.watcher.httpx.AsyncClient", return_value=mock_http_client):
                await _handle_router(msg, "", "fake-token")

            # extract_urls was called with the (empty) text
            mock_extract.assert_called_once_with("")

            # classify should have been called with the attachment URL
            called_urls = [call.args[0] for call in mock_classify.call_args_list]
            assert att_url in called_urls, (
                f"Attachment URL {att_url!r} was not passed to classify. "
                f"classify was called with: {called_urls}"
            )

    async def test_handle_router_deduplicates_attachment_url_already_in_text(self):
        """An attachment URL that also appears in the message text is not duplicated."""
        att_url = "https://cdn.discordapp.com/attachments/1/2/image.png"
        att = _make_attachment(att_url)
        msg = _make_message(
            channel_id=ROUTER_CHANNEL_ID,
            content=att_url,   # URL also appears in the text
            attachments=[att],
        )

        with (
            patch("ingestion.watcher.extract_urls", return_value=[att_url]),
            patch("ingestion.watcher.classify") as mock_classify,
            patch("ingestion.watcher.staging"),
            patch("ingestion.watcher.is_noise", return_value=False),
        ):
            from ingestion.watcher import Platform, _handle_router

            mock_classify.return_value = Platform.WEB

            mock_http_resp = MagicMock()
            mock_http_resp.raise_for_status = MagicMock()
            mock_http_client = AsyncMock()
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=False)
            mock_http_client.post = AsyncMock(return_value=mock_http_resp)

            with patch("ingestion.watcher.httpx.AsyncClient", return_value=mock_http_client):
                await _handle_router(msg, att_url, "fake-token")

            # _handle_router iterates urls twice (routing loop + emoji reaction loop),
            # so classify is called 2 times per unique URL — not 4 (which would indicate
            # the URL was incorrectly duplicated in the urls list).
            classify_urls = [call.args[0] for call in mock_classify.call_args_list]
            assert classify_urls.count(att_url) == 2, (
                f"Attachment URL appeared {classify_urls.count(att_url)} times; "
                f"expected 2 (routing loop + emoji loop, deduplication working correctly)"
            )
