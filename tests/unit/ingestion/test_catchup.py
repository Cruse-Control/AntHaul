"""Unit tests for ingestion/catchup.py.

All network I/O (httpx) and DB calls (staging) are mocked.
Tests cover:
- New messages get staged
- Already-staged messages are skipped
- 403 channels are skipped gracefully
- after_timestamp generates correct snowflake parameter
- Attachment-only messages are staged as web type
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_discord_msg(
    msg_id: str = "1498836204689428592",
    content: str = "Check this out https://example.com",
    author: str = "testuser",
    author_id: str = "100",
    guild_id: str = "999",
    timestamp: str = "2024-01-01T12:00:00+00:00",
    attachments: list | None = None,
) -> dict:
    return {
        "id": msg_id,
        "content": content,
        "author": {"id": author_id, "username": author},
        "guild_id": guild_id,
        "timestamp": timestamp,
        "attachments": attachments or [],
    }


def _make_http_response(status_code: int = 200, json_data=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else []
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunCatchup:
    async def test_run_catchup_stages_new_message(self):
        """A new (not-yet-staged) message should be staged via _stage_catchup_message."""
        msg = _make_discord_msg()

        mock_response = _make_http_response(200, [msg])
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with patch("ingestion.catchup.httpx.AsyncClient", return_value=mock_async_client), \
             patch("ingestion.catchup.staging.get_by_discord_msg_id", return_value=None), \
             patch("ingestion.catchup._stage_catchup_message", return_value=1) as mock_stage:
            from ingestion.catchup import run_catchup
            count = await run_catchup("fake-token", {123456789})

        mock_stage.assert_called_once_with(123456789, msg)
        assert count == 1

    async def test_run_catchup_skips_already_staged(self):
        """A message already in the staging table should NOT be staged again."""
        msg = _make_discord_msg()

        mock_response = _make_http_response(200, [msg])
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        existing_row = {"id": "some-uuid"}

        with patch("ingestion.catchup.httpx.AsyncClient", return_value=mock_async_client), \
             patch("ingestion.catchup.staging.get_by_discord_msg_id", return_value=existing_row), \
             patch("ingestion.catchup._stage_catchup_message") as mock_stage:
            from ingestion.catchup import run_catchup
            count = await run_catchup("fake-token", {123456789})

        mock_stage.assert_not_called()
        assert count == 0

    async def test_run_catchup_skips_403_channel(self):
        """A 403 response should log debug and continue without raising."""
        mock_response = _make_http_response(403)
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with patch("ingestion.catchup.httpx.AsyncClient", return_value=mock_async_client), \
             patch("ingestion.catchup._stage_catchup_message") as mock_stage:
            from ingestion.catchup import run_catchup
            # Should not raise
            count = await run_catchup("fake-token", {111111111})

        mock_stage.assert_not_called()
        assert count == 0

    async def test_run_catchup_skips_404_channel(self):
        """A 404 response should log debug and continue without raising."""
        mock_response = _make_http_response(404)
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with patch("ingestion.catchup.httpx.AsyncClient", return_value=mock_async_client), \
             patch("ingestion.catchup._stage_catchup_message") as mock_stage:
            from ingestion.catchup import run_catchup
            count = await run_catchup("fake-token", {222222222})

        mock_stage.assert_not_called()
        assert count == 0

    async def test_run_catchup_uses_after_snowflake(self):
        """When after_timestamp is given, the params must include an 'after' snowflake."""
        mock_response = _make_http_response(200, [])
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        after_ts = "2024-01-15T10:30:00+00:00"
        # Pre-compute expected snowflake
        ts_ms = int(datetime.fromisoformat(after_ts).timestamp() * 1000)
        expected_snowflake = str((ts_ms - 1420070400000) << 22)

        with patch("ingestion.catchup.httpx.AsyncClient", return_value=mock_async_client):
            from ingestion.catchup import run_catchup
            await run_catchup("fake-token", {333333333}, after_timestamp=after_ts)

        call_kwargs = mock_async_client.get.call_args
        # params can be positional or keyword
        params = call_kwargs.kwargs.get("params") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert params is not None, "No params passed to httpx GET"
        assert "after" in params, f"'after' key missing from params: {params}"
        assert params["after"] == expected_snowflake

    async def test_run_catchup_no_after_timestamp(self):
        """Without after_timestamp, params should NOT contain 'after'."""
        mock_response = _make_http_response(200, [])
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with patch("ingestion.catchup.httpx.AsyncClient", return_value=mock_async_client):
            from ingestion.catchup import run_catchup
            await run_catchup("fake-token", {444444444}, after_timestamp=None)

        call_kwargs = mock_async_client.get.call_args
        params = call_kwargs.kwargs.get("params") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert params is not None
        assert "after" not in params


class TestStageCatchupMessage:
    def test_stage_catchup_attachment_only(self):
        """Attachment-only messages (no content) are staged as web source type."""
        msg = _make_discord_msg(
            content="",
            attachments=[{"url": "https://cdn.discordapp.com/attachments/1/2/image.png"}],
        )

        with patch("ingestion.catchup.staging.stage", return_value="uuid-123") as mock_stage:
            from ingestion.catchup import _stage_catchup_message
            count = _stage_catchup_message(123456, msg)

        assert count == 1
        mock_stage.assert_called_once()
        call_kwargs = mock_stage.call_args.kwargs
        assert call_kwargs["source_type"] == "web"
        assert call_kwargs["source_uri"] == "https://cdn.discordapp.com/attachments/1/2/image.png"
        assert call_kwargs["metadata"]["backfilled"] is True

    def test_stage_catchup_no_content_no_attachments(self):
        """Empty message returns 0 without calling staging.stage."""
        msg = _make_discord_msg(content="", attachments=[])

        with patch("ingestion.catchup.staging.stage") as mock_stage:
            from ingestion.catchup import _stage_catchup_message
            count = _stage_catchup_message(123456, msg)

        assert count == 0
        mock_stage.assert_not_called()

    def test_stage_catchup_url_message(self):
        """Message with URL gets classified and staged."""
        msg = _make_discord_msg(content="Watch this https://www.youtube.com/watch?v=abc123")

        with patch("ingestion.catchup.staging.stage", return_value="uuid-456") as mock_stage:
            from ingestion.catchup import _stage_catchup_message
            count = _stage_catchup_message(123456, msg)

        assert count == 1
        call_kwargs = mock_stage.call_args.kwargs
        assert call_kwargs["source_type"] == "youtube"
        assert call_kwargs["metadata"]["backfilled"] is True

    def test_stage_catchup_plain_text(self):
        """Non-noise plain text message is staged as plain_text."""
        msg = _make_discord_msg(content="This is an insightful observation about ants")

        with patch("ingestion.catchup.staging.stage", return_value="uuid-789") as mock_stage:
            from ingestion.catchup import _stage_catchup_message
            count = _stage_catchup_message(123456, msg)

        assert count == 1
        call_kwargs = mock_stage.call_args.kwargs
        assert call_kwargs["source_type"] == "plain_text"

    def test_stage_catchup_noise_skipped(self):
        """Noise messages return 0 without staging."""
        msg = _make_discord_msg(content="ok")

        with patch("ingestion.catchup.staging.stage") as mock_stage:
            from ingestion.catchup import _stage_catchup_message
            count = _stage_catchup_message(123456, msg)

        assert count == 0
        mock_stage.assert_not_called()

    def test_stage_catchup_metadata_includes_backfilled_flag(self):
        """Backfilled metadata key must be True for all staged catchup messages."""
        msg = _make_discord_msg(content="Something worth keeping")

        captured_meta = {}

        def capture_stage(**kwargs):
            captured_meta.update(kwargs.get("metadata", {}))
            return "uuid-meta"

        with patch("ingestion.catchup.staging.stage", side_effect=capture_stage):
            from ingestion.catchup import _stage_catchup_message
            _stage_catchup_message(99999, msg)

        assert captured_meta.get("backfilled") is True
        assert captured_meta.get("discord_msg_id") == msg["id"]
        assert captured_meta.get("discord_channel_id") == "99999"
