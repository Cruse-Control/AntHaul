"""Unit tests for seed_storage/query/search.py.

Tests cover:
- hybrid_search is called with correct parameters
- embed_text is called for the query
- Empty results handling
- Error propagation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestSearchCallsHybrid:
    @pytest.mark.asyncio
    async def test_search_calls_hybrid_search(self):
        """search() must call hybrid_search with embedded query."""
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        mock_hybrid = AsyncMock(return_value=[])

        with patch("seed_storage.query.search.embed_text", mock_embed), \
             patch("seed_storage.query.search.hybrid_search", mock_hybrid):
            from seed_storage.query.search import search
            await search("test query")

        mock_embed.assert_called_once_with("test query")
        mock_hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_passes_num_results_as_limit(self):
        """num_results parameter maps to limit in hybrid_search."""
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        mock_hybrid = AsyncMock(return_value=[])

        with patch("seed_storage.query.search.embed_text", mock_embed), \
             patch("seed_storage.query.search.hybrid_search", mock_hybrid):
            from seed_storage.query.search import search
            await search("query", num_results=25)

        call_kwargs = mock_hybrid.call_args
        assert call_kwargs.kwargs.get("limit") == 25


class TestReturnValues:
    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(self):
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        mock_hybrid = AsyncMock(return_value=[])

        with patch("seed_storage.query.search.embed_text", mock_embed), \
             patch("seed_storage.query.search.hybrid_search", mock_hybrid):
            from seed_storage.query.search import search
            results = await search("no match")

        assert results == []

    @pytest.mark.asyncio
    async def test_results_are_dicts(self):
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        mock_hybrid = AsyncMock(return_value=[
            {"node": {"id": "1", "name": "Test"}, "score": 0.9},
        ])

        with patch("seed_storage.query.search.embed_text", mock_embed), \
             patch("seed_storage.query.search.hybrid_search", mock_hybrid):
            from seed_storage.query.search import search
            results = await search("query")

        assert len(results) == 1
        assert results[0]["node"]["name"] == "Test"
        assert results[0]["score"] == 0.9


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_embed_error_propagates(self):
        mock_embed = AsyncMock(side_effect=RuntimeError("API error"))

        with patch("seed_storage.query.search.embed_text", mock_embed):
            from seed_storage.query.search import search
            with pytest.raises(RuntimeError, match="API error"):
                await search("query")

    @pytest.mark.asyncio
    async def test_hybrid_search_error_propagates(self):
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        mock_hybrid = AsyncMock(side_effect=ConnectionError("Neo4j down"))

        with patch("seed_storage.query.search.embed_text", mock_embed), \
             patch("seed_storage.query.search.hybrid_search", mock_hybrid):
            from seed_storage.query.search import search
            with pytest.raises(ConnectionError, match="Neo4j down"):
                await search("query")
