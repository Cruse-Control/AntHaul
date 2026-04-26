"""Graphiti client singleton with provider branching and vision client factory.

All add_episode() calls must use group_id="seed-storage". Never per-channel.

The singleton is event-loop-aware: when called from a new asyncio.run()
(which creates a new event loop), the old Graphiti instance is closed and
a fresh one is created.  This avoids the "Future attached to a different
loop" RuntimeError that occurs when the Neo4j async driver's connections
are used on a loop they weren't created on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig

from seed_storage.config import settings

logger = logging.getLogger(__name__)

GROUP_ID = "seed-storage"

_graphiti: Graphiti | None = None
_loop_id: int | None = None
_init_lock: asyncio.Lock | None = None


def _get_init_lock() -> asyncio.Lock:
    """Get or create the init lock for the current event loop."""
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def _build_llm_client():
    """Build the Graphiti LLM client based on settings.LLM_PROVIDER.

    Provider branching:
    - openai  → OpenAIClient
    - anthropic → AnthropicClient
    - groq    → GroqClient
    """
    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL

    if provider == "anthropic":
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        return AnthropicClient(config=LLMConfig(api_key=settings.ANTHROPIC_API_KEY, model=model))

    if provider == "groq":
        from graphiti_core.llm_client.groq_client import GroqClient

        return GroqClient(config=LLMConfig(api_key=settings.GROQ_API_KEY, model=model))

    # default: openai
    from graphiti_core.llm_client.openai_client import OpenAIClient

    return OpenAIClient(config=LLMConfig(api_key=settings.OPENAI_API_KEY, model=model))


def _build_embedder() -> OpenAIEmbedder:
    """Build the embedder. Always OpenAIEmbedder (requires OPENAI_API_KEY)."""
    return OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=settings.OPENAI_API_KEY,
            embedding_model="text-embedding-3-small",
        )
    )


async def get_graphiti() -> Graphiti:
    """Return the Graphiti singleton, initializing on first call.

    Calls build_indices_and_constraints() on first init.
    Provider branching: openai→OpenAIClient, anthropic→AnthropicClient, groq→GroqClient.
    Embedder: always OpenAIEmbedder (requires OPENAI_API_KEY regardless of LLM_PROVIDER).

    Event-loop-aware: if the running loop differs from the one the singleton
    was created on, the old instance is closed and a new one is built.  This
    is necessary because Celery tasks call asyncio.run() per invocation,
    creating a fresh event loop each time.

    Uses an asyncio.Lock to prevent concurrent callers from racing during
    initialization (one task closing the driver while another is mid-query).
    """
    global _graphiti, _loop_id, _init_lock
    current_loop_id = id(asyncio.get_running_loop())

    # Fast path — already initialized on this loop.
    if _graphiti is not None and _loop_id == current_loop_id:
        return _graphiti

    async with _get_init_lock():
        # Re-check after acquiring lock (another task may have initialized).
        if _graphiti is not None and _loop_id == current_loop_id:
            return _graphiti

        # Close stale instance whose connections are on a dead loop
        if _graphiti is not None:
            logger.debug("graphiti: event loop changed, closing stale instance")
            try:
                await _graphiti.close()
            except Exception:
                pass
            _graphiti = None

        _graphiti = Graphiti(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            llm_client=_build_llm_client(),
            embedder=_build_embedder(),
        )
        _loop_id = current_loop_id  # Set before build_indices so concurrent waiters see it

        await _graphiti.build_indices_and_constraints()
        return _graphiti


def reset_graphiti() -> None:
    """Reset the singleton (used in tests)."""
    global _graphiti, _loop_id, _init_lock
    _graphiti = None
    _loop_id = None
    _init_lock = None


def get_vision_client() -> Any:
    """Return an SDK client for VISION_PROVIDER (defaults to LLM_PROVIDER).

    Used by the image resolver. Separate from the Graphiti LLM client.
    Returns the native SDK client (openai.OpenAI, anthropic.Anthropic, or groq.Groq).
    """
    provider = settings.VISION_PROVIDER or settings.LLM_PROVIDER

    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    if provider == "groq":
        import groq

        return groq.Groq(api_key=settings.GROQ_API_KEY)

    # default: openai
    import openai

    return openai.OpenAI(api_key=settings.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Convenience wrappers used by v1 ingestion code (loader.py, test_pipeline.py)
# ---------------------------------------------------------------------------


async def add_episode(
    *,
    name: str,
    content: str,
    source: str = "text",
    source_description: str = "",
    reference_time: Any = None,
) -> None:
    """Add an episode to the knowledge graph via Graphiti."""
    from graphiti_core.nodes import EpisodeType

    g = await get_graphiti()
    await g.add_episode(
        name=name,
        episode_body=content,
        source=EpisodeType.text,
        source_description=source_description,
        reference_time=reference_time,
        group_id=GROUP_ID,
    )


async def close() -> None:
    """Close the Graphiti singleton and release connections."""
    global _graphiti, _loop_id
    if _graphiti:
        try:
            await _graphiti.close()
        except Exception:
            pass
        _graphiti = None
        _loop_id = None


async def search(query: str, limit: int = 10) -> list:
    """Search the knowledge graph."""
    g = await get_graphiti()
    return await g.search(query, group_ids=[GROUP_ID], num_results=limit)
