"""Embedding client — always OpenAI (text-embedding-3-small, 1536 dims)."""

from __future__ import annotations

from .config import settings

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    from openai import OpenAI
    _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


async def embed_text(text: str) -> list[float]:
    """Embed a text string. Returns vector of 1536 dimensions."""
    client = _get_client()
    result = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
    return result.data[0].embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts."""
    client = _get_client()
    result = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in result.data]
