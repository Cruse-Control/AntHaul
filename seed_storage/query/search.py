"""Graph search -- direct Neo4j vector + fulltext hybrid queries."""
from __future__ import annotations

from seed_storage.embeddings import embed_text
from seed_storage.graph import hybrid_search, vector_search, fulltext_search


async def search(query: str, num_results: int = 10) -> list[dict]:
    """Search the knowledge graph for entities matching the query.

    Uses hybrid vector + fulltext search on __Entity__ and Fact nodes.
    """
    embedding = await embed_text(query)
    return await hybrid_search(query=query, embedding=embedding, limit=num_results)
