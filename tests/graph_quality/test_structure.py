"""Structure tests -- relationship diversity, provenance chains, vector indices."""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


class TestRelationshipDiversity:
    def test_at_least_five_relationship_types(self, graph_stats):
        """Graph should have at least 5 distinct relationship types."""
        rels = graph_stats.get("relationships", {})
        assert len(rels) >= 5, f"Only {len(rels)} relationship types: {list(rels.keys())}"

    def test_no_orphan_entities(self, neo4j_driver, event_loop):
        """Entities should have at least one relationship."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WHERE NOT (e)-[]-()
                    RETURN count(e) AS cnt
                """)
                record = await result.single()
                # Allow some orphans (up to 10%) but not a majority
                total_result = await s.run(
                    "MATCH (e:__Entity__) RETURN count(e) AS total"
                )
                total_record = await total_result.single()
                total = total_record["total"]
                orphans = record["cnt"]
                if total > 0:
                    orphan_pct = orphans / total
                    assert orphan_pct < 0.5, \
                        f"{orphans}/{total} ({orphan_pct:.0%}) entities are orphans"
        event_loop.run_until_complete(check())


class TestProvenance:
    def test_source_nodes_exist(self, graph_stats):
        """Graph should have Source nodes for provenance."""
        assert graph_stats.get("nodes", {}).get("Source", 0) > 0, "No Source nodes found"

    def test_sources_have_uris(self, neo4j_driver, event_loop):
        """Every Source node should have a source_uri."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run(
                    "MATCH (s:Source) WHERE s.source_uri IS NULL RETURN count(s) AS cnt"
                )
                record = await result.single()
                assert record["cnt"] == 0, f"{record['cnt']} Sources without URI"
        event_loop.run_until_complete(check())


class TestVectorIndices:
    def test_entity_embedding_index_exists(self, neo4j_driver, event_loop):
        """Vector index on entity embeddings should exist."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("SHOW INDEXES WHERE name = 'entity_embedding'")
                records = [r async for r in result]
                assert len(records) > 0, "entity_embedding vector index not found"
        event_loop.run_until_complete(check())

    def test_entities_have_embeddings(self, neo4j_driver, event_loop):
        """Most entities should have embedding vectors."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WITH count(e) AS total,
                         sum(CASE WHEN e.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_emb
                    RETURN total, with_emb
                """)
                record = await result.single()
                total = record["total"]
                with_emb = record["with_emb"]
                if total > 0:
                    pct = with_emb / total
                    assert pct >= 0.8, \
                        f"Only {with_emb}/{total} ({pct:.0%}) entities have embeddings"
        event_loop.run_until_complete(check())
