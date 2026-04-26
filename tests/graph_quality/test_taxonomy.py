"""Taxonomy tests -- verify typed entities, label distribution, no untyped nodes."""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


class TestEntityTyping:
    def test_entity_nodes_have_type_label(self, neo4j_driver, event_loop):
        """Every __Entity__ node must also have a type label (Person, Concept, etc.)."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run(
                    "MATCH (e:__Entity__) WHERE size(labels(e)) < 2 RETURN count(e) AS cnt"
                )
                record = await result.single()
                assert record["cnt"] == 0, f"{record['cnt']} entities missing type label"
        event_loop.run_until_complete(check())

    def test_at_least_three_entity_types(self, graph_stats):
        """Graph should have at least 3 distinct entity types."""
        entity_types = {"Person", "Organization", "Product", "Concept", "Location", "Event"}
        present = [t for t in entity_types if t in graph_stats.get("nodes", {})]
        assert len(present) >= 3, f"Only {len(present)} entity types: {present}"

    def test_no_url_entities(self, neo4j_driver, event_loop):
        """No entity names that look like URLs."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run(
                    "MATCH (e:__Entity__) WHERE e.name STARTS WITH 'http' RETURN count(e) AS cnt"
                )
                record = await result.single()
                assert record["cnt"] == 0, f"{record['cnt']} URL entities found"
        event_loop.run_until_complete(check())

    def test_no_generic_phrase_entities(self, neo4j_driver, event_loop):
        """No entities that are generic phrases."""
        async def check():
            generic = ["the project", "this tool", "the system", "the team"]
            async with neo4j_driver.session() as s:
                for phrase in generic:
                    result = await s.run(
                        "MATCH (e:__Entity__) WHERE toLower(e.name) = $name "
                        "RETURN count(e) AS cnt",
                        name=phrase,
                    )
                    record = await result.single()
                    assert record["cnt"] == 0, f"Generic entity found: '{phrase}'"
        event_loop.run_until_complete(check())

    def test_entity_count_reasonable(self, graph_stats):
        """Graph should have between 10 and 5000 entities."""
        count = graph_stats.get("nodes", {}).get("__Entity__", 0)
        assert 10 <= count <= 5000, f"Entity count {count} outside expected range"

    def test_multiple_node_types(self, graph_stats):
        """Graph should use multiple distinct node labels."""
        node_count = len(graph_stats.get("nodes", {}))
        assert node_count >= 4, \
            f"Only {node_count} node types: {list(graph_stats.get('nodes', {}).keys())}"
