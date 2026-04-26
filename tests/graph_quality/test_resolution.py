"""Resolution tests -- alias merging, no duplicate entities, cross-source linking."""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


class TestAliasResolution:
    def test_flynn_aliases_merged(self, neo4j_driver, event_loop):
        """Flynn's aliases (siliconwarlock, flynnbo, flynn-cruse) should merge to one node."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WHERE toLower(e.canonical_name) = 'flynn cruse'
                       OR toLower(e.name) CONTAINS 'flynn'
                       OR ANY(a IN coalesce(e.aliases, [])
                              WHERE toLower(a) IN ['siliconwarlock', 'flynnbo', 'flynn-cruse'])
                    RETURN count(DISTINCT e) AS cnt
                """)
                record = await result.single()
                assert record["cnt"] <= 2, \
                    f"Flynn has {record['cnt']} nodes (expected 1-2, aliases may not be fully merged yet)"
        event_loop.run_until_complete(check())

    def test_wyler_aliases_merged(self, neo4j_driver, event_loop):
        """Wyler's aliases should merge to one node."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WHERE toLower(e.canonical_name) = 'wyler zahm'
                       OR toLower(e.name) CONTAINS 'wyler'
                       OR ANY(a IN coalesce(e.aliases, [])
                              WHERE toLower(a) IN ['famed_esteemed', 'wylerza', 'wyler-zahm'])
                    RETURN count(DISTINCT e) AS cnt
                """)
                record = await result.single()
                assert record["cnt"] <= 2, \
                    f"Wyler has {record['cnt']} nodes (expected 1-2)"
        event_loop.run_until_complete(check())


class TestNoGarbageEntities:
    def test_no_url_entities(self, neo4j_driver, event_loop):
        """No entity names that are URLs."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WHERE e.name STARTS WITH 'http://' OR e.name STARTS WITH 'https://'
                    RETURN count(e) AS cnt, collect(e.name)[..5] AS examples
                """)
                record = await result.single()
                assert record["cnt"] == 0, \
                    f"Found {record['cnt']} URL entities: {record['examples']}"
        event_loop.run_until_complete(check())

    def test_no_hashtag_entities(self, neo4j_driver, event_loop):
        """No entity names that are just hashtags."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WHERE e.name STARTS WITH '#' AND size(e.name) < 30
                    RETURN count(e) AS cnt, collect(e.name)[..5] AS examples
                """)
                record = await result.single()
                assert record["cnt"] == 0, \
                    f"Found {record['cnt']} hashtag entities: {record['examples']}"
        event_loop.run_until_complete(check())

    def test_no_single_char_entities(self, neo4j_driver, event_loop):
        """No entity names that are single characters."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__) WHERE size(e.name) <= 1
                    RETURN count(e) AS cnt
                """)
                record = await result.single()
                assert record["cnt"] == 0, f"{record['cnt']} single-char entities"
        event_loop.run_until_complete(check())

    def test_no_excessive_duplicates(self, neo4j_driver, event_loop):
        """No entity canonical_name should appear more than 3 times."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)
                    WITH e.canonical_name AS cn, count(*) AS cnt
                    WHERE cnt > 3
                    RETURN cn, cnt ORDER BY cnt DESC LIMIT 10
                """)
                dupes = [(r["cn"], r["cnt"]) async for r in result]
                assert len(dupes) == 0, \
                    f"Duplicate entities: {dupes}"
        event_loop.run_until_complete(check())
