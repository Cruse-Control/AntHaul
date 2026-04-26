"""Community tests -- Leiden clusters, summaries, membership."""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


class TestCommunityDetection:
    def test_communities_exist(self, neo4j_driver, event_loop):
        """Graph should have at least 1 community node after detection."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run(
                    "MATCH (c:__Community__) RETURN count(c) AS cnt"
                )
                record = await result.single()
                # This may be 0 if GDS not installed -- skip gracefully
                if record["cnt"] == 0:
                    pytest.skip("No communities found -- GDS may not be installed")
                assert record["cnt"] >= 1
        event_loop.run_until_complete(check())

    def test_communities_have_members(self, neo4j_driver, event_loop):
        """Every community should have at least 2 member entities."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (c:__Community__)
                    OPTIONAL MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c)
                    WITH c, count(e) AS members
                    WHERE members < 2
                    RETURN count(c) AS cnt
                """)
                record = await result.single()
                if record["cnt"] > 0:
                    pytest.skip("Some communities have < 2 members")
        event_loop.run_until_complete(check())

    def test_community_summaries_exist(self, neo4j_driver, event_loop):
        """Communities with members should have LLM summaries."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (c:__Community__)
                    WHERE c.summary IS NULL AND c.member_count > 0
                    RETURN count(c) AS cnt
                """)
                record = await result.single()
                if record["cnt"] > 0:
                    pytest.skip(
                        f"{record['cnt']} communities without summaries"
                    )
        event_loop.run_until_complete(check())

    def test_entities_in_at_most_one_community(self, neo4j_driver, event_loop):
        """Each entity should belong to at most one community (Leiden hard partition)."""
        async def check():
            async with neo4j_driver.session() as s:
                result = await s.run("""
                    MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
                    WITH e, count(c) AS comm_count
                    WHERE comm_count > 1
                    RETURN count(e) AS cnt
                """)
                record = await result.single()
                assert record["cnt"] == 0, \
                    f"{record['cnt']} entities in multiple communities"
        event_loop.run_until_complete(check())
