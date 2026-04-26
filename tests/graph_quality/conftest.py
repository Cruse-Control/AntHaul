"""Fixtures for graph quality tests. Requires loaded Neo4j + Postgres."""
from __future__ import annotations

import asyncio

import pytest

from seed_storage.graph import get_driver, get_stats


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def neo4j_driver(event_loop):
    """Provide Neo4j driver for graph quality tests."""
    return event_loop.run_until_complete(get_driver())


@pytest.fixture(scope="session")
def graph_stats(event_loop, neo4j_driver):
    """Pre-load graph stats for all quality tests."""
    return event_loop.run_until_complete(get_stats())


def _run(coro, loop):
    """Helper to run async code in the session loop."""
    return loop.run_until_complete(coro)
