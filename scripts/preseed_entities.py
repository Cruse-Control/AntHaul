"""Pre-seed core CruseControl entities into the knowledge graph.

Seeds known people, products, and organizations as Graphiti episodes so that
future ingestion can resolve mentions ("siliconwarlock" → Wyler Zahm, etc.).

Usage:
    python -m scripts.preseed_entities [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("preseed_entities")

# Core entities with aliases that should be recognized across all content.
CORE_ENTITIES = [
    {
        "name": "Flynn Cruse",
        "type": "person",
        "aliases": ["Flynn", "Flynn A. Cruse", "flynn-cruse", "flynncruse",
                    "siliconwarlock", "silicon_warlock", "SiliconWarlock", "flynnbo"],
        "description": "Co-founder and CEO of CruseControl. Business strategy, sales, client relations.",
    },
    {
        "name": "Wyler Zahm",
        "type": "person",
        "aliases": ["Wyler", "wyler-zahm", "famed_esteemed", "wylerza", "Wyler Z"],
        "description": "Co-founder and CTO of CruseControl. AI engineering, infrastructure, agent systems.",
    },
    {
        "name": "CruseControl",
        "type": "organization",
        "aliases": ["Cruse Control", "cruse-control", "CC"],
        "description": "AI agent consulting startup. Builds autonomous agent systems for enterprise clients.",
    },
    {
        "name": "AntKeeper",
        "type": "product",
        "aliases": ["ant-keeper", "Ant Keeper", "antkeeper"],
        "description": "CruseControl's task orchestration platform. Manages daemons, scheduled tasks, and agent runs.",
    },
    {
        "name": "AntHaul",
        "type": "product",
        "aliases": ["ant-haul", "Ant Haul", "anthaul", "seed-storage"],
        "description": "CruseControl's knowledge graph ingestion pipeline. Discord messages, web content, and media into Neo4j.",
    },
    {
        "name": "AntForge",
        "type": "product",
        "aliases": ["ant-forge", "Ant Forge", "antforge", "Forge"],
        "description": "CruseControl's autonomous build system. Spec-driven code generation with gate validation.",
    },
    {
        "name": "AntFarm",
        "type": "product",
        "aliases": ["ant-farm", "Ant Farm", "antfarm"],
        "description": "CruseControl's Discord bot. Routes content, manages reactions, and sends alerts.",
    },
    {
        "name": "Prospector",
        "type": "product",
        "aliases": ["prospector"],
        "description": "CruseControl's autonomous lead generation agent. Finds and qualifies potential clients.",
    },
]


def _build_episode_body(entity: dict) -> str:
    """Build a rich episode body that gives Graphiti enough context for entity extraction."""
    lines = [
        f"Entity: {entity['name']}",
        f"Type: {entity['type']}",
        f"Description: {entity['description']}",
    ]
    if entity.get("aliases"):
        lines.append(f"Also known as: {', '.join(entity['aliases'])}")
    return "\n".join(lines)


async def preseed(dry_run: bool = False):
    """Seed all core entities into the knowledge graph via Graphiti episodes."""
    from seed_storage.graphiti_client import add_episode, close

    now = datetime.now(timezone.utc)
    loaded = 0
    failed = 0

    for entity in CORE_ENTITIES:
        body = _build_episode_body(entity)
        name = f"preseed:{entity['name'].lower().replace(' ', '-')}"

        if dry_run:
            log.info("DRY RUN: would seed %s\n%s", name, body)
            continue

        try:
            await add_episode(
                name=name,
                content=body,
                source="text",
                source_description="preseed_entities",
                reference_time=now,
            )
            loaded += 1
            log.info("Seeded: %s", entity["name"])
        except Exception:
            failed += 1
            log.exception("Failed to seed: %s", entity["name"])

    if not dry_run:
        await close()

    log.info("Pre-seed complete: %d loaded, %d failed", loaded, failed)
    return {"loaded": loaded, "failed": failed}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load .env for local development
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    dry_run = "--dry-run" in sys.argv
    result = asyncio.run(preseed(dry_run=dry_run))
    print(result)
