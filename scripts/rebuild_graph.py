#!/usr/bin/env python3
"""Rebuild the Neo4j graph from scratch using Postgres staging data.

Usage:
    python scripts/rebuild_graph.py --dry-run
    python scripts/rebuild_graph.py --confirm
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

log = logging.getLogger("rebuild")


async def rebuild(dry_run: bool = False) -> None:
    from seed_storage.graph import get_driver, init_schema, close
    from seed_storage import staging
    from seed_storage.preseed import init_preseed_table, seed_defaults

    driver = await get_driver()

    if dry_run:
        counts = staging.count_by_status()
        extracted = counts.get("extracted", 0) + counts.get("loaded", 0)
        print(f"Would rebuild graph from {extracted} items")
        print(f"  Current status counts: {counts}")
        return

    # 1. Wipe Neo4j
    log.info("Wiping Neo4j graph...")
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    # 2. Re-init schema
    log.info("Re-initializing schema...")
    await init_schema()

    # 3. Seed preseed entities
    init_preseed_table()
    seed_defaults()

    # 4. Reset all loaded items back to extracted
    count = staging.reset_to_status("extracted", source_statuses=["loaded"])
    log.info("Reset %d items from loaded to extracted", count)

    # 5. Run loader
    from ingestion.loader import load_batch
    result = await load_batch(limit=5000)
    log.info("Loader result: %s", result)

    # 6. Run community detection (if GDS available)
    try:
        from seed_storage.communities import check_gds_available, run_leiden, summarize_communities
        if await check_gds_available():
            leiden_result = await run_leiden()
            log.info("Leiden result: %s", leiden_result)
            summarized = await summarize_communities()
            log.info("Summarized %d communities", summarized)
    except Exception as exc:
        log.warning("Community detection skipped: %s", exc)

    await close()
    log.info("Rebuild complete")


def main():
    parser = argparse.ArgumentParser(description="Rebuild Neo4j graph from Postgres")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true",
                        help="Required to actually wipe and rebuild")
    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        print("This will WIPE the Neo4j graph and rebuild. Use --confirm to proceed.")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(rebuild(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
