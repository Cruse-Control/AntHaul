#!/usr/bin/env python3
"""CLI query interface for the AntHaul knowledge graph.

Usage:
    python scripts/query.py "your query here"
    python scripts/query.py "your query here" --limit 20
    python scripts/query.py "your query here" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _run_query(query: str, limit: int, output_json: bool) -> None:
    """Execute a search query and print results."""
    from seed_storage.query.search import search

    results = await search(query=query, num_results=limit)

    if output_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if not results:
            print("No results found.")
            return

        print(f"Found {len(results)} result(s) for: {query!r}\n")
        for i, r in enumerate(results, 1):
            node = r.get("node", {})
            score = r.get("score", 0)
            name = node.get("name", node.get("statement", "unknown"))
            entity_type = node.get("entity_type", "")
            description = node.get("description", "")
            print(f"[{i}] {name}" + (f" ({entity_type})" if entity_type else ""))
            if description:
                print(f"    {description[:100]}")
            print(f"    Score: {score:.3f}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the AntHaul knowledge graph")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--limit", "-n", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="output_json")

    args = parser.parse_args()
    if not args.query.strip():
        print("Error: query cannot be empty", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run_query(args.query, args.limit, args.output_json))


if __name__ == "__main__":
    main()
