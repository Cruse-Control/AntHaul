"""Batch CLI -- replay pipeline stages from Postgres staging.

Usage:
    python -m seed_storage.batch status
    python -m seed_storage.batch run --from enriched --limit 50
    python -m seed_storage.batch run --from enriched --all --dry-run
    python -m seed_storage.batch run --from enriched --batch-api
    python -m seed_storage.batch poll <batch-id>
    python -m seed_storage.batch reset --to enriched
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from seed_storage import staging
from seed_storage.models import PIPELINE_STATUSES

log = logging.getLogger("batch")


def cmd_status(args):
    """Show pipeline status counts."""
    counts = staging.count_by_status()
    total = sum(counts.values())
    print(f"Pipeline status ({total} total):")
    for status in PIPELINE_STATUSES + ["failed", "rejected", "deduped", "loading", "extracting"]:
        if status in counts:
            print(f"  {status:>12}: {counts[status]}")


def cmd_reset(args):
    """Reset items to a target status for replay."""
    count = staging.reset_to_status(
        args.to,
        batch_id=args.batch_id,
        limit=args.limit if not args.all else None,
    )
    print(f"Reset {count} items to '{args.to}'")


def cmd_run(args):
    """Run pipeline from a given status."""
    target = args.from_status
    limit = None if args.all else (args.limit or 200)

    # Batch API path for extraction
    if getattr(args, "batch_api", False) and target == "enriched":
        from pathlib import Path
        import tempfile
        from seed_storage.batch.batch_api import build_extraction_jsonl, submit_batch
        from seed_storage.preseed import init_preseed_table

        init_preseed_table()
        items = staging.get_staged(status=target, limit=limit or 5000)
        if not items:
            print(f"No items at status '{target}'")
            return
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            jsonl_path = Path(f.name)
        count = build_extraction_jsonl(items, jsonl_path)
        batch_id = submit_batch(jsonl_path)
        print(f"Batch submitted: {batch_id} ({count} requests)")
        print(f"JSONL: {jsonl_path}")
        print(f"Poll with: python -m seed_storage.batch poll {batch_id}")
        return

    items = staging.get_staged(status=target, limit=limit or 1000)
    if not items:
        print(f"No items at status '{target}'")
        return
    if args.dry_run:
        print(f"Would process {len(items)} items from '{target}':")
        for item in items[:10]:
            print(f"  [{item['source_type']}] {item['source_uri'][:60]}")
        if len(items) > 10:
            print(f"  ...and {len(items) - 10} more")
        return
    # Dispatch to appropriate worker based on source status
    if target == "staged":
        from ingestion.processor import process_batch
        asyncio.run(process_batch(limit=limit))
    elif target == "processed":
        from ingestion.enricher import enrich_batch
        asyncio.run(enrich_batch(limit=limit))
    elif target == "enriched":
        from seed_storage.extraction import extract_batch
        asyncio.run(extract_batch(limit=limit))
    elif target == "extracted":
        from ingestion.loader import load_batch
        asyncio.run(load_batch(limit=limit))
    else:
        print(f"Cannot run from status '{target}'")
        sys.exit(1)


def cmd_poll(args):
    """Poll an OpenAI Batch API job and apply results."""
    from seed_storage.batch.batch_api import poll_batch, download_results, apply_batch_results

    print(f"Polling batch {args.batch_id}...")
    result = poll_batch(args.batch_id, poll_interval=args.interval, max_polls=args.max_polls)
    print(f"Batch status: {result['status']}")

    if result["status"] == "completed" and result.get("output_file_id"):
        print("Downloading results...")
        results = download_results(result["output_file_id"])
        print(f"Downloaded {len(results)} results")
        if not args.dry_run:
            counts = apply_batch_results(results)
            print(f"Applied: {counts}")
        else:
            print("Dry run -- results not applied")
    elif result["status"] == "failed":
        print("Batch failed.")
        if result.get("error_file_id"):
            print(f"Error file: {result['error_file_id']}")
    elif result["status"] == "timeout":
        print("Polling timed out. Re-run to continue polling.")


def cmd_progress(args):
    """Show progress for a batch job."""
    from seed_storage.batch.coordinator import get_batch_progress
    progress = get_batch_progress(args.batch_id)
    print(f"Batch {args.batch_id}: {progress['total']} items")
    for status, count in progress["by_status"].items():
        print(f"  {status:>12}: {count}")


def main():
    parser = argparse.ArgumentParser(prog="python -m seed_storage.batch")
    sub = parser.add_subparsers(dest="command")

    # run pipeline
    run_p = sub.add_parser("run", help="Run pipeline from a status")
    run_p.add_argument("--from", dest="from_status", required=True,
                       choices=PIPELINE_STATUSES[:-1])
    run_p.add_argument("--limit", type=int, default=200)
    run_p.add_argument("--all", action="store_true")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--batch-api", action="store_true",
                       help="Use OpenAI Batch API (50%% discount, 24hr SLA)")

    # Reset
    reset_p = sub.add_parser("reset", help="Reset items for replay")
    reset_p.add_argument("--to", required=True, choices=PIPELINE_STATUSES[:-1])
    reset_p.add_argument("--batch-id", default=None)
    reset_p.add_argument("--limit", type=int, default=None)
    reset_p.add_argument("--all", action="store_true")

    # Status
    sub.add_parser("status", help="Show pipeline counts")

    # Poll batch API job
    poll_p = sub.add_parser("poll", help="Poll OpenAI Batch API job")
    poll_p.add_argument("batch_id", help="OpenAI batch ID")
    poll_p.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    poll_p.add_argument("--max-polls", type=int, default=1440, help="Max poll attempts")
    poll_p.add_argument("--dry-run", action="store_true",
                        help="Download results but don't apply")

    # Progress for a batch
    progress_p = sub.add_parser("progress", help="Show batch progress")
    progress_p.add_argument("batch_id", help="Batch UUID")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "reset":
        cmd_reset(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "poll":
        cmd_poll(args)
    elif args.command == "progress":
        cmd_progress(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
