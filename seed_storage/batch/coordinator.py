"""Batch job coordinator -- lifecycle management for large batch operations.

Tracks batch jobs via Postgres staging batch_id, manages progress, handles
pause/resume/cancel.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from seed_storage import staging

log = logging.getLogger("batch.coordinator")


def create_batch(
    *,
    from_status: str,
    target_status: str,
    limit: int | None = None,
    use_batch_api: bool = False,
) -> dict:
    """Create a new batch job. Returns batch metadata including batch_id.

    Claims items by setting their batch_id and transitioning to transient status.
    """
    batch_id = str(uuid.uuid4())
    items = staging.get_staged(status=from_status, limit=limit or 5000)
    if not items:
        return {"batch_id": batch_id, "item_count": 0, "status": "empty"}

    item_ids = [str(i["id"]) for i in items]

    # Map from_status to transient status
    transient_map = {"enriched": "extracting", "extracted": "loading"}
    transient = transient_map.get(from_status, from_status)

    staging.update_status(item_ids, transient, batch_id)

    return {
        "batch_id": batch_id,
        "item_count": len(items),
        "from_status": from_status,
        "target_status": target_status,
        "use_batch_api": use_batch_api,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "created",
    }


def get_batch_progress(batch_id: str) -> dict:
    """Get progress for a batch job."""
    import psycopg2.extras
    from seed_storage.staging import _connect

    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT status, count(*) AS cnt
                   FROM seed_staging WHERE batch_id = %s::uuid
                   GROUP BY status""",
                (batch_id,),
            )
            counts = {r["status"]: r["cnt"] for r in cur.fetchall()}
    total = sum(counts.values())
    return {"batch_id": batch_id, "total": total, "by_status": counts}


def cancel_batch(batch_id: str, reset_to: str = "enriched") -> int:
    """Cancel a batch -- reset all non-terminal items back to a replayable status."""
    return staging.reset_to_status(
        reset_to,
        source_statuses=["extracting", "loading"],
        batch_id=batch_id,
    )
