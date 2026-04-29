"""Staging table for two-phase ingestion — extract first, vectorize later."""

from __future__ import annotations

import psycopg2
import psycopg2.extras

from . import config

# Use the existing task_manager PostgreSQL
PG_DSN = config.PG_DSN


def _connect():
    return psycopg2.connect(PG_DSN)


def init_tables():
    """Create staging and cost tables. Idempotent."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS seed_staging (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_type TEXT NOT NULL,
                    source_uri TEXT UNIQUE NOT NULL,
                    raw_content TEXT,
                    media_urls TEXT[],
                    word_count INT DEFAULT 0,
                    token_estimate INT DEFAULT 0,
                    author TEXT,
                    channel TEXT,
                    created_at TIMESTAMPTZ,
                    staged_at TIMESTAMPTZ DEFAULT NOW(),
                    status TEXT DEFAULT 'staged',
                    batch_id UUID,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_staging_status ON seed_staging(status);
                CREATE INDEX IF NOT EXISTS idx_staging_batch ON seed_staging(batch_id);

                CREATE TABLE IF NOT EXISTS seed_costs (
                    id SERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ DEFAULT NOW(),
                    operation TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INT DEFAULT 0,
                    output_tokens INT DEFAULT 0,
                    cost_usd NUMERIC(10,6) NOT NULL,
                    source_id UUID,
                    batch_id UUID
                );
                CREATE INDEX IF NOT EXISTS idx_costs_batch ON seed_costs(batch_id);
            """)
        conn.commit()


def stage(
    *,
    source_type: str,
    source_uri: str,
    raw_content: str,
    author: str = "",
    channel: str = "",
    created_at: str | None = None,
    media_urls: list[str] | None = None,
    metadata: dict | None = None,
    status: str = "staged",
) -> str | None:
    """Stage a piece of content. Returns the id, or None if already staged (dedup by URI)."""
    word_count = len(raw_content.split())
    token_estimate = int(word_count * 1.33)
    meta = psycopg2.extras.Json(metadata or {})

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO seed_staging (source_type, source_uri, raw_content, media_urls,
                    word_count, token_estimate, author, channel, created_at, metadata, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_uri) DO NOTHING
                RETURNING id
                """,
                (
                    source_type,
                    source_uri,
                    raw_content,
                    media_urls or [],
                    word_count,
                    token_estimate,
                    author,
                    channel,
                    created_at,
                    meta,
                    status,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row[0]) if row else None


def get_staged(status: str = "staged", limit: int = 1000) -> list[dict]:
    """Get staged items by status."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM seed_staging WHERE status = %s ORDER BY staged_at LIMIT %s",
                (status, limit),
            )
            return [dict(r) for r in cur.fetchall()]


def get_by_uri(source_uri: str) -> dict | None:
    """Get a single staging item by source_uri (unique index)."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM seed_staging WHERE source_uri = %s", (source_uri,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_by_id(item_id: str) -> dict | None:
    """Get a single staging item by UUID."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM seed_staging WHERE id = %s::uuid", (item_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def count_by_status() -> dict[str, int]:
    """Count staged items grouped by status."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, count(*) FROM seed_staging GROUP BY status")
            return dict(cur.fetchall())


def summary() -> dict:
    """Summary stats for all staged content."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    source_type,
                    count(*) AS items,
                    sum(word_count) AS total_words,
                    sum(token_estimate) AS total_tokens
                FROM seed_staging
                WHERE status = 'staged'
                GROUP BY source_type
                ORDER BY total_tokens DESC
            """)
            rows = [dict(r) for r in cur.fetchall()]
            total_tokens = sum(r["total_tokens"] or 0 for r in rows)
            return {
                "by_type": rows,
                "total_tokens": total_tokens,
                "total_items": sum(r["items"] for r in rows),
            }


def get_recently_loaded(hours: int = 24) -> list[dict]:
    """Get items loaded in the last N hours."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM seed_staging
                   WHERE status = 'loaded'
                   AND staged_at >= NOW() - INTERVAL '%s hours'
                   ORDER BY staged_at DESC""",
                (hours,),
            )
            return [dict(r) for r in cur.fetchall()]


def update_content(
    item_id: str, raw_content: str, metadata: dict | None = None, status: str = "processed"
):
    """Update a staged item with processed content (Step 2)."""
    word_count = len(raw_content.split())
    token_estimate = int(word_count * 1.33)
    meta = psycopg2.extras.Json(metadata or {})

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE seed_staging
                   SET raw_content = %s, word_count = %s, token_estimate = %s,
                       metadata = %s, status = %s
                   WHERE id = %s""",
                (raw_content, word_count, token_estimate, meta, status, item_id),
            )
        conn.commit()


def update_status(ids: list[str], status: str, batch_id: str | None = None):
    """Update status for a list of staged item IDs."""
    with _connect() as conn:
        with conn.cursor() as cur:
            if batch_id:
                cur.execute(
                    "UPDATE seed_staging SET status = %s, batch_id = %s WHERE id = ANY(%s::uuid[])",
                    (status, batch_id, ids),
                )
            else:
                cur.execute(
                    "UPDATE seed_staging SET status = %s WHERE id = ANY(%s::uuid[])",
                    (status, ids),
                )
        conn.commit()


def patch_metadata(item_id: str, patch: dict):
    """Merge keys into an item's metadata without changing status or content.

    Uses jsonb concatenation (||) — existing keys are preserved unless overwritten.
    Idempotent: running twice with the same patch is safe.
    """
    import psycopg2.extras

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE seed_staging SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb WHERE id = %s::uuid",
                (psycopg2.extras.Json(patch), item_id),
            )
        conn.commit()


# ── Circuit Breaker ──────────────────────────────────────────────────


def init_circuit_breaker_table():
    """Create the circuit breaker table. Idempotent."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS seed_circuit_breaker (
                    id SERIAL PRIMARY KEY,
                    tripped_at TIMESTAMPTZ DEFAULT NOW(),
                    reason TEXT NOT NULL,
                    reset_after TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ
                );
            """)
        conn.commit()


def trip_breaker(reason: str, cooldown_hours: int | None = None):
    """Trip the circuit breaker. cooldown_hours=None means manual reset only."""
    init_circuit_breaker_table()
    with _connect() as conn:
        with conn.cursor() as cur:
            reset_after = None
            if cooldown_hours:
                cur.execute("SELECT NOW() + INTERVAL '%s hours'", (cooldown_hours,))
                reset_after = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO seed_circuit_breaker (reason, reset_after)
                   VALUES (%s, %s)""",
                (reason, reset_after),
            )
        conn.commit()


def is_breaker_tripped() -> dict | None:
    """Check if circuit breaker is active. Returns the active breaker row or None."""
    init_circuit_breaker_table()
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM seed_circuit_breaker
                WHERE resolved_at IS NULL
                  AND (reset_after IS NULL OR reset_after > NOW())
                ORDER BY tripped_at DESC LIMIT 1
            """)
            row = cur.fetchone()
            return dict(row) if row else None


def reset_breaker():
    """Resolve all active circuit breakers."""
    init_circuit_breaker_table()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE seed_circuit_breaker SET resolved_at = NOW() WHERE resolved_at IS NULL"
            )
        conn.commit()


def reset_orphaned_loading(timeout_hours: int = 1) -> int:
    """Reset items stuck in 'loading' status back to 'enriched'. Returns count."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE seed_staging SET status = 'enriched'
                   WHERE status = 'loading'
                   AND staged_at < NOW() - INTERVAL '%s hours'
                   RETURNING id""",
                (timeout_hours,),
            )
            count = cur.rowcount
        conn.commit()
    return count


def reset_orphaned_extracting(timeout_hours: int = 1) -> int:
    """Reset items stuck in 'extracting' status back to 'enriched'. Returns count."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE seed_staging SET status = 'enriched'
                   WHERE status = 'extracting'
                   AND staged_at < NOW() - INTERVAL '%s hours'
                   RETURNING id""",
                (timeout_hours,),
            )
            count = cur.rowcount
        conn.commit()
    return count


def get_by_discord_msg_id(msg_id: str) -> dict | None:
    """Return the staging row for a Discord message ID, or None if not staged."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id FROM seed_staging WHERE metadata->>'discord_msg_id' = %s LIMIT 1",
                (msg_id,)
            )
            return cur.fetchone()


def upsert_bot_last_seen(timestamp: str) -> None:
    """Record the bot's last successful Discord connection time.

    Creates the seed_bot_state table if it doesn't exist.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS seed_bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                INSERT INTO seed_bot_state (key, value, updated_at)
                VALUES ('bot_last_seen', %s, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """, (timestamp,))
        conn.commit()


def get_bot_last_seen() -> str | None:
    """Return the ISO timestamp of the bot's last connection, or None if never recorded."""
    with _connect() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT value FROM seed_bot_state WHERE key = 'bot_last_seen'"
                )
                row = cur.fetchone()
                return row[0] if row else None
            except Exception:
                return None


def reset_to_status(
    target_status: str,
    *,
    source_statuses: list[str] | None = None,
    batch_id: str | None = None,
    limit: int | None = None,
) -> int:
    """Reset items from downstream statuses back to a target for replay.

    Args:
        target_status: Status to reset TO (e.g. 'enriched').
        source_statuses: Which statuses to reset FROM. If None, uses all
            statuses downstream of target_status.
        batch_id: Optional -- only reset items in this batch.
        limit: Optional -- cap on how many items to reset.

    Returns count of items reset.
    """
    DOWNSTREAM = {
        "staged": ["processed", "enriched", "extracted", "loaded"],
        "processed": ["enriched", "extracted", "loaded"],
        "enriched": ["extracted", "loaded"],
        "extracted": ["loaded"],
    }

    if source_statuses is None:
        source_statuses = DOWNSTREAM.get(target_status)
        if source_statuses is None:
            raise ValueError(f"No downstream statuses for '{target_status}'")

    with _connect() as conn:
        with conn.cursor() as cur:
            if limit:
                batch_filter = "AND batch_id = %s::uuid" if batch_id else ""
                query = f"""
                    UPDATE seed_staging SET status = %s
                    WHERE id IN (
                        SELECT id FROM seed_staging
                        WHERE status = ANY(%s)
                        {batch_filter}
                        ORDER BY staged_at LIMIT %s
                    )
                """
                params: list = [target_status, source_statuses]
                if batch_id:
                    params.append(batch_id)
                params.append(limit)
            else:
                query = "UPDATE seed_staging SET status = %s WHERE status = ANY(%s)"
                params = [target_status, source_statuses]
                if batch_id:
                    query += " AND batch_id = %s::uuid"
                    params.append(batch_id)
            cur.execute(query, params)
            count = cur.rowcount
        conn.commit()
    return count
