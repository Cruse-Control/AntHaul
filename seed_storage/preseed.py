"""Preseed entity table -- extensible alias/entity configuration in Postgres."""
from __future__ import annotations

import psycopg2
import psycopg2.extras

from . import config

PG_DSN = config.PG_DSN


def _connect():
    return psycopg2.connect(PG_DSN)


def init_preseed_table():
    """Create the preseed entities table. Idempotent."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS seed_preseed_entities (
                    id SERIAL PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases TEXT[] NOT NULL DEFAULT '{}',
                    description TEXT DEFAULT '',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_preseed_canonical
                    ON seed_preseed_entities(canonical_name);
            """)
        conn.commit()


def seed_defaults():
    """Insert default preseed entities. Idempotent (ON CONFLICT DO NOTHING)."""
    defaults = [
        ("flynn cruse", "Person", ["siliconwarlock", "flynnbo", "flynn-cruse", "flynn"],
         "Co-founder of CruseControl"),
        ("wyler zahm", "Person", ["famed_esteemed", "wylerza", "wyler-zahm", "wyler"],
         "Co-founder of CruseControl"),
        ("crusecontrol", "Organization", ["cruse-control", "cruse control", "cc"],
         "AI agent consulting startup"),
    ]
    with _connect() as conn:
        with conn.cursor() as cur:
            for canonical, etype, aliases, desc in defaults:
                cur.execute(
                    """INSERT INTO seed_preseed_entities (canonical_name, entity_type, aliases, description)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (canonical_name) DO NOTHING""",
                    (canonical, etype, aliases, desc),
                )
        conn.commit()


def get_alias_map() -> dict[str, str]:
    """Return {alias_lowercase: canonical_name} for all preseed entities.

    Used by coreference pre-processing and Tier 1 resolution.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT canonical_name, aliases FROM seed_preseed_entities")
            alias_map: dict[str, str] = {}
            for canonical, aliases in cur.fetchall():
                alias_map[canonical] = canonical
                for alias in (aliases or []):
                    alias_map[alias.lower()] = canonical
            return alias_map


def get_all() -> list[dict]:
    """Return all preseed entities."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM seed_preseed_entities ORDER BY canonical_name")
            return [dict(r) for r in cur.fetchall()]


def add_entity(canonical_name: str, entity_type: str, aliases: list[str] = None,
               description: str = "") -> bool:
    """Add a preseed entity. Returns True if inserted, False if already exists."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO seed_preseed_entities (canonical_name, entity_type, aliases, description)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (canonical_name) DO NOTHING
                   RETURNING id""",
                (canonical_name.lower(), entity_type, aliases or [], description),
            )
            inserted = cur.fetchone() is not None
        conn.commit()
    return inserted


def add_alias(canonical_name: str, alias: str) -> bool:
    """Add an alias to an existing preseed entity. Returns True if updated."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE seed_preseed_entities
                   SET aliases = array_append(aliases, %s)
                   WHERE canonical_name = %s
                   AND NOT (%s = ANY(aliases))
                   RETURNING id""",
                (alias.lower(), canonical_name.lower(), alias.lower()),
            )
            updated = cur.fetchone() is not None
        conn.commit()
    return updated
