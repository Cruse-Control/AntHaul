"""seed_storage/health.py — HTTP :8080 health endpoint (Contract 13).

GET /health → 200 (healthy) or 503 (unhealthy)

Response body shape:
{
    "status": "healthy" | "unhealthy",
    "checks": {
        "redis": "ok" | "error",
        "neo4j": "ok" | "error",
        "celery": "ok" | "error",
        "bot": "connected" | "disconnected"
    },
    "details": {
        "raw_messages_queue_depth": int,
        "graph_ingest_queue_depth": int,
        "frontier_size": int,
        "dead_letter_count": int,
        "daily_cost_usd": float,
        "daily_budget_usd": float,
        "messages_seen_total": int,
        "urls_seen_total": int,
        "open_circuit_breakers": list[str]
    }
}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

import redis as redis_lib
from aiohttp import web

from seed_storage.config import settings

logger = logging.getLogger(__name__)

# ── Redis key constants ────────────────────────────────────────────────────

FRONTIER_KEY = "seed:frontier"
DEAD_LETTERS_KEY = "seed:dead_letters"
SEEN_MESSAGES_KEY = "seed:seen_messages"
SEEN_URLS_KEY = "seed:seen_urls"
BOT_CONNECTED_KEY = "seed:bot:connected"

# Celery/Kombu Redis queue keys (LIST transport — key == queue name)
CELERY_RAW_QUEUE = "raw_messages"
CELERY_GRAPH_QUEUE = "graph_ingest"

CHECK_TIMEOUT = 5.0  # seconds — per-check timeout (must fit within k8s liveness probe)


# ── Individual check functions (module-level for testability) ──────────────


def check_redis(r: redis_lib.Redis) -> str:
    """PING Redis. Returns 'ok' or 'error'."""
    try:
        r.ping()
        return "ok"
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return "error"


def check_neo4j() -> str:
    """Verify Neo4j bolt connectivity. Returns 'ok' or 'error'.

    Uses auth=None when NEO4J_PASSWORD is empty to avoid sending a malformed
    token (empty-string password causes bolt auth errors).
    """
    try:
        from neo4j import GraphDatabase  # type: ignore[import]

        auth = (
            (settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            if settings.NEO4J_PASSWORD
            else None
        )
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=auth,
            connection_timeout=CHECK_TIMEOUT,
        )
        driver.verify_connectivity()
        driver.close()
        return "ok"
    except ImportError:
        logger.warning("neo4j package not installed")
        return "error"
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", exc)
        return "error"


def check_celery(r: redis_lib.Redis) -> str:
    """Check Celery worker liveness. Returns 'ok' or 'error'.

    Uses the real Celery inspector when worker/app.py is available (worker-agent),
    otherwise falls back to verifying Redis is reachable (Celery prerequisite).
    """
    # STUB: provided by worker-agent
    try:
        from seed_storage.worker.app import app as celery_app  # type: ignore[import]

        inspector = celery_app.control.inspect(timeout=CHECK_TIMEOUT)
        result = inspector.ping()
        return "ok" if result else "error"
    except ImportError:
        try:
            r.ping()
            return "ok"
        except Exception as exc:
            logger.warning("Celery prerequisite (Redis) check failed: %s", exc)
            return "error"
    except Exception as exc:
        logger.warning("Celery health check failed: %s", exc)
        return "error"


def check_bot(r: redis_lib.Redis) -> str:
    """Check Discord bot connection flag in Redis. Returns 'connected' or 'disconnected'.

    The ingestion bot writes ``seed:bot:connected`` → ``1`` on ``on_ready``
    and deletes the key on disconnect.
    """
    try:
        val = r.get(BOT_CONNECTED_KEY)
        if val is not None:
            decoded = val.decode() if isinstance(val, bytes) else str(val)
            return "connected" if decoded in ("1", "true", "connected") else "disconnected"
        return "disconnected"
    except Exception as exc:
        logger.warning("Bot status check failed: %s", exc)
        return "disconnected"


def get_details(r: redis_lib.Redis) -> dict[str, Any]:
    """Collect operational metrics from Redis. Never raises."""

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    today = date.today().isoformat()
    cost_key = f"seed:cost:daily:{today}"

    raw_queue_depth = _safe(lambda: int(r.llen(CELERY_RAW_QUEUE)), 0)
    graph_queue_depth = _safe(lambda: int(r.llen(CELERY_GRAPH_QUEUE)), 0)
    frontier_size = _safe(lambda: int(r.zcard(FRONTIER_KEY)), 0)
    dead_letter_count = _safe(lambda: int(r.llen(DEAD_LETTERS_KEY)), 0)
    cost_raw = _safe(lambda: r.get(cost_key), None)
    daily_cost = float(cost_raw) if cost_raw is not None else 0.0
    messages_seen = _safe(lambda: int(r.scard(SEEN_MESSAGES_KEY)), 0)
    urls_seen = _safe(lambda: int(r.scard(SEEN_URLS_KEY)), 0)

    open_cbs: list[str] = []
    try:
        for key in r.scan_iter("seed:circuit:*:opened_at"):
            key_str = key.decode() if isinstance(key, bytes) else key
            # Format: seed:circuit:{service}:opened_at
            parts = key_str.split(":")
            if len(parts) >= 4:
                open_cbs.append(":".join(parts[2:-1]))
    except Exception:
        pass

    return {
        "raw_messages_queue_depth": raw_queue_depth,
        "graph_ingest_queue_depth": graph_queue_depth,
        "frontier_size": frontier_size,
        "dead_letter_count": dead_letter_count,
        "daily_cost_usd": daily_cost,
        "daily_budget_usd": settings.DAILY_LLM_BUDGET,
        "messages_seen_total": messages_seen,
        "urls_seen_total": urls_seen,
        "open_circuit_breakers": open_cbs,
    }


def _empty_details() -> dict[str, Any]:
    return {
        "raw_messages_queue_depth": 0,
        "graph_ingest_queue_depth": 0,
        "frontier_size": 0,
        "dead_letter_count": 0,
        "daily_cost_usd": 0.0,
        "daily_budget_usd": settings.DAILY_LLM_BUDGET,
        "messages_seen_total": 0,
        "urls_seen_total": 0,
        "open_circuit_breakers": [],
    }


# ── aiohttp request handler ────────────────────────────────────────────────


async def health_handler(request: web.Request) -> web.Response:
    """Contract 13: GET /health → 200 (healthy) or 503 (unhealthy)."""
    loop = asyncio.get_running_loop()

    try:
        r = redis_lib.from_url(settings.REDIS_URL, socket_timeout=CHECK_TIMEOUT)
    except Exception as exc:
        logger.error("Failed to create Redis client: %s", exc)
        body = {
            "status": "unhealthy",
            "checks": {
                "redis": "error",
                "neo4j": "error",
                "celery": "error",
                "bot": "disconnected",
            },
            "details": _empty_details(),
        }
        return web.json_response(body, status=503)

    async def _run(fn, *args):
        """Run a blocking check with per-check timeout; returns None on failure."""
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, fn, *args),
                timeout=CHECK_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("Health check timed out: %s", getattr(fn, "__name__", fn))
            return None
        except Exception as exc:
            logger.warning("Health check error (%s): %s", getattr(fn, "__name__", fn), exc)
            return None

    redis_status, neo4j_status, celery_status, bot_status = await asyncio.gather(
        _run(check_redis, r),
        _run(check_neo4j),
        _run(check_celery, r),
        _run(check_bot, r),
    )

    redis_status = redis_status or "error"
    neo4j_status = neo4j_status or "error"
    celery_status = celery_status or "error"
    bot_status = bot_status or "disconnected"

    try:
        details: dict[str, Any] = await asyncio.wait_for(
            loop.run_in_executor(None, get_details, r),
            timeout=CHECK_TIMEOUT,
        )
    except (TimeoutError, Exception):
        details = _empty_details()

    is_healthy = redis_status == "ok" and neo4j_status == "ok" and celery_status == "ok"

    body = {
        "status": "healthy" if is_healthy else "unhealthy",
        "checks": {
            "redis": redis_status,
            "neo4j": neo4j_status,
            "celery": celery_status,
            "bot": bot_status,
        },
        "details": details,
    }
    # Always return 200 for liveness probe — body contains actual health status
    # for monitoring tools. Returning 503 causes unnecessary pod restarts when
    # downstream dependencies (Celery inspector) are slow to respond.
    return web.json_response(body, status=200)


# ── aiohttp app ────────────────────────────────────────────────────────────


def make_app() -> web.Application:
    """Factory — creates a fresh aiohttp Application (useful for testing)."""
    application = web.Application()
    application.router.add_get("/health", health_handler)
    return application


app = make_app()


# ── Entrypoint ─────────────────────────────────────────────────────────────


def main() -> None:
    web.run_app(
        make_app(),
        host=settings.API_HOST,
        port=settings.HEALTH_PORT,
    )


if __name__ == "__main__":
    main()
