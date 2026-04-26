"""Journey tests for issues #3, #10, #11, #12, #13, #14.

Tests the full pipeline end-to-end against live Postgres and Neo4j.
Each test creates fresh staging items, runs pipeline steps, and verifies results.

Usage:
    python tests/journey_test.py [--issue N] [--all]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

log = logging.getLogger("journey_test")

PASS = 0
FAIL = 0


def _assert(condition: bool, msg: str):
    global PASS, FAIL
    if condition:
        PASS += 1
        log.info("  PASS: %s", msg)
    else:
        FAIL += 1
        log.error("  FAIL: %s", msg)


# ── #3: X/Twitter resolver via FxTwitter API ────────────────���─────────

async def test_issue_3():
    """X/Twitter URL -> full tweet content extracted via FxTwitter API."""
    log.info("=== Issue #3: X/Twitter resolver ===")
    import httpx
    from ingestion.processor import _process_x

    # Use a real tweet from staging
    test_url = "https://x.com/nesquena/status/2040830449413763155"

    async with httpx.AsyncClient(timeout=30) as http:
        content, meta = await _process_x(http, test_url)

    _assert(len(content) > 50, f"Content length {len(content)} > 50")
    _assert("Tweet by" in content, "Content starts with 'Tweet by'")
    _assert("nesquena" in content.lower() or "Nathan" in content, "Author name present")
    _assert(meta.get("tweet_id") == "2040830449413763155", f"tweet_id={meta.get('tweet_id')}")
    _assert(len(meta.get("speakers", [])) > 0, "Has speakers metadata")
    _assert(meta.get("published_at"), f"Has published_at: {meta.get('published_at')}")

    # Test with a second tweet to verify robustness
    test_url2 = "https://x.com/RampLabs/status/2036165188899012655"
    async with httpx.AsyncClient(timeout=30) as http:
        content2, meta2 = await _process_x(http, test_url2)
    _assert(len(content2) > 20, f"Second tweet content length {len(content2)} > 20")

    # Test full pipeline: stage -> process -> verify content updated
    from seed_storage import staging
    staging.init_tables()

    # source_uri must be the real URL (processor passes it to _process_x)
    # Use a unique URL by appending a timestamp query param
    pipeline_url = f"https://x.com/nesquena/status/2040830449413763155?_jt={int(time.time())}"
    sid = staging.stage(
        source_type="x_twitter",
        source_uri=pipeline_url,
        raw_content=pipeline_url,
        author="journey_test",
        channel="test",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _assert(sid is not None, f"Staged test item: {sid}")

    if sid:
        item = staging.get_by_id(sid)
        from ingestion.processor import process_one

        async with httpx.AsyncClient(timeout=60) as http:
            await process_one(item, http)

        item = staging.get_by_id(sid)
        _assert(item["status"] == "processed", f"Status after process: {item['status']}")
        _assert("Tweet by" in (item["raw_content"] or ""), "Processed content has tweet text")

        # Clean up
        staging.update_status([sid], "deleted")


# ── #10: Enrichment metadata in loader ────────────────────────────────

async def test_issue_10():
    """Enriched metadata (tags, summary, speakers) included in episode body."""
    log.info("=== Issue #10: Enrichment metadata ===")
    from ingestion.loader import _build_enriched_content

    # Test the enrichment builder
    content = "This is an article about machine learning."
    metadata = {
        "tags": ["machine-learning", "ai-research"],
        "summary": "Overview of ML techniques for knowledge graphs",
        "speakers": [
            {"name": "Dr. Smith", "role": "author"},
            {"name": "Flynn Cruse", "role": "curator"},
        ],
    }

    enriched = _build_enriched_content(content, metadata)
    _assert("Tags: machine-learning, ai-research" in enriched, "Tags in enriched content")
    _assert("Summary: Overview" in enriched, "Summary in enriched content")
    _assert("Dr. Smith (author)" in enriched, "Speaker in enriched content")
    _assert("Flynn Cruse (curator)" in enriched, "Curator in enriched content")
    _assert(content in enriched, "Original content preserved")

    # Test with empty metadata
    plain = _build_enriched_content(content, {})
    _assert(plain == content, "Empty metadata returns original content")

    # Test with uncategorized tags (should NOT be prepended)
    uncategorized = _build_enriched_content(content, {"tags": ["uncategorized"]})
    _assert("Tags:" not in uncategorized, "Uncategorized tags excluded from header")

    # Test with discord_context (#4 source tagging)
    ctx_meta = {"discord_context": "Check out this great ML overview"}
    with_ctx = _build_enriched_content(content, ctx_meta)
    _assert("Shared with context: Check out this great ML overview" in with_ctx,
            "Discord context in enriched content")


# ── #11: Instagram image resolver ─────────────────────────────────────

async def test_issue_11():
    """Instagram image URL -> caption/author extracted via instaloader."""
    log.info("=== Issue #11: Instagram image resolver ===")
    import httpx
    from ingestion.processor import _process_instagram_image, _extract_instagram_shortcode

    # Test shortcode extraction
    _assert(_extract_instagram_shortcode("https://www.instagram.com/p/DWlZ6EIjw7P/") == "DWlZ6EIjw7P",
            "Shortcode extraction from /p/ URL")
    _assert(_extract_instagram_shortcode("https://www.instagram.com/reel/ABC123/") == "ABC123",
            "Shortcode extraction from /reel/ URL")

    # Test with a real Instagram post
    test_url = "https://www.instagram.com/p/DWlZ6EIjw7P/"
    async with httpx.AsyncClient(timeout=60) as http:
        content, meta = await _process_instagram_image(http, test_url)

    _assert(len(content) > 30, f"Content length {len(content)} > 30")
    _assert("instagram" in content.lower() or "post" in content.lower(), "Content references Instagram")
    if "okaashish" in content.lower() or meta.get("author"):
        _assert(True, f"Author extracted: {meta.get('author', 'in content')}")
    else:
        _assert(len(content) > 50, "Content fallback still meaningful")

    # Test full pipeline: stage -> process -> verify
    from seed_storage import staging
    staging.init_tables()

    # source_uri must be the real URL (processor passes it to the resolver)
    pipeline_url = f"https://www.instagram.com/p/DWlZ6EIjw7P/?_jt={int(time.time())}"
    sid = staging.stage(
        source_type="instagram_image",
        source_uri=pipeline_url,
        raw_content=pipeline_url,
        author="journey_test",
        channel="test",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _assert(sid is not None, f"Staged test item: {sid}")

    if sid:
        item = staging.get_by_id(sid)
        from ingestion.processor import process_one

        async with httpx.AsyncClient(timeout=120) as http:
            await process_one(item, http)

        item = staging.get_by_id(sid)
        _assert(item["status"] == "processed", f"Status after process: {item['status']}")
        _assert(len(item["raw_content"] or "") > 30, "Processed content is non-trivial")

        staging.update_status([sid], "deleted")


# ── #12: Content quality gate ─────────────────────────────────────────

async def test_issue_12():
    """Quality gate rejects garbage content, passes good content."""
    log.info("=== Issue #12: Content quality gate ===")
    from ingestion.loader import _is_loadable

    # Good content should pass
    ok, reason = _is_loadable(
        "This is a detailed article about machine learning and neural networks with lots of useful information",
        "web",
    )
    _assert(ok and reason == "ok", f"Good content passes: {reason}")

    # Auth wall should be rejected
    ok, reason = _is_loadable("Please sign in to continue to view this page content and data", "web")
    _assert(not ok and "auth_wall" in reason, f"Auth wall rejected: {reason}")

    # Short content rejected
    ok, reason = _is_loadable("hi", "web")
    _assert(not ok and reason == "content_too_short", f"Short content rejected: {reason}")

    # Tweet stub rejected
    ok, reason = _is_loadable("[Tweet by @someone] https://x.com/someone/status/123", "x_twitter")
    _assert(not ok and reason == "tweet_stub", f"Tweet stub rejected: {reason}")

    # Cookie wall rejected
    ok, reason = _is_loadable("cookie cookie cookie cookie consent", "web")
    _assert(not ok and reason == "cookie_wall", f"Cookie wall rejected: {reason}")

    # Empty/stub bracket content rejected
    ok, reason = _is_loadable("[Instagram image post]", "instagram_image")
    _assert(not ok and reason == "stub_content", f"Stub bracket rejected: {reason}")

    # Real X/Twitter content from old pipeline (all "Something went wrong")
    ok, reason = _is_loadable(
        "Something went wrong, but don't fret — let's give it another shot.\n"
        "Try again\nSome privacy related extensions may cause issues on x.com.",
        "x_twitter",
    )
    _assert(not ok, f"Old failed X/Twitter content rejected: {reason}")

    # Long content with login mention should still pass
    long_content = "This article discusses how to sign in to machine learning platforms. " * 20
    ok, reason = _is_loadable(long_content, "web")
    _assert(ok and reason == "ok", f"Long content with login mention passes: {reason}")


# ── #13: Threaded items pipeline flow ─────────────────────────────────

async def test_issue_13():
    """Threaded conversation items flow through processor -> enricher."""
    log.info("=== Issue #13: Threaded items flow ===")
    from seed_storage import staging
    staging.init_tables()

    # Create a conversation_thread item (simulates what threader.py produces)
    thread_content = (
        "**Flynn A. Cruse**: Hey, check out this new AI framework\n"
        "**Wyler Zahm**: Which one? I saw a few announcements today\n"
        "**Flynn A. Cruse**: The one from Anthropic - looks promising for agent workflows\n"
        "**Wyler Zahm**: Interesting, let me dig into the docs"
    )
    thread_metadata = {
        "thread_size": 4,
        "fragment_ids": ["fake-id-1", "fake-id-2", "fake-id-3", "fake-id-4"],
        "speakers": [
            {"name": "Flynn A. Cruse", "role": "speaker", "platform": "discord"},
            {"name": "Wyler Zahm", "role": "speaker", "platform": "discord"},
        ],
        "channel": "imessages",
    }

    test_uri = f"thread://imessages/journey-test-{int(time.time())}"
    sid = staging.stage(
        source_type="conversation_thread",
        source_uri=test_uri,
        raw_content=thread_content,
        author="Flynn A. Cruse, Wyler Zahm",
        channel="imessages",
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata=thread_metadata,
    )
    _assert(sid is not None, f"Staged conversation_thread: {sid}")

    if sid:
        # Process the thread
        item = staging.get_by_id(sid)
        import httpx
        from ingestion.processor import process_one

        async with httpx.AsyncClient(timeout=30) as http:
            await process_one(item, http)

        item = staging.get_by_id(sid)
        _assert(item["status"] == "processed", f"Thread processed: {item['status']}")

        # Verify metadata is preserved (not wiped to empty dict)
        meta = item.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        _assert(meta.get("thread_size") == 4, f"Thread metadata preserved: thread_size={meta.get('thread_size')}")
        _assert(len(meta.get("speakers", [])) == 2, f"Speakers preserved: {len(meta.get('speakers', []))}")

        # Clean up
        staging.update_status([sid], "deleted")

    # Also verify that existing conversation_thread items in staging can flow
    import psycopg2
    conn = psycopg2.connect(os.environ.get("PG_DSN", "postgresql://taskman:postgres@127.0.0.1:30433/task_manager"))
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM seed_staging WHERE source_type = 'conversation_thread'")
    thread_count = cur.fetchone()[0]
    conn.close()
    _assert(thread_count > 0, f"Existing conversation_threads in staging: {thread_count}")


# ── #14: Pre-seed core entities ───────────────────────────────────────

async def test_issue_14():
    """Core entities script creates valid episode bodies for Graphiti."""
    log.info("=== Issue #14: Pre-seed entities ===")
    from scripts.preseed_entities import CORE_ENTITIES, _build_episode_body

    # Verify entity definitions
    _assert(len(CORE_ENTITIES) >= 7, f"At least 7 core entities defined: {len(CORE_ENTITIES)}")

    # Check key entities exist
    names = {e["name"] for e in CORE_ENTITIES}
    _assert("Flynn Cruse" in names, "Flynn Cruse entity defined")
    _assert("Wyler Zahm" in names, "Wyler Zahm entity defined")
    _assert("CruseControl" in names, "CruseControl entity defined")
    _assert("AntKeeper" in names, "AntKeeper entity defined")

    # Verify aliases are set for key entities
    wyler = next(e for e in CORE_ENTITIES if e["name"] == "Wyler Zahm")
    _assert("famed_esteemed" in wyler["aliases"], "famed_esteemed is a Wyler alias")
    _assert("wyler-zahm" in wyler["aliases"], "wyler-zahm is a Wyler alias")

    flynn = next(e for e in CORE_ENTITIES if e["name"] == "Flynn Cruse")
    _assert("flynn-cruse" in flynn["aliases"], "flynn-cruse is a Flynn alias")
    _assert("siliconwarlock" in flynn["aliases"], "siliconwarlock is a Flynn alias")
    _assert("flynnbo" in flynn["aliases"], "flynnbo is a Flynn alias")

    # Verify episode body format
    body = _build_episode_body(wyler)
    _assert("Entity: Wyler Zahm" in body, "Episode body has entity name")
    _assert("Type: person" in body, "Episode body has type")
    _assert("famed_esteemed" in body, "Episode body includes aliases")
    _assert("CTO" in body or "engineering" in body.lower(), "Episode body has description")


# ── Runner ────────────────────────────────────────────────────────────

TEST_MAP = {
    3: test_issue_3,
    10: test_issue_10,
    11: test_issue_11,
    12: test_issue_12,
    13: test_issue_13,
    14: test_issue_14,
}


async def run_all():
    for issue_num in sorted(TEST_MAP.keys()):
        try:
            await TEST_MAP[issue_num]()
        except Exception:
            log.exception("Issue #%d test CRASHED", issue_num)
            global FAIL
            FAIL += 1
        log.info("")


async def run_one(issue_num: int):
    if issue_num not in TEST_MAP:
        log.error("No test for issue #%d. Available: %s", issue_num, sorted(TEST_MAP.keys()))
        return
    try:
        await TEST_MAP[issue_num]()
    except Exception:
        log.exception("Issue #%d test CRASHED", issue_num)
        global FAIL
        FAIL += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    args = sys.argv[1:]
    issue = None
    for a in args:
        if a.startswith("--issue="):
            issue = int(a.split("=")[1])
        elif a == "--issue" and args.index(a) + 1 < len(args):
            issue = int(args[args.index(a) + 1])

    if issue:
        asyncio.run(run_one(issue))
    else:
        asyncio.run(run_all())

    log.info("=" * 60)
    log.info("RESULTS: %d passed, %d failed", PASS, FAIL)
    log.info("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
