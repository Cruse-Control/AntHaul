# AntHaul (seed-storage) — Agent Instructions

## What this project is

A Discord-first knowledge graph for CruseControl. Ingests Discord messages and linked content, extracts typed entities via LLM structured output, resolves duplicates with 3-tier entity resolution, and writes to Neo4j with dual-label nodes and vector indices. Runs as an ant-keeper daemon (K8s pod, `ant-keeper` namespace).

## Pipeline architecture

```
Discord Bot / Batch Import / Express Ingest
    ↓ stage() [Postgres]
    ↓ status='staged'

Processor [reads staged, writes processed]
    ↓ Extract content (yt-dlp, trafilatura, FxTwitter, GitHub API, etc.)
    ↓ status='processed'

Enricher [reads processed, writes enriched]
    ↓ LLM tagging + summary (1 call, gpt-4o-mini)
    ↓ status='enriched'

Extractor [reads enriched, writes extracted]
    ↓ LLM entity extraction via OpenAI structured output (1 call per chunk)
    ↓ Per-source Pydantic schemas (Person, Organization, Concept, etc.)
    ↓ metadata.extraction={entities: [...], relationships: [...]}
    ↓ status='extracted'

Loader [reads extracted, writes loaded]
    ↓ 3-tier entity resolution (normalize → embed → LLM judge)
    ↓ Write typed nodes (:Person:__Entity__) via graph.py
    ↓ Create vector indices, semantic relationships
    ↓ status='loaded'

Community Detection [periodic batch job]
    ↓ Neo4j GDS Leiden algorithm
    ↓ LLM community summaries on __Community__ nodes
```

**Postgres staging is the backbone.** Every pipeline step reads from and writes to the `seed_staging` table. No data lives only in ephemeral Celery messages.

## DEPLOY PREREQUISITES (read before registering the daemon)

**Env-mode credentials require `proxy_target` before daemon registration.** `openai` and `github-pat` are env-mode, proxy-enabled credentials. They **must** have a `proxy_target` URL configured in ant-keeper before the daemon is registered.

```bash
./infra/scripts/proxy-enable.sh openai https://api.openai.com
./infra/scripts/proxy-enable.sh github-pat https://api.github.com  # if using github-pat
```

File-mode credentials (`discord-bot-seed-storage`, `neo4j-seed-storage`, `discord-alerts-webhook`) do NOT need proxy targets.

## Architecture decisions

### Graph: Custom extraction + Neo4j-native writes

All graph writes go through `seed_storage/graph.py`. Entity extraction uses OpenAI structured output (1 LLM call per chunk). Entity resolution uses 3 tiers:

1. **Tier 1 — Canonical name normalization** (O(1), deterministic): lowercase, strip @/#, preseed alias lookup
2. **Tier 2 — Embedding similarity** via Neo4j vector index (threshold 0.65)
3. **Tier 3 — LLM judge** for the 0.65-0.90 ambiguous band (~5% of entities)

Entity nodes use **dual labels**: `(:Person:__Entity__)`, `(:Organization:__Entity__)`, etc. The `__Entity__` base label enables cross-type queries. Type-specific labels enable filtered queries.

`group_id` is always `"ant-haul"`. Never per-channel — the entire knowledge base is one unified graph.

Schema is managed by `graph.init_schema()` called on startup (idempotent). No migration chain.

**Preseed entities** are stored in Postgres (`seed_preseed_entities` table) and loaded via `preseed.py`. These provide known aliases for coreference pre-processing and Tier 1 resolution. Add new preseed entities via `preseed.add_entity()`.

### Extraction: structured output, per-source schemas

`seed_storage/extraction.py` uses OpenAI `response_format` with `json_schema` for deterministic structured output. Per-source entity type priorities are in `SOURCE_ENTITY_TYPES`. Items with <50 words are skipped.

Coreference pre-processing replaces known aliases (from preseed) before sending to the LLM.

### Credentials: file-mode via ant-keeper

Sensitive credentials (`DISCORD_BOT_TOKEN`, `NEO4J_PASSWORD`, `DISCORD_ALERTS_WEBHOOK_URL`) are stored in ant-keeper and injected as file paths (`*_PATH` env vars). `config.py` reads the file at startup. **Never hardcode credentials. Never bypass iron-proxy.**

### Redis: DB 2

All seed-storage Redis keys are on **DB 2** (`redis://redis.ant-keeper.svc:6379/2`). Ant-keeper uses DB 0.

Key namespaces:
- `seed:seen_messages` — message dedup SET
- `seed:seen_urls` — URL dedup SET (SHA256 hashes)
- `seed:ingested_content` — graph-ingested URL SET
- `seed:frontier` — expansion frontier ZSET (score = priority)
- `seed:frontier:meta:{hash}` — frontier metadata HASH
- `seed:dead_letters` — failed task LIST
- `seed:circuit:{service}:*` — circuit breaker state
- `seed:cost:daily:YYYY-MM-DD` — daily LLM spend counter
- `seed:reactions` — Discord reaction pubsub channel
- `seed:bot:connected` — bot liveness flag

### Async/sync boundary

- **Resolvers:** `async def resolve()` — non-blocking HTTP via httpx
- **Celery tasks:** synchronous — bridge with `asyncio.run()` per-task invocation
- **`send_alert()`:** synchronous `httpx.Client` — never use `asyncio.run()`
- **Worker pool:** default prefork. Never `--pool=gevent`

**Mocking async functions in integration tests:** When a Celery task calls `asyncio.run(some_async_fn(...))`, the mock must be an `AsyncMock`. Use `new=AsyncMock(return_value=[...])`.

### Embeddings: always OpenAI

`OPENAI_API_KEY` is required. Embeddings use `text-embedding-3-small` (1536 dimensions) via `seed_storage/embeddings.py`.

## Key paths

| Path | Purpose |
|------|---------|
| `seed_storage/config.py` | All configuration (pydantic-settings `Settings` singleton) |
| `seed_storage/models.py` | Shared Pydantic types: `ExtractedEntity`, `ExtractionResult`, status constants |
| `seed_storage/extraction.py` | LLM entity extraction via OpenAI structured output |
| `seed_storage/resolution.py` | 3-tier entity resolution (normalize → embed → LLM judge) |
| `seed_storage/graph.py` | Neo4j client: typed dual-label entities, vector indices, CRUD |
| `seed_storage/preseed.py` | Preseed entity table in Postgres (aliases, known entities) |
| `seed_storage/embeddings.py` | OpenAI embedding client (text-embedding-3-small, 1536 dims) |
| `seed_storage/staging.py` | Postgres staging table: stage, get, update, reset |
| `seed_storage/communities.py` | Neo4j GDS Leiden community detection + LLM summaries |
| `seed_storage/mcp_server.py` | MCP server for Claude Code (hybrid search, explore, ingest) |
| `seed_storage/query/search.py` | Hybrid vector + fulltext search wrapper |
| `seed_storage/batch/__main__.py` | Batch CLI: run, reset, status, poll, progress |
| `seed_storage/batch/coordinator.py` | Batch job lifecycle (create, progress, cancel) |
| `seed_storage/batch/batch_api.py` | OpenAI Batch API integration (50% cost savings) |
| `seed_storage/worker/tasks.py` | Celery tasks (enrich_message, ingest_episode, frontier) |
| `seed_storage/worker/app.py` | Celery app + queue routing + beat schedule |
| `seed_storage/enrichment/dispatcher.py` | Routes URLs to resolvers |
| `seed_storage/enrichment/models.py` | Shared types: `ResolvedContent`, `ContentType` |
| `seed_storage/ingestion/bot.py` | Discord bot real-time ingestion |
| `seed_storage/health.py` | Health endpoint on :8080 |
| `seed_storage/expansion/frontier.py` | Redis frontier operations |
| `seed_storage/dedup.py` | Dedup store + URL canonicalization |
| `ingestion/loader.py` | Loader: extraction→resolution→typed-graph-write pipeline |
| `ingestion/express.py` | Single-shot URL ingest (stage→process→enrich→extract→load) |
| `scripts/query.py` | CLI query interface |
| `scripts/rebuild_graph.py` | Full graph rebuild from Postgres staging |
| `scripts/rollback.py` | Graph rollback by timestamp |

## Neo4j schema

**Node labels:**
- Entity types: `(:Person:__Entity__)`, `(:Organization:__Entity__)`, `(:Product:__Entity__)`, `(:Concept:__Entity__)`, `(:Location:__Entity__)`, `(:Event:__Entity__)`
- Other: `Source`, `Fact`, `Tag`, `__Community__`

**Relationship types:**
- Entity→Entity: `WORKS_FOR`, `FOUNDED`, `DISCUSSES`, `CITES`, `CREATED`, `USES`, `PART_OF`, `LOCATED_IN`, `RELATED_TO`, `SUPPORTS`, `MADE_DECISION`, `APPLIES_MODEL`
- Source→Fact: `EXTRACTED_FROM`
- Fact→Entity: `MENTIONS`
- Entity→Community: `IN_COMMUNITY`
- Source→Tag: `HAS_TAG`

**Vector indices:** `entity_embedding` (1536d), `fact_embedding` (1536d), `source_embedding` (1536d) — all cosine similarity.

**Fulltext indices:** `entity_name_fulltext`, `fact_statement`, `source_content`.

## Batch CLI

```bash
# Show pipeline status
python -m seed_storage.batch status

# Run extraction on enriched items
python -m seed_storage.batch run --from enriched --limit 50

# Dry run (inspect without executing)
python -m seed_storage.batch run --from enriched --limit 50 --dry-run

# Use OpenAI Batch API (50% discount, 24hr SLA)
python -m seed_storage.batch run --from enriched --batch-api

# Poll batch API job
python -m seed_storage.batch poll <batch-id>

# Reset items for replay
python -m seed_storage.batch reset --to enriched

# Rebuild graph from scratch
python scripts/rebuild_graph.py --confirm
```

## Resolver quirks

**Twitter/X:** Stub only. Returns `error_result()`.

**YouTube:** Uses yt-dlp for metadata and transcript. Manual captions preferred over auto-generated. Falls back to Whisper transcription if no captions available.

**Video:** Downloads to temp file -> ffmpeg -> Whisper. Temp file cleaned up in `finally` block.

**PDF:** docling primary, unstructured fallback. Both are heavy imports; test with mocks.

**Image:** Calls vision LLM. Returns description in `summary` field and copies it to `text`.

**Webpage:** trafilatura primary, readability-lxml fallback. Both-fail returns `error_result()`.

**Dispatcher priority order (highest to lowest):** Twitter -> YouTube -> GitHub -> Image -> PDF -> Video -> Webpage -> Fallback

## Celery configuration

**Two queues:**
- `raw_messages` — `enrich_message` tasks (concurrency: `WORKER_CONCURRENCY_RAW`, default 2)
- `graph_ingest` — `ingest_episode`, `expand_from_frontier`, `scan_frontier` tasks (concurrency: `WORKER_CONCURRENCY_GRAPH`, default 4)

All tasks use `acks_late=True` and `reject_on_worker_lost=True`.

**Beat:** `scan_frontier` runs every 60s. No-op when `FRONTIER_AUTO_ENABLED=false`.

## Ingestion contracts

### Contract 1 — `raw_payload` shape

```python
{
    "source_type": str,       # "discord", "expansion", ...
    "source_id": str,         # Discord snowflake or frontier hash
    "source_channel": str,    # channel name
    "author": str,            # display name
    "content": str,           # raw text including URLs
    "timestamp": str,         # ISO 8601
    "attachments": list[str], # direct URLs
    "metadata": dict,         # source-specific
}
```

### Contract 2 — `enriched_payload` shape

```python
{
    "message": raw_payload,
    "resolved_contents": [rc.to_dict(), ...],
}
```

### Contract 3 — extraction result (in metadata.extraction)

```python
{
    "entities": [{"name": str, "canonical_name": str, "entity_type": str, "description": str, "aliases": list, "confidence": float}],
    "relationships": [{"source": str, "target": str, "relationship_type": str, "description": str, "confidence": float}],
    "model_used": str,
    "tokens_input": int,
    "tokens_output": int,
    "extracted_at": str  # ISO 8601
}
```

## Dead letters

Stored as JSON in `seed:dead_letters` LIST (RPUSH). Each entry includes `task_name`, `payload`, `traceback` (sanitized), `retries`, `timestamp`.

Replay via `python -m seed_storage.worker.replay`.

## Known limitations

- **Twitter/X:** Stub only — returns error for all twitter.com and x.com URLs.
- **Frontier auto-expansion:** Disabled by default (`FRONTIER_AUTO_ENABLED=false`).
- **Single graph:** All sources share `group_id="ant-haul"`. No per-channel graph isolation.
- **Loader concurrency=1:** Entity resolution requires sequential writes to see current graph state. Parallel writes cause merge races.
- **Community detection requires GDS:** Neo4j GDS plugin must be installed for Leiden algorithm. Falls back gracefully if not available.
- **Reactions require bot + Redis pubsub:** If the bot is disconnected, reaction events are dropped silently.
- **Ruff baseline:** Run `ruff check . && ruff format .` early when modifying code.
- **`asyncio.run()` inside Celery tasks:** Each task uses `asyncio.run()` — creates a new event loop per invocation. Integration tests must use `AsyncMock` for async function mocks.
- **Health endpoint startup:** Serves 503 until dependencies are healthy.
- **Env-mode credentials require proxy_target before deploy.**
- **Neo4j `execute_query` parameter syntax:** Use `parameters_={"key": value}` or `**kwargs` directly. Never `params=`.

## Running tests

```bash
# Unit tests (no infrastructure required)
uv run pytest tests/unit/ -v

# Integration tests (requires docker compose up)
docker compose -p seed-storage-dev up -d
uv run pytest tests/integration/ -m integration -v

# Graph quality tests (requires loaded Neo4j)
uv run pytest tests/graph_quality/ -m integration -v

# E2E tests (requires full stack)
uv run pytest tests/e2e/ -v

# Security tests
uv run pytest tests/security/ -v
```
