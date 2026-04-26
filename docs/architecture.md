# Seed Storage — Architecture

## System overview

AntHaul (seed-storage) is a Discord-first knowledge graph pipeline. It ingests Discord messages and linked content, extracts typed entities via LLM structured output, resolves duplicates with 3-tier entity resolution, and stores everything in Neo4j with dual-label nodes and vector indices for semantic search.

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                          │
│                                                                  │
│   Discord Bot (bot.py)          Batch Import (batch.py)          │
│   discord.py + reactions        DiscordChatExporter JSON         │
│        │                               │                         │
│        └──────────────┬────────────────┘                        │
│                       │ raw_payload (Contract 1)                 │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CELERY: raw_messages queue                   │
│                                                                  │
│                    enrich_message task                           │
│                  dedup → URL extraction → dispatch               │
│                       │                                          │
│          ┌────────────┼────────────────┐                        │
│          │            │                │                         │
│        URL 1        URL 2           URL N                        │
│          │            │                │                         │
│          └────────────┼────────────────┘                        │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONTENT RESOLUTION LAYER                       │
│                                                                  │
│              ContentDispatcher (dispatcher.py)                   │
│                                                                  │
│  Twitter  YouTube  GitHub  Image  PDF  Video  Webpage  Fallback  │
│  (stub)                                        (primary) (last)  │
│                                                                  │
│           Returns ResolvedContent (models.py)                    │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼ enriched_payload (Contract 2)
┌─────────────────────────────────────────────────────────────────┐
│                     CELERY: graph_ingest queue                   │
│                                                                  │
│                    ingest_episode task                           │
│              budget check → rate limit → circuit breaker         │
│                    → extraction → resolution → graph.py          │
│                    → expansion_urls → frontier                   │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTRACTION + RESOLUTION                        │
│                                                                  │
│  extraction.py — OpenAI structured output (1 LLM call/chunk)    │
│  resolution.py — 3-tier entity dedup (canonical→embed→judge)    │
│  preseed.py — known entities + aliases (Postgres)               │
│                         │                                        │
│                    GRAPH LAYER                                   │
│              graph.py — typed dual-label entities                │
│              (:Person:__Entity__), vector indices                │
│                    Neo4j (K8s)                                   │
│              Bolt :30687 / HTTP :30474                           │
└─────────────────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTIER EXPANSION                           │
│                                                                  │
│  scan_frontier (beat, 60s) ──► expand_from_frontier task        │
│     Redis ZSET (seed:frontier)     │                             │
│     priority-sorted URLs           └──► ingest_episode.delay()  │
└─────────────────────────────────────────────────────────────────┘
```

## Component details

### Ingestion

**`seed_storage/ingestion/bot.py`** — Discord bot that listens on configured channel IDs (`DISCORD_CHANNEL_IDS`). For each message:
1. Publishes 📥 reaction to `seed:reactions`
2. Calls `enrich_message.delay(raw_payload)`

Also subscribes to `seed:reactions` pubsub channel to forward reaction events back to Discord (📥 received → ⚙️ processing → 🏷️ tagged → 🧠 graph updated + platform emojis).

**`seed_storage/ingestion/batch.py`** — Parses DiscordChatExporter JSON exports and calls `enrich_message.delay()` for each message. Caps at 5,000 messages per run. Bot-authored messages skipped. Malformed entries logged and skipped.

### Queue design

Two Celery queues with separate worker pools to prevent head-of-line blocking:

| Queue | Tasks | Default concurrency |
|-------|-------|---------------------|
| `raw_messages` | `enrich_message` | 2 |
| `graph_ingest` | `ingest_episode`, `expand_from_frontier`, `scan_frontier` | 4 |

Worker pool: prefork (default). **Never gevent** — resolvers use `asyncio.run()` which is incompatible with gevent.

All tasks use `acks_late=True, reject_on_worker_lost=True` to prevent message loss on worker crash.

### Deduplication

Three independent Redis SETs on DB 2:

| Set | Key format | Purpose |
|-----|-----------|---------|
| `seed:seen_messages` | `{source_type}:{source_id}` | Skip duplicate Discord messages |
| `seed:seen_urls` | SHA256(canonical URL) | Skip duplicate URL resolutions |
| `seed:ingested_content` | SHA256(canonical URL) | Skip duplicate graph writes |

Message dedup is checked in `enrich_message`. URL dedup (resolution) is checked in `enrich_message`. Content dedup (graph write) is checked in `ingest_episode`.

**URL canonicalization** (`dedup.py`): strips tracking params (`utm_*`, `fbclid`, `ref`, `si`, `t`, `s`), lowercases scheme+host, sorts remaining query params, removes trailing slash and fragment. Idempotent.

### Content resolution

`ContentDispatcher` routes URLs to resolvers by priority order. First resolver where `can_handle()` returns `True` wins.

**Priority order:**
1. TwitterResolver — twitter.com / x.com (stub, returns error)
2. YouTubeResolver — youtube.com / youtu.be (yt-dlp + transcription)
3. GitHubResolver — github.com repos (REST API + README)
4. ImageResolver — image extensions (vision LLM)
5. PDFResolver — .pdf (docling + unstructured fallback)
6. VideoResolver — video extensions (ffmpeg + Whisper)
7. WebpageResolver — generic HTTP(S) (trafilatura + readability fallback)
8. FallbackResolver — all HTTP(S) (BeautifulSoup, never raises)

All resolvers return `ResolvedContent`. On failure they return `ResolvedContent.error_result(url, error)`. Resolvers must never raise — all exceptions are caught internally.

`ContentDispatcher` also catches unexpected resolver exceptions as a safety net.

`resolved_at` is stamped by the dispatcher (not the resolver) after resolution completes.

### Entity extraction + graph writes

**Extraction** (`extraction.py`): Uses OpenAI structured output (`response_format` with `json_schema`) for deterministic entity extraction. One LLM call per content chunk. Per-source schemas control which entity types to prioritize. Coreference pre-processing replaces known aliases (from preseed) before sending to the LLM.

**Resolution** (`resolution.py`): 3-tier entity deduplication:
1. **Tier 1** — Canonical name normalization + preseed alias lookup (O(1))
2. **Tier 2** — Embedding cosine similarity via Neo4j vector index (threshold 0.65)
3. **Tier 3** — LLM-as-judge for the ambiguous 0.65-0.90 band (~5% of entities)

**Graph writes** (`graph.py`): All entity nodes have dual labels `(:Person:__Entity__)`. MERGE by `canonical_name`. Relationships are typed (`WORKS_FOR`, `DISCUSSES`, etc.). Vector indices on 1536-dim embeddings (text-embedding-3-small).

**`ingest_episode` task**: Reads enriched payload, persists to Postgres staging, extracts entities via `extract_one()`, resolves via `resolve_entity()`, writes typed nodes via `graph.py`.

`group_id` is always `"ant-haul"`. Never per-channel. All knowledge is in one unified graph.

**Loader concurrency is 1** — entity resolution requires seeing the current graph state. Parallel writes cause merge races.

### Frontier expansion

Discovered URLs (from `expansion_urls` in `ResolvedContent`) are added to a Redis sorted set:
- Key: `seed:frontier` (ZSET, score = priority float)
- Metadata: `seed:frontier:meta:{url_hash}` (HASH)

`scan_frontier` (Celery beat, every 60s) picks the top-N URLs by priority and enqueues `expand_from_frontier` tasks. Disabled when `FRONTIER_AUTO_ENABLED=false`.

**Priority scoring:**
```
score = 0.5
      - 0.15 * depth
      + resolver_bonus  (youtube/github: +0.2, twitter: -1.0)
      + domain_bonus    (arxiv.org: +0.3, github.com/youtube.com: +0.2)
      + channel_bonus   (caller-supplied)
      max(0.0, score)
```

Hard limits: `HARD_DEPTH_CEILING=5` (never expand beyond 5 hops), `MAX_EXPANSION_BREADTH=20` (max child URLs per resolution).

ZADD uses NX — existing frontier entries keep their original priority score.

**Depth policies** (max hops by resolver type):
- youtube, github: 3
- pdf, image, video, unknown: 2
- webpage: 4
- twitter/tweet: 0 (never expand)
- fallback: 1

### Resilience

**Circuit breaker** (`circuit_breaker.py`): Per-service Redis-backed breaker. Tracks failure count in `seed:circuit:{service}:*`. Opens after 5 failures (configurable). Cooldown: 300s. When opens/closes → Discord alert.

**Cost tracking** (`cost_tracking.py`): Daily counter in `seed:cost:daily:YYYY-MM-DD` (TTL 48h). Default budget $5.00/day. When exceeded → `graph_ingest` tasks sleep 5 min and retry. When 80% reached → warning alert.

**Rate limiting** (`rate_limiting.py`): Sliding window on `seed:ratelimit:graphiti`. Default 100 `add_episode()` calls/minute. When exceeded → retry with backoff.

**Dead letters** (`worker/dead_letters.py`): Tasks that exhaust retries are RPUSH'd to `seed:dead_letters`. Traceback is sanitized (API keys masked). Replay via `worker/replay.py`.

**Notifications** (`notifications.py`): Fire-and-forget Discord webhook alerts. Sync `httpx.Client`. Debounce via Redis (`seed:alert:debounce:{key}`). Empty webhook URL → silently skipped.

### Health endpoint

`GET :8080/health` (aiohttp) — returns 200 (healthy) or 503 (unhealthy).

Checks (5s timeout each): Redis PING, Neo4j bolt connectivity, Celery inspector ping, bot liveness flag (`seed:bot:connected`).

Details: queue depths, frontier size, dead letter count, daily cost, budget, messages/URLs seen, open circuit breakers.

## Data flow

```
Discord message
    │
    ├── raw_payload (Contract 1)
    │       source_type, source_id, source_channel, author,
    │       content, timestamp, attachments, metadata
    │
    ├── enrich_message (Celery task)
    │       dedup by source_type:source_id
    │       extract URLs from content + attachments
    │       dedup URLs by SHA256(canonical)
    │       dispatch new URLs → ContentDispatcher
    │
    ├── enriched_payload (Contract 2)
    │       message: raw_payload
    │       resolved_contents: [ResolvedContent.to_dict(), ...]
    │
    ├── ingest_episode (Celery task)
    │       check budget + rate limit + circuit breaker
    │       extract entities (extraction.py, structured output)
    │       resolve entities (resolution.py, 3-tier dedup)
    │       write typed nodes (graph.py, dual-label entities)
    │       add expansion_urls to frontier
    │
    └── Neo4j (via graph.py)
            Typed entity nodes (:Person:__Entity__, :Concept:__Entity__, etc.)
            Source nodes (provenance), Fact nodes, Tag nodes
            Typed relationships (WORKS_FOR, DISCUSSES, CITES, etc.)
            Vector indices (1536d cosine)
```

## Redis key namespaces

All keys on **DB 2** (`/2` suffix in `REDIS_URL`). Ant-keeper uses DB 0.

| Key | Type | Purpose |
|-----|------|---------|
| `seed:seen_messages` | SET | Message dedup (`{source_type}:{source_id}`) |
| `seed:seen_urls` | SET | URL resolution dedup (SHA256 hashes) |
| `seed:ingested_content` | SET | Graph write dedup (SHA256 hashes) |
| `seed:frontier` | ZSET | Expansion frontier (score = priority) |
| `seed:frontier:meta:{hash}` | HASH | Frontier URL metadata |
| `seed:dead_letters` | LIST | Failed tasks (RPUSH/LPOP FIFO) |
| `seed:circuit:{service}:*` | various | Circuit breaker state per service |
| `seed:cost:daily:YYYY-MM-DD` | STRING | Daily LLM spend counter (TTL 48h) |
| `seed:ratelimit:extraction` | ZSET | Sliding window rate limiter |
| `seed:reactions` | pubsub | Discord reaction events channel |
| `seed:bot:connected` | STRING | Bot liveness flag (`1` = connected) |

## Infrastructure

### Kubernetes (production)

- **Namespace:** `ant-keeper`
- **Neo4j:** StatefulSet with PVC (5Gi). NodePort: Bolt 30687, HTTP 30474.
- **Redis:** Existing service on NodePort 30679. Seed-storage uses DB 2.
- **Seed Storage pod:** Single Docker container running supervisord with 5 processes.

### supervisord processes

| Process | Command | Purpose |
|---------|---------|---------|
| `bot` | `python -m seed_storage.ingestion.bot` | Discord bot — listens for messages + reaction pubsub |
| `worker-raw` | `celery worker -Q raw_messages --concurrency 2` | Enrich message tasks (URL extraction + dispatch) |
| `worker-graph` | `celery worker -Q graph_ingest --concurrency 4` | Graph ingest + frontier expansion tasks |
| `beat` | `celery beat` | scan_frontier schedule (every 60s) |
| `health` | `python -m seed_storage.health` | Health endpoint on :8080 |

### Local development

`docker-compose.yml` provides Redis 7 (port 6379, appendonly) and Neo4j 5 (ports 7474, 7687, auth `neo4j/localdev`, APOC plugin) for local development.

### Credential injection (ant-keeper)

Sensitive credentials are never stored in environment literals or code. They are managed by ant-keeper and injected at pod startup.

| Credential | ant-keeper name | Env var injected | Mode | Notes |
|-----------|----------------|-----------------|------|-------|
| OpenAI API key | `openai` | `OPENAI_API_KEY` | env-mode (proxy-enabled) | Requires `proxy_target` before daemon registration |
| Discord bot token | `discord-bot-seed-storage` | `DISCORD_BOT_TOKEN_PATH` | file-mode | `config.py` reads file at path on startup |
| Neo4j password | `neo4j-seed-storage` | `NEO4J_PASSWORD_PATH` | file-mode | `config.py` reads file at path on startup |
| GitHub PAT | `github-pat` | `GITHUB_TOKEN` | env-mode (proxy-enabled, optional) | Requires `proxy_target` before daemon registration |
| Discord alerts webhook | `discord-alerts-webhook` | `DISCORD_ALERTS_WEBHOOK_PATH` | file-mode | `config.py` reads file at path on startup |

**How file-mode works:** `config.py` `Settings._resolve_file_credentials()` runs as a pydantic `model_validator`. If `*_PATH` is set and the corresponding value field is empty, it reads the file at the path and populates the value. This happens once at startup.

**How env-mode (proxy-enabled) works:** Env-mode credentials are injected into the process environment directly. When marked as proxy-enabled, iron-proxy intercepts outbound HTTP to the credential's `proxy_target` domain and injects authentication headers.

**CRITICAL:** The `proxy_target` must be configured before daemon registration:
```bash
./infra/scripts/proxy-enable.sh openai https://api.openai.com
./infra/scripts/proxy-enable.sh github-pat https://api.github.com
```
Omitting this step causes deploy-time failure: `Credential '<name>' missing proxy_target`.

**External HTTP:** All outbound HTTP to external APIs goes through iron-proxy (enforces allowlist, injects auth for proxy-enabled credentials). Internal services (`redis.ant-keeper.svc`, `neo4j.ant-keeper.svc`) are accessed directly.

## Security model

- All secrets managed by ant-keeper file-mode credentials. Never in code or environment literals.
- External HTTP via iron-proxy sidecar (enforces allowlist, handles auth injection).
- Redis DB isolation: DB 2 for seed-storage, DB 0 for ant-keeper.
- API keys masked in all log output by `_SecretMaskingFilter` in `config.py`.
- Dead letter tracebacks sanitized before storage.
- No raw Cypher execution from user input. All graph writes through graph.py typed functions.
