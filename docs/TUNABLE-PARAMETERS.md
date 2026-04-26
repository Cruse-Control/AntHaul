# Tunable Parameters

All parameters are configured via `seed_storage/config.py` (pydantic-settings `Settings` class) or module-level constants. Environment variables override defaults.

## Entity Resolution

| Parameter | Default | Config key | Effect |
|-----------|---------|------------|--------|
| Similarity threshold | 0.65 | `ENTITY_SIMILARITY_THRESHOLD` | Minimum cosine similarity for Tier 2 entity matching. Lower = more merges, higher = more new entities. |
| Ambiguous threshold | 0.90 | `ENTITY_AMBIGUOUS_THRESHOLD` | Score above which Tier 2 auto-merges without LLM judge. Between similarity and ambiguous thresholds triggers Tier 3 LLM judge. |
| Embedding model | text-embedding-3-small | `EMBEDDING_MODEL` | Model for entity/fact/source embeddings. text-embedding-3-small is 1536 dims. |
| Embedding dimensions | 1536 | `EMBEDDING_DIM` | Must match the embedding model's output dimensions. |

**Tuning guidance:**
- Start with 0.65/0.90. If too many false merges, raise similarity threshold to 0.70.
- If too many near-duplicates, lower ambiguous threshold to 0.85 (more LLM judge calls, higher cost).
- Monitor resolution quality via `tests/graph_quality/test_resolution.py`.

## Extraction

| Parameter | Default | Config key | Effect |
|-----------|---------|------------|--------|
| Extraction model | gpt-4o-mini | `EXTRACTION_MODEL` | LLM for entity extraction. Cheapest model that supports structured output. |
| Enrichment model | gpt-4o-mini | `ENRICHMENT_MODEL` | LLM for tag/summary enrichment. |
| Extraction concurrency | 3 | `EXTRACTION_CONCURRENCY` | Parallel extraction calls per batch. Higher = faster but more API pressure. |
| Content truncation | 8000 chars | Hardcoded in extraction.py | Max content sent to LLM. ~2k tokens. |
| Min content words | 50 | Hardcoded in extraction.py | Items with fewer words skip extraction. |

## Loader

| Parameter | Default | Config key | Effect |
|-----------|---------|------------|--------|
| Loader concurrency | 1 | `LOADER_CONCURRENCY` | Sequential graph writes. Must be 1 for consistent entity resolution. |
| Batch size | 200 | `BATCH_SIZE_DEFAULT` | Items per batch processing run. |

## Community Detection

| Parameter | Default | Location | Effect |
|-----------|---------|----------|--------|
| Leiden gamma | 1.0 | `communities.run_leiden(gamma=)` | Resolution parameter. Higher = more communities. 1.0 is standard. |
| Summary limit | 20 | `communities.summarize_communities(limit=)` | Max communities to summarize per run. |

## Cost Controls

| Parameter | Default | Config key | Effect |
|-----------|---------|------------|--------|
| Daily budget | $5.00 | `DAILY_BUDGET_USD` | All LLM-calling workers pause when exceeded. |
| Batch cost ceiling | $2.00 | `BATCH_COST_CEILING_USD` | Per-batch cost limit. |
| Circuit breaker threshold | 5 | `CIRCUIT_BREAKER_THRESHOLD` | Consecutive failures before circuit opens. |

## Pipeline

| Parameter | Default | Config key | Effect |
|-----------|---------|------------|--------|
| Orphan timeout | 1 hour | Hardcoded in staging.py | Items stuck in transient status (extracting/loading) reset after this. |
| Quality gate | varies | `_is_loadable()` in loader.py | Rejects auth walls, cookie walls, stub content before extraction. |

## Neo4j

| Parameter | Default | Config key | Effect |
|-----------|---------|------------|--------|
| Neo4j URI | bolt://127.0.0.1:7687 | `NEO4J_URI` | Neo4j connection. |
| Neo4j user | neo4j | `NEO4J_USER` | |
| Group ID | ant-haul | `GROUP_ID` | Graph partition identifier. Always "ant-haul". |

## Preseed Entities

Managed via `seed_storage/preseed.py`. Stored in Postgres `seed_preseed_entities` table.

Default preseed entities:
- **flynn cruse** (Person): aliases siliconwarlock, flynnbo, flynn-cruse, flynn
- **wyler zahm** (Person): aliases famed_esteemed, wylerza, wyler-zahm, wyler
- **crusecontrol** (Organization): aliases cruse-control, cruse control, cc

Add new preseed entities:
```python
from seed_storage.preseed import add_entity, add_alias
add_entity("new entity", "Person", aliases=["alias1", "alias2"], description="...")
add_alias("existing entity", "new_alias")
```
