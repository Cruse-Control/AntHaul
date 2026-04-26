"""Entity extraction via OpenAI structured output.

One LLM call per content chunk. Per-source schemas control which entity types
to prioritize. Coreference pre-processing normalizes known aliases before extraction.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from openai import OpenAI

from seed_storage import staging
from seed_storage.config import settings, BATCH_SIZE_DEFAULT, EXTRACTION_CONCURRENCY
from seed_storage.models import ExtractedEntity, ExtractedRelationship, ExtractionResult
from seed_storage.preseed import get_alias_map, init_preseed_table

log = logging.getLogger("extraction")

# Per-source entity type priorities
SOURCE_ENTITY_TYPES: dict[str, list[str]] = {
    "discord": ["Person", "Organization", "Product", "Concept"],
    "imessage": ["Person", "Organization", "Product", "Concept"],
    "x_twitter": ["Person", "Concept", "Product"],
    "youtube": ["Person", "Concept", "Product", "Event"],
    "github": ["Person", "Organization", "Product", "Concept"],
    "pdf": ["Concept", "Product", "Organization"],
    "webpage": ["Concept", "Product", "Organization", "Person"],
    "image": ["Person", "Product", "Event"],
    "video": ["Person", "Concept", "Product", "Event"],
}

RELATIONSHIP_TYPES = [
    "WORKS_FOR", "FOUNDED", "DISCUSSES", "CITES", "CREATED", "USES",
    "PART_OF", "LOCATED_IN", "RELATED_TO", "SUPPORTS",
]

# Extraction JSON schema for OpenAI structured output
EXTRACTION_SCHEMA = {
    "name": "extraction_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {"type": "string"},
                        "description": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "entity_type", "description", "aliases"],
                    "additionalProperties": False,
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "relationship_type": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "target", "relationship_type", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities", "relationships"],
        "additionalProperties": False,
    },
}


def _build_system_prompt(source_type: str, alias_map: dict[str, str]) -> str:
    """Build the extraction system prompt with per-source entity types and known aliases."""
    entity_types = SOURCE_ENTITY_TYPES.get(source_type, ["Person", "Concept", "Product"])

    known_aliases = []
    seen = set()
    for alias, canonical in alias_map.items():
        if canonical not in seen:
            aliases_for = [a for a, c in alias_map.items() if c == canonical and a != canonical]
            if aliases_for:
                known_aliases.append(f"  {canonical} (also known as: {', '.join(aliases_for)})")
            seen.add(canonical)

    alias_block = "\n".join(known_aliases[:20]) if known_aliases else "  (none)"

    return f"""You extract structured entities and relationships from content.

Extract entities of these types: {', '.join(entity_types)}.
For each entity: name (as mentioned in text), type (from the list above), description (1 sentence), aliases if known.
For each relationship: source entity name, target entity name, type (from: {', '.join(RELATIONSHIP_TYPES)}), description (1 sentence).

Rules:
- Only extract entities clearly mentioned in the text. Do not infer entities not present.
- Do not extract URLs, hashtags, or generic phrases ("the project", "this tool") as entities.
- Use the canonical name for known entities listed below.
- Be concise. Description should be one sentence.

Known entities and aliases:
{alias_block}"""


def _apply_coreference(text: str, alias_map: dict[str, str]) -> str:
    """Replace known aliases in text with canonical names."""
    for alias, canonical in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        if alias == canonical:
            continue
        # Word-boundary replacement to avoid partial matches
        pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
        text = pattern.sub(canonical, text)
    return text


_ENTITY_TYPE_MAP = {
    "Person": "Person", "Organization": "Organization", "Product": "Product",
    "Concept": "Concept", "Location": "Location", "Event": "Event",
    # Common LLM variants that aren't in our strict schema
    "Company": "Organization", "Startup": "Organization", "Institution": "Organization",
    "Tool": "Product", "Framework": "Product", "Library": "Product", "Software": "Product",
    "Technology": "Product", "Platform": "Product", "Service": "Product",
    "Place": "Location", "City": "Location", "Country": "Location",
    "Topic": "Concept", "Idea": "Concept", "Method": "Concept", "Theory": "Concept",
}


def _normalize_entity_type(raw_type: str) -> str:
    """Map LLM entity type to our canonical EntityType, defaulting to Concept."""
    return _ENTITY_TYPE_MAP.get(raw_type, "Concept")


def _parse_extraction(raw: dict, model_used: str, input_tokens: int,
                      output_tokens: int) -> ExtractionResult:
    """Parse LLM structured output into ExtractionResult."""
    entities = []
    for e in raw.get("entities", []):
        canonical = e["name"].lower().strip().lstrip("@#")
        entities.append(ExtractedEntity(
            name=e["name"],
            canonical_name=canonical,
            entity_type=_normalize_entity_type(e.get("entity_type", "Concept")),
            description=e.get("description", ""),
            aliases=e.get("aliases", []),
        ))
    relationships = []
    for r in raw.get("relationships", []):
        relationships.append(ExtractedRelationship(
            source=r["source"].lower().strip(),
            target=r["target"].lower().strip(),
            relationship_type=r.get("relationship_type", "RELATED_TO"),
            description=r.get("description", ""),
        ))
    return ExtractionResult(
        entities=entities,
        relationships=relationships,
        model_used=model_used,
        tokens_input=input_tokens,
        tokens_output=output_tokens,
    )


def extract_one(item: dict, client: OpenAI | None = None,
                alias_map: dict[str, str] | None = None) -> ExtractionResult:
    """Extract entities and relationships from a single staging item.

    Args:
        item: Postgres staging row dict (must have raw_content, source_type, metadata).
        client: OpenAI client. Created from settings if None.
        alias_map: Preseed alias map. Loaded from DB if None.

    Returns ExtractionResult with entities, relationships, and token counts.
    """
    if client is None:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
    if alias_map is None:
        alias_map = get_alias_map()

    content = item.get("raw_content", "") or ""
    source_type = item.get("source_type", "unknown")
    meta = item.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)

    # Skip tiny content
    if len(content.split()) < 50 and not meta.get("media_urls"):
        return ExtractionResult(
            entities=[], relationships=[],
            model_used="skipped", tokens_input=0, tokens_output=0,
        )

    # Apply coreference pre-processing
    processed_content = _apply_coreference(content, alias_map)

    # Build prompt with enrichment context
    enrichment_header = ""
    tags = meta.get("tags", [])
    summary = meta.get("summary", "")
    if tags and tags != ["uncategorized"]:
        enrichment_header += f"Tags: {', '.join(tags)}\n"
    if summary:
        enrichment_header += f"Summary: {summary}\n"
    discord_ctx = meta.get("discord_context", "")
    if discord_ctx:
        enrichment_header += f"Shared with context: {discord_ctx}\n"

    user_content = f"Source type: {source_type}\n"
    if enrichment_header:
        user_content += enrichment_header + "\n"
    user_content += f"Content:\n{processed_content[:8000]}"  # ~2k tokens max

    system_prompt = _build_system_prompt(source_type, alias_map)

    response = client.chat.completions.create(
        model=settings.EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_schema", "json_schema": EXTRACTION_SCHEMA},
        temperature=0.1,
    )

    usage = response.usage
    raw = json.loads(response.choices[0].message.content)
    return _parse_extraction(
        raw,
        model_used=settings.EXTRACTION_MODEL,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


async def extract_batch(limit: int = None, dry_run: bool = False) -> dict:
    """Extract entities from a batch of enriched items.

    Reads items at status='enriched', runs extract_one() on each,
    stores result in metadata.extraction, updates status to 'extracted'.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    limit = limit or BATCH_SIZE_DEFAULT
    items = staging.get_staged(status="enriched", limit=limit)
    if not items:
        log.info("No enriched items to extract")
        return {"extracted": 0, "failed": 0, "skipped": 0}

    if dry_run:
        log.info("Would extract %d items", len(items))
        return {"would_extract": len(items)}

    # Mark items as extracting
    item_ids = [str(i["id"]) for i in items]
    staging.update_status(item_ids, "extracting")

    # Load shared resources once
    init_preseed_table()
    alias_map = get_alias_map()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    extracted = 0
    failed = 0
    skipped = 0

    def _do_extract(item: dict) -> tuple[str, ExtractionResult | None, str | None]:
        """Thread worker: extract one item, return (item_id, result, error)."""
        item_id = str(item["id"])
        try:
            result = extract_one(item, client=client, alias_map=alias_map)
            return (item_id, result, None)
        except Exception as exc:
            return (item_id, None, str(exc)[:500])

    # Run extractions with bounded concurrency via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=EXTRACTION_CONCURRENCY) as pool:
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(pool, _do_extract, item) for item in items]
        results = await asyncio.gather(*futures)

    for item_id, result, error in results:
        if error:
            log.warning("Extraction failed for %s: %s", item_id, error)
            staging.update_status([item_id], "failed")
            staging.patch_metadata(item_id, {"extraction_error": error})
            failed += 1
        elif result and result.model_used == "skipped":
            staging.update_status([item_id], "extracted")
            staging.patch_metadata(item_id, {
                "extraction": {"entities": [], "relationships": [],
                               "model_used": "skipped", "tokens_input": 0, "tokens_output": 0,
                               "extracted_at": datetime.now(timezone.utc).isoformat()},
            })
            skipped += 1
        elif result:
            staging.patch_metadata(item_id, {
                "extraction": {
                    **result.model_dump(),
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
            })
            staging.update_status([item_id], "extracted")
            extracted += 1
            log.info("Extracted %d entities, %d rels from %s",
                     len(result.entities), len(result.relationships), item_id)
        else:
            staging.update_status([item_id], "failed")
            failed += 1

    log.info("Extraction batch: %d extracted, %d failed, %d skipped", extracted, failed, skipped)
    return {"extracted": extracted, "failed": failed, "skipped": skipped}
