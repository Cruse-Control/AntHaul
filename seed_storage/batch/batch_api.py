"""OpenAI Batch API integration -- 50% cost savings for offline extraction + embeddings.

Workflow: build JSONL -> upload -> create batch -> poll -> parse results.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openai import OpenAI

from seed_storage import staging
from seed_storage.config import settings
from seed_storage.extraction import EXTRACTION_SCHEMA, _build_system_prompt
from seed_storage.preseed import get_alias_map

log = logging.getLogger("batch.api")


def build_extraction_jsonl(items: list[dict], output_path: Path) -> int:
    """Build JSONL file for batch extraction requests. Returns request count."""
    alias_map = get_alias_map()
    count = 0
    with open(output_path, "w") as f:
        for item in items:
            item_id = str(item["id"])
            content = item.get("raw_content", "") or ""
            source_type = item.get("source_type", "unknown")
            meta = item.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)

            system_prompt = _build_system_prompt(source_type, alias_map)

            # Build user content with enrichment context
            user_content = f"Source type: {source_type}\n"
            tags = meta.get("tags", [])
            summary = meta.get("summary", "")
            if tags and tags != ["uncategorized"]:
                user_content += f"Tags: {', '.join(tags)}\n"
            if summary:
                user_content += f"Summary: {summary}\n"
            user_content += f"Content:\n{content[:8000]}"

            request = {
                "custom_id": item_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": settings.EXTRACTION_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": EXTRACTION_SCHEMA,
                    },
                    "temperature": 0.1,
                },
            }
            f.write(json.dumps(request) + "\n")
            count += 1
    return count


def submit_batch(jsonl_path: Path, client: OpenAI | None = None) -> str:
    """Upload JSONL and create batch job. Returns OpenAI batch_id."""
    if client is None:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    log.info("Batch created: %s (file: %s)", batch.id, file_obj.id)
    return batch.id


def poll_batch(
    batch_id: str,
    client: OpenAI | None = None,
    poll_interval: int = 60,
    max_polls: int = 1440,
) -> dict:
    """Poll batch until complete or failed. Returns batch status dict."""
    if client is None:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    for _ in range(max_polls):
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        log.info("Batch %s: status=%s", batch_id, status)

        if status in ("completed", "failed", "expired", "cancelled"):
            return {
                "status": status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "request_counts": {
                    "total": batch.request_counts.total if batch.request_counts else 0,
                    "completed": batch.request_counts.completed if batch.request_counts else 0,
                    "failed": batch.request_counts.failed if batch.request_counts else 0,
                },
            }
        time.sleep(poll_interval)

    return {"status": "timeout", "batch_id": batch_id}


def download_results(output_file_id: str, client: OpenAI | None = None) -> list[dict]:
    """Download and parse batch results JSONL. Returns list of {custom_id, result}."""
    if client is None:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    content = client.files.content(output_file_id)
    results = []
    for line in content.text.strip().split("\n"):
        if line.strip():
            data = json.loads(line)
            custom_id = data.get("custom_id")
            response_body = data.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if choices:
                message_content = choices[0].get("message", {}).get("content", "{}")
                results.append({
                    "custom_id": custom_id,
                    "result": json.loads(message_content),
                    "usage": response_body.get("usage", {}),
                })
    return results


def apply_batch_results(results: list[dict]) -> dict:
    """Apply parsed batch results to staging items.

    For each result, stores extraction in metadata and updates status to 'extracted'.
    Returns counts of {applied, failed, skipped}.
    """
    from datetime import datetime, timezone
    from seed_storage.extraction import _parse_extraction

    applied = 0
    failed = 0
    skipped = 0

    for r in results:
        item_id = r["custom_id"]
        raw = r.get("result", {})
        usage = r.get("usage", {})

        if not raw:
            skipped += 1
            continue

        try:
            extraction = _parse_extraction(
                raw,
                model_used=f"{settings.EXTRACTION_MODEL}-batch",
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
            staging.patch_metadata(item_id, {
                "extraction": {
                    **extraction.model_dump(),
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "batch_api": True,
                },
            })
            staging.update_status([item_id], "extracted")
            applied += 1
        except Exception as exc:
            log.warning("Failed to apply batch result for %s: %s", item_id, exc)
            staging.update_status([item_id], "failed")
            staging.patch_metadata(item_id, {"extraction_error": str(exc)[:500]})
            failed += 1

    log.info("Batch results applied: %d applied, %d failed, %d skipped",
             applied, failed, skipped)
    return {"applied": applied, "failed": failed, "skipped": skipped}
