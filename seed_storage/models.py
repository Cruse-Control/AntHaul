"""Shared types for the seed-storage extraction pipeline."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

EntityType = Literal["Person", "Organization", "Product", "Concept", "Location", "Event"]

class ExtractedEntity(BaseModel):
    name: str
    canonical_name: str          # lowercase, stripped
    entity_type: EntityType
    description: str
    aliases: list[str] = []
    confidence: float = 0.8

class ExtractedRelationship(BaseModel):
    source: str                   # entity canonical_name
    target: str                   # entity canonical_name
    relationship_type: str        # WORKS_FOR, DISCUSSES, etc.
    description: str
    confidence: float = 0.8

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
    model_used: str
    tokens_input: int
    tokens_output: int

# Pipeline status constants
PIPELINE_STATUSES = ["staged", "processed", "enriched", "extracted", "loaded"]
TERMINAL_STATUSES = ["failed", "rejected", "deduped"]
TRANSIENT_STATUSES = ["loading", "extracting"]
