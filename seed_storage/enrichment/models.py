from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

ContentType = Literal[
    "webpage", "youtube", "video", "image", "pdf", "github", "tweet", "instagram", "unknown"
]

_CONTENT_TYPES = {
    "webpage",
    "youtube",
    "video",
    "image",
    "pdf",
    "github",
    "tweet",
    "instagram",
    "unknown",
}


@dataclass
class ResolvedContent:
    source_url: str
    content_type: ContentType
    title: str | None
    text: str  # clean extracted text; empty string on failure
    transcript: str | None  # for video/audio content
    summary: str | None  # populated by vision LLM for images
    expansion_urls: list[str]  # secondary URLs found within this content
    metadata: dict[str, Any]  # source-specific extras
    extraction_error: str | None  # None on success, error message on failure
    resolved_at: datetime  # UTC, set by dispatcher after resolution completes

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict. datetime → ISO 8601 string."""
        return {
            "source_url": self.source_url,
            "content_type": self.content_type,
            "title": self.title,
            "text": self.text,
            "transcript": self.transcript,
            "summary": self.summary,
            "expansion_urls": list(self.expansion_urls),
            "metadata": dict(self.metadata),
            "extraction_error": self.extraction_error,
            "resolved_at": self.resolved_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedContent":
        """Deserialize from dict. Ignores unknown keys (forward compatibility)."""
        resolved_at_raw = data["resolved_at"]
        if isinstance(resolved_at_raw, datetime):
            resolved_at = resolved_at_raw
        else:
            resolved_at = datetime.fromisoformat(resolved_at_raw)

        content_type = data.get("content_type", "unknown")
        if content_type not in _CONTENT_TYPES:
            content_type = "unknown"

        return cls(
            source_url=data["source_url"],
            content_type=content_type,
            title=data.get("title"),
            text=data.get("text", ""),
            transcript=data.get("transcript"),
            summary=data.get("summary"),
            expansion_urls=list(data.get("expansion_urls", [])),
            metadata=dict(data.get("metadata", {})),
            extraction_error=data.get("extraction_error"),
            resolved_at=resolved_at,
        )

    @classmethod
    def error_result(cls, url: str, error: str) -> "ResolvedContent":
        """Factory for failed resolutions. text='', extraction_error=error, resolved_at=utcnow()."""
        return cls(
            source_url=url,
            content_type="unknown",
            title=None,
            text="",
            transcript=None,
            summary=None,
            expansion_urls=[],
            metadata={},
            extraction_error=error,
            resolved_at=datetime.now(tz=UTC),
        )
