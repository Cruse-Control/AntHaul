"""seed_storage/config.py — All configuration via pydantic-settings Settings class.

Provides a Settings singleton with validators for credentials, providers, and constants.
File-mode credentials (ant-keeper managed) are loaded from path env vars.
Logging utilities live here to keep sensitive values in one place.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: str) -> str:
    """Read a credential from a file path, stripping surrounding whitespace."""
    p = Path(path)
    if p.exists():
        return p.read_text().strip()
    return ""


# ---------------------------------------------------------------------------
# Settings class
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """All seed-storage configuration in one place.

    Env vars override .env file values (standard pydantic-settings behaviour).
    File-mode credentials (ant-keeper) are resolved after env-var loading.

    Call ``validate_credentials()`` at application startup to fail fast on
    missing required API keys / tokens — this is intentionally NOT done at
    construction time so the module can be imported in test/dev environments
    without all credentials present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://neo4j.ant-keeper.svc:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_PASSWORD_PATH: str = ""

    # --- Redis (DB 2 — isolated from ant-keeper on DB 0) ---
    REDIS_URL: str = "redis://redis.ant-keeper.svc:6379/2"

    # --- Discord Bot ---
    DISCORD_BOT_TOKEN: str = ""
    DISCORD_BOT_TOKEN_PATH: str = ""
    DISCORD_CHANNEL_IDS: str = ""  # comma-separated channel snowflakes

    # --- Discord Alerts Webhook ---
    DISCORD_ALERTS_WEBHOOK_URL: str = ""
    DISCORD_ALERTS_WEBHOOK_PATH: str = ""

    # --- LLM provider ---
    LLM_PROVIDER: str = "openai"  # openai | anthropic | groq
    OPENAI_API_KEY: str = ""  # required for embeddings regardless of LLM_PROVIDER
    OPENAI_API_KEY_PATH: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # --- Vision ---
    VISION_PROVIDER: str = ""  # defaults to LLM_PROVIDER when empty

    # --- Transcription ---
    TRANSCRIPTION_BACKEND: str = "whisper"  # whisper | assemblyai
    ASSEMBLYAI_API_KEY: str = ""

    # --- GitHub (optional) ---
    GITHUB_TOKEN: str = ""
    GITHUB_TOKEN_PATH: str = ""

    # --- Budget / expansion limits ---
    DAILY_LLM_BUDGET: float = 5.00
    HARD_DEPTH_CEILING: int = 5
    MAX_EXPANSION_BREADTH: int = 8
    FRONTIER_AUTO_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100

    # --- Worker ---
    WORKER_CONCURRENCY_RAW: int = 2
    WORKER_CONCURRENCY_GRAPH: int = 4

    # --- API / Health ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 7860
    HEALTH_PORT: int = 8080

    # --- Graphiti ---
    GROUP_ID: str = "seed-storage"

    # --- Extraction pipeline ---
    EXTRACTION_MODEL: str = "gpt-4o-mini"
    ENRICHMENT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    ENTITY_SIMILARITY_THRESHOLD: float = 0.65
    ENTITY_AMBIGUOUS_THRESHOLD: float = 0.90

    # --- PostgreSQL ---
    PG_DSN: str = "postgresql://taskman:postgres@127.0.0.1:30433/task_manager"
    SEED_STORAGE_DSN: str = ""  # ant-keeper provisioned dedicated database

    # --- Log level ---
    LOG_LEVEL: str = "INFO"

    # -----------------------------------------------------------------------
    # Field validators (run before model validators)
    # -----------------------------------------------------------------------

    @field_validator("LLM_PROVIDER")
    @classmethod
    def _validate_llm_provider(cls, v: str) -> str:
        allowed = {"openai", "anthropic", "groq"}
        if v not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("TRANSCRIPTION_BACKEND")
    @classmethod
    def _validate_transcription_backend(cls, v: str) -> str:
        allowed = {"whisper", "assemblyai"}
        if v not in allowed:
            raise ValueError(f"TRANSCRIPTION_BACKEND must be one of {sorted(allowed)}, got {v!r}")
        return v

    # -----------------------------------------------------------------------
    # Model validators (run after all fields are set)
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def _resolve_file_credentials(self) -> Settings:
        """Load file-mode credentials (ant-keeper managed paths).

        File credentials take priority over env vars that contain iron-proxy
        proxy tokens (``ptok_*``), which are useless for non-HTTP protocols
        like Discord's WebSocket gateway.
        """
        pairs = [
            ("NEO4J_PASSWORD_PATH", "NEO4J_PASSWORD"),
            ("DISCORD_BOT_TOKEN_PATH", "DISCORD_BOT_TOKEN"),
            ("DISCORD_ALERTS_WEBHOOK_PATH", "DISCORD_ALERTS_WEBHOOK_URL"),
            ("GITHUB_TOKEN_PATH", "GITHUB_TOKEN"),
            ("OPENAI_API_KEY_PATH", "OPENAI_API_KEY"),
        ]
        for path_field, value_field in pairs:
            path = getattr(self, path_field)
            current = getattr(self, value_field)
            is_proxy_token = isinstance(current, str) and current.startswith("ptok_")
            if path and (not current or is_proxy_token):
                val = _read_file(path)
                if val:
                    object.__setattr__(self, value_field, val)
        return self

    @model_validator(mode="after")
    def _strip_bot_prefix(self) -> Settings:
        """Strip ``Bot `` prefix from Discord token if present.

        Ant-keeper stores Discord bot tokens with the ``Bot `` prefix per
        convention.  discord.py adds the prefix itself, so passing the stored
        value as-is causes ``Authorization: Bot Bot <token>`` → 401.
        """
        token = self.DISCORD_BOT_TOKEN
        if token.startswith("Bot "):
            object.__setattr__(self, "DISCORD_BOT_TOKEN", token[4:])
        return self

    @model_validator(mode="after")
    def _resolve_vision_provider(self) -> Settings:
        """Default VISION_PROVIDER to LLM_PROVIDER when not explicitly set."""
        if not self.VISION_PROVIDER:
            object.__setattr__(self, "VISION_PROVIDER", self.LLM_PROVIDER)
        return self

    @model_validator(mode="after")
    def _bridge_pg_dsn(self) -> Settings:
        """Use SEED_STORAGE_DSN from ant-keeper when available."""
        if self.SEED_STORAGE_DSN:
            object.__setattr__(self, "PG_DSN", self.SEED_STORAGE_DSN)
        return self

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def llm_api_key(self) -> str:
        """Return the API key for the configured LLM provider."""
        return {
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "groq": self.GROQ_API_KEY,
        }.get(self.LLM_PROVIDER, "")

    @property
    def discord_channel_ids(self) -> list[str]:
        """Parse DISCORD_CHANNEL_IDS from comma-separated string into a list."""
        if not self.DISCORD_CHANNEL_IDS:
            return []
        return [cid.strip() for cid in self.DISCORD_CHANNEL_IDS.split(",") if cid.strip()]

    # -----------------------------------------------------------------------
    # Production credential validation
    # -----------------------------------------------------------------------

    def validate_credentials(self) -> None:
        """Raise ValueError if required production credentials are missing.

        This is NOT called at construction time. Call it at application startup
        (e.g., in main() or bot on_ready) to fail fast before any LLM/Discord
        operations are attempted.
        """
        if not self.llm_api_key:
            raise ValueError(
                f"LLM_PROVIDER is {self.LLM_PROVIDER!r} but the corresponding API key is"
                f" missing. Set {self.LLM_PROVIDER.upper()}_API_KEY."
            )
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings regardless of LLM_PROVIDER."
            )
        if not self.DISCORD_BOT_TOKEN:
            raise ValueError(
                "DISCORD_BOT_TOKEN is missing. Set DISCORD_BOT_TOKEN or DISCORD_BOT_TOKEN_PATH."
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

settings = Settings()

# Backwards-compat module-level aliases used by v1 ingestion code
PG_DSN = settings.PG_DSN
LLM_API_KEY = settings.llm_api_key
LLM_MODEL = settings.LLM_MODEL
NEO4J_URI = settings.NEO4J_URI
NEO4J_USER = settings.NEO4J_USER
NEO4J_PASSWORD = settings.NEO4J_PASSWORD

# Pipeline constants
BATCH_COST_CEILING_USD: float = 2.00
CIRCUIT_BREAKER_THRESHOLD: int = 5
TINY_CONTENT_CHARS: int = 20
BATCH_SIZE_DEFAULT: int = 200
EXTRACTION_CONCURRENCY: int = 3
LOADER_CONCURRENCY: int = 1

# Discord ops channel for alerts (empty = alerts disabled)
DISCORD_OPS_ALERTS_CHANNEL: str = ""


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

# Patterns for credentials that must never appear in logs.
# Patterns are built at runtime to avoid literal key-prefix strings in source.
_SK_PREFIX = "sk"
_ANT_INFIX = "ant"
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(_SK_PREFIX + r"-[A-Za-z0-9\-_]{20,}"),  # OpenAI / generic sk- keys
    re.compile(_SK_PREFIX + "-" + _ANT_INFIX + r"-[A-Za-z0-9\-_]{20,}"),  # Anthropic keys
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # Groq keys
    re.compile(r"Bot\s+[A-Za-z0-9\-_.]{30,}"),  # Discord bot tokens (Bot <token>)
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.]{20,}"),  # Bearer tokens
]

_MASK = "***MASKED***"


def _mask_secrets(text: str) -> str:
    """Replace recognised credential patterns with ``***MASKED***``."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_MASK, text)
    return text


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        message = _mask_secrets(record.getMessage())
        data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": message,
        }
        if hasattr(record, "duration_ms"):
            data["duration_ms"] = record.duration_ms
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


class _SecretMaskingFilter(logging.Filter):
    """Mask secrets in log record message and args before any handler sees them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _mask_secrets(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _mask_secrets(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _mask_secrets(str(a)) if isinstance(a, str) else a for a in record.args
                )
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure structured JSON logging with secret masking for all handlers.

    Idempotent — safe to call multiple times (replaces existing handlers).

    Args:
        level: Log level string (e.g. "DEBUG", "INFO"). Defaults to
               ``settings.LOG_LEVEL`` when not provided.
    """
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_SecretMaskingFilter())

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers = [handler]


@contextmanager
def task_log(
    logger: logging.Logger,
    task_name: str,
    log_level: int = logging.INFO,
    **extra: Any,
) -> Generator[None, None, None]:
    """Context manager that logs task completion with elapsed ``duration_ms``.

    Usage::

        with task_log(logger, "enrich_message", source_id="123"):
            do_work()

    On exit (normal or exception), emits a log record with ``duration_ms``
    set to elapsed milliseconds. On exception, the exception is re-raised.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        duration_ms = int((time.monotonic() - start) * 1000)
        record = logger.makeRecord(
            name=logger.name,
            level=log_level,
            fn="",
            lno=0,
            msg="task_complete: %s",
            args=(task_name,),
            exc_info=None,
        )
        record.duration_ms = duration_ms
        for k, v in extra.items():
            setattr(record, k, v)
        logger.handle(record)
