"""seed_storage/expansion/policies.py — Per-resolver depth policies and priority scoring.

Priority is a float score used to order the frontier (higher = more urgent).
It is computed from a base value adjusted by depth penalty, resolver bonus,
domain bonus, and channel bonus, then clamped to [0.0, ∞).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Depth policies — maximum hop depth allowed per resolver type
# ---------------------------------------------------------------------------

#: Maps resolver_hint → maximum allowed expansion depth.
#: URLs whose depth exceeds the policy for their resolver are skipped by pick_top.
#: "default" is the fallback when a resolver_hint is not in the dict.
DEPTH_POLICIES: dict[str, int] = {
    "youtube": 3,
    "github": 3,
    "pdf": 2,
    "image": 2,
    "video": 2,
    "webpage": 4,
    "tweet": 1,
    "twitter": 1,
    "unknown": 2,
    "fallback": 1,
    "default": 3,
}

# ---------------------------------------------------------------------------
# Priority scoring constants
# ---------------------------------------------------------------------------

#: Base priority assigned to newly discovered URLs.
BASE_PRIORITY: float = 0.5

#: Score deducted per hop depth level.
DEPTH_PENALTY_PER_LEVEL: float = 0.15

#: Additive bonus per resolver type. Negative values reduce urgency.
RESOLVER_BONUS: dict[str, float] = {
    "youtube": 0.2,
    "github": 0.2,
    "pdf": 0.1,
    "image": 0.0,
    "video": 0.1,
    "webpage": 0.0,
    "tweet": 0.1,
    "twitter": 0.1,
    "unknown": 0.0,
    "fallback": -0.1,
}

#: Additive bonus for high-value domains.
DOMAIN_BONUS: dict[str, float] = {
    "github.com": 0.2,
    "youtube.com": 0.2,
    "youtu.be": 0.2,
    "arxiv.org": 0.3,
    "papers.ssrn.com": 0.2,
    "docs.python.org": 0.1,
}


# ---------------------------------------------------------------------------
# Priority computation
# ---------------------------------------------------------------------------


def compute_priority(
    *,
    depth: int = 0,
    resolver_hint: str = "unknown",
    domain: str = "",
    source_channel: str = "",
    channel_bonuses: dict[str, float] | None = None,
    base_priority: float = BASE_PRIORITY,
) -> float:
    """Compute priority score for a frontier URL.

    Components applied in order:
    1. Start from *base_priority* (default 0.5).
    2. Subtract ``DEPTH_PENALTY_PER_LEVEL * depth``.
    3. Add resolver bonus for the expected content type.
    4. Add domain bonus for high-value domains.
    5. Add channel bonus for high-value source channels (caller-supplied).
    6. Clamp result to ``[0.0, ∞)`` — score is never negative.

    Args:
        depth:           Hop depth from the original message (0 = direct link).
        resolver_hint:   Expected resolver type (e.g. "youtube", "github").
        domain:          Registered domain of the URL (e.g. "github.com").
        source_channel:  Source channel name for channel bonus lookup.
        channel_bonuses: Optional dict mapping channel name → additive bonus.
        base_priority:   Starting score before any adjustments.

    Returns:
        float ≥ 0.0 — higher means the URL should be expanded sooner.
    """
    score = base_priority
    score -= DEPTH_PENALTY_PER_LEVEL * depth
    score += RESOLVER_BONUS.get(resolver_hint, 0.0)
    score += DOMAIN_BONUS.get(domain, 0.0)
    if channel_bonuses:
        score += channel_bonuses.get(source_channel, 0.0)
    return max(0.0, score)
