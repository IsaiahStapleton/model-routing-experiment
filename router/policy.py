"""Tier selection policy. Pure functions only, no I/O.

The live router and the offline threshold sweep both import decide(), so a
threshold chosen from a sweep graph is literally the one running in production.
"""
from dataclasses import dataclass

TIER_ORDER = ["qwen3-1.7b", "qwen3-30b-a3b", "gpt-oss-120b"]

TIER_CONTEXT = {
    "qwen3-1.7b": 4096,
    "qwen3-30b-a3b": 32768,
    "gpt-oss-120b": 32768,
}

DEFAULT_TIER = "qwen3-30b-a3b"

# Reserved so a prompt that only just fits does not fail during generation.
OUTPUT_HEADROOM = 1024


@dataclass(frozen=True)
class Decision:
    tier: str
    reason: str


def _fits(tier: str, prompt_tokens: int) -> bool:
    return prompt_tokens + OUTPUT_HEADROOM <= TIER_CONTEXT[tier]


def _tier_for_score(score: int, thresholds: tuple[int, int]) -> str:
    cheap_max, mid_max = thresholds
    if score <= cheap_max:
        return TIER_ORDER[0]
    if score <= mid_max:
        return TIER_ORDER[1]
    return TIER_ORDER[2]


def decide(
    score: int | None,
    prompt_tokens: int,
    thresholds: tuple[int, int] = (3, 7),
) -> Decision:
    """Choose a serving tier.

    score is None when classification failed; we fail safe to DEFAULT_TIER
    rather than guessing. The context guardrail then overrides any choice that
    cannot physically hold the prompt.
    """
    if score is None:
        tier, reason = DEFAULT_TIER, "guardrail_error"
    else:
        tier, reason = _tier_for_score(score, thresholds), "score"

    if not _fits(tier, prompt_tokens):
        for candidate in TIER_ORDER:
            if _fits(candidate, prompt_tokens):
                return Decision(tier=candidate, reason="guardrail_context")
        # Nothing fits. Return the largest tier and let it report the error
        # honestly rather than pretending we made a valid choice.
        return Decision(tier=TIER_ORDER[-1], reason="guardrail_context")

    return Decision(tier=tier, reason=reason)
