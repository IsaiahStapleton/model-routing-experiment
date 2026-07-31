"""Real agent turns must never be routed somewhere their prompt cannot fit.

This needs no judge: it is a behavioural invariant with a deterministic answer.

Correction: eval/corpus.py records prompt_tokens from the user message only.
The largest guardrail item is ~2586 tokens, well under the 1.7b's 4096 context,
so a test that used prompt_tokens as-is would never see a violation and the
key assertion below would run vacuously. A real podling request is much
larger than its user message: the system prompt, identity files, and tool
schemas add a large fixed overhead before the user's words are even counted.
OPENCLAW_REQUEST_OVERHEAD models that so the guardrail is actually exercised
against the request size the serving model would receive.
"""
import json
import os

import pytest

from router.policy import OUTPUT_HEADROOM, TIER_CONTEXT, decide

GUARDRAIL = os.path.expanduser("~/isaiah-routing-lab/eval/out/guardrail.jsonl")

# Measured on podling 2026-07-30: system prompt + identity files + tool
# schemas total ~23,493 tokens before any user message. The corpus records
# only the user message, so add this to model the request the serving
# model would actually receive.
OPENCLAW_REQUEST_OVERHEAD = 23493


def _items():
    if not os.path.exists(GUARDRAIL):
        return []
    with open(GUARDRAIL) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _effective_prompt_tokens(item: dict) -> int:
    return item["prompt_tokens"] + OPENCLAW_REQUEST_OVERHEAD


@pytest.mark.skipif(not _items(), reason="guardrail corpus not built")
@pytest.mark.parametrize("score", [None, 1, 5, 10])
def test_chosen_tier_can_always_hold_the_prompt(score):
    violations = []
    for item in _items():
        effective = _effective_prompt_tokens(item)
        decision = decide(score, effective)
        limit = TIER_CONTEXT[decision.tier]
        if effective + OUTPUT_HEADROOM > limit:
            # Only acceptable if no tier could hold it at all.
            if any(effective + OUTPUT_HEADROOM <= c
                   for c in TIER_CONTEXT.values()):
                violations.append((item["id"], effective, decision.tier))
    assert not violations, f"guardrail failed for: {violations[:5]}"


@pytest.mark.skipif(not _items(), reason="guardrail corpus not built")
def test_large_real_turns_never_reach_the_small_tier():
    big = [i for i in _items() if _effective_prompt_tokens(i) > 3072]
    if not big:
        pytest.skip("no large real turns in corpus")
    for item in big:
        assert decide(1, _effective_prompt_tokens(item)).tier != "qwen3-1.7b"
