import pytest
from router.policy import decide, Decision, TIER_ORDER, TIER_CONTEXT, DEFAULT_TIER


@pytest.mark.parametrize("score,expected_tier", [
    (1, "qwen3-1.7b"),
    (3, "qwen3-1.7b"),
    (4, "qwen3-30b-a3b"),
    (7, "qwen3-30b-a3b"),
    (8, "gpt-oss-120b"),
    (10, "gpt-oss-120b"),
])
def test_score_maps_to_tier(score, expected_tier):
    d = decide(score=score, prompt_tokens=100)
    assert d.tier == expected_tier
    assert d.reason == "score"


def test_failed_classification_uses_default_tier():
    d = decide(score=None, prompt_tokens=100)
    assert d.tier == DEFAULT_TIER
    assert d.reason == "guardrail_error"


def test_context_guardrail_promotes_off_small_tier():
    # score says cheapest, but 4000 + 1024 headroom exceeds the 1.7b 4096 limit
    d = decide(score=1, prompt_tokens=4000)
    assert d.tier == "qwen3-30b-a3b"
    assert d.reason == "guardrail_context"


def test_context_guardrail_does_not_fire_when_it_fits():
    # 3000 + 1024 = 4024, just under the 4096 limit
    d = decide(score=1, prompt_tokens=3000)
    assert d.tier == "qwen3-1.7b"
    assert d.reason == "score"


def test_context_guardrail_applies_to_failed_classification_too():
    d = decide(score=None, prompt_tokens=40000)
    # nothing fits 40k; must not silently pick a tier that will 400
    assert d.reason == "guardrail_context"
    assert d.tier == TIER_ORDER[-1]


def test_thresholds_are_tunable():
    # widen the cheap band so 5 routes to the 1.7b
    d = decide(score=5, prompt_tokens=100, thresholds=(6, 8))
    assert d.tier == "qwen3-1.7b"


def test_decision_is_immutable():
    d = decide(score=5, prompt_tokens=100)
    with pytest.raises(Exception):
        d.tier = "other"


def test_tier_order_is_cheapest_first():
    assert TIER_ORDER == ["qwen3-1.7b", "qwen3-30b-a3b", "gpt-oss-120b"]
    assert TIER_CONTEXT["qwen3-1.7b"] == 4096
