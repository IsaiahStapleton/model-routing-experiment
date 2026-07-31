import json

from eval.dual_run import already_done, answer_budget, build_body, tier_can_serve


def test_small_tier_rejects_oversized_prompt():
    assert not tier_can_serve("qwen3-1.7b", 4000)
    assert tier_can_serve("qwen3-1.7b", 1000)


def test_large_tiers_accept_moderate_prompts():
    assert tier_can_serve("qwen3-30b-a3b", 20000)
    assert tier_can_serve("gpt-oss-120b", 20000)
    assert not tier_can_serve("gpt-oss-120b", 40000)


def test_already_done_reads_existing_pairs(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps({"id": "a", "tier": "qwen3-1.7b"}) + "\n"
        + json.dumps({"id": "b", "tier": "gpt-oss-120b"}) + "\n"
    )
    done = already_done(str(path))
    assert ("a", "qwen3-1.7b") in done
    assert ("b", "gpt-oss-120b") in done
    assert ("a", "gpt-oss-120b") not in done


def test_already_done_handles_missing_file(tmp_path):
    assert already_done(str(tmp_path / "nope.jsonl")) == set()


def test_build_body_disables_thinking():
    body = build_body("qwen3-30b-a3b", "hello", 500)
    assert body["model"] == "qwen3-30b-a3b"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_tokens"] == 4000
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_answer_budget_respects_small_tier_context():
    # 4096 context, so a 97-token prompt cannot also take 4000 output
    b = answer_budget("qwen3-1.7b", 97)
    assert b <= 4096 - 97 - 128
    assert b >= 256


def test_answer_budget_uses_full_budget_on_large_tiers():
    assert answer_budget("gpt-oss-120b", 500) == 4000
    assert answer_budget("qwen3-30b-a3b", 500) == 4000


def test_answer_budget_never_returns_below_minimum():
    # even an absurd prompt yields a usable floor rather than zero or negative
    assert answer_budget("qwen3-1.7b", 5000) == 256
