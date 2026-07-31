import json

from eval.dual_run import already_done, tier_can_serve


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
