import json
import os

from eval.corpus import (
    estimate_tokens,
    extract_real_turns,
    is_self_contained,
    load_synthetic,
)

SYNTH = os.path.expanduser("~/isaiah-routing-lab/eval/synthetic.json")


def test_synthetic_loads_and_is_well_formed():
    items = load_synthetic(SYNTH)
    assert len(items) >= 30
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for item in items:
        assert item["user_msg"].strip()
        assert isinstance(item["tool_names"], list)
        assert item["prompt_tokens"] > 0
        assert item["source"] == "synthetic"


def test_estimate_tokens_is_roughly_chars_over_four():
    assert estimate_tokens("a" * 400) == 100


def test_self_contained_accepts_standalone_question():
    assert is_self_contained("What is the capital of Japan?")


def test_self_contained_rejects_back_references():
    assert not is_self_contained("now do the same for the other file")
    assert not is_self_contained("continue")
    assert not is_self_contained("what did we decide about that?")
    assert not is_self_contained("fix it")


def test_self_contained_rejects_too_short():
    assert not is_self_contained("ok")


def test_extract_real_turns_pulls_user_messages():
    dump = "\n".join([
        json.dumps({"message": {"role": "user", "content": "deploy the operator to staging please"}}),
        json.dumps({"message": {"role": "assistant", "content": "done"}}),
        json.dumps({"not_a_message": True}),
        "this line is not json",
    ])
    turns = extract_real_turns(dump)
    assert len(turns) == 1
    assert turns[0]["user_msg"] == "deploy the operator to staging please"
    assert turns[0]["source"] == "real"
    assert turns[0]["prompt_tokens"] > 0
