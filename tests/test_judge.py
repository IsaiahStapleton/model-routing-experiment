from eval.judge import build_judge_prompt, parse_verdict


def test_prompt_contains_task_and_answer():
    p = build_judge_prompt("What is 2+2?", "4")
    assert "What is 2+2?" in p
    assert "4" in p
    assert "ADEQUATE" in p


def test_parse_adequate():
    assert parse_verdict("ADEQUATE") is True
    assert parse_verdict("  adequate  ") is True


def test_parse_inadequate():
    assert parse_verdict("INADEQUATE") is False
    assert parse_verdict("inadequate, it misses the point") is False


def test_inadequate_wins_when_both_appear():
    # "INADEQUATE" contains "ADEQUATE" as a substring; must not misread it
    assert parse_verdict("The answer is INADEQUATE") is False


def test_parse_unknown_returns_none():
    assert parse_verdict("") is None
    assert parse_verdict("I am not sure") is None


def test_empty_answer_is_never_adequate():
    # an empty answer cannot be adequate regardless of what the judge says
    from eval.judge import prejudge
    assert prejudge("") is False
    assert prejudge("   ") is False
    assert prejudge("a real answer") is None


def test_ask_judge_retries_on_transient_failure(monkeypatch):
    # A fake transport that fails twice (transient errors) then succeeds.
    # ask_judge must retry through the failures and return the real verdict,
    # not give up and return None. No network calls: the transport is a
    # plain injected callable, and time.sleep is stubbed so the test is fast.
    import eval.judge as judge

    monkeypatch.setattr(judge.time, "sleep", lambda seconds: None)

    calls = {"n": 0}

    def flaky_transport(task, answer, base, key):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient failure")
        return "ADEQUATE"

    result = judge.ask_judge("task", "answer", "base", "key", request=flaky_transport)

    assert result is True
    assert calls["n"] == 3
