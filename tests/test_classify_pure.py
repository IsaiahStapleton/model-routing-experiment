from router.classify import build_prompt, parse_score


def test_prompt_includes_task_and_tool_names():
    p = build_prompt("check why the deploy failed", ["read_file", "bash"])
    assert "check why the deploy failed" in p
    assert "read_file" in p and "bash" in p


def test_prompt_handles_no_tools():
    p = build_prompt("what is 2+2", [])
    assert "2+2" in p
    assert "none" in p.lower()


def test_prompt_truncates_very_long_task():
    # must stay well inside the 1.7b 4096 context
    p = build_prompt("x" * 100000, [])
    assert len(p) < 8000


def test_parse_plain_integer():
    assert parse_score({"content": "7"}) == 7


def test_parse_integer_with_surrounding_prose():
    assert parse_score({"content": "Score: 7 out of 10"}) == 7


def test_parse_falls_back_to_reasoning_content():
    # thinking models can leave content empty
    assert parse_score({"content": "", "reasoning_content": "I think 4"}) == 4


def test_parse_prefers_content_over_reasoning():
    assert parse_score({"content": "9", "reasoning_content": "2"}) == 9


def test_parse_rejects_out_of_range():
    assert parse_score({"content": "42"}) is None
    assert parse_score({"content": "0"}) is None


def test_parse_rejects_garbage():
    assert parse_score({"content": "banana"}) is None


def test_parse_rejects_negative_numbers():
    assert parse_score({"content": "-5"}) is None
    assert parse_score({"content": "I'd say -3"}) is None


def test_parse_handles_empty_and_missing():
    assert parse_score({"content": ""}) is None
    assert parse_score({}) is None
    assert parse_score({"content": None}) is None


def test_parse_word_categories():
    assert parse_score({"content": "EASY"}) == 2
    assert parse_score({"content": "MEDIUM"}) == 5
    assert parse_score({"content": "HARD"}) == 9


def test_parse_word_is_case_insensitive():
    assert parse_score({"content": "medium"}) == 5


def test_parse_word_wins_over_stray_number():
    # a reply like "HARD (9/10)" must not be read as some other number
    assert parse_score({"content": "HARD (9/10)"}) == 9


def test_numeric_fallback_still_works():
    assert parse_score({"content": "7"}) == 7
    assert parse_score({"content": "-5"}) is None
    assert parse_score({"content": "42"}) is None


def test_prompt_asks_for_category():
    p = build_prompt("what is 2+2", ["bash"])
    assert "EASY" in p and "MEDIUM" in p and "HARD" in p
    assert "bash" in p
    assert "what is 2+2" in p
