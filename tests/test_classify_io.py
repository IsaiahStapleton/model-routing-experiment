from router.classify import classify


def _ok_response(score="6"):
    return {"choices": [{"message": {"content": score}}]}


def test_sends_required_fields():
    captured = {}

    def fake(url, body, headers, timeout):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _ok_response()

    score = classify("hello", ["bash"], "http://x/v1", "k", transport=fake)

    assert score == 6
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["body"]["model"] == "qwen3-1.7b"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 8
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["timeout"] == 2.0


def test_returns_none_on_transport_exception():
    def boom(url, body, headers, timeout):
        raise TimeoutError("too slow")

    assert classify("hi", [], "http://x/v1", "k", transport=boom) is None


def test_returns_none_on_malformed_response():
    def weird(url, body, headers, timeout):
        return {"unexpected": True}

    assert classify("hi", [], "http://x/v1", "k", transport=weird) is None


def test_returns_none_on_unparseable_score():
    def prose(url, body, headers, timeout):
        return _ok_response("I cannot rate this")

    assert classify("hi", [], "http://x/v1", "k", transport=prose) is None
