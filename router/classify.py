"""Difficulty classification via qwen3-1.7b.

build_prompt and parse_score are pure and unit-tested. Only classify() does I/O,
and it takes its transport as an argument so tests need no HTTP mocking.
"""
import json
import re
import urllib.error
import urllib.request

CLASSIFIER_MODEL = "qwen3-1.7b"

# The 1.7b has a 4096 context. Cap the task text well below it so the
# instructions and tool list always survive.
MAX_TASK_CHARS = 6000

_INSTRUCTIONS = (
    "Rate how difficult this task is for a language model, from 1 to 10.\n"
    "1 means trivial (arithmetic, greetings, simple lookups).\n"
    "5 means moderate (short code, summarising, routine explanation).\n"
    "10 means very hard (multi-step reasoning, subtle debugging, deep analysis).\n"
    "Answer with the number only, no words."
)


def build_prompt(user_msg: str, tool_names: list[str]) -> str:
    task = (user_msg or "")[:MAX_TASK_CHARS]
    tools = ", ".join(tool_names) if tool_names else "none"
    return f"TASK: {task}\nAVAILABLE TOOLS: {tools}\n\n{_INSTRUCTIONS}"


def parse_score(message: dict) -> int | None:
    """Extract a 1-10 score, tolerating prose and empty content.

    Thinking-capable models may return an empty content field with the answer
    in reasoning_content, so fall back to it rather than reporting failure.
    """
    if not isinstance(message, dict):
        return None
    for field in ("content", "reasoning_content"):
        text = message.get(field)
        if not text:
            continue
        match = re.search(r"(?<!-)\b(10|[1-9])\b", str(text))
        if match:
            return int(match.group(1))
    return None


def post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """Default transport. Separated so tests can inject a fake."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def classify(
    user_msg: str,
    tool_names: list[str],
    api_base: str,
    api_key: str,
    transport=post_json,
    timeout: float = 2.0,
) -> int | None:
    """Return a 1-10 difficulty score, or None if classification failed.

    Never raises: a router must not fail because its classifier did.
    """
    body = {
        "model": CLASSIFIER_MODEL,
        "messages": [{"role": "user", "content": build_prompt(user_msg, tool_names)}],
        "temperature": 0,
        "max_tokens": 8,
        # Without this the model spends its 8 token budget thinking and
        # returns empty content.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        data = transport(
            f"{api_base}/chat/completions",
            body,
            {"Authorization": f"Bearer {api_key}"},
            timeout,
        )
        return parse_score(data["choices"][0]["message"])
    except Exception:
        return None
