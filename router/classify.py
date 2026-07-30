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
    "Classify this task as EASY, MEDIUM, or HARD.\n"
    "EASY = trivial facts, arithmetic, greetings.\n"
    "MEDIUM = short code, routine explanation.\n"
    "HARD = multi-step reasoning, derivation, subtle debugging.\n"
    "Answer with one word only."
)

# Word categories map onto the same 1-10 scale parse_score has always
# returned, chosen so the default policy.decide() thresholds (3, 7) still
# route EASY to the cheap tier, MEDIUM to the mid tier, and HARD to the
# expensive tier without any change to policy.py.
_WORD_SCORES = {"EASY": 2, "MEDIUM": 5, "HARD": 9}
_WORD_RE = re.compile(r"\b(EASY|MEDIUM|HARD)\b", re.IGNORECASE)


def build_prompt(user_msg: str, tool_names: list[str]) -> str:
    task = (user_msg or "")[:MAX_TASK_CHARS]
    tools = ", ".join(tool_names) if tool_names else "none"
    return f"TASK: {task}\nAVAILABLE TOOLS: {tools}\n\n{_INSTRUCTIONS}"


def parse_score(message: dict) -> int | None:
    """Extract a 1-10 score, tolerating prose and empty content.

    Thinking-capable models may return an empty content field with the answer
    in reasoning_content, so fall back to it rather than reporting failure.

    Checks the EASY/MEDIUM/HARD category words first (case-insensitively),
    since that is what the classifier prompt now asks for, then falls back
    to the original bare-integer parsing for backward compatibility.
    """
    if not isinstance(message, dict):
        return None
    for field in ("content", "reasoning_content"):
        text = message.get(field)
        if not text:
            continue
        text = str(text)
        word_match = _WORD_RE.search(text)
        if word_match:
            return _WORD_SCORES[word_match.group(1).upper()]
        match = re.search(r"(?<!-)\b(10|[1-9])\b", text)
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
