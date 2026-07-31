"""Build the two evaluation corpora.

judged.jsonl   self-contained items, gradeable from the task alone
guardrail.jsonl real agent turns, used only to prove the context guardrail fires

Real turns are kept out of the judged set because the judge sees only the task
and the answer, while the serving model saw the full ~23k context. A correct
answer citing MEMORY.md would look unverifiable, producing noisy labels.
"""
import json
import re
import sys

# Phrases that signal the request depends on earlier conversation.
_BACK_REFERENCE = re.compile(
    r"\b(the same|again|that|this|it|those|these|continue|as before|"
    r"we decided|you said|earlier|previous|the other)\b",
    re.IGNORECASE,
)

MIN_SELF_CONTAINED_CHARS = 15


def estimate_tokens(text: str) -> int:
    """Cheap proxy: ~4 characters per token. Good enough for guardrail sizing."""
    return max(1, len(text) // 4)


def is_self_contained(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < MIN_SELF_CONTAINED_CHARS:
        return False
    return not _BACK_REFERENCE.search(stripped)


def load_synthetic(path: str) -> list[dict]:
    with open(path) as fh:
        raw = json.load(fh)
    items = []
    for entry in raw:
        items.append({
            "id": entry["id"],
            "user_msg": entry["user_msg"],
            "tool_names": entry.get("tool_names", []),
            "prompt_tokens": estimate_tokens(entry["user_msg"]),
            "source": "synthetic",
        })
    return items


def extract_real_turns(session_dump: str) -> list[dict]:
    """Pull user messages out of concatenated OpenClaw session jsonl."""
    turns = []
    for index, line in enumerate(session_dump.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message") if isinstance(record, dict) else None
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            continue
        turns.append({
            "id": f"real-{index:05d}",
            "user_msg": content.strip(),
            "tool_names": [],
            "prompt_tokens": estimate_tokens(content),
            "source": "real",
        })
    return turns


def _write(path: str, items: list[dict]) -> None:
    with open(path, "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def main(synthetic_path: str, real_dump_path: str | None, out_dir: str) -> None:
    judged = load_synthetic(synthetic_path)
    guardrail: list[dict] = []

    if real_dump_path:
        with open(real_dump_path) as fh:
            real = extract_real_turns(fh.read())
        # Real turns with a big prompt are guardrail material. Short
        # self-contained ones can also be judged, giving the judged set some
        # real-world items rather than only synthetic ones.
        for turn in real:
            if is_self_contained(turn["user_msg"]):
                judged.append(turn)
            else:
                guardrail.append(turn)

    _write(f"{out_dir}/judged.jsonl", judged)
    _write(f"{out_dir}/guardrail.jsonl", guardrail)
    print(f"judged={len(judged)} guardrail={len(guardrail)}")


if __name__ == "__main__":
    real_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    main(sys.argv[1], real_arg, sys.argv[3])
