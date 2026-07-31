"""Run every judged corpus item on every tier that can serve it.

Resumable: re-running skips (id, tier) pairs already present in answers.jsonl,
because a full run takes ~20 minutes and losing it to a dropped SSH session
would be tedious.
"""
import json
import os
import re
import sys
import time
import urllib.request

from router.policy import OUTPUT_HEADROOM, TIER_CONTEXT, TIER_ORDER

API_BASE = "http://127.0.0.1:4000/v1"
# 4000 is the budget at which all three tiers reach finish_reason=stop on
# their own (gpt-oss-120b needs ~2876 tokens on hard items); below that, the
# harness itself truncates mid-reasoning or mid-answer for every tier. It is
# a ceiling, not a flat allocation: answer_budget() below caps it per tier so
# prompt + output never exceeds that tier's context window (qwen3-1.7b's
# context is only 4096, smaller than this ceiling on its own).
MAX_ANSWER_TOKENS = 4000
# Tokeniser estimate error (corpus prompt_tokens is chars//4) plus
# chat-template overhead not captured by that estimate.
SAFETY_MARGIN = 128
MIN_ANSWER_TOKENS = 256
REQUEST_TIMEOUT = 300.0


def read_key() -> str:
    path = os.path.expanduser("~/isaiah-routing-lab/.env")
    with open(path) as fh:
        match = re.search(r"(?<=LITELLM_MASTER_KEY=).*", fh.read())
    return match.group(0).strip()


def tier_can_serve(tier: str, prompt_tokens: int) -> bool:
    return prompt_tokens + OUTPUT_HEADROOM <= TIER_CONTEXT[tier]


def already_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((record.get("id"), record.get("tier")))
    return done


def answer_budget(tier: str, prompt_tokens: int) -> int:
    room = TIER_CONTEXT[tier] - prompt_tokens - SAFETY_MARGIN
    return max(MIN_ANSWER_TOKENS, min(MAX_ANSWER_TOKENS, room))


def build_body(tier: str, user_msg: str, prompt_tokens: int) -> dict:
    return {
        "model": tier,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": answer_budget(tier, prompt_tokens),
        "temperature": 0,
        # Without this, hybrid-thinking tiers spend the whole answer-token
        # budget reasoning and return empty content.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def ask(tier: str, user_msg: str, key: str, prompt_tokens: int) -> dict:
    body = build_body(tier, user_msg, prompt_tokens)
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.load(resp)
    elapsed = time.time() - started
    message = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return {
        "answer": (message.get("content") or "").strip(),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "latency_s": round(elapsed, 2),
        "status": "ok",
    }


def main(judged_path: str, out_path: str) -> None:
    key = read_key()
    done = already_done(out_path)
    with open(judged_path) as fh:
        items = [json.loads(line) for line in fh if line.strip()]

    with open(out_path, "a") as out:
        for item in items:
            for tier in TIER_ORDER:
                if (item["id"], tier) in done:
                    continue
                if not tier_can_serve(tier, item["prompt_tokens"]):
                    record = {"id": item["id"], "tier": tier, "answer": "",
                              "completion_tokens": 0, "prompt_tokens": item["prompt_tokens"],
                              "latency_s": 0.0, "status": "skipped_context"}
                else:
                    try:
                        record = {"id": item["id"], "tier": tier,
                                  **ask(tier, item["user_msg"], key, item["prompt_tokens"])}
                    except Exception as exc:
                        record = {"id": item["id"], "tier": tier, "answer": "",
                                  "completion_tokens": 0, "prompt_tokens": 0,
                                  "latency_s": 0.0, "status": f"error: {type(exc).__name__}"}
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"{item['id']:<12} {tier:<16} {record['status']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
