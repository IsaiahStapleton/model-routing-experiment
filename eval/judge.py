"""Label each answer adequate or inadequate using gpt-5.5.

Runs INSIDE the podling pod: the real OpenAI key never exists in podling's
config, it is a placeholder that the claw-proxy substitutes, so only something
egressing through that proxy can authenticate. Stdlib only for that reason.

Judges each answer independently rather than comparing them, which avoids
position bias and lets several tiers be adequate for one prompt. That is
exactly what identifies the cheapest adequate tier.
"""
import json
import os
import re
import sys
import time
import urllib.request

MODEL = "gpt-5.5"
# gpt-5.5 is a reasoning model: it spends completion tokens thinking before
# answering, and rejects max_tokens outright. Too small a budget returns empty.
# Measured reasoning cost on real cases ranged 257-469 tokens; 2000 gives
# comfortable headroom for the longest answers without being wasteful.
MAX_COMPLETION_TOKENS = 2000

# Bounded retry for transient request failures (timeouts, connection resets).
# Does not retry an unparseable-but-successful response; that is a distinct
# failure mode and is logged separately instead.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (2, 4)

_PROMPT = (
    "You are grading whether an ANSWER is adequate for a TASK.\n"
    "Adequate means: correct, responsive to what was asked, and useful as-is.\n"
    "Do not reward extra length or penalise brevity.\n\n"
    "TASK:\n{task}\n\nANSWER:\n{answer}\n\n"
    "Reply with exactly one word: ADEQUATE or INADEQUATE."
)


def build_judge_prompt(task: str, answer: str) -> str:
    return _PROMPT.format(task=task, answer=answer)


def parse_verdict(text: str) -> bool | None:
    upper = (text or "").upper()
    # Check INADEQUATE first: ADEQUATE is a substring of it.
    if "INADEQUATE" in upper:
        return False
    if "ADEQUATE" in upper:
        return True
    return None


def prejudge(answer: str) -> bool | None:
    """Short-circuit cases needing no judge. Returns None to mean 'ask'."""
    if not (answer or "").strip():
        return False
    return None


def _openai_config() -> tuple[str, str]:
    with open(os.path.expanduser("~/.openclaw/openclaw.json")) as fh:
        cfg = json.load(fh)
    provider = cfg["models"]["providers"]["openai"]
    return provider["baseUrl"], provider["apiKey"]


def _http_request(task: str, answer: str, base: str, key: str) -> str:
    """Make the real HTTP call to the judge model. Returns raw message content.

    Kept separate from ask_judge so tests can inject a fake in its place
    without making network calls.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": build_judge_prompt(task, answer)}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"].get("content") or ""


def ask_judge(task: str, answer: str, base: str, key: str, request=_http_request) -> bool | None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = request(task, answer, base, key)
        except Exception as exc:
            print(f"request failed (attempt {attempt}/{MAX_ATTEMPTS}): {type(exc).__name__}",
                  flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])
                continue
            return None
        verdict = parse_verdict(content)
        if verdict is None:
            print(f"unparseable verdict: {content[:60]!r}", flush=True)
        return verdict
    return None


def main(judged_path: str, answers_path: str, out_path: str) -> None:
    base, key = _openai_config()

    with open(judged_path) as fh:
        tasks = {j["id"]: j["user_msg"] for j in (json.loads(l) for l in fh if l.strip())}

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((rec.get("id"), rec.get("tier")))

    with open(answers_path) as fh:
        answers = [json.loads(l) for l in fh if l.strip()]

    with open(out_path, "a") as out:
        for record in answers:
            key_pair = (record["id"], record["tier"])
            if key_pair in done:
                continue
            if record.get("status") != "ok":
                verdict = False
            else:
                verdict = prejudge(record["answer"])
                if verdict is None:
                    try:
                        verdict = ask_judge(tasks[record["id"]], record["answer"], base, key)
                    except Exception as exc:
                        print(f"judge error {key_pair}: {type(exc).__name__}", flush=True)
                        verdict = None
            out.write(json.dumps({"id": record["id"], "tier": record["tier"],
                                  "adequate": verdict}) + "\n")
            out.flush()
            print(f"{record['id']:<12} {record['tier']:<16} {verdict}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
