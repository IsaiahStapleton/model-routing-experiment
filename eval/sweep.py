"""Replay the policy across threshold grids and compare against baselines.

Cost is GPU-seconds of exclusive decode time, since there is no dollar cost on
owned hardware. This counts decode only and ignores prefill, which understates
long prompts; acceptable because judged items are short by construction.
"""
import json
import os
import random
import re
import sys

from router.classify import classify
from router.policy import TIER_ORDER, decide

API_BASE = "http://127.0.0.1:4000/v1"

THROUGHPUT = {
    "gpt-oss-120b": 33.6,
    "qwen3-30b-a3b": 55.2,
    "qwen3-1.7b": 93.1,
}


def read_key() -> str:
    with open(os.path.expanduser("~/isaiah-routing-lab/.env")) as fh:
        return re.search(r"(?<=LITELLM_MASTER_KEY=).*", fh.read()).group(0).strip()


def gpu_seconds(tier: str, completion_tokens: int) -> float:
    return completion_tokens / THROUGHPUT[tier]


def cheapest_adequate(labels_for_id: dict) -> str | None:
    for tier in TIER_ORDER:
        if labels_for_id.get(tier) is True:
            return tier
    return None


def evaluate(items, labels, costs, chooser) -> dict:
    """Score a routing strategy. chooser(item) -> tier."""
    scored = under = over = 0
    total_cost = 0.0
    for item in items:
        per_tier = labels.get(item["id"], {})
        best = cheapest_adequate(per_tier)
        if best is None:
            # No tier produced an adequate answer; the routing choice is not
            # measurable for this item.
            continue
        scored += 1
        chosen = chooser(item)
        if per_tier.get(chosen) is not True:
            under += 1
        elif TIER_ORDER.index(chosen) > TIER_ORDER.index(best):
            over += 1
        total_cost += costs.get((item["id"], chosen), 0.0)
    if scored == 0:
        return {"n": 0, "accuracy": 0.0, "under_route": 0.0, "over_route": 0.0,
                "gpu_seconds": 0.0}
    return {
        "n": scored,
        "accuracy": round((scored - under) / scored, 3),
        "under_route": round(under / scored, 3),
        "over_route": round(over / scored, 3),
        "gpu_seconds": round(total_cost, 1),
    }


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def classify_all(items: list[dict], out_path: str) -> dict:
    """Classify once and cache, so sweeping costs no model calls."""
    cached = {}
    if os.path.exists(out_path):
        for rec in load(out_path):
            cached[rec["id"]] = rec["score"]
    key = read_key()
    with open(out_path, "a") as out:
        for item in items:
            if item["id"] in cached:
                continue
            score = classify(item["user_msg"], item["tool_names"], API_BASE, key, timeout=30.0)
            cached[item["id"]] = score
            out.write(json.dumps({"id": item["id"], "score": score}) + "\n")
            out.flush()
            print(f"{item['id']:<12} score={score}", flush=True)
    return cached


def main(out_dir: str) -> None:
    # judged-final.jsonl is the deduplicated 32-item corpus that matches the
    # answers and labels exactly. judged.jsonl is a stale 88-item file with 56
    # duplicate cron entries and must not be used.
    items = load(f"{out_dir}/judged-final.jsonl")
    answers = load(f"{out_dir}/answers.jsonl")
    label_rows = load(f"{out_dir}/labels.jsonl")

    labels: dict = {}
    for row in label_rows:
        labels.setdefault(row["id"], {})[row["tier"]] = row["adequate"]

    costs = {
        (a["id"], a["tier"]): gpu_seconds(a["tier"], a.get("completion_tokens", 0))
        for a in answers
    }

    scores = classify_all(items, f"{out_dir}/scores.jsonl")

    rows = []

    # Theoretical floor: always pick the cheapest tier that is actually
    # adequate for this item, read straight from the labels. No routing
    # strategy can beat this on cost without also beating it on quality.
    rows.append(("oracle (cheapest adequate)", evaluate(
        items, labels, costs,
        lambda i: cheapest_adequate(labels.get(i["id"], {})),
    )))

    # Baselines next, so the classifier has something to beat.
    for tier in TIER_ORDER:
        rows.append((f"always {tier}", evaluate(items, labels, costs, lambda i, t=tier: t)))

    random.seed(0)
    rows.append(("random", evaluate(items, labels, costs,
                                    lambda i: random.choice(TIER_ORDER))))

    def length_heuristic(item):
        # Dumb but honest: long prompt or any tools means escalate.
        if item["tool_names"] or item["prompt_tokens"] > 200:
            return "gpt-oss-120b"
        if item["prompt_tokens"] > 60:
            return "qwen3-30b-a3b"
        return "qwen3-1.7b"

    rows.append(("length heuristic", evaluate(items, labels, costs, length_heuristic)))

    # The classifier, swept over threshold pairs.
    best = None
    for cheap_max in range(1, 9):
        for mid_max in range(cheap_max + 1, 10):
            chooser = lambda i, c=cheap_max, m=mid_max: decide(
                scores.get(i["id"]), i["prompt_tokens"], thresholds=(c, m)
            ).tier
            result = evaluate(items, labels, costs, chooser)
            rows.append((f"classifier ({cheap_max},{mid_max})", result))
            # Prefer fewest under-routes, then lowest cost.
            candidate = (result["under_route"], result["gpu_seconds"])
            if best is None or candidate < best[0]:
                best = (candidate, cheap_max, mid_max, result)

    lines = ["# Routing sweep results", "",
             "| strategy | n | accuracy | under-route | over-route | GPU-s |",
             "|---|---|---|---|---|---|"]
    for name, r in rows:
        lines.append(f"| {name} | {r['n']} | {r['accuracy']} | {r['under_route']} "
                     f"| {r['over_route']} | {r['gpu_seconds']} |")
    if best:
        _, cheap_max, mid_max, r = best
        lines += ["", f"**Best classifier thresholds: ({cheap_max}, {mid_max})** "
                      f"with under-route {r['under_route']} and {r['gpu_seconds']} GPU-s."]
    unparsed = sum(1 for v in scores.values() if v is None)
    lines += ["", f"Classifier returned no parseable score for {unparsed}/{len(scores)} items."]
    lines += ["", "Note: the classifier maps EASY->2, MEDIUM->5, HARD->9 and in "
                  "practice never emits HARD, so scores are only ever 2 or 5. "
                  "Threshold pairs whose lower bound (cheap_max) is 5 or greater "
                  "therefore behave identically to each other, since no score "
                  "ever falls above 5."]

    with open(f"{out_dir}/results.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
