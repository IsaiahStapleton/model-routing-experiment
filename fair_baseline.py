"""Was the classifier compared against a FAIRLY TUNED heuristic?

The original sweep gave the classifier a 36-cell threshold grid but gave the
length heuristic a single hand-picked configuration. That is an unfair
comparison: the classifier was tuned, the baseline was not.

This sweeps the heuristic's thresholds over the same data and rebuilds the
Pareto frontier so both are tuned before being compared.
"""
import json
import itertools

TP = {"gpt-oss-120b": 33.6, "qwen3-30b-a3b": 55.2, "qwen3-1.7b": 93.1}
ORDER = ["qwen3-1.7b", "qwen3-30b-a3b", "gpt-oss-120b"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


items = load("results/judged-final.jsonl")
answers = {(a["id"], a["tier"]): a for a in load("results/answers.jsonl")}
labels = {}
for r in load("results/labels.jsonl"):
    labels.setdefault(r["id"], {})[r["tier"]] = r["adequate"]
scores = {s["id"]: s["score"] for s in load("results/scores.jsonl")}

cost = {k: a["completion_tokens"] / TP[k[1]] for k, a in answers.items()}


def cheapest_adequate(d):
    for t in ORDER:
        if d.get(t) is True:
            return t
    return None


def evaluate(chooser):
    scored = under = 0
    total = 0.0
    for it in items:
        per = labels.get(it["id"], {})
        if cheapest_adequate(per) is None:
            continue
        scored += 1
        c = chooser(it)
        if per.get(c) is not True:
            under += 1
        total += cost.get((it["id"], c), 0.0)
    return (scored - under) / scored, total, scored


def length_chooser(a, b, use_tools):
    def f(it):
        if use_tools and it["tool_names"]:
            return "gpt-oss-120b"
        pt = it["prompt_tokens"]
        if pt <= a:
            return "qwen3-1.7b"
        if pt <= b:
            return "qwen3-30b-a3b"
        return "gpt-oss-120b"
    return f


def classifier_chooser(a, b):
    def f(it):
        s = scores.get(it["id"])
        if s is None:
            return "qwen3-30b-a3b"
        if s <= a:
            return "qwen3-1.7b"
        if s <= b:
            return "qwen3-30b-a3b"
        return "gpt-oss-120b"
    return f


rows = []

# tuned length heuristic: sweep both cut points and the tool signal
grid = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 10000]
for a, b in itertools.combinations(grid, 2):
    for use_tools in (True, False):
        acc, gpu, n = evaluate(length_chooser(a, b, use_tools))
        rows.append((f"length({a},{b},tools={use_tools})", acc, gpu, "heuristic"))

# classifier, same treatment
for a, b in itertools.combinations(range(1, 10), 2):
    acc, gpu, n = evaluate(classifier_chooser(a, b))
    rows.append((f"classifier({a},{b})", acc, gpu, "classifier"))

# reference points
acc, gpu, n = evaluate(lambda it: "qwen3-30b-a3b")
rows.append(("always qwen3-30b-a3b", acc, gpu, "fixed"))
acc, gpu, n = evaluate(lambda it: cheapest_adequate(labels[it["id"]]) or "gpt-oss-120b")
rows.append(("oracle", acc, gpu, "oracle"))


def frontier(rs):
    out = []
    for name, acc, gpu, kind in rs:
        if not any(a2 >= acc and g2 < gpu or a2 > acc and g2 <= gpu
                   for n2, a2, g2, k2 in rs if n2 != name):
            out.append((name, acc, gpu, kind))
    return sorted(out, key=lambda x: x[2])


print(f"  evaluated on n={n} items\n")
print("  PARETO FRONTIER with BOTH strategies tuned (cheapest first):")
print(f"  {'strategy':<38} {'acc':>6} {'GPU-s':>8}  kind")
for name, acc, gpu, kind in frontier(rows):
    print(f"  {name:<38} {acc:>6.3f} {gpu:>8.1f}  {kind}")

print("\n  BEST OF EACH KIND at comparable accuracy levels:")
for target in (0.60, 0.70, 0.733, 0.80, 0.867):
    print(f"    at accuracy >= {target:.3f}:")
    for kind in ("heuristic", "classifier"):
        cands = [r for r in rows if r[3] == kind and r[1] >= target]
        if cands:
            best = min(cands, key=lambda x: x[2])
            print(f"      {kind:<11} {best[0]:<32} acc={best[1]:.3f} {best[2]:.1f} GPU-s")
        else:
            print(f"      {kind:<11} (cannot reach this accuracy)")
