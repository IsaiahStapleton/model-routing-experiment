# Intelligent model routing on a DGX Spark

Does a small LLM classifier route requests to the cheapest adequate model tier better than a trivial heuristic, and is routing worth doing at all?

Short answers: **no, a tuned length heuristic beats it**, and **no, routing does not pay off on this workload either**. Details below.

Run on an NVIDIA DGX Spark (GB10, 128 GB unified memory, arm64) serving three vLLM tiers behind a LiteLLM proxy, 2026-07-30.

## Result

**A tuned length heuristic beats the LLM classifier.** Counting prompt tokens routes at least as well as asking a 1.7B model to rate difficulty, and does it more cheaply.

Compared at matched accuracy, with both strategies tuned over their own threshold grids on the same data:

| accuracy | best heuristic | best classifier |
|---|---|---|
| 0.733 | `length(20,150)` **322.1 GPU-s** | `classifier(2,5)` 351.2 GPU-s |
| 0.800 | `length(10,150)` **364.9 GPU-s** | cannot reach 0.800 |
| 0.867 | (equals always-30b) 438.7 GPU-s | `classifier(1,5)` 438.7 GPU-s, which ignores the classifier and sends everything to the 30b |

The Pareto frontier below 0.867 accuracy consists **entirely of length heuristics**. The classifier's only appearance on the frontier is a configuration that routes every request to qwen3-30b-a3b regardless of what the classifier said.

Why: prompt length predicts difficulty about as well as the 1.7B's judgement on this corpus, and length is continuous while the classifier emits only two distinct values (EASY and MEDIUM, since it never emits HARD). Continuous input gives the heuristic finer control over the cost/accuracy tradeoff.

### And routing itself does not pay off

Separately from which router is better, routing at all loses to doing nothing clever:

| strategy | accuracy | GPU-seconds |
|---|---|---|
| always qwen3-30b-a3b | 0.867 | 438.7 |
| oracle (cheapest adequate, unattainable) | 1.000 | 489.4 |

Perfect routing costs 489.4 GPU-seconds against 438.7 for simply always using qwen3-30b-a3b, so even an oracle is 1.12x the cost of the simplest strategy. No routing configuration is both cheaper and more accurate than always-30b.

Routing fails here because of the workload: the cheap tier is adequate only 43.8% of the time, and 4 of 32 items are adequate *only* on gpt-oss-120b, so an optimal router must pay for the expensive tier precisely where it hurts. A workload with more easy traffic would change this.

**Recommendation: use qwen3-30b-a3b for everything. Do not build a router, and if you ever do, use prompt length rather than an LLM classifier.**

### A correction worth recording

An earlier version of this README claimed the classifier beat the heuristic by 20 percentage points. That comparison was **unfair and wrong**: the classifier had been tuned across a 36-cell threshold grid while the heuristic used a single hand-picked configuration. Tuning both on the same data reverses the conclusion. `fair_baseline.py` reproduces the corrected comparison.

## Per-tier behaviour

| tier | adequate | cost per answer | notes |
|---|---|---|---|
| gpt-oss-120b | 26/32 (81.2%) | 43.5 GPU-s | MoE, 5.1B active, MXFP4, 32k context |
| qwen3-30b-a3b | 26/32 (81.2%) | 15.1 GPU-s | MoE, 3B active, FP8, 32k context |
| qwen3-1.7b | 14/32 (43.8%) | 5.1 GPU-s | FP8, 4k context, also serves as the classifier |

The two large tiers tie in aggregate but are **complementary, not interchangeable**: 4 items are adequate only on gpt-oss (syn-e04, syn-h02, syn-h05, syn-h09) and 4 different ones only on qwen3-30b (syn-m05, syn-m07, syn-m09, real-02406). That is why an oracle needs both and ends up costing more than either alone.

Router overhead is negligible: the classifier call takes 57 ms against 10.3 s of serving on qwen3-30b and 32.0 s on gpt-oss, so 0.2 to 0.6% of end-to-end latency. Routing is rejected on cost economics, not on overhead.

## Classifier capability ceiling

qwen3-1.7b separates trivial from non-trivial reliably (Spearman rho 0.828 against hand-assigned ordinal truth, holding on a held-out set) but **cannot separate moderate from hard**, which is a large part of why prompt length outperforms it: length varies continuously while the classifier collapses every request into one of two buckets and never emits the top category. In practice it only ever outputs EASY or MEDIUM, mapped to scores 2 and 5, which collapses the 36-cell threshold grid to four functionally distinct behaviours.

An earlier prompt variant appeared to fix this, lifting rho to 0.898 with the top category emitted 3 times. It contained the line "if the task contains the words derive, prove, design ... it is HARD", and the hard probe items began with exactly those words. On a held-out set with none of that vocabulary the advantage vanished completely (rho 0.828, top category emitted 0 times). That variant was **rejected as test-set leakage**.

## How it works

```
corpus.py    build a judged corpus (self-contained items, gradeable from the task alone)
             and a guardrail corpus (real agent turns, used only to test the context guard)
dual_run.py  run every judged item on every tier, recording answers, tokens, latency
judge.py     label each answer ADEQUATE or INADEQUATE using gpt-5.5
sweep.py     replay policy.decide() across a threshold grid, compare against baselines
```

`router/policy.py` holds `decide(score, prompt_tokens, thresholds)`, a pure function with no I/O. Both the offline sweep and any future live router import the same function, so a threshold chosen from the results is literally the code that would run in production.

Two guardrails override the classifier: a failed classification falls back to qwen3-30b-a3b, and a request whose prompt plus 1024 tokens of output headroom exceeds a tier's context is promoted to the smallest tier that fits.

## Reproducing

Requires three vLLM tiers behind a LiteLLM proxy at `127.0.0.1:4000` with the key in `.env` as `LITELLM_MASTER_KEY`, plus a way to reach gpt-5.5 for judging.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q                       # 67 tests

.venv/bin/python -m eval.corpus eval/synthetic.json <real-turns.jsonl or -> eval/out
.venv/bin/python -m eval.dual_run eval/out/judged-final.jsonl eval/out/answers.jsonl
python3 eval/judge.py eval/out/judged-final.jsonl eval/out/answers.jsonl eval/out/labels.jsonl
.venv/bin/python -m eval.sweep eval/out
```

`judge.py` is stdlib-only because it runs wherever the judging credential lives, which in the original setup was inside a Kubernetes pod rather than on the Spark.

`docker-compose.yml` and `litellm-config.yaml` are the tier definitions used for the run. See `LAB.md` for the memory tuning, including the load-order constraint: vLLM checks free memory at startup, so the largest model must be started first.

## Data in `results/`

| file | contents |
|---|---|
| `judged-final.jsonl` | the 32 corpus items actually evaluated |
| `answers.jsonl` | 96 answers (32 items x 3 tiers), all complete, no empties |
| `labels.jsonl` | 96 adequacy labels from gpt-5.5 |
| `scores.jsonl` | classifier score per item |
| `results.md` | the full threshold sweep table |
| `guardrail-sanitised.jsonl` | guardrail corpus with message text redacted, see below |

## Privacy

`guardrail-sanitised.jsonl` has had its message text replaced. The original contained real agent session content including conversation fragments, internal cron UUIDs, and workspace paths. Only `prompt_tokens` is used by the guardrail tests, so redaction costs nothing.

The raw 58 MB session dump (`real-turns.jsonl`) is gitignored and must never be committed.

`judged-final.jsonl` retains two real items, both innocuous: an automated git-push cron prompt and the question "testing, what model are you using?". **Review these before publishing this repository publicly.**

## Limitations

These bound what the numbers support, and they matter:

- 32 items, of which 2 have no adequate tier anywhere, so every row is n=30. One item flipping moves accuracy by ~3.3 points, so differences under about 7 points are not meaningful.
- The judged corpus is 30 synthetic items plus only 2 real ones. The real agent traffic sampled proved to be 57 copies of one cron job plus one throwaway question, carrying almost no diversity, so it was deduplicated.
- One sample per tier per item at temperature 0. Variance is unmeasured.
- Binary adequate/inadequate labels, no graded quality scale.
- Cost is decode-only GPU-seconds and ignores prefill, which understates long prompts.
- qwen3-1.7b was evaluated with thinking disabled while gpt-oss reasoned freely, because the cheap tier exists to be fast. This understates what the 1.7b could achieve.
- 3 of 96 answers hit the 4000-token ceiling and were truncated; all 3 were judged inadequate.
- 1 of 96 labels remains null after retries and is treated as not-adequate.
- The judge is gpt-5.5, which is not a candidate tier, so there is no self-preference bias. But it saw only the task and the answer, not the full context the serving model had.

## Incidental finding

The item "Spell the word 'necessary' backwards" sent qwen3-30b-a3b into a 4000-token decode loop costing 72.5 GPU-seconds, and it was judged inadequate. gpt-oss-120b answered the same prompt in 126 tokens. A trivial character-manipulation task defeating the mid tier through a degenerate loop is worth investigating separately.
