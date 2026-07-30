# Isaiah's model routing lab (dgx-spark-4)

Three vLLM tiers behind a LiteLLM proxy, exposed to the `isaiah-claw` OpenShift namespace over Skupper.

## Layout

| Component | Port (loopback) | Pool | Context | Notes |
|---|---|---|---|---|
| gpt-oss-120b | 8001 | 0.60 | 32k | MoE 5.1B active, MXFP4, 1.80x concurrency |
| qwen3-30b-a3b | 8002 | 0.29 | 32k | MoE 3B active, FP8, 1.13x concurrency |
| qwen3-1.7b | 8004 | 0.030 | **4k** | FP8, triage/router tier only, NOT an agent model |
| qwen3-4b | 8003 | - | - | stopped: 5 GB of weights does not fit |
| LiteLLM | 4000 | - | - | one OpenAI-compatible endpoint, bearer auth |
| skupper router | host net | - | - | docker site `spark-box`, connector to 127.0.0.1:4000 |

Everything binds to 127.0.0.1 only. The bearer token lives in `.env` (mode 600) and is mirrored into the cluster Secret `spark-models-api-key`.

## All three tiers fit, but the box runs at 98% memory

Verified under concurrent load: 18k-token prompts to both large tiers plus a hit on the triage tier all succeeded, with peak usage **120,598 MB of 122,502 (98.4%)**. It works, but the margin is ~1.9 GB. This is a **shared** machine, so an OOM could kill someone else's job, not just a tier here. If anyone else starts using the box, drop a tier.

Weights alone account for 92.3 GiB of the 119.6 GiB:

| | weights |
|---|---|
| gpt-oss-120b (MXFP4) | 60.8 GiB |
| qwen3-30b-a3b (FP8) | 29.0 GiB |
| qwen3-1.7b (FP8) | 2.5 GiB |

Measured overheads that `--gpu-memory-utilization` does **not** cover: ~2.7 GiB OS/docker/litellm/skupper baseline, plus ~1-2 GiB per vLLM engine outside its pool.

Two configurations with real headroom, if you want them:

- **gpt-oss-120b + qwen3-30b-a3b only** — both at full 32k, ~7 GB free.
- **qwen3-30b + 4b + 1.7b (drop gpt-oss)** — ~40 GiB total, ~75 GiB free, generous context everywhere.

## Sizing rules learned the hard way

- **Read the concurrency line, not just pass/fail.** `Maximum concurrency for N tokens per request: 1.01x` means it fit by luck and will fail after any unrelated change. Aim for >1.1x.
- **A pool cannot go below weights + KV + activations.** qwen3-30b's 29.0 GiB of weights put a hard floor near 0.28; 0.245 left 0.27 GiB and could not allocate a single cache block.
- **`--kv-cache-dtype fp8` does NOT free memory.** It doubles tokens-per-byte, so it only helps when you have spare pool. It needs *more* room during init and made qwen3-30b fail to start at a pool size that worked without it.
- **Untried lever:** `--enforce-eager` on gpt-oss would free several GiB of CUDA graph memory at a real throughput cost.

## Tool calling needs an explicit parser per model family

vLLM will not emit `tool_calls` without one, and it fails **silently**: the response comes back 200 with empty content and no tool calls, which an agent reports as a generic failure. Each family needs its own parser:

| model | flags |
|---|---|
| gpt-oss-120b | `--enable-auto-tool-choice --tool-call-parser openai` |
| qwen3-* | `--enable-auto-tool-choice --tool-call-parser hermes` |

`openai` is the registered name for `OpenAIToolParser`, which handles gpt-oss's harmony format. Note that gpt-oss's *reasoning* is parsed automatically without a `--reasoning-parser` flag, so seeing `reasoning_content` come back is not evidence that tool calling works.

Always verify with a payload that actually carries `tools`. A short "reply with one word" prompt passes even when tool calling is completely broken.

## Context length: OpenClaw needs 32k minimum

Do not trim `--max-model-len` below 32768 for any tier an OpenClaw agent will call. Its system prompt plus tool definitions is **23,493 tokens** before any conversation history, so a 16k limit fails every request with a vLLM 400:

```
ValueError: Input length (23493) exceeds model's maximum context length (16384)
```

Agent-side this surfaces only as "The agent run failed before producing a reply", so check the vLLM or LiteLLM logs rather than the agent's message.

Raising `--max-model-len` does **not** enlarge the reserved pool; it only changes how vLLM carves the existing KV cache. Check the startup line `GPU KV cache size: N tokens` — as long as N exceeds your `--max-model-len`, the engine will start. Qwen3-30B had 37,392 tokens of KV at `0.26`, so it moved from 16k to 32k with no memory change at all. Lower `--max-num-seqs` if you need to trade concurrency for context.

There are two distinct failure modes for memory, and they need opposite fixes:

- `Free memory ... less than desired GPU memory utilization` — it cannot reserve its pool. Fix the load order, or lower the fraction.
- `No available memory for the cache blocks` — the pool fits the weights but leaves no KV cache. Fix by *raising* the fraction or lowering `--max-model-len`.

```
docker compose up -d --no-deps vllm-gptoss120b   # wait for /health, ~8 min
docker compose up -d --no-deps vllm-qwen3-30b    # wait for /health
docker compose up -d --no-deps litellm
```

## Usage

```
K=$(grep -oP '(?<=LITELLM_MASTER_KEY=).*' .env)
curl -s http://127.0.0.1:4000/v1/models -H "Authorization: Bearer $K"
curl -s http://127.0.0.1:4000/v1/chat/completions -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"hi"}]}'
```

## Skupper

The cluster cannot dial in (the Spark is on a private network), so the Spark holds an outbound mTLS link to the cluster's inter-router Route on 443. In the cluster this surfaces as Service `spark-models:4000` in namespace `isaiah-claw`.

Status commands:

```
SKUPPER_PLATFORM=docker skupper site status
SKUPPER_PLATFORM=docker skupper connector status
```

Note: the non-kube CLI reports link/connector status unreliably (it showed `Pending / Not Operational` while the link was demonstrably carrying traffic). Trust the cluster side (`oc get site spark-lab -n isaiah-claw`) and the router logs instead.

If the Spark router is recreated (`skupper system reload` regenerates its router identity), the cluster router can hold a stale link-state record and refuse to route. The fix is to restart the cluster router pod:

```
oc delete pod -n isaiah-claw -l skupper.io/component=router
```

## Downloading more models

Use `HF_HUB_DISABLE_XET=1`. The xet transfer backend stalled silently partway through both large downloads (bytes flatlined while the process stayed alive). Plain HTTPS sustained ~90 MB/s.

For repos carrying multiple checkpoint formats, exclude what vLLM does not need via `snapshot_download(..., ignore_patterns=[...])` rather than the CLI's `--exclude`, whose trailing patterns get misparsed as positional filenames.

## Known box issues

- The system clock is ~8 hours ahead of real UTC and `System clock synchronized: no`. Fixing it needs root. Container log timestamps and any traces emitted from this box will be wrong until then.
- Docker has the NVIDIA container toolkit installed but never registered as a runtime, so `runtime: nvidia` fails. The compose file uses `gpus: all` instead, which works without root.
