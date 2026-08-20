# AgentWorld-35B vs Qwen3.8-27B on AGX Orin — head-to-head verdict

**Question.** Should Qwen3.8-27B (dense hybrid GDN VLM) replace Qwen-AgentWorld-35B-A3B
(A3B MoE, ~3B active) as the `:8010` swap-router primary?

**Measured 2026-08-19** on agx-node (JetPack 6.x / CUDA 12.6 / SM87). All three configs
served by the same rebuilt image `llamacpp-jetson:qwen38` (llama.cpp master with GDN/MTP)
with identical flags: `--ctx-size 8192 --parallel 6 --flash-attn on --batch-size 2048
--ubatch-size 512 --n-gpu-layers 99`. Sweep tool: `bench/engine_sweep.sh` (`n_predict=200`,
`temperature 0`, generic completions endpoint). Raw outputs: `results/raw/*.txt`.

## Throughput (aggregate + per-stream tok/s)

| Model | N=1 | N=3 (AGG / per) | N=6 (AGG / per) |
|---|---|---|---|
| **qwen-agentworld-35b (IQ4_NL)** | **29.3** | **66.0 / 22.0** | **79.8 / 13.3** |
| qwen3.8-27b (Q4_K_M, MTP) | 11.1 | 26.1 / 8.7 (accept 59%) | 38.5 / 6.4 (accept 60%) |
| qwen3.8-27b (Q4_K_M, no MTP) | 8.1 | 16.9 / 5.6 | 21.1 / 3.5 |

**AgentWorld wins throughput at every concurrency level by ~2.1–2.6×.** This is expected:
A3B MoE routes ~3B active parameters per token, so decode is bandwidth-cheap; Qwen3.8-27B
is 27B dense, so every decoded token touches all 27B weights. MTP recovers some ground
(+37–83% over plain Q4) but doesn't close the gap.

## Latency (single-request chat/completions, temperature 0)

Same image + flags. Prompts from the quality mini-eval below.

| Test | agentworld | 3.8-27B Q4 | 3.8-27B MTP |
|---|---|---|---|
| json_extract  | 1.3s | 3.0s | 2.1s |
| tool_call     | 1.5s | 5.0s | 3.2s |
| instr_follow  | 0.6s | 1.7s | 1.6s |

AgentWorld is 2–3× faster end-to-end on short agent-shaped prompts.

## Quality mini-eval (3 prompts × 3 models, `enable_thinking=false`)

Prompts: (a) JSON extraction with strict schema, (b) single tool-call formatted as JSON,
(c) instruction-following (three UPPERCASE fruits, comma-separated).

| Model | Pass rate |
|---|---|
| qwen-agentworld-35b | 3/3 |
| qwen3.8-27b (Q4)    | 3/3 |
| qwen3.8-27b (MTP)   | 3/3 |

All three pass. Sample too small to differentiate quality; the throughput gap is the
decisive signal for a serving decision. Raw responses in `results/raw/quality-*.txt`.

## What Qwen3.8-27B has that AgentWorld doesn't

- **Native vision** (VLM, `mmproj-F16/F32` bundled). AgentWorld is text-only.
- **Native 262K context** with the GDN hybrid — only 16/64 layers hold full KV, so long
  context is memory-cheap on this box (verified upstream: ~5 GB KV at 128K).
- **Newer arch / newer training data** (shipped 2026-08-14).

None of these are load-bearing for the primary swap-router slot today — most `:8010`
traffic is text agent orchestration, at concurrency 1–3 (subagent bursts), where
AgentWorld's per-token cost dominates.

## VERDICT — **COEXIST (keep AgentWorld primary, add 3.8-27B as an on-demand preset)**

1. **AgentWorld-35B stays the `:8010` default.** No config change needed. It beats
   Qwen3.8-27B on every concurrency point by ~2.5× at parity quality on the agentic
   prompts that dominate its actual traffic.
2. **Add Qwen3.8-27B (MTP) as a `:8010` swap-loadable preset** for the two workloads
   it uniquely serves:
   - **Vision-input tasks** (only qwen3.8-27b and llama.cpp `mmproj` on this box handle
     images natively).
   - **Long-context sweeps** (>65K, the AgentWorld preset ceiling), leveraging the
     GDN hybrid's cheap KV.
3. **Do NOT add non-MTP Qwen3.8-27B.** MTP is strictly better on this hardware
   (+37–83% throughput, no quality regression, same disk footprint).

### Router.ini diff (proposed, not yet applied)

```ini
[qwen3.8-27b-mtp]
model            = /models/Jackrong_Qwen3.8-27B-MTP-Q4_K_M/Qwen3.8-27B-MTP-Q4_K_M.gguf
ctx-size         = 65536      ; can raise to 131072 if a caller needs it (~5GB extra KV)
n-gpu-layers     = 99
parallel         = 1          ; matches other :8010 presets — bump only for a bench window
no-warmup        = 1
jinja            = 1
flash-attn       = on
spec-type        = draft-mtp
spec-draft-n-max = 3
batch-size       = 2048
ubatch-size      = 512
```

No Hermes slot change unless a caller specifically wants VLM/long-ctx routing (the
current slots are text-only and better served by AgentWorld).

## Method notes / limits

- Fair comparison only on the same image + flags. Prior 2026-08-19 numbers (memory
  `project_qwen38_engine_benchoff_20260819`) used a different container config; the
  matched-config MTP sanity run here (11.1/26.1/38.5) reproduces the prior baseline at
  low N (9.9/27.9) within ±15% and is cleaner at N=6 (38.5 vs prior 20.9), consistent
  with a correctly-set `--parallel 6`.
- Quality eval is 3 prompts × 3 models — enough to rule out catastrophic regression, not
  enough to detect nuanced quality differences. If we ever want to promote 3.8-27B for
  reasoning-heavy work, run it through `~/grounder-bench` or a proper synth-harness slot
  eval before deciding.
- No vLLM number here — the prior bench-off already established vLLM 0.20 as the
  concurrency winner for 3.8-27B; that comparison stands and is orthogonal to
  "which model in the llama.cpp swap slot". Deploying vLLM as a permanent slot is
  question #2 in the queued next-session split.

## Deliverables

- Sweep raw: `results/raw/{agentworld-35b-Q4,qwen38-27b-Q4,qwen38-27b-MTP}.txt`.
- Quality raw: `results/raw/quality-{agentworld,qwen38-Q4,qwen38-MTP}.txt`.
- Bench script (reusable, unchanged): `bench/engine_sweep.sh`.
- Quality eval script: `/tmp/.../scratchpad/quality_eval.py` (small; not committed).
