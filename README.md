# jetson-llm-benchoff

Concurrency-aware, multi-engine inference benchmarks for **Qwen3.8-27B** (the `qwen3_5`
Gated-DeltaNet hybrid vision-language family) on the **NVIDIA Jetson AGX Orin (SM 8.7 / JetPack 6 /
CUDA 12.6)**, plus a reusable eval harness and a reproducible vLLM install recipe.

Most Jetson LLM writeups answer "how do I run it". This repo answers **"which engine and quant do I
pick, and how fast is it under real concurrency"** for a hybrid-attention VLM, with the negative
results (which engines do *not* work on SM87, and exactly why) documented so you do not repeat the
dead ends.

## What is new here (vs prior art)

This builds on and credits [MarkWind85/vllm-jetson-orin](https://github.com/MarkWind85/vllm-jetson-orin)
(single-stream vLLM-vs-llama.cpp numbers for Qwen3.5-35B-A3B GPTQ) and the
[Jetson AI Lab](https://www.jetson-ai-lab.com/models/) run guides. It adds what neither has:

| Contribution | Prior art | Here |
|---|---|---|
| Concurrency scaling (1 / 3 / 6 aggregate tok/s) | not covered | measured, and it flips the verdict |
| MTP self-speculation impact + accept rate | not covered | measured (+24 to +64%) |
| Prefix-caching TTFT impact | not covered | measured (5x) |
| Dense Gated-DeltaNet **VLM** (`qwen3_5`) | A3B MoE only | Qwen3.8-27B, vision included |
| Engine-viability matrix with root cause | not covered | why SGLang / TokenSpeed do not run on SM87 |
| Reusable multi-engine eval harness | single-stream scripts | concurrency + prefix + MTP, any OpenAI endpoint |

## Headline result

Aggregate decode throughput, ctx 8192, Qwen3.8-27B, AGX Orin 64GB. `[measured]` 2026-08-19.

| Concurrency | llama.cpp Q4 | llama.cpp Q4 + MTP | vLLM 0.20 AWQ-INT4 |
|---|---|---|---|
| 1 | 8.0 | 9.9 | 8.9 |
| 3 | 17.0 | **27.9** | 27.2 |
| 6 | 15.7 | 20.9 | **51.4** |

**The decision rule this produces:**

- **llama.cpp continuous batching saturates at about 3 concurrent** (~17 to 28 tok/s ceiling,
  regardless of MTP). Best for low concurrency and simplicity, and its 128K context is cheap
  (~5 GB KV, because the hybrid keeps KV only on the 16 full-attention layers of 64).
- **vLLM PagedAttention scales close to linearly** (8.9 to 27 to 51, per-stream barely drops) and was
  not saturating at 6. Best for many concurrent agents or pipelines. At 6 concurrent it is 2.5x
  llama.cpp+MTP.
- **MTP self-speculation is a real win and does not collapse under batching here** (+24 to +64%
  across N=1 to 6, 50 to 58% draft accept) because 27B decode on Orin is memory-bandwidth-bound, so
  the GPU is not compute-saturated and spec amortizes weight reads. Contrast: a draft-*model* spec
  setup was net-negative on an A3B MoE in prior testing on this platform.
- **vLLM `--enable-prefix-caching`: 5x faster TTFT** on a shared 3,285-token prompt (7.0s cold to
  1.4s warm). This captures the headline advantage of SGLang RadixAttention without needing SGLang.

## Engine viability on SM 8.7 (only two of five run this model)

`[measured]` 2026-08-19 unless noted.

| Engine | Runs Qwen3.8-27B on SM87 | Root cause if not |
|---|---|---|
| **llama.cpp** | yes | rebuilt from master with `CUDA_ARCH=87`; needs a build past GDN support |
| **vLLM 0.20** | yes | Triton/FLA GDN kernels run; the older #40124 "FLA broken on Ampere" is resolved |
| **SGLang 0.5.12** | installs, supports `qwen3_5`, but cannot load this quant | its compressed-tensors path lacks the asymmetric int4 group scheme (vLLM has it) |
| **SGLang 0.5.13+** | no | moved to CUDA 13; incompatible with JetPack 6 / CUDA 12.6 |
| **TokenSpeed 0.1.0** | no | aarch64 kernels are cp311+, Jetson prebuilt torch is cp310 only |
| **TensorRT-LLM** | not tested | GDN arch support + host-side engine compile; deprioritized once vLLM won |

The reusable lesson: on an edge board, check **(arch supported) x (CUDA major matches) x (cpXY matches)
x (SM in the prebuilt wheel)** before assuming an engine runs. Any one of these silently blocks it.

## Follow-up benches

Two follow-ups extend the flagship bench-off along different axes. Both use the same box, same
image, and the same reusable harness in `bench/`.

### Model vs model: is Qwen3.8-27B better than the incumbent primary?

The engine bench-off answers "which engine serves Qwen3.8-27B best". A separate question is
whether Qwen3.8-27B should replace the current router primary, **Qwen-AgentWorld-35B-A3B** (an A3B
MoE, ~3B active parameters). Same image, matched flags (`--parallel 6 --flash-attn on --ctx-size
8192`), llama.cpp on both. `[measured]` 2026-08-19.

| Model | N=1 | N=3 (AGG / per) | N=6 (AGG / per) |
|---|---|---|---|
| **qwen-agentworld-35b (IQ4_NL)** | **29.3** | **66.0 / 22.0** | **79.8 / 13.3** |
| qwen3.8-27b (Q4_K_M, MTP) | 11.1 | 26.1 / 8.7 | 38.5 / 6.4 |
| qwen3.8-27b (Q4_K_M) | 8.1 | 16.9 / 5.6 | 21.1 / 3.5 |

**AgentWorld wins ~2.1 to 2.6x at every concurrency point** because A3B MoE decode touches ~3B
active params vs 3.8-27B's 27B dense. Verdict: **coexist** — keep AgentWorld as the primary
`:8010` slot, add 3.8-27B-MTP as a swap-preset only for VLM + long-context (>65K) work that
AgentWorld cannot serve. Full doc: [`results/agentworld_vs_qwen38.md`](results/agentworld_vs_qwen38.md).

### Perf-angles: TTFT, long-ctx degradation, prefix-cache, KV footprint

Four additional serving-relevant lenses on the same three models, measured at `--parallel 1`
(matches the reference `:8010` serving config used throughout this repo; safer than the
`parallel=6` bench, which triggered a Tegra NVRM DCE-RPC hard-hang once and needs a physical
power cycle to recover).
`[measured]` 2026-08-19.

- **TTFT p50 (warm, 20 gen tokens):** Qwen3.6-35B 0.20s < AgentWorld 0.40s < Qwen3.8-27B-MTP
  1.11s. Tight P99 (within ~10% of P50) — no long-tail hitches on any model.
- **Long-context tps at 32K:** AgentWorld 0.96 / 3.6-35B 0.71 / 3.8-27B-MTP 0.22. Prompt-eval,
  not decode, dominates at scale.
- **Prefix-cache speedup on 4K shared prefix:** 3.8-27B-MTP **2.26x**, 3.6-35B 1.59x,
  AgentWorld 1.30x. 3.8-27B benefits most because its prompt-eval cost is highest to elide.
- **KV memory at max ctx:** 3.8-27B-MTP holds **128K ctx in 27.87 GB** total (weights + KV),
  vs AgentWorld's 65K ceiling in 20.08 GB. GDN hybrid keeps KV on only 16 of 64 layers, so
  128K is memory-viable on 61 GB unified LPDDR5 even with co-resident auxiliary models loaded.

Full doc: [`results/perf_angles_2026-08-19.md`](results/perf_angles_2026-08-19.md).

## Layout

- `results/` measured numbers with full hardware and checkpoint provenance
  - `qwen3.8-27b-agx-orin.md` — the flagship engine bench-off
  - `agentworld_vs_qwen38.md` — model-vs-model head-to-head (matched flags, llama.cpp on both)
  - `perf_angles_2026-08-19.md` — TTFT, long-ctx, prefix-cache, KV footprint
  - `raw/` — raw sweep + probe outputs, one file per (engine × model) or (angle × model)
- `bench/` reusable probe scripts (concurrency sweep, prefix-cache probe, perf-angles)
  for any OpenAI-compatible endpoint
- `recipes/` the vLLM-0.20-on-JetPack-6 install recipe (the non-obvious dependency fixes) and serve script

## Reproduce

See `bench/README.md`. In short: serve the model on any engine, then

```
bench/engine_sweep.sh http://localhost:8100/v1/completions qwen38-vllm "1 3 6"
bench/prefix_cache_probe.sh http://localhost:8100/v1/completions qwen38-vllm
bench/perf_angles.sh <alias> <gguf_relpath> <max_ctx> [spec_flags]
```

The perf-angles script is llama.cpp-specific (it manages a container lifecycle for
memory-footprint measurement); the other two work against any OpenAI-compatible
`/v1/completions` endpoint.

`perf_angles.sh` mounts a GGUF directory into the container at `/models`. Set
`MODELS_DIR=/path/to/your/gguf` before invoking; defaults to `/models` if unset.

## Scope and honesty

Single-board, single-run numbers, meant as a directional decision aid and a reproducible harness, not
a leaderboard. Hardware, checkpoints, flags, and dates are recorded in `results/` so you can rerun and
disagree. Corrections and other-board results welcome.

License: Apache-2.0.
