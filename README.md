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
  setup was net-negative on an A3B MoE in prior fleet testing.
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

## Layout

- `results/` measured numbers with full hardware and checkpoint provenance
- `bench/` the eval harness (concurrency sweep, prefix-cache probe, MTP accept-rate) for any
  OpenAI-compatible endpoint
- `recipes/` the vLLM-0.20-on-JetPack-6 install recipe (the non-obvious dependency fixes) and serve script

## Reproduce

See `bench/README.md`. In short: serve the model on any engine, then

```
bench/engine_sweep.sh http://localhost:8100/v1/completions qwen38-vllm "1 3 6"
bench/prefix_cache_probe.sh http://localhost:8100/v1/completions qwen38-vllm
```

## Scope and honesty

Single-board, single-run numbers, meant as a directional decision aid and a reproducible harness, not
a leaderboard. Hardware, checkpoints, flags, and dates are recorded in `results/` so you can rerun and
disagree. Corrections and other-board results welcome.

License: Apache-2.0.
