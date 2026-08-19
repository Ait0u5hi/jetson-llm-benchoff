# Jetson Orin LLM Engine Bench-off: Qwen3.8-27B (GDN hybrid VLM) on SM87

A concurrency-aware, multi-engine benchmark and reproducible eval harness for running the
**Qwen3.8-27B** family (architecture `Qwen3_5ForConditionalGeneration`, a dense Gated-DeltaNet
hybrid vision-language model) on the **NVIDIA Jetson AGX Orin** (JetPack 6, CUDA 12.6, compute
capability 8.7).

Measured on real hardware, 2026-08-19. This repo answers the question the existing Jetson guides do
not: **how do the engines actually behave under concurrency, and which ones even run this
architecture on SM87?**

## Why this exists (and how it differs from prior work)

Two good resources already cover "how to run vLLM on Jetson Orin":
[MarkWind85/vllm-jetson-orin](https://github.com/MarkWind85/vllm-jetson-orin) (single-stream vLLM vs
llama.cpp numbers, GPTQ, Qwen3.5-35B-A3B MoE) and
[jetson-ai-lab.com/models](https://www.jetson-ai-lab.com/models/) (deployment instructions, no
numbers). This repo builds on them and adds what neither has:

| Contribution | MarkWind85 | jetson-ai-lab | This repo |
|---|---|---|---|
| Concurrency scaling (1/3/6 aggregate tok/s) | no | no | **yes** |
| MTP self-speculation, measured + accept rate | no | no | **yes** |
| Prefix-caching TTFT impact | no | no | **yes** |
| Gated-DeltaNet hybrid VLM family (`qwen3_5`) | no (A3B MoE) | partial | **yes** |
| Engine-viability matrix with root cause (why some engines do not run) | no | no | **yes** |
| Reusable multi-engine eval harness | single-stream | no | **yes** |

## Headline results

Qwen3.8-27B, context 8192, aggregate decode throughput (tokens/sec):

| Concurrency | llama.cpp | llama.cpp + MTP | vLLM 0.20 (AWQ-INT4) |
|---|---|---|---|
| 1 | 8.0 | 9.9 | 8.9 |
| 3 | 17.0 | 27.9 | 27.2 |
| 6 | 15.7 | 20.9 | **51.4** |

Two findings drive the recommendation:
1. **llama.cpp saturates at about 3 concurrent** (its aggregate ceiling is ~17 tok/s and it regresses
   at 6). **vLLM scales near-linearly** (8.9, 27, 51) with per-stream throughput staying flat, and it
   was not saturated at 6.
2. **MTP self-speculation helps at every concurrency and does not collapse under batching** here
   (+24 to +64 percent), because 27B decode on the Orin is memory-bandwidth-bound, so the GPU is not
   compute-saturated and speculation amortizes weight reads. Draft accept rate 50 to 58 percent.

Extras:
- **Prefix caching** (vLLM `--enable-prefix-caching`): 5x faster time-to-first-token on a shared
  3,285-token prompt (7.0s cold, 1.4s warm). This captures the practical benefit of SGLang
  RadixAttention without SGLang.
- **Long context is cheap** on this hybrid: 128K context costs about 5 GB of KV on llama.cpp
  (only 16 of 64 layers hold KV; the 48 Gated-DeltaNet layers keep a small fixed recurrent state),
  versus 20 to 40 GB for a same-size dense transformer. Decode barely degrades with depth.

## Recommendation

- **Many concurrent requests (subagents, pipelines): vLLM 0.20, AWQ-INT4, prefix caching on.**
- **Low concurrency or minimal footprint: llama.cpp + MTP.** Also the simplest to deploy.
- Both use INT4 / Q4 quants under 24 GB, so both are portable to a 24 GB Ampere card (SM86).

## Engine viability on SM87 (the part nobody documents)

Only two of five engines actually run this model on JetPack 6. The failures are version-entanglement,
not fundamental incompatibility, and the root causes are the useful part:

| Engine | Runs Qwen3.8 on SM87 | Root cause |
|---|---|---|
| **llama.cpp** | yes | needs build >= b10419 for the GDN arch; rebuild image with `CUDA_ARCH=87` |
| **vLLM 0.20** | yes | GDN Triton/FLA kernels run; the older "FLA broken on Ampere" issue (#40124) is fixed |
| **SGLang** | no | latest (0.5.13+) moved to CUDA 13 (incompatible with JetPack 6); the last CUDA-12 version (0.5.12) supports the arch but its compressed-tensors path cannot load an asymmetric-int4-group scheme |
| **TokenSpeed** | no | supports the arch, but its aarch64 kernels are cp311+ while the Jetson prebuilt torch is cp310-only |
| **TensorRT-LLM** | not tested | GDN arch not yet in the support matrix; requires host-side engine compilation |

See [`results/engine-viability.md`](results/engine-viability.md) for the full breakdown.

## Reproduce

1. Stand up an engine (see [`recipes/`](recipes/)). The vLLM-on-JetPack-6 recipe documents the
   non-obvious dependency fixes (cuDSS, torch ABI match, undeclared deps, flashinfer version bypass).
2. Run the harness against any OpenAI-compatible endpoint:
   ```bash
   bench/engine_sweep.sh http://localhost:8100/v1/completions qwen38-vllm "1 3 6"
   bench/prefix_cache_probe.sh http://localhost:8100/v1/completions qwen38-vllm
   ```
   The harness is engine-agnostic (works against vLLM, llama.cpp, SGLang, any OpenAI endpoint) and
   parses native llama.cpp timings for MTP draft accept rate when present.

## Hardware

NVIDIA Jetson AGX Orin 64 GB, L4T R36.5.2 (JetPack 6.x), CUDA 12.6, GPU compute capability 8.7,
61 GiB unified LPDDR5.

## License

Apache-2.0. Model weights are the property of their respective publishers (Qwen, Apache-2.0).

## Credits

Builds on prior Jetson inference work by
[MarkWind85](https://github.com/MarkWind85/vllm-jetson-orin),
[thehighnotes](https://huggingface.co/thehighnotes/vllm-jetson-orin), and the
[NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/).
