# Results: Qwen3.8-27B on Jetson AGX Orin 64 GB (SM87)

Hardware: NVIDIA Jetson AGX Orin 64 GB, L4T R36.5.2 (JetPack 6.x), CUDA 12.6, SM 8.7, 61 GiB unified.
Date: 2026-08-19. Context length 8192 unless noted. Aggregate = sum of decode tokens / wall clock.

## Throughput vs concurrency

| Concurrency | llama.cpp Q4_K_M | llama.cpp + MTP | vLLM 0.20 AWQ-INT4 |
|---|---|---|---|
| 1 | 8.0 | 9.9 | 8.9 |
| 3 | 17.0 | 27.9 | 27.2 |
| 6 | 15.7 | 20.9 | 51.4 |

- llama.cpp aggregate peaks at N=3 and regresses at N=6 (fixed `--parallel` slots saturate).
- vLLM scales near-linearly; per-stream stays ~8.6 tok/s at N=6 (from 8.9 at N=1). Not saturated at 6.

## MTP self-speculation (llama.cpp, `--spec-type draft-mtp --spec-draft-n-max 3`)

| Concurrency | no-MTP agg | MTP agg | gain | draft accept |
|---|---|---|---|---|
| 1 | 8.0 | 9.9 | +24% | 50% |
| 3 | 17.0 | 27.9 | +64% | 58% |
| 6 | 15.7 | 20.9 | +33% | 56% |

MTP wins at every concurrency (decode is memory-bandwidth-bound, so the GPU is not compute-saturated
and speculation amortizes weight reads). flash-attn was neutral for decode.

## Prefix caching (vLLM `--enable-prefix-caching`)

Shared 3,285-token prompt, `max_tokens=1` to isolate prefill:
- cold (first sight): 7.0 s
- warm (cache hit, new suffix): 1.4 s
- **5x faster time-to-first-token.** Decode throughput unchanged (prefix caching affects prefill only).

## Long context (llama.cpp)

128K context loads and costs ~5 GiB additional KV (only 16 of 64 layers hold KV; the 48
Gated-DeltaNet layers keep a small fixed recurrent state). Prefill ~242 tok/s on a 14K-token prompt;
decode at 16K depth 7.9 tok/s vs 8.3 shallow (minimal degradation).

## Quant / sampling used

- llama.cpp: `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_M + mmproj-F16; MTP build `Jackrong/Qwen3.8-27B-MTP-GGUF`.
- vLLM: `cyankiwi/Qwen3.8-27B-AWQ-INT4` (compressed-tensors, asymmetric int4, group_size 32, W4A16), 20 GB.
- Sampling (thinking): temp 1.0, top_p 0.95, top_k 20, min_p 0. Instruct: temp 0.7, top_p 0.80, presence_penalty 1.5.

## Software versions (for reproducibility)

- Board: L4T R36.5.2 (JetPack 6.x), CUDA 12.6, GPU SM 8.7.
- vLLM: `vllm==0.20.0+cu126`, `torch==2.11.0` (jetson-ai-lab cu126 index), `nvidia-cudss-cu12==0.8.0.10`,
  `compressed-tensors==0.18.0`, `transformers==5.15.1`, `xgrammar`. Run with `FLASHINFER_DISABLE_VERSION_CHECK=1`.
- llama.cpp: built from `master` commit `dc72703`, `-DCMAKE_CUDA_ARCHITECTURES=87`.
- SGLang (blocked): `sglang==0.5.12`, `sglang-kernel==0.4.2.post2`, `flashinfer-python==0.6.11.post1`.

Raw per-run harness output is in [`raw/`](raw/): `llamacpp-Q4-noFA.txt`, `llamacpp-Q4-flashattn.txt`,
`llamacpp-Q4-MTP.txt`, `vllm-0.20-AWQ-INT4.txt`.
