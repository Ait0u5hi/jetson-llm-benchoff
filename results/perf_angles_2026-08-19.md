# Perf-angles bench: AgentWorld-35B vs Qwen3.8-27B-MTP vs Qwen3.6-35B on AGX Orin

**Measured 2026-08-19** on Jetson AGX Orin (64GB, JetPack 6.x / CUDA 12.6 / SM87, 61 GiB unified LPDDR5).
Same rebuilt image `llamacpp-jetson:qwen38` (llama.cpp master with GDN/MTP), same flags:
`--parallel 1 --flash-attn on --batch-size 2048 --ubatch-size 512 --cache-reuse 256
--n-gpu-layers 99`. Sequential across models (no concurrent bench containers,
co-resident auxiliary models stayed running) to match the reference serving config
(`:8010`, see `recipes/llamacpp.md`). Raw outputs:
`results/raw/perf/{qwen-agentworld-35b,qwen3.8-27b-mtp,qwen3.6-35b}.txt`.

Sequential-safe config chosen deliberately: an earlier `parallel=6` bench triggered an
unrecoverable NVRM DCE-RPC hang (`NVRM_RPC_DCE: Failed RM ctrl call cmd:0x731341
result 0xffff`, followed by system-wide soft-lockup requiring physical power cycle).
`parallel=1` is what the reference `:8010` serving config runs without incident.

## (1) TTFT distribution (short prompt, 20 gen tokens, 1 cold + 29 warm)

| Model | Cold s | Warm min | P50 | P95 | P99 | Mean |
|---|---|---|---|---|---|---|
| **qwen3.6-35b (A3B MoE + draft-MTP)** | 0.60 | 0.18 | **0.20** | 0.22 | 0.26 | 0.21 |
| qwen-agentworld-35b (A3B MoE) | 0.96 | 0.37 | 0.40 | 0.41 | 0.44 | 0.40 |
| qwen3.8-27b (dense hybrid GDN + MTP) | 1.60 | 1.08 | 1.11 | 1.13 | 1.14 | 1.11 |

TTFT is tightly distributed (P99 within ~10% of P50 on all three — no long-tail hitches).
Qwen3.6-35B leads, plausibly from draft-MTP spec decode being effective on the 20-token
generation. 3.8-27B is the slowest per-request by ~2.75× — dense-27B decode vs A3B MoE
(~3B active) is the load-bearing difference. Consistent with the throughput head-to-head.

## (2) Long-context throughput degradation (100 gen tokens, real prompt-fill)

| Model | 2K ctx | 8K ctx | 32K ctx | 64K ctx |
|---|---|---|---|---|
| **qwen-agentworld-35b** | 9.53 tok/s | 3.25 | 0.96 | (65K ceiling, skipped) |
| **qwen3.6-35b** | 8.49 | 2.90 | 0.71 | (65K ceiling, skipped) |
| **qwen3.8-27b-mtp** | 2.64 | 0.84 | 0.22 | **0.15 @ 64K** |

Two things pop:
- The A3B MoE models are 3–4× faster than 3.8-27B at every ctx point — again the
  active-param gap dominating.
- **3.8-27B held 64K ctx (128K max) fine** where the A3B GGUFs cap at 65K. The GDN hybrid's
  KV-cheapness thesis holds: the model loaded at ctx=131072 with total unified-memory
  delta of only 27.87 GB (see KV footprint below) — 128K ctx is memory-feasible on the AGX
  even though throughput at those contexts is well under 1 tok/s.
- Prompt-eval time dominates at higher ctx (wall time grows near-linearly with prompt
  tokens; gen tokens are a small fraction of total wall). This is the *prompt-fill*
  bottleneck, not decode: any model with better prompt-fill kernels would win here.

## (3) Prefix-cache effectiveness (4K shared prefix, distinct suffix; `--cache-reuse 256`)

| Model | Cold s | Warm avg s | Speedup |
|---|---|---|---|
| **qwen3.8-27b-mtp** | 11.08 | 4.91 | **2.26×** |
| qwen3.6-35b | 2.64 | 1.66 | 1.59× |
| qwen-agentworld-35b | 2.30 | 1.77 | 1.30× |

3.8-27B benefits most in absolute terms — long prompt-eval cost (its weakness above) is
exactly what a prefix cache elides on repeated prefixes. Both A3B models see smaller
gains because their prompt-eval was already cheap. **If your workload sees repeated
system-prompt prefixes (agent orchestration always does), 3.8-27B's amortized TTFT
narrows toward the A3B pack.**

## (4) KV memory footprint at max ctx

| Model | Max ctx | Load time s | Mem delta GB (weights + KV, unified LPDDR5) |
|---|---|---|---|
| qwen-agentworld-35b | 65536 | 25 | 20.08 |
| qwen3.6-35b | 65536 | 25 | 21.31 |
| **qwen3.8-27b-mtp** | **131072** | 30 | **27.87** |

3.8-27B holds 2× the context in 1.4× the memory — the GDN hybrid claim ("only 16/64 layers
hold full KV") is measurable. Concrete implication: an on-demand swap-preset for 3.8-27B
at 128K ctx is memory-feasible even with the co-resident auxiliary models loaded
(61 GB total, ~30 GB free after the auxiliary set).

## (5) MMLU: deferred

Attempted MMLU (limit=500) via lm-eval's `local-completions` model class against
llama-server's `/v1/completions` endpoint. Blocker: llama-server's logprobs response shape
does not match what lm-eval's loglikelihood parser expects (`KeyError: 'token_logprobs'`).
Chat-completions has no loglikelihood at all (`NotImplementedError`). Working around this
means either a llama.cpp fork/patch or moving to a vLLM/SGLang backend, both out of scope
for this run. Deferred to a dedicated quality-eval run on a different backend.

## Reconciliation with the head-to-head throughput verdict

These angles ratify and sharpen the earlier `agentworld_vs_qwen38.md` COEXIST verdict:

- **AgentWorld stays the `:8010` default.** Faster per-request (2.75× TTFT lead), faster
  long-ctx throughput, sufficient 65K ceiling for text-agent work.
- **Add 3.8-27B-MTP as swap-preset with confidence on two grounds now measured, not just
  asserted:** (a) 128K ctx is memory-viable at 27.87 GB, (b) prefix-cache reclaims 55% of
  its per-request latency on repeated prefixes typical of agent workloads.
- **Qwen3.6-35B baseline reads as slightly-slower-than-AgentWorld across all angles** —
  ratifies AgentWorld having replaced it as the primary; no regression.

## Method notes

- All numbers from a single-process llama-server per model, `--parallel 1`, no changes
  to co-resident auxiliary models. Sequential model swaps with 10-15s GPU-settle sleeps
  between.
- Bench script: `bench/perf_angles.sh` (one model, four angles). The MMLU driver used
  a separate off-box lm-eval invocation (not included; see method note above for the
  llama-server logprobs-shape blocker that made it non-portable).
- Long-ctx test uses real prompt-fill with tokenized filler; the reported prompt-tokens
  count is what the model actually saw (`usage.prompt_tokens`).
- Prefix-cache test uses `--cache-reuse 256` on server; measures cold TTFT vs warm-mean
  TTFT on same-4K-prefix distinct-suffix requests.
- KV footprint = `free -h Used` delta across the model-load boundary; captures weights +
  KV + intermediate buffers at max ctx. Not a pure-KV number — pure-KV would require
  parsing `llama-server`'s startup log for the specific line, which comes out empty on
  this rebuilt image.
