# Small extraction + embedding models for the nano — market pass 2026-09-11/12

Research-harness market run `20260912T020749Z` (report
`~/research-harness/pipelines/market/reports/2026-09-12-small-local-models-for-an-agent.md`)
had DEGRADED search (11 ddgs fallbacks) and found no IFEval/BFCL or MTEB retrieval score for any
candidate; verdict "cannot rank". The numbers below are a hand pass over primary model cards and
papers (vendor-reported unless noted). Deployed today: gemma-4-E2B (extraction) + Qwen3-Embedding-0.6B.

## Sub-3B extraction candidates (vendor cards; IFEval = instruction/format compliance)

| Model | IFEval | BFCL | Other | Source date |
|---|---|---|---|---|
| LFM2.5-1.2B-Instruct | **86.2** | v3 49.1 | IFBench 47.3, Multi-IF 61.0, MMLU-Pro 44.4; ~700 MB Q4 | HF card (2026) |
| Qwen3.5-2B non-thinking (default) | 61.2 | — | MMLU-Pro 55.3 | HF card, Feb 2026 |
| Qwen3.5-2B thinking | 78.6 | v4 43.6 | TAU2 48.8, MMLU-Pro 66.5 | same |
| Gemma 4 E2B (deployed) | not published | not published | Tau2 avg 24.5 (E4B 42.2), MMLU-Pro 60.0 | HF card |
| Qwen3-1.7B instruct (Liquid's table) | 73.7 | — | IFBench 21.3 | LFM card |
| Granite 4.0-1B (Liquid's table) | 79.6 | — | IFBench 21.0 | LFM card |
| Gemma 3 1B (Liquid's table) | 63.3 | — | | LFM card |

Caveats: cross-vendor tables, thinking-mode differences, no independent replication; none of these
is a strict-JSON extraction eval. Our own measured extraction metric (identifier recall, n=5) is
in `nano-extraction-20260911.md`: gemma-E2B 0.39, 8B 0.30. The June 2026 LFM2.5 rejection
(memory `project_lfm25_eval`) was the 8B-A1B as Hermes ORCHESTRATOR on tool selection; it says
nothing about the 1.2B as an extractor.

Nano throughput reference (yuvrajsingh.io, 2026-05-29, Orin Nano Super 8GB, llama.cpp CUDA
b9292, ctx 2048): LFM2.5-1.2B Q4_K_M 54.1 tok/s at 698 MB; LFM2.5-350M 115 tok/s at 219 MB;
SmolLM2-360M Q8 102 tok/s. Same post: after ~3 sequential model loads in one OS session, CUDA IOVA
fragmentation blocks large allocations (reboot workaround) — consistent with our daily recycle.

## Small embedders (MTEB; retrieval = NDCG@10)

| Model | Params | Dim | Ctx | MTEB Eng v2 retrieval | Other | Source |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B (deployed) | 596M | 1024 (MRL 32-1024) | 32k | 61.8 | Eng v2 mean 70.7, multilingual 64.3 | HF card |
| EmbeddingGemma-300M | 300M | 768 (MRL 128-768) | 2k | retrieval n/a | Eng v2 mean 69.7 (256d: 68.4); Code 68.8; Q4/Q8 GGUF | HF card |
| snowflake-arctic-embed-m-v2.0 | 305M | 768 | 8k | 58.4 | BEIR 55.5 | Granite R2 paper 2508.21085 |
| granite-embedding-english-r2 | 149M | 768 | 8k | 56.4 | BEIR 53.1, LongEmbed 67.8 | same |
| bge-base-en-v1.5 | 109M | 768 | 512 | 54.8 | BEIR 54.2 | same |
| granite-embedding-small-english-r2 | 47M | 384 | 8k | 53.9 | 199 docs/s on H100 vs 138-145 peers | same |
| bge-small-en-v1.5 | 33M | 384 | 512 | 53.9 | | same |
| nomic-embed-text-v1.5 | 137M | 768 | 8k | ~62 (legacy MTEB, not comparable) | AGX :8020 today | secondary |

## Read-out
- Embedding: Qwen3-Embedding-0.6B is the top retrieval scorer among sub-1B open models on the
  same table; EmbeddingGemma-300M is within ~1 point on the English mean at half the params and
  a quarter of the dims (256d), and is the only real RAM lever (~0.3 GB Q8 vs ~0.6-1.1 GB). Its
  2k context is a constraint for long closeouts (chunk first).
- Extraction: LFM2.5-1.2B is the strongest published instruction-following score at this size
  (86 IFEval, ~0.7 GB Q4, ~54 tok/s on a nano) and is a hybrid architecture (constant-size
  state, good at long prefixes). Gemma 4 E2B publishes no IFEval. Worth one measured arm in
  `bench/extract_compare.py` against gemma on identifier recall before any swap.
- Concurrency within ~4 GB: no external evidence; plausible on footprints (1.2B Q4 0.7 GB +
  0.6B Q8 ~0.6 GB + KV). Our own nano runs both today at 2.4 GB available post-recycle.
