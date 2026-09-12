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
| Gemma 4 E2B (deployed) | **94.6** | not published | IFBench 38.0; Tau2 avg 24.5; MMLU-Pro 60.0 | Gemma 4 tech report arXiv 2607.02770 Table 5 (HF card omits IFEval) |
| Gemma 4 E4B | **96.7** | not published | IFBench 44.0; Tau2 avg 42.2; ~3 GB Q4 (fits nano) | same |
| Qwen3.5-4B | 89.8 | v4 50.3 | TAU2 79.9, MMLU-Pro 79.1 (mode not split) | HF card |
| Granite 4.0 H-Micro 3B (hybrid Mamba2, 128k) | 84.3 avg | v3 57.6 | MMLU 67.4; GGUF | HF card |
| Granite 4.0 Micro 3B (dense) | 82.3 avg | v3 60.0 | | HF card |
| Phi-4-mini 3.8B | 70.1 | — | | tech report 2503.01743 |
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
- Extraction (CORRECTED 2026-09-12): the Gemma 4 technical report gives E2B IFEval 94.6, the
  highest of any sub-3B general model here, so the deployed gemma is NOT behind on instruction
  following; the earlier line 'Gemma 4 E2B publishes no IFEval' was wrong (the HF card omits it,
  the paper has it). LFM2.5-1.2B (86.2) remains interesting only for its 0.7 GB / 54 tok/s
  footprint. See the purpose-built section below for the real challengers.
- Concurrency within ~4 GB: no external evidence; plausible on footprints (1.2B Q4 0.7 GB +
  0.6B Q8 ~0.6 GB + KV). Our own nano runs both today at 2.4 GB available post-recycle.

## Second pass 2026-09-12: purpose-built extractors + structured-output benches

Hindsight's retain path is schema-driven [measured on the nano container,
`hindsight_api/engine/retain/fact_extraction.py`]: the LLM must return `{"facts":[{what, when,
where, who, why, fact_type, entities[], causal_relations?}]}` validated by pydantic, with
"verbose, never summarize" field instructions. That is a fixed JSON template over free text, i.e.
exactly what schema-conditioned extractor models are trained on.

| Candidate | Size / base | Evidence | Fit notes |
|---|---|---|---|
| **NuExtract-2.0-2B** | 2B, Qwen2-VL-2B, MIT | VAREX (arXiv 2603.15118, 2026-03): 90.8% exact match, zero schema echo, vs Qwen3-VL-2B 34.2%, InternVL3.5-2B 85.6%, Gemma 3n E2B 71.0% (image modality, text-normalised) | GGUF Q4_K_M 986 MB, Q8 1.65 GB; llama.cpp needs `--mmproj` even for text; template = `# Template:\n<json>\n<text>`; `verbatim-string` type maps to our identifier-recall metric; temp 0 |
| **NuExtract3 (4B)** | Qwen3.5-4B base, Apache 2.0, 131k ctx | NuMind internal bench (~600 docs): 0.651 vs Gemma-4-E4B 0.538, Qwen3.5-9B 0.479, Qwen3.5-4B 0.417; fewest failed outputs (27) | Vendor bench; no GGUF listed yet (W4A16 repo exists); ~2.5-3 GB at Q4, tight beside the embedder |
| **LFM2-1.2B-Extract** | 1.2B hybrid, GGUF Q4_K_M 731 MB / Q8 1.25 GB | Liquid: 5,000-doc eval (syntax, format, keyword faithfulness, judge); claims above Gemma 3 27B on nested multilingual objects; chart only, no table | Greedy temp 0; JSON/XML/YAML; 9 languages; 350M sibling; 2025-11-28 |
| Schematron-8B | 8B | Structured Output Benchmark (arXiv 2604.25359): JSON pass 0.990, value acc 0.754 (best open <=14B on text) | Too big for the nano; AGX `cheap` class |
| GLiNER2 / 2.5 | 74M-340M DeBERTa encoders, Apache 2.0, CPU-first | Schema JSON extraction API (`extract_json`), near GPT-5 on CrossNER (vendor) | Span extractor, not generative: cannot write the verbose `what` sentences Hindsight wants; only viable for an entities-only pre-pass |
| compact-relex (Qwen2.5-0.5B fine-tune) | 0.5B | arXiv 2606.22606 (2026-06): general RE micro-F1 0.83 vs GPT-5.4 0.69 zero-shot; checkpoint released | Relation triples, not our fact schema; shows fine-tuning beats scale for extraction |

Benchmarks worth knowing (none scores our exact candidates on text): VAREX (multimodal
structured extraction, 20 models, EM + compliance), Structured Output Benchmark (21 models, text +
image + audio, schema-then-value; finding: near-perfect JSON validity but only ~83% field accuracy
even at the top, and size does not predict quality), JSONSchemaBench (10k schemas; measures the
constrained-decoding ENGINE, e.g. llama.cpp grammar, not the model), IFEval/IFBench (format
compliance proxies), BFCL (tool calls, weak proxy).

Read-out v2: gemma-4-E2B is not weak on instruction following (94.6). The evidence that a
purpose-built extractor beats a general model at 2B is real but independent only for
NuExtract-2.0-2B on VAREX; NuExtract3 and LFM2-Extract are vendor-benched. Because Hindsight's
task is a fixed schema, the right next measurement is a 3-arm run of `bench/extract_compare.py`
on the nano: gemma-4-E2B (incumbent) vs NuExtract-2.0-2B Q8 vs LFM2-1.2B-Extract Q8, same 5
closeouts, identifier recall + pydantic-valid rate + s/doc, with a `--json-schema` grammar
(llama.cpp) so JSON validity is engine-enforced and the arms compare on values only. Gemma 4 E4B
(IFEval 96.7, ~3 GB) is the general-model upgrade path if the nano keeps >3.5 GB free after the
embedder.
