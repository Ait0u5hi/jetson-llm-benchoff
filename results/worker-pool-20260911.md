# Worker-pool concurrency bench on AGX Orin (2026-09-11)

Question: does `--parallel 3` on ONE resident preset pay on this box, and which engine/image?
Method: `bench/pool_arm.sh` — standalone container per arm on :8011 (production router untouched,
27B/8B unloaded to make room), ctx 8192 per slot, `engine_sweep.sh` n_predict 200, GPU idle
(pi-harness bake-off finished; 0 foreign router requests during every sweep, checked per arm).
Raw: `results/raw/pool-20260911/`.

| Arm | Image | N=1 | N=3 agg / per-stream | N=6 agg / per-stream |
|---|---|---|---|---|
| Qwen3.8-27B MTP Q4_K_M, parallel 3 | `llamacpp-jetson:latest` (build 9552, PRODUCTION) | 10.0 | 10.6 / 3.5 | – |
| Qwen3.8-27B MTP Q4_K_M, parallel 3 | `llamacpp-jetson:qwen38` (master dc72703) | 11.2 | 25.7 / 8.6 (accept 59%) | – |
| AgentWorld-35B-A3B IQ4_NL, parallel 3 | `qwen38` | 29.8 | 66.0 / 22.0 | – |
| Qwen3-Coder-30B-A3B IQ4_NL, parallel 3 | `qwen38` | 45.9 | 97.1 / 32.4 | – |
| Qwen3.8-27B AWQ-INT4, vLLM 0.20 (alone) | `~/vllm-venv` | 9.5 | 27.2 / 9.1 | 51.6 / 8.6 |

Findings:
1. **The production image does not batch this model at `parallel = 3`** — three requests
   serialized (1.06x). The `qwen38` image batches (2.3x) and reproduces the 2026-08-19 numbers
   within 2%. Any pool on :8010 needs the router moved to the `qwen38` build (or a rebuild).
2. August numbers reproduce on `qwen38`: 27B-MTP 26.1→25.7, AgentWorld 66.0→66.0, vLLM
   27.2/51.4→27.2/51.6.
3. Criterion 1 of the plan (aggregate ≥ 1.5x and per-stream ≥ 4 tok/s): PASS for all four
   `qwen38`/vLLM arms; FAIL for the production image.
4. Worker-model choice still goes through the quality gate (graded tasks, 27B default) — this
   bench measures throughput only.
