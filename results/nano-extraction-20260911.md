# Hindsight extraction: nano gemma-4-E2B vs AGX `cheap` (Qwen3-8B) — 2026-09-11

Question: would moving Hindsight's fact-extraction LLM off the nano (to the AGX `cheap` role via the
MLflow gateway) lose quality? Pre-registered metric: identifier recall (run ids, hashes, numbers,
PR numbers present in the source closeout that survive verbatim into the extracted facts), JSON
validity, fact count, wall seconds. Same instruction, temperature 0.1, max_tokens 1500.
Script: `bench/extract_compare.py`. Raw: `results/raw/nano-extract-20260911/`.

Caveats stated up front: n = 5 closeouts (effective n = 4 shared between arms — a new closeout
entered the "5 most recent" set between runs); the first 8B run was in THINKING mode and is
reported only as a confound; recall counts verbatim identifiers, not semantic correctness.

| Arm | Docs | Mean recall | JSON ok | Mean s/doc | Mean facts |
|---|---|---|---|---|---|
| nano gemma-4-E2B Q4_K_M (`--reasoning off`, ctx 32k) | 5 | 0.39 | 5/5 | 111.1 | 9.0 |
| AGX cheap 8B, thinking ON (confounded) | 5 | 0.21 | 4/5 | 37.9 | 2.4 |
| AGX cheap 8B, thinking OFF | 5 | 0.30 | 5/5 | 23.5 | 10.2 |

On the 4 shared docs: gemma 0.45 vs 8B-no-think 0.30. The gap is one long document (this
session's own closeout, 19-22 identifiers): gemma extracted 23 facts / recall 0.64, the 8B 6 facts /
recall 0.05. Elsewhere they tie.

Verdict (measure-first, per the user's 2026-09-11 decision): NO quality win from moving; gemma is at
least as good on identifier recall, the 8B is ~5x faster. Keep extraction on the nano for now; the
nano RAM problem is handled by the daily recycle + (pending) zram. Revisit with a larger n and a
semantic judge if the nano keeps thrashing.
