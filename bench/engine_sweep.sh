#!/bin/bash
# Concurrency throughput sweep for any OpenAI-compatible /v1/completions endpoint.
# Engine-agnostic (vLLM, llama.cpp llama-server, SGLang, ...). Reports aggregate and
# per-stream decode tok/s at each concurrency level, plus MTP draft accept rate when the
# server returns native llama.cpp timings.
#
# Usage: engine_sweep.sh <completions_url> <model_name> ["1 3 6"] [n_predict]
#   engine_sweep.sh http://localhost:8100/v1/completions qwen38-vllm "1 3 6" 200
set -euo pipefail
URL="${1:?completions url}"; MODEL="${2:?model name}"; LEVELS="${3:-1 3 6}"; NPRED="${4:-200}"
PROMPT="Explain the theory of general relativity in detail, step by step."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

req() { printf '{"model":"%s","prompt":"%s","max_tokens":%s,"temperature":0,"cache_prompt":false}' "$MODEL" "$PROMPT" "$NPRED"; }

sweep() {
  local n=$1; rm -f "$TMP"/r_*.json
  local start end; start=$(date +%s.%N)
  for i in $(seq 1 "$n"); do
    curl -s --max-time 600 "$URL" -H "Content-Type: application/json" -d "$(req)" -o "$TMP/r_$i.json" &
  done
  wait
  end=$(date +%s.%N)
  python3 - "$n" "$start" "$end" "$TMP" <<'PY'
import json, glob, sys
n, start, end, tmp = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
wall = end - start; toks = 0; dn = 0; da = 0
for f in glob.glob(f"{tmp}/r_*.json"):
    try:
        d = json.load(open(f))
        # OpenAI usage (vLLM/SGLang) or native llama.cpp timings
        toks += d.get("usage", {}).get("completion_tokens", 0) or d.get("timings", {}).get("predicted_n", 0)
        t = d.get("timings", {})
        dn += t.get("draft_n", 0); da += t.get("draft_n_accepted", 0)
    except Exception:
        pass
acc = f" | draft_accept={100*da/dn:.0f}%" if dn else ""
print(f"N={n:>2} | wall={wall:5.1f}s | gen_tokens={toks:>5} | AGG={toks/wall:6.1f} tok/s | per-stream~{toks/wall/int(n):5.1f}{acc}")
PY
}

echo "== engine_sweep: $MODEL @ $URL (n_predict=$NPRED) =="
for lvl in $LEVELS; do sweep "$lvl"; done
