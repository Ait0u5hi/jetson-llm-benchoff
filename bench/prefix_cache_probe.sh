#!/bin/bash
# Prefix-caching / prefill-reuse probe for an OpenAI-compatible /v1/completions endpoint.
# Sends a long shared prefix twice (different suffix). With prefix caching enabled, the second
# request should skip prefill and return time-to-first-token far faster.
#
# Usage: prefix_cache_probe.sh <completions_url> <model_name> [prefix_repeats]
#   prefix_cache_probe.sh http://localhost:8100/v1/completions qwen38-vllm 180
set -euo pipefail
URL="${1:?completions url}"; MODEL="${2:?model name}"; REPEATS="${3:-180}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 -c "print('You are an expert assistant. Follow these detailed guidelines carefully. '*$REPEATS)" > "$TMP/prefix.txt"
echo "shared prefix ~$(( $(wc -c < "$TMP/prefix.txt") / 4 )) tokens"

fire() { # suffix -> wall seconds for a max_tokens=1 request (isolates prefill / TTFT)
  local body; body=$(python3 -c "import json,sys;p=open('$TMP/prefix.txt').read();print(json.dumps({'model':'$MODEL','prompt':p+' Question: '+sys.argv[1],'max_tokens':1,'temperature':0}))" "$1")
  local t; t=$(date +%s.%N)
  curl -s --max-time 120 "$URL" -H "Content-Type: application/json" -d "$body" -o /dev/null
  python3 -c "print(round($(date +%s.%N)-$t,2))"
}

cold=$(fire "What is 2+2?")
warm=$(fire "Name a color.")
echo "cold prefill (first sight of prefix): ${cold}s"
echo "warm prefill (prefix cached, new suffix): ${warm}s"
python3 -c "print(f'speedup: {$cold/$warm:.1f}x faster TTFT on cache hit')"
