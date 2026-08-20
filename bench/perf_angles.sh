#!/bin/bash
# Perf-angles bench for one model: TTFT distribution, long-ctx throughput,
# prefix-cache effectiveness, KV memory footprint. Parallel=1, matches the
# :8010 production config to avoid the DCE-RPC hang seen 2026-08-19 at parallel=6.
set -uo pipefail

MODEL_ALIAS=$1        # e.g. qwen-agentworld-35b
MODEL_PATH=$2         # relative to /models
MAX_CTX=$3            # e.g. 65536 or 131072
SPEC_FLAGS=${4:-}     # e.g. "--spec-type draft-mtp --spec-draft-n-max 3"

RAW=~/jetson-llm-benchoff/results/raw/perf
mkdir -p "$RAW"
OUT="$RAW/${MODEL_ALIAS}.txt"
exec > >(tee -a "$OUT") 2>&1

echo "=== $(date -Is) PERF ANGLES: $MODEL_ALIAS (max_ctx=$MAX_CTX) ==="

# Start server: parallel=1, cache-reuse enabled so prefix-cache probe measures the effect.
docker rm -f benchoff-run 2>/dev/null || true
BEFORE_MEM=$(free -b | awk '/Mem/ {print $3}')
echo "MEM_USED_BEFORE_LOAD_BYTES=$BEFORE_MEM"

docker run -d --rm --name benchoff-run --runtime nvidia --network host \
  -v /generative-AI-models/gguf:/models \
  -v /usr/local/cuda-12.6/lib64:/jetpack-cuda:ro \
  -e NVIDIA_VISIBLE_DEVICES=all -e LD_LIBRARY_PATH=/jetpack-cuda \
  llamacpp-jetson:qwen38 \
  --model "/models/$MODEL_PATH" \
  --alias "$MODEL_ALIAS" --host 0.0.0.0 --port 8100 \
  --n-gpu-layers 99 --ctx-size "$MAX_CTX" --parallel 1 --flash-attn on \
  --batch-size 2048 --ubatch-size 512 --no-warmup --jinja --metrics \
  --cache-reuse 256 $SPEC_FLAGS >/dev/null

echo "loading @ ctx=$MAX_CTX..."
for i in {1..90}; do
  if curl -sSf http://localhost:8100/health >/dev/null 2>&1; then
    echo "LOADED_AFTER_S=$((i*5))"
    break
  fi
  sleep 5
done
curl -sSf http://localhost:8100/health >/dev/null || { echo "LOAD_FAILED"; docker logs benchoff-run 2>&1 | tail -10; exit 1; }

# KV MEMORY FOOTPRINT — read from container startup log + free
KV_LINE=$(docker logs benchoff-run 2>&1 | grep -iE 'kv cache size|KV self size|kv_cache size' | tail -3)
AFTER_MEM=$(free -b | awk '/Mem/ {print $3}')
echo "MEM_USED_AFTER_LOAD_BYTES=$AFTER_MEM"
echo "MEM_DELTA_GB=$(python3 -c "print(round(($AFTER_MEM-$BEFORE_MEM)/1e9,2))")"
echo "KV_SELF_LINES:"
echo "$KV_LINE"

# === (1) TTFT DISTRIBUTION ===
echo
echo "=== (1) TTFT DISTRIBUTION (short prompt, 20 gen tokens, 30 requests) ==="
python3 - <<PY
import json, time, urllib.request
url="http://localhost:8100/v1/chat/completions"
def probe():
    body=json.dumps({"model":"$MODEL_ALIAS","messages":[{"role":"user","content":"Say hi."}],"max_tokens":20,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"stream":False}).encode()
    t0=time.time()
    urllib.request.urlopen(urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"}),timeout=120).read()
    return time.time()-t0
cold=probe(); print(f"COLD_S={cold:.3f}")
warm=[probe() for _ in range(29)]
warm.sort()
p50=warm[len(warm)//2]; p95=warm[int(len(warm)*0.95)]; p99=warm[-1]
print(f"WARM_N=29 P50_S={p50:.3f} P95_S={p95:.3f} P99_S={p99:.3f} MIN_S={warm[0]:.3f} MEAN_S={sum(warm)/len(warm):.3f}")
PY

# === (2) LONG-CONTEXT THROUGHPUT DEGRADATION ===
echo
echo "=== (2) LONG-CTX THROUGHPUT (prompt-fill @ N tokens, 100 gen tokens) ==="
python3 - <<PY
import json, time, urllib.request
url="http://localhost:8100/v1/chat/completions"
max_ctx=$MAX_CTX
points=[2048,8192,32768]
if max_ctx>=65536: points.append(65536)
if max_ctx>=131072: points.append(131072)
# Filler prompt: repeat token-y content to reach the target prompt length.
# ~4 chars/token: 4*N chars of prompt lands near N tokens after templating.
for n in points:
    if n>max_ctx-200: continue
    filler=("The quick brown fox jumps over the lazy dog. "*max(1,(n*4)//45))[:max(200,n*4-100)]
    body=json.dumps({"model":"$MODEL_ALIAS","messages":[{"role":"user","content":filler+"\n\nSummarize in one sentence."}],"max_tokens":100,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"stream":False}).encode()
    t0=time.time()
    try:
        d=json.loads(urllib.request.urlopen(urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"}),timeout=300).read())
        wall=time.time()-t0
        usage=d.get("usage",{})
        pt=usage.get("prompt_tokens",0); ct=usage.get("completion_tokens",0)
        tps=ct/wall if wall else 0
        print(f"CTX_TARGET={n} PROMPT_TOKENS={pt} GEN_TOKENS={ct} WALL_S={wall:.2f} GEN_TPS={tps:.2f}")
    except Exception as e:
        print(f"CTX_TARGET={n} ERROR={e}")
PY

# === (3) PREFIX-CACHE EFFECTIVENESS ===
echo
echo "=== (3) PREFIX-CACHE (same 4K prefix, distinct suffix; cold + 3 warm) ==="
python3 - <<PY
import json, time, urllib.request
url="http://localhost:8100/v1/chat/completions"
prefix=("Context: a large body of text about kubernetes networking, ingress, and services. "*80)[:3800]
def probe(suffix):
    body=json.dumps({"model":"$MODEL_ALIAS","messages":[{"role":"user","content":prefix+"\n\nQ: "+suffix}],"max_tokens":30,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"stream":False}).encode()
    t0=time.time()
    urllib.request.urlopen(urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"}),timeout=120).read()
    return time.time()-t0
cold=probe("first question about services")
warm=[probe(f"question {i} about ingress") for i in range(3)]
avg_warm=sum(warm)/len(warm)
print(f"PREFIX_COLD_S={cold:.3f} PREFIX_WARM_AVG_S={avg_warm:.3f} PREFIX_SPEEDUP={cold/avg_warm:.2f}x")
print(f"WARM_INDIVIDUAL_S={warm}")
PY

echo
echo "=== $(date -Is) PERF ANGLES DONE for $MODEL_ALIAS ==="
