#!/bin/bash
# One worker-pool bench arm in a STANDALONE llama-server container (production router untouched).
# Usage: pool_arm.sh <name> <gguf-relative-to-/models> <parallel> <ctx-per-slot> [extra llama-server flags...]
# Mirrors ~/llamacpp-compose/compose.yml mounts (llamacpp-jetson:latest, /models, JetPack cuBLAS).
set -euo pipefail
NAME="$1"; GGUF="$2"; NP="$3"; CTX_SLOT="$4"; shift 4; EXTRA=("$@")
PORT=8011; CTX=$((NP*CTX_SLOT)); OUT=~/jetson-llm-benchoff/results/raw/pool-20260911; mkdir -p "$OUT"
CN="bench-pool-$NAME"
docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d --name "$CN" --runtime nvidia --network host \
  -v /generative-AI-models/gguf:/models -v /usr/local/cuda-12.6/lib64:/jetpack-cuda:ro \
  -e NVIDIA_VISIBLE_DEVICES=all -e LD_LIBRARY_PATH=/jetpack-cuda \
  "${IMG:-llamacpp-jetson:latest}" -m "/models/$GGUF" --alias "$NAME" -ngl 99 \
  --ctx-size "$CTX" --parallel "$NP" --jinja --flash-attn on --batch-size 2048 --ubatch-size 512 \
  --host 0.0.0.0 --port $PORT --no-warmup --metrics "${EXTRA[@]}" >/dev/null
echo "[$NAME] started $CN parallel=$NP ctx=$CTX ($CTX_SLOT/slot) extra=${EXTRA[*]:-none}"
for i in $(seq 1 120); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then echo "[$NAME] healthy after ${i}x5s"; break; fi
  if ! docker ps -q -f name="$CN" | grep -q .; then echo "[$NAME] CONTAINER DIED"; docker logs "$CN" 2>&1 | tail -20; exit 2; fi
  sleep 5
done
curl -s -m 3 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "[$NAME] never healthy"; docker logs "$CN" 2>&1 | tail -20; docker rm -f "$CN"; exit 3; }
{ docker logs "$CN" 2>&1 | grep -E "n_ctx_seq|n_seq_max|KV .*size|new slot|compute buffer|model buffer" | head -12 || true; } | tee "$OUT/$NAME.load.txt"
free -g | head -2 | tee -a "$OUT/$NAME.load.txt"
{ P=$(pgrep -f "alias $NAME" | head -1); [ -n "$P" ] && echo "RSS_kB $(grep VmRSS /proc/$P/status)" || echo "RSS_kB n/a"; } | tee -a "$OUT/$NAME.load.txt"
# --- idle gate: wait until the resident 8B on :8010 is not decoding (contamination guard) ---
idle8b() { curl -s -m 3 http://127.0.0.1:60197/slots 2>/dev/null | python3 -c "import sys,json; print(int(any(s.get('is_processing') for s in json.load(sys.stdin))))" 2>/dev/null || echo 0; }
for i in $(seq 1 90); do
  b=$(idle8b); if [ "$b" = "0" ]; then sleep 15; [ "$(idle8b)" = "0" ] && { echo "[$NAME] 8B idle, starting sweep (waited $((i*10))s)"; break; }; fi
  sleep 10
done
REQ0=$(docker logs --since 1s llamacpp-swap-router-8010 2>&1 | grep -c "proxying request" || true)
T0=$(date -Is)
# warm once, then sweep 1 and N
curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$NAME\",\"prompt\":\"warmup\",\"max_tokens\":8}" >/dev/null
~/jetson-llm-benchoff/bench/engine_sweep.sh "http://127.0.0.1:$PORT/v1/completions" "$NAME" "1 $NP" 200 | tee "$OUT/$NAME.sweep.txt"
REQN=$(docker logs --since "$T0" llamacpp-swap-router-8010 2>&1 | grep -c "proxying request" || true)
echo "router requests to :8010 during sweep: $REQN (0 = clean)" | tee -a "$OUT/$NAME.sweep.txt"
docker rm -f "$CN" >/dev/null; echo "[$NAME] container removed"
