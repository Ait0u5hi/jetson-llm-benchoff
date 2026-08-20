#!/bin/bash
# Sequential IFEval overnight bench, one model at a time, parallel=1.
# Designed to avoid the parallel=6 DCE-RPC fault mode from 2026-08-19.
# Runs on AGX; drives lm-eval on Arthur pointed at AGX :8100.
set -uo pipefail

RESULTS=~/jetson-llm-benchoff/results
LOG=$RESULTS/overnight-$(date +%Y%m%d-%H%M).log
mkdir -p "$RESULTS/raw"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) SEQUENTIAL IFEVAL BENCH START ==="
echo "Server config: parallel=1, ctx=8192, flash-attn on (matches :8010 primary router config)"
echo "Client: lm-eval num_concurrent=1 (via arthur)"

MODELS=(
  "qwen-agentworld-35b:unsloth_Qwen-AgentWorld-35B-A3B_UD-IQ4_NL/Qwen-AgentWorld-35B-A3B-UD-IQ4_NL.gguf::"
  "qwen3.8-27b-mtp:Jackrong_Qwen3.8-27B-MTP-Q4_K_M/Qwen3.8-27B-MTP-Q4_K_M.gguf:--spec-type draft-mtp --spec-draft-n-max 3:"
  "qwen3.6-35b:unsloth_Qwen3.6-35B-A3B-MTP_UD-IQ4_NL/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf:--spec-type draft-mtp --spec-draft-n-max 3:"
)

for entry in "${MODELS[@]}"; do
  IFS=":" read -r MODEL GGUF SPEC _ <<< "$entry"
  echo
  echo "=== $(date -Is) MODEL: $MODEL ==="
  free -h | head -2

  # kill any leftover benchoff container
  docker rm -f benchoff-run 2>/dev/null || true

  # start benchoff container at parallel=1 (matches production :8010 config)
  docker run -d --rm --name benchoff-run --runtime nvidia --network host \
    -v /generative-AI-models/gguf:/models \
    -v /usr/local/cuda-12.6/lib64:/jetpack-cuda:ro \
    -e NVIDIA_VISIBLE_DEVICES=all -e LD_LIBRARY_PATH=/jetpack-cuda \
    llamacpp-jetson:qwen38 \
    --model "/models/$GGUF" \
    --alias "$MODEL" --host 0.0.0.0 --port 8100 \
    --n-gpu-layers 99 --ctx-size 8192 --parallel 1 --flash-attn on \
    --batch-size 2048 --ubatch-size 512 --no-warmup --jinja --metrics $SPEC

  echo "waiting for model load..."
  for i in {1..60}; do
    if curl -sSf http://localhost:8100/health >/dev/null 2>&1; then
      echo "LOADED after ~$((i*5))s"
      break
    fi
    sleep 5
  done
  if ! curl -sSf http://localhost:8100/health >/dev/null 2>&1; then
    echo "FAILED to load $MODEL after 300s, moving on"
    docker logs benchoff-run 2>&1 | tail -20
    docker rm -f benchoff-run 2>/dev/null || true
    continue
  fi

  # run IFEval on arthur
  echo "=== $(date -Is) firing IFEval on $MODEL via arthur ==="
  START=$(date +%s)
  ssh arthur "~/run_bench.sh ifeval $MODEL http://192.168.1.152:8100/v1/chat/completions" 2>&1 | tail -25
  END=$(date +%s)
  echo "=== $(date -Is) IFEval on $MODEL done in $((END-START))s ==="

  # pull results back to AGX
  rsync -a arthur:~/quality-results/$MODEL/ "$RESULTS/quality-$MODEL/" 2>&1 | tail -3

  # stop container so next model gets a clean slot
  docker rm -f benchoff-run 2>/dev/null || true
  sleep 10  # let GPU state settle
done

echo
echo "=== $(date -Is) SEQUENTIAL BENCH DONE ==="
free -h | head -2
