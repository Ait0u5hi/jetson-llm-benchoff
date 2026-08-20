#!/bin/bash
# Sequential MMLU (limit=500) sweep across 3 models, parallel=1.
set -uo pipefail
RAW=~/jetson-llm-benchoff/results/raw/perf
mkdir -p "$RAW"
LOG=$RAW/mmlu-sweep.log
exec > >(tee -a "$LOG") 2>&1

MODELS=(
  "qwen-agentworld-35b:unsloth_Qwen-AgentWorld-35B-A3B_UD-IQ4_NL/Qwen-AgentWorld-35B-A3B-UD-IQ4_NL.gguf::"
  "qwen3.8-27b-mtp:Jackrong_Qwen3.8-27B-MTP-Q4_K_M/Qwen3.8-27B-MTP-Q4_K_M.gguf:--spec-type draft-mtp --spec-draft-n-max 3:"
  "qwen3.6-35b:unsloth_Qwen3.6-35B-A3B-MTP_UD-IQ4_NL/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf:--spec-type draft-mtp --spec-draft-n-max 3:"
)

for entry in "${MODELS[@]}"; do
  IFS=":" read -r M P S _ <<< "$entry"
  echo; echo "=== $(date -Is) MMLU $M ==="
  docker rm -f benchoff-run 2>/dev/null || true

  docker run -d --rm --name benchoff-run --runtime nvidia --network host \
    -v /generative-AI-models/gguf:/models \
    -v /usr/local/cuda-12.6/lib64:/jetpack-cuda:ro \
    -e NVIDIA_VISIBLE_DEVICES=all -e LD_LIBRARY_PATH=/jetpack-cuda \
    llamacpp-jetson:qwen38 \
    --model "/models/$P" \
    --alias "$M" --host 0.0.0.0 --port 8100 \
    --n-gpu-layers 99 --ctx-size 8192 --parallel 1 --flash-attn on \
    --batch-size 2048 --ubatch-size 512 --no-warmup --jinja --metrics $S >/dev/null

  for i in {1..60}; do curl -sSf http://localhost:8100/health >/dev/null 2>&1 && break || sleep 5; done
  curl -sSf http://localhost:8100/health >/dev/null || { echo "LOAD_FAILED $M"; continue; }

  START=$(date +%s)
  ssh arthur "mkdir -p ~/quality-results/$M && NLTK_DATA=~/nltk_data ~/evals-venv/bin/lm_eval \
    --model local-chat-completions --tasks mmlu --limit 500 --apply_chat_template \
    --model_args 'model=$M,base_url=http://192.168.1.152:8100/v1/chat/completions,num_concurrent=1,max_length=8192,tokenizer_backend=None,eos_string=<|im_end|>' \
    --gen_kwargs 'max_gen_toks=32' --output_path ~/quality-results/$M/mmlu" 2>&1 | tail -20
  echo "MMLU_$M DONE in $(($(date +%s)-START))s"

  rsync -a arthur:~/quality-results/$M/mmlu/ "$RAW/mmlu-$M/" 2>&1 | tail -2
  docker rm -f benchoff-run 2>/dev/null || true
  sleep 10
done
echo; echo "=== $(date -Is) MMLU SWEEP DONE ==="
