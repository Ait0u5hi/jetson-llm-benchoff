#!/usr/bin/env bash
# VAREX text_flow arms on the AGX (2026-09-12), one llama-server at a time on :8012. Waits for the gemma run first.
set -u; cd ~/varex-bench
until grep -qE "^DONE|Traceback" results/run-gemma.log; do sleep 15; done
docker rm -f varex-llm >/dev/null 2>&1
M=/home/agx_orin/models/gguf-extract
arm() { # name style extra-args...
  local name=$1 style=$2; shift 2
  echo "$(date -Is) == $name ($style)"
  docker rm -f varex-llm >/dev/null 2>&1
  docker run -d --name varex-llm --runtime nvidia -p 127.0.0.1:8012:8012 -v $M:/m:ro llamacpp-jetson:latest --host 0.0.0.0 --port 8012 --alias $name --ctx-size 12288 --parallel 1 --no-warmup -ngl 99 "$@" >/dev/null
  for i in $(seq 1 40); do curl -sf http://127.0.0.1:8012/health >/dev/null && break; sleep 5; done
  python3 run_local.py --model $name --style $style --n 150 --ids-file results/sample_ids.txt --out results 2>&1 | tail -3
  docker logs varex-llm 2>&1 | grep -iE "CUDA error|abort" | tail -2
  docker rm -f varex-llm >/dev/null 2>&1
}
arm nuextract-2b-q8 nuextract --model /m/NuExtract-2.0-2B-Q8_0.gguf --mmproj /m/NuExtract-2.0-2B-mmproj-BF16.gguf
arm nuextract-2b-q8-sysprompt json_object --model /m/NuExtract-2.0-2B-Q8_0.gguf --mmproj /m/NuExtract-2.0-2B-mmproj-BF16.gguf
arm lfm2-1.2b-extract-q8 json_object --model /m/LFM2-1.2B-Extract-Q8_0.gguf
arm qwen35-2b-q4 json_object --model /m/Qwen3.5-2B-Q4_K_M.gguf --reasoning off
echo "$(date -Is) ALL VAREX ARMS DONE"
