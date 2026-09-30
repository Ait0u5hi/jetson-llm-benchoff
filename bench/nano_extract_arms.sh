#!/usr/bin/env bash
# Drive the 3 nano extractor arms (2026-09-12): stop production gemma, run each candidate on :8081, bench, restore gemma.
set -u
N="${NANO_SSH:?set NANO_SSH=user@host of the nano}"; NANO_HOST="${NANO_HOST:?set NANO_HOST=address of the nano}"; G=/generative-AI-models/gguf
cd ~/jetson-llm-benchoff
OUT=results/raw/nano-extract-20260912
until ssh -o ConnectTimeout=8 $N 'grep -q ALLDONE ~/dl-extract.log' </dev/null 2>/dev/null; do sleep 20; done
echo "$(date -Is) downloads done"; ssh $N 'cat ~/dl-extract.log' </dev/null
run_arm() { # name model-args...
  local name=$1; shift
  case ",${ONLY_ARMS:-all}," in *,all,*|*,$name,*) ;; *) return;; esac
  echo "$(date -Is) == arm $name"
  ssh $N "docker stop llama-server >/dev/null; docker rm -f bench-extract >/dev/null 2>&1; docker run -d --rm --name bench-extract --runtime nvidia --network host -v /generative-AI-models:/generative-AI-models:ro llamacpp-jetson:latest --host 0.0.0.0 --port 8081 --alias bench --ctx-size 8192 --parallel 1 --no-warmup -ngl 99 $*" </dev/null
  for i in $(seq 1 60); do curl -sf http://$NANO_HOST:8081/health >/dev/null && break; sleep 5; done
  curl -s http://$NANO_HOST:8081/health; echo
  ssh $N 'free -m | tail -2' </dev/null
  DOCS_FILE=$OUT/docs.txt OUTDIR=$OUT ARMS=$name MAX_TOKENS=${MAX_TOKENS:-1500} python3 bench/extract_compare.py 2>&1 | tee $OUT/arm-$name${TAG:-}.log
  ssh $N "docker logs bench-extract 2>&1 | tail -5; docker stop bench-extract >/dev/null" </dev/null
}
run_arm nano-nuextract-2b --model $G/NuExtract-2.0-2B-Q8_0.gguf
run_arm nano-lfm2-extract --model $G/LFM2-1.2B-Extract-Q8_0.gguf
run_arm nano-qwen35-2b --model $G/unsloth_Qwen3.5-2B_Q4_K_M/*.gguf --reasoning off
ssh $N 'docker start llama-server; sleep 20; docker ps --format "{{.Names}} {{.Status}}"; free -m | tail -2' </dev/null
echo "$(date -Is) ALL ARMS DONE"
