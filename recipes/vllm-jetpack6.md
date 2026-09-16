# vLLM 0.20 for Qwen3.8-27B on JetPack 6 (CUDA 12.6 / SM87): the working recipe

Standing vLLM up for this model on JetPack 6 took roughly fifteen debugging steps. This is the
sequence that works. The core traps are: pip pulls a CUDA-13 torch from PyPI, the jetson torch needs
external CUDA libs it does not declare, and vLLM has undeclared runtime deps.

```bash
# 1. venv (JetPack 6 python3.10 has no ensurepip; bootstrap pip)
python3 -m venv ~/vllm-venv
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/vllm-venv/bin/python3 /tmp/get-pip.py

# 2. vLLM from the jetson-ai-lab cu126 index (NOT pypi.jetson-ai-lab.dev, that host is stale)
~/vllm-venv/bin/pip install \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  --extra-index-url https://pypi.org/simple  vllm==0.20.0

# 3. force the ABI-matched jetson torch (step 2 pulls torch 2.13+cu130 from PyPI otherwise,
#    which reports cuda_avail=False on the Orin: "NVIDIA driver too old, found 12060")
~/vllm-venv/bin/pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  --force-reinstall --no-deps  torch==2.11.0

# 4. torch 2.11 needs libcudss.so.0, absent from the jetson index and the system.
#    --no-deps avoids a 575MB cublas-12.9 it does not actually need (system cublas 12.6 is used).
~/vllm-venv/bin/pip install --no-deps nvidia-cudss-cu12==0.8.0.10

# 5. undeclared runtime deps, then repair transformers/hub which xgrammar downgrades
~/vllm-venv/bin/pip install xgrammar compressed-tensors loguru
~/vllm-venv/bin/pip install --no-deps "transformers==5.15.1" "huggingface_hub>=1.5,<2"
```

Serve:
```bash
VENV=~/vllm-venv
export LD_LIBRARY_PATH="$(ls -d $VENV/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')/usr/local/cuda-12.6/lib64"
export FLASHINFER_DISABLE_VERSION_CHECK=1     # cubin/python version mismatch is benign here
$VENV/bin/vllm serve ~/models/Qwen3.8-27B-AWQ-INT4 \
  --served-model-name qwen38-vllm \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.60 \             # checks FREE memory, not total; size to headroom
  --enable-prefix-caching \                   # 5x TTFT on shared prompts
  --trust-remote-code --host 0.0.0.0 --port 8100
```

Notes:
- Model load is about 340 s for the 20 GB INT4 checkpoint.
- If you script the launch, guard it with `if __name__ == "__main__":` (vLLM uses spawn once CUDA
  is initialized, and re-imports the entry module).
- Memory: at util 0.60 vLLM reserves about 37 GB. On a 64 GB Orin it cannot coexist with another
  large resident model; treat it as an on-demand endpoint.

## Tool calling (agent workloads) — added 2026-09-15

A vLLM server started without tool-call flags **cannot run an agent**, and the failure is a hard 400
rather than a degraded reply:

> `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`

That is correct, fail-closed behaviour. The model emits tool calls as TEXT in its own markup; the
OpenAI contract needs them back as a structured `tool_calls` array, and only a tool-call parser does
that translation. Without one, the alternative would be returning raw `<tool_call>` markup as
assistant content — the client renders it as the model *talking about* calling a tool while the agent
loop never dispatches. `vllm/entrypoints/openai/cli_args.py` enforces the pairing: auto tool choice
without a parser is a startup error.

**Pick the parser by the model's chat template, not by model family.** For
`Qwen3.8-27B-AWQ-INT4` the template specifies XML-style calls —
`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` — so the parser
is `qwen3_xml`, NOT `hermes` (hermes expects `<tool_call>{json}</tool_call>` and would silently fail
to match). Check `tokenizer_config.json`'s `chat_template` before choosing; vLLM 0.20 registers the
names in `vllm/tool_parsers/__init__.py`.

```
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
```

Keep the reasoning parser alongside it: on a reasoning model it strips the thinking block first, so
the tool parser sees the actual output.

Verified on the AGX worker (`~/.config/systemd/user/vllm-worker.service` ->
`~/llamacpp-compose/vllm/start-vllm.sh`, backup `.bak-20260915`) on 2026-09-15: a `tool_choice: auto`
request returns `finish_reason: tool_calls` with parsed arguments, reasoning still populated, the
3x32k capacity flags unchanged, and neighbouring services (llama-swap 8B, nomic embeddings) alive.
A full agent seat then ran end to end against it, locally.
