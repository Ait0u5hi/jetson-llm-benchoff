# Eval harness

Two engine-agnostic probes against any OpenAI-compatible `/v1/completions` endpoint
(vLLM, llama.cpp `llama-server`, SGLang, and so on). No dependencies beyond `bash`,
`curl`, and `python3`.

## `engine_sweep.sh`: concurrency throughput

Fires N concurrent completion requests and reports aggregate + per-stream decode tok/s
at each concurrency level. If the server returns native llama.cpp `timings`, it also
reports the MTP draft accept rate.

```
engine_sweep.sh <completions_url> <model_name> ["1 3 6"] [n_predict]
```

Example:
```
engine_sweep.sh http://localhost:8100/v1/completions qwen38-vllm "1 3 6" 200
```

Sample output:
```
== engine_sweep: qwen38-vllm @ http://localhost:8100/v1/completions (n_predict=200) ==
N= 1 | wall= 22.4s | gen_tokens=  200 | AGG=   8.9 tok/s | per-stream~  8.9
N= 3 | wall= 22.1s | gen_tokens=  600 | AGG=  27.2 tok/s | per-stream~  9.1
N= 6 | wall= 23.3s | gen_tokens= 1200 | AGG=  51.4 tok/s | per-stream~  8.6
```

Aggregate = total generated tokens / wall clock of the whole concurrent batch. It reads
`usage.completion_tokens` (OpenAI shape) and falls back to `timings.predicted_n`
(llama.cpp native), so the same script scores every engine comparably.

## `prefix_cache_probe.sh`: shared-prefix TTFT

Sends a long shared prefix twice with different suffixes at `max_tokens=1` to isolate
prefill. With prefix caching (vLLM `--enable-prefix-caching`) or RadixAttention (SGLang)
enabled, the warm request skips prefill and returns far faster.

```
prefix_cache_probe.sh <completions_url> <model_name> [prefix_repeats]
```

Example and sample output:
```
$ prefix_cache_probe.sh http://localhost:8100/v1/completions qwen38-vllm 180
shared prefix ~3285 tokens
cold prefill (first sight of prefix): 7.0s
warm prefill (prefix cached, new suffix): 1.4s
speedup: 5.0x faster TTFT on cache hit
```

The prefix is synthetic (a repeated instruction sentence); prefill cost depends on token
count, not content, so this is a valid TTFT probe. Do not read its decode throughput.

## Notes on fair comparison

- Use the same `n_predict`, context length, and sampling across engines.
- Run one engine at a time; a 27B model reserves most of the board's memory.
- Numbers are single-board, single-run. Re-run and average for anything load-bearing.
