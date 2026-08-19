# Engine viability for Qwen3.8-27B (arch `qwen3_5`, GDN hybrid VLM) on Jetson SM87 / JetPack 6

The general lesson: before assuming an engine runs on an edge board, check four axes independently:
**(1) is the model architecture supported, (2) does the engine's CUDA major match the board
(JetPack 6 = CUDA 12.6), (3) does the Python ABI (cpXY) match an available torch build, (4) does the
prebuilt wheel include the board's SM (8.7)?** Each failure below is one of these axes, not a
fundamental limitation.

## llama.cpp: WORKS
- The current serving image was build 9552; the GDN arch needs >= b10419. Rebuild from `master`
  with `-DCMAKE_CUDA_ARCHITECTURES=87`. Runtime needs the JetPack cuBLAS mounted
  (`/usr/local/cuda-12.6/lib64`) because generic aarch64 cuBLAS fails on Tegra unified memory.
- GGUF Q4_K_M + mmproj (vision) both load. MTP via `--spec-type draft-mtp`.

## vLLM 0.20: WORKS (supersedes the "FLA broken on Ampere" lore)
- `vllm==0.20.0` from the jetson-ai-lab cu126 index runs the Gated-DeltaNet path on SM87:
  log shows "Using Triton/FLA GDN prefill kernel" plus FLASH_ATTN for the full-attention layers.
  The older vLLM issue #40124 (FLA/Mamba broken on Ampere) is resolved in current vLLM.
- It loads the asymmetric-int4 compressed-tensors checkpoint that SGLang cannot (see below).
- Install is fiddly; see `recipes/vllm-jetpack6.md`.

## SGLang: DOES NOT RUN (this model, this board)
- Latest SGLang (0.5.13+) migrated to CUDA 13 (`flashinfer[cu13]`), incompatible with JetPack 6.
- The last CUDA-12 version, **0.5.12**, does support `qwen3_5` + MTP and installs on aarch64 from
  prebuilt wheels (`sglang-kernel==0.4.2.post2` aarch64 exists, so no source build is needed).
  It loads, detects the qwen3 reasoning parser, and begins model init.
- It then fails at the quant layer: `NotImplementedError: No compressed-tensors compatible scheme
  was found`. The checkpoint is asymmetric int4, group_size 32, W4A16 (`symmetric: false`), which
  SGLang 0.5.12's compressed-tensors path does not implement (vLLM 0.20 does).
- Workaround would be re-quantizing to a symmetric or GPTQ scheme SGLang supports. Not pursued:
  vLLM already works and `--enable-prefix-caching` provides SGLang RadixAttention's headline benefit.

## TokenSpeed: DOES NOT RUN (this board)
- `tokenspeed 0.1.0` supports `qwen3_5` (ships `Qwen3_5BaseTextConfig`), and its aarch64 native
  kernels exist. But those kernels are built for cp311/cp312/cp313 only, while the Jetson prebuilt
  torch is cp310-only. No Python version has both, so it cannot be assembled from prebuilt wheels.

## TensorRT-LLM: NOT TESTED
- The GDN `qwen3_5` architecture is almost certainly not in the TRT-LLM support matrix yet, and
  TRT-LLM requires ahead-of-time engine compilation on the target GPU. Highest effort, lowest
  near-term odds; skipped once vLLM proved the batching-engine case.
