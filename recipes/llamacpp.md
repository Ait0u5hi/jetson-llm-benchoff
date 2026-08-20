# llama.cpp for Qwen3.8-27B on JetPack 6 (CUDA 12.6 / SM87)

The stock/older llama.cpp images predate the Gated-DeltaNet arch. Rebuild from `master`.

## Build the image (arch 8.7)
```dockerfile
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates libssl-dev
ARG LLAMA_REF=master
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggerganov/llama.cpp /opt/llama.cpp
RUN cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=ON \
 && cmake --build build --config Release -j"$(nproc)" --target llama-server
# ... copy build/bin/llama-server into a cuda:12.6.2-runtime stage
```
Runtime note: the generic aarch64 cuBLAS in the CUDA base image fails on Tegra unified memory. Mount
the JetPack cuBLAS and point the loader at it:
```
-v /usr/local/cuda-12.6/lib64:/jetpack-cuda:ro  -e LD_LIBRARY_PATH=/jetpack-cuda
```

## Serve (with MTP self-speculation — +64% throughput at 3 concurrent)
```bash
llama-server \
  -m /models/Qwen3.8-27B-MTP-Q4_K_M.gguf \
  --mmproj /models/mmproj-F32.gguf \
  -ngl 99 --ctx-size 8192 --parallel 6 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 0.0.0.0 --port 8098
```
- Use the MTP GGUF build for `draft-mtp` (self-speculation, no separate draft model).
- Vision: pass `--mmproj`. Long context is cheap on this hybrid (128K adds ~5 GiB KV).
- Tegra cuBLAS does not support KV cache quantization, so long-context times concurrency is
  full-precision KV. The hybrid architecture keeps that affordable anyway.
- Sampling (thinking): temp 1.0, top_p 0.95, top_k 20, min_p 0.
