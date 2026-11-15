#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for one VLM (8 GB VRAM friendly defaults).
# Usage: scripts/serve_vllm.sh [MODEL] [extra vllm args...]
#   scripts/serve_vllm.sh Qwen/Qwen2.5-VL-3B-Instruct-AWQ
#   scripts/serve_vllm.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ
# The 7B-AWQ weights take 6.7 GiB, more than the ~6.8 GiB free on the 8 GB laptop GPU under WSL,
# so for 7B we offload 3 GB of weights to CPU, disable CUDA graphs and shorten the context.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"; shift || true
# FlashInfer sampler JIT-compiles kernels and needs nvcc (absent in WSL); use the PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
EXTRA=()
if [[ "$MODEL" == *7B* ]]; then
  GPU_UTIL="${GPU_UTIL:-0.85}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"; MAX_SEQS="${MAX_SEQS:-8}"
  MAX_PIXELS=307200
  EXTRA=(--enforce-eager --max-num-batched-tokens 2048 --cpu-offload-gb "${CPU_OFFLOAD_GB:-3}")
fi
exec .venv/bin/vllm serve "$MODEL" \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.82}" \
  --max-num-seqs "${MAX_SEQS:-16}" \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}" \
  --enable-prefix-caching \
  "${EXTRA[@]}" "$@"
