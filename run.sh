#!/usr/bin/env bash
# Run a Python script with the repo venv (built by setup.sh) and the GPU environment settings below.
#   ./run.sh generate.py --image photo.png      (generate.sh adds a host-RAM cap)
# Platform: TRELLIS_PLATFORM=cuda|rocm, else the one setup.sh built for (.venv/trellis_platform), else detected.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${TRELLIS_PYTHON:-$HERE/.venv/bin/python}"

PLATFORM="${TRELLIS_PLATFORM:-}"
[ -z "$PLATFORM" ] && [ -f "$HERE/.venv/trellis_platform" ] && PLATFORM="$(cat "$HERE/.venv/trellis_platform")"
if [ -z "$PLATFORM" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then PLATFORM=cuda; else PLATFORM=rocm; fi
fi

# ~/.local can carry another torch that would shadow the venv's.
export PYTHONNOUSERSITE=1
export OPENCV_IO_ENABLE_OPENEXR=1
unset TORCH_LOGS

if [ "$PLATFORM" = "rocm" ]; then
    export TRITON_DEFAULT_BACKEND="${TRITON_DEFAULT_BACKEND:-amd}"
    # flash-attn on ROCm is the Triton build (setup.sh): without this it looks for the CUDA extension flash_attn_2_cuda
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    # PyTorch SDPA by default: flash-attn 2.8.3's Triton AMD kernels do not import with Triton >= 3.9 (torch >= 2.14)
    export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    # expandable_segments corrupts tensors on this ROCm stack.
    unset PYTORCH_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
    # Shell debug defaults that only add noise / fill /tmp with comgr temp files.
    unset AMD_COMGR_SAVE_TEMPS AMD_COMGR_EMIT_VERBOSE_LOGS
    export AMD_LOG_LEVEL=0
else
    # GPUs before Ampere (compute capability < 8.0): no flash-attn kernels (setup.sh skips it) -> PyTorch SDPA.
    # The bf16 DiTs are loaded in fp16 there (trellis2/models/__init__.py).
    MIN_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -1 || true)"
    if [ -n "$MIN_CC" ] && [ "${MIN_CC%%.*}" -lt 8 ]; then
        export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
    fi
fi

cd "$HERE"
exec "$PY" "$@"
