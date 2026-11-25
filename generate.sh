#!/usr/bin/env bash
# Image -> 3D (AMD or NVIDIA) with INT4 DiTs + INT8 decoders (generate.py) in the repo venv (run.sh), with a host-RAM cap:
# the low-VRAM pipeline keeps the models in host RAM (512: ~11-13 GB RSS).
#   ./generate.sh --image photo.png
#   ./generate.sh --image photo.png --pipeline 1024_cascade
# env: RB_MEM (RAM cap, default 13G; needs systemd, skipped otherwise), RB_MODEL_REPO (Hugging Face repo of the
#      INT model files), TRELLIS_PYTHON (default .venv/bin/python)
set -euo pipefail
HERE="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAP=()
if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope -q true 2>/dev/null; then
    CAP=(systemd-run --user --scope -q -p MemoryMax="${RB_MEM:-13G}" -p MemorySwapMax=0)
fi
exec "${CAP[@]}" env -u TORCH_LOGS MALLOC_MMAP_THRESHOLD_=65536 "$HERE/run.sh" "$HERE/generate.py" "$@"
