#!/usr/bin/env bash
# Run scripts/rb_trellis2.py inside the patched TRELLIS.2-ROCm checkout (its venv via run_amd.sh), with a RAM cap:
# the low-VRAM pipeline keeps the models in host RAM (512: ~11-13 GB RSS).
#   ./rb.sh gen --image path/to/image.png                   # default: INT4 DiTs + INT8 decoders from runs/trellis2/t1
#   ./rb.sh gen --pipeline 1024_cascade --image ...
#   ./rb.sh gen --stock --tag base --image ...              # unmodified pipeline (reference)
# env: TRELLIS2_DIR (default ~/3d_gen/TRELLIS.2-ROCm), RB_MEM (default 13G)
set -euo pipefail
HERE="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRELLIS2_DIR="${TRELLIS2_DIR:-$HOME/3d_gen/TRELLIS.2-ROCm}"
[ -f "$TRELLIS2_DIR/run_amd.sh" ] || { echo "rb.sh: $TRELLIS2_DIR/run_amd.sh not found (patched TRELLIS.2-ROCm, see README)" >&2; exit 1; }
[ -d "$HERE/runs/trellis2/t1" ] || echo "rb.sh: warning: no models in $HERE/runs/trellis2/t1 (unzip t1.zip into runs/trellis2/)" >&2
builtin cd "$TRELLIS2_DIR"
exec systemd-run --user --scope -q -p MemoryMax="${RB_MEM:-13G}" -p MemorySwapMax=0 \
    env -u TORCH_LOGS MALLOC_MMAP_THRESHOLD_=65536 ./run_amd.sh "$HERE/scripts/rb_trellis2.py" "$@"
