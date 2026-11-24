# TRELLIS.2-ROCm-INT4

Run [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (image to 3D) on an AMD RX 7900 XTX with calibrated
**INT4 flow DiTs** and **INT8 latent decoders**, made with RoundBit (block-wise AutoRound-style calibration).
This repository holds the code; the model files are not included (see [Models](#models)).

## Results (seed 42, T.png / bear / gear example images)

| | weights the pipeline loads | peak VRAM | quality |
|---|---|---|---|
| 512, stock (bf16 DiTs, fp16 decoders, fp32 image encoder) | 11.0 GB | 2.72 GiB (T) | reference |
| **512, INT4 DiTs + INT8 decoders (default)** | **4.4 GB** | **2.02 GiB** (T), 1.96 (bear), 1.25 (gear) | geometry checked against the source images at full resolution: gear 7 spokes like the source, rim intact; bear on par |
| 1024_cascade, stock | – | 7.19 GiB (T), 6.85 (bear), 5.14 (gear) | reference |
| 1024_cascade, INT4 + INT8 decoders | – | 6.26 GiB (T), 6.34 (bear), 2.39 (gear) | geometry on par; T loses the teal glass cylinder (dark metal tube instead) |

The INT4 DiTs change the sample a little (a 3D generator turns small weight changes into a slightly different
object); the INT8 decoders are practically lossless (identical renders to the INT4-DiT run without them).
Known loss: on T the glass cylinder comes out as dark metal (512 and 1024). Judge results on full-resolution views
against the source image; chamfer distance (~0.004) does not show missing or invented parts.

## Setup

1. TRELLIS.2-ROCm at the upstream commit in `patches/UPSTREAM_COMMIT`, plus the patch from this repo:
   ```bash
   git clone https://github.com/Mateusz-Dera/TRELLIS.2-ROCm.git ~/3d_gen/TRELLIS.2-ROCm
   cd ~/3d_gen/TRELLIS.2-ROCm
   git checkout aa1528a607b22c0a320848fef61b0e2fc9c9b39c
   git apply /path/to/TRELLIS.2-ROCm-INT4/patches/trellis2-rocm-roundbit.patch
   ```
   The patch adds what the INT models need and what this setup was tested with: building models on the meta device
   with load hooks (`trellis2/models/__init__.py`, `META_HOOKS`: quantized layers are swapped in before the bf16 /
   fp16 weights are allocated), lazy model loading + low-VRAM mode (`trellis2/pipelines/base.py`,
   `trellis2_image_to_3d.py`), the transformers 5 fix for DINOv3, a rembg fallback, symmetry support, `run_amd.sh`
   and the nvdiffrast-hip RDNA3 fixes used by `setup.sh`.
2. Build the environment as the TRELLIS.2-ROCm README describes (`setup.sh`, `.venv`); `run_amd.sh` uses
   `.venv/bin/python` (override with `TRELLIS_PYTHON`). Triton is needed for the INT4 kernels.
3. Point this repo at the checkout if it is not `~/3d_gen/TRELLIS.2-ROCm`: `export TRELLIS2_DIR=/path/to/it`.

## Models

`t1.zip` (~4 GB) unpacked into `runs/trellis2/`, so that `runs/trellis2/t1/` contains:

| file | used by | size |
|---|---|---|
| `sparse_structure_flow_model.pt` | 512, 1024_cascade | 678 MB |
| `shape_slat_flow_model_512.pt` | 512, 1024_cascade | 678 MB |
| `tex_slat_flow_model_512.pt` | 512 | 678 MB |
| `shape_slat_flow_model_1024.pt` | 1024_cascade | 678 MB |
| `tex_slat_flow_model_1024.pt` | 1024_cascade | 678 MB |
| `shape_slat_decoder.int8.pt`, `tex_slat_decoder.int8.pt` | both | 482 MB each |

They are RoundBit's packed format (INT codes + fp16 group scales), not `.safetensors`: each file replaces only the
quantized linear layers; everything else (norms, embeddings, the sparse-structure decoder, the DINOv3 image encoder)
still comes from the original `microsoft/TRELLIS.2-4B` checkpoints in the Hugging Face cache, so those are needed too.
The decoder files can be rebuilt from the original decoders with `./rb.sh save_dec --tag t1`.

## Usage

```bash
./rb.sh gen --image path/to/image.png                          # 512, INT4 DiTs + INT8 decoders (default)
./rb.sh gen --pipeline 1024_cascade --image path/to/image.png
./rb.sh gen --stock --tag base --image path/to/image.png       # unmodified pipeline, as a reference
./rb.sh gen --dec_bits 0 --image ...                           # INT4 DiTs, fp16 decoders
```

Outputs go to `runs/trellis2/<tag>_<image>_s<seed>.{ply,png}` (mesh + a 2x2 render grid: shaded / normals, PBR
channels); a run with a `base_...ply` for the same image and seed also reports the chamfer distance to it. `rb.sh`
caps host RAM at 13 GB (`RB_MEM`): the low-VRAM pipeline keeps all models in host RAM. Options: `--seed`, `--sampler`
(JSON, e.g. step counts), `--load ''` (no INT DiTs), `--dec_bits 0|4|8`, `--cond bf16|int8` (image encoder;
**experimental**, changes the sample; fp16 overflows to NaN). The other options of `scripts/rb_trellis2.py`
(`capture`, `tune`, `--bridges`, `--teacache`, `--dvec`, ...) belong to the calibration / research tools of the full
RoundBit repository and need files that are not part of this one.

## Layout

```
rb.sh                        wrapper: runs scripts/rb_trellis2.py in the TRELLIS.2 checkout with a RAM cap
scripts/rb_trellis2.py       gen (load INT models, generate, render, chamfer), save_dec, capture / tune
scripts/dec_quant.py         weight-only INT8 / INT4 for the latent decoders
roundbit/                    runtime subset of RoundBit: IntLinear / Int4Linear packed format, Triton INT4 kernels
patches/                     TRELLIS.2-ROCm patch + the upstream commit it applies to
```

## Licence

TRELLIS.2 and TRELLIS.2-ROCm are MIT (the patch is a change to that code). The RoundBit layer code in `roundbit/`
builds on LittleBit (CC BY-NC 4.0): private, non-commercial use.
