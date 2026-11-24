"""Weight-only INT8 / INT4 for the TRELLIS.2 latent decoders (shape + texture, 474M fp16 weights each:
SparseConv3d 402M + Linear 72M), stored packed; each layer dequantizes its fp16 weight only while it runs.

    cd ~/3d_gen/TRELLIS.2-ROCm
    CAP="systemd-run --user --scope -q -p MemoryMax=13G -p MemorySwapMax=0 env MALLOC_MMAP_THRESHOLD_=65536"
    $CAP ./run_amd.sh $RB/trellis2/dec_quant.py --variants int8,int4g64

One generation (t1 INT4 DiTs) gives the latents; then the fp16 decoders and every variant decode the same latents:
  shape: mesh faces, chamfer vs the fp16 mesh (same sampling seed: identical meshes give 0)
  texture: decoded on the fp16 shape's subdivisions, so the voxels match; PSNR / max error of the PBR attributes
  memory: decoder weight bytes and decode peak (allocated / reserved)
One JSON line per variant -> runs/trellis2/dec_quant.jsonl as it finishes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import rb_trellis2 as T  # noqa: E402
from roundbit.int_layer import _group_scale  # noqa: E402

GiB = 2 ** 30
DECODERS = ["shape_slat_decoder", "tex_slat_decoder"]


# ── packed weight-only quantization ────────────────────────────────────────────────────────────────────────────
class _QWeight:
    """Mixin placed in front of the layer's class: `weight` becomes the dequantized fp16 tensor (built per access)."""

    @property
    def weight(self):
        q = self.wq
        if self.wbits == 4:  # two codes per byte
            q = torch.stack([(q & 0xF).to(torch.int8) - 8, (q >> 4).to(torch.int8) - 8], -1).flatten(-2)
        W = q.to(self.ws.dtype).reshape(*self.ws.shape[:2], -1)
        W.mul_(self.ws)  # in place: one fp16 temporary per layer call
        W = W.reshape(self.ws.shape[0], -1)
        if W.shape[1] != self.wfan:  # padded groups; flex_gemm needs a contiguous weight
            W = W[:, : self.wfan].contiguous()
        return W.reshape(self.wshape).to(self.wdtype)


_classes = {}


def quantize_layer_(m: nn.Module, bits: int, group: int) -> int:
    W = m.weight.detach()
    shape, dtype = W.shape, W.dtype
    Wf = W.float().reshape(shape[0], -1)
    fan = Wf.shape[1]
    g = fan if group <= 0 or group >= fan else group
    pad = (-fan) % g
    Wg = torch.nn.functional.pad(Wf, (0, pad)).reshape(shape[0], -1, g)
    L = 2 ** (bits - 1)
    s = _group_scale(Wg, L)
    q = (Wg / s).round().clamp(-L, L - 1).to(torch.int8).reshape(shape[0], -1)
    if bits == 4:
        assert q.shape[1] % 2 == 0
        u = (q + 8).to(torch.uint8).reshape(shape[0], -1, 2)
        q = u[..., 0] | (u[..., 1] << 4)
    del m._parameters["weight"]
    m.register_buffer("wq", q.contiguous())
    m.register_buffer("ws", s.to(torch.float16))
    m.wshape, m.wdtype, m.wfan, m.wbits = shape, dtype, fan, bits
    cls = type(m)
    if cls not in _classes:
        _classes[cls] = type("Q" + cls.__name__, (_QWeight, cls), {})
    m.__class__ = _classes[cls]
    return W.numel()


def quantize_decoder_(model: nn.Module, bits: int, group: int, kinds=("SparseConv3d", "Linear"), skip=()) -> dict:
    n = layers = 0
    for name, m in list(model.named_modules()):
        if type(m).__name__ in kinds and isinstance(getattr(m, "weight", None), nn.Parameter) \
                and not any(k in name for k in skip):
            n += quantize_layer_(m, bits, group)
            layers += 1
    return {"layers": layers, "weights": n}


def model_bytes(m) -> int:
    return sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))


# ── comparison ─────────────────────────────────────────────────────────────────────────────────────────────────
def psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="int8,int4g64", help="comma list: int8 (per row), int8g<g>, int4g<g>")
    ap.add_argument("--load", default="t1")
    ap.add_argument("--image", default=str(T.EXAMPLES / "T.png"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from PIL import Image

    pipe = T.load_pipeline()
    T.meta_swap(pipe, T.PIPELINES["512"], lambda k: T.OUT / a.load / f"{k}.pt")
    img = Image.open(a.image)
    _, (shape_slat, tex_slat, res) = pipe.run(img, pipeline_type="512", seed=a.seed, return_latent=True)
    lm = pipe.models

    @torch.no_grad()  # as the pipeline's decode_latent: with autograd the decoders kept every activation (22.9 GiB)
    def decode():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        meshes, subs = pipe.decode_shape_slat(shape_slat, res)
        tex = pipe.decode_tex_slat(tex_slat, subs_fp if subs_fp else subs)
        torch.cuda.synchronize()
        mem = {"decode_s": round(time.time() - t0, 2),
               "peak_gib": round(torch.cuda.max_memory_allocated() / GiB, 3),
               "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GiB, 3)}
        return meshes[0], subs, tex[0], mem

    subs_fp = None
    decode()  # warm-up: decoder loads + flex_gemm autotune
    m_fp, subs_fp, tex_fp, mem_fp = decode()
    v_fp, f_fp = m_fp.vertices.float().cpu().numpy(), m_fp.faces.cpu().numpy()
    base = {"image": Path(a.image).name, "seed": a.seed, "faces_fp": int(len(f_fp))}
    rec = {**base, "variant": "fp16", **mem_fp,
           "weights_gib": {k: round(model_bytes(dict.__getitem__(lm, k)) / GiB, 3) for k in DECODERS}}
    print("RESULT " + json.dumps(rec), flush=True)
    with open(T.OUT / "dec_quant.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

    for variant in [v for v in a.variants.split(",") if v]:
        bits = 8 if variant.startswith("int8") else 4
        group = int(variant.split("g")[1]) if "g" in variant else 0
        info = {}
        for k in DECODERS:  # fresh fp16 decoders from disk, then quantize
            if dict.__contains__(lm, k):
                dict.__delitem__(lm, k)
            info[k] = quantize_decoder_(lm[k], bits, group)
            lm[k].cpu()  # as low-VRAM mode keeps them between stages (else both sit on the GPU during decode)
        m_q, _, tex_q, mem = decode()
        v_q, f_q = m_q.vertices.float().cpu().numpy(), m_q.faces.cpu().numpy()
        rec = {**base, "variant": variant, **mem, "quantized": info,
               "weights_gib": {k: round(model_bytes(dict.__getitem__(lm, k)) / GiB, 3) for k in DECODERS},
               "faces": int(len(f_q)), **T.chamfer(v_q, f_q, v_fp, f_fp)}
        same = tex_q.coords.shape == tex_fp.coords.shape and bool((tex_q.coords == tex_fp.coords).all())
        rec["tex_same_voxels"] = same
        if same:
            rec["tex_psnr"] = round(psnr(tex_q.feats.clamp(0, 1), tex_fp.feats.clamp(0, 1)), 2)
            rec["tex_max_abs"] = round((tex_q.feats - tex_fp.feats).abs().max().item(), 4)
        print("RESULT " + json.dumps(rec), flush=True)
        with open(T.OUT / "dec_quant.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        for k in DECODERS:  # back to fp16 for the next variant
            dict.__delitem__(lm, k)


if __name__ == "__main__":
    main()
