"""RoundBit INT on TRELLIS.2-4B (~/3d_gen/TRELLIS.2-ROCm), 512 pipeline: sparse-structure flow, shape flow 512,
texture flow 512 (three 1.3B DiTs, 30 blocks of width 1536 each).

Run every command through the fork's runner with a RAM cap (see memory: low-VRAM runs reach 11-13 GB RSS):
    cd ~/3d_gen/TRELLIS.2-ROCm
    CAP="systemd-run --user --scope -q -p MemoryMax=13G -p MemorySwapMax=0 env MALLOC_MMAP_THRESHOLD_=65536"
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py gen                                  # DEFAULT: t1 INT4 DiTs + INT8 decoders
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py gen --stock --tag base               # unmodified reference
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py save_dec --tag t1                    # write the INT8 decoder files once
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py gen --quant int  --tag int4g64       # training-free INT4
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py capture --n_images 6                 # calibration calls
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py tune --tag t1 --model shape_slat_flow_model_512 --budget 240
    $CAP ./run_amd.sh $RB/trellis2/rb_trellis2.py gen --load t1 --tag t1               # tuned layers

gen      load -> quantize / swap in tuned layers (inside the model loader: the fork's LazyModels evicts and reloads
         models beyond TRELLIS_MODEL_CACHE_GB) -> 1-step warm-up (stage 1 differs on a process's first call)
         -> 512 generation (seed) -> .ply, PBR render grid .png, chamfer vs base_<img>_s<seed>.ply.
capture  hooks on block 0 of each DiT record every call's inputs (block input, t_emb, cond, rope phases ...)
         over n calibration images (example images, T.png excluded) -> runs/trellis2/cap/<model>.pt
tune     per block: FP reference on the FP cascade, linears -> roundbit IntLinear (activity-weighted clipping
         start), Adam on rounding + scales against the reference with a held-out best snapshot, freeze; the
         quantized cascade feeds the next block. Saved after every block (--resume continues, --budget seconds).
         Blocks stay bf16 (flash-attn); only the INT parameters are fp32.
"""
from __future__ import annotations

import argparse
import json
import re
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

RB = Path(__file__).resolve().parents[1]
OUT = RB / "runs" / "trellis2"
T2 = Path(__import__("os").environ.get("TRELLIS2_DIR", str(Path.home() / "3d_gen" / "TRELLIS.2-ROCm"))).expanduser()
sys.path[:0] = [str(T2), str(RB)]
FLOWS = ["sparse_structure_flow_model", "shape_slat_flow_model_512", "tex_slat_flow_model_512"]
# DiTs each pipeline type runs (1024_cascade is the fork's default: structure, shape 512, shape 1024, texture 1024)
PIPELINES = {"512": FLOWS,
             "1024_cascade": ["sparse_structure_flow_model", "shape_slat_flow_model_512",
                              "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"]}
ALL_FLOWS = sorted({m for v in PIPELINES.values() for m in v})
EXAMPLES = T2 / "assets" / "example_image"

from roundbit.int_layer import IntLinear  # noqa: E402


# ── training-free INT (gen --quant int) ─────────────────────────────────────────────────────────────────────────
def group_scale(Wg: torch.Tensor, L: int, clip_min=0.5, steps=10) -> torch.Tensor:
    amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    best_s, best_e = None, None
    for r in torch.linspace(1.0, clip_min, steps).tolist():
        s = amax * r / (L - 1)
        e = (Wg - (Wg / s).round().clamp(-L, L - 1) * s).pow(2).sum(-1, keepdim=True)
        if best_e is None:
            best_s, best_e = s, e
        else:
            better = e < best_e
            best_e, best_s = torch.where(better, e, best_e), torch.where(better, s, best_s)
    return best_s


def rtn_rotate() -> bool:
    """Training-free RTN path: rotation ON by default (2026-11-24, chain_rot.sh: q4rot clearly beats q4 on T/bear/gear
    and is free, the weights are rotated back). ROUNDBIT_ROTATE=0 turns it off. The tuned path (IntLinear) keeps its
    own default OFF: t1rot == t1 visually, but costs +3.3 MiB/DiT and a runtime x@R per layer in Int4Linear."""
    import os as _os
    return _os.environ.get("ROUNDBIT_ROTATE", "1") != "0"


@torch.no_grad()
def quantize_inplace(model: nn.Module, bits: int, group: int, only: str = "blocks", bits_of=None) -> dict:
    """RTN INT<bits> g<group> in place; bits_of(name) -> per-layer bits (e.g. 8 for the detail path)"""
    n_w = n_bits = n_layers = n8 = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or only not in name:
            continue
        b = bits_of(name) if bits_of else bits
        L = 2 ** (b - 1)
        n8 += m.weight.numel() if b > bits else 0
        w = m.weight
        W = w.detach().to("cuda", torch.float32)
        out, a = W.shape
        g = a if group <= 0 or group >= a else group
        Wg = torch.nn.functional.pad(W, (0, (-a) % g)).reshape(out, -1, g)
        from roundbit.int_layer import rotation
        R = rotation(g, W.device) if (rtn_rotate() and g & (g - 1) == 0 and a % g == 0) else None
        if R is not None:  # randomized Hadamard per group, quantize, rotate back (dense weights: no runtime cost)
            Wg = Wg @ R
        s = group_scale(Wg, L).half().float()
        Q = (Wg / s).round().clamp(-L, L - 1) * s
        if R is not None:
            Q = Q @ R.t()
        w.data.copy_(Q.reshape(out, -1)[:, :a].to(w.device, w.dtype))
        n_w += W.numel()
        n_bits += W.numel() * b + 16 * out * Wg.shape[1]
        n_layers += 1
    total = sum(p.numel() for p in model.parameters())
    return {"layers": n_layers, "q_weights": n_w, "bpw": n_bits / max(n_w, 1), "params": total, "high_share": n8 / max(n_w, 1),
            "mb_q": (n_bits + 16 * (total - n_w)) / 8e6, "mb_fp16": 16 * total / 8e6}


# ── layer plumbing ─────────────────────────────────────────────────────────────────────────────────────────────
class SparseInt(nn.Module):
    """Stand-in for trellis2 SparseLinear (input may be a VarLenTensor/SparseTensor or a plain tensor)."""

    def __init__(self, layer: IntLinear):
        super().__init__()
        self.lb = layer

    @property
    def bias(self):
        return self.lb.bias

    def forward(self, x):
        return x.replace(self.lb(x.feats)) if hasattr(x, "feats") else self.lb(x)


def _set(root: nn.Module, name: str, new: nn.Module):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, new)


def block_linears(block: nn.Module):
    return [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]


def apply_packed(model: nn.Module, path: Path, device="cuda") -> int:
    state = torch.load(path, map_location="cpu")
    from roundbit.int_layer import load_packed

    for name, (sd, meta) in state["layers"].items():
        layer = load_packed(sd, meta, device)
        _set(model.blocks, name, SparseInt(layer) if meta["sparse"] else layer)
    return len(state["layers"])


def chamfer(va, fa, vb, fb, n=100_000, seed=0) -> dict:
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    A, B = trimesh.Trimesh(va, fa, process=False), trimesh.Trimesh(vb, fb, process=False)
    np.random.seed(seed)
    pa, pb = A.sample(n), B.sample(n)
    da, db = cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0]
    diag = float(np.linalg.norm(B.extents))
    return {"chamfer_mean": float((da.mean() + db.mean()) / 2 / diag),
            "chamfer_p95": float(max(np.percentile(da, 95), np.percentile(db, 95)) / diag),
            "hausdorff": float(max(da.max(), db.max()) / diag)}


def load_pipeline():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    return pipe


def meta_swap(pipe, keys, path_of) -> bool:
    """Swap tuned INT layers in while the fork's loader still has the model on the meta device
    (trellis2.models.META_HOOKS): the bf16 block linears are never allocated, so loading a DiT costs ~0.7 GB of
    VRAM instead of 2.4 (4.95 GiB with the old fp32 build). Marks the model with _rb_loaded = #layers.
    Returns False when the fork has no META_HOOKS (then on_load's swap after loading is the only path)."""
    from trellis2 import models as tm

    if not hasattr(tm, "META_HOOKS"):
        return False
    specs = {pipe.models._specs[k]: k for k in keys}

    def hook(path, model, device):
        for spec, key in specs.items():
            if path.endswith(spec):
                f = path_of(key)
                if f is not None and f.exists():
                    model._rb_loaded = apply_packed(model, f, device)

    tm.META_HOOKS.append(hook)
    return True


DECODERS = ["shape_slat_decoder", "tex_slat_decoder"]


def dec_file(tag: str, key: str, bits: int) -> Path:
    return OUT / tag / f"{key}.int{bits}.pt"


def apply_dec_packed(model: nn.Module, path: Path, device="cuda") -> int:
    """Swap the saved weight-only INT layers (dec_quant.py format) into a decoder; works on the meta device too,
    so the fp16 weights are never allocated."""
    import dec_quant

    state = torch.load(path, map_location="cpu")
    for name, st in state["layers"].items():
        m = model.get_submodule(name)
        m._parameters.pop("weight", None)
        m.register_buffer("wq", st["wq"].to(device))
        m.register_buffer("ws", st["ws"].to(device))
        m.wshape, m.wdtype, m.wfan, m.wbits = torch.Size(st["wshape"]), getattr(torch, st["wdtype"]), st["wfan"], st["wbits"]
        cls = type(m)
        if not cls.__name__.startswith("Q"):
            if cls not in dec_quant._classes:
                dec_quant._classes[cls] = type("Q" + cls.__name__, (dec_quant._QWeight, cls), {})
            m.__class__ = dec_quant._classes[cls]
    return len(state["layers"])


def cmd_save_dec(a):
    """Quantize the latent decoders once (weight-only INT, dec_quant.py) and save them next to the tuned DiTs:
    runs/trellis2/<tag>/<decoder>.int<bits>.pt. gen then loads these files directly (no fp16 decoder load)."""
    import dec_quant

    pipe = load_pipeline()
    (OUT / a.tag).mkdir(parents=True, exist_ok=True)
    for key in DECODERS:
        model = pipe.models[key]
        before = dec_quant.model_bytes(model)
        r = dec_quant.quantize_decoder_(model, a.bits, a.group)
        layers = {n: {"wq": m.wq.cpu(), "ws": m.ws.cpu(), "wshape": list(m.wshape), "wdtype": str(m.wdtype).split(".")[-1],
                      "wfan": m.wfan, "wbits": m.wbits}
                  for n, m in model.named_modules() if hasattr(m, "wq")}
        f = dec_file(a.tag, key, a.bits)
        torch.save({"layers": layers, "info": {"bits": a.bits, "group": a.group, "weights": r["weights"]}}, f)
        print(f"SAVED {key}: {r['layers']} layers, weights {before / 1e6:.0f} -> {dec_quant.model_bytes(model) / 1e6:.0f} MB "
              f"in memory, file {f} ({f.stat().st_size / 1e6:.0f} MB)", flush=True)
        del pipe.models[key]
        torch.cuda.empty_cache()


class SpanBridge(nn.Module):
    """replaces a span of DiT blocks: x + [x, 1] W (fp32), dense or sparse input; args (t_emb, cond, ...) ignored"""

    def __init__(self, W):
        super().__init__()
        self.register_buffer("W", W.float())

    def forward(self, x, *args, **kwargs):
        f = x.feats if hasattr(x, "feats") else x
        shp = f.shape
        f2 = f.reshape(-1, shp[-1]).float()
        y = (f2 + f2 @ self.W[:-1] + self.W[-1]).to(f.dtype).reshape(shp)
        return x.replace(y) if hasattr(x, "feats") else y


class SpanSkip(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def apply_bridges(model, spans, device="cuda") -> int:
    """spans from span_probe.py --fit: block i -> bridge (or skip), blocks i+1 .. i+len-1 -> skip"""
    n = 0
    for s in spans:
        i, L = s["i"], s["len"]
        model.blocks[i] = SpanBridge(s["W"]).to(device) if s["kind"] == "linear" else SpanSkip()
        for j in range(i + 1, i + L):
            model.blocks[j] = SpanSkip()
        n += L
    return n


def on_load(pipe, keys, fn):
    """Call fn(key, model) for each model in keys now (if resident) and every time LazyModels (re)loads it."""
    lm = pipe.models
    spec_key = {lm._specs[k]: k for k in keys}
    orig = lm._loader

    def loader(spec, device):
        model = orig(spec, device)
        if spec in spec_key:
            fn(spec_key[spec], model)
        return model

    lm._loader = loader
    for k in list(dict.keys(lm)):
        if k in keys:
            fn(k, dict.__getitem__(lm, k))


# ── call capture: pack / unpack block arguments ────────────────────────────────────────────────────────────────
def _pack(o, tab: list):
    from trellis2.modules.sparse import SparseTensor, VarLenTensor

    if isinstance(o, SparseTensor):
        return {"_sp": _pack(o.coords, tab), "feats": _pack(o.feats, tab), "shape": list(o.shape)}
    if isinstance(o, VarLenTensor):
        return {"_vl": _pack(o.feats, tab), "layout": [(s.start, s.stop) for s in o.layout]}
    if torch.is_tensor(o):  # dedupe by content (cond, rope phases, coords repeat across calls)
        t = o.detach().cpu()
        for i, u in enumerate(tab):
            if u.shape == t.shape and u.dtype == t.dtype and torch.equal(u, t):
                return {"_t": i}
        tab.append(t.clone())
        return {"_t": len(tab) - 1}
    if isinstance(o, (list, tuple)):
        return [_pack(v, tab) for v in o]
    return o


def _unpack(o, tab: list, cache: dict, dev="cuda"):
    from trellis2.modules.sparse import SparseTensor, VarLenTensor

    if isinstance(o, dict) and "_t" in o:
        k = ("t", o["_t"])
        if k not in cache:
            cache[k] = tab[o["_t"]].to(dev)
        return cache[k]
    if isinstance(o, dict) and "_vl" in o:
        return VarLenTensor(_unpack(o["_vl"], tab, cache, dev), [slice(a, b) for a, b in o["layout"]])
    if isinstance(o, dict) and "_sp" in o:
        return SparseTensor(feats=_unpack(o["feats"], tab, cache, dev), coords=_unpack(o["_sp"], tab, cache, dev),
                            shape=torch.Size(o["shape"]))
    if isinstance(o, list):
        return [_unpack(v, tab, cache, dev) for v in o]
    return o


def cmd_capture(a):
    from PIL import Image

    imgs = sorted(p for p in EXAMPLES.iterdir() if p.suffix in (".png", ".webp", ".jpg") and p.name != "T.png")
    step = max(1, len(imgs) // a.n_images)
    imgs = imgs[a.offset::step][: a.n_images]
    pipe = load_pipeline()
    targets = [m for m in (a.models.split(",") if a.models else PIPELINES[a.pipeline]) if m]
    cap = {k: {"tab": [], "calls": [], "images": []} for k in targets}
    hooked = set()
    cur = {"gen": 0}

    def hook_model(key, model):
        if id(model) in hooked:
            return
        hooked.add(id(model))

        def pre(_m, args, kwargs):
            x = args[0]
            feats = x.feats if hasattr(x, "feats") else x
            c = cap[key]
            c["calls"].append({"gen": cur["gen"], "x": feats.detach().cpu().clone(),
                               "x_meta": {"coords": _pack(x.coords, c["tab"]), "shape": list(x.shape)}
                               if hasattr(x, "feats") else None,
                               "args": _pack(list(args[1:]), c["tab"]), "kwargs": _pack(dict(kwargs), c["tab"])})
        model.blocks[0].register_forward_pre_hook(pre, with_kwargs=True)

    on_load(pipe, targets, hook_model)
    t0 = time.time()
    for gi, p in enumerate(imgs):
        cur["gen"] = gi
        before = {k: len(cap[k]["calls"]) for k in targets}
        try:
            pipe.run(Image.open(p), pipeline_type=a.pipeline, seed=a.seed)
        except RuntimeError as e:
            # the calls are recorded during sampling; mesh decoding afterwards can run out of VRAM on large objects
            # (1024_cascade, CuMesh OOM) -- that does not touch the captured calls, so keep them if every model ran
            if not all(len(cap[k]["calls"]) > before[k] for k in targets):
                raise
            print(f"note: {p.name}: error after sampling ({str(e).splitlines()[0][:120]}); calls kept", flush=True)
            torch.cuda.empty_cache()
        print(f"captured {p.name} ({time.time() - t0:.0f}s): " +
              ", ".join(f"{k}={len(cap[k]['calls'])}" for k in targets), flush=True)
        for k in targets:
            cap[k]["images"].append(p.name)
    (OUT / "cap").mkdir(parents=True, exist_ok=True)
    for k in targets:
        path = OUT / "cap" / f"{k}{a.suffix}.pt"
        torch.save(cap[k], path)
        n_tok = sorted({c["x"].shape[-2] if c["x"].dim() == 3 else c["x"].shape[0] for c in cap[k]["calls"]})
        print(f"CAPTURE {k}: {len(cap[k]['calls'])} calls, tokens {n_tok[:3]}..{n_tok[-1:]}, -> {path}", flush=True)


# ── tune ───────────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_tune(a):
    dev = "cuda"
    t_start = time.time()
    pipe = load_pipeline()
    model = pipe.models[a.model].to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = model.blocks
    from trellis2.modules.sparse import SparseTensor

    # captures one at a time, every call_stride-th call; one SparseTensor geometry per distinct coords (= per
    # generation), fwd() swaps the features in via replace() -- a geometry per call kept a second GPU copy of every
    # block input (the 1024 models see up to ~20k tokens per call)
    calls, feats = [], []
    for sfx in a.cap.split(","):
        cap = torch.load(OUT / "cap" / f"{a.model}{sfx}.pt", map_location="cpu")
        cache, geos = {}, {}
        for c in cap["calls"][:: a.call_stride]:
            geo = None
            if c["x_meta"] is not None:
                key = c["x_meta"]["coords"]["_t"]
                if key not in geos:
                    geos[key] = SparseTensor(feats=c["x"].to(dev), shape=torch.Size(c["x_meta"]["shape"]),
                                             coords=_unpack(c["x_meta"]["coords"], cap["tab"], cache, dev))
                geo = geos[key]
            calls.append({"geo": geo, "args": _unpack(c["args"], cap["tab"], cache, dev),
                          "kwargs": _unpack(c["kwargs"], cap["tab"], cache, dev)})
            feats.append(c["x"])
        del cap
    # three copies live at once (FP cascade, INT cascade, references): host RAM when they would crowd the card
    total_gb = 3 * sum(t.numel() * t.element_size() for t in feats) / 2**30
    offload = a.offload == "on" or (a.offload == "auto" and total_gb > a.offload_gb)
    keep_dev = (lambda t: t.cpu()) if offload else (lambda t: t)
    x_fp = feats if offload else [t.to(dev) for t in feats]
    del feats
    n = len(calls)
    x_q = [t.clone() for t in x_fp]
    val = sorted(set(range(0, n, max(2, n // max(1, a.val))))) if a.val else []
    train_idx = [i for i in range(n) if i not in set(val)]
    print(f"tune {a.model}: {n} calls ({len(val)} held out), block input {tuple(x_fp[0].shape)}, "
          f"{len(blocks)} blocks, load {time.time() - t_start:.0f}s", flush=True)

    def fwd(block, feats, i):
        c = calls[i]
        if c["geo"] is not None:
            return block(c["geo"].replace(feats.to(dev)), *c["args"], **c["kwargs"]).feats
        return block(feats.to(dev), *c["args"], **c["kwargs"])

    @torch.no_grad()
    def rel_err(block, idx, ref):
        num = den = 0.0
        for i in idx:
            r = ref[i].to(dev).float()
            num += (fwd(block, x_q[i], i).float() - r).pow(2).mean().item()
            den += r.pow(2).mean().item()
        return num / max(den, 1e-30)

    run_dir = OUT / a.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / f"{a.model}.pt"
    layers_out, start = {}, 0
    # cascades after the last finished block, saved when a chunk stops on its budget: resuming by recomputing both
    # cascades through all finished blocks cost 130 s at block 20 of the 1024 shape model (2 blocks per 5-min chunk)
    cache_path = run_dir / f"{a.model}.cascade.pt"
    cache_key = {"cap": a.cap, "call_stride": a.call_stride, "n": n}
    cached = None
    if a.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        layers_out, start = state["layers"], state["info"]["blocks_done"]
        if cache_path.exists():
            c = torch.load(cache_path, map_location="cpu", mmap=True)
            if c.get("key") == cache_key and c.get("blocks_done") == start:
                cached = c
    if cached is not None:
        x_fp = [t if offload else t.to(dev) for t in cached["x_fp"]]
        x_q = [t if offload else t.to(dev) for t in cached["x_q"]]
        del cached
        print(f"resumed after block {start - 1} from the cascade cache ({time.time() - t_start:.0f}s)", flush=True)
    elif a.resume and state_path.exists():
        with torch.no_grad():
            for bi in range(start):
                block = blocks[bi]
                ref = [keep_dev(fwd(block, x_fp[i], i)) for i in range(n)]
                for name, (sd, meta) in layers_out.items():
                    if name.startswith(f"{bi}."):
                        layer = IntLinear.from_packed(sd, meta, dev)
                        _set(block, name.split(".", 1)[1], SparseInt(layer) if meta["sparse"] else layer)
                x_q = [keep_dev(fwd(block, x_q[i], i)) for i in range(n)]
                x_fp = ref
        print(f"resumed after block {start - 1} ({time.time() - t_start:.0f}s)", flush=True)
    ginit, gb_out = None, {}
    if a.gbits_from:
        from roundbit import group_bits as gb

        gpath = run_dir / f"{a.model}.gb_init.pt"
        if a.resume and state_path.exists():
            gb_out = torch.load(state_path, map_location="cpu").get("gb", {})
        if gpath.exists() and a.resume:
            ginit = torch.load(gpath, map_location="cpu")
        else:
            tA = time.time()
            src = torch.load(OUT / a.gbits_from / f"{a.model}.pt", map_location="cpu")["layers"]
            st_ = {}

            def mk2(key):
                def pre(_m, args):
                    x = args[0]
                    x = (x.feats if hasattr(x, "feats") else x).detach().float()
                    x = x.reshape(-1, x.shape[-1])
                    acc = st_.setdefault(key, [0, 0, 0, 0, 0])
                    acc[0] = acc[0] + x.pow(2).sum(0)
                    acc[1] += x.shape[0]

                def post(_m, args, o):
                    o = (o.feats if hasattr(o, "feats") else o).detach().float()
                    o = o.reshape(-1, o.shape[-1])
                    acc = st_[key]
                    acc[2] = acc[2] + o.sum(0)
                    acc[3] = acc[3] + o.pow(2).sum(0)
                    acc[4] += o.shape[0]
                return pre, post
            hooks = []
            for bi_, blk in enumerate(blocks):
                for k_, m_ in block_linears(blk):
                    pre, post = mk2(f"{bi_}.{k_}")
                    hooks += [m_.register_forward_pre_hook(pre), m_.register_forward_hook(post)]
            with torch.no_grad():  # FP activity of every block linear, on every k-th call
                for i in range(0, n, a.gbits_stat_stride):
                    f_ = x_fp[i]
                    for blk in blocks:
                        f_ = fwd(blk, f_, i)
            for h in hooks:
                h.remove()
            holder, score = nn.Module(), {}
            for j_, (name, (sd, meta)) in enumerate(src.items()):
                layer = IntLinear.from_packed(sd, meta, dev)
                holder.add_module(f"l{j_}", layer)
                acc = st_[name]
                xc = torch.nn.functional.pad(acc[0] / acc[1], (0, layer.pad)).reshape(layer.ng, layer.g)
                if getattr(layer, "R", None) is not None:
                    xc = xc.mean(-1, keepdim=True).expand(layer.ng, layer.g)
                mean = acc[2] / acc[4]
                var = (acc[3] / acc[4] - mean.pow(2)).clamp_min(1e-6)
                score[id(layer)] = (xc[None], 1.0 / var[:, None, None])
            del st_
            gb.attach(holder, light=True)  # widths only: a float mask per weight would be 5 GB
            n_dem = gb.demote(holder, a.gbits_step, score)
            bs_ = gb.bit_stats(holder)
            ginit = {"layers": {name: (holder.get_submodule(f"l{j_}").packed_state_dict(), src[name][1],
                                       holder.get_submodule(f"l{j_}").gb.to(torch.int8).cpu())
                                for j_, name in enumerate(src)}, "stats": bs_, "groups": n_dem}
            del holder, score, src
            torch.cuda.empty_cache()
            torch.save(ginit, gpath)
            print(f"GBITS demoted {n_dem} groups from {a.gbits_from}: INT {bs_['uniform_mb']} -> {bs_['int_mb']} MB, "
                  f"widths {bs_['bits_share']} ({time.time() - tA:.0f}s)", flush=True)
    gen = torch.Generator().manual_seed(start)
    last_s, info = 0.0, None
    for bi in range(start, len(blocks)):
        if a.budget and time.time() - t_start + 1.2 * last_s > a.budget:
            print(f"budget: stopping before block {bi} ({time.time() - t_start:.0f}s used)", flush=True)
            tmp = cache_path.with_suffix(".tmp")
            torch.save({"key": cache_key, "blocks_done": bi, "x_fp": [t.cpu() for t in x_fp],
                        "x_q": [t.cpu() for t in x_q]}, tmp)
            tmp.replace(cache_path)
            print(f"saved cascade cache ({time.time() - t_start:.0f}s)", flush=True)
            break
        t0 = time.time()
        block = blocks[bi]
        act = {}

        def mk(key):
            def hook(_m, args):
                x = args[0]
                x = (x.feats if hasattr(x, "feats") else x).detach().float()
                sq = x.reshape(-1, x.shape[-1]).pow(2)
                acc = act.setdefault(key, [torch.zeros(x.shape[-1], device=x.device), 0])
                acc[0] += sq.sum(0)
                acc[1] += sq.shape[0]
            return hook
        hooks = [m.register_forward_pre_hook(mk(k)) for k, m in block_linears(block)]
        with torch.no_grad():
            ref = [keep_dev(fwd(block, x_fp[i], i)) for i in range(n)]
        for h in hooks:
            h.remove()
        converted = {}
        for name, lin in block_linears(block):
            sparse = type(lin) is not nn.Linear
            if ginit is not None:  # the demoted layer of the source tag, re-tuned below inside its group widths
                sd_, meta_, gbw = ginit["layers"][f"{bi}.{name}"]
                layer = IntLinear.from_packed(sd_, meta_, dev)
            else:
                scale = (act[name][0] / act[name][1]).sqrt() if name in act else None
                layer = IntLinear(lin, a.bits, a.group, in_scale=scale).to(dev)
            _set(block, name, SparseInt(layer) if sparse else layer)
            converted[name] = (layer, sparse)
        if ginit is not None:
            gb.attach(block)
            for name, (layer, _) in converted.items():
                layer.gb = ginit["layers"][f"{bi}.{name}"][2].to(dev, torch.int16)
                gb.set_mask(layer, "bits")
        st = {"block": bi, "rel_init": rel_err(block, val or train_idx[:8], ref)}
        if a.iters > 0:
            if ginit is not None:
                lat = [gb.start_recal(l) for l, _ in converted.values()]
            else:
                lat = [p for l, _ in converted.values() for p in l.latent_parameters()]
            sc = [p for l, _ in converted.values() for p in l.scale_parameters()]
            for p in lat + sc:
                p.requires_grad_(True)
            opt = torch.optim.AdamW(lat + sc, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.iters // 20)) * (
                0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, a.iters) / a.iters))))
            ref_pow = sum(r.to(dev).float().pow(2).mean().item() for r in ref) / n
            best = (st["rel_init"], 0, [p.detach().clone() for p in lat + sc])
            every = max(1, a.iters // 5)
            for step in range(a.iters):
                idx = [train_idx[int(j)] for j in torch.randint(0, len(train_idx), (a.calls_per_step,), generator=gen)]
                loss = sum((fwd(block, x_q[i], i).float() - ref[i].to(dev).float()).pow(2).mean() for i in idx) / (
                    ref_pow * len(idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                if val and ((step + 1) % every == 0 or step == a.iters - 1):
                    r = rel_err(block, val, ref)
                    if r < best[0]:
                        best = (r, step + 1, [p.detach().clone() for p in lat + sc])
            with torch.no_grad():
                for p, b in zip(lat + sc, best[2]):
                    p.copy_(b)
            for p in lat + sc:
                p.requires_grad_(False)
                p.grad = None
            del opt
            st["best_step"] = best[1]
        if ginit is not None:
            for name, (layer, _) in converted.items():
                gb.fold(layer)
                gb_out[f"{bi}.{name}"] = layer.gb.to(torch.int8).cpu()
            gb.detach(block)
        for name, (layer, sparse) in converted.items():
            layer.freeze()
            sd = layer.packed_state_dict()
            layers_out[f"{bi}.{name}"] = (sd, {**layer.packed_meta(), "sparse": sparse})
            # deploy form right away: frees the fp32 training copies (Hi3D: ~0.7 GB per block, OOM at block 9)
            # and the later blocks are tuned on what the real INT4 kernels compute
            from roundbit.int_layer import load_packed

            dep = load_packed(sd, layers_out[f"{bi}.{name}"][1], dev)
            _set(block, name, SparseInt(dep) if sparse else dep)
        st["rel_final"] = rel_err(block, val or train_idx[:8], ref)
        with torch.no_grad():
            x_q = [keep_dev(fwd(block, x_q[i], i)) for i in range(n)]
        x_fp = ref
        st["cascade_rel"] = sum((x_q[i].to(dev).float() - x_fp[i].to(dev).float()).pow(2).mean().item()
                                for i in range(n)) / max(sum(x_fp[i].to(dev).float().pow(2).mean().item()
                                                             for i in range(n)), 1e-30)
        st["s"] = round(time.time() - t0, 1)
        last_s = time.time() - t0
        print("BLOCK " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in st.items()}), flush=True)
        with open(run_dir / f"{a.model}.blocks.jsonl", "a") as f:
            f.write(json.dumps(st) + "\n")
        info = {"args": vars(a), "blocks_done": bi + 1, "n_blocks": len(blocks), "layers": len(layers_out),
                "cascade_rel": st["cascade_rel"]}
        if ginit is not None:
            info["gbits"] = ginit["stats"]
        torch.save({"layers": layers_out, "info": info, **({"gb": gb_out} if ginit is not None else {})}, state_path)
    if info is not None and info["blocks_done"] == len(blocks):
        cache_path.unlink(missing_ok=True)
        print("TUNED " + json.dumps(info), flush=True)


# ── coarse meshes for Hi3D v1.0 (its calibration / test objects) ─────────────────────────────────────────────
def cmd_coarse(a):
    """bf16 512 generation per example image -> simplified, y-up (x, z, -y) like the TRELLIS GLB export ->
    runs/hi3d/coarse/<stem>.glb + .png (the image Hi3D conditions on). Skips stems that already exist."""
    import numpy as np
    import trimesh
    from PIL import Image

    dst = RB / "runs" / "hi3d" / "coarse"
    dst.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in EXAMPLES.iterdir() if p.suffix in (".png", ".webp", ".jpg") and p.name != "T.png")
    imgs = imgs[a.offset::a.stride][: a.n]
    todo = [p for p in imgs if not (dst / f"{p.stem[:10]}.glb").exists()]
    pipe = load_pipeline() if todo else None
    t0 = time.time()
    for p in todo:
        if a.budget and time.time() - t0 > a.budget:
            print(f"budget: stopping before {p.name}", flush=True)
            break
        img = Image.open(p)
        mesh = pipe.run(img, pipeline_type="512", seed=a.seed)[0]
        mesh.simplify(a.faces)
        v = mesh.vertices.float().cpu().numpy()
        v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)  # z-up -> y-up, as o_voxel.postprocess.to_glb
        stem = p.stem[:10]
        trimesh.Trimesh(v, mesh.faces.cpu().numpy(), process=False).export(str(dst / f"{stem}.glb"))
        img.convert("RGBA").save(dst / f"{stem}.png")
        print(f"COARSE {stem}: {len(mesh.faces)} faces, extents {np.ptp(v, 0).round(3).tolist()} "
              f"({time.time() - t0:.0f}s)", flush=True)


# ── generate + evaluate ───────────────────────────────────────────────────────────────────────────────────────
def cmd_gen(a):
    import imageio
    import numpy as np
    import trimesh
    from PIL import Image

    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils

    OUT.mkdir(parents=True, exist_ok=True)
    if a.stock:
        a.load, a.dec_bits = "", 0
    if a.quant == "int" and not any(x.startswith("--load") for x in sys.argv):  # training-free INT runs: no t1 underneath
        a.load = ""
    tag = a.tag or ("base" if a.quant == "none" and not a.load and not a.dec_bits else
                    f"{a.load}{f'_dec{a.dec_bits}' if a.dec_bits else ''}" if a.load else f"int{a.bits}g{a.group}")
    pt = "" if a.pipeline == "512" else f"_{a.pipeline}"
    stem = OUT / f"{tag}{pt}_{Path(a.image).stem[:12]}_s{a.seed}"
    import os as _os
    rec = {"tag": tag, "args": vars(a), "loads": [], "rotate_rtn": rtn_rotate(), "rotate_env": _os.environ.get("ROUNDBIT_ROTATE", "unset")}
    t0 = time.time()
    pipe = load_pipeline()
    rec["load_s"] = round(time.time() - t0, 1)
    targets = [k for k in (a.models.split(",") if a.models else PIPELINES[a.pipeline]) if k]
    decs = ["shape_slat_decoder", "tex_slat_decoder"] if a.dec_bits else []
    if decs:
        targets = sorted(set(targets) | set(decs))
    if a.cond != "fp32":  # image encoder: not in LazyModels, lives on the pipeline object
        cm = pipe.image_cond_model.model
        cm.to(torch.float16 if a.cond == "fp16" else torch.bfloat16)  # fp16 overflows inside DINOv3 (NaN)
        if a.cond == "int8":
            import dec_quant

            r = dec_quant.quantize_decoder_(cm, 8, 128, kinds=("Linear",))
            rec["loads"].append(("image_cond_model", "int8", r["layers"]))
        rec["cond_mb"] = round(sum(t.numel() * t.element_size() for t in list(cm.parameters()) + list(cm.buffers()))
                               / 1e6, 1)

    bridges = torch.load(OUT / "bridges" / f"{a.bridges}.pt", map_location="cpu")["models"] if a.bridges else {}
    if a.bridges:
        targets = sorted(set(targets) | set(bridges))

    dvecs = torch.load(OUT / "dvec" / f"{a.dvec}.pt", map_location="cpu")["vecs"] if a.dvec else {}
    if a.dvec:
        targets = sorted(set(targets) | set(dvecs))

    dec_saved = {k: dec_file(a.load or "t1", k, a.dec_bits) for k in decs}
    dec_saved = {k: f for k, f in dec_saved.items() if f.exists()}

    def prepare(key, model):
        if key in decs:
            if getattr(model, "_rb_dec_loaded", None):  # saved INT decoder swapped in on the meta device
                rec["loads"].append((key, f"int{a.dec_bits}-saved", model._rb_dec_loaded))
                model._rb_dec = True
                import dec_quant

                rec.setdefault("dec_mb", {})[key] = round(dec_quant.model_bytes(model) / 1e6, 1)
            elif not getattr(model, "_rb_dec", False):
                import dec_quant

                model._rb_dec = True
                r = dec_quant.quantize_decoder_(model, a.dec_bits, a.dec_group)
                rec.setdefault("dec_mb", {})[key] = round(dec_quant.model_bytes(model) / 1e6, 1)
                rec["loads"].append((key, f"int{a.dec_bits}", r["layers"]))
                torch.cuda.empty_cache()
            print(f"PREPARE {rec['loads'][-1]}", flush=True)
            return
        if key in bridges and not getattr(model, "_rb_bridged", False):
            model._rb_bridged = True
            rec["loads"].append((key, "bridges", apply_bridges(model, bridges[key])))
        path = OUT / a.load / f"{key}.pt" if a.load else None
        if getattr(model, "_rb_loaded", None):  # swapped on the meta device already (meta_swap)
            rec["loads"].append((key, "tuned-meta", model._rb_loaded))
        elif path is not None and path.exists():
            info = torch.load(path, map_location="cpu")["info"]
            assert info["blocks_done"] == info["n_blocks"], f"{path}: only {info['blocks_done']} blocks tuned"
            rec["loads"].append((key, "tuned", apply_packed(model, path)))
        elif a.quant == "int":
            hb = (lambda n, key=key: a.hi_bits if re.search(a.hi, f"{key}/{n}") else a.bits) if a.hi else None
            r = quantize_inplace(model, a.bits, a.group, bits_of=hb)
            rec["loads"].append((key, "rtn", r["layers"], round(r["bpw"], 3), round(r["high_share"], 3)))
        if a.dvec and key in dvecs and not getattr(model, "_rb_dvec", False) and (
                not a.dvec_models or key in a.dvec_models.split(",")):
            import dvec

            model._rb_dvec = True
            blks = None
            if a.dvec_blocks:
                lo, hi = (int(v) for v in a.dvec_blocks.split("-"))
                blks = set(range(lo, hi + 1))
            rec["loads"].append((key, "dvec", dvec.apply(model, key, dvecs[key], a.dvec_alpha, blks)))
        if a.comp_gain and key in a.comp_models.split(",") and not getattr(model, "_rb_cg", False):
            model._rb_cg = True
            rec["loads"].append((key, "comp_gain", apply_comp_gain(model, a.comp_gain)))
        if a.teacache >= 0 and not getattr(model, "_rb_tea", False) and (not a.tea_models or key in a.tea_models.split(",")):
            import teacache

            rs = [float(v) for v in a.tea_rescale.split(",")] if a.tea_rescale else None
            rec["loads"].append((key, "teacache", teacache.install(model, key, a.teacache, OUT / f"tea_{tag}.jsonl",
                                                                  a.tea_max_skip, rs, a.tea_pred)))
        print(f"PREPARE {rec['loads'][-1] if rec['loads'] else key}", flush=True)

    if a.load:
        rec["meta_swap"] = meta_swap(pipe, [k for k in targets if k not in decs], lambda k: OUT / a.load / f"{k}.pt")
    if dec_saved:
        from trellis2 import models as tm

        if hasattr(tm, "META_HOOKS"):
            dspec = {pipe.models._specs[k]: k for k in dec_saved}

            def dhook(path, model, device):
                for spec, key in dspec.items():
                    if path.endswith(spec):
                        model._rb_dec_loaded = apply_dec_packed(model, dec_saved[key], device)
            tm.META_HOOKS.append(dhook)
    if a.comp_gain:
        targets = sorted(set(targets) | set(a.comp_models.split(",")))
    if a.load or a.quant == "int" or a.bridges or a.teacache >= 0 or a.dvec or a.comp_gain or decs:
        on_load(pipe, targets, prepare)
    img = Image.open(a.image)
    one = {"steps": 1}
    pipe.run(img, pipeline_type=a.pipeline, seed=a.seed, sparse_structure_sampler_params=one,
             shape_slat_sampler_params=one, tex_slat_sampler_params=one)  # warm-up / autotune
    if a.teacache >= 0:
        import teacache

        teacache.STATS.update(calls=0, skipped=0)
        with open(OUT / f"tea_{tag}.jsonl", "a") as fh:
            fh.write(json.dumps({"run": "timed", "image": a.image, "seed": a.seed}) + "\n")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sp = json.loads(a.sampler) if a.sampler else {}
    mesh = pipe.run(img, pipeline_type=a.pipeline, seed=a.seed,
                    sparse_structure_sampler_params=sp.get("ss", {}), shape_slat_sampler_params=sp.get("shape", {}),
                    tex_slat_sampler_params=sp.get("tex", {}))[0]
    torch.cuda.synchronize()
    rec["gen_s"] = round(time.time() - t0, 1)
    if a.teacache >= 0:
        import teacache

        rec["tea"] = dict(teacache.STATS)
    rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    v, f = mesh.vertices.float().cpu().numpy(), mesh.faces.cpu().numpy()
    rec["verts"], rec["faces"] = int(len(v)), int(len(f))
    rec["finite"] = bool(np.isfinite(v).all()) and bool(torch.isfinite(mesh.attrs).all())
    trimesh.Trimesh(v, f, process=False).export(str(stem) + ".ply")
    base = OUT / f"base{pt}_{Path(a.image).stem[:12]}_s{a.seed}.ply"
    if base.exists() and tag != "base":
        B = trimesh.load(str(base), process=False)
        rec.update(chamfer(v, f, np.asarray(B.vertices), np.asarray(B.faces)))
    print("EVAL " + json.dumps({k: rec[k] for k in rec if k != "args"}), flush=True)
    envmap = EnvMap(torch.tensor(render_utils.load_hdri(str(T2 / "assets/hdri/forest.exr")),
                                 dtype=torch.float32, device="cuda"))
    frames = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap, num_frames=4,
                                                                        resolution=512))
    grid = np.concatenate([np.concatenate(frames[:2], 1), np.concatenate(frames[2:4], 1)], 0)
    imageio.imwrite(str(stem) + ".png", grid)
    rec["png"] = str(stem) + ".png"
    with open(OUT / "results.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("DONE", flush=True)


def apply_comp_gain(model, spec):
    """forward hooks multiplying the output of blocks[i].<component> by a gain; returns the hooked count"""
    n = 0
    for part in filter(None, spec.split(";")):
        where, g = part.split("=")
        blocks, comp = where.split(".")
        idx = set()
        for b in blocks.split(","):
            lo, _, hi = b.partition("-")
            idx |= set(range(int(lo), int(hi or lo) + 1))
        for i in sorted(idx):
            def hook(_m, _a, o, g=float(g)):
                return o.replace(o.feats * g) if hasattr(o, "feats") else o * g
            getattr(model.blocks[i], comp).register_forward_hook(hook)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--quant", default="none", choices=["none", "int"], help="int = training-free INT for --models")
    g.add_argument("--load", default="t1", help="tune tag: swap in runs/trellis2/<tag>/<model>.pt where present "
                   "(default t1 = calibrated INT4 DiTs; '' = none)")
    g.add_argument("--stock", action="store_true",
                   help="the unmodified pipeline (bf16 DiTs, fp16 decoders): sets --load '' --dec_bits 0 (reference runs)")
    g.add_argument("--bits", type=int, default=4)
    g.add_argument("--group", type=int, default=64)
    g.add_argument("--pipeline", default="512", choices=sorted(PIPELINES))
    g.add_argument("--models", default="", help="comma list of pipeline models to quantize / load (default: all the "
                                                "pipeline runs)")
    g.add_argument("--image", default=str(EXAMPLES / "T.png"))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--tag", default="")
    g.add_argument("--sampler", default="",
                   help='JSON sampler overrides per stage, e.g. {"shape": {"steps": 24}, "tex": {"steps": 24}}')
    g.add_argument("--dvec", default="", help="dvec.py build name: add detail control vectors to block outputs")
    g.add_argument("--dvec_alpha", type=float, default=1.0)
    g.add_argument("--hi", default="", help="regex on '<model>/<layer name>': these layers get --hi_bits (detail path)")
    g.add_argument("--hi_bits", type=int, default=8)
    g.add_argument("--comp_gain", default="", help="scale block component outputs, e.g. '11.cross_attn=1.5;17,19.self_attn=1.3;"
                   "14-26.mlp=0.8' (circuit_probe.py findings)")
    g.add_argument("--comp_models", default="shape_slat_flow_model_512")
    g.add_argument("--dvec_models", default="", help="comma list (default: every DiT in the dvec file)")
    g.add_argument("--dvec_blocks", default="", help="block range lo-hi (default: all)")
    g.add_argument("--teacache", type=float, default=-1,
                   help="TeaCache step caching threshold (teacache.py); 0 = probe (log, never skip), -1 = off")
    g.add_argument("--tea_max_skip", type=int, default=2, help="max consecutive skipped calls per CFG branch")
    g.add_argument("--tea_pred", default="reuse", choices=["reuse", "lin"],
                   help="skipped call: reuse the last residual (TeaCache) or extrapolate it linearly in t (TaylorSeer-like)")
    g.add_argument("--tea_models", default="", help="comma list of DiTs to cache (default: all)")
    g.add_argument("--tea_rescale", default="", help="polynomial coefficients for the indicator (highest first)")
    g.add_argument("--dec_bits", type=int, default=8, choices=[0, 4, 8],
                   help="weight-only INT for the latent decoders (shape + texture; dec_quant.py), 0 = fp16. Default 8: "
                        "loaded from runs/trellis2/<--load>/<decoder>.int8.pt when saved (save_dec), else quantized on load")
    g.add_argument("--dec_group", type=int, default=128)
    g.add_argument("--cond", default="fp32", choices=["fp32", "fp16", "bf16", "int8"],
                   help="DINOv3 image encoder weights (stock: fp32); int8 = weight-only INT8 g128 on its linears, bf16 rest. "
                        "fp16 overflows (NaN features, empty structure)")
    g.add_argument("--bridges", default="", help="span_probe.py --fit name: replace block spans by linear bridges")
    c = sub.add_parser("capture")
    c.add_argument("--n_images", type=int, default=6)
    c.add_argument("--offset", type=int, default=0)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--suffix", default="", help="capture file suffix (several captures can be combined in tune)")
    c.add_argument("--pipeline", default="512", choices=sorted(PIPELINES))
    c.add_argument("--models", default="", help="comma list of models to record (default: all the pipeline runs)")
    o = sub.add_parser("coarse")
    o.add_argument("--n", type=int, default=8)
    o.add_argument("--offset", type=int, default=3)
    o.add_argument("--stride", type=int, default=9, help="spread the picks over the 72 example images")
    o.add_argument("--faces", type=int, default=200_000)
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--budget", type=float, default=230)
    sd_ = sub.add_parser("save_dec", help="quantize the latent decoders once and save them (gen loads the files)")
    sd_.add_argument("--tag", default="t1")
    sd_.add_argument("--bits", type=int, default=8, choices=[4, 8])
    sd_.add_argument("--group", type=int, default=128)
    t = sub.add_parser("tune")
    t.add_argument("--model", required=True, choices=ALL_FLOWS)
    t.add_argument("--tag", required=True)
    t.add_argument("--cap", default="", help="comma list of capture suffixes ('' = the default capture)")
    t.add_argument("--bits", type=int, default=4)
    t.add_argument("--group", type=int, default=64)
    t.add_argument("--iters", type=int, default=50)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--calls_per_step", type=int, default=2)
    t.add_argument("--val", type=int, default=12, help="held-out calls for the best snapshot")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--call_stride", type=int, default=1, help="use every k-th captured call (all timesteps stay)")
    t.add_argument("--offload", default="auto", choices=["auto", "on", "off"])
    t.add_argument("--offload_gb", type=float, default=10.0, help="auto: host RAM when the 3 cascades exceed this")
    t.add_argument("--budget", type=float, default=0, help="seconds; do not start a block that would exceed it")
    t.add_argument("--gbits_from", default="",
                   help="per-group bit allocation round (roundbit/group_bits.py): start from this tuned tag, demote the "
                        "least important groups one width step (4>3>2>0), then re-tune every block (codes +-1 inside "
                        "the group's width, scales). The result stays in the packed 4-bit format (gen --load works)")
    t.add_argument("--gbits_step", type=float, default=0.04, help="share of (4 bit x INT weights) saved by the round")
    t.add_argument("--gbits_stat_stride", type=int, default=4, help="every k-th call for the activity statistics")
    a = ap.parse_args()
    {"gen": cmd_gen, "capture": cmd_capture, "tune": cmd_tune, "coarse": cmd_coarse, "save_dec": cmd_save_dec}[a.cmd](a)


if __name__ == "__main__":
    main()
