"""Inference-only layers for the packed RoundBit files (INT4 flow DiTs, INT8 latent decoders).

DiT files (<model>.pt): {"layers": {"<block>.<linear>": (state, meta)}, "info": {...}}
    state: Qp uint8 (out, n_groups, group/2) two codes per byte (low nibble = even column, value = q + 8),
           s fp16 (out, n_groups), optional bias fp16, optional R fp32 (per-group rotation, codes in rotated basis)
    meta:  {"in", "out", "bits", "group", "bias", "sparse"}
Decoder files (<decoder>.int8.pt): {"layers": {"<module name>": {"wq", "ws", "wshape", "wdtype", "wfan", "wbits"}}}
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int4Linear(nn.Module):
    """INT4 linear holding only the packed codes (uint8, two per byte) and fp16 group scales.
    forward = int4_kernels.int4_linear on the GPU (fused kernel for small M, dequantize + vendor GEMM for large M);
    elsewhere a PyTorch dequantization."""

    def __init__(self, qp: torch.Tensor, s: torch.Tensor, group: int, bias: torch.Tensor | None, in_features: int,
                 R: torch.Tensor | None = None):
        super().__init__()
        self.in_features, self.out_features, self.g = in_features, qp.shape[0], group
        self.register_buffer("qp", qp.reshape(qp.shape[0], -1).contiguous())
        self.register_buffer("s", s.reshape(qp.shape[0], -1).half().contiguous())
        self.register_buffer("bias", None if bias is None else bias.half().contiguous())
        # codes stored in a rotated basis (W_g = Q_g R^T per group): W x = sum_g Q_g (x_g R) -> rotate x
        self.register_buffer("R", None if R is None else R.float().contiguous())

    def _rot_in(self, x):
        if self.R is None:
            return x
        sh = x.shape
        return (x.reshape(*sh[:-1], -1, self.g).float() @ self.R).reshape(sh).to(x.dtype)

    def dense_weight(self, dtype=torch.float16) -> torch.Tensor:
        u = self.qp
        q = torch.stack([(u & 15), (u >> 4)], -1).reshape(u.shape[0], -1).to(dtype) - 8
        return (q * self.s.to(dtype).repeat_interleave(self.g, dim=1))[:, : self.in_features]

    def forward(self, x):
        x = self._rot_in(x)
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and self.in_features % self.g == 0:
            from roundbit.int4_kernels import int4_linear

            return int4_linear(x, self.qp, self.s, self.g, self.bias)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dense_weight(x.dtype), b)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int4, group={self.g}"


class PackedIntLinear(nn.Module):
    """Fallback for packed layers the INT4 kernel path cannot take (other bit widths, padded groups,
    ROUNDBIT_INT4_KERNEL=0): keeps int8 codes + scales and builds the dense weight per call."""

    def __init__(self, sd: dict, meta: dict, device="cuda"):
        super().__init__()
        self.in_features, self.out_features, self.g = meta["in"], meta["out"], meta["group"]
        if "Qp" in sd:
            p = sd["Qp"].to(device)
            Q = (torch.stack([(p & 15), (p >> 4)], -1).reshape(*p.shape[:-1], -1).to(torch.int16) - 8).to(torch.int8)
        else:
            Q = sd["Q"].to(device)
        self.register_buffer("Q", Q.reshape(self.out_features, -1, self.g))
        self.register_buffer("s", sd["s"].to(device, torch.float32).reshape(self.out_features, -1, 1))
        self.register_buffer("bias", sd["bias"].to(device, torch.float32) if meta.get("bias") else None)
        self.register_buffer("R", sd["R"].to(device, torch.float32) if "R" in sd else None)

    def forward(self, x):
        Wg = self.Q.float() * self.s
        if self.R is not None:
            Wg = Wg @ self.R.t()
        W = Wg.reshape(self.out_features, -1)[:, : self.in_features].to(x.dtype)
        return F.linear(x, W, None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, group={self.g}"


def load_packed(sd: dict, meta: dict, device="cuda") -> nn.Module:
    """Layer for one packed DiT linear: Int4Linear (real 4-bit + Triton kernels) when possible."""
    if ("Qp" in sd and meta.get("bits") == 4 and meta["in"] % meta["group"] == 0 and meta["group"] % 16 == 0
            and os.environ.get("ROUNDBIT_INT4_KERNEL", "1") != "0"):
        return Int4Linear(sd["Qp"].to(device), sd["s"].to(device), meta["group"],
                          sd["bias"].to(device) if meta.get("bias") else None, meta["in"],
                          sd["R"].to(device) if "R" in sd else None)
    return PackedIntLinear(sd, meta, device)


class SparseInt(nn.Module):
    """Stand-in for trellis2 SparseLinear (input may be a VarLenTensor/SparseTensor or a plain tensor)."""

    def __init__(self, layer: nn.Module):
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


def apply_dit_file(model: nn.Module, path, device="cuda") -> int:
    """Swap the packed INT linears of a flow DiT file into model.blocks (works on the meta device: the replaced
    bf16 weights are never allocated). Returns the number of layers."""
    state = torch.load(path, map_location="cpu")
    info = state.get("info", {})
    if "blocks_done" in info and info["blocks_done"] != info.get("n_blocks"):
        raise RuntimeError(f"{path}: incomplete file ({info['blocks_done']} of {info.get('n_blocks')} blocks)")
    for name, (sd, meta) in state["layers"].items():
        layer = load_packed(sd, meta, device)
        _set(model.blocks, name, SparseInt(layer) if meta["sparse"] else layer)
    return len(state["layers"])


# ── latent decoders: weight-only INT8 / INT4, dequantized per layer call ─────────────────────────────────────
class _QWeight:
    """Mixin placed in front of the layer's class: `weight` becomes the dequantized tensor (built per access)."""

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


_classes: dict = {}


def apply_decoder_file(model: nn.Module, path, device="cuda") -> int:
    """Swap the saved weight-only INT layers into a latent decoder (works on the meta device too, so the fp16
    weights are never allocated). Returns the number of layers."""
    state = torch.load(path, map_location="cpu")
    for name, st in state["layers"].items():
        m = model.get_submodule(name)
        m._parameters.pop("weight", None)
        m.register_buffer("wq", st["wq"].to(device))
        m.register_buffer("ws", st["ws"].to(device))
        m.wshape, m.wdtype, m.wfan, m.wbits = torch.Size(st["wshape"]), getattr(torch, st["wdtype"]), st["wfan"], st["wbits"]
        cls = type(m)
        if not cls.__name__.startswith("Q"):
            if cls not in _classes:
                _classes[cls] = type("Q" + cls.__name__, (_QWeight, cls), {})
            m.__class__ = _classes[cls]
    return len(state["layers"])
