"""LittleBit factorized binary convolutions (RoundBit, generic; model-specific selection lives in yolo/).

A kernel W (out, in, kh, kw) is treated as the matrix (out, in*kh*kw) and factorized with LittleBitLinear
(rank from eff_bit, SVD + optional Joint-ITQ, residual pair, optional activity-aware init):
    W ~= sum_pairs diag(u1) sign(U) diag(u2 v1) sign(V) diag(v2)
Training rebuilds W from signs (surrogate gradient) + scales each forward and runs conv2d. Deployed, this is a
(kh x kw, in -> s) binary conv followed by a 1x1 (s -> out) binary conv plus per-channel scales.
Groups != 1 (depthwise) are not supported.

Findings from YOLO26s (yolo/README.md): at equal bits, greedy stages do not beat one main + residual pair, but once
the rank is capped at the matrix size (~3 bpw for 3x3 convs) stages are the way to spend more bits:
3 stages (4.55 bpw) reached 86 % of fp mAP50-95 vs 71 % for one pair at 3.03 bpw.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from roundbit.layer import LittleBitLinear, binarize


class LittleBitConv2d(nn.Module):
    """stages == 1: one RoundBit LittleBitLinear (main + optional residual pair), stored as ``lb``.
    stages > 1: K greedy stages ``lbs[k]`` (each with its own residual pair if residual), stage k fit to W minus the binarised
    reconstruction of stages < k, each with eff_bit / K bits and its own scales (plan phase 1.1)."""

    def __init__(self, conv: nn.Conv2d, *, eff_bit: float, residual: bool, use_itq: bool, quant_func="SmoothSign",
                 calc_device=None, in_scale: torch.Tensor | None = None, stages: int = 1):
        super().__init__()
        assert conv.groups == 1
        out_c, in_c, kh, kw_ = conv.weight.shape
        self.kshape = (out_c, in_c, kh, kw_)
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        dev = conv.weight.device
        target = conv.weight.detach().reshape(out_c, -1).float()
        # per-input-channel rms -> per flattened column (channel-major, like the weight reshape)
        a_in = None if in_scale is None else in_scale.float().repeat_interleave(kh * kw_)
        fit = dict(use_itq=use_itq, quant_func=quant_func, calc_device=calc_device, in_scale=a_in)
        # plain attributes (not buffers): the fp kernel is only needed while calibrating, see drop_orig()
        self._orig = (target.clone(), None if conv.bias is None else conv.bias.detach().float().clone(), a_in)
        if stages == 1:
            lin = self._lin(target, conv.bias, dev)
            self.lb = LittleBitLinear.from_linear(lin, eff_bit=eff_bit, residual=residual, **fit)
            return
        self.lbs = nn.ModuleList()
        for k in range(stages):
            lin = self._lin(target, conv.bias if k == 0 else None, dev)
            lb = LittleBitLinear.from_linear(lin, eff_bit=eff_bit / stages, residual=residual, **fit)
            self.lbs.append(lb)
            target = target - lb.dense_weight().to(target.device)

    @staticmethod
    def _lin(W, bias, dev):
        lin = nn.Linear(W.shape[1], W.shape[0], bias=bias is not None, device=dev)
        with torch.no_grad():
            lin.weight.copy_(W)
            if bias is not None:
                lin.bias.copy_(bias.float())
        return lin

    def parts(self):
        return [self.lb] if hasattr(self, "lb") else list(self.lbs)

    def weight(self, dtype=torch.float32) -> torch.Tensor:
        W = 0
        for lb in self.parts():
            for sfx in lb.pairs:
                g = lambda n: getattr(lb, n + sfx)  # noqa: E731
                U = lb._signs(g("U"), dtype)
                V = lb._signs(g("V"), dtype)
                W = W + (U * (g("u1").t() @ g("u2")).to(dtype)) @ (V * (g("v1").t() @ g("v2")).to(dtype))
        return W.reshape(self.kshape)

    def forward(self, x):
        b0 = self.parts()[0].bias
        b = b0.to(x.dtype) if b0 is not None else None
        return F.conv2d(x, self.weight(x.dtype), b, self.stride, self.padding, self.dilation)

    # ── training-free fitting (roundbit/closed_form.py) ──
    @torch.no_grad()
    def closed_form(self, rounds: int = 2, sign_sweeps: int = 1) -> dict:
        """Exact least-squares scales + sign flips against the fp kernel, activity-weighted if in_scale was given."""
        from roundbit.closed_form import closed_form_fit

        W, _, a = self._orig
        return closed_form_fit(self.parts(), W, a, rounds, sign_sweeps)

    @torch.no_grad()
    def out_stats(self, x: torch.Tensor, acc: dict | None = None) -> dict:
        """Accumulate per-output-channel moments of this layer (y) and of the fp conv (t) on the same input."""
        W, b, _ = self._orig
        x = x.float()
        y = self.forward(x)
        t = F.conv2d(x, W.reshape(self.kshape), b, self.stride, self.padding, self.dilation)
        red = (0, 2, 3)
        cur = {"n": y.numel() // y.shape[1], "y": y.sum(red), "t": t.sum(red), "yy": (y * y).sum(red),
               "yt": (y * t).sum(red)}
        return cur if acc is None else {k: acc[k] + cur[k] for k in cur}

    @staticmethod
    @torch.no_grad()
    def moments(y: torch.Tensor, t: torch.Tensor, acc: dict | None = None) -> dict:
        """Per-channel moments of an output y and a reference t (e.g. the fp network's activation here)."""
        y, t = y.float(), t.float()
        red = (0, 2, 3)
        cur = {"n": y.numel() // y.shape[1], "y": y.sum(red), "t": t.sum(red), "yy": (y * y).sum(red),
               "yt": (y * t).sum(red)}
        return cur if acc is None else {k: acc[k] + cur[k] for k in cur}

    @torch.no_grad()
    def apply_out_correction(self, acc: dict, max_gain: float = 2.0):
        """y -> g*y + beta per output channel (least squares to the fp conv output); g folds into u1, beta into bias."""
        n = acc["n"]
        my, mt = acc["y"] / n, acc["t"] / n
        var = (acc["yy"] / n - my * my)
        cov = acc["yt"] / n - my * mt
        g = torch.where(var > 1e-12, cov / var.clamp_min(1e-12), torch.ones_like(var)).clamp(1 / max_gain, max_gain)
        beta = mt - g * my
        for lb in self.parts():
            for sfx in lb.pairs:
                u1 = getattr(lb, "u1" + sfx)
                u1.mul_(g.reshape(u1.shape).to(u1.dtype))
        b0 = self.parts()[0].bias
        if b0 is not None:
            b0.copy_((g * b0.float() + beta).to(b0.dtype))
        return g, beta

    def drop_orig(self):
        self._orig = None

    def latent_parameters(self):
        return [p for lb in self.parts() for p in lb.latent_parameters()]

    def scale_parameters(self):
        return [p for lb in self.parts() for p in lb.scale_parameters() + ([lb.bias] if lb.bias is not None else [])]

    def bits(self) -> float:
        return sum(lb.bits_used() for lb in self.parts())

    def n_weights(self) -> int:
        lb = self.parts()[0]
        return lb.in_features * lb.out_features


@torch.no_grad()
def freeze_module_signs(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "freeze_codes"):  # roundbit.int_layer.IntConv2d: fix the integer codes
            m.freeze_codes()
        elif isinstance(m, LittleBitConv2d):
            for p in m.latent_parameters():
                p.data = binarize(p.data)
                p.requires_grad_(False)


def lb_convs(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LittleBitConv2d)]


def train_params(model):
    lat, sc = [], []
    for _, m in lb_convs(model):
        lat += m.latent_parameters()
        sc += m.scale_parameters()
    return lat, sc


@torch.no_grad()
def freeze_signs(model):
    """Replace latents by their signs (what deployment stores); forward stays the same."""
    freeze_module_signs(model)
