"""RoundBit's default layer format (converter v2): group-wise INT-b weights with learned rounding (AutoRound idea)
and learned group scales, trained by RoundBit's block-wise loops. IntLinear for linears, IntConv2d for convs.
Defaults (DEFAULTS): INT4, groups of 64, 50 steps per block; LittleBit stays available as --fmt littlebit
(only worth it below ~2 bpw).

    W_q = clamp(round_ste(W / s + V), -2^(b-1), 2^(b-1)-1) * s      per group of `group` input columns
    V   rounding offset in [-0.5, 0.5], stored as V / v_mult (Adam steps v_mult x larger)
    s   = s0 * exp(lg): Adam trains the log gain (relative steps); s0 from a clipping search at init
freeze() fixes the integer codes (Q, int8); afterwards only s and bias change the output.
Same training interface as LittleBitLinear: latent_parameters(), scale_parameters(), freeze(), bits_used().
On YOLO26s (convs) INT4 g64 reached 97 % of fp mAP50-95 at 4.25 bpw vs 85 % for LittleBit at 4.55 bpw.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = {"fmt": "int", "bits": 4, "group": 64, "iters": 50}


def round_ste(x):
    return x + (x.round() - x).detach()


@torch.no_grad()
def _group_scale(Wg: torch.Tensor, L: int, clip_min=0.5, steps=10) -> torch.Tensor:
    """Per-row scale for one group of columns Wg (out, g): best clipping ratio by plain rounding MSE."""
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


@torch.no_grad()
def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, group: int, damp: float = 0.01, blocksize: int = 128):
    """GPTQ (Frantar et al. 2022) with static groups: quantize columns left to right, push each column's rounding
    error onto the not yet quantized columns through the inverse Hessian H = X^T X of the layer input.
    Returns the dequantized weight (out, in) and the group scales (out, n_groups)."""
    W = W.float().clone()
    H = H.float().clone()
    out, a = W.shape
    L = 2 ** (bits - 1)
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    mean_diag = H.diagonal().mean()
    H0 = H
    # Near-singular X^T X (seen on Pixal3D's tex 1024 DiT): retry with stronger damping (in float64 at the end)
    # instead of crashing. Layers whose first attempt works are unchanged.
    for d in (damp, 0.03, 0.1, 0.3, 1.0):
        H = H0.clone()
        H.diagonal().add_(d * mean_diag)
        try:
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
            break
        except torch.linalg.LinAlgError:
            try:
                Hd = H.double()
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True).float()
                break
            except torch.linalg.LinAlgError:
                continue
    else:
        raise torch.linalg.LinAlgError("gptq: H not positive-definite even with damp 1.0")
    if d != damp:
        print(f"[gptq] cholesky needed damp {d} (requested {damp})", flush=True)
    blocksize = max(blocksize, group) // group * group
    Q = torch.zeros_like(W)
    scales = torch.zeros(out, a // group, device=W.device)
    for i1 in range(0, a, blocksize):
        i2 = min(i1 + blocksize, a)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        s = None
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                s = _group_scale(W1[:, i:i + group], L)
                scales[:, (i1 + i) // group] = s[:, 0]
            w, d = W1[:, i], Hinv1[i, i]
            q = (w / s[:, 0]).round().clamp(-L, L - 1) * s[:, 0]
            Q[:, i1 + i] = q
            err = (w - q) / d
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scales


def want_rotation(rotate) -> bool:
    """rotate=None -> environment ROUNDBIT_ROTATE ('1' = on); otherwise the given bool"""
    import os as _os
    return _os.environ.get("ROUNDBIT_ROTATE", "0") == "1" if rotate is None else bool(rotate)


def rotation(g: int, device, seed: int = 0) -> torch.Tensor:
    """randomized Hadamard (g a power of two): orthogonal, spreads outlier weights over the group"""
    import math as _m
    H = torch.ones(1, 1)
    while H.shape[0] < g:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    sign = torch.randint(0, 2, (g,), generator=torch.Generator().manual_seed(seed)).float() * 2 - 1
    return (H / _m.sqrt(g) * sign[None]).to(device)


class IntLinear(nn.Module):
    def __init__(self, lin: nn.Linear, bits: int = 4, group: int = 64, v_mult: float = 10.0,
                 in_scale: torch.Tensor | None = None, clip_min: float = 0.5, clip_steps: int = 10,
                 H: torch.Tensor | None = None, rotate: bool | None = None):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.bits, self.L, self.v_mult = bits, 2 ** (bits - 1), v_mult
        a = lin.in_features
        g = a if group <= 0 or group >= a else group
        self.g, self.ng, self.pad = g, -(-a // g), (-a) % g
        W = lin.weight.detach().float()
        if H is not None and self.pad == 0:
            # GPTQ start: W0 = the GPTQ weights, so V = 0 reproduces them and training explores their neighbours
            Qd, sc = gptq(W, H.to(W.device), bits, g)
            Wg = self._groups(Qd)
            s0 = sc[..., None].clamp_min(1e-8)
        else:
            Wg = self._groups(W)
            if want_rotation(rotate) and g & (g - 1) == 0 and self.pad == 0:
                # rotation per group (QuIP/RaBitQ idea); activity weighting becomes the group mean
                self.register_buffer("R", rotation(g, W.device))
                Wg = Wg @ self.R
                if in_scale is not None:
                    in_scale = self._groups(in_scale.float().to(W.device)[None].pow(2)).mean(-1, keepdim=True) \
                        .sqrt().expand(1, self.ng, g).reshape(-1)
            s0 = self._clip_search(Wg, in_scale, clip_min, clip_steps)
        if "R" not in self._buffers:
            self.R = None
        self.register_buffer("W0", Wg.clone())
        self.register_buffer("s0", s0)
        self.register_buffer("Q", torch.zeros_like(Wg, dtype=torch.int8))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self.V = nn.Parameter(torch.zeros_like(Wg))
        self.lg = nn.Parameter(torch.zeros_like(self.s0))
        self.bias = nn.Parameter(lin.bias.detach().float().clone()) if lin.bias is not None else None

    def _groups(self, W):
        return F.pad(W, (0, self.pad)).reshape(W.shape[0], self.ng, self.g)

    @torch.no_grad()
    def _clip_search(self, Wg, in_scale, clip_min, steps):
        """Per group: the clipping ratio with the lowest (activity-weighted) rounding error."""
        d2 = torch.ones(self.in_features, device=Wg.device) if in_scale is None else \
            in_scale.to(Wg.device).float().pow(2)
        d2 = self._groups(d2[None]).expand_as(Wg)
        amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        best_s, best_e = None, None
        for r in torch.linspace(1.0, clip_min, steps).tolist():
            s = amax * r / (self.L - 1)
            e = ((Wg - (Wg / s).round().clamp(-self.L, self.L - 1) * s).pow(2) * d2).sum(-1, keepdim=True)
            if best_e is None:
                best_s, best_e = s, e
            else:
                better = e < best_e
                best_e, best_s = torch.where(better, e, best_e), torch.where(better, s, best_s)
        return best_s

    def scale(self):
        return self.s0 * self.lg.exp()

    def weight(self, dtype=torch.float32):
        s = self.scale()
        if bool(self.frozen):
            q = self.Q.float()
        else:
            q = round_ste(self.W0 / s + (self.V * self.v_mult).clamp(-0.5, 0.5)).clamp(-self.L, self.L - 1)
        Wg = q * s
        if getattr(self, "R", None) is not None:
            Wg = Wg @ self.R.t()
        return Wg.reshape(self.out_features, -1)[:, : self.in_features].to(dtype)

    def forward(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight(x.dtype), b)

    def latent_parameters(self):
        return [] if bool(self.frozen) else [self.V]

    def scale_parameters(self):
        return [self.lg] + ([self.bias] if self.bias is not None else [])

    @torch.no_grad()
    def freeze(self, scale_dtype=None):
        if not bool(self.frozen):
            v = (self.V * self.v_mult).clamp(-0.5, 0.5)
            self.Q.copy_((self.W0 / self.scale() + v).round().clamp(-self.L, self.L - 1).to(torch.int8))
            self.frozen.fill_(True)
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def bits_used(self) -> float:
        """Codes + one fp16 scale per group (bias not counted, as for LittleBitLinear)."""
        return self.out_features * (self.in_features * self.bits + 16 * self.ng)

    @torch.no_grad()
    def packed_state_dict(self) -> dict:
        """Deployable form: int8 codes (out, n_groups, group), fp16 scales (out, n_groups), fp16 bias."""
        self.freeze()
        sd = {"s": self.scale().reshape(self.out_features, -1).half().cpu()}
        if self.bits <= 4 and self.g % 2 == 0:  # two codes per byte: (q + 8) in the low / high nibble
            u = (self.Q.to(torch.int16) + 8).to(torch.uint8).cpu()
            sd["Qp"] = u[..., 0::2] | (u[..., 1::2] << 4)
        else:
            sd["Q"] = self.Q.cpu().clone()
        if self.bias is not None:
            sd["bias"] = self.bias.detach().half().cpu()
        if getattr(self, "R", None) is not None:
            sd["R"] = self.R.detach().float().cpu()
        return sd

    def packed_meta(self) -> dict:
        return {"fmt": "int", "in": self.in_features, "out": self.out_features, "bits": self.bits, "group": self.g,
                "bias": self.bias is not None}

    @classmethod
    def from_packed(cls, sd: dict, meta: dict, device="cuda", dtype=torch.float16) -> "IntLinear":
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.in_features, self.out_features = meta["in"], meta["out"]
        self.bits, self.L, self.v_mult = meta["bits"], 2 ** (meta["bits"] - 1), 1.0
        self.g = meta["group"]
        self.ng, self.pad = -(-meta["in"] // self.g), (-meta["in"]) % self.g
        if "Qp" in sd:
            p_ = sd["Qp"].to(device)
            Q = torch.stack([(p_ & 15), (p_ >> 4)], dim=-1).reshape(*p_.shape[:-1], -1).to(torch.int16) - 8
            Q = Q.to(torch.int8)
        else:
            Q = sd["Q"].to(device)
        self.register_buffer("W0", torch.zeros(0, device=device))
        self.register_buffer("s0", sd["s"].to(device, torch.float32)[..., None])
        self.register_buffer("Q", Q)
        self.register_buffer("frozen", torch.ones((), dtype=torch.bool, device=device))
        self.V = nn.Parameter(torch.zeros(0, device=device), requires_grad=False)
        self.lg = nn.Parameter(torch.zeros_like(self.s0), requires_grad=False)
        self.bias = nn.Parameter(sd["bias"].to(device, torch.float32), requires_grad=False) if meta["bias"] else None
        if "R" in sd:
            self.register_buffer("R", sd["R"].to(device, torch.float32))
        else:
            self.R = None
        return self

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, group={self.g}"


# ── convs ─────────────────────────────────────────────────────────────────────────────────────────────────────
from roundbit.conv import LittleBitConv2d  # noqa: E402


class IntConv2d(LittleBitConv2d):
    """Group-wise INT-b conv (kernel as the matrix (out, in*kh*kw)) with learned rounding. A LittleBitConv2d
    subclass so the YOLO tooling's isinstance checks, size report and train_params keep working. Tensors live in
    the child module ``lb`` (rb_run finds converted blocks by '.conv.lb' in a state dict).
    closed_form(): activity-weighted clipping search per group; apply_out_correction(): per-row gain into the
    scales; freeze_codes(): fixes the integer codes (called by conv.freeze_module_signs)."""

    def __init__(self, conv: nn.Conv2d, *, bits: int = 4, group: int = 64, v_mult: float = 10.0, amp: bool = False,
                 in_scale=None, clip_min: float = 0.5, rotate: bool | None = None, **_ignored):
        nn.Module.__init__(self)
        assert conv.groups == 1
        out_c, in_c, kh, kw_ = conv.weight.shape
        self.kshape = (out_c, in_c, kh, kw_)
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        self.v_mult, self.amp, self.clip_min = v_mult, amp, clip_min
        W = conv.weight.detach().reshape(out_c, -1).float()
        a = W.shape[1]
        self.bits_, self.L = bits, 2 ** (bits - 1)
        g = a if group <= 0 or group >= a else group
        self.g, self.ng, self.pad = g, -(-a // g), (-a) % g
        a_in = None if in_scale is None else in_scale.float().repeat_interleave(kh * kw_)
        self._orig = (W.clone(), None if conv.bias is None else conv.bias.detach().float().clone(), a_in)
        Wg = self._groups(W)
        self.lb = nn.Module()
        if want_rotation(rotate) and g & (g - 1) == 0 and self.pad == 0:
            # rotation per group (QuIP/RaBitQ idea): codes and scales live in the rotated basis, weight() rotates back
            self.lb.register_buffer("R", rotation(g, W.device))
            Wg = Wg @ self.lb.R
        else:
            self.lb.R = None
        self.lb.register_buffer("W0", Wg.clone())
        self.lb.register_buffer("s0", (Wg.abs().amax(-1, keepdim=True) / (self.L - 1)).clamp_min(1e-8))
        self.lb.register_buffer("Q", torch.zeros_like(Wg, dtype=torch.int8))
        self.lb.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self.lb.V = nn.Parameter(torch.zeros_like(Wg))
        self.lb.lg = nn.Parameter(torch.zeros_like(self.lb.s0))
        self.bias = nn.Parameter(conv.bias.detach().float().clone()) if conv.bias is not None else None

    def _groups(self, W):
        return F.pad(W, (0, self.pad)).reshape(W.shape[0], self.ng, self.g)

    def scale(self):
        return self.lb.s0 * self.lb.lg.exp()

    def weight(self, dtype=torch.float32) -> torch.Tensor:
        s = self.scale()
        if bool(self.lb.frozen):
            q = self.lb.Q.float()
        else:
            q = round_ste(self.lb.W0 / s + (self.lb.V * self.v_mult).clamp(-0.5, 0.5)).clamp(-self.L, self.L - 1)
        Wg = q * s
        if getattr(self.lb, "R", None) is not None:
            Wg = Wg @ self.lb.R.t()
        Wq = Wg.reshape(self.kshape[0], -1)[:, : self.kshape[1] * self.kshape[2] * self.kshape[3]]
        return Wq.reshape(self.kshape).to(dtype)

    def forward(self, x):
        dt = torch.bfloat16 if self.amp and torch.is_grad_enabled() else x.dtype  # bf16 only in training steps
        b = self.bias.to(dt) if self.bias is not None else None
        return F.conv2d(x.to(dt), self.weight(dt), b, self.stride, self.padding, self.dilation).to(x.dtype)

    def parts(self):
        return []

    @torch.no_grad()
    def closed_form(self, rounds: int = 2, sign_sweeps: int = 1) -> dict:
        W, _, a = self._orig
        d2 = torch.ones(W.shape[1], device=W.device) if a is None else a.to(W.device).float().pow(2)
        d2 = self._groups(d2[None]).expand_as(self.lb.W0)
        if getattr(self.lb, "R", None) is not None:  # rotation mixes the columns of a group: use the group mean
            d2 = d2.mean(-1, keepdim=True).expand_as(self.lb.W0)
        Wg, amax = self.lb.W0, self.lb.W0.abs().amax(-1, keepdim=True)

        def err(s):
            return ((Wg - (Wg / s).round().clamp(-self.L, self.L - 1) * s).pow(2) * d2).sum(-1, keepdim=True)

        s_init = self.scale()
        best_s, best_e = s_init.clone(), err(s_init)
        e0 = best_e.sum().item()
        for r in torch.linspace(1.0, self.clip_min, 2 * max(2, 4 * rounds + 1)).tolist():
            s = (amax * r / (self.L - 1)).clamp_min(1e-8)
            e = err(s)
            better = e < best_e
            best_e, best_s = torch.where(better, e, best_e), torch.where(better, s, best_s)
        self.lb.s0.copy_(best_s)
        self.lb.lg.zero_()
        tot = (Wg.pow(2) * d2).sum().item()
        e1 = (best_e.sum().item() / tot) ** 0.5
        return {"err_init": (e0 / tot) ** 0.5, "err_als": e1, "err_final": e1}

    @torch.no_grad()
    def apply_out_correction(self, acc: dict, max_gain: float = 2.0):
        n = acc["n"]
        my, mt = acc["y"] / n, acc["t"] / n
        var = acc["yy"] / n - my * my
        cov = acc["yt"] / n - my * mt
        g = torch.where(var > 1e-12, cov / var.clamp_min(1e-12), torch.ones_like(var)).clamp(1 / max_gain, max_gain)
        beta = mt - g * my
        self.lb.s0.mul_(g.reshape(-1, 1, 1).to(self.lb.s0.dtype))
        if self.bias is not None:
            self.bias.copy_((g * self.bias.float() + beta).to(self.bias.dtype))
        return g, beta

    @torch.no_grad()
    def freeze_codes(self):
        if bool(self.lb.frozen):
            return
        v = (self.lb.V * self.v_mult).clamp(-0.5, 0.5)
        self.lb.Q.copy_((self.lb.W0 / self.scale() + v).round().clamp(-self.L, self.L - 1).to(torch.int8))
        self.lb.frozen.fill_(True)
        self.lb.V.requires_grad_(False)

    def latent_parameters(self):
        return [] if bool(self.lb.frozen) else [self.lb.V]

    def scale_parameters(self):
        return [self.lb.lg] + ([self.bias] if self.bias is not None else [])

    def bits(self) -> float:
        return self.n_weights() * self.bits_ + 16 * self.kshape[0] * self.ng

    def n_weights(self) -> int:
        return self.kshape[0] * self.kshape[1] * self.kshape[2] * self.kshape[3]


def bits_report(model: nn.Module, keep_bpw: float = 0.0) -> dict:
    """roundbit.layer.model_bits_report for models with IntLinear and/or LittleBitLinear layers."""
    from roundbit.layer import LittleBitLinear

    layers = [m for m in model.modules() if isinstance(m, (IntLinear, LittleBitLinear))]
    bits = sum(m.bits_used() for m in layers)
    w = sum(m.in_features * m.out_features for m in layers)
    kept = sum(m.weight.numel() for n, m in model.named_modules() if type(m) is nn.Linear and "lm_head" not in n)
    rep = {"layers": len(layers), "weights": w, "avg_bpw": bits / w if w else math.nan, "kept_weights": kept}
    if kept and keep_bpw:
        rep["mixed_bpw"] = (bits + kept * keep_bpw) / (w + kept)
    return rep


# ── deployment: real 4-bit storage + Triton kernels (roundbit/int4_kernels.py) ─────────────────────────────────
class Int4Linear(nn.Module):
    """Inference-only INT4 linear holding only the packed codes (uint8, two per byte) and fp16 group scales:
    0.5 byte + 16/group bits per weight in memory. forward = int4_kernels.int4_linear (fused kernel for small M,
    dequantize + vendor GEMM for large M); on CPU a PyTorch dequantization."""

    def __init__(self, qp: torch.Tensor, s: torch.Tensor, group: int, bias: torch.Tensor | None, in_features: int,
                 R: torch.Tensor | None = None):
        super().__init__()
        self.in_features, self.out_features, self.g, self.bits = in_features, qp.shape[0], group, 4
        self.register_buffer("qp", qp.reshape(qp.shape[0], -1).contiguous())
        self.register_buffer("s", s.reshape(qp.shape[0], -1).half().contiguous())
        self.register_buffer("bias", None if bias is None else bias.half().contiguous())
        # codes stored in a rotated basis (W_g = Q_g R^T per group): W x = sum_g Q_g (x_g R)  -> rotate x, no weight change
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

    def bits_used(self) -> float:
        return self.out_features * (self.in_features * 4 + 16 * self.s.shape[1])

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int4, group={self.g}"


def load_packed(sd: dict, meta: dict, device="cuda"):
    """Layer for a packed IntLinear state: Int4Linear (real 4-bit + kernels) when possible, else IntLinear."""
    import os

    if ("Qp" in sd and meta.get("bits") == 4 and meta["in"] % meta["group"] == 0 and meta["group"] % 16 == 0
            and os.environ.get("ROUNDBIT_INT4_KERNEL", "1") != "0"):
        return Int4Linear(sd["Qp"].to(device), sd["s"].to(device), meta["group"],
                          sd["bias"].to(device) if meta.get("bias") else None, meta["in"],
                          sd["R"].to(device) if "R" in sd else None)
    return IntLinear.from_packed(sd, meta, device)
