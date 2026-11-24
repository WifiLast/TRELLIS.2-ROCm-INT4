"""LittleBit linear layer as a standalone module.

Math, rank selection and initialisation follow LittleBit (quantization/modules/littlebit.py):

    W ~= diag(u1) sign(U) diag(u2) . diag(v1) sign(V) diag(v2)       (+ the same for a residual pair)
    y  = ((((x * v2) @ sign(V)^T) * (v1 * u2)) @ sign(U)^T) * u1 + bias

Differences from the original: a normal nn.Module built from an nn.Linear (no __class__ patching),
SVD/ITQ run on the weight's own device, latents can be kept in fp32 while the block runs in bf16,
and ``freeze()`` turns the latents into int8 signs once tuning is done.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from roundbit.functions import get_act_quant, get_quant_func
from roundbit.packing import pack_signs

PAIRS = ("", "_R")  # main and residual factor sets
SIGN_NAMES = ("U", "V")
SCALE_NAMES = ("u1", "u2", "v1", "v2")


# ── rank selection (identical formulas to LittleBit) ───────────────────────────
def estimate_split_dim(a: int, b: int, eff_bit: float | None, residual: bool) -> float | None:
    if eff_bit is None or a * b == 0:
        return None
    base = a + b + 16
    if residual:
        return (a * b * eff_bit - 32 * (a + b)) / (2 * base)
    return (a * b * eff_bit - 16 * (a + b)) / base


def finalize_split_dim(split_float: float | None, split_default: int, min_split_dim: int) -> int:
    cand = split_float if split_float is not None else split_default
    cand = int(cand) if cand is not None else 0
    cand = (cand // 8) * 8
    if cand == 0:
        cand = min_split_dim
    return max(cand, min_split_dim)


def compute_eff_bits(a: int, b: int, s: int, residual: bool) -> float:
    if a * b == 0:
        return float("inf")
    if residual:
        return (s * 2 * (a + b + 16) + 32 * (a + b)) / (a * b)
    return (s * (a + b + 16) + 16 * (a + b)) / (a * b)


def max_eff_bit(a: int, b: int, residual: bool) -> float:
    """Highest bits/weight a LittleBit layer can spend: the rank is capped at min(a, b) (rounded to 8).
    Narrow layers saturate early, e.g. 384-wide ViT-S linears at ~2.78 bpw with the residual pair, so a
    larger eff_bit buys nothing (found on DA3-SMALL, see da3/README.md)."""
    full = min(a, b)
    return compute_eff_bits(a, b, (full // 8) * 8 or full, residual)


def choose_split_dim(
    in_features: int,
    out_features: int,
    eff_bit: float | None,
    residual: bool,
    split_dim: int = 1024,
    ratio_factor: float = 1.0,
    min_split_dim: int = 8,
) -> int:
    est = estimate_split_dim(in_features, out_features, eff_bit, residual)
    if est:
        est *= ratio_factor
    return finalize_split_dim(est, split_dim, min_split_dim)


_RANK_CAP_WARNED: set = set()


def _warn_rank_cap(a: int, b: int, eff_bit, residual: bool) -> None:
    """Warn once per shape when eff_bit exceeds what the layer can hold (the rank is clamped silently otherwise)."""
    key = (a, b, eff_bit, residual)
    if key not in _RANK_CAP_WARNED:
        _RANK_CAP_WARNED.add(key)
        import warnings

        warnings.warn(f"LittleBit {b}x{a}: eff_bit {eff_bit} exceeds the full-rank maximum "
                      f"{max_eff_bit(a, b, residual):.3f} bpw; rank clamped to {min(a, b)}", stacklevel=3)


# ── initialisation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def polar_factor(M: torch.Tensor) -> torch.Tensor:
    """Orthogonal polar factor U_p @ Vt_p of a square M.

    On GPU this uses an fp64 eigendecomposition of M^T M (0.07 s for 1000x1000 on the 7900 XTX vs 0.28 s
    for torch.linalg.svd; fp32 eigh is too ill-conditioned). On CPU the SVD is faster.
    """
    if M.device.type == "cpu":
        Up, _, Vtp = torch.linalg.svd(M, full_matrices=False)
        return Up @ Vtp
    Md = M.double()
    w, Q = torch.linalg.eigh(Md.t() @ Md)
    return (Md @ (Q * w.clamp_min(1e-30).rsqrt()) @ Q.t()).to(M.dtype)


@torch.no_grad()
def itq_rotation(X: torch.Tensor, n_iter: int = 50, seed: int = 0) -> torch.Tensor:
    """Orthogonal R minimising ||sign(XR) - XR||_F (Joint-ITQ, LittleBit-2).

    Same iteration as LittleBit: B = sign(XR); M = B^T X = U S Vt; R = Vt^T U^T = polar(M)^T.
    """
    dim = X.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.linalg.qr(torch.randn(dim, dim, generator=gen))[0].to(X.device, torch.float32)
    Xf = X.float()
    for _ in range(n_iter):
        B = torch.sign(Xf @ R)
        R = polar_factor(B.t() @ Xf).t()
    return R


@torch.no_grad()
def rank_one(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-negative rank-one factors (row (1, m), row (1, n)) of a non-negative matrix X (m, n)."""
    U, S, V = torch.svd_lowrank(X, q=1)
    s0 = torch.sqrt(S[0])
    left = (U[:, :1] * s0).t().contiguous()
    right = (s0 * V[:, :1].t()).contiguous()
    if left.sum() < 0:  # the top singular pair of a non-negative matrix is single-signed
        left, right = -left, -right
    return left, right


@torch.no_grad()
def decompose(W: torch.Tensor, split_dim: int, use_itq: bool = False, itq_n_iter: int = 50, niter: int = 4):
    """W (out, in) -> U (out, s), V (s, in), u1 (1,out), u2 (1,s), v1 (1,s), v2 (1,in), all fp32."""
    W = W.float()
    q = min(split_dim, *W.shape)
    try:
        U_t, S_t, V_t = torch.svd_lowrank(W, q=q, niter=niter)
    except torch._C._LinAlgError:  # rocSOLVER may not converge when q reaches the matrix size; LAPACK copes
        U_t, S_t, V_t = (t.to(W.device) for t in torch.svd_lowrank(W.cpu(), q=q, niter=niter))
    sq = torch.sqrt(S_t)
    U = U_t * sq
    V = sq[:, None] * V_t.t()
    if q < split_dim:  # rank larger than the matrix allows: pad with tiny noise so signs stay defined
        U = torch.cat([U, 1e-6 * torch.randn(U.shape[0], split_dim - q, device=W.device)], dim=1)
        V = torch.cat([V, 1e-6 * torch.randn(split_dim - q, V.shape[1], device=W.device)], dim=0)
    if use_itq:
        R = itq_rotation(torch.cat([U, V.t()], dim=0), n_iter=itq_n_iter)
        U = U @ R
        V = R.t() @ V
    v1, v2 = rank_one(V.abs())
    u1, u2 = rank_one(U.abs())
    return U.contiguous(), V.contiguous(), u1, u2, v1, v2


def binarize(x: torch.Tensor) -> torch.Tensor:
    y = x.sign()
    y[y == 0] = 1
    return y


# ── the layer ──────────────────────────────────────────────────────────────────
class LittleBitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        split_dim: int,
        *,
        bias: bool = True,
        residual: bool = False,
        quant_func: str = "SmoothSign",
        eff_bit_target: float | None = None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_dim = split_dim
        self.residual = residual
        self.quant_func_name = quant_func
        self.quant_func = get_quant_func(quant_func)
        self.frozen = False
        self.act_quant = "none"

        def p(*shape):
            return nn.Parameter(torch.empty(*shape, device=device, dtype=dtype))

        for sfx in self.pairs:
            setattr(self, "U" + sfx, p(out_features, split_dim))
            setattr(self, "V" + sfx, p(split_dim, in_features))
            setattr(self, "u1" + sfx, p(1, out_features))
            setattr(self, "u2" + sfx, p(1, split_dim))
            setattr(self, "v1" + sfx, p(1, split_dim))
            setattr(self, "v2" + sfx, p(1, in_features))
        if bias:
            self.bias = p(out_features)
        else:
            self.register_parameter("bias", None)

        eff_actual = compute_eff_bits(in_features, out_features, split_dim, residual)
        buf_dev = None if device == "meta" else device
        self.register_buffer(
            "_eff_bit_target", torch.tensor(-1.0 if eff_bit_target is None else float(eff_bit_target), device=buf_dev)
        )
        self.register_buffer("_split_dim_final", torch.tensor(split_dim, device=buf_dev))
        self.register_buffer("_eff_bit_actual", torch.tensor(eff_actual, device=buf_dev))

    @property
    def pairs(self):
        return PAIRS if self.residual else PAIRS[:1]

    # ── construction ──
    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        eff_bit: float | None = 1.0,
        residual: bool = False,
        split_dim: int = 1024,
        ratio_factor: float = 1.0,
        min_split_dim: int = 8,
        use_itq: bool = False,
        itq_n_iter: int = 50,
        quant_func: str = "SmoothSign",
        latent_dtype: torch.dtype = torch.float32,
        calc_device=None,
        in_scale: torch.Tensor | None = None,
    ) -> "LittleBitLinear":
        a, b = linear.in_features, linear.out_features
        s = choose_split_dim(a, b, eff_bit, residual, split_dim, ratio_factor, min_split_dim)
        full = min(a, b)
        if s > full:
            # A rank above the matrix size adds no information; decompose() would pad it with noise columns whose
            # signs still count at full weight (worse fit) and the GPU ITQ step turns them into NaN/inf.
            s = (full // 8) * 8 or full
            _warn_rank_cap(a, b, eff_bit, residual)
        dev = linear.weight.device
        calc = torch.device(calc_device) if calc_device is not None else dev
        layer = cls(
            a,
            b,
            s,
            bias=linear.bias is not None,
            residual=residual,
            quant_func=quant_func,
            eff_bit_target=eff_bit,
            device=dev,
            dtype=latent_dtype,
        )
        W = linear.weight.detach().to(calc, torch.float32)
        target = W
        for sfx in layer.pairs:
            if in_scale is None:
                U, V, u1, u2, v1, v2 = decompose(target, s, use_itq, itq_n_iter)
            else:
                # Activity-aware init: factorise W * diag(a) so the low-rank fit spends its accuracy on the input
                # channels the layer is actually driven by (a = rms input per channel), then undo the scaling in
                # V and in the per-input scale v2. Signs of V are unchanged (a > 0).
                a_in = in_scale.to(calc, torch.float32).clamp_min(1e-4)
                a_in = a_in / a_in.mean()
                U, V, u1, u2, v1, v2 = decompose(target * a_in, s, use_itq, itq_n_iter)
                V = V / a_in
                v2 = v2 / a_in
            for name, t in zip(("U", "V", "u1", "u2", "v1", "v2"), (U, V, u1, u2, v1, v2)):
                getattr(layer, name + sfx).copy_(t.to(dev, latent_dtype))
            if sfx == "":  # residual target = what the binarised main pair misses
                approx = (binarize(U) * (u1.t() @ u2)) @ (binarize(V) * (v1.t() @ v2))
                target = W - approx
                del approx
        if linear.bias is not None:
            layer.bias.copy_(linear.bias.detach().to(dev, latent_dtype))
        return layer

    # ── forward ──
    def _signs(self, t: torch.Tensor, dtype) -> torch.Tensor:
        if self.frozen:
            return t.to(dtype)
        return self.quant_func(t.to(dtype))

    def _pair_forward(self, x, sfx):
        dt = x.dtype
        Vq = self._signs(getattr(self, "V" + sfx), dt)
        Uq = self._signs(getattr(self, "U" + sfx), dt)
        v2 = getattr(self, "v2" + sfx).to(dt)
        mid = (getattr(self, "v1" + sfx) * getattr(self, "u2" + sfx)).to(dt)
        u1 = getattr(self, "u1" + sfx).to(dt)
        aq = get_act_quant(self.act_quant)
        if aq is None:
            return ((((x * v2) @ Vq.t()) * mid) @ Uq.t()) * u1
        # the GGUF runner folds mid into V's rows, so both matmul inputs are x*v2 and (x*v2 @ V^T)*mid
        return (aq(aq(x * v2) @ Vq.t() * mid) @ Uq.t()) * u1

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = self._pair_forward(x2, "")
        if self.residual:
            y = y + self._pair_forward(x2, "_R")
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shape[:-1], self.out_features)

    # ── helpers ──
    def latent_parameters(self):
        return [getattr(self, n + sfx) for sfx in self.pairs for n in SIGN_NAMES]

    def scale_parameters(self):
        return [getattr(self, n + sfx) for sfx in self.pairs for n in SCALE_NAMES]

    @torch.no_grad()
    def dense_weight(self) -> torch.Tensor:
        """Reconstructed dense weight (out, in) in fp32, for tests and diagnostics."""
        W = 0
        for sfx in self.pairs:
            g = lambda n: getattr(self, n + sfx).float()  # noqa: E731
            U = g("U") if self.frozen else binarize(g("U"))
            V = g("V") if self.frozen else binarize(g("V"))
            W = W + (U * (g("u1").t() @ g("u2"))) @ (V * (g("v1").t() @ g("v2")))
        return W

    @torch.no_grad()
    def freeze(self, scale_dtype: torch.dtype | None = None):
        """Replace latents by int8 signs (no grad) and optionally cast scales/bias."""
        if self.frozen:
            return self
        for sfx in self.pairs:
            for n in SIGN_NAMES:
                old = getattr(self, n + sfx)
                setattr(self, n + sfx, nn.Parameter(binarize(old.data).to(torch.int8), requires_grad=False))
        for p in self.scale_parameters() + ([self.bias] if self.bias is not None else []):
            if scale_dtype is not None:
                p.data = p.data.to(scale_dtype)
            p.requires_grad_(False)
        self.frozen = True
        return self

    @torch.no_grad()
    def packed_state_dict(self, prefix: str = "", scale_dtype=torch.bfloat16) -> dict:
        """State dict in LittleBit's packed checkpoint layout."""
        out = {}
        for sfx in self.pairs:
            for n in SIGN_NAMES:
                t = getattr(self, n + sfx)
                signs = t if self.frozen else binarize(t)
                out[f"{prefix}{n}{sfx}_packed"] = pack_signs(signs).cpu()
                out[f"{prefix}{n}{sfx}_shape"] = torch.tensor(tuple(t.shape), dtype=torch.long)
            for n in SCALE_NAMES:
                out[f"{prefix}{n}{sfx}"] = getattr(self, n + sfx).detach().to(scale_dtype).cpu().contiguous()
        if self.bias is not None:
            out[f"{prefix}bias"] = self.bias.detach().to(scale_dtype).cpu().contiguous()
        for b in ("_eff_bit_target", "_split_dim_final", "_eff_bit_actual"):
            out[f"{prefix}{b}"] = getattr(self, b).detach().cpu()
        return out

    def bits_used(self) -> float:
        return float(self._eff_bit_actual) * self.in_features * self.out_features

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, split_dim={self.split_dim}, "
            f"residual={self.residual}, eff_bit={float(self._eff_bit_actual):.4f}, frozen={self.frozen}"
            + (f", act_quant={self.act_quant}" if self.act_quant != "none" else "")
        )


def littlebit_layers(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LittleBitLinear)]


def set_act_quant(model: nn.Module, mode: str) -> int:
    """Turn activation fake quantization ("none" | "q8_1") on/off for every LittleBit layer."""
    get_act_quant(mode)
    layers = littlebit_layers(model)
    for _, m in layers:
        m.act_quant = mode
    return len(layers)


def model_bits_report(model: nn.Module, keep_bpw: float = 0.0) -> dict:
    """Average bits per weight over LittleBit layers (scales and signs, as in the papers)."""
    layers = littlebit_layers(model)
    bits = sum(m.bits_used() for _, m in layers)
    weights = sum(m.in_features * m.out_features for _, m in layers)
    # mixed precision: plain linears left at base weights (lm_head/embeddings excluded, as for avg_bpw)
    kept = sum(m.weight.numel() for n, m in model.named_modules()
               if type(m) is nn.Linear and "lm_head" not in n)
    rep = {
        "layers": len(layers),
        "weights": weights,
        "avg_bpw": bits / weights if weights else math.nan,
        "kept_weights": kept,
    }
    if kept and keep_bpw:
        rep["mixed_bpw"] = (bits + kept * keep_bpw) / (weights + kept)
    return rep
