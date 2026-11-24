"""Sign functions with surrogate gradients (ported from LittleBit quantization/functions/binary.py)."""

import torch


class _STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * ((x > -1) & (x < 1))


class _SmoothSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=100.0):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        t = torch.tanh(ctx.alpha * x)
        return grad_output * ctx.alpha * (1 - t * t), None


def STEBinary(x):
    return _STEBinary.apply(x)


def SmoothSign(x):
    return _SmoothSign.apply(x)


QUANT_FUNCS = {"STEBinary": STEBinary, "SmoothSign": SmoothSign}


# ── activation fake quantization (what the inference kernels do to matmul inputs) ──
def fake_quant_q8_block(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """llama.cpp Q8_1/Q8_0 activation rounding along the last dim, straight-through gradient.

    Per block of 32: d = max|x| / 127, q = round(x / d), x -> q * fp16(d) (ggml rounds with the fp32 scale and
    stores it as fp16). A short last block is zero-padded, as the Q1_0 runner pads the rank to 128 with zero
    rows. Computed in fp32, returned in x's dtype.
    """
    n = x.shape[-1]
    pad = (-n) % block
    xf = x.float()
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    xb = xf.reshape(*xf.shape[:-1], -1, block)
    d = xb.abs().amax(dim=-1, keepdim=True) / 127.0
    q = torch.where(d > 0, torch.round(xb / d.clamp_min(1e-30)) * d.half().float(), torch.zeros_like(xb))
    q = q.reshape(xf.shape)[..., :n].to(x.dtype)
    return x + (q - x).detach()


ACT_QUANTS = {"none": None, "q8_1": fake_quant_q8_block}


def get_act_quant(name: str):
    try:
        return ACT_QUANTS[name]
    except KeyError:
        raise ValueError(f"unknown act_quant {name!r}; choose from {sorted(ACT_QUANTS)}") from None


def get_quant_func(name: str):
    try:
        return QUANT_FUNCS[name]
    except KeyError:
        raise ValueError(f"unknown quant_func {name!r}; choose from {sorted(QUANT_FUNCS)}") from None
