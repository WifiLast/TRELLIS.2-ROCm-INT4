"""Real INT4 weight-only matmul (W4A16) for the packed RoundBit DiT layers, in Triton.

Weight format (packed DiT files, see int_layer.py): Qp uint8 (N, K/2), two codes per byte, low nibble = even column,
high nibble = odd column, value = q + 8 (q in [-8, 7]); scales fp16 (N, K/group). W[n, k] = (nib - 8) * s[n, k//group].

    y = int4_linear(x, Qp, s, group, bias)        x (..., K) fp16/bf16 -> (..., N) in x's dtype

Two paths, picked by M = rows of x (crossover measured on a 7900 XTX; on other GPUs, override with ROUNDBIT_INT4_FUSED_MAX_M):
  fused   (M <= fused_max_m): one kernel reads the packed codes, unpacks + scales them in registers and feeds
          tl.dot (fp32 accumulate). Memory-bound small-M shapes read 4x fewer weight bytes than fp16.
  dequant (larger M): one kernel unpacks the whole layer to a temporary fp16/bf16 weight, then torch.matmul
          (hipBLASLt / cuBLAS). Compute-bound shapes keep the vendor GEMM speed; only one layer's dense weight
          exists at a time.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

FUSED_MAX_M = int(os.environ.get("ROUNDBIT_INT4_FUSED_MAX_M", "256"))  # 7900 XTX: fused beats dequant+GEMM up to M=256


@triton.jit
def _dequant_kernel(qp_ptr, s_ptr, w_ptr, N, K, n_groups,
                    GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """W[n, k] for a BLOCK_N x BLOCK_K tile (BLOCK_K a multiple of GROUP, K even)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rkh = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)  # byte columns
    mask = (rn[:, None] < N) & (rkh[None, :] < K // 2)
    b = tl.load(qp_ptr + rn[:, None] * (K // 2) + rkh[None, :], mask=mask, other=0).to(tl.int32)
    lo = (b & 15) - 8
    hi = (b >> 4) - 8
    q = tl.join(lo, hi).reshape(BLOCK_N, BLOCK_K)  # interleave: column 2j = lo, 2j + 1 = hi
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    sc = tl.load(s_ptr + rn[:, None] * n_groups + rk[None, :] // GROUP,
                 mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0).to(tl.float32)
    w = q.to(tl.float32) * sc
    tl.store(w_ptr + rn[:, None] * K + rk[None, :], w.to(w_ptr.dtype.element_ty),
             mask=(rn[:, None] < N) & (rk[None, :] < K))


@triton.jit
def _fused_kernel(x_ptr, qp_ptr, s_ptr, b_ptr, y_ptr, M, N, K, n_groups,
                  stride_xm, stride_ym,
                  HAS_BIAS: tl.constexpr, GROUP: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """y[m, n] = sum_k x[m, k] * W[n, k]; BLOCK_K == GROUP so one scale per row and K tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k0 = kt * BLOCK_K
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        rkh = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        b = tl.load(qp_ptr + rn[:, None] * (K // 2) + rkh[None, :],
                    mask=(rn[:, None] < N) & (rkh[None, :] < K // 2), other=8).to(tl.int32)
        q = tl.join((b & 15) - 8, (b >> 4) - 8).reshape(BLOCK_N, BLOCK_K)
        sc = tl.load(s_ptr + rn * n_groups + k0 // GROUP, mask=rn < N, other=0.0).to(tl.float32)
        w = (q.to(tl.float32) * sc[:, None]).to(x.dtype)
        acc += tl.dot(x, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def dequant(qp: torch.Tensor, s: torch.Tensor, group: int, dtype=torch.float16) -> torch.Tensor:
    """Packed codes (N, K/2) + scales (N, K/group) -> dense (N, K)."""
    N, K = qp.shape[0], qp.shape[-1] * 2
    qp = qp.reshape(N, K // 2).contiguous()
    s = s.reshape(N, -1).contiguous()
    w = torch.empty(N, K, device=qp.device, dtype=dtype)
    BN, BK = 32, max(group, 128) // group * group
    _dequant_kernel[(triton.cdiv(N, BN), triton.cdiv(K, BK))](qp, s, w, N, K, s.shape[1],
                                                              GROUP=group, BLOCK_N=BN, BLOCK_K=BK)
    return w


def int4_linear(x: torch.Tensor, qp: torch.Tensor, s: torch.Tensor, group: int, bias: torch.Tensor | None = None,
                fused_max_m: int | None = None) -> torch.Tensor:
    N, K = qp.shape[0], qp.shape[-1] * 2
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]
    fmax = FUSED_MAX_M if fused_max_m is None else fused_max_m
    if M > fmax or K % group or group % 16:
        w = dequant(qp, s, group, x.dtype)
        y = torch.nn.functional.linear(x2, w, None if bias is None else bias.to(x.dtype))
        return y.reshape(*lead, N)
    x2 = x2.contiguous()
    qp = qp.reshape(N, K // 2).contiguous()
    s = s.reshape(N, -1).contiguous()
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    BM = 16 if M <= 16 else 32
    BN = 64
    _fused_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x2, qp, s, bias if bias is not None else s, y, M, N, K, s.shape[1], x2.stride(0), y.stride(0),
        HAS_BIAS=bias is not None, GROUP=group, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=group)
    return y.reshape(*lead, N)
