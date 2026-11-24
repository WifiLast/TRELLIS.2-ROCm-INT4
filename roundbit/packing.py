"""1-bit packing, bit-compatible with LittleBit quantization/utils/binary_packer.py.

Layout: row-major int32 words, lsb first, +1 -> bit 0, -1 -> bit 1, rows padded with +1.
"""

import torch


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack a 2-D {-1,+1} tensor (any dtype) into int32 words of shape (rows, ceil(cols/32))."""
    if signs.dim() != 2:
        raise ValueError(f"expected 2-D tensor, got shape {tuple(signs.shape)}")
    rows, cols = signs.shape
    words = (cols + 31) // 32
    bits = (signs < 0).to(torch.int64)
    pad = words * 32 - cols
    if pad:
        bits = torch.cat([bits, bits.new_zeros(rows, pad)], dim=1)
    weights = torch.ones(32, dtype=torch.int64, device=bits.device) << torch.arange(32, device=bits.device)
    packed = (bits.view(rows, words, 32) * weights).sum(dim=2)
    # wrap into signed int32 range (bit 31 set -> negative), same bits as LittleBit's int32 sum
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32).contiguous()


def unpack_signs(packed: torch.Tensor, shape, dtype=torch.int8) -> torch.Tensor:
    """Inverse of pack_signs; returns {-1,+1} in ``dtype``."""
    rows, cols = (int(s) for s in shape)
    words = (cols + 31) // 32
    if tuple(packed.shape) != (rows, words):
        raise ValueError(f"packed shape {tuple(packed.shape)} does not match {(rows, words)}")
    shifts = torch.arange(32, device=packed.device, dtype=torch.int32)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(rows, words * 32)[:, :cols]
    return (1 - 2 * bits).to(dtype)
