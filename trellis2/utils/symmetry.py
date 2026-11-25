"""
Mirror-symmetry support for the TRELLIS.2 image-to-3D pipeline.

All mirrors are about the grid centre plane of one of the three axes
(voxel i -> R-1-i). That plane stays consistent through every level of the
pipeline: the 64^3 sparse structure, the 32/64 latent grid, the decoder's 2x
subdivisions up to 512/1024/1536 and the cascade's re-quantisation, so a
symmetric coarse structure stays symmetric all the way down.

What is symmetrised (no retraining, everything is post-hoc on decoder outputs):
  1. sparse-structure occupancy logits (dense 64^3)          -> symmetric token set
  2. subdivision logits of every shape-decoder upsample block -> symmetric voxel set
  3. the 7 flexible-dual-grid output channels                 -> exactly symmetric mesh
  4. PBR voxel attributes, gated by disagreement              -> keeps logos / text
Mirrored UVs for the GLB export live in `trellis2.utils.symmetry_glb`.

Axes are pipeline/voxel axes: 0 = x, 1 = y, 2 = z (z is up; GLB y = z).
"""
from contextlib import contextmanager
from typing import *
import torch

AXIS_NAMES = 'xyz'

# Shell IoU of the 64^3 occupancy with its mirror; above this an axis counts as
# symmetric in 'auto' mode. Calibrated on the 72 assets/example_image inputs
# (tests_amd/sym_calibrate.py, 2026-11-20): clearly mirror-symmetric objects score
# 0.80-1.00, posed characters / plants / dioramas 0.05-0.60; 0.75-0.79 held
# borderline cases (roaring bear, mech with one gun arm), so 0.80 favours precision.
DEFAULT_DETECT_THRESHOLD = 0.80

# Texture gate: per-voxel max |a - mirror(a)| over base colour / metallic /
# roughness. Below `lo` both sides are averaged, above `hi` they are kept.
DEFAULT_TEX_GATE = (0.08, 0.20)


# ----------------------------------------------------------------------------
# Spec parsing
# ----------------------------------------------------------------------------

def parse_symmetry(spec: Union[None, str, Sequence[int]]) -> Union[str, Tuple[int, ...]]:
    """
    'off' / None / '' -> ()      'auto' -> 'auto'
    'x', 'xy', 'x,z', [0, 2]     -> sorted tuple of axis indices
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ('', 'off', 'none', 'false', '0'):
            return ()
        if s == 'auto':
            return 'auto'
        axes = []
        for ch in s.replace(',', '').replace(' ', ''):
            if ch not in AXIS_NAMES:
                raise ValueError(f"Invalid symmetry spec {spec!r}: use 'off', 'auto' or axes from 'xyz'")
            axes.append(AXIS_NAMES.index(ch))
        return tuple(sorted(set(axes)))
    axes = tuple(sorted(set(int(a) for a in spec)))
    if any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"Invalid symmetry axes {spec!r}")
    return axes


def axes_to_str(axes: Sequence[int]) -> str:
    return ''.join(AXIS_NAMES[a] for a in axes) or 'off'


# ----------------------------------------------------------------------------
# Dense (sparse-structure) stage
# ----------------------------------------------------------------------------

@torch.no_grad()
def dense_mirror_scores(logits: torch.Tensor) -> List[float]:
    """
    Shell IoU of occupancy (logits > 0) with its grid-centre mirror, per axis.
    logits: [B, 1, R, R, R]; the score is the minimum over the batch.
    """
    occ = logits > 0
    scores = []
    for a in range(3):
        m = occ.flip(2 + a)
        inter = (occ & m).flatten(1).sum(1).float()
        union = (occ | m).flatten(1).sum(1).float().clamp(min=1)
        scores.append((inter / union).min().item())
    return scores


def dense_symmetrize(logits: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    for a in axes:
        logits = 0.5 * (logits + logits.flip(2 + a))
    return logits


# ----------------------------------------------------------------------------
# Sparse coordinate lookup
# ----------------------------------------------------------------------------

class CoordIndex:
    """Sorted int64 keys of [N, 4] (batch, x, y, z) coords for O(log N) lookup."""

    def __init__(self, coords: torch.Tensor, R: int):
        self.R = R
        keys = self.key(coords)
        self.sorted_keys, self.order = torch.sort(keys)

    def key(self, coords: torch.Tensor) -> torch.Tensor:
        c = coords.long()
        R = self.R
        return ((c[:, 0] * R + c[:, 1]) * R + c[:, 2]) * R + c[:, 3]

    def lookup(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (index into the original coords, found mask)."""
        q = self.key(coords)
        pos = torch.searchsorted(self.sorted_keys, q).clamp(max=self.sorted_keys.numel() - 1)
        found = self.sorted_keys[pos] == q
        # out-of-range queries (mirror with shift may leave the grid)
        inb = ((coords[:, 1:] >= 0) & (coords[:, 1:] < self.R)).all(dim=1)
        found = found & inb
        return self.order[pos], found


def mirror_coords(coords: torch.Tensor, axis: int, R: int, shift: int = 0) -> torch.Tensor:
    """Voxel i -> R-1-shift-i on `axis` (coords are [N, 4] with batch first)."""
    out = coords.clone()
    out[:, 1 + axis] = R - 1 - shift - out[:, 1 + axis]
    return out


def _mirror_partner(coords: torch.Tensor, axis: int, R: int, shift: int = 0, index: CoordIndex = None):
    index = index or CoordIndex(coords, R)
    return index.lookup(mirror_coords(coords, axis, R, shift))


# ----------------------------------------------------------------------------
# Decoder hooks: subdivision logits and flexible-dual-grid outputs
# ----------------------------------------------------------------------------

def _child_perm(axis: int, device) -> torch.Tensor:
    # child index bit i = offset on axis i (see SparseUpsample / SparseChannel2Spatial)
    return torch.arange(8, device=device) ^ (1 << axis)


def symmetrize_subdiv(coords: torch.Tensor, logits: torch.Tensor, axes: Sequence[int], R: int) -> torch.Tensor:
    """Average each voxel's 8 child logits with the mirrored voxel's mirrored children."""
    feats = logits.float()
    index = CoordIndex(coords, R)
    for a in axes:
        idx, found = _mirror_partner(coords, a, R, index=index)
        partner = feats[idx][:, _child_perm(a, feats.device)]
        feats = torch.where(found[:, None], 0.5 * (feats + partner), feats)
    return feats.to(logits.dtype)


def symmetrize_dual_grid(coords: torch.Tensor, h: torch.Tensor, axes: Sequence[int], R: int) -> torch.Tensor:
    """
    Symmetrise the raw 7-channel FlexiDualGrid decoder output.

      0:3  dual vertex, v = (1+2m)*sigmoid(f) - m in voxel-local [0,1] -> mirror v' = 1-v, i.e. f' = -f
      3:6  'intersected' logit for the primal edge along axis k owned by the voxel.
           The edge along the mirror axis maps to the mirrored voxel (R-1-i); edges along the
           other two axes sit on the voxel's max face and map to voxel R-2-i.
      6    quad split weight (per voxel scalar)

    Voxels whose partner is missing keep their own value.
    """
    feats = h.float()
    index = CoordIndex(coords, R)
    for a in axes:
        idx0, found0 = _mirror_partner(coords, a, R, 0, index)
        idx1, found1 = _mirror_partner(coords, a, R, 1, index)
        p0 = feats[idx0]
        p1 = feats[idx1]
        mirrored = p0.clone()
        mirrored[:, a] = -p0[:, a]
        use = found0[:, None].expand_as(feats).clone()
        for k in range(3):
            if k != a:
                mirrored[:, 3 + k] = p1[:, 3 + k]
                use[:, 3 + k] = found1
        feats = torch.where(use, 0.5 * (feats + mirrored), feats)
    return feats.to(h.dtype)


@contextmanager
def shape_decoder_symmetry(decoder, axes: Sequence[int], base_res: int, final_res: Optional[int] = None):
    """
    While active, symmetrise the shape decoder's subdivision logits (all upsample blocks)
    and, if `final_res` is given, its 7-channel output.

    base_res: grid resolution of the latent fed to the decoder (resolution // 16).
    """
    handles = []
    if axes:
        n_levels = len(decoder.blocks)
        for i in range(n_levels - 1):
            up = decoder.blocks[i][-1]
            if not hasattr(up, 'to_subdiv'):
                continue
            R = base_res * 2 ** i

            def hook(module, inputs, output, R=R):
                return output.replace(symmetrize_subdiv(output.coords, output.feats, axes, R))
            handles.append(up.to_subdiv.register_forward_hook(hook))
        if final_res is not None:
            def out_hook(module, inputs, output):
                return output.replace(symmetrize_dual_grid(output.coords, output.feats, axes, final_res))
            handles.append(decoder.output_layer.register_forward_hook(out_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------------------
# PBR attributes
# ----------------------------------------------------------------------------

@torch.no_grad()
def symmetrize_attrs(
    coords: torch.Tensor,
    attrs: torch.Tensor,
    axes: Sequence[int],
    R: int,
    gate: Tuple[float, float] = DEFAULT_TEX_GATE,
    gate_channels: slice = slice(0, 5),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Gated average of per-voxel attributes with their mirror partner.
    Returns new attrs and per-axis stats {axis: fraction of matched voxels kept asymmetric}.
    """
    lo, hi = gate
    feats = attrs.float()
    index = CoordIndex(coords, R)
    stats = {}
    for a in axes:
        idx, found = _mirror_partner(coords, a, R, index=index)
        partner = feats[idx]
        d = (feats[:, gate_channels] - partner[:, gate_channels]).abs().amax(dim=1)
        w = ((hi - d) / (hi - lo)).clamp(0, 1) * found.float()
        feats = feats + 0.5 * w[:, None] * (partner - feats)
        n = found.sum().clamp(min=1).float()
        stats[AXIS_NAMES[a]] = ((w < 0.5) & found).sum().float().div(n).item()
    return feats.to(attrs.dtype), stats


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

@torch.no_grad()
def sparse_mirror_iou(coords: torch.Tensor, R: int) -> List[float]:
    """IoU of a sparse voxel set ([N,4] coords) with its grid-centre mirror, per axis."""
    index = CoordIndex(coords, R)
    out = []
    for a in range(3):
        _, found = _mirror_partner(coords, a, R, index=index)
        hit = found.sum().item()
        out.append(hit / max(1, 2 * coords.shape[0] - hit))
    return out
