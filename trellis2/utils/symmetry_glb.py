"""
GLB export with mirrored UVs for symmetric meshes.

Same cleaning / remeshing / baking as `o_voxel.postprocess.to_glb`, but the mesh is
cut at each mirror plane (grid-centre planes, coordinate 0 in [-0.5, 0.5]^3), only
the kept half (or quarter) is UV-unwrapped and baked, and the result is mirrored
back with shared UVs. Effects:
  - the exported mesh is exactly symmetric (vertices, faces and UVs),
  - each mirror axis doubles the texel area per surface for the same texture size,
  - no UV seam along the mirror plane (plane vertices are shared by both halves).

Asymmetric texture details (logos, text) are taken from the kept (+axis) side and
mirrored, so only use this when the PBR volume is symmetric; see
`choose_uv_mirror_axes`.
"""
from typing import *
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual

from .symmetry import AXIS_NAMES

# Max fraction of mirrored voxel pairs that the texture gate left asymmetric for an
# axis to still get mirrored UVs in 'auto' mode. Measured 2026-11-20 at 512: the
# steampunk T 2.6-3.3 % (nothing worth keeping), white armour 9 % (side-specific trim).
DEFAULT_UV_ASYM_LIMIT = 0.05


def choose_uv_mirror_axes(symmetry: Optional[dict], limit: float = DEFAULT_UV_ASYM_LIMIT) -> Tuple[int, ...]:
    """Axes from mesh.symmetry whose texture is (nearly) symmetric."""
    if not symmetry or not symmetry.get('axes'):
        return ()
    asym = symmetry.get('tex_asym', {})
    return tuple(a for a in symmetry['axes'] if asym.get(AXIS_NAMES[a], 1.0) <= limit)


def _cut(v: np.ndarray, f: np.ndarray, axis: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the part with coordinate[axis] >= 0. Crossing triangles are clipped
    (Sutherland-Hodgman); each cut edge gets exactly one new vertex, shared by both
    adjacent faces, so the cut adds no cracks. Faces lying in the plane are dropped
    (their mirror image would duplicate them).
    """
    v = np.asarray(v, dtype=np.float64).copy()
    f = np.asarray(f, dtype=np.int64)
    d = v[:, axis]
    d[np.abs(d) < eps] = 0.0
    s = np.sign(d)[f]
    keep = (s >= 0).all(1) & (s > 0).any(1)
    cross = (s > 0).any(1) & (s < 0).any(1)
    tris = [f[keep]]
    edge_pt, new_v = {}, []
    clipped = []
    for face in f[cross].tolist():
        poly = []
        for k in range(3):
            i, j = face[k], face[(k + 1) % 3]
            if d[i] >= 0:
                poly.append(i)
            if d[i] * d[j] < 0:
                key = (i, j) if i < j else (j, i)
                if key not in edge_pt:
                    lo, hi = key
                    t = d[lo] / (d[lo] - d[hi])
                    p = v[lo] + t * (v[hi] - v[lo])
                    p[axis] = 0.0
                    edge_pt[key] = v.shape[0] + len(new_v)
                    new_v.append(p)
                poly.append(edge_pt[key])
        for k in range(1, len(poly) - 1):
            clipped.append((poly[0], poly[k], poly[k + 1]))
    if clipped:
        tris.append(np.asarray(clipped, dtype=np.int64))
    if new_v:
        v = np.concatenate([v, np.asarray(new_v)])
    f = np.concatenate(tris)
    used = np.zeros(v.shape[0], dtype=bool)
    used[f.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return v[used].astype(np.float32), remap[f]


def _mirror_back(v, f, uv, n, axis):
    """Append the mirror image; plane vertices (coord == 0) are shared, not duplicated."""
    N = v.shape[0]
    on = v[:, axis] == 0
    n = n.clone()
    n[on, axis] = 0
    n = torch.nn.functional.normalize(n, dim=1)
    v2 = v.clone(); v2[:, axis] = -v2[:, axis]
    n2 = n.clone(); n2[:, axis] = -n2[:, axis]
    remap = torch.arange(2 * N, device=v.device)
    on_idx = on.nonzero().squeeze(1)
    remap[N + on_idx] = on_idx
    f2 = remap[f[:, [0, 2, 1]].long() + N]
    V = torch.cat([v, v2]); UV = torch.cat([uv, uv]); NN = torch.cat([n, n2])
    F = torch.cat([f.long(), f2])
    used = torch.zeros(2 * N, dtype=torch.bool, device=v.device)
    used[F.reshape(-1)] = True
    new_idx = torch.cumsum(used.long(), 0) - 1
    return V[used], new_idx[F].int(), UV[used], NN[used]


def to_glb_mirrored(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    mirror_axes: Sequence[int],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Like o_voxel.postprocess.to_glb, plus `mirror_axes` (0=x, 1=y, 2=z in pipeline
    coordinates). With no axes it simply calls to_glb.
    """
    import o_voxel
    mirror_axes = tuple(sorted(set(mirror_axes)))
    common = dict(
        vertices=vertices, faces=faces, attr_volume=attr_volume, coords=coords,
        attr_layout=attr_layout, aabb=aabb, voxel_size=voxel_size, grid_size=grid_size,
        decimation_target=decimation_target, texture_size=texture_size, remesh=remesh,
        remesh_band=remesh_band, remesh_project=remesh_project,
        mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
        mesh_cluster_refine_iterations=mesh_cluster_refine_iterations,
        mesh_cluster_global_iterations=mesh_cluster_global_iterations,
        mesh_cluster_smooth_strength=mesh_cluster_smooth_strength,
        verbose=verbose, use_tqdm=use_tqdm,
    )
    if not mirror_axes:
        return o_voxel.postprocess.to_glb(**common)

    import cumesh
    import nvdiffrast.torch as dr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from tqdm import tqdm

    # --- Input normalisation (as in to_glb) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    centre = aabb.mean(dim=0)
    assert centre.abs().max().item() < 1e-6, "mirrored export expects an AABB centred at the origin"

    pbar = tqdm(total=6, desc="Extracting GLB (mirrored UVs)") if use_tqdm else None
    step = lambda desc: (pbar.set_description(desc), pbar.update(1)) if pbar else None
    log = print if verbose else (lambda *a, **k: None)

    # --- Cleaning, remeshing / simplification (as in to_glb) ---
    vertices = vertices.cuda(); faces = faces.cuda()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()
    bvh = cumesh.cuBVH(vertices, faces)
    step("Cleaning mesh")
    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center=centre,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution,
            band=remesh_band,
            project_back=remesh_project,
            verbose=verbose,
            bvh=bvh,
        ))
        mesh.simplify(decimation_target, verbose=verbose)
    log(f"Full mesh before cutting: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    step("Cutting at mirror planes")

    # --- Cut to the fundamental domain ---
    v_np, f_np = [t.cpu().numpy() for t in mesh.read()]
    eps = 1e-6
    for a in mirror_axes:
        v_np, f_np = _cut(v_np, f_np.astype(np.int64), a, eps)
        log(f"Cut at {AXIS_NAMES[a]}=0: {v_np.shape[0]} vertices, {f_np.shape[0]} faces")
    if f_np.shape[0] == 0:
        log("Mirror cut left no faces, falling back to plain export")
        if pbar: pbar.close()
        return o_voxel.postprocess.to_glb(**common)

    half = cumesh.CuMesh()
    half.init(torch.from_numpy(v_np).cuda(), torch.from_numpy(f_np).int().cuda())
    step("Parameterizing half mesh")
    out_vertices, out_faces, out_uvs, out_vmaps = half.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda(); out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda(); out_vmaps = out_vmaps.cuda()
    half.compute_vertex_normals()
    out_normals = half.read_vertex_normals()[out_vmaps]
    # unwrap copies positions, so plane vertices are still exactly 0; re-snap anyway
    for a in mirror_axes:
        out_vertices[out_vertices[:, a].abs() < eps, a] = 0.0

    # --- Bake the half's texture (identical to to_glb) ---
    step("Sampling attributes")
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(ctx, uvs_rast, out_faces[i:i + 100000], resolution=[texture_size, texture_size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )

    # --- Mirror geometry back (reverse order of the cuts) ---
    step("Mirroring")
    V, F, UV, NN = out_vertices, out_faces, out_uvs, out_normals
    for a in reversed(mirror_axes):
        V, F, UV, NN = _mirror_back(V, F, UV, NN, a)
    log(f"Mirrored mesh: {V.shape[0]} vertices, {F.shape[0]} faces "
        f"(texel area x{2 ** len(mirror_axes)} per surface)")

    # --- Texture post-processing & material (as in to_glb) ---
    step("Finalizing mesh")
    mask = mask.cpu().numpy()
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=True if not remesh else False,
    )
    vertices_np = V.cpu().numpy()
    faces_np = F.cpu().numpy()
    uvs_np = UV.cpu().numpy()
    normals_np = NN.cpu().numpy()
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1]
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )
    if pbar:
        pbar.update(1); pbar.close()
    return textured_mesh
