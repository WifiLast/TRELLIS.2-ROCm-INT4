"""TRELLIS.2 image -> 3D (AMD ROCm or NVIDIA CUDA) with INT4 flow DiTs and INT8 latent decoders.

    ./generate.sh --image photo.png                               # 512 pipeline -> outputs/photo_s42.glb + preview
    ./generate.sh --image photo.png --pipeline 1024_cascade
    ./generate.sh --image photo.png --stock                       # unmodified bf16 / fp16 models (reference)

The INT model files are downloaded from the Hugging Face model repo in $RB_MODEL_REPO (or read from --models_dir);
everything else (norms, embeddings, sparse-structure decoder, DINOv3 image encoder) comes from microsoft/TRELLIS.2-4B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import torch  # noqa: E402

from roundbit.int_layer import apply_decoder_file, apply_dit_file  # noqa: E402

MODEL_REPO = os.environ.get("RB_MODEL_REPO", "JustAnotherUser1/TRELLIS.2-ROCm-INT4")
# flow DiTs each pipeline type runs
PIPELINES = {
    "512": ["sparse_structure_flow_model", "shape_slat_flow_model_512", "tex_slat_flow_model_512"],
    "1024_cascade": ["sparse_structure_flow_model", "shape_slat_flow_model_512",
                     "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"],
}
DECODERS = ["shape_slat_decoder", "tex_slat_decoder"]


def model_files(keys: list[str], models_dir: str) -> dict[str, Path]:
    """key -> local path of its INT file (DiT: <key>.pt, decoder: <key>.int8.pt); downloads what is missing."""
    names = {k: f"{k}.int8.pt" if k in DECODERS else f"{k}.pt" for k in keys}
    if models_dir:
        files = {k: Path(models_dir) / n for k, n in names.items()}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            sys.exit(f"missing model files: {', '.join(missing)}")
        return files
    from huggingface_hub import hf_hub_download

    return {k: Path(hf_hub_download(MODEL_REPO, n)) for k, n in names.items()}


def install_int_models(pipe, files: dict[str, Path]) -> dict:
    """Swap the INT layers in while the loader still has the model on the meta device (the replaced bf16 / fp16
    weights are never allocated), and after loading for models built the stock way. Runs again whenever the lazy
    model cache reloads a model."""
    from trellis2 import models as tm

    lm = pipe.models
    by_spec = {lm._specs[k]: k for k in files}
    done = {}

    def apply(key, model, device):
        fn = apply_decoder_file if key in DECODERS else apply_dit_file
        model._rb_int = fn(model, files[key], device)
        done[key] = model._rb_int

    def meta_hook(path, model, device):
        for spec, key in by_spec.items():
            if path.endswith(spec):
                apply(key, model, device)

    tm.META_HOOKS.append(meta_hook)
    orig = lm._loader

    def loader(spec, device):
        model = orig(spec, device)
        key = by_spec.get(spec)
        if key is not None and not getattr(model, "_rb_int", 0):  # meta build not taken: swap after loading
            apply(key, model, next(model.parameters()).device)
            torch.cuda.empty_cache()
        return model

    lm._loader = loader
    return done


def export(mesh, stem: Path, a):
    import imageio
    import numpy as np
    import o_voxel

    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils

    mesh.simplify(16777216)  # nvdiffrast limit
    if a.preview:
        envmap = EnvMap(torch.tensor(render_utils.load_hdri(str(HERE / "assets/hdri/forest.exr")),
                                     dtype=torch.float32, device="cuda"))
        frames = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap, num_frames=4,
                                                                            resolution=512))
        grid = np.concatenate([np.concatenate(frames[:2], 1), np.concatenate(frames[2:4], 1)], 0)
        imageio.imwrite(f"{stem}.png", grid)
        print(f"preview: {stem}.png", flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=a.faces, texture_size=a.texture_size, remesh=True, remesh_band=1, remesh_project=0,
        verbose=True)
    glb.export(f"{stem}.glb", extension_webp=True)
    print(f"model: {stem}.glb", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="input image (background is removed automatically)")
    ap.add_argument("--pipeline", default="512", choices=sorted(PIPELINES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "outputs"), help="output directory")
    ap.add_argument("--models_dir", default="", help="local folder with the INT model files (default: download "
                                                     "from the Hugging Face repo in $RB_MODEL_REPO)")
    ap.add_argument("--stock", action="store_true", help="unmodified pipeline (bf16 DiTs, fp16 decoders)")
    ap.add_argument("--fp16_decoders", action="store_true", help="INT4 DiTs, but the original fp16 decoders")
    ap.add_argument("--symmetry", default="off", help="off | auto | explicit axes such as x, xy (z is up)")
    ap.add_argument("--sampler", default="",
                    help='JSON sampler overrides per stage, e.g. {"shape": {"steps": 24}, "tex": {"steps": 24}}')
    ap.add_argument("--faces", type=int, default=1_000_000, help="GLB decimation target")
    ap.add_argument("--texture_size", type=int, default=4096)
    ap.add_argument("--no_preview", dest="preview", action="store_false", help="skip the 2x2 PBR render grid")
    ap.add_argument("--no_warmup", dest="warmup", action="store_false",
                    help="skip the 1-step warm-up run (faster; the first sample of a process can differ slightly)")
    a = ap.parse_args()

    from PIL import Image

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pt = "" if a.pipeline == "512" else f"_{a.pipeline}"
    stem = out / f"{Path(a.image).stem}{pt}{'_stock' if a.stock else ''}_s{a.seed}"

    keys = [] if a.stock else PIPELINES[a.pipeline] + ([] if a.fp16_decoders else DECODERS)
    files = model_files(keys, a.models_dir)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    loaded = install_int_models(pipe, files) if files else {}

    img = Image.open(a.image)
    run = dict(pipeline_type=a.pipeline, seed=a.seed, symmetry=a.symmetry)
    if a.warmup:  # also triggers the Triton / flex_gemm autotuning
        one = {"steps": 1}
        pipe.run(img, sparse_structure_sampler_params=one, shape_slat_sampler_params=one,
                 tex_slat_sampler_params=one, **run)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sp = json.loads(a.sampler) if a.sampler else {}
    mesh = pipe.run(img, sparse_structure_sampler_params=sp.get("ss", {}), shape_slat_sampler_params=sp.get("shape", {}),
                    tex_slat_sampler_params=sp.get("tex", {}), **run)[0]
    torch.cuda.synchronize()
    print(json.dumps({"generate_s": round(time.time() - t0, 1),
                      "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                      "int_layers": loaded}), flush=True)
    export(mesh, stem, a)


if __name__ == "__main__":
    main()
