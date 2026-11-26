"""TRELLIS.2 image -> 3D (AMD ROCm or NVIDIA CUDA) with INT4 flow DiTs and INT8 latent decoders.

    ./generate.sh --image photo.png                               # 512 pipeline -> outputs/photo_s42.glb + preview
    ./generate.sh --image photo.png --pipeline 1024_cascade
    ./generate.sh --image photo.png --stock                       # unmodified bf16 / fp16 models (reference)

The INT model files are downloaded from the Hugging Face model repo in $RB_MODEL_REPO (or read from --models_dir);
everything else (norms, embeddings, sparse-structure decoder) comes from microsoft/TRELLIS.2-4B.
Optional stack files next to them (used when present, old model sets without them still work):
    bridges.pt                 block 29 of the shape / texture 512 DiTs replaced by a linear bridge (the DiT files
                               then lack that block and need it; --no_bridges only works with unstripped files)
    image_cond_model.int8.pt   DINOv3 image encoder in INT8 (309 MB instead of the 1213 MB fp32 checkpoint, which
                               is then never downloaded or read; only its config.json). Not used on GPUs without
                               native bf16 (NVIDIA before Ampere): fp16 overflows inside DINOv3, so fp32 there.
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

from roundbit.int_layer import apply_bridges, apply_decoder_file, apply_dit_file, build_cond  # noqa: E402

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


def optional_file(name: str, models_dir: str) -> Path | None:
    """a stack file (bridges.pt, image_cond_model.int8.pt) if the model set has it, else None"""
    if models_dir:
        f = Path(models_dir) / name
        return f if f.exists() else None
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    try:
        return Path(hf_hub_download(MODEL_REPO, name))
    except EntryNotFoundError:
        return None


def load_pipeline(cond_file: Path | None):
    """TRELLIS.2 pipeline; with cond_file the DINOv3 encoder is built from its config + the INT8 file
    (DinoV3FeatureExtractor calls DINOv3ViTModel.from_pretrained through the module global)."""
    import trellis2.modules.image_feature_extractor as ife
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    orig = ife.DINOv3ViTModel
    if cond_file is not None:
        class _Saved:
            @staticmethod
            def from_pretrained(name, *args, **kw):
                return build_cond(orig, name, cond_file)
        ife.DINOv3ViTModel = _Saved
    try:
        pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    finally:
        ife.DINOv3ViTModel = orig
    pipe.cuda()
    return pipe


def install_int_models(pipe, files: dict[str, Path], bridges: dict | None = None) -> dict:
    """Swap the INT layers in while the loader still has the model on the meta device (the replaced bf16 / fp16
    weights are never allocated), and after loading for models built the stock way. Runs again whenever the lazy
    model cache reloads a model."""
    from trellis2 import models as tm

    lm = pipe.models
    by_spec = {lm._specs[k]: k for k in files}
    done = {}

    def apply(key, model, device):
        if bridges and key in bridges:  # first: the bridged blocks are skipped by the DiT file (never allocated)
            done[f"{key}:bridged_blocks"] = apply_bridges(model, bridges[key], device)
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
    ap.add_argument("--no_bridges", action="store_true",
                    help="ignore bridges.pt (only for DiT files that still contain the bridged blocks)")
    ap.add_argument("--cond", default="auto", choices=["auto", "int8", "fp32"],
                    help="DINOv3 image encoder: int8 = image_cond_model.int8.pt from the model set, fp32 = stock; "
                         "auto = int8 when the file exists and the GPU has native bf16, else fp32")
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

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pt = "" if a.pipeline == "512" else f"_{a.pipeline}"
    stem = out / f"{Path(a.image).stem}{pt}{'_stock' if a.stock else ''}_s{a.seed}"

    keys = [] if a.stock else PIPELINES[a.pipeline] + ([] if a.fp16_decoders else DECODERS)
    files = model_files(keys, a.models_dir)
    bridges = {}
    if files and not a.no_bridges:
        bf = optional_file("bridges.pt", a.models_dir)
        if bf is not None:
            bridges = {k: v for k, v in torch.load(bf, map_location="cpu")["models"].items() if k in files}
    cond_file = None
    if not a.stock and a.cond != "fp32":
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        if a.cond == "int8" or bf16:
            cond_file = optional_file("image_cond_model.int8.pt", a.models_dir)
            if cond_file is None and a.cond == "int8":
                sys.exit("--cond int8: the model set has no image_cond_model.int8.pt")
            if cond_file is not None and not bf16:
                print("warning: no native bf16 on this GPU, the INT8 DINOv3 (bf16 base) runs emulated", flush=True)
    pipe = load_pipeline(cond_file)
    loaded = install_int_models(pipe, files, bridges) if files else {}
    if cond_file is not None:
        loaded["image_cond_model"] = "int8"

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
