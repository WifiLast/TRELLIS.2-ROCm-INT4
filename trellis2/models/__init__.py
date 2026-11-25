import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _stream_into(model, model_file: str, device, only=None) -> None:
    """Copy checkpoint tensors one at a time into the (already allocated) model.

    Same result as load_state_dict(load_file(...), strict=False), but host RAM never
    holds more than one tensor of the checkpoint.
    """
    import torch
    from safetensors import safe_open
    targets = model.state_dict(keep_vars=True)
    with torch.no_grad(), safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            dst = targets.get(key)
            if dst is None or (only is not None and key not in only):  # strict=False: unexpected keys ignored
                continue
            src = f.get_tensor(key)
            if dst.shape != src.shape:
                raise RuntimeError(f"size mismatch for {key}: checkpoint {tuple(src.shape)} vs model {tuple(dst.shape)}")
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype, non_blocking=False))
            del src


def _recompute_rope(model, device):
    """Tensors __init__ computes that are not in the checkpoint; on the meta device they hold no values."""
    import torch
    from ..modules.attention import RotaryPositionEmbedder
    # computed on `device` exactly as __init__ does under torch.device(device): CPU and GPU pow differ in the
    # last bit, which changes the sampled mesh
    with torch.device(device):
        for m in model.modules():   # RotaryPositionEmbedder and SparseRotaryPositionEmbedder: freqs is no buffer
            f = getattr(m, 'freqs', None)
            if isinstance(f, torch.Tensor) and f.is_meta and hasattr(m, 'freq_dim') and hasattr(m, 'rope_freq'):
                freqs = torch.arange(m.freq_dim, dtype=torch.float32) / m.freq_dim
                m.freqs = m.rope_freq[0] / (m.rope_freq[1] ** freqs)
        if getattr(model, 'rope_phases', None) is not None and getattr(model, 'pe_mode', None) == 'rope':
            res = model.resolution
            coords = torch.meshgrid(*[torch.arange(res, device=device) for _ in range(3)], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            with torch.no_grad():
                model.rope_phases.copy_(RotaryPositionEmbedder(model.model_channels // model.num_heads, 3)(coords))


# checkpoint-less tensors _build_on_meta knows how to fill (anything else -> old construction path)
_RECOMPUTED = {'rope_phases'}
# callables fn(path, model, device) run on the meta-built model before anything is allocated, e.g. to swap layers
# for quantized ones that load their own weights (RoundBit): the replaced weights are then never allocated
META_HOOKS = []


def _materialize(model, device) -> None:
    """to_empty for the tensors still on the meta device only (layers a hook already loaded stay as they are)."""
    import torch
    for m in model.modules():
        for name, p in list(m._parameters.items()):
            if p is not None and p.is_meta:
                m._parameters[name] = torch.nn.Parameter(torch.empty_like(p, device=device),
                                                         requires_grad=p.requires_grad)
        for name, b in list(m._buffers.items()):
            if b is not None and b.is_meta:
                m._buffers[name] = torch.empty_like(b, device=device)


def _build_on_meta(config, model_file: str, device, kwargs, path: str = ""):
    """Build the model on the meta device (no memory), allocate it on `device` in its final dtypes and stream the
    checkpoint in. The old path builds it on the GPU in fp32 first: a 1.3B DiT peaked at 4.95 GiB while loading
    instead of 2.4 GiB. Returns None (caller falls back) when a tensor would stay uninitialised."""
    import torch
    from safetensors import safe_open
    with torch.device('meta'):
        model = __getattr__(config['name'])(**config['args'], **kwargs)
    with safe_open(model_file, framework="pt", device="cpu") as f:
        keys = set(f.keys())
    for hook in META_HOOKS:
        hook(path, model, device)
    missing = [k for k, v in model.state_dict().items() if k not in keys and v.is_meta]
    if any(k.split('.')[-1] not in _RECOMPUTED for k in missing):
        return None
    fresh = {k for k, v in model.state_dict(keep_vars=True).items() if v.is_meta}  # not loaded by a hook
    _materialize(model, device)
    _stream_into(model, model_file, device, only=fresh)
    _recompute_rope(model, device)
    for m in model.modules():       # plain tensor attributes created in __init__ (not params / buffers)
        for name, v in vars(m).items():
            if isinstance(v, torch.Tensor) and v.is_meta:
                del model
                torch.cuda.empty_cache()
                return None
    return model


def _override_dtype(config) -> None:
    """bf16 models (the flow DiTs) run in fp16 on GPUs without native bf16 (NVIDIA before Ampere): there xformers
    has no bf16 attention kernel and bf16 math is emulated. TRELLIS_DIT_DTYPE=float16|bfloat16 forces a dtype."""
    import os
    import torch
    args = config.get('args', {})
    if args.get('dtype') not in ('bf16', 'bfloat16'):
        return
    forced = os.environ.get('TRELLIS_DIT_DTYPE', '')
    if forced:
        args['dtype'] = forced
    elif torch.cuda.is_available() and not torch.cuda.is_bf16_supported(including_emulation=False):
        args['dtype'] = 'float16'


def from_pretrained(path: str, device=None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        device: When set to a GPU device, the model is built directly on it and the
                checkpoint is streamed tensor by tensor (no fp32 CPU init, no full
                state-dict copy in host RAM). Default: stock CPU loading.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    _override_dtype(config)
    if device is not None and str(device) != 'cpu':
        import torch
        if os.environ.get('TRELLIS_META_INIT', '1') != '0':
            model = _build_on_meta(config, model_file, device, kwargs, path)
            if model is not None:
                return model
        with torch.device(device):
            model = __getattr__(config['name'])(**config['args'], **kwargs)
        _stream_into(model, model_file, device)
        return model
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
