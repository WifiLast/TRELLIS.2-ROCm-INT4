from typing import *
import torch
import torch.nn as nn
from .. import models


class LazyModels(dict):
    """
    Model dict that loads checkpoints on first access and keeps only a bounded
    amount of weights resident in host RAM.

    TRELLIS.2-4B is ~16 GB of weights; loading all of it up front (as the
    stock pipeline does) does not fit next to anything else on a 32 GB host.
    In low-VRAM mode the pipeline moves one model at a time to the GPU, so the
    rest can stay on disk. Models sitting on the CPU are dropped least-recently
    -used first once they exceed TRELLIS_MODEL_CACHE_GB (default 5).
    """
    def __init__(self, specs: dict, loader: Callable):
        super().__init__()
        self._specs = dict(specs)
        self._loader = loader
        self._budget = float(__import__('os').environ.get('TRELLIS_MODEL_CACHE_GB', '5')) * 2**30
        # Set by the pipeline: device new models are built on (GPU -> no fp32 CPU init,
        # weights streamed tensor by tensor; TRELLIS_LOAD_ON_GPU=0 keeps CPU loading).
        self._device_fn = None

    @staticmethod
    def _cpu_bytes(model: nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in model.parameters() if p.device.type == 'cpu')

    def _evict(self, keep: str = None):
        import gc
        resident = [(k, self._cpu_bytes(m)) for k, m in super().items() if k != keep]
        total = sum(b for _, b in resident)
        dropped = False
        for k, b in resident:           # dict order == least recently used first
            if total <= self._budget:
                break
            if b > 0:
                super().__delitem__(k)
                total -= b
                dropped = True
        if dropped:
            gc.collect()

    def __getitem__(self, key):
        if super().__contains__(key):
            model = super().pop(key)    # re-insert: mark as most recently used
            super().__setitem__(key, model)
            return model
        if key not in self._specs:
            raise KeyError(key)
        self._evict()
        device = self._device_fn() if self._device_fn is not None else None
        if __import__('os').environ.get('TRELLIS_LOAD_ON_GPU', '1') == '0':
            device = None
        model = self._loader(self._specs[key], device)
        model.eval()
        super().__setitem__(key, model)
        return model

    def __contains__(self, key):
        return key in self._specs or super().__contains__(key)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def keys(self):
        return list(dict.fromkeys(list(self._specs) + list(super().keys())))

    def values(self):
        return (self[k] for k in self.keys())

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def loaded(self) -> dict:
        """Models currently in memory (no loading)."""
        return dict(super().items())


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        if isinstance(models, LazyModels):
            return
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        def _load(v, device=None):
            try:
                return models.from_pretrained(f"{path}/{v}", device=device)
            except Exception as e:
                return models.from_pretrained(v, device=device)

        specs = {
            k: v for k, v in args['models'].items()
            if not hasattr(cls, 'model_names_to_load') or k in cls.model_names_to_load
        }
        if os.environ.get('TRELLIS_LAZY_LOAD', '1') != '0':
            _models = LazyModels(specs, _load)
        else:
            _models = {k: _load(v) for k, v in specs.items()}

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        if isinstance(_models, LazyModels):
            _models._device_fn = lambda: getattr(new_pipeline, '_device', None)
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        loaded = self.models.loaded() if isinstance(self.models, LazyModels) else self.models
        for model in loaded.values():
            if hasattr(model, 'device'):
                return model.device
        for model in loaded.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))