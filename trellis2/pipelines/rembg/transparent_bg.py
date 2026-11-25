from typing import *
import os
from PIL import Image


class TransparentBackground:
    """
    Background removal with InSPyReNet (the `transparent-background` package).

    Ungated replacement for briaai/RMBG-2.0. The network is loaded on first
    use, so pipelines that only ever see RGBA inputs never pay for it.
    """
    def __init__(self, mode: str = 'base', **kwargs):
        self.mode = mode
        self.device = 'cpu'
        self.remover = None

    def _load(self):
        if self.remover is None:
            from transparent_background import Remover
            self.remover = Remover(mode=self.mode, jit=False, device=str(self.device))
        return self.remover

    def to(self, device):
        self.device = device
        if self.remover is not None:
            self.remover.model.to(device)
            self.remover.device = str(device)

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    def __call__(self, image: Image.Image) -> Image.Image:
        return self._load().process(image.convert('RGB'), type='rgba')


def build_rembg(name: str, args: dict):
    """
    Build the background remover named in pipeline.json.

    TRELLIS_REMBG=transparent_background | birefnet | auto (default).
    `auto` uses the configured model when it can be loaded and falls back to
    transparent-background otherwise (RMBG-2.0 is a gated repository).
    """
    from . import BiRefNet
    choice = os.environ.get('TRELLIS_REMBG', 'auto').lower()
    if choice in ('transparent_background', 'transparent-background', 'inspyrenet'):
        return TransparentBackground()
    if choice == 'birefnet':
        return BiRefNet(**args)
    try:
        return {'BiRefNet': BiRefNet, 'TransparentBackground': TransparentBackground}[name](**args)
    except Exception as e:
        print(f"[REMBG] {name}({args}) unavailable ({type(e).__name__}); using transparent-background instead.")
        return TransparentBackground()
