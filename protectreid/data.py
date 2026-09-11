"""Small dependency-light image I/O helpers used by the CLI."""

from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from PIL import Image
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: Union[str, Path]) -> List[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def image_id(path: Union[str, Path]) -> str:
    """Return the filename stem used by the gallery exclusion interface."""

    return Path(path).stem


def load_image(path: Union[str, Path], size: int) -> torch.Tensor:
    """Load an RGB image as ``[3,size,size]`` in the generator's [-1,1] range."""

    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    """Save a generator-range tensor as an RGB image."""

    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("image tensor must have shape [3,H,W] or [1,3,H,W]")
    array = ((tensor.permute(1, 2, 0).clamp(-1, 1) + 1.0) * 127.5).round().byte().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
