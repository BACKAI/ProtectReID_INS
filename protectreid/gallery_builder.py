"""Build the paired identity/W+ gallery used by ProtectReID."""

from pathlib import Path
from typing import Union

import numpy as np
import torch

from .data import image_id, list_images, load_image
from .models import E4EEncoder, IdentityExtractor
from .retrieval import LatentGallery


def build_gallery(
    image_root: Union[str, Path],
    output_path: Union[str, Path],
    identity_extractor: IdentityExtractor,
    e4e_encoder: E4EEncoder,
    image_size: int = 256,
) -> LatentGallery:
    """Encode every image as ``(identity feature, 14x512 W+)`` and save NPZ."""

    paths = list_images(image_root)
    if not paths:
        raise ValueError(f"no images found in {image_root}")
    device = next(identity_extractor.model.parameters()).device
    features, latents, ids = [], [], []
    for path in paths:
        image = load_image(path, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = identity_extractor(image, no_grad=True)
            latent = e4e_encoder(image)
        features.append(feature[0].detach().cpu())
        latents.append(latent[0].detach().cpu())
        ids.append(image_id(path))

    gallery = LatentGallery(torch.stack(features), torch.stack(latents), ids, device="cpu")
    gallery.save_npz(output_path)
    return gallery
