"""Adapters for the frozen StyleGAN3, e4e and re-ID checkpoints."""

import importlib
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_generator(
    checkpoint: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    stylegan_root: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """Load ``G_ema`` from an official StyleGAN3 pickle.

    ``stylegan_root`` should point to a checkout of NVLabs' stylegan3-
    ada-pytorch repository.  The root is only used to import its ``legacy``
    loader; the checkpoint itself is never modified.
    """

    if stylegan_root is not None:
        root = str(Path(stylegan_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    checkpoint = Path(checkpoint)
    try:
        legacy = importlib.import_module("legacy")
        with checkpoint.open("rb") as handle:
            data = legacy.load_network_pkl(handle)
    except (ImportError, ModuleNotFoundError):
        # A self-contained pickle can be loaded without the StyleGAN helper.
        with checkpoint.open("rb") as handle:
            data = pickle.load(handle)

    if "G_ema" not in data:
        raise KeyError("generator checkpoint does not contain G_ema")
    generator = data["G_ema"].to(device)
    _freeze(generator)
    num_ws = getattr(generator, "num_ws", getattr(getattr(generator, "mapping", None), "num_ws", None))
    if num_ws is not None and int(num_ws) != 14:
        raise ValueError(f"expected a 14-layer W+ generator, found num_ws={num_ws}")
    return generator


class _ClassBlock(nn.Module):
    """State-dict-compatible head used by the released ResNet50 re-ID model."""

    def __init__(self, input_dim: int = 2048, class_num: int = 751, dropout: float = 0.5):
        super().__init__()
        self.add_block = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.add_block(x))


class ResNet50ReID(nn.Module):
    """The released Market-1501 ResNet50 embedding extractor.

    The original repository builds a torchvision ResNet50 and replaces the
    classifier with an empty sequential module at inference.  This class keeps
    the same ``model.*`` and ``classifier.*`` state-dict names.
    """

    def __init__(self, num_classes: int = 751):
        super().__init__()
        try:
            from torchvision.models import resnet50
        except ImportError as exc:
            raise ImportError("torchvision is required for the ResNet50 extractor") from exc
        self.model = resnet50(weights=None)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Identity()
        self.classifier = _ClassBlock(class_num=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return self.classifier(features)


def _state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint)
    return checkpoint


def load_reid_extractor(
    checkpoint: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    num_classes: int = 751,
) -> nn.Module:
    """Load the released ResNet50 re-ID network and expose its 512-D head."""

    loaded = torch.load(checkpoint, map_location="cpu")
    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        model = ResNet50ReID(num_classes=num_classes)
        state = _state_dict(loaded)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    _freeze(model)
    # The released code sets classifier.classifier = Sequential() before use.
    if hasattr(model, "classifier") and hasattr(model.classifier, "classifier"):
        model.classifier.classifier = nn.Identity()
    return model


class IdentityExtractor:
    """Differentiable image-to-identity wrapper used by both retrieval and loss."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=next(model.parameters()).device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=next(model.parameters()).device).view(1, 3, 1, 1)

    def __call__(self, images: torch.Tensor, no_grad: bool = False) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = F.interpolate(images, size=(256, 128), mode="bicubic", align_corners=False)
        images = (images + 1.0) / 2.0
        images = (images - self.mean.to(images)) / self.std.to(images)
        if no_grad:
            with torch.no_grad():
                return self.model(images)
        return self.model(images)


class E4EEncoder:
    """Load only the e4e encoder, avoiding the old repository's PTI path."""

    def __init__(self, encoder: nn.Module, device: Union[str, torch.device]):
        self.encoder = _freeze(encoder.to(device))
        self.device = torch.device(device)

    @classmethod
    def load(
        cls,
        checkpoint: Union[str, Path],
        e4e_root: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
    ) -> "E4EEncoder":
        root = str(Path(e4e_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            encoders = importlib.import_module("models.e4e_reid.encoders.psp_encoders")
        except ImportError as exc:
            raise ImportError(
                "Could not import e4e encoders. Pass the root of the released "
                "ProtectReID/e4e repository with --e4e-root."
            ) from exc

        state = torch.load(checkpoint, map_location="cpu")
        raw_opts = state.get("opts", {}) if isinstance(state, dict) else {}
        opts_dict = vars(raw_opts).copy() if isinstance(raw_opts, Namespace) else dict(raw_opts)
        opts_dict.setdefault("stylegan_size", 256)
        opts_dict.setdefault("encoder_type", "Encoder4Editing")
        opts = Namespace(**opts_dict)
        encoder_type = opts.encoder_type
        if encoder_type == "GradualStyleEncoder":
            encoder = encoders.GradualStyleEncoder(50, "ir_se", opts)
        elif encoder_type == "Encoder4Editing":
            encoder = encoders.Encoder4Editing(50, "ir_se", opts)
        elif encoder_type == "SingleStyleCodeEncoder":
            encoder = encoders.BackboneEncoderUsingLastLayerIntoW(50, "ir_se", opts)
        else:
            raise ValueError(f"unsupported e4e encoder_type: {encoder_type}")

        raw_state = state.get("state_dict", state)
        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in raw_state.items()
            if key.startswith("encoder.")
        }
        encoder.load_state_dict(encoder_state or raw_state, strict=True)
        return cls(encoder, device)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        with torch.no_grad():
            latents = self.encoder(images.to(self.device))
        if latents.ndim != 3 or latents.shape[1:] != (14, 512):
            raise ValueError(f"e4e must return [B,14,512], got {tuple(latents.shape)}")
        return latents
