"""Reference implementation of ProtectReID.

The defense has no trainable protector network.  It retrieves W+ codes from a
fixed gallery, fuses them with reciprocal self-attention, and optimizes one
W+ code per input image at inference time.
"""

from .attention import reciprocal_self_attention
from .config import ProtectConfig
from .pipeline import ProtectReIDPipeline, ProtectionResult
from .retrieval import LatentGallery, RetrievalResult

__all__ = [
    "LatentGallery",
    "ProtectConfig",
    "ProtectReIDPipeline",
    "ProtectionResult",
    "RetrievalResult",
    "reciprocal_self_attention",
]
