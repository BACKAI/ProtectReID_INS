"""End-to-end ProtectReID inference."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .attention import reciprocal_self_attention
from .config import ProtectConfig
from .retrieval import LatentGallery, RetrievalResult


@dataclass
class ProtectionResult:
    protected_images: torch.Tensor
    final_latents: torch.Tensor
    top_indices: torch.Tensor
    bottom_indices: torch.Tensor
    top_scores: torch.Tensor
    bottom_scores: torch.Tensor
    history: List[Dict[str, torch.Tensor]]


class ProtectReIDPipeline:
    """Frozen-model ProtectReID pipeline.

    ``generator`` must expose ``synthesis(ws, noise_mode='const')`` and accept
    W+ codes shaped ``[B,14,512]``.  ``identity_extractor`` is any callable
    mapping generator-range images to 512-D embeddings; the adapter in
    :mod:`protectreid.models` supplies the released ResNet50 model.
    """

    def __init__(
        self,
        generator,
        identity_extractor,
        gallery: LatentGallery,
        config: Optional[ProtectConfig] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        self.generator = generator
        self.identity_extractor = identity_extractor
        self.gallery = gallery
        self.config = config or ProtectConfig()
        self.device = torch.device(device)
        self.generator.eval()
        self.config.validate(self.num_ws)

    @property
    def num_ws(self) -> int:
        value = getattr(self.generator, "num_ws", None)
        if value is None:
            value = getattr(getattr(self.generator, "mapping", None), "num_ws", None)
        if value is None:
            value = self.gallery.latents.shape[1]
        return int(value)

    def _synthesis(self, latents: torch.Tensor) -> torch.Tensor:
        try:
            return self.generator.synthesis(
                latents, noise_mode=self.config.noise_mode, force_fp32=True
            )
        except TypeError:
            return self.generator.synthesis(latents, noise_mode=self.config.noise_mode)

    @staticmethod
    def _squared_l2_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        # Equation (8) uses squared l2 norms, not a perceptual or feature loss.
        return (a - b).flatten(1).square().sum(dim=1)

    def _hierarchical_latent_manipulation(
        self,
        top_aggregate: torch.Tensor,
        bottom_aggregate: torch.Tensor,
        original_images: torch.Tensor,
        original_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            bottom_image = self._synthesis(bottom_aggregate.detach())

        latent = top_aggregate.detach().clone()
        history: List[Dict[str, torch.Tensor]] = []
        coarse = list(self.config.coarse_layers)
        fine = list(self.config.fine_layers)

        for _ in range(self.config.refinement_steps):
            # First update only coarse W+ layers with the visual margin loss.
            latent_for_visual = latent.detach().requires_grad_(True)
            visual_image = self._synthesis(latent_for_visual)
            visual_loss = F.relu(
                self._squared_l2_per_sample(visual_image, bottom_image)
                - self._squared_l2_per_sample(visual_image, original_images)
                + self.config.visual_margin
            )
            visual_grad = torch.autograd.grad(visual_loss.sum(), latent_for_visual)[0]
            latent_after_coarse = latent_for_visual.detach().clone()
            latent_after_coarse[:, coarse, :] -= self.config.step_size * visual_grad[:, coarse, :]

            # Then update only fine W+ layers with the identity cosine loss.
            latent_for_identity = latent_after_coarse.detach().requires_grad_(True)
            identity_image = self._synthesis(latent_for_identity)
            protected_features = self.identity_extractor(identity_image, no_grad=False)
            identity_loss = 1.0 - F.cosine_similarity(
                protected_features, original_features, dim=1
            )
            identity_grad = torch.autograd.grad(identity_loss.sum(), latent_for_identity)[0]
            latent = latent_for_identity.detach().clone()
            latent[:, fine, :] -= self.config.step_size * identity_grad[:, fine, :]

            history.append(
                {
                    "visual_loss": visual_loss.detach(),
                    "identity_loss": identity_loss.detach(),
                }
            )

        final_images = self._synthesis(latent)
        return final_images, latent, history

    def protect(
        self,
        images: torch.Tensor,
        query_ids: Optional[Sequence[Optional[str]]] = None,
        exclude_indices: Optional[Sequence[Optional[Union[int, Sequence[int]]]]] = None,
    ) -> ProtectionResult:
        """Protect a batch of generator-range images.

        The main CLI uses batch size one because each query may have a
        different source-gallery exclusion.  The algorithm itself supports a
        batch and performs independent latent refinement for every item.
        """

        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            query_features = self.identity_extractor(images, no_grad=True)

        if query_ids is None and exclude_indices is None:
            # The official .mat gallery has no IDs. When the exact source image
            # is present, its normalized feature is still an exact-match key.
            # Cross-dataset queries normally produce no rows and use FAISS.
            auto_rows = self.gallery.find_exact_rows(query_features)
            if any(auto_rows):
                exclude_indices = auto_rows

        retrieval: RetrievalResult = self.gallery.retrieve(
            query_features,
            k=self.config.retrieval_k,
            exclude_ids=query_ids,
            exclude_indices=exclude_indices,
        )
        top_latents = self.gallery.latents[retrieval.top_indices]
        bottom_latents = self.gallery.latents[retrieval.bottom_indices]
        top_aggregate, bottom_aggregate = reciprocal_self_attention(
            top_latents,
            bottom_latents,
            temperature=self.config.attention_temperature,
            reciprocal_epsilon=self.config.reciprocal_epsilon,
            reciprocal_clip=self.config.reciprocal_clip,
        )

        with torch.no_grad():
            original_features = self.identity_extractor(images, no_grad=True)
        protected, final_latents, history = self._hierarchical_latent_manipulation(
            top_aggregate,
            bottom_aggregate,
            images,
            original_features,
        )
        return ProtectionResult(
            protected_images=protected,
            final_latents=final_latents,
            top_indices=retrieval.top_indices,
            bottom_indices=retrieval.bottom_indices,
            top_scores=retrieval.top_scores,
            bottom_scores=retrieval.bottom_scores,
            history=history,
        )
