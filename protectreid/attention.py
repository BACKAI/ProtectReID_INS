"""The parameter-free reciprocal self-attention module from Section 3.3."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def reciprocal_self_attention(
    top_latents: torch.Tensor,
    bottom_latents: torch.Tensor,
    temperature: float = 0.1,
    reciprocal_epsilon: Optional[float] = None,
    reciprocal_clip: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate top-k and bottom-k W+ codes.

    Args:
        top_latents: ``[K, L, D]`` or ``[B, K, L, D]`` top-k codes.
        bottom_latents: Tensor with the same shape for bottom-k codes.
        temperature: Softmax temperature ``tau``.
        reciprocal_epsilon: Optional epsilon in ``1 / (M_a + epsilon)``.
        reciprocal_clip: Optional symmetric clipping value before softmax.

    Returns:
        ``(f_star, b_star)`` with shape ``[L, D]`` for unbatched input or
        ``[B, L, D]`` for batched input.

    The default is deliberately the paper's unbounded reciprocal attention:
    ``M_a = Q K^T / D`` followed by ``M = 1 / M_a`` and row-wise softmax.
    There are no learned projections, no cosine normalization, and no
    implicit epsilon or clipping in the default path.
    """

    if top_latents.ndim not in (3, 4):
        raise ValueError("top_latents must have shape [K,L,D] or [B,K,L,D]")
    if top_latents.shape != bottom_latents.shape:
        raise ValueError("top_latents and bottom_latents must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    unbatched = top_latents.ndim == 3
    if unbatched:
        top_latents = top_latents.unsqueeze(0)
        bottom_latents = bottom_latents.unsqueeze(0)

    # [B,K,L,D] -> [B,L,K,D], i.e. process the 14 StyleGAN channels/layers
    # independently as specified in Algorithm 2.
    x = top_latents.permute(0, 2, 1, 3)
    y = bottom_latents.permute(0, 2, 1, 3)
    d = x.shape[-1]

    # Q = K = V = X.  The paper uses /D, not the conventional /sqrt(D).
    similarity = torch.matmul(x, x.transpose(-1, -2)) / float(d)
    if reciprocal_epsilon is None:
        reciprocal = torch.reciprocal(similarity)
    else:
        reciprocal = torch.reciprocal(similarity + reciprocal_epsilon)
    if reciprocal_clip is not None:
        reciprocal = reciprocal.clamp(-reciprocal_clip, reciprocal_clip)

    weights = F.softmax(reciprocal / temperature, dim=-1)
    attended = torch.matmul(weights, x)
    f_star = attended.mean(dim=-2)
    b_star = y.mean(dim=-2)

    if unbatched:
        return f_star[0], b_star[0]
    return f_star, b_star
