from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProtectConfig:
    """Hyperparameters from the paper's main configuration.

    ``reciprocal_epsilon`` and ``reciprocal_clip`` are intentionally disabled
    by default.  They are optional numerical-stability ablations, not part of
    the default formulation reported in the paper.
    """

    retrieval_k: int = 10
    refinement_steps: int = 10
    step_size: float = 1e-3
    visual_margin: float = 0.5
    attention_temperature: float = 0.1
    coarse_layers: Tuple[int, ...] = (0, 1, 2)
    fine_layers: Tuple[int, ...] = tuple(range(3, 14))
    reciprocal_epsilon: Optional[float] = None
    reciprocal_clip: Optional[float] = None
    noise_mode: str = "const"

    def validate(self, num_ws: int) -> None:
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be positive")
        if self.refinement_steps < 0:
            raise ValueError("refinement_steps cannot be negative")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if not self.coarse_layers or not self.fine_layers:
            raise ValueError("coarse_layers and fine_layers must be non-empty")
        all_layers = (*self.coarse_layers, *self.fine_layers)
        if len(set(all_layers)) != len(all_layers):
            raise ValueError("coarse_layers and fine_layers must be disjoint")
        if min(all_layers) < 0 or max(all_layers) >= num_ws:
            raise ValueError(f"layer indices must be in [0, {num_ws - 1}]")
