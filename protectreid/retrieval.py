"""Identity-to-latent retrieval and gallery serialization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    top_indices: torch.Tensor
    bottom_indices: torch.Tensor
    top_scores: torch.Tensor
    bottom_scores: torch.Tensor


class LatentGallery:
    """A gallery of paired identity embeddings and W+ latent codes.

    The paper uses FAISS for fast cosine retrieval. This class uses FAISS when
    it is installed and falls back to an exact torch matrix product, so the
    reference algorithm remains usable in minimal environments. Retrieval is
    exact in both paths; the FAISS path uses ``IndexFlatIP``.
    """

    def __init__(
        self,
        features: torch.Tensor,
        latents: torch.Tensor,
        ids: Optional[Sequence[str]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        if features.ndim != 2:
            raise ValueError("features must have shape [N,D]")
        if latents.ndim != 3:
            raise ValueError("latents must have shape [N,L,D]")
        if features.shape[0] != latents.shape[0]:
            raise ValueError("features and latents must contain the same N")
        if ids is not None and len(ids) != features.shape[0]:
            raise ValueError("ids must contain one entry per gallery sample")
        if latents.shape[1] != 14 or latents.shape[2] != 512:
            raise ValueError(
                "ProtectReID expects W+ latents with shape [N,14,512], "
                f"got {tuple(latents.shape)}"
            )

        self.features = F.normalize(features.float(), dim=-1).to(device)
        self.latents = latents.float().to(device)
        self.ids = list(ids) if ids is not None else None
        self.device = torch.device(device)
        self._faiss_index = None

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        feature_key: str = "gallery_f",
        latent_key: str = "gallery_e4e",
        id_key: str = "ids",
    ) -> "LatentGallery":
        """Load the official MATLAB gallery or this implementation's NPZ."""

        path = Path(path)
        if path.suffix.lower() == ".mat":
            try:
                from scipy.io import loadmat
            except ImportError as exc:
                raise ImportError("scipy is required to load a .mat gallery") from exc
            data = loadmat(path)
            features = data[feature_key]
            latents = data[latent_key]
            ids = None
        elif path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            features = data[feature_key]
            latents = data[latent_key]
            ids = data[id_key].tolist() if id_key in data else None
        elif path.suffix.lower() in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
            features = data.get(feature_key, data.get("features"))
            latents = data.get(latent_key, data.get("latents"))
            ids = data.get(id_key)
        else:
            raise ValueError("gallery must be .mat, .npz, .pt, or .pth")

        if features is None or latents is None:
            raise KeyError(f"gallery must contain {feature_key!r} and {latent_key!r}")
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return cls(torch.as_tensor(features), torch.as_tensor(latents), ids, device)

    def save_npz(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gallery_f": self.features.detach().cpu().numpy().astype(np.float32),
            "gallery_e4e": self.latents.detach().cpu().numpy().astype(np.float32),
        }
        if self.ids is not None:
            payload["ids"] = np.asarray(self.ids, dtype=str)
        np.savez(path, **payload)

    def _get_faiss_index(self):
        if self._faiss_index is not None:
            return self._faiss_index
        try:
            import faiss
        except ImportError:
            return None
        features = np.ascontiguousarray(self.features.detach().cpu().numpy(), dtype=np.float32)
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)
        self._faiss_index = index
        return index

    def find_exact_rows(self, query_features: torch.Tensor, atol: float = 1e-4):
        """Find exact source rows when a query is also present in an ID-less gallery."""

        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)
        queries = F.normalize(query_features.float(), dim=-1).to(self.device)
        scores = queries @ self.features.T
        return [torch.where(row >= 1.0 - atol)[0].tolist() for row in scores]

    def retrieve(
        self,
        query_features: torch.Tensor,
        k: int = 10,
        exclude_ids: Optional[Sequence[Optional[str]]] = None,
        exclude_indices: Optional[Sequence[Optional[Union[int, Sequence[int]]]]] = None,
    ) -> RetrievalResult:
        """Return descending top-k and ascending bottom-k neighbors.

        The exclusion is per image, not per identity: it removes the exact
        source gallery entry as required by Algorithm 1.  Pass ``exclude_ids``
        when gallery IDs are available, or ``exclude_indices`` when the caller
        already knows gallery row indices.
        """

        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)
        if query_features.ndim != 2 or query_features.shape[1] != self.features.shape[1]:
            raise ValueError("query_features must have shape [B,D]")
        if k < 1 or 2 * k > self.features.shape[0]:
            raise ValueError("gallery must contain at least 2*k entries")
        if exclude_ids is not None and len(exclude_ids) != query_features.shape[0]:
            raise ValueError("exclude_ids must have one entry per query")
        if exclude_indices is not None and len(exclude_indices) != query_features.shape[0]:
            raise ValueError("exclude_indices must have one entry per query")

        queries = F.normalize(query_features.float(), dim=-1).to(self.device)

        # With no per-query exclusion, use the same exact FAISS flat-IP search
        # as the paper. Negative queries return the bottom-k original scores.
        if exclude_ids is None and exclude_indices is None:
            index = self._get_faiss_index()
            if index is not None:
                query_np = np.ascontiguousarray(queries.detach().cpu().numpy(), dtype=np.float32)
                top_scores, top_indices = index.search(query_np, k)
                bottom_neg, bottom_indices = index.search(-query_np, k)
                return RetrievalResult(
                    torch.as_tensor(top_indices, device=self.device, dtype=torch.long),
                    torch.as_tensor(bottom_indices, device=self.device, dtype=torch.long),
                    torch.as_tensor(top_scores, device=self.device),
                    torch.as_tensor(-bottom_neg, device=self.device),
                )

        scores = queries @ self.features.T
        invalid = torch.zeros_like(scores, dtype=torch.bool)

        if exclude_ids is not None:
            if self.ids is None:
                raise ValueError("exclude_ids were supplied but gallery has no ids")
            id_to_rows = {}
            for row, sample_id in enumerate(self.ids):
                id_to_rows.setdefault(str(sample_id), []).append(row)
            for batch_row, sample_id in enumerate(exclude_ids):
                if sample_id is not None:
                    invalid[batch_row, id_to_rows.get(str(sample_id), [])] = True

        if exclude_indices is not None:
            for batch_row, rows in enumerate(exclude_indices):
                if rows is None:
                    continue
                if isinstance(rows, (int, np.integer)):
                    rows = [int(rows)]
                invalid[batch_row, list(rows)] = True

        available = (~invalid).sum(dim=1)
        if (available < 2 * k).any():
            raise ValueError("an exclusion mask leaves fewer than 2*k gallery entries")

        top_scores = scores.masked_fill(invalid, -torch.inf)
        top_scores, top_indices = torch.topk(top_scores, k=k, dim=1, largest=True, sorted=True)

        bottom_scores = scores.masked_fill(invalid, torch.inf)
        # top-k of the negative score is the bottom-k of the original score.
        bottom_neg, bottom_indices = torch.topk(-bottom_scores, k=k, dim=1, largest=True, sorted=True)
        bottom_scores = -bottom_neg
        return RetrievalResult(top_indices, bottom_indices, top_scores, bottom_scores)
