from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA


@dataclass
class PCACurveProjector:
    mean_: np.ndarray
    components_: np.ndarray

    @classmethod
    def fit(cls, curves: np.ndarray, n_components: int = 3) -> "PCACurveProjector":
        components = min(n_components, curves.shape[0], curves.shape[1])
        pca = PCA(n_components=components)
        pca.fit(curves)
        return cls(mean_=pca.mean_.astype(np.float32), components_=pca.components_.astype(np.float32))

    def project_numpy(self, curves: np.ndarray) -> np.ndarray:
        centered = curves - self.mean_
        factors = centered @ self.components_.T
        return factors @ self.components_ + self.mean_

    def project_torch(self, curves: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean_, device=curves.device, dtype=curves.dtype)
        components = torch.as_tensor(self.components_, device=curves.device, dtype=curves.dtype)
        centered = curves - mean
        factors = centered @ components.t()
        return factors @ components + mean
