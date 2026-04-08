from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HARFactorBaseline:
    def __init__(
        self,
        grid_size: int,
        n_factors: int = 3,
        windows: tuple[int, ...] = (1, 6, 24),
        ridge_alpha: float = 1.0,
        use_exogenous: bool = True,
    ):
        self.grid_size = grid_size
        self.n_factors = n_factors
        self.windows = windows
        self.ridge_alpha = ridge_alpha
        self.use_exogenous = use_exogenous
        self.pca: PCA | None = None
        self.regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ]
        )

    def _build_features(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("HARFactorBaseline must be fitted before features can be built.")
        curve_seq = X[:, :, : self.grid_size]
        factors = self.pca.transform(curve_seq.reshape(-1, self.grid_size)).reshape(
            curve_seq.shape[0], curve_seq.shape[1], -1
        )
        har_parts: list[np.ndarray] = []
        for window in self.windows:
            width = max(1, min(int(window), factors.shape[1]))
            har_parts.append(factors[:, -width:, :].mean(axis=1))
        features = np.concatenate(har_parts, axis=1)
        if self.use_exogenous and X.shape[2] > self.grid_size:
            features = np.concatenate([features, X[:, -1, self.grid_size :]], axis=1)
        return features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HARFactorBaseline":
        curve_seq = X[:, :, : self.grid_size]
        components = min(self.n_factors, self.grid_size, max(1, len(y)))
        self.pca = PCA(n_components=components)
        self.pca.fit(np.vstack([curve_seq.reshape(-1, self.grid_size), y]))
        X_features = self._build_features(X)
        y_factors = self.pca.transform(y)
        self.regressor.fit(X_features, y_factors)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("HARFactorBaseline must be fitted before predict().")
        X_features = self._build_features(X)
        pred_factors = self.regressor.predict(X_features)
        return self.pca.inverse_transform(pred_factors)
