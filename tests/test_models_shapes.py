from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.data.csv_panel_loader import load_panel, load_underlying_csv
from src.data.feature_engineering import build_sequence_dataset
from src.models.ar1_per_grid import AR1PerGridModel
from src.models.elastic_net_baseline import ElasticNetBaseline
from src.models.extra_trees_baseline import ExtraTreesBaseline
from src.models.factor_ar_var import FactorARVARModel
from src.models.garch_baseline import GARCHStyleBaseline
from src.models.har_factor_baseline import HARFactorBaseline
from src.models.hist_gradient_boosting_baseline import HistGradientBoostingBaseline
from src.models.lstm_curve import LSTMCurveForecaster
from src.models.mlp_baseline import MLPBaseline
from src.models.persistence import PersistenceModel
from src.models.smile_coefficient_baseline import SmileCoefficientBaseline
from src.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


class ModelShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_yaml_config(ROOT / "configs" / "data_spy_daily.yaml")
        panel = load_panel(ROOT / "data" / "samples" / "spy_iv_panel_sample.csv")
        underlying = load_underlying_csv(ROOT / "data" / "samples" / "spy_underlying_sample.csv")
        cls.bundle = build_sequence_dataset(panel, underlying, config)
        cls.X = cls.bundle.X[:20]
        cls.y = cls.bundle.y[:20]
        cls.grid_size = len(cls.bundle.curve_columns)
        cls.atm_index = cls.bundle.curve_columns.index(cls.bundle.metadata["atm_column"])

    def test_baselines_return_curve_shaped_output(self) -> None:
        models = [
            PersistenceModel(grid_size=self.grid_size),
            AR1PerGridModel(grid_size=self.grid_size),
            FactorARVARModel(grid_size=self.grid_size, n_factors=3, mode="var"),
            GARCHStyleBaseline(grid_size=self.grid_size, atm_index=self.atm_index),
            MLPBaseline(hidden_layer_sizes=(16,), max_iter=50, random_state=7),
            ElasticNetBaseline(alpha=1e-3, l1_ratio=0.5, max_iter=2000, random_state=7),
            ExtraTreesBaseline(n_estimators=20, min_samples_leaf=1, random_state=7, n_jobs=1),
            HistGradientBoostingBaseline(max_iter=20, max_depth=3, min_samples_leaf=2, random_state=7),
            HARFactorBaseline(grid_size=self.grid_size, n_factors=3, windows=(1, 3, 5), ridge_alpha=1.0),
            SmileCoefficientBaseline(
                moneyness_grid=list(self.bundle.metadata["moneyness_grid"]),
                degree=3,
                ridge_alpha=1e-3,
                windows=(1, 3, 5),
            ),
        ]
        for model in models:
            model.fit(self.X, self.y)
            pred = model.predict(self.X[:5])
            self.assertEqual(pred.shape, (5, self.grid_size))

    def test_lstm_forward_shape(self) -> None:
        model = LSTMCurveForecaster(
            input_size=self.X.shape[-1],
            output_size=self.grid_size,
            hidden_size=16,
            num_layers=1,
            dropout=0.1,
        )
        pred = model(torch.tensor(self.X[:4], dtype=torch.float32))
        self.assertEqual(tuple(pred.shape), (4, self.grid_size))


if __name__ == "__main__":
    unittest.main()
