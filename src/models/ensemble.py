"""Soft-voting ensemble combining Random Forest, XGBoost, and Neural Network."""

from __future__ import annotations

import os

import numpy as np

from .base_model import BaseIDSModel
from .neural_network import NeuralNetworkIDS
from .random_forest_model import RandomForestIDS
from .xgboost_model import XGBoostIDS


class EnsembleIDS(BaseIDSModel):
    """Soft-voting ensemble: averages predicted probabilities of all 3 models."""

    name = "Ensemble (RF + XGB + NN)"

    def __init__(
        self,
        rf: RandomForestIDS,
        xgb: XGBoostIDS,
        nn: NeuralNetworkIDS,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.rf = rf
        self.xgb = xgb
        self.nn = nn
        self._weights = np.array(weights, dtype=float) / sum(weights)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all sub-models (or call individually and pass fitted models)."""
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.nn.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_rf = self.rf.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        p_nn = self.nn.predict_proba(X)
        return (
            self._weights[0] * p_rf
            + self._weights[1] * p_xgb
            + self._weights[2] * p_nn
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Factory: load all three pre-trained models and return ensemble
    # ------------------------------------------------------------------

    @classmethod
    def from_saved(
        cls,
        task: str = "multiclass",
        config_path: str = "config/config.yaml",
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> "EnsembleIDS":
        """Load saved RF, XGB, and NN models and wrap in an ensemble."""
        rf = RandomForestIDS.load(f"models/saved/rf_{task}.pkl")
        xgb = XGBoostIDS.load(f"models/saved/xgb_{task}.pkl")
        nn = NeuralNetworkIDS.load(f"models/saved/nn_{task}.pt", config_path)
        return cls(rf, xgb, nn, weights)
