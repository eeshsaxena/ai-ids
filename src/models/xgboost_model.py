"""XGBoost IDS model."""

import numpy as np
import yaml
from xgboost import XGBClassifier

from .base_model import BaseIDSModel


class XGBoostIDS(BaseIDSModel):
    name = "XGBoost"

    def __init__(self, config_path: str = "config/config.yaml", **kwargs):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["models"]["xgboost"]
        cfg.update(kwargs)
        # Remove keys XGBClassifier doesn't accept
        cfg.pop("use_label_encoder", None)
        self.model = XGBClassifier(**cfg, verbosity=0)
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str], top_n: int = 20) -> dict:
        if self.feature_importances_ is None:
            raise RuntimeError("Model not trained yet.")
        pairs = sorted(
            zip(feature_names, self.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        return {name: float(imp) for name, imp in pairs}
