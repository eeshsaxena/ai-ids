"""Abstract base class for all IDS models."""

import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


class BaseIDSModel(ABC):
    """Base interface every model must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def evaluate(self, X: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted", zero_division=0),
            "recall": recall_score(y, preds, average="weighted", zero_division=0),
            "f1": f1_score(y, preds, average="weighted", zero_division=0),
            "report": classification_report(
                y, preds, target_names=class_names, zero_division=0
            ),
            "predictions": preds,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseIDSModel":
        with open(path, "rb") as f:
            return pickle.load(f)
