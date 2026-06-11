"""PyTorch Neural Network IDS model."""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseIDSModel


class _IDSNet(nn.Module):
    def __init__(self, in_features: int, hidden: list[int], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralNetworkIDS(BaseIDSModel):
    name = "Neural Network"

    def __init__(self, n_features: int, n_classes: int, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["models"]["neural_network"]

        self.cfg = cfg
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = _IDSNet(
            in_features=n_features,
            hidden=cfg["hidden_layers"],
            n_classes=n_classes,
            dropout=cfg["dropout_rate"],
        ).to(self.device)

        self._classes: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._classes = np.unique(y)

        # Use 10% of training data as a held-out validation set for early stopping
        val_ratio = self.cfg.get("validation_split", 0.1)
        n_val = max(1, int(len(X) * val_ratio))
        idx = np.random.default_rng(42).permutation(len(X))
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
            ),
            batch_size=self.cfg["batch_size"],
            shuffle=True,
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(self.cfg["learning_rate"]),
            weight_decay=float(self.cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.cfg["epochs"]):
            self.net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()

            # Evaluate on validation split (no gradient)
            self.net.eval()
            with torch.no_grad():
                val_loss = criterion(self.net(X_val_t), y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.cfg["early_stopping_patience"]:
                    break

        if hasattr(self, "_best_state"):
            self.net.load_state_dict(self._best_state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(Xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Persistence — override base to handle torch state dict
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save net state dict separately; pickle the rest
        state = {
            "cfg": self.cfg,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "net_state": self.net.state_dict(),
            "_classes": self._classes,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, config_path: str = "config/config.yaml") -> "NeuralNetworkIDS":
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            n_features=state["n_features"],
            n_classes=state["n_classes"],
            config_path=config_path,
        )
        obj.net.load_state_dict(state["net_state"])
        obj._classes = state["_classes"]
        return obj
