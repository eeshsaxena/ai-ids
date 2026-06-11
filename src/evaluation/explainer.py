"""SHAP-based model explainability for IDS models."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def explain_rf_xgb(
    model,
    X: np.ndarray,
    feature_names: list[str],
    title: str = "Feature Importance",
    save_dir: str = "results/plots",
    top_n: int = 20,
) -> None:
    """Generate SHAP summary plot for tree-based models (RF / XGBoost)."""
    try:
        import shap
    except ImportError:
        print("[!] shap not installed. Run: pip install shap")
        return

    os.makedirs(save_dir, exist_ok=True)

    # TreeExplainer is fast and exact for tree models
    underlying = getattr(model, "model", model)  # unwrap our wrapper
    explainer = shap.TreeExplainer(underlying)

    sample = X[:min(2000, len(X))]
    shap_values = explainer.shap_values(sample)

    # Multi-class: shap_values is a list; take mean abs across classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values)

    importance = mean_abs.mean(axis=0)
    idx = np.argsort(importance)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importance[idx[::-1]],
        color="#3498db",
    )
    ax.set_title(f"SHAP Feature Importance — {title}")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    path = os.path.join(save_dir, f"shap_{title.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] SHAP plot saved: {path}")

    # SHAP summary (beeswarm) for top features
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        mean_abs[:, idx] if isinstance(shap_values, list) else shap_values[:, idx],
        sample[:, idx],
        feature_names=[feature_names[i] for i in idx],
        show=False,
        plot_size=(10, 8),
    )
    path2 = os.path.join(save_dir, f"shap_summary_{title.lower().replace(' ', '_')}.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP summary plot saved: {path2}")


def explain_nn(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    title: str = "Neural Network",
    save_dir: str = "results/plots",
    top_n: int = 20,
) -> None:
    """Generate permutation importance for neural network (SHAP DeepExplainer requires GPU for large data)."""
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("[!] sklearn not available for permutation importance")
        return

    os.makedirs(save_dir, exist_ok=True)

    class _Wrapper:
        """sklearn-compatible wrapper for our NN."""
        def __init__(self, nn): self.nn = nn
        def predict(self, X): return self.nn.predict(X)
        def score(self, X, y):
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, self.predict(X))

    # Use a subset for speed
    n = min(1000, len(X_explain))
    X_sub = X_explain[:n]

    # Get true labels from predictions (unsupervised — use model's own preds as reference)
    y_ref = model.predict(X_sub)

    from sklearn.inspection import permutation_importance as pi
    wrapper = _Wrapper(model)

    result = pi(wrapper, X_sub, y_ref, n_repeats=5, random_state=42, n_jobs=-1)
    importance = result.importances_mean
    idx = np.argsort(importance)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importance[idx[::-1]],
        color="#9b59b6",
    )
    ax.set_title(f"Permutation Importance — {title}")
    ax.set_xlabel("Mean accuracy decrease")
    plt.tight_layout()
    path = os.path.join(save_dir, f"perm_importance_{title.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Permutation importance plot saved: {path}")
