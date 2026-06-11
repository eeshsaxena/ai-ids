"""Evaluate trained IDS models and generate reports."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.models import NeuralNetworkIDS, RandomForestIDS, XGBoostIDS
from src.preprocessing import NSLKDDPreprocessor


def load_model(key: str, task: str, config_path: str = "config/config.yaml"):
    ext = ".pt" if key == "nn" else ".pkl"
    path = f"models/saved/{key}_{task}{ext}"
    if key == "rf":
        return RandomForestIDS.load(path)
    if key == "xgb":
        return XGBoostIDS.load(path)
    if key == "nn":
        return NeuralNetworkIDS.load(path, config_path)
    raise ValueError(key)


def evaluate_models(
    model_filter: str = "all",
    config_path: str = "config/config.yaml",
) -> dict:
    os.makedirs("results/plots", exist_ok=True)

    prep = NSLKDDPreprocessor.load("models/saved/preprocessor.pkl")
    train_df, test_df = prep.load_raw()
    data = prep.fit_transform(train_df, test_df)

    keys = ["rf", "xgb", "nn"] if model_filter == "all" else [model_filter]
    all_results: dict = {}

    for task, y_test, class_names in [
        ("binary", data["y_test_binary"], data["binary_classes"]),
        ("multiclass", data["y_test_multi"], data["multi_classes"]),
    ]:
        for key in keys:
            model_path = f"models/saved/{key}_{task}.{'pt' if key == 'nn' else 'pkl'}"
            if not os.path.exists(model_path):
                continue

            model = load_model(key, task, config_path)
            metrics = model.evaluate(data["X_test"], y_test, list(class_names))
            probs = model.predict_proba(data["X_test"])

            # Confusion matrix
            _plot_confusion_matrix(
                y_test, metrics["predictions"], class_names,
                title=f"{model.name} — {task}",
                save_path=f"results/plots/{key}_{task}_confusion.png",
            )

            # ROC curves
            _plot_roc(
                y_test, probs, class_names,
                title=f"{model.name} — {task} ROC",
                save_path=f"results/plots/{key}_{task}_roc.png",
            )

            result_key = f"{key}_{task}"
            all_results[result_key] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "report": metrics["report"],
            }
            print(f"\n[{key.upper()} | {task}]")
            print(metrics["report"])

    with open("results/evaluation_results.json", "w") as f:
        # report is not JSON-serializable as-is; store other fields
        serializable = {
            k: {m: v for m, v in d.items() if m != "report"}
            for k, d in all_results.items()
        }
        json.dump(serializable, f, indent=2)

    return all_results


def _plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.4), max(5, len(class_names) * 1.2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_roc(y_true, probs, class_names, title, save_path):
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        auc = roc_auc_score(y_true, probs[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc = roc_auc_score(y_bin[:, i], probs[:, i])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
