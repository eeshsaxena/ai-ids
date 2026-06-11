"""Train all IDS models and persist results."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from src.models import NeuralNetworkIDS, RandomForestIDS, XGBoostIDS
from src.preprocessing import NSLKDDPreprocessor

console = Console()


def _model_path(name: str, task: str) -> str:
    ext = ".pt" if name == "nn" else ".pkl"
    return f"models/saved/{name}_{task}{ext}"


def train_models(
    model_filter: str = "all",
    task_filter: str = "both",
    config_path: str = "config/config.yaml",
) -> dict:
    """Train selected models on binary and/or multi-class tasks."""
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    console.rule("[bold cyan]AI Intrusion Detection System — Training")

    # Load & preprocess data
    console.print("[yellow]Loading and preprocessing NSL-KDD dataset...[/yellow]")
    prep = NSLKDDPreprocessor(config_path)
    train_df, test_df = prep.load_raw()
    data = prep.fit_transform(train_df, test_df)
    prep.save("models/saved/preprocessor.pkl")

    console.print(
        f"[green]Train: {data['X_train'].shape}  Test: {data['X_test'].shape}[/green]"
    )

    tasks = []
    if task_filter in ("binary", "both"):
        tasks.append(("binary", data["y_train_binary"], data["y_test_binary"], data["binary_classes"]))
    if task_filter in ("multiclass", "both"):
        tasks.append(("multiclass", data["y_train_multi"], data["y_test_multi"], data["multi_classes"]))

    model_keys = ["rf", "xgb", "nn"] if model_filter == "all" else [model_filter]
    all_results: dict = {}

    for task_name, y_train, y_test, class_names in tasks:
        n_classes = len(np.unique(y_train))
        console.rule(f"[bold]Task: {task_name}  ({n_classes} classes)")

        for key in model_keys:
            model = _build_model(key, data["n_features"], n_classes, config_path)
            console.print(f"\n[cyan]Training {model.name}...[/cyan]")
            t0 = time.time()
            model.fit(data["X_train"], y_train)
            elapsed = time.time() - t0
            console.print(f"  [dim]Trained in {elapsed:.1f}s[/dim]")

            metrics = model.evaluate(data["X_test"], y_test, list(class_names))
            _print_metrics(model.name, metrics)

            save_path = _model_path(key, task_name)
            if key == "nn":
                model.save(save_path)
            else:
                model.save(save_path)

            result_key = f"{key}_{task_name}"
            all_results[result_key] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "train_time_s": elapsed,
                "model_path": save_path,
            }

    # Persist summary
    results_path = "results/training_summary.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.rule("[bold green]Training complete")
    console.print(f"[green]Summary saved to {results_path}[/green]")
    return all_results


def _build_model(key: str, n_features: int, n_classes: int, config_path: str):
    if key == "rf":
        return RandomForestIDS(config_path)
    if key == "xgb":
        return XGBoostIDS(config_path)
    if key == "nn":
        return NeuralNetworkIDS(n_features, n_classes, config_path)
    raise ValueError(f"Unknown model key: {key}")


def _print_metrics(name: str, metrics: dict) -> None:
    tbl = Table(title=name, show_header=True)
    tbl.add_column("Metric")
    tbl.add_column("Score", justify="right")
    for k in ("accuracy", "precision", "recall", "f1"):
        tbl.add_row(k.capitalize(), f"{metrics[k]:.4f}")
    console.print(tbl)
