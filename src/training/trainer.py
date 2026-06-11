"""Train all IDS models, with optional K-fold cross-validation."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
    run_cv: bool = False,
) -> dict:
    """Train selected models on binary and/or multi-class tasks.

    Args:
        model_filter: 'rf', 'xgb', 'nn', or 'all'
        task_filter: 'binary', 'multiclass', or 'both'
        config_path: path to config YAML
        run_cv: if True, run K-fold cross-validation on RF and XGBoost
    """
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    console.rule("[bold cyan]AI Intrusion Detection System — Training")

    console.print("[yellow]Loading and preprocessing NSL-KDD dataset...[/yellow]")
    prep = NSLKDDPreprocessor(config_path)
    train_df, test_df = prep.load_raw()
    data = prep.fit_transform(train_df, test_df)
    prep.save("models/saved/preprocessor.pkl")

    console.print(
        f"[green]Train: {data['X_train'].shape}  Test: {data['X_test'].shape}[/green]"
    )
    console.print(
        f"Binary classes: {data['binary_classes']}  |  "
        f"Multi classes: {data['multi_classes']}"
    )

    tasks = []
    if task_filter in ("binary", "both"):
        tasks.append(("binary", data["y_train_binary"], data["y_test_binary"], data["binary_classes"]))
    if task_filter in ("multiclass", "both"):
        tasks.append(("multiclass", data["y_train_multi"], data["y_test_multi"], data["multi_classes"]))

    model_keys = ["rf", "xgb", "nn"] if model_filter == "all" else [model_filter]
    all_results: dict = {}

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    n_cv_folds = cfg.get("training", {}).get("cv_folds", 5)

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

            # Use frozen preprocessor's transform_labeled for honest test evaluation
            labeled = prep.transform_labeled(test_df)
            if task_name == "binary":
                X_eval, y_eval = labeled["X_binary"], labeled["y_binary"]
            else:
                X_eval, y_eval = labeled["X_multi"], labeled["y_multi"]

            metrics = model.evaluate(X_eval, y_eval, list(class_names))
            _print_metrics(model.name, metrics)

            cv_score = None
            if run_cv and key in ("rf", "xgb"):
                console.print(f"  [yellow]Running {n_cv_folds}-fold CV...[/yellow]")
                cv_scores = cross_val_score(
                    model.model, data["X_train"], y_train,
                    cv=StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42),
                    scoring="f1_weighted",
                    n_jobs=-1,
                )
                cv_score = float(cv_scores.mean())
                console.print(
                    f"  [green]CV F1: {cv_score:.4f} ± {cv_scores.std():.4f}[/green]"
                )

            save_path = _model_path(key, task_name)
            model.save(save_path)

            result = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "train_time_s": elapsed,
                "model_path": save_path,
            }
            if cv_score is not None:
                result["cv_f1"] = cv_score

            all_results[f"{key}_{task_name}"] = result

    results_path = "results/training_summary.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.rule("[bold green]Training complete")
    console.print(f"[green]Summary → {results_path}[/green]")
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
