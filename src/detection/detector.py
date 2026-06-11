"""Real-time-style intrusion detection on new network traffic."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.models import NeuralNetworkIDS, RandomForestIDS, XGBoostIDS
from src.preprocessing import NSLKDDPreprocessor
from src.preprocessing.preprocessor import ATTACK_CATEGORIES, CATEGORY_COLORS

console = Console()

ATTACK_EMOJI = {
    "normal": "✅",
    "DoS": "💥",
    "Probe": "🔍",
    "R2L": "🔑",
    "U2R": "☠️",
    "attack": "⚠️",
    "Unknown": "❓",
}


def _load_model(key: str, task: str, config_path: str):
    ext = ".pt" if key == "nn" else ".pkl"
    path = f"models/saved/{key}_{task}{ext}"
    if key == "rf":
        return RandomForestIDS.load(path)
    if key == "xgb":
        return XGBoostIDS.load(path)
    return NeuralNetworkIDS.load(path, config_path)


def detect(
    input_path: str,
    model_key: str = "rf",
    output_path: str | None = None,
    task: str = "multiclass",
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """Run detection on a CSV of raw NSL-KDD format records."""
    console.rule("[bold cyan]Intrusion Detection — Live Scan")

    if not os.path.exists("models/saved/preprocessor.pkl"):
        console.print("[red]Preprocessor not found. Run training first.[/red]")
        return pd.DataFrame()

    prep = NSLKDDPreprocessor.load("models/saved/preprocessor.pkl")
    model = _load_model(model_key, task, config_path)

    df = pd.read_csv(input_path)
    console.print(f"[yellow]Scanning {len(df)} connections...[/yellow]")

    X = prep.transform(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidence = np.max(probs, axis=1)

    if task == "multiclass":
        label_names = prep.le_multi.classes_
    else:
        label_names = prep.le_binary.classes_

    pred_labels = label_names[preds]

    results = df.copy()
    results["predicted_category"] = pred_labels
    results["confidence"] = confidence
    results["is_attack"] = pred_labels != "normal"

    # Summary table
    tbl = Table(title=f"Detection Results — {model.name}", show_lines=True)
    tbl.add_column("#", style="dim")
    tbl.add_column("Protocol")
    tbl.add_column("Service")
    tbl.add_column("Category")
    tbl.add_column("Confidence", justify="right")
    tbl.add_column("Status")

    for i, row in results.head(20).iterrows():
        cat = row["predicted_category"]
        emoji = ATTACK_EMOJI.get(cat, "❓")
        status = "[red]ATTACK[/red]" if row["is_attack"] else "[green]NORMAL[/green]"
        tbl.add_row(
            str(i),
            str(row.get("protocol_type", "?")),
            str(row.get("service", "?")),
            f"{emoji} {cat}",
            f"{row['confidence']:.1%}",
            status,
        )

    console.print(tbl)

    attack_count = results["is_attack"].sum()
    console.print(f"\n[bold]Summary:[/bold] {attack_count}/{len(results)} attacks detected")

    if output_path:
        results.to_csv(output_path, index=False)
        console.print(f"[green]Results saved to {output_path}[/green]")

    return results


def simulate_traffic(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic NSL-KDD-like traffic for demo purposes."""
    rng = np.random.default_rng(seed)

    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "ftp", "smtp", "ssh", "dns", "private", "other"]
    flags = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2"]

    n = n_samples
    data = {
        "duration": rng.integers(0, 58329, n),
        "protocol_type": rng.choice(protocols, n),
        "service": rng.choice(services, n),
        "flag": rng.choice(flags, n),
        "src_bytes": rng.integers(0, 1379963888, n),
        "dst_bytes": rng.integers(0, 1309937401, n),
        "land": rng.integers(0, 2, n),
        "wrong_fragment": rng.integers(0, 3, n),
        "urgent": rng.integers(0, 14, n),
        "hot": rng.integers(0, 77, n),
        "num_failed_logins": rng.integers(0, 5, n),
        "logged_in": rng.integers(0, 2, n),
        "num_compromised": rng.integers(0, 884, n),
        "root_shell": rng.integers(0, 2, n),
        "su_attempted": rng.integers(0, 2, n),
        "num_root": rng.integers(0, 993, n),
        "num_file_creations": rng.integers(0, 28, n),
        "num_shells": rng.integers(0, 5, n),
        "num_access_files": rng.integers(0, 9, n),
        "num_outbound_cmds": rng.integers(0, 1, n),
        "is_host_login": rng.integers(0, 2, n),
        "is_guest_login": rng.integers(0, 2, n),
        "count": rng.integers(0, 511, n),
        "srv_count": rng.integers(0, 511, n),
        "serror_rate": rng.random(n),
        "srv_serror_rate": rng.random(n),
        "rerror_rate": rng.random(n),
        "srv_rerror_rate": rng.random(n),
        "same_srv_rate": rng.random(n),
        "diff_srv_rate": rng.random(n),
        "srv_diff_host_rate": rng.random(n),
        "dst_host_count": rng.integers(0, 256, n),
        "dst_host_srv_count": rng.integers(0, 256, n),
        "dst_host_same_srv_rate": rng.random(n),
        "dst_host_diff_srv_rate": rng.random(n),
        "dst_host_same_src_port_rate": rng.random(n),
        "dst_host_srv_diff_host_rate": rng.random(n),
        "dst_host_serror_rate": rng.random(n),
        "dst_host_srv_serror_rate": rng.random(n),
        "dst_host_rerror_rate": rng.random(n),
        "dst_host_srv_rerror_rate": rng.random(n),
    }
    return pd.DataFrame(data)
