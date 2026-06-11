#!/usr/bin/env python3
"""
AI-Based Intrusion Detection System
=====================================
Detects DoS, Port Scan, Brute Force, and other attacks using ML trained on NSL-KDD.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-ids",
        description="AI-Based Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py download
  python main.py train --model all --task both
  python main.py evaluate --model all
  python main.py detect --input data/sample.csv --model rf
  python main.py dashboard
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # download
    sub.add_parser("download", help="Download NSL-KDD dataset")

    # train
    t = sub.add_parser("train", help="Train ML models")
    t.add_argument("--model", choices=["rf", "xgb", "nn", "all"], default="all")
    t.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    t.add_argument("--cv", action="store_true", help="Run K-fold cross-validation (RF+XGB only)")

    # evaluate
    e = sub.add_parser("evaluate", help="Evaluate trained models and generate plots")
    e.add_argument("--model", choices=["rf", "xgb", "nn", "all"], default="all")

    # detect
    d = sub.add_parser("detect", help="Run detection on new CSV data")
    d.add_argument("--input", required=True, help="Input CSV (NSL-KDD format)")
    d.add_argument("--model", choices=["rf", "xgb", "nn", "ensemble"], default="rf")
    d.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    d.add_argument("--output", default=None, help="Output CSV path")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    # api
    a = sub.add_parser("api", help="Start FastAPI REST server")
    a.add_argument("--host", default="0.0.0.0")
    a.add_argument("--port", type=int, default=8000)

    # simulate
    s = sub.add_parser("simulate", help="Generate synthetic traffic CSV for testing")
    s.add_argument("--n", type=int, default=200, help="Number of packets")
    s.add_argument("--output", default="data/simulated_traffic.csv")

    args = parser.parse_args()

    if args.cmd == "download":
        from data.download_data import download_nsl_kdd
        download_nsl_kdd()

    elif args.cmd == "train":
        from src.training.trainer import train_models
        train_models(args.model, args.task, run_cv=getattr(args, "cv", False))

    elif args.cmd == "evaluate":
        from src.evaluation.evaluator import evaluate_models
        evaluate_models(args.model)

    elif args.cmd == "detect":
        from src.detection.detector import detect
        detect(args.input, args.model, args.output, args.task)

    elif args.cmd == "dashboard":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"],
            check=True,
        )

    elif args.cmd == "api":
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn", "app.server:app",
                "--host", args.host, "--port", str(args.port), "--reload",
            ],
            check=True,
        )

    elif args.cmd == "simulate":
        from src.detection.detector import simulate_traffic
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df = simulate_traffic(args.n)
        df.to_csv(args.output, index=False)
        print(f"[✓] Simulated {args.n} packets saved to {args.output}")


if __name__ == "__main__":
    main()
