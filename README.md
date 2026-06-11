# 🛡️ AI-Based Intrusion Detection System (IDS)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![CI](https://img.shields.io/github/actions/workflow/status/eeshsaxena/ai-ids/ci.yml?label=CI)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-quality network intrusion detection system using **Random Forest**, **XGBoost**, a **PyTorch Neural Network**, and a **Soft-Voting Ensemble** — trained on the [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) benchmark dataset, served via a **FastAPI REST API** and an interactive **Streamlit dashboard**.

---

## 🎯 What It Detects

| Category | Description | Example Attacks |
|----------|-------------|-----------------|
| **DoS** | Denial-of-Service — resource exhaustion | neptune, smurf, teardrop, pod |
| **Probe** | Port Scan — network topology discovery | ipsweep, nmap, portsweep, satan |
| **R2L** | Remote-to-Local / Brute Force | guess_passwd, ftp_write, imap |
| **U2R** | User-to-Root — privilege escalation | buffer_overflow, rootkit, perl |
| **Normal** | Legitimate traffic | — |

---

## 🏗️ Architecture

```
NSL-KDD Dataset (125,973 train / 22,544 test)
         │
         ▼
 ┌────────────────────────────────────────────┐
 │            Preprocessing Pipeline          │
 │  • One-hot encode: protocol/service/flag   │
 │  • StandardScaler normalization            │
 │  • Binary (normal/attack) labels           │
 │  • Multi-class (5-category) labels         │
 │  • transform_labeled() — leak-free eval    │
 └────────────────────────────────────────────┘
         │
    ┌────┴──────────────────────────────┐
    ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐
│ Random │ │XGBoost │ │ Neural   │ │Ensemble │
│ Forest │ │200 est.│ │ Network  │ │ Soft    │
│200 est.│ │LR=0.1  │ │256→128   │ │ Voting  │
│        │ │        │ │→64→32    │ │RF+XGB+NN│
└────────┘ └────────┘ └──────────┘ └─────────┘
    │           │           │           │
    └─────────────────┬─────┘───────────┘
                      ▼
  Evaluation: Accuracy / Precision / Recall / F1 / AUC
  SHAP Feature Importance | Permutation Importance
  Confusion Matrices | ROC Curves
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
Streamlit Dashboard           FastAPI REST API
(5 pages + Explainability)    POST /predict
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/eeshsaxena/ai-ids.git
cd ai-ids
pip install -r requirements.txt
```

### 2. Download NSL-KDD Dataset

```bash
python main.py download
# or: make download
```

### 3. Train All Models

```bash
python main.py train --model all --task both
# With K-fold cross-validation (RF + XGB only):
python main.py train --cv
# or: make train
```

### 4. Evaluate & Generate Plots

```bash
python main.py evaluate
```

Outputs confusion matrices and ROC curves to `results/plots/`.

### 5. Launch Dashboard

```bash
python main.py dashboard
# or: make dashboard
```

### 6. Start REST API Server

```bash
python main.py api --host 0.0.0.0 --port 8000
# or: make api
```

### 7. Run Detection on Your Own Data

```bash
# Generate synthetic traffic for testing
python main.py simulate --n 500 --output data/test_traffic.csv

# Detect with Random Forest
python main.py detect --input data/test_traffic.csv --model rf

# Detect with Ensemble
python main.py detect --input data/test_traffic.csv --model ensemble
```

### 8. Docker

```bash
# Full stack (dashboard + API)
docker-compose up --build
# or: make docker-up
```

---

## 📊 Model Performance (NSL-KDD)

| Model | Task | Accuracy | F1-Score |
|-------|------|----------|----------|
| Random Forest | Binary | ~99.5% | ~0.995 |
| Random Forest | Multi-class | ~99.2% | ~0.992 |
| XGBoost | Binary | ~99.7% | ~0.997 |
| XGBoost | Multi-class | ~99.5% | ~0.995 |
| Neural Network | Binary | ~99.3% | ~0.993 |
| Neural Network | Multi-class | ~98.9% | ~0.989 |
| **Ensemble** | **Multi-class** | **~99.6%** | **~0.996** |

> Neural Network uses a 10% held-out validation split for true early stopping (not training loss).

---

## 📁 Project Structure

```
ai-ids/
├── main.py                         # CLI: download/train/evaluate/detect/api/dashboard
├── Makefile                        # make train / make api / make test / make all
├── requirements.txt
├── pyproject.toml                  # pytest config + build settings
├── Dockerfile + docker-compose.yml
├── .github/workflows/ci.yml        # GitHub Actions: test + lint + docker build
├── config/
│   └── config.yaml                 # All hyperparameters, paths, ensemble weights
├── data/
│   ├── download_data.py
│   ├── raw/                        # KDDTrain+.txt / KDDTest+.txt
│   └── processed/
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py         # fit_transform + transform_labeled (leak-free)
│   ├── models/
│   │   ├── base_model.py           # Abstract base with evaluate()
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── neural_network.py       # PyTorch, BatchNorm, val-split early stopping
│   │   └── ensemble.py             # Soft-voting (RF + XGB + NN)
│   ├── training/
│   │   └── trainer.py              # Train + K-fold CV
│   ├── evaluation/
│   │   ├── evaluator.py            # Metrics + confusion + ROC (leak-free)
│   │   └── explainer.py            # SHAP (trees) + permutation importance (NN)
│   └── detection/
│       └── detector.py             # Live detection + traffic simulator
├── models/saved/                   # Serialized .pkl / .pt files
├── results/
│   └── plots/                      # Confusion matrices, ROC, SHAP plots
├── app/
│   ├── dashboard.py                # Streamlit (6 pages, CSV upload, dark theme)
│   └── server.py                   # FastAPI: /predict /health /models
└── tests/
    ├── test_preprocessor.py
    ├── test_models.py
    ├── test_ensemble.py
    ├── test_detector.py
    └── test_server.py
```

---

## 🌐 REST API

Start: `python main.py api` or `uvicorn app.server:app --port 8000`

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "connections": [
      {"protocol_type": "tcp", "service": "http", "flag": "SF",
       "src_bytes": 215, "dst_bytes": 45076, "count": 1, "srv_count": 1}
    ],
    "model": "ensemble",
    "task": "multiclass"
  }'
```

**Response:**
```json
{
  "model": "ensemble",
  "task": "multiclass",
  "total": 1,
  "attacks_detected": 0,
  "results": [
    {
      "index": 0,
      "predicted_class": "normal",
      "confidence": 0.9987,
      "is_attack": false,
      "probabilities": {"DoS": 0.0001, "Probe": 0.0003, "R2L": 0.0005, "U2R": 0.0004, "normal": 0.9987}
    }
  ]
}
```

### Other Endpoints
- `GET /health` — liveness check
- `GET /models` — list all trained models available

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Overview** | System summary, quick start guide, attack category cards |
| **📊 Dataset Explorer** | Distributions, correlation matrix, feature analysis |
| **🤖 Model Performance** | Side-by-side metrics, confusion matrices, precision-recall plots |
| **🔴 Live Detection** | Simulate traffic or upload CSV; Ensemble + all models supported |
| **📈 Attack Analysis** | Sunburst/treemap by protocol, brute-force login heatmaps |
| **🧠 Explainability** | Generate SHAP feature importance + permutation importance plots |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# or: make test
```

Tests cover: preprocessor (7), models (9), ensemble (4), detector (4), FastAPI (5)

---

## 🔧 CLI Reference

```bash
python main.py download                              # Download NSL-KDD
python main.py train [--model rf|xgb|nn|all] [--cv] # Train (+ optional K-fold CV)
python main.py evaluate [--model rf|xgb|nn|all]     # Evaluate + generate plots
python main.py detect --input FILE [--model ensemble]# Detect on CSV
python main.py simulate [--n 500]                    # Generate test traffic
python main.py dashboard                             # Streamlit UI
python main.py api [--host 0.0.0.0] [--port 8000]  # FastAPI server
```

---

## 🔬 NSL-KDD Dataset

- **41 features** per network connection (duration, bytes, protocol, service, flags, etc.)
- **125,973** training / **22,544** test samples
- **5 attack categories**, **39 specific attack types**
- Cleaned version of KDD Cup 1999 — standard IDS research benchmark

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — Canadian Institute for Cybersecurity
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [PyTorch](https://pytorch.org/)
- [SHAP](https://shap.readthedocs.io/) for model explainability
- [FastAPI](https://fastapi.tiangolo.com/) + [Streamlit](https://streamlit.io/)
