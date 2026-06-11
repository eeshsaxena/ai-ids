# 🛡️ AI-Based Intrusion Detection System (IDS)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-quality network intrusion detection system using **Random Forest**, **XGBoost**, and a **PyTorch Neural Network** trained on the [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) benchmark dataset.

---

## 🎯 What It Detects

| Category | Description | Example Attacks |
|----------|-------------|-----------------|
| **DoS** | Denial-of-Service — resource exhaustion | neptune, smurf, teardrop, pod |
| **Probe** | Port Scan — network topology discovery | ipsweep, nmap, portsweep, satan |
| **R2L** | Remote-to-Local / Brute Force — unauthorized remote access | guess_passwd, ftp_write, imap |
| **U2R** | User-to-Root — privilege escalation | buffer_overflow, rootkit, perl |
| **Normal** | Legitimate traffic | — |

---

## 🏗️ Architecture

```
NSL-KDD Dataset (125,973 train / 22,544 test)
         │
         ▼
 ┌───────────────────────────────────────────┐
 │           Preprocessing Pipeline          │
 │  • One-hot encode: protocol/service/flag  │
 │  • StandardScaler normalization           │
 │  • Binary (normal/attack) labels          │
 │  • Multi-class (5-category) labels        │
 └───────────────────────────────────────────┘
         │
    ┌────┴─────────────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌──────────────────┐
│ Random │ │XGBoost │ │ Neural Network   │
│ Forest │ │200 est.│ │ 256→128→64→32    │
│200 est.│ │LR=0.1  │ │ BatchNorm+Drop   │
└────────┘ └────────┘ └──────────────────┘
    │           │           │
    └─────┬─────┘           │
          ▼                 ▼
     Evaluation: Accuracy / Precision / Recall / F1 / AUC
     Confusion Matrix | ROC Curves | Feature Importance
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
```

### 3. Train All Models

```bash
python main.py train --model all --task both
```

This trains:
- Random Forest (binary + multiclass)
- XGBoost (binary + multiclass)
- Neural Network (binary + multiclass)

### 4. Evaluate & Generate Plots

```bash
python main.py evaluate
```

Outputs confusion matrices and ROC curves to `results/plots/`.

### 5. Launch Dashboard

```bash
python main.py dashboard
# or
streamlit run app/dashboard.py
```

### 6. Detect on New Data

```bash
# Generate synthetic traffic
python main.py simulate --n 500 --output data/test_traffic.csv

# Run detection
python main.py detect --input data/test_traffic.csv --model rf --task multiclass
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

> Results may vary slightly depending on system and random seed.

---

## 📁 Project Structure

```
ai-ids/
├── main.py                    # CLI entry point
├── requirements.txt
├── config/
│   └── config.yaml            # Hyperparameters & paths
├── data/
│   ├── download_data.py       # Auto-download NSL-KDD
│   ├── raw/                   # Raw .txt files
│   └── processed/             # Pickled train/test
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── base_model.py      # Abstract base
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   └── neural_network.py  # PyTorch deep network
│   ├── training/
│   │   └── trainer.py         # Orchestrate training
│   ├── evaluation/
│   │   └── evaluator.py       # Metrics + plots
│   └── detection/
│       └── detector.py        # Live detection + simulation
├── models/saved/              # Serialized model files
├── results/
│   └── plots/                 # Confusion matrices, ROC curves
├── app/
│   └── dashboard.py           # Streamlit UI
└── tests/
    ├── test_preprocessor.py
    ├── test_models.py
    └── test_detector.py
```

---

## 🔬 NSL-KDD Dataset

The **NSL-KDD** dataset is the cleaned version of the classic KDD Cup 1999 dataset — the standard benchmark for IDS research.

- **41 features** per network connection (duration, bytes, protocol, service, flags, etc.)
- **125,973** training samples, **22,544** test samples
- **5 attack categories**, **39 specific attack types**

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📖 Configuration

All hyperparameters live in `config/config.yaml`:

```yaml
models:
  random_forest:
    n_estimators: 200
    max_depth: 20

  xgboost:
    n_estimators: 200
    learning_rate: 0.1

  neural_network:
    hidden_layers: [256, 128, 64, 32]
    dropout_rate: 0.3
    epochs: 100
```

---

## 🖥️ Dashboard Features

| Page | Description |
|------|-------------|
| **Overview** | System summary, quick start guide |
| **Dataset Explorer** | Attack distributions, feature correlations |
| **Model Performance** | Side-by-side metric comparison, confusion matrices |
| **Live Detection** | Simulate & scan real-time network traffic |
| **Attack Analysis** | Threat intelligence, protocol heatmaps |

---

## 🛠️ CLI Reference

```bash
python main.py download                          # Download NSL-KDD
python main.py train --model [rf|xgb|nn|all]    # Train models
python main.py evaluate --model [rf|xgb|nn|all] # Evaluate + plots
python main.py detect --input FILE --model rf   # Detect on CSV
python main.py simulate --n 500                  # Generate test data
python main.py dashboard                         # Launch Streamlit
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — Canadian Institute for Cybersecurity
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/) for the dashboard
