"""Streamlit dashboard for the AI Intrusion Detection System."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a5298;
    }
    .attack-badge-dos   { color: #e74c3c; font-weight: bold; }
    .attack-badge-probe { color: #f39c12; font-weight: bold; }
    .attack-badge-r2l   { color: #9b59b6; font-weight: bold; }
    .attack-badge-u2r   { color: #e67e22; font-weight: bold; }
    .attack-badge-normal{ color: #2ecc71; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------
st.sidebar.image(
    "https://img.shields.io/badge/AI-IDS-blue?style=for-the-badge&logo=shield",
    use_container_width=True,
)
st.sidebar.title("🛡️ AI-IDS")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Dataset Explorer", "🤖 Model Performance",
     "🔴 Live Detection", "📈 Attack Analysis"],
)

CATEGORY_COLORS = {
    "normal": "#2ecc71",
    "DoS": "#e74c3c",
    "Probe": "#f39c12",
    "R2L": "#9b59b6",
    "U2R": "#e67e22",
    "attack": "#e74c3c",
    "Unknown": "#95a5a6",
}


# -----------------------------------------------------------------------
# Helper: load results
# -----------------------------------------------------------------------
@st.cache_data
def load_results() -> dict:
    path = ROOT / "results" / "training_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset():
    """Load raw NSL-KDD if available."""
    train_path = ROOT / "data" / "raw" / "KDDTrain+.txt"
    if not train_path.exists():
        return None, None

    COLUMNS = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty",
    ]
    ATTACK_MAP = {
        "normal": "normal",
        "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
        "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
        "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
        "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
        "satan": "Probe", "mscan": "Probe", "saint": "Probe",
        "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
        "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
        "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
        "snmpgetattack": "R2L", "snmpguess": "R2L", "httptunnel": "R2L",
        "xlock": "R2L", "xsnoop": "R2L",
        "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
        "rootkit": "U2R", "xterm": "U2R", "sqlattack": "U2R", "ps": "U2R",
    }
    train = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_path = ROOT / "data" / "raw" / "KDDTest+.txt"
    test = pd.read_csv(test_path, header=None, names=COLUMNS) if test_path.exists() else None

    train["attack_category"] = train["label"].map(lambda x: ATTACK_MAP.get(x, "Unknown"))
    if test is not None:
        test["attack_category"] = test["label"].map(lambda x: ATTACK_MAP.get(x, "Unknown"))
    return train, test


# -----------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------

if page == "🏠 Overview":
    st.title("🛡️ AI-Based Intrusion Detection System")
    st.markdown(
        "> Detecting network intrusions in real-time using **Random Forest**, "
        "**XGBoost**, and **Neural Networks** trained on the NSL-KDD dataset."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    results = load_results()
    best_f1 = max((v["f1"] for v in results.values()), default=None)
    trained = len(results)

    c1.metric("Models Trained", trained, help="Number of model-task combinations trained")
    c2.metric("Best F1-Score", f"{best_f1:.4f}" if best_f1 else "—")
    c3.metric("Attack Classes", "5", help="normal, DoS, Probe, R2L, U2R")
    c4.metric("Dataset", "NSL-KDD", help="Industry-standard IDS benchmark dataset")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Attack Categories Detected")
        attack_info = {
            "DoS": ("💥", "Denial of Service — floods the target to exhaust resources"),
            "Probe": ("🔍", "Port Scan — maps network topology and open services"),
            "R2L": ("🔑", "Remote-to-Local / Brute Force — unauthorized remote access"),
            "U2R": ("☠️", "User-to-Root — privilege escalation attacks"),
            "normal": ("✅", "Legitimate traffic — no attack detected"),
        }
        for cat, (icon, desc) in attack_info.items():
            st.markdown(f"**{icon} {cat}** — {desc}")

    with col_r:
        st.subheader("ML Pipeline")
        st.markdown(
            """
            ```
            NSL-KDD Dataset (125,973 train / 22,544 test)
                    ↓
            Preprocessing
            • One-hot encode protocol/service/flag
            • StandardScaler normalization
            • Binary + Multi-class label encoding
                    ↓
            ┌──────────────┬──────────────┬────────────────┐
            │ Random Forest│   XGBoost    │ Neural Network │
            │  200 trees   │  200 rounds  │  256→128→64→32 │
            └──────────────┴──────────────┴────────────────┘
                    ↓
            Evaluation: Accuracy / Precision / Recall / F1
            Confusion Matrix / ROC AUC
            ```
            """
        )

    st.divider()
    st.subheader("Quick Start")
    st.code(
        """# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NSL-KDD dataset
python main.py download

# 3. Train all models
python main.py train --model all --task both

# 4. Evaluate
python main.py evaluate

# 5. Launch this dashboard
streamlit run app/dashboard.py""",
        language="bash",
    )


elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer — NSL-KDD")
    train_df, test_df = load_dataset()

    if train_df is None:
        st.warning("Dataset not found. Run `python main.py download` first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Overview", "Attack Distribution", "Feature Analysis"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Train Samples", f"{len(train_df):,}")
            c2.metric("Test Samples", f"{len(test_df):,}" if test_df is not None else "—")
            c3.metric("Features", "41")

            st.subheader("Sample Records")
            st.dataframe(train_df.head(100), use_container_width=True, height=300)

        with tab2:
            cat_counts = train_df["attack_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            cat_counts["Color"] = cat_counts["Category"].map(CATEGORY_COLORS)

            fig = px.bar(
                cat_counts, x="Category", y="Count",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Attack Category Distribution (Train Set)",
                text="Count",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.pie(
                cat_counts, values="Count", names="Category",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Proportion of Attack Categories",
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Top 15 Specific Attack Types")
            top_attacks = (
                train_df[train_df["label"] != "normal"]["label"]
                .value_counts()
                .head(15)
                .reset_index()
            )
            top_attacks.columns = ["Attack", "Count"]
            fig3 = px.bar(top_attacks, x="Count", y="Attack", orientation="h",
                          title="Top Attack Types", color="Count",
                          color_continuous_scale="Reds")
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.subheader("Feature Correlations (numeric features)")
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ["difficulty"]]
            sample = train_df[numeric_cols[:20]].sample(min(5000, len(train_df)), random_state=42)
            corr = sample.corr()
            fig4 = px.imshow(
                corr, title="Correlation Matrix (top 20 numeric features)",
                color_continuous_scale="RdBu", aspect="auto",
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader("Feature Distribution")
            feat = st.selectbox("Select feature", numeric_cols)
            fig5 = px.histogram(
                train_df.sample(min(5000, len(train_df)), random_state=42),
                x=feat, color="attack_category",
                color_discrete_map=CATEGORY_COLORS,
                title=f"Distribution of {feat} by Attack Category",
                marginal="rug", nbins=50,
            )
            st.plotly_chart(fig5, use_container_width=True)


elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Comparison")
    results = load_results()

    if not results:
        st.warning("No trained models found. Run `python main.py train` first.")
    else:
        # Build comparison dataframe
        rows = []
        for k, v in results.items():
            parts = k.split("_", 1)
            model_key, task = parts[0], parts[1] if len(parts) > 1 else "unknown"
            model_name = {"rf": "Random Forest", "xgb": "XGBoost", "nn": "Neural Network"}.get(model_key, model_key)
            rows.append({
                "Model": model_name,
                "Task": task,
                "Accuracy": v["accuracy"],
                "Precision": v["precision"],
                "Recall": v["recall"],
                "F1-Score": v["f1"],
                "Train Time (s)": v.get("train_time_s", 0),
            })
        df_results = pd.DataFrame(rows)

        tab1, tab2, tab3 = st.tabs(["Metrics Table", "Visual Comparison", "Confusion Matrices"])

        with tab1:
            st.dataframe(
                df_results.style.format({
                    "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                    "Recall": "{:.4f}", "F1-Score": "{:.4f}",
                    "Train Time (s)": "{:.1f}",
                }).background_gradient(subset=["Accuracy", "F1-Score"], cmap="Greens"),
                use_container_width=True,
            )

        with tab2:
            task_filter = st.selectbox("Task", ["binary", "multiclass", "all"])
            df_plot = df_results if task_filter == "all" else df_results[df_results["Task"] == task_filter]

            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
            df_melted = df_plot.melt(
                id_vars=["Model", "Task"],
                value_vars=metrics,
                var_name="Metric", value_name="Score",
            )
            fig = px.bar(
                df_melted, x="Metric", y="Score", color="Model",
                barmode="group", title=f"Model Comparison ({task_filter})",
                text_auto=".3f",
            )
            fig.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.scatter(
                df_plot, x="Recall", y="Precision", color="Model",
                size="F1-Score", hover_data=["Task", "Accuracy"],
                title="Precision vs Recall (bubble size = F1)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            plot_dir = ROOT / "results" / "plots"
            plot_files = list(plot_dir.glob("*_confusion.png"))
            if not plot_files:
                st.info("Run `python main.py evaluate` to generate confusion matrices.")
            else:
                cols = st.columns(min(len(plot_files), 2))
                for i, p in enumerate(plot_files):
                    cols[i % 2].image(str(p), caption=p.stem, use_container_width=True)


elif page == "🔴 Live Detection":
    st.title("🔴 Live Intrusion Detection")
    st.markdown("Simulate real-time network traffic scanning.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        model_choice = st.selectbox("Model", ["Random Forest", "XGBoost", "Neural Network"])
        task_choice = st.selectbox("Task", ["multiclass", "binary"])
        n_packets = st.slider("Packets to simulate", 10, 500, 100)
        auto_refresh = st.toggle("Auto-refresh (every 3s)", value=False)

        model_map = {"Random Forest": "rf", "XGBoost": "xgb", "Neural Network": "nn"}
        model_key = model_map[model_choice]

        run_btn = st.button("▶ Run Detection", type="primary", use_container_width=True)

    with col2:
        st.subheader("Detection Stream")
        placeholder = st.empty()

        def run_detection():
            prep_path = ROOT / "models" / "saved" / "preprocessor.pkl"
            ext = ".pt" if model_key == "nn" else ".pkl"
            model_path = ROOT / "models" / "saved" / f"{model_key}_{task_choice}{ext}"

            if not prep_path.exists() or not model_path.exists():
                placeholder.error("Models not found. Run training first.")
                return

            from src.detection.detector import simulate_traffic
            from src.preprocessing import NSLKDDPreprocessor

            prep = NSLKDDPreprocessor.load(str(prep_path))

            if model_key == "rf":
                from src.models import RandomForestIDS
                model = RandomForestIDS.load(str(model_path))
            elif model_key == "xgb":
                from src.models import XGBoostIDS
                model = XGBoostIDS.load(str(model_path))
            else:
                from src.models import NeuralNetworkIDS
                model = NeuralNetworkIDS.load(str(model_path))

            df = simulate_traffic(n_packets)
            X = prep.transform(df)
            preds = model.predict(X)
            probs = model.predict_proba(X)
            confidence = np.max(probs, axis=1)

            classes = prep.le_multi.classes_ if task_choice == "multiclass" else prep.le_binary.classes_
            pred_labels = classes[preds]

            df["predicted"] = pred_labels
            df["confidence"] = confidence
            df["is_attack"] = pred_labels != "normal"

            # Summary
            attack_count = df["is_attack"].sum()
            normal_count = len(df) - attack_count
            attack_pct = attack_count / len(df) * 100

            m1, m2, m3 = placeholder.columns(3)
            placeholder.empty()

            with placeholder.container():
                a, b, c = st.columns(3)
                a.metric("Total Packets", len(df))
                b.metric("Attacks Detected", int(attack_count), delta=f"{attack_pct:.0f}%", delta_color="inverse")
                c.metric("Avg Confidence", f"{confidence.mean():.1%}")

                # Category breakdown
                if task_choice == "multiclass":
                    cat_counts = df["predicted"].value_counts().reset_index()
                    cat_counts.columns = ["Category", "Count"]
                    fig = px.bar(
                        cat_counts, x="Category", y="Count",
                        color="Category", color_discrete_map=CATEGORY_COLORS,
                        title="Detected Attack Categories",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Timeline
                df["packet_id"] = range(len(df))
                df["threat_level"] = df["predicted"].map(
                    lambda x: 0 if x == "normal" else (
                        1 if x == "Probe" else (
                        2 if x == "R2L" else (
                        3 if x == "DoS" else 4)))
                )
                fig2 = px.scatter(
                    df, x="packet_id", y="confidence",
                    color="predicted", symbol="is_attack",
                    color_discrete_map=CATEGORY_COLORS,
                    title="Confidence Score by Packet",
                    labels={"packet_id": "Packet #", "confidence": "Confidence"},
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Table of attacks only
                attacks = df[df["is_attack"]].head(20)
                if len(attacks):
                    st.subheader(f"🚨 Top Threats ({len(attacks)} shown)")
                    st.dataframe(
                        attacks[["protocol_type", "service", "src_bytes", "dst_bytes", "predicted", "confidence"]]
                        .rename(columns={"predicted": "Category"}),
                        use_container_width=True,
                    )

        if run_btn:
            run_detection()

        if auto_refresh:
            time.sleep(3)
            run_detection()
            st.rerun()


elif page == "📈 Attack Analysis":
    st.title("📈 Threat Intelligence & Analysis")
    train_df, test_df = load_dataset()

    if train_df is None:
        st.warning("Dataset not found. Run `python main.py download` first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Protocol Analysis", "Feature Importance", "Threat Heatmap"])

        with tab1:
            st.subheader("Attack Types by Protocol")
            proto_cat = (
                train_df.groupby(["protocol_type", "attack_category"])
                .size()
                .reset_index(name="count")
            )
            fig = px.sunburst(
                proto_cat, path=["protocol_type", "attack_category"],
                values="count",
                color="attack_category",
                color_discrete_map=CATEGORY_COLORS,
                title="Attack Distribution by Protocol",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.treemap(
                train_df.groupby(["service", "attack_category"]).size().reset_index(name="count").nlargest(50, "count"),
                path=["service", "attack_category"],
                values="count",
                color="attack_category",
                color_discrete_map=CATEGORY_COLORS,
                title="Top Services by Attack Category (Treemap)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            plot_dir = ROOT / "results" / "plots"
            st.subheader("Feature Importance — Numeric Summary")
            results = load_results()
            if not results:
                st.info("Train models to generate feature importance.")
            else:
                st.markdown("Train Random Forest / XGBoost and check `results/` for feature importance plots.")

            st.subheader("DoS vs Normal: Key Differentiating Features")
            dos_df = train_df[train_df["attack_category"].isin(["DoS", "normal"])].sample(
                min(5000, len(train_df)), random_state=42
            )
            feat = st.selectbox("Feature", ["count", "srv_count", "serror_rate", "src_bytes", "dst_bytes"])
            fig3 = px.box(
                dos_df, x="attack_category", y=feat, color="attack_category",
                color_discrete_map=CATEGORY_COLORS,
                title=f"{feat} Distribution: DoS vs Normal",
                points="outliers",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.subheader("Flag × Protocol Attack Heatmap")
            hm_data = (
                train_df[train_df["attack_category"] != "normal"]
                .groupby(["flag", "protocol_type"])
                .size()
                .unstack(fill_value=0)
            )
            fig4 = px.imshow(
                hm_data,
                title="Attack Volume: Flag vs Protocol",
                color_continuous_scale="YlOrRd",
                aspect="auto",
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader("Brute Force (R2L) — Failed Logins Analysis")
            r2l = train_df[train_df["attack_category"] == "R2L"]
            if len(r2l):
                fig5 = px.histogram(
                    r2l, x="num_failed_logins", color="label",
                    title="Brute Force Attacks — Failed Login Attempts Distribution",
                    nbins=20,
                )
                st.plotly_chart(fig5, use_container_width=True)
