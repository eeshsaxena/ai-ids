"""NSL-KDD dataset preprocessing pipeline."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

ATTACK_CATEGORIES = {
    "normal": "normal",
    # DoS
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    # Probe (port scan)
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (Remote-to-Local / Brute Force)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "httptunnel": "R2L",
    "xlock": "R2L", "xsnoop": "R2L",
    # U2R (User-to-Root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "sqlattack": "U2R", "ps": "U2R",
}

CATEGORY_COLORS = {
    "normal": "#2ecc71",
    "DoS": "#e74c3c",
    "Probe": "#f39c12",
    "R2L": "#9b59b6",
    "U2R": "#e67e22",
}


class NSLKDDPreprocessor:
    """End-to-end preprocessing for NSL-KDD dataset."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
        self.le_binary = LabelEncoder()
        self.le_multi = LabelEncoder()
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.read_csv(
            self.config["data"]["train_path"], header=None, names=COLUMNS
        )
        test = pd.read_csv(
            self.config["data"]["test_path"], header=None, names=COLUMNS
        )
        return train, test

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["attack_category"] = df["label"].map(
            lambda x: ATTACK_CATEGORIES.get(x, "Unknown")
        )
        df["binary_label"] = df["label"].apply(
            lambda x: "normal" if x == "normal" else "attack"
        )
        return df

    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> dict:
        """Fit on train, transform both. Returns processed arrays + metadata."""
        label_cols = {"label", "attack_category", "binary_label", "difficulty"}

        train_df = self._add_labels(train_df)
        test_df = self._add_labels(test_df)

        # Drop non-feature columns
        drop = label_cols.intersection(train_df.columns)
        feat_cols = [c for c in train_df.columns if c not in drop]

        X_train_raw = pd.get_dummies(train_df[feat_cols], columns=CATEGORICAL_COLS)
        X_test_raw = pd.get_dummies(test_df[feat_cols], columns=CATEGORICAL_COLS)

        # Align test to training columns
        X_train_raw, X_test_raw = X_train_raw.align(
            X_test_raw, join="left", axis=1, fill_value=0
        )
        self.feature_names = list(X_train_raw.columns)

        X_train = self.scaler.fit_transform(X_train_raw.values.astype(np.float32))
        X_test = self.scaler.transform(X_test_raw.values.astype(np.float32))

        y_train_b = self.le_binary.fit_transform(train_df["binary_label"])
        y_test_b = self.le_binary.transform(test_df["binary_label"])

        y_train_m = self.le_multi.fit_transform(train_df["attack_category"])
        y_test_m = self.le_multi.transform(test_df["attack_category"])

        return {
            "X_train": X_train.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "y_train_binary": y_train_b,
            "y_test_binary": y_test_b,
            "y_train_multi": y_train_m,
            "y_test_multi": y_test_m,
            "train_df": train_df,
            "test_df": test_df,
            "binary_classes": list(self.le_binary.classes_),
            "multi_classes": list(self.le_multi.classes_),
            "n_features": len(self.feature_names),
        }

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform unseen data using fitted scaler."""
        df = df.copy()
        if "difficulty" in df.columns:
            df = df.drop(columns=["difficulty"])
        label_cols = {"label", "attack_category", "binary_label"}
        feat_cols = [c for c in df.columns if c not in label_cols]

        X = pd.get_dummies(df[feat_cols], columns=CATEGORICAL_COLS)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        return self.scaler.transform(X.values.astype(np.float32))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/saved/preprocessor.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = "models/saved/preprocessor.pkl") -> "NSLKDDPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
