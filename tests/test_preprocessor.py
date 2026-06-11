"""Tests for NSL-KDD preprocessor."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocessor import (
    ATTACK_CATEGORIES,
    COLUMNS,
    NSLKDDPreprocessor,
)


def _make_sample_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "ftp", "smtp"]
    flags = ["SF", "S0", "REJ"]
    labels = list(ATTACK_CATEGORIES.keys())

    data = {
        "duration": rng.integers(0, 1000, n),
        "protocol_type": rng.choice(protocols, n),
        "service": rng.choice(services, n),
        "flag": rng.choice(flags, n),
        "src_bytes": rng.integers(0, 100000, n),
        "dst_bytes": rng.integers(0, 100000, n),
        "land": rng.integers(0, 2, n),
        "wrong_fragment": rng.integers(0, 3, n),
        "urgent": rng.integers(0, 2, n),
        "hot": rng.integers(0, 10, n),
        "num_failed_logins": rng.integers(0, 5, n),
        "logged_in": rng.integers(0, 2, n),
        "num_compromised": rng.integers(0, 10, n),
        "root_shell": rng.integers(0, 2, n),
        "su_attempted": rng.integers(0, 2, n),
        "num_root": rng.integers(0, 5, n),
        "num_file_creations": rng.integers(0, 5, n),
        "num_shells": rng.integers(0, 3, n),
        "num_access_files": rng.integers(0, 3, n),
        "num_outbound_cmds": rng.integers(0, 1, n),
        "is_host_login": rng.integers(0, 2, n),
        "is_guest_login": rng.integers(0, 2, n),
        "count": rng.integers(0, 256, n),
        "srv_count": rng.integers(0, 256, n),
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
        "label": rng.choice(labels, n),
        "difficulty": rng.integers(0, 21, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dfs():
    return _make_sample_df(80), _make_sample_df(20)


@pytest.fixture
def preprocessor(tmp_path):
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    return NSLKDDPreprocessor(config_path=str(cfg_path))


def test_attack_categories_complete():
    cats = set(ATTACK_CATEGORIES.values())
    assert "normal" in cats
    assert "DoS" in cats
    assert "Probe" in cats
    assert "R2L" in cats
    assert "U2R" in cats


def test_fit_transform_shapes(preprocessor, sample_dfs):
    train, test = sample_dfs
    out = preprocessor.fit_transform(train, test)
    assert out["X_train"].shape[0] == len(train)
    assert out["X_test"].shape[0] == len(test)
    assert out["X_train"].shape[1] == out["X_test"].shape[1]


def test_binary_labels(preprocessor, sample_dfs):
    train, test = sample_dfs
    out = preprocessor.fit_transform(train, test)
    unique = set(np.unique(out["y_train_binary"]))
    assert unique.issubset({0, 1})


def test_multi_labels_bounded(preprocessor, sample_dfs):
    train, test = sample_dfs
    out = preprocessor.fit_transform(train, test)
    n_classes = len(np.unique(out["y_train_multi"]))
    assert 1 <= n_classes <= 5


def test_feature_count(preprocessor, sample_dfs):
    train, test = sample_dfs
    out = preprocessor.fit_transform(train, test)
    assert out["n_features"] > 41  # one-hot expands categorical cols


def test_transform_aligns_columns(preprocessor, sample_dfs):
    train, test = sample_dfs
    preprocessor.fit_transform(train, test)
    new = _make_sample_df(5)
    # Remove label column
    new = new.drop(columns=["label", "difficulty"])
    X = preprocessor.transform(new)
    assert X.shape[1] == len(preprocessor.feature_names)


def test_save_load(preprocessor, sample_dfs, tmp_path):
    train, test = sample_dfs
    preprocessor.fit_transform(train, test)
    path = str(tmp_path / "prep.pkl")
    preprocessor.save(path)
    loaded = NSLKDDPreprocessor.load(path)
    assert loaded.feature_names == preprocessor.feature_names
