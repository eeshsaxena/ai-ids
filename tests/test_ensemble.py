"""Tests for soft-voting ensemble model."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import EnsembleIDS, NeuralNetworkIDS, RandomForestIDS, XGBoostIDS

CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
N, F, C = 150, 25, 3


@pytest.fixture
def xy():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.integers(0, C, N)
    return X, y


@pytest.fixture
def ensemble(xy):
    X, y = xy
    rf = RandomForestIDS(CONFIG, n_estimators=5)
    xgb = XGBoostIDS(CONFIG, n_estimators=5)
    nn = NeuralNetworkIDS(F, C, CONFIG)
    rf.fit(X, y)
    xgb.fit(X, y)
    nn.fit(X, y)
    return EnsembleIDS(rf, xgb, nn)


def test_predict_shape(ensemble, xy):
    X, _ = xy
    assert ensemble.predict(X).shape == (N,)


def test_proba_shape(ensemble, xy):
    X, _ = xy
    probs = ensemble.predict_proba(X)
    assert probs.shape == (N, C)


def test_proba_sums_to_one(ensemble, xy):
    X, _ = xy
    probs = ensemble.predict_proba(X)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_custom_weights(xy):
    X, y = xy
    rf = RandomForestIDS(CONFIG, n_estimators=5)
    xgb = XGBoostIDS(CONFIG, n_estimators=5)
    nn = NeuralNetworkIDS(F, C, CONFIG)
    rf.fit(X, y); xgb.fit(X, y); nn.fit(X, y)
    ens = EnsembleIDS(rf, xgb, nn, weights=(2.0, 1.0, 1.0))
    assert ens.predict(X).shape == (N,)
