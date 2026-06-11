"""Tests for all three IDS models."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import NeuralNetworkIDS, RandomForestIDS, XGBoostIDS

CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
N, F, C = 200, 30, 5


@pytest.fixture
def xy():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = rng.integers(0, C, N)
    return X, y


@pytest.fixture
def rf():
    return RandomForestIDS(CONFIG, n_estimators=10)


@pytest.fixture
def xgb():
    return XGBoostIDS(CONFIG, n_estimators=10)


@pytest.fixture
def nn():
    return NeuralNetworkIDS(F, C, CONFIG)


class TestRandomForest:
    def test_predict_shape(self, rf, xy):
        X, y = xy
        rf.fit(X, y)
        assert rf.predict(X).shape == (N,)

    def test_proba_shape(self, rf, xy):
        X, y = xy
        rf.fit(X, y)
        assert rf.predict_proba(X).shape == (N, C)

    def test_feature_importance(self, rf, xy):
        X, y = xy
        rf.fit(X, y)
        names = [f"f{i}" for i in range(F)]
        imp = rf.get_feature_importance(names, top_n=5)
        assert len(imp) == 5

    def test_save_load(self, rf, xy, tmp_path):
        X, y = xy
        rf.fit(X, y)
        path = str(tmp_path / "rf.pkl")
        rf.save(path)
        loaded = RandomForestIDS.load(path)
        np.testing.assert_array_equal(rf.predict(X), loaded.predict(X))


class TestXGBoost:
    def test_predict_shape(self, xgb, xy):
        X, y = xy
        xgb.fit(X, y)
        assert xgb.predict(X).shape == (N,)

    def test_proba_shape(self, xgb, xy):
        X, y = xy
        xgb.fit(X, y)
        assert xgb.predict_proba(X).shape == (N, C)

    def test_save_load(self, xgb, xy, tmp_path):
        X, y = xy
        xgb.fit(X, y)
        path = str(tmp_path / "xgb.pkl")
        xgb.save(path)
        loaded = XGBoostIDS.load(path)
        np.testing.assert_array_equal(xgb.predict(X), loaded.predict(X))


class TestNeuralNetwork:
    def test_predict_shape(self, nn, xy):
        X, y = xy
        nn.fit(X, y)
        assert nn.predict(X).shape == (N,)

    def test_proba_shape(self, nn, xy):
        X, y = xy
        nn.fit(X, y)
        probs = nn.predict_proba(X)
        assert probs.shape == (N, C)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load(self, nn, xy, tmp_path):
        X, y = xy
        nn.fit(X, y)
        path = str(tmp_path / "nn.pt")
        nn.save(path)
        loaded = NeuralNetworkIDS.load(path, CONFIG)
        np.testing.assert_array_equal(nn.predict(X), loaded.predict(X))
