"""Tests for FastAPI REST server endpoints."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi.testclient import TestClient
    from app.server import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_models" in data
    assert "total" in data


def test_predict_no_models(client):
    """When no models are trained, /predict should return 404 or 503."""
    payload = {
        "connections": [
            {
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 215,
                "dst_bytes": 45076,
                "duration": 0,
            }
        ],
        "model": "rf",
        "task": "multiclass",
    }
    resp = client.post("/predict", json=payload)
    # Either 200 (models exist) or 503/404 (no models trained yet)
    assert resp.status_code in (200, 404, 503)


def test_predict_schema_validation(client):
    """Send invalid payload — FastAPI should return 422."""
    resp = client.post("/predict", json={"connections": [], "model": "rf"})
    assert resp.status_code == 422  # min_length=1 violated


def test_predict_invalid_model(client):
    """Unknown model name should fail schema validation."""
    payload = {
        "connections": [{"protocol_type": "tcp", "service": "http", "flag": "SF"}],
        "model": "unknown_model",
        "task": "multiclass",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
