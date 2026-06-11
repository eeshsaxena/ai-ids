"""Tests for simulate_traffic helper."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import simulate_traffic


def test_simulate_shape():
    df = simulate_traffic(50)
    assert len(df) == 50


def test_simulate_has_required_cols():
    df = simulate_traffic(10)
    required = {"protocol_type", "service", "flag", "src_bytes", "dst_bytes"}
    assert required.issubset(df.columns)


def test_simulate_deterministic():
    df1 = simulate_traffic(20, seed=0)
    df2 = simulate_traffic(20, seed=0)
    assert (df1["src_bytes"] == df2["src_bytes"]).all()


def test_simulate_different_seeds():
    df1 = simulate_traffic(20, seed=1)
    df2 = simulate_traffic(20, seed=2)
    assert not (df1["src_bytes"] == df2["src_bytes"]).all()
