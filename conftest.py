"""Shared pytest configuration — adds project root to sys.path."""

import sys
from pathlib import Path

# Ensure src/ and project root are importable from any test
sys.path.insert(0, str(Path(__file__).resolve().parent))
