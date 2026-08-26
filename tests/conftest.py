"""Shared pytest setup."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _clear_label_horizons():
    """
    `temporal_split.LABEL_HORIZONS` is module-level state — it describes the
    feed, so one process-wide map is the right shape in production. In tests it
    would leak a horizon registered by one case into another's maturity
    decision, making failures depend on test order. Clear it around each test.
    """
    from src.data import temporal_split

    temporal_split.LABEL_HORIZONS.clear()
    yield
    temporal_split.LABEL_HORIZONS.clear()
