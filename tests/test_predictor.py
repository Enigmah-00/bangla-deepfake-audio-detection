from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.predictor import decision_from_probability, load_assets


def test_model_and_scaler_load() -> None:
    assets = load_assets()
    assert assets.model is not None
    assert assets.scaler is not None


def test_model_accepts_one_227_feature_row() -> None:
    assets = load_assets()
    row = np.zeros((1, config.EXPECTED_FEATURE_COUNT), dtype=np.float32)
    scaled = assets.scaler.transform(row)
    proba = assets.model.predict_proba(scaled)
    assert proba.shape[0] == 1
    assert proba.shape[1] == 2


def test_probabilities_are_valid() -> None:
    assets = load_assets()
    row = np.zeros((1, config.EXPECTED_FEATURE_COUNT), dtype=np.float32)
    proba = assets.model.predict_proba(assets.scaler.transform(row))[0]
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)
    assert float(np.sum(proba)) == pytest.approx(1.0, abs=1e-5)


def test_decision_boundaries() -> None:
    assert decision_from_probability(0.70)["label"] == "Likely Deepfake"
    assert decision_from_probability(0.30)["label"] == "Likely Genuine"
    assert decision_from_probability(0.50)["label"] == "Inconclusive"

