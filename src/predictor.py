from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import config
from .audio_io import UserSafeAudioError


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedAssets:
    model: Any
    scaler: Any
    model_path: Path
    scaler_path: Path


@lru_cache(maxsize=1)
def load_assets() -> LoadedAssets:
    if not config.MODEL_PATH.exists():
        raise ModelLoadError(f"Model asset not found: {config.MODEL_PATH}")
    if not config.SCALER_PATH.exists():
        raise ModelLoadError(f"Scaler asset not found: {config.SCALER_PATH}")

    try:
        model = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
    except Exception as exc:
        raise ModelLoadError(
            "Could not load trained model assets. Confirm compatible xgboost, scikit-learn, and joblib versions."
        ) from exc

    probe = np.zeros((1, config.EXPECTED_FEATURE_COUNT), dtype=np.float32)
    try:
        transformed = scaler.transform(probe)
    except Exception as exc:
        raise ModelLoadError("Scaler did not accept the expected 227-feature input.") from exc
    if transformed.shape != probe.shape:
        raise ModelLoadError("Scaler returned an unexpected feature shape.")
    if not hasattr(model, "predict_proba"):
        raise ModelLoadError("Loaded model does not expose predict_proba().")
    return LoadedAssets(model=model, scaler=scaler, model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH)


def decision_from_probability(p_fake: float) -> dict[str, str]:
    if p_fake >= config.FAKE_THRESHOLD:
        return dict(config.LABELS["deepfake"])
    if p_fake <= config.REAL_THRESHOLD:
        return dict(config.LABELS["genuine"])
    return dict(config.LABELS["inconclusive"])


def _probability_indices(model: Any) -> tuple[int, int]:
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != 2:
        return 0, 1
    class_list = list(classes)
    fake_idx = class_list.index(1) if 1 in class_list else 1
    real_idx = class_list.index(0) if 0 in class_list else 0
    return real_idx, fake_idx


def predict_from_features(features: np.ndarray) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (config.EXPECTED_FEATURE_COUNT,):
        raise UserSafeAudioError(
            f"Expected {config.EXPECTED_FEATURE_COUNT} features for prediction.",
            code="feature_count_mismatch",
        )
    assets = load_assets()
    row = features.reshape(1, -1)
    scaled = assets.scaler.transform(row)
    probabilities = np.asarray(assets.model.predict_proba(scaled), dtype=float)
    if probabilities.shape[0] != 1 or probabilities.shape[1] < 2:
        raise ModelLoadError("Model returned an unexpected probability shape.")

    real_idx, fake_idx = _probability_indices(assets.model)
    p_fake = float(probabilities[0, fake_idx])
    p_real = float(probabilities[0, real_idx])
    total = p_fake + p_real
    if total > 0:
        p_fake = p_fake / total
        p_real = p_real / total

    decision = decision_from_probability(p_fake)
    return {
        **decision,
        "fake_probability": round(p_fake, 6),
        "real_probability": round(p_real, 6),
        "model_score_name": "model probability",
        "thresholds": {
            "likely_genuine_at_or_below": config.REAL_THRESHOLD,
            "likely_deepfake_at_or_above": config.FAKE_THRESHOLD,
        },
        "model_version": config.MODEL_VERSION,
        "disclaimer": config.DISCLAIMER,
        "scaled_features": scaled.astype(np.float32),
    }

