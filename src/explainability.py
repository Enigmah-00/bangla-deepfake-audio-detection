from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import config


def _normalize_group_values(values: dict[str, float]) -> list[dict[str, float | str]]:
    total = float(sum(max(v, 0.0) for v in values.values()))
    rows: list[dict[str, float | str]] = []
    for group in config.FEATURE_GROUPS:
        raw = float(max(values.get(str(group["key"]), 0.0), 0.0))
        percent = 0.0 if total <= 0 else (raw / total) * 100.0
        rows.append(
            {
                "key": str(group["key"]),
                "label": str(group["label"]),
                "value": round(raw, 6),
                "percent": round(percent, 2),
            }
        )
    return rows


def _group_vector_abs(vector: np.ndarray) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for group in config.FEATURE_GROUPS:
        start = int(group["start"])
        end = int(group["end"])
        grouped[str(group["key"])] = float(np.sum(np.abs(vector[start:end])))
    return grouped


def grouped_acoustic_evidence(model: Any, scaled_features: np.ndarray) -> dict[str, Any]:
    row = np.asarray(scaled_features, dtype=np.float32)
    if row.ndim == 1:
        row = row.reshape(1, -1)

    enable_global = os.getenv("KONTHOPROHORI_ENABLE_GLOBAL_IMPORTANCE", "").lower() in {"1", "true", "yes"}
    enable_local = os.getenv("KONTHOPROHORI_ENABLE_LOCAL_CONTRIBS", "").lower() in {"1", "true", "yes"}
    if not enable_local:
        if enable_global:
            importances = getattr(model, "feature_importances_", None)
        else:
            importances = None

        if importances is not None:
            values = _group_vector_abs(np.asarray(importances, dtype=float))
            method = "global_xgboost_feature_importance"
            caution = "This chart shows global model importance, not clip-specific proof."
        else:
            values = _group_vector_abs(np.asarray(row[0], dtype=float))
            method = "scaled_feature_magnitude_fallback"
            caution = (
                "Model contribution output is disabled for reliability; this chart summarizes "
                "scaled acoustic feature magnitude, not proof of manipulation."
            )
        if sum(values.values()) <= 0:
            values = {str(group["key"]): 1.0 for group in config.FEATURE_GROUPS}

        return {
            "title": "Relative acoustic evidence used by the model",
            "method": method,
            "groups": _normalize_group_values(values),
            "caution": caution,
        }

    try:
        import xgboost as xgb

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(row)
        contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
        local = contribs[0, :-1]
        return {
            "title": "Relative acoustic evidence used by the model",
            "method": "local_xgboost_contributions",
            "groups": _normalize_group_values(_group_vector_abs(local)),
            "caution": "High contribution means the feature group influenced this model score; it does not prove manipulation.",
        }
    except Exception:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            values = {str(group["key"]): 1.0 for group in config.FEATURE_GROUPS}
            method = "feature_groups_only"
            caution = "Local contribution output was unavailable; groups are shown without importance weighting."
        else:
            values = _group_vector_abs(np.asarray(importances, dtype=float))
            method = "global_xgboost_feature_importance"
            caution = "Local contribution output was unavailable; this chart shows global model importance, not clip-specific proof."

        return {
            "title": "Relative acoustic evidence used by the model",
            "method": method,
            "groups": _normalize_group_values(values),
            "caution": caution,
        }
