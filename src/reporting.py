from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import config
from .audio_io import ProcessedAudio


def build_analysis_report(
    audio: ProcessedAudio,
    prediction: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": config.PROJECT_NAME,
        "filename": audio.filename,
        "duration_seconds": round(audio.duration_seconds, 4),
        "analyzed_seconds": round(audio.analyzed_seconds, 4),
        "sample_rate": audio.sample_rate,
        "predicted_label": prediction["label"],
        "predicted_label_bn": prediction["label_bn"],
        "fake_probability": prediction["fake_probability"],
        "real_probability": prediction["real_probability"],
        "decision_band": prediction["band"],
        "model_score_name": prediction["model_score_name"],
        "feature_group_evidence": evidence,
        "model_version": prediction["model_version"],
        "thresholds": prediction["thresholds"],
        "limitations": [
            config.DISCLAIMER,
            config.VALIDATION_CAVEAT,
            "The result does not identify the speaker and must not be used as standalone legal or forensic proof.",
        ],
    }

