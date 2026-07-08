from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.audio_io import UserSafeAudioError, process_audio_bytes
from src.explainability import grouped_acoustic_evidence
from src.features import extract_features, feature_group_index_map
from src.predictor import ModelLoadError, load_assets, predict_from_features
from src.reporting import build_analysis_report
from src.visualization import build_visualizations


app = FastAPI(
    title=f"{config.PROJECT_NAME} API",
    description="FastAPI inference backend for Bangla deepfake audio screening.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    model_ready = True
    error = None
    try:
        load_assets()
    except Exception as exc:  # pragma: no cover - environment-specific.
        model_ready = False
        error = str(exc)
    return {
        "status": "ok" if model_ready else "degraded",
        "project": config.PROJECT_NAME,
        "model_ready": model_ready,
        "error": error,
    }


@app.get("/api/metadata")
def metadata() -> dict[str, Any]:
    return {
        "project": config.PROJECT_NAME,
        "tagline": config.PROJECT_TAGLINE,
        "tagline_bn": config.PROJECT_TAGLINE_BN,
        "sample_rate": config.SAMPLE_RATE,
        "analysis_seconds": config.ANALYSIS_SECONDS,
        "expected_feature_count": config.EXPECTED_FEATURE_COUNT,
        "supported_extensions": sorted(config.SUPPORTED_EXTENSIONS),
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "feature_groups": feature_group_index_map(),
        "dataset_url": config.DATASET_URL,
        "disclaimer": config.DISCLAIMER,
        "validation_caveat": config.VALIDATION_CAVEAT,
    }


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        data = await file.read()
        processed = process_audio_bytes(data, file.filename)
        features = extract_features(processed.audio, processed.sample_rate)
        prediction = predict_from_features(features)
        assets = load_assets()
        evidence = grouped_acoustic_evidence(assets.model, prediction["scaled_features"])
        visualizations = build_visualizations(processed.audio, processed.sample_rate, prediction, evidence)
        report = build_analysis_report(processed, prediction, evidence)

        public_prediction = {key: value for key, value in prediction.items() if key != "scaled_features"}
        return {
            "ok": True,
            "audio": {
                "filename": processed.filename,
                "duration_seconds": round(processed.duration_seconds, 4),
                "analyzed_seconds": round(processed.analyzed_seconds, 4),
                "sample_rate": processed.sample_rate,
                "rms": round(processed.rms, 8),
                "peak": round(processed.peak, 8),
            },
            "prediction": public_prediction,
            "evidence": evidence,
            "visualizations": visualizations,
            "report": report,
        }
    except UserSafeAudioError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "The model could not be loaded. Confirm the backend dependencies and trained assets.",
            },
        ) from exc
    except Exception as exc:  # pragma: no cover - safety net.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "analysis_failed",
                "message": "The audio could not be analyzed safely. Please try a clearer supported file.",
            },
        ) from exc

