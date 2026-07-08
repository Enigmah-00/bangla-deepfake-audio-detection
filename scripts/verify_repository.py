from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require(path: str) -> None:
    target = ROOT / path
    if not target.exists():
        raise AssertionError(f"Missing required path: {path}")


def main() -> int:
    required = [
        "app.py",
        "backend/main.py",
        "src/config.py",
        "src/audio_io.py",
        "src/features.py",
        "src/predictor.py",
        "src/explainability.py",
        "src/visualization.py",
        "src/reporting.py",
        "frontend/package.json",
        "requirements.txt",
        "packages.txt",
        "models/bangla_deepfake_detector.joblib",
        "models/audio_scaler.joblib",
        "notebooks/pipeline.ipynb",
        "artifacts/metrics.json",
        "artifacts/feature_schema.json",
        "artifacts/model_metadata.json",
        "README.md",
    ]
    for path in required:
        require(path)

    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    model_hash = sha256(ROOT / metadata["file_names"]["model"])
    scaler_hash = sha256(ROOT / metadata["file_names"]["scaler"])
    if model_hash != metadata["sha256"]["model"]:
        raise AssertionError("Model SHA-256 does not match artifacts/model_metadata.json")
    if scaler_hash != metadata["sha256"]["scaler"]:
        raise AssertionError("Scaler SHA-256 does not match artifacts/model_metadata.json")

    schema = json.loads((ROOT / "artifacts" / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["feature_count"] != 227:
        raise AssertionError("Feature schema must declare 227 features")
    if schema["groups"][-1]["end"] != 227:
        raise AssertionError("Feature schema groups must end at index 227")

    print(json.dumps({"ok": True, "checked_paths": len(required)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        raise SystemExit(1)

