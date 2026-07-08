from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True)
        return output.strip()
    except Exception:
        return None


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    from src import config
    from src.predictor import load_assets

    assets = load_assets()
    model = assets.model
    scaler = assets.scaler
    params = model.get_params() if hasattr(model, "get_params") else None

    metadata = {
        "file_names": {
            "model": str(config.MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "scaler": str(config.SCALER_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "sha256": {
            "model": sha256(config.MODEL_PATH),
            "scaler": sha256(config.SCALER_PATH),
        },
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_parameters": json_safe(params),
        "scaler_class": f"{scaler.__class__.__module__}.{scaler.__class__.__name__}",
        "input_feature_count": config.EXPECTED_FEATURE_COUNT,
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": version("numpy"),
            "scikit-learn": version("sklearn"),
            "xgboost": version("xgboost"),
            "joblib": version("joblib"),
            "librosa": version("librosa"),
        },
        "generation_date": datetime.now(timezone.utc).isoformat(),
        "repository_commit_hash": git_commit(),
        "load_status": "verified",
        "compatibility_note": (
            "This metadata records the environment that loaded the assets. "
            "Use requirements.txt for the recommended deployment stack to reduce Joblib version warnings."
        ),
    }

    target = ROOT / "artifacts" / "model_metadata.json"
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": True, "wrote": str(target.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
