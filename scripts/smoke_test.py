from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.audio_io import process_audio_path
from src.explainability import grouped_acoustic_evidence
from src.features import extract_features
from src.predictor import load_assets, predict_from_features
from src.reporting import build_analysis_report


def _analyze(path: Path) -> dict[str, object]:
    processed = process_audio_path(path)
    features = extract_features(processed.audio, processed.sample_rate)
    prediction = predict_from_features(features)
    assets = load_assets()
    evidence = grouped_acoustic_evidence(assets.model, prediction["scaled_features"])
    report = build_analysis_report(processed, prediction, evidence)
    return {
        "file": path.name,
        "label": report["predicted_label"],
        "fake_probability": report["fake_probability"],
        "real_probability": report["real_probability"],
        "duration_seconds": report["duration_seconds"],
    }


def _make_synthetic_clip(path: Path, frequency: float) -> None:
    seconds = 1.5
    t = np.linspace(0, seconds, int(config.SAMPLE_RATE * seconds), endpoint=False)
    audio = 0.16 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, audio.astype(np.float32), config.SAMPLE_RATE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a backend inference smoke test.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated audio when real/fake example files are not available.",
    )
    args = parser.parse_args()

    load_assets()
    example_paths = [ROOT / "examples" / "real_sample.wav", ROOT / "examples" / "fake_sample.wav"]

    if all(path.exists() for path in example_paths):
        outputs = [_analyze(path) for path in example_paths]
    elif args.synthetic:
        with tempfile.TemporaryDirectory(prefix="konthoprohori_smoke_") as tmp_dir:
            real_like = Path(tmp_dir) / "synthetic_real_placeholder.wav"
            fake_like = Path(tmp_dir) / "synthetic_fake_placeholder.wav"
            _make_synthetic_clip(real_like, 220.0)
            _make_synthetic_clip(fake_like, 440.0)
            outputs = [_analyze(real_like), _analyze(fake_like)]
    else:
        missing = [str(path.relative_to(ROOT)) for path in example_paths if not path.exists()]
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Missing example audio files. Add consented dataset examples or run with --synthetic.",
                    "missing": missing,
                },
                indent=2,
            )
        )
        return 2

    print(json.dumps({"ok": True, "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

