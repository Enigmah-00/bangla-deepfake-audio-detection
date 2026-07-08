from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_NAME = "KonthoProhori"
PROJECT_TAGLINE = "Bangla Deepfake Audio Detection and Voice Forensics"
PROJECT_TAGLINE_BN = "বাংলা কণ্ঠস্বর ডিপফেক শনাক্তকরণ ও ফরেনসিক বিশ্লেষণ"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "bangla_deepfake_detector.joblib"
SCALER_PATH = MODEL_DIR / "audio_scaler.joblib"

SAMPLE_RATE = 16000
ANALYSIS_SECONDS = 5.0
EXPECTED_FEATURE_COUNT = 227

FAKE_THRESHOLD = 0.70
REAL_THRESHOLD = 0.30
MIN_DURATION_SECONDS = 0.50
MAX_UPLOAD_MB = 20
SILENCE_RMS_THRESHOLD = 1e-4
SILENCE_PEAK_THRESHOLD = 5e-4

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".webm",
}

FEATURE_GROUPS = [
    {"key": "mfcc_mean", "label": "MFCC mean", "start": 0, "end": 40},
    {"key": "mfcc_std", "label": "MFCC variability", "start": 40, "end": 80},
    {"key": "chroma_mean", "label": "Chroma", "start": 80, "end": 92},
    {"key": "mel_mean", "label": "Mel-spectrum", "start": 92, "end": 220},
    {
        "key": "spectral_contrast_mean",
        "label": "Spectral contrast",
        "start": 220,
        "end": 227,
    },
]

DISCLAIMER = (
    "This tool provides AI-assisted screening, not legal or forensic proof. "
    "Performance may vary for unseen speakers, generators, accents, compression "
    "levels, background noise, and recording devices."
)

VALIDATION_CAVEAT = (
    "The model achieved 99.57% accuracy on the notebook's held-out random split. "
    "Broader cross-speaker, cross-generator, and real-world validation remains "
    "future work."
)

DATASET_URL = "https://www.kaggle.com/datasets/ahanaf101/bangla-deepfake-dataset"
MODEL_VERSION = "xgboost-handcrafted-227-v1"


def configure_runtime_environment() -> None:
    numba_cache_dir = ROOT_DIR / ".numba_cache"
    matplotlib_cache_dir = ROOT_DIR / ".matplotlib_cache"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    numba_cache_dir.mkdir(exist_ok=True)
    matplotlib_cache_dir.mkdir(exist_ok=True)

LABELS = {
    "deepfake": {
        "label": "Likely Deepfake",
        "label_bn": "সম্ভাব্য ডিপফেক",
        "band": "high-risk",
        "summary": (
            "The model found acoustic patterns associated with synthetic or "
            "manipulated speech. Verify the clip through its original source "
            "before acting or sharing."
        ),
    },
    "genuine": {
        "label": "Likely Genuine",
        "label_bn": "সম্ভাব্য আসল কণ্ঠ",
        "band": "low-risk",
        "summary": (
            "The model found stronger similarity to genuine speech in its "
            "training data. This does not prove authenticity; source "
            "verification is still recommended."
        ),
    },
    "inconclusive": {
        "label": "Inconclusive",
        "label_bn": "অনির্ধারিত",
        "band": "uncertain",
        "summary": (
            "The model score is not strong enough for a reliable screening "
            "decision. Try a clearer clip or request expert review."
        ),
    },
}
