from __future__ import annotations

import numpy as np

from . import config
from .audio_io import UserSafeAudioError

config.configure_runtime_environment()

import librosa


def feature_group_index_map() -> list[dict[str, int | str]]:
    return [dict(group) for group in config.FEATURE_GROUPS]


def extract_features(audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    if sample_rate != config.SAMPLE_RATE:
        raise UserSafeAudioError(
            f"Expected audio sampled at {config.SAMPLE_RATE} Hz after preprocessing.",
            code="bad_sample_rate",
        )

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = librosa.to_mono(audio)
    if audio.size == 0:
        raise UserSafeAudioError("Cannot extract features from empty audio.", code="empty_audio")
    if not np.all(np.isfinite(audio)):
        raise UserSafeAudioError("Cannot extract features from invalid sample values.", code="non_finite_audio")

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    except Exception as exc:  # pragma: no cover - librosa raises version-specific errors.
        raise UserSafeAudioError(
            "The acoustic feature extraction step failed for this audio.",
            code="feature_extraction_failed",
        ) from exc

    features = np.hstack(
        [
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
        ]
    )

    features = np.asarray(features, dtype=np.float32)
    if features.shape != (config.EXPECTED_FEATURE_COUNT,):
        raise UserSafeAudioError(
            f"Expected {config.EXPECTED_FEATURE_COUNT} features but got {features.shape[0]}.",
            code="feature_count_mismatch",
        )
    if not np.all(np.isfinite(features)):
        raise UserSafeAudioError("Extracted features contain invalid values.", code="non_finite_features")
    return features
