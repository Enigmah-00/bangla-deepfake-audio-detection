from __future__ import annotations

import tempfile
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config


class UserSafeAudioError(ValueError):
    """Validation error that can be returned to users without a stack trace."""

    def __init__(self, message: str, code: str = "invalid_audio") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class ProcessedAudio:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    analyzed_seconds: float
    filename: str
    original_size_bytes: int
    rms: float
    peak: float


def safe_filename(filename: str | None) -> str:
    if not filename:
        return "uploaded_audio"
    return Path(filename).name.replace("\x00", "").strip() or "uploaded_audio"


def _validate_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise UserSafeAudioError(
            f"Unsupported audio format. Please upload one of: {supported}.",
            code="unsupported_format",
        )
    return suffix


def _validate_size(size_bytes: int) -> None:
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes <= 0:
        raise UserSafeAudioError("The uploaded file is empty.", code="empty_file")
    if size_bytes > max_bytes:
        raise UserSafeAudioError(
            f"The uploaded file is too large. Maximum size is {config.MAX_UPLOAD_MB} MB.",
            code="file_too_large",
        )


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise UserSafeAudioError("The decoded audio has an unsupported channel layout.", code="bad_channel_layout")


def _resample_if_needed(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate == config.SAMPLE_RATE:
        return audio.astype(np.float32), sample_rate
    common = gcd(sample_rate, config.SAMPLE_RATE)
    up = config.SAMPLE_RATE // common
    down = sample_rate // common
    resampled = resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float32), config.SAMPLE_RATE


def _load_audio_from_path(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        audio = _to_mono(np.asarray(audio, dtype=np.float32))
        return _resample_if_needed(audio, int(sample_rate))
    except Exception:
        pass

    try:
        config.configure_runtime_environment()
        import librosa

        audio, sample_rate = librosa.load(
            path,
            sr=config.SAMPLE_RATE,
            mono=True,
            duration=None,
        )
    except Exception as exc:  # pragma: no cover - exact decoder errors vary.
        raise UserSafeAudioError(
            "The audio could not be decoded. Try a clear WAV, MP3, FLAC, OGG, M4A, or WEBM file.",
            code="decode_failed",
        ) from exc

    audio = np.asarray(audio, dtype=np.float32)
    audio = _to_mono(audio)
    if audio.size == 0:
        raise UserSafeAudioError("The decoded audio has no samples.", code="empty_audio")
    if not np.all(np.isfinite(audio)):
        raise UserSafeAudioError("The decoded audio contains invalid sample values.", code="non_finite_audio")
    return audio, sample_rate


def _validate_signal(audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    duration_seconds = float(audio.size / sample_rate)
    if duration_seconds < config.MIN_DURATION_SECONDS:
        raise UserSafeAudioError(
            f"Audio is too short. Please provide at least {config.MIN_DURATION_SECONDS:.1f} seconds.",
            code="too_short",
        )

    audio64 = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(audio64))))
    peak = float(np.max(np.abs(audio)))
    if rms < config.SILENCE_RMS_THRESHOLD or peak < config.SILENCE_PEAK_THRESHOLD:
        raise UserSafeAudioError(
            "The audio is too quiet or silent for reliable analysis.",
            code="near_silent",
        )
    return duration_seconds, rms, peak


def process_audio_path(path: str | Path, filename: str | None = None) -> ProcessedAudio:
    audio_path = Path(path)
    display_name = safe_filename(filename or audio_path.name)
    _validate_extension(display_name)

    if not audio_path.exists():
        raise UserSafeAudioError("The audio file was not found.", code="missing_file")
    if not audio_path.is_file():
        raise UserSafeAudioError("The selected path is not an audio file.", code="not_a_file")

    size_bytes = audio_path.stat().st_size
    _validate_size(size_bytes)

    audio, sample_rate = _load_audio_from_path(audio_path)
    duration_seconds, rms, peak = _validate_signal(audio, sample_rate)
    max_samples = int(config.ANALYSIS_SECONDS * sample_rate)
    analyzed = np.asarray(audio[:max_samples], dtype=np.float32)
    analyzed_seconds = float(analyzed.size / sample_rate)

    return ProcessedAudio(
        audio=analyzed,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        analyzed_seconds=analyzed_seconds,
        filename=display_name,
        original_size_bytes=size_bytes,
        rms=rms,
        peak=peak,
    )


def process_audio_bytes(data: bytes, filename: str | None) -> ProcessedAudio:
    display_name = safe_filename(filename)
    suffix = _validate_extension(display_name)
    _validate_size(len(data))

    with tempfile.TemporaryDirectory(prefix="konthoprohori_") as tmp_dir:
        path = Path(tmp_dir) / f"upload{suffix}"
        path.write_bytes(data)
        return process_audio_path(path, filename=display_name)


def read_upload_file(file_obj: BinaryIO) -> bytes:
    data = file_obj.read()
    if not isinstance(data, bytes):
        raise UserSafeAudioError("The uploaded content could not be read.", code="read_failed")
    return data
