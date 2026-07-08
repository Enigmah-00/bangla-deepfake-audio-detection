from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src import config
from src.audio_io import UserSafeAudioError, process_audio_path


def test_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "clip.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(UserSafeAudioError, match="Unsupported"):
        process_audio_path(path)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(UserSafeAudioError, match="not found"):
        process_audio_path(tmp_path / "missing.wav")


def test_oversized_file(tmp_path) -> None:
    path = tmp_path / "large.wav"
    path.write_bytes(b"0" * ((config.MAX_UPLOAD_MB * 1024 * 1024) + 1))
    with pytest.raises(UserSafeAudioError, match="too large"):
        process_audio_path(path)


def test_zero_byte_file(tmp_path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(UserSafeAudioError, match="empty"):
        process_audio_path(path)


def test_too_short_audio(tmp_path) -> None:
    path = tmp_path / "short.wav"
    audio = np.ones(int(config.SAMPLE_RATE * 0.1), dtype=np.float32) * 0.1
    sf.write(path, audio, config.SAMPLE_RATE)
    with pytest.raises(UserSafeAudioError, match="too short"):
        process_audio_path(path)


def test_near_silent_audio(tmp_path) -> None:
    path = tmp_path / "quiet.wav"
    audio = np.ones(config.SAMPLE_RATE, dtype=np.float32) * 1e-6
    sf.write(path, audio, config.SAMPLE_RATE)
    with pytest.raises(UserSafeAudioError, match="too quiet|silent"):
        process_audio_path(path)

