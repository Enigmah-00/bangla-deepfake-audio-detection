from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src import config
from src.audio_io import UserSafeAudioError, process_audio_path
from src.features import extract_features


def tone(seconds: float = 1.0, frequency: float = 220.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(config.SAMPLE_RATE * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_valid_generated_waveform_returns_227_finite_values() -> None:
    features = extract_features(tone(), config.SAMPLE_RATE)
    assert features.shape == (227,)
    assert np.all(np.isfinite(features))


def test_stereo_file_is_converted_to_mono(tmp_path) -> None:
    mono = tone()
    stereo = np.column_stack([mono, mono * 0.5])
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, config.SAMPLE_RATE)
    processed = process_audio_path(path)
    assert processed.audio.ndim == 1
    assert processed.sample_rate == config.SAMPLE_RATE


def test_short_valid_audio_does_not_crash() -> None:
    features = extract_features(tone(seconds=0.55), config.SAMPLE_RATE)
    assert features.shape == (227,)


def test_silence_is_rejected(tmp_path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(config.SAMPLE_RATE, dtype=np.float32), config.SAMPLE_RATE)
    with pytest.raises(UserSafeAudioError, match="too quiet|silent"):
        process_audio_path(path)


def test_corrupt_file_does_not_raise_raw_exception(tmp_path) -> None:
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not a wave file")
    with pytest.raises(UserSafeAudioError):
        process_audio_path(path)

