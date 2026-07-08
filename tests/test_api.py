from __future__ import annotations

from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app
from src import config


def make_wav_bytes() -> bytes:
    t = np.linspace(0, 1.0, config.SAMPLE_RATE, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, audio, config.SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def test_analyze_endpoint_returns_safe_structured_result() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        files={"file": ("tone.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["prediction"]["label"] in {"Likely Genuine", "Likely Deepfake", "Inconclusive"}
    assert payload["audio"]["sample_rate"] == config.SAMPLE_RATE
    assert "waveform" in payload["visualizations"]
    assert "report" in payload


def test_analyze_endpoint_rejects_unsupported_format() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_format"

