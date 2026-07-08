from __future__ import annotations

import base64
import io
from typing import Any

from . import config

config.configure_runtime_environment()

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


INK = "#18212f"
MUTED = "#64748b"
BLUE = "#2563eb"
AMBER = "#d97706"
TEAL = "#0f766e"
VIOLET = "#7c3aed"
PAPER = "#ffffff"


def _figure_to_data_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def waveform_image(audio: np.ndarray, sample_rate: int) -> str:
    audio = np.asarray(audio, dtype=float)
    times = np.arange(audio.size) / sample_rate
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    ax.plot(times, audio, color=BLUE, linewidth=0.9)
    ax.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax.set_title("Waveform", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", color=MUTED)
    ax.set_ylabel("Amplitude", color=MUTED)
    ax.grid(True, color="#e2e8f0", linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    return _figure_to_data_uri(fig)


def mel_spectrogram_image(audio: np.ndarray, sample_rate: int) -> str:
    mel = librosa.feature.melspectrogram(y=np.asarray(audio, dtype=np.float32), sr=sample_rate)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    img = librosa.display.specshow(log_mel, sr=sample_rate, x_axis="time", y_axis="mel", cmap="magma", ax=ax)
    ax.set_title("Log-mel spectrogram", color=INK, fontsize=12, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    return _figure_to_data_uri(fig)


def evidence_chart_image(evidence: dict[str, Any]) -> str:
    groups = evidence.get("groups", [])
    labels = [str(row["label"]) for row in groups]
    values = [float(row["percent"]) for row in groups]
    colors = [BLUE, TEAL, AMBER, VIOLET, "#334155"]

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors[: len(labels)])
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative contribution (%)", color=MUTED)
    ax.set_title("Relative acoustic evidence used by the model", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(100, max(values, default=0) * 1.15))
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for i, value in enumerate(values):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=8, color=INK)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    return _figure_to_data_uri(fig)


def probability_chart_image(fake_probability: float, real_probability: float) -> str:
    labels = ["Genuine model score", "Deepfake model score"]
    values = [real_probability * 100.0, fake_probability * 100.0]
    fig, ax = plt.subplots(figsize=(8.5, 2.7))
    bars = ax.bar(labels, values, color=[TEAL, AMBER])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)", color=MUTED)
    ax.set_title("Model probability", color=INK, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    return _figure_to_data_uri(fig)


def build_visualizations(audio: np.ndarray, sample_rate: int, prediction: dict[str, Any], evidence: dict[str, Any]) -> dict[str, str]:
    return {
        "waveform": waveform_image(audio, sample_rate),
        "mel_spectrogram": mel_spectrogram_image(audio, sample_rate),
        "evidence": evidence_chart_image(evidence),
        "probabilities": probability_chart_image(
            float(prediction["fake_probability"]),
            float(prediction["real_probability"]),
        ),
    }
