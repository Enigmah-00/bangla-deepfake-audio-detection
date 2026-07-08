from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np


ASSETS = ROOT / "assets"


def write_logo() -> None:
    logo = """<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180" viewBox="0 0 720 180" role="img" aria-labelledby="title desc">
  <title id="title">KonthoProhori logo</title>
  <desc id="desc">A shield mark with waveform bars and the KonthoProhori wordmark.</desc>
  <rect width="720" height="180" rx="8" fill="#ffffff"/>
  <path d="M90 24l54 20v42c0 36-20 56-54 72-34-16-54-36-54-72V44l54-20z" fill="#0f766e"/>
  <path d="M66 93h10v-26h-10v26zm18 18h10V50H84v61zm18-9h10V64h-10v38zm18-17h10V73h-10v12z" fill="#ffffff"/>
  <text x="174" y="80" font-family="Inter, Arial, sans-serif" font-size="40" font-weight="800" fill="#18212f">KonthoProhori</text>
  <text x="176" y="116" font-family="Inter, Arial, sans-serif" font-size="20" fill="#64748b">Bangla Deepfake Audio Detection</text>
</svg>
"""
    (ASSETS / "logo.svg").write_text(logo, encoding="utf-8")


def write_confusion_matrix() -> None:
    matrix = np.array([[2535, 15], [7, 2545]])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted real", "Predicted fake"])
    ax.set_yticks([0, 1], labels=["Actual real", "Actual fake"])
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            ax.text(col, row, str(value), ha="center", va="center", color="#18212f", fontweight="bold")
    ax.set_title("Notebook random-split confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def write_architecture() -> None:
    steps = [
        "Upload / Record",
        "Validate",
        "Mono 16 kHz",
        "227 Features",
        "Scaler",
        "XGBoost",
        "Report",
    ]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(steps))
    for index, (x, label) in enumerate(zip(xs, steps)):
        ax.add_patch(plt.Rectangle((x - 0.055, 0.42), 0.11, 0.20, fill=True, color="#eef6ff", ec="#2563eb", lw=1.2))
        ax.text(x, 0.52, label, ha="center", va="center", fontsize=8, color="#18212f", fontweight="bold")
        if index < len(steps) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.065, 0.52), xytext=(x + 0.065, 0.52), arrowprops={"arrowstyle": "->", "color": "#64748b"})
    ax.set_title("KonthoProhori inference flow", color="#18212f", fontweight="bold")
    fig.savefig(ASSETS / "architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    write_logo()
    write_confusion_matrix()
    write_architecture()
    print("Generated assets/logo.svg, assets/confusion_matrix.png, and assets/architecture.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
