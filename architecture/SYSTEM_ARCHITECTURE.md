# System Architecture

## Overview

KonthoProhori is split into two deployable surfaces:

- A Next.js browser frontend for upload, microphone recording, result display,
  chart rendering, and JSON report download.
- A FastAPI backend for audio validation, preprocessing, feature extraction,
  model inference, evidence summaries, visualization generation, and report
  construction.

The trained model assets are kept in `models/` and are not modified during
runtime.

## Component Boundaries

| Component | Path | Responsibility |
|---|---|---|
| Frontend | `frontend/` | User workflow, microphone capture, upload, result display |
| API app | `backend/main.py` | HTTP endpoints, safe errors, CORS, response shaping |
| Audio IO | `src/audio_io.py` | Size/type checks, decode, mono conversion, resampling, silence checks |
| Features | `src/features.py` | Notebook-compatible 227-dimensional feature extraction |
| Predictor | `src/predictor.py` | Cached Joblib loading, scaler transform, XGBoost probability |
| Explainability | `src/explainability.py` | Grouped acoustic evidence summaries |
| Visualization | `src/visualization.py` | Waveform, spectrogram, probability, and evidence charts |
| Reporting | `src/reporting.py` | Downloadable JSON report without raw audio |

## Runtime Flow

1. Browser sends an audio file to `POST /api/analyze`.
2. Backend validates file type, file size, duration, and signal loudness.
3. Audio is converted to mono 16 kHz and clipped to the first five seconds.
4. The exact 227-feature vector is extracted.
5. The saved StandardScaler transforms the feature row.
6. The saved XGBoost classifier returns real/fake probabilities.
7. The backend returns a three-way decision, charts, evidence summary, and JSON
   report payload.

## Reliability Decisions

- The model and scaler are loaded once per process through an LRU-cached loader.
- User-safe exceptions are converted into structured HTTP errors.
- Raw audio and absolute server paths are excluded from reports.
- Local XGBoost contribution calls are disabled by default for deployment
  reliability with the existing serialized model.

