# KonthoProhori

Bangla Deepfake Audio Detection and Voice Forensics.

| Item | Status |
|---|---|
| Live Demo | TBD |
| Demo Video | TBD |
| Track | Track E: National Defence |
| Main Result | 99.57% on the notebook random held-out split |
| Important Limitation | Cross-speaker, cross-generator, and real-world validation remains future work |

KonthoProhori is a privacy-aware Bangla voice-forensics web application for screening suspicious speech clips. It uses the existing trained XGBoost model and 227 handcrafted acoustic features to estimate whether an uploaded or recorded voice clip resembles genuine or AI-generated Bangla speech.

## Current Product

- FastAPI backend for audio validation, feature extraction, inference, explainability, charts, and JSON reports.
- Next.js frontend for upload, browser microphone recording, result cards, probabilities, acoustic evidence, and report download.
- Exact notebook-compatible 227-feature layout.
- Existing Joblib assets preserved without retraining.
- Honest uncertainty bands: Likely Genuine, Inconclusive, Likely Deepfake.

## Why It Matters

Bangla voice deepfakes can support impersonation, scams, fabricated emergency announcements, public-information manipulation, and reputational attacks. The tool is designed as a first-line screening and triage aid for journalists, fact-checkers, government communication teams, cybercrime units, moderators, institutions, and citizens.

This project is a decision-support system. It is not legal, forensic, or scientific proof.

## Architecture

```text
Browser / Next.js
  -> POST /api/analyze
FastAPI backend
  -> validate file
  -> decode audio
  -> mono 16 kHz
  -> first 5 seconds
  -> 227 acoustic features
  -> saved StandardScaler
  -> saved XGBoost classifier
  -> probabilities, evidence, charts, JSON report
```

## AI Pipeline

The inference pipeline preserves the training notebook feature order:

| Index range | Feature family | Dimensions |
|---|---|---:|
| `0:40` | MFCC mean | 40 |
| `40:80` | MFCC standard deviation | 40 |
| `80:92` | Chroma mean | 12 |
| `92:220` | Mel-spectrogram mean | 128 |
| `220:227` | Spectral-contrast mean | 7 |

The model output is shown as a model probability, not guaranteed confidence.

## Dataset

Dataset reference: [Bangla DeepFake Dataset by ahanaf101 on Kaggle](https://www.kaggle.com/datasets/ahanaf101/bangla-deepfake-dataset).

The existing notebook reports 25,506 audio files: 12,749 real and 12,757 fake. Dataset license status was not verified from accessible page content during implementation and should be confirmed manually before final submission.

## Reported Evaluation

The notebook reports:

- Accuracy: 99.57%
- Confusion matrix:

```text
[[2535, 15],
 [   7, 2545]]
```

Important caveat: this is a random stratified hold-out result. It should not be presented as real-world accuracy across all Bangla audio.

## Local Setup

Use Python 3.11 or 3.12 for the backend.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```

Open the Next.js local URL shown by `npm run dev`.

## Tests

```bash
python -m pytest
python scripts/verify_repository.py
python scripts/smoke_test.py --synthetic
```

For the strict smoke test, add consented files:

```text
examples/real_sample.wav
examples/fake_sample.wav
```

Then run:

```bash
python scripts/smoke_test.py
```

## Repository Structure

```text
app.py
backend/main.py
frontend/
src/
tests/
scripts/
models/
notebooks/
examples/
assets/
artifacts/
docs/
```

## Privacy And Ethics

- Uploaded audio is processed for the request and is not intentionally retained by the application.
- Uploaded audio is not used for retraining.
- Reports exclude raw audio and absolute server paths.
- The tool does not identify who is speaking.
- The result must not be used for automatic punishment, censorship, prosecution, or other unattended high-stakes decisions.

## Deployment Notes

Recommended deployment:

- Backend: Render, Railway, Fly.io, Hugging Face Docker Space, or another Python service that supports FFmpeg.
- Frontend: Vercel, Netlify, or a static Next.js-capable host.
- Set `NEXT_PUBLIC_API_BASE_URL` to the public FastAPI backend URL.
- Confirm CORS policy before final deployment.

## Team

TBD.

## License

TBD by the team. Third-party dataset and library licenses must be confirmed before final public submission.

