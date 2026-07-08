# KonthoProhori Project Report Source

## Problem Statement

AI-generated Bangla speech can imitate public officials, family members, journalists, executives, or security personnel. This creates risk for misinformation, scams, reputational harm, fake emergency messages, and public-trust manipulation. Bangla-speaking users need accessible screening tools that communicate uncertainty clearly.

## Proposed Solution

KonthoProhori is a web application for AI-assisted Bangla voice deepfake screening. Users upload or record audio, and the system returns a three-way result: Likely Genuine, Inconclusive, or Likely Deepfake. The interface also shows probabilities, waveform, spectrogram, grouped acoustic evidence, and a downloadable JSON report.

## Methodology

The system validates audio, converts it to mono, resamples to 16 kHz, and analyzes up to the first five seconds. It extracts 227 notebook-compatible features: MFCC means, MFCC variability, chroma means, mel-spectrogram means, and spectral contrast means. The saved StandardScaler transforms the feature vector before inference.

## AI/ML Approach

The classifier is a saved XGBoost binary model trained in the existing notebook. The project does not retrain the model during inference. The backend loads the scaler and model once per process, applies `predict_proba`, and uses heuristic product-level thresholds: fake probability at or above 0.70 is Likely Deepfake, at or below 0.30 is Likely Genuine, and the middle band is Inconclusive.

## Results

The notebook reports 99.57% accuracy on a random stratified held-out split with confusion matrix:

```text
[[2535, 15],
 [   7, 2545]]
```

This metric is not presented as universal field accuracy. Broader speaker-disjoint, cross-generator, and real-world validation is future work.

## Implementation

The product uses a FastAPI backend and Next.js frontend. The backend handles validation, feature extraction, inference, explainability, chart generation, and JSON report construction. The frontend provides upload, microphone recording, analysis state, safe errors, model probability display, charts, and report download.

## Privacy And Ethics

The application processes audio for the request and does not intentionally store uploaded recordings. It does not identify the speaker and must not be used as standalone proof, automatic punishment, or legal evidence. Users are encouraged to verify suspicious clips through original sources and official channels.

## Limitations And Future Work

Future work includes speaker-disjoint evaluation, external challenge-set testing, robustness checks under noise and compression, multi-segment analysis for longer audio, and comparison with pretrained speech representation models such as wav2vec-style systems.

## References

- Bangla DeepFake Dataset: <https://www.kaggle.com/datasets/ahanaf101/bangla-deepfake-dataset>
- XGBoost
- Librosa
- FastAPI
- Next.js

