# Inference Pipeline

## Input Contract

Supported file extensions:

```text
.wav, .mp3, .flac, .ogg, .m4a, .webm
```

The backend rejects unsupported, missing, empty, oversized, too-short, corrupt,
or near-silent files with user-safe error messages.

## Preprocessing

- Decode audio.
- Convert stereo or multichannel audio to mono.
- Resample to 16 kHz.
- Require at least 0.50 seconds.
- Analyze the first 5.0 seconds.
- Reject near-silent audio using RMS and peak thresholds.

## Feature Schema

The production inference code preserves the notebook feature layout exactly:

| Index range | Feature family | Dimensions |
|---|---|---:|
| `0:40` | MFCC mean | 40 |
| `40:80` | MFCC standard deviation | 40 |
| `80:92` | Chroma mean | 12 |
| `92:220` | Mel-spectrogram mean | 128 |
| `220:227` | Spectral-contrast mean | 7 |
| Total |  | 227 |

## Model Path

```text
audio -> 227 features -> saved StandardScaler -> saved XGBoost classifier
```

The app uses heuristic product decision bands:

| Fake probability | Decision |
|---:|---|
| `>= 0.70` | Likely Deepfake |
| `<= 0.30` | Likely Genuine |
| otherwise | Inconclusive |

These are product-level uncertainty bands, not newly trained thresholds.

