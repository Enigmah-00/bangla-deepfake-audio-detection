# Model And Data Card

## Dataset

- Name: Bangla DeepFake Dataset
- Source: <https://www.kaggle.com/datasets/ahanaf101/bangla-deepfake-dataset>
- Files used in notebook: 25,506
- Real/fake distribution: 12,749 real, 12,757 fake
- Audio format: WAV files discovered under `Real` and `Fake` folders in the notebook
- License status: not verified from accessible Kaggle page content during implementation
- Known limitations: possible speaker overlap, recording-domain overlap, generator-specific artifacts, and unknown robustness under compression, noise, phone calls, or unseen synthesis systems

## Model

- Type: XGBoost binary classifier
- Input: 227 handcrafted acoustic features
- Preprocessing: mono audio, 16 kHz, first 5 seconds
- Scaler: saved StandardScaler
- Output: model probability for real/fake classes
- Assets: `models/bangla_deepfake_detector.joblib`, `models/audio_scaler.joblib`
- Reported held-out metric: 99.57% accuracy on the notebook random stratified split

## Intended Uses

- Bangla audio screening
- Media triage
- Awareness and education
- Research demonstration
- Human-in-the-loop verification support

## Out-Of-Scope Uses

- Criminal conviction
- Automatic censorship
- Biometric identity verification
- Proof of speaker identity
- Unattended high-stakes decisions
- Legal or forensic proof without external validation and chain-of-custody procedures

## Limitations

- False positives and false negatives are possible.
- Performance on unseen TTS or voice-cloning systems is not established.
- Speaker and recording-domain shift may reduce reliability.
- Compression, music, overlapping speakers, and background noise may change scores.
- The random split may overestimate field performance.
- Uploaded voice data is sensitive and must be handled with consent and care.

