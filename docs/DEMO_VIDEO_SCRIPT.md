# Demo Video Script

## 0:00-0:25: Problem

Bangla voice messages are trusted in families, workplaces, media, and public communication. With modern voice cloning, a fake clip can impersonate a public figure, security officer, relative, or executive and spread quickly before anyone verifies the source.

## 0:25-0:50: Solution

KonthoProhori is a Bangla deepfake audio screening tool. It helps users upload or record a suspicious clip, see a model score, inspect acoustic evidence, and understand uncertainty before sharing or acting.

## 0:50-1:45: Genuine-Audio Demo

Upload a known genuine Bangla sample. Show the waveform, log-mel spectrogram, three-way result, genuine and deepfake model probabilities, grouped acoustic evidence, and disclaimer.

## 1:45-2:40: Deepfake-Audio Demo

Upload a known synthetic Bangla sample. Explain that the output is a model probability, not proof. Emphasize source verification.

## 2:40-3:20: Technical Pipeline

Show the architecture: validation, mono conversion, 16 kHz resampling, first five seconds, 227 acoustic features, saved StandardScaler, XGBoost classifier, uncertainty band, explainability, and downloadable report.

## 3:20-4:00: Results And Honesty

Show the notebook result: 99.57% accuracy on a random held-out split. State that broader cross-speaker, cross-generator, and real-world validation remains future work.

## 4:00-4:30: Impact And Future Work

Discuss use by journalists, fact-checkers, cybercrime units, institutions, and citizens. Mention robustness testing, external challenge sets, and safer interpretation guidance.

