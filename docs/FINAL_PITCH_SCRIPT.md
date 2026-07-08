# Final Pitch Script

## 0:00-0:30: Problem

Bangla voice deepfakes can impersonate trusted people and institutions. A fake voice message can trigger panic, scams, misinformation, or reputational harm before the original source is checked.

## 0:30-1:05: Users And National Relevance

KonthoProhori targets journalists, fact-checkers, public communication teams, cybercrime units, social-media moderators, financial institutions, and ordinary Bangla-speaking citizens. This fits Track E because public-information integrity is a national-security concern.

## 1:05-2:15: Product Demo

Show upload and microphone recording. Analyze a known sample, show the result card, probabilities, waveform, spectrogram, acoustic evidence, and JSON report download.

## 2:15-3:10: Technical Architecture

The backend validates audio, converts it to mono 16 kHz, analyzes the first five seconds, extracts 227 acoustic features, applies the saved StandardScaler, and scores the saved XGBoost classifier. The model assets are loaded once per backend process.

## 3:10-3:55: Dataset And Results

The notebook used the Bangla DeepFake Dataset with 25,506 clips. It reports 99.57% accuracy on a random stratified held-out split. We state this carefully and do not claim universal real-world accuracy.

## 3:55-4:35: Privacy, Ethics, And Limitations

The tool is for screening, not legal proof. It does not identify the speaker, does not intentionally retain uploaded audio, and includes an inconclusive band for uncertain cases.

## 4:35-5:00: Roadmap And Close

Next steps are external challenge-set evaluation, speaker-disjoint validation, robustness checks, and segment-level analysis for longer clips. KonthoProhori lowers the barrier for responsible Bangla audio verification.

