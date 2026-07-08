# Security And Privacy Architecture

## Privacy Controls

- Uploaded audio is processed for the request only.
- The app does not intentionally retain uploaded recordings.
- Reports exclude raw audio and absolute server paths.
- Uploaded audio is not used for retraining.
- No user account is required for the core demo.

## Safety Controls

- The UI describes the output as AI-assisted screening.
- Results are not described as legal, forensic, or scientific proof.
- The app does not identify the speaker.
- The app uses an inconclusive band to avoid overconfident communication.
- The app encourages source verification before acting or sharing.

## Error Handling

Validation errors are converted into user-safe messages such as:

- unsupported format;
- file too large;
- empty file;
- too short;
- near-silent;
- decode failed.

Raw Python stack traces are not exposed through API responses.

## Future Hardening

- Restrict CORS to the production Vercel domain after final deployment.
- Add request-size limits at the hosting platform edge.
- Add abuse-rate protection if the public demo receives heavy traffic.
- Add a small external challenge set and robustness report.

