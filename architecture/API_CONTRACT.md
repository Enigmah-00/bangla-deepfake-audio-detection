# API Contract

Base URL:

```text
https://<backend-domain>
```

## `GET /api/health`

Checks whether the backend can load the model assets.

Response:

```json
{
  "status": "ok",
  "project": "KonthoProhori",
  "model_ready": true,
  "error": null
}
```

## `GET /api/metadata`

Returns static product and model metadata used by clients or deployment checks.

Includes:

- project name;
- sample rate;
- analysis duration;
- supported extensions;
- feature groups;
- dataset URL;
- disclaimer.

## `POST /api/analyze`

Multipart upload endpoint.

Request:

```text
Content-Type: multipart/form-data
field name: file
```

Success response shape:

```json
{
  "ok": true,
  "audio": {
    "filename": "clip.wav",
    "duration_seconds": 4.2,
    "analyzed_seconds": 4.2,
    "sample_rate": 16000
  },
  "prediction": {
    "label": "Likely Deepfake",
    "fake_probability": 0.91,
    "real_probability": 0.09,
    "band": "high-risk"
  },
  "evidence": {},
  "visualizations": {},
  "report": {}
}
```

Error response shape:

```json
{
  "detail": {
    "code": "unsupported_format",
    "message": "Unsupported audio format."
  }
}
```

The API never returns raw audio or absolute server paths.

