# Deployment Architecture

## Recommended Production Layout

```text
Vercel frontend
  -> NEXT_PUBLIC_API_BASE_URL
Railway backend
  -> FastAPI app
  -> model/scaler assets
```

## Backend: Railway

Deploy from the repository root.

Required settings:

```text
Root Directory: /
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Recommended variables:

```text
NIXPACKS_PYTHON_VERSION=3.11
PYTHONUNBUFFERED=1
NUMBA_CACHE_DIR=/tmp/numba
MPLCONFIGDIR=/tmp/matplotlib
```

Health check:

```text
https://<railway-domain>/api/health
```

## Frontend: Vercel

Deploy from the `frontend/` directory.

Required variable:

```text
NEXT_PUBLIC_API_BASE_URL=https://<railway-domain>
```

Do not append `/api` to this value.

## Public Demo Checklist

- Backend `/api/health` returns `status: ok`.
- Frontend loads without login.
- Upload works from the deployed frontend URL.
- Microphone recording works in a browser with permission.
- Downloaded report contains no raw audio.

