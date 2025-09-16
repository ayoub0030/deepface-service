# DeepFace Local Service (FastAPI)

Run a local CPU-only verification service for the app.

## Setup

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Start the server:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

3) Test health:

```powershell
curl http://127.0.0.1:8000/health
```

## API

- POST /verify (multipart/form-data)
  - fields: `selfie` (file), `candidate` (file)
  - optional query params: `model_name` (default ArcFace), `detector_backend` (default retinaface), `distance_metric` (default cosine)
  - returns: `{ verified, distance, threshold, model, detector_backend, time_ms }`

Notes:
- `enforce_detection=False` to avoid hard errors on no-face cases.
- For Windows CPU, default wheels are fine; GPU is not required.


