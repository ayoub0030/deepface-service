import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from deepface import DeepFace
from PIL import Image
import numpy as np
import io


app = FastAPI(title="DeepFace Service", version="0.1.0")

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
cors_origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    time_ms: Optional[float] = None


def read_image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    selfie: UploadFile = File(...),
    candidate: UploadFile = File(...),
    model_name: str = os.environ.get("MODEL_NAME", "SFace"),
    detector_backend: str = os.environ.get("DETECTOR_BACKEND", "opencv"),
    distance_metric: str = os.environ.get("DISTANCE_METRIC", "cosine"),
):
    selfie_bytes = await selfie.read()
    candidate_bytes = await candidate.read()

    img1 = read_image_bytes_to_rgb(selfie_bytes)
    img2 = read_image_bytes_to_rgb(candidate_bytes)

    result = DeepFace.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )

    return VerifyResponse(
        verified=bool(result.get("verified", False)),
        distance=float(result.get("distance", 1.0)),
        threshold=float(result.get("threshold", 0.0)),
        model=str(result.get("model", model_name)),
        detector_backend=str(result.get("detector_backend", detector_backend)),
        time_ms=float(result.get("time", 0.0)) if result.get("time") is not None else None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


