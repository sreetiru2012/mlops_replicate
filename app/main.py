from prometheus_client import Counter, generate_latest
from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os

from src.inference.predictor import Predictor

app = FastAPI(title="Cats vs Dogs API")

PRED_REQUESTS = Counter("prediction_requests_total", "Total prediction requests")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
predictor = Predictor(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predictor.predict(image)
    PRED_REQUESTS.inc()
    return result

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")