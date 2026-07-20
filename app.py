from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import (
    predict,
    DEFAULT_MODEL,
)

app = FastAPI(
    title="Cyberbullying Tweet Classifier",
    description="API for classifying tweets using different NLP models.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL

class PredictionResponse(BaseModel):
    prediction: str

@app.get("/")
def home():
    return {
        "message": "Cyberbullying Tweet Classifier API"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
)

def classify(request: PredictionRequest):
    prediction = predict(
        request.text,
        model_name=request.model,
    )

    return PredictionResponse(
        prediction=prediction
    )
