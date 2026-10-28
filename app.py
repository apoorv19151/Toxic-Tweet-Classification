from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import (
    predict,
    DEFAULT_MODEL,
)

app = FastAPI(
    title="Cyberbullying Tweet Classifier",
    description="API for classifying tweets into six cyberbullying categories using different NLP models.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Tweet text to classify"
    )
    model: str = DEFAULT_MODEL

class PredictionResponse(BaseModel):
    prediction: str

@app.get("/")
def home():
    return {
        "message": "Cyberbullying Tweet Classifier API",
        "satus": "running"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=PredictionResponse,
)

def classify(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )
        
    try:
        prediction = predict(
            text,
            model_name=request.model,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    return PredictionResponse(
        prediction=prediction
    )
