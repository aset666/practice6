from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import os
app = FastAPI(
    title="Iris ML API",
    description="A simple FastAPI service that predicts Iris flower species.",
    version="1.0.0",
)
# load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model file '{MODEL_PATH}' not found. Run train.py first."
    )
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        example=[5.1, 3.5, 1.4, 0.2],
        description=(
            "Four Iris features in order: "
            "sepal length (cm), sepal width (cm), "
            "petal length (cm), petal width (cm)"
        ),
    )
class PredictResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    probabilities: dict
@app.get("/", summary="Health check")
def root():
    """Returns a simple message confirming the API is running."""
    return {"message": "ML API is running"}
@app.post("/predict", response_model=PredictResponse, summary="Predict Iris species")
def predict(request: PredictRequest):
    if len(request.features) != 4:
        raise HTTPException(
            status_code=422,
            detail="Exactly 4 features are required.",
        )
    X = np.array(request.features).reshape(1, -1)
    class_id = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return PredictResponse(
        predicted_class_id=class_id,
        predicted_class_name=CLASS_NAMES[class_id],
        probabilities={
            name: round(float(prob), 4)
            for name, prob in zip(CLASS_NAMES, proba)
        },
    )
