"""Credit Default Prediction API.

Serves the Module 8 tuned MLP over a REST interface (FastAPI), following the
pattern from the Module 10 lecture. Endpoints:

- GET  /            health check and model metadata
- POST /predict     JSON body with the 23 client features, returns the
                    default probability and a 0/1 prediction at threshold 0.5
"""

import json
import os

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "credit_mlp.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    TEST_METRICS = json.load(f)

app = FastAPI(
    title="Credit Default Prediction API",
    description=(
        "Tuned MLP trained on the Default of Credit Card Clients dataset "
        "(Yeh & Lien, 2009). Predicts probability of default next month. "
        "Continuously deployed from GitHub Actions."
    ),
)


class Client(BaseModel):
    """One credit card client. Field names follow the dataset columns."""

    LIMIT_BAL: float = Field(..., description="Credit limit (NT$)")
    SEX: int = Field(..., description="1=male, 2=female")
    EDUCATION: int = Field(..., description="1=grad school, 2=university, 3=high school, 4=other")
    MARRIAGE: int = Field(..., description="1=married, 2=single, 3=other")
    AGE: int = Field(..., description="Age in years")
    PAY_1: int = Field(..., description="Repayment status, most recent month")
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "Tuned MLP (Module 8 configuration)",
        "features_expected": FEATURE_NAMES,
        "held_out_test_metrics": TEST_METRICS,
    }


@app.post("/predict")
def predict(client: Client):
    payload = client.model_dump()
    row = np.array([[payload[name] for name in FEATURE_NAMES]], dtype=float)
    prob = float(model.predict(scaler.transform(row), verbose=0).ravel()[0])
    return {
        "default_probability": round(prob, 4),
        "prediction": int(prob >= 0.5),
        "threshold": 0.5,
    }
