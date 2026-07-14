"""CI tests: artifact integrity, model sanity, and API contract.

These run on every push via GitHub Actions. They test the committed
artifacts in model/, not a fresh training run, so a bad artifact or a
broken API change blocks deployment.
"""

import json
import os
import sys

import joblib
import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, "model")

# A plausible low-risk client: high limit, no delinquency, steady payments.
LOW_RISK = {
    "LIMIT_BAL": 300000, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 35,
    "PAY_1": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 5000, "BILL_AMT2": 4800, "BILL_AMT3": 5100,
    "BILL_AMT4": 4900, "BILL_AMT5": 5000, "BILL_AMT6": 5200,
    "PAY_AMT1": 5000, "PAY_AMT2": 4800, "PAY_AMT3": 5100,
    "PAY_AMT4": 4900, "PAY_AMT5": 5000, "PAY_AMT6": 5200,
}

# A plausible high-risk client: low limit, months of delinquency, no payments.
HIGH_RISK = {
    "LIMIT_BAL": 20000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 24,
    "PAY_1": 3, "PAY_2": 3, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
    "BILL_AMT1": 19000, "BILL_AMT2": 18500, "BILL_AMT3": 18000,
    "BILL_AMT4": 17500, "BILL_AMT5": 17000, "BILL_AMT6": 16500,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


@pytest.fixture(scope="module")
def artifacts():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "credit_mlp.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        features = json.load(f)
    return model, scaler, features


@pytest.fixture(scope="module")
def client():
    from app import app
    return TestClient(app)


def test_artifacts_exist():
    for name in ("credit_mlp.keras", "scaler.joblib", "feature_names.json", "metrics.json"):
        assert os.path.exists(os.path.join(MODEL_DIR, name)), f"missing {name}"


def test_model_expects_23_features(artifacts):
    model, scaler, features = artifacts
    assert len(features) == 23
    assert model.input_shape[-1] == 23
    assert scaler.n_features_in_ == 23


def test_prediction_is_valid_probability(artifacts):
    model, scaler, features = artifacts
    row = np.array([[LOW_RISK[name] for name in features]], dtype=float)
    prob = float(model.predict(scaler.transform(row), verbose=0).ravel()[0])
    assert 0.0 <= prob <= 1.0


def test_risk_ordering(artifacts):
    """The delinquent low-limit client must score riskier than the clean one."""
    model, scaler, features = artifacts
    rows = np.array(
        [[LOW_RISK[n] for n in features], [HIGH_RISK[n] for n in features]],
        dtype=float,
    )
    probs = model.predict(scaler.transform(rows), verbose=0).ravel()
    assert probs[1] > probs[0]


def test_shipped_metrics_meet_floor():
    """Guardrail: don't deploy an artifact worse than the Module 8 baseline."""
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        m = json.load(f)
    assert m["roc_auc"] >= 0.75
    assert m["recall_default"] >= 0.50


def test_api_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_predict(client):
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
