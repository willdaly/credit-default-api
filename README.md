---
title: Credit Default Prediction API
emoji: 💳
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Credit Default Prediction API

FastAPI deployment of the tuned MLP from the AAI6610 credit risk project.
The model predicts the probability that a credit card client defaults on
their payment next month, trained on the Default of Credit Card Clients
dataset (Yeh & Lien, 2009): 30,000 clients, 23 features, binary target.

## Model

Feed-forward MLP (Keras): 23 inputs → Dense(64, ReLU) → Dropout(0.3) →
Dense(32, ReLU) → Dropout(0.3) → sigmoid output. Adam at the tuned learning
rate of 0.003, class weights for the 22% minority default class, early
stopping on validation AUC. 80/20 stratified split, random_state=2026.

Held-out test metrics of the shipped artifact: accuracy 0.781, ROC AUC
0.781, default-class precision 0.505, recall 0.593, F1 0.546.

## API

- `GET /` — health check, expected feature list, shipped test metrics
- `POST /predict` — JSON body with the 23 client features, returns
  `default_probability` and a 0/1 `prediction` at threshold 0.5
- `/docs` — FastAPI interactive documentation

Example:

```bash
curl -X POST "$SPACE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 20000, "SEX": 1, "EDUCATION": 3, "MARRIAGE": 1, "AGE": 24,
       "PAY_1": 3, "PAY_2": 3, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
       "BILL_AMT1": 19000, "BILL_AMT2": 18500, "BILL_AMT3": 18000,
       "BILL_AMT4": 17500, "BILL_AMT5": 17000, "BILL_AMT6": 16500,
       "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
       "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}'
```

## CI/CD

- `.github/workflows/ci.yml` — on every push/PR: run the pytest suite
  against the committed artifacts, then smoke-run `train.py` end to end
- `.github/workflows/deploy.yml` — after CI succeeds on main: push the
  repo to this Hugging Face Space, triggering a Docker rebuild and redeploy

## Repository layout

```
app.py              FastAPI serving app
train.py            training pipeline (reproduces the Module 8 tuned MLP)
model/              committed artifacts: credit_mlp.keras, scaler.joblib,
                    feature_names.json, metrics.json
tests/test_model.py CI test suite (artifact integrity, sanity, API contract)
data/               Default of Credit Card Clients dataset (.xls)
Dockerfile          Space container (python:3.12-slim, uvicorn on 7860)
```
