"""Train the tuned credit-default MLP and save the serving artifacts.

Reproduces the Module 8 tuned configuration:
- Default of Credit Card Clients dataset (Yeh & Lien, 2009), 30,000 rows
- 80/20 stratified train/test split, random_state=2026
- 15% validation carve-out from the training portion for early stopping
- StandardScaler fit on the training partition only
- MLP: 23 inputs -> Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu)
  -> Dropout(0.3) -> Dense(1, sigmoid)
- Adam, learning rate 0.003 (tuned), binary cross-entropy, batch size 64
- Class weights inversely proportional to class frequency (training only)
- Early stopping on validation AUC, patience 5, restore best weights

Artifacts written to model/:
- credit_mlp.keras   (trained network)
- scaler.joblib      (fitted StandardScaler)
- feature_names.json (input feature order the API must respect)
- metrics.json       (held-out test metrics for the shipped artifact)
"""

import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 2026
DATA_PATH = os.path.join("data", "default_of_credit_card_clients.xls")
MODEL_DIR = "model"


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(path: str = DATA_PATH):
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1"})
    y = df["default payment next month"]
    X = df.drop(columns=["ID", "default payment next month"])
    return X, y


def build_model(n_features: int, learning_rate: float = 0.003) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    set_seeds()
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    scaler = StandardScaler().fit(X_tr.values)
    X_tr_s = scaler.transform(X_tr.values)
    X_val_s = scaler.transform(X_val.values)
    X_test_s = scaler.transform(X_test.values)

    # Class weights inversely proportional to class frequency (training only).
    n = len(y_tr)
    n_pos = int(y_tr.sum())
    n_neg = n - n_pos
    class_weight = {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}

    model = build_model(X.shape[1])
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )
    model.fit(
        X_tr_s,
        y_tr,
        validation_data=(X_val_s, y_val),
        epochs=50,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early],
        verbose=2,
    )

    prob = model.predict(X_test_s, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 3),
        "precision_default": round(float(precision_score(y_test, pred)), 3),
        "recall_default": round(float(recall_score(y_test, pred)), 3),
        "f1_default": round(float(f1_score(y_test, pred)), 3),
    }
    print("Held-out test metrics:", metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "credit_mlp.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(list(X.columns), f, indent=2)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Artifacts saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
