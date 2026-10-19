"""
CSI-4170/5170 Final Project — shared pipeline

The SAME run_pipeline() is applied to both datasets.
Only the data going in is different. That is the design of the study.

Benchmark : Wisconsin Diagnostic Breast Cancer (scikit-learn, raw/unprocessed)
Ours      : the group's daily log, exported from Google Sheets
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

RANDOM_STATE = 42


def algorithms():
    """Three algorithms, all named in the course learning outcomes."""
    return {
        "Naive Bayes":         GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF)":           SVC(kernel="rbf", random_state=RANDOM_STATE),
    }


# ============================================================
# THE PIPELINE  — identical for every dataset
# ============================================================

def run_pipeline(X, y, dataset_name, test_size=0.2, scale=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    if scale:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)    # fit on TRAIN only
        X_test = sc.transform(X_test)          # test is only transformed

    rows = []
    print("=" * 64)
    print(dataset_name, " | train:", X_train.shape, " test:", X_test.shape)
    print("=" * 64)

    for name, model in algorithms().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        rows.append({
            "dataset":   dataset_name,
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred, average="macro", zero_division=0), 3),
            "recall":    round(recall_score(y_test, pred, average="macro", zero_division=0), 3),
            "f1":        round(f1_score(y_test, pred, average="macro", zero_division=0), 3),
            "cv_mean":   round(cv.mean(), 3),
            "cv_std":    round(cv.std(), 3),
        })

        print()
        print(name)
        print("  test accuracy :", round(accuracy_score(y_test, pred), 3))
        print("  5-fold CV     :", round(cv.mean(), 3), "+/-", round(cv.std(), 3))
        print("  confusion matrix (rows = true, cols = predicted):")
        print(confusion_matrix(y_test, pred))

    print()
    return pd.DataFrame(rows)


# ============================================================
# DATASET 1 — the public benchmark (available now)
# ============================================================

def load_benchmark():
    """Wisconsin Diagnostic Breast Cancer. Natively two-class, raw feature values."""
    d = load_breast_cancer()
    X, y = d.data, d.target
    print("Benchmark:", X.shape[0], "rows,", X.shape[1], "features")
    print("Classes  :", dict(zip(d.target_names, np.bincount(y))))
    print("Raw value range:", round(float(X.min()), 3), "to", round(float(X.max()), 3),
          "  <- very different scales, so preprocessing matters here")
    print()
    return X, y


# ============================================================
# DATASET 2 — our own logs (available in November)
# ============================================================

FEATURES = ["Sleep (hours)", "Exercise (min)", "Screen time (hours)",
            "Study (hours)", "Caffeine (cups)"]
TARGET = "Stress (1-10)"
STRESS_CUTOFF = 6          # 6 or above = high stress


def load_our_logs(csv_path, anonymize=True):
    """Read the exported group log and shape it like the benchmark."""
    df = pd.read_csv(csv_path)

    if anonymize:
        df["Person"] = df["Person"].map({"Jing": "P1", "Christina": "P2", "Pablo": "P3"})

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].values
    y = (df[TARGET] >= STRESS_CUTOFF).astype(int).values

    print("Our logs:", X.shape[0], "rows,", X.shape[1], "features")
    print("Class balance (0 = low stress, 1 = high):", np.bincount(y))
    print()
    return X, y


if __name__ == "__main__":
    X_bench, y_bench = load_benchmark()
    results = run_pipeline(X_bench, y_bench, "Benchmark (Wisconsin breast cancer)")
    print("SUMMARY")
    print(results.to_string(index=False))
