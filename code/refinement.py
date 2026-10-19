"""
CSI-4170/5170 Final Project — the refinement ladder

What the professor asked for:
  1. three algorithms, run on the benchmark dataset and on our own, separately
  2. preprocessing / feature selection that MEASURABLY IMPROVES the metrics

Design: a cumulative ladder. Each stage adds one refinement and keeps the
previous ones, so the table shows exactly what each step was worth.

  STAGE 0  raw features, default settings        <- baseline
  STAGE 1  + feature scaling
  STAGE 2  + outlier removal (training data only)
  STAGE 3  + feature selection
  STAGE 4  + hyperparameter tuning               <- final

Every stage is evaluated on the SAME held-out test set, so the numbers
are directly comparable. All fitting happens inside a Pipeline, so nothing
learned from the test set ever leaks into training.
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RANDOM_STATE = 42


def algorithms():
    """Three algorithms, all named in the course learning outcomes."""
    return {
        "Naive Bayes":         GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF)":           SVC(kernel="rbf", random_state=RANDOM_STATE),
    }


PARAM_GRIDS = {
    "Naive Bayes":         {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]},
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)":           {"clf__C": [0.1, 1, 10, 100],
                            "clf__gamma": ["scale", 0.01, 0.1]},
}


# ------------------------------------------------------------
# refinement helpers
# ------------------------------------------------------------

def remove_outliers(X, y, z_thresh=3.0):
    """Drop training rows where any feature is more than z_thresh SDs from the mean.
    TRAINING DATA ONLY — never touch the test set."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    z = np.abs((X - mu) / sd)
    keep = (z < z_thresh).all(axis=1)
    return X[keep], y[keep], int((~keep).sum())


def build_pipeline(model, scale=False, select_k=None):
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if select_k is not None:
        steps.append(("select", SelectKBest(f_classif, k=select_k)))
    steps.append(("clf", model))
    return Pipeline(steps)


def metrics(y_true, pred):
    return {
        "accuracy":  round(float(accuracy_score(y_true, pred)), 3),
        "precision": round(float(precision_score(y_true, pred, average="macro", zero_division=0)), 3),
        "recall":    round(float(recall_score(y_true, pred, average="macro", zero_division=0)), 3),
        "f1":        round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 3),
    }


# ------------------------------------------------------------
# the ladder
# ------------------------------------------------------------

def refinement_ladder(X, y, dataset_name, select_k=None, verbose=True):
    """Run all five stages for all three algorithms. Returns a tidy DataFrame."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    n_features = X.shape[1]
    if select_k is None:
        select_k = max(2, n_features // 2)

    Xtr_clean, ytr_clean, n_dropped = remove_outliers(X_train, y_train)

    rows = []
    for name, model in algorithms().items():

        stages = [
            ("0 · baseline",            dict(scale=False, select=None, tune=False, clean=False)),
            ("1 · + scaling",           dict(scale=True,  select=None, tune=False, clean=False)),
            ("2 · + outlier removal",   dict(scale=True,  select=None, tune=False, clean=True)),
            ("3 · + feature selection", dict(scale=True,  select=select_k, tune=False, clean=True)),
            ("4 · + tuning",            dict(scale=True,  select=select_k, tune=True,  clean=True)),
        ]

        for stage_name, cfg in stages:
            Xt = Xtr_clean if cfg["clean"] else X_train
            yt = ytr_clean if cfg["clean"] else y_train

            pipe = build_pipeline(algorithms()[name],
                                  scale=cfg["scale"], select_k=cfg["select"])

            if cfg["tune"]:
                search = GridSearchCV(pipe, PARAM_GRIDS[name], cv=5,
                                      scoring="accuracy", n_jobs=-1)
                search.fit(Xt, yt)
                best = search.best_estimator_
                pred = best.predict(X_test)
                note = str(search.best_params_)
            else:
                pipe.fit(Xt, yt)
                pred = pipe.predict(X_test)
                note = ""

            row = {"dataset": dataset_name, "model": name, "stage": stage_name}
            row.update(metrics(y_test, pred))
            row["note"] = note
            rows.append(row)

    df = pd.DataFrame(rows)

    if verbose:
        print("=" * 78)
        print(dataset_name, f" | train {X_train.shape}  test {X_test.shape}"
              f"  | outliers dropped from train: {n_dropped}"
              f"  | features kept at stage 3+: {select_k} of {n_features}")
        print("=" * 78)
        for name in algorithms():
            sub = df[df.model == name]
            print()
            print(name)
            print(sub[["stage", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
            gain = sub.iloc[-1]["f1"] - sub.iloc[0]["f1"]
            print(f"   F1 change from baseline to final: {gain:+.3f}")
        print()

    return df


def which_features(X, y, feature_names, k):
    """Which features did SelectKBest keep, and what were their scores?"""
    sel = SelectKBest(f_classif, k=k).fit(StandardScaler().fit_transform(X), y)
    out = pd.DataFrame({"feature": feature_names,
                        "F_score": np.round(sel.scores_, 2),
                        "kept": sel.get_support()})
    return out.sort_values("F_score", ascending=False).reset_index(drop=True)
