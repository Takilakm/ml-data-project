"""
CSI-4170/5170 Final Project — experiments

Algorithms chosen to match the course learning outcomes:
Naive Bayes (probabilistic), Logistic Regression (linear),
K-Nearest Neighbors (instance-based), SVM (non-probabilistic, kernel).

EXP 1  baseline comparison       both datasets, identical pipeline
EXP 2  learning curve            isolates SAMPLE SIZE
EXP 3  label noise               isolates NOISE
EXP 4  significance testing      is a difference real or luck?
EXP 5  regularization sweep      how regularization interacts with sample size
EXP 6  PCA comparison            does dimension reduction help on small data?
EXP 7  per-person calibration    does fixing "your 7 != my 7" help?   [our data]
EXP 8  k-means day types         unsupervised structure              [our data]
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import silhouette_score
from scipy.stats import binomtest, ttest_rel

RANDOM_STATE = 42


def models():
    """The four algorithms named in the course learning outcomes."""
    return {
        "Naive Bayes":         GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)":           SVC(kernel="rbf", random_state=RANDOM_STATE),
    }


# ============================================================
# EXP 2 — LEARNING CURVE   (isolates sample size)
# ============================================================

def learning_curve(X, y, sizes=(50, 100, 150, 200, 250, 300), n_repeats=5):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name in models():
        for n in sizes:
            if n > len(y):
                continue
            scores = []
            for _ in range(n_repeats):
                idx = rng.choice(len(y), n, replace=False)
                cv = cross_val_score(models()[name], X[idx], y[idx], cv=5)
                scores.append(cv.mean())
            rows.append({"model": name, "n": n,
                         "accuracy": round(float(np.mean(scores)), 3),
                         "std": round(float(np.std(scores)), 3)})
    return pd.DataFrame(rows)


# ============================================================
# EXP 3 — LABEL NOISE   (isolates noise)
# ============================================================

def noise_curve(X, y, levels=(0, 5, 10, 20, 30), n_repeats=5):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name in models():
        for pct in levels:
            scores = []
            for _ in range(n_repeats):
                yn = y.copy()
                k = int(len(yn) * pct / 100)
                if k:
                    flip = rng.choice(len(yn), k, replace=False)
                    yn[flip] = 1 - yn[flip]
                scores.append(cross_val_score(models()[name], X, yn, cv=5).mean())
            rows.append({"model": name, "noise_pct": pct,
                         "accuracy": round(float(np.mean(scores)), 3),
                         "std": round(float(np.std(scores)), 3)})
    return pd.DataFrame(rows)


# ============================================================
# EXP 4 — SIGNIFICANCE TESTING
# ============================================================

def mcnemar(y_true, pred_a, pred_b, label_a="A", label_b="B"):
    """Exact McNemar test on the discordant pairs."""
    a_ok = (pred_a == y_true); b_ok = (pred_b == y_true)
    n01 = int(np.sum(~a_ok & b_ok)); n10 = int(np.sum(a_ok & ~b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)
    return {"comparison": f"{label_a} vs {label_b}", "n01": n01, "n10": n10,
            "p_value": round(p, 4), "significant": bool(p < 0.05)}


def compare_all(X, y, test_size=0.2):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=RANDOM_STATE, stratify=y)
    s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
    preds = {}
    for name, m in models().items():
        m.fit(Xtr, ytr); preds[name] = m.predict(Xte)
    names = list(preds); rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append(mcnemar(yte, preds[names[i]], preds[names[j]], names[i], names[j]))
    return pd.DataFrame(rows)


def paired_ttest(X, y, name_a, name_b, cv=5):
    a = cross_val_score(models()[name_a], X, y, cv=cv)
    b = cross_val_score(models()[name_b], X, y, cv=cv)
    t, p = ttest_rel(a, b)
    return {"comparison": f"{name_a} vs {name_b}",
            "mean_a": round(float(a.mean()), 3), "mean_b": round(float(b.mean()), 3),
            "t": round(float(t), 3), "p_value": round(float(p), 4),
            "significant": bool(p < 0.05)}


# ============================================================
# EXP 5 — REGULARIZATION  x  SAMPLE SIZE
# ============================================================

def regularization_sweep(X, y, Cs=(0.01, 0.1, 1, 10, 100),
                         sizes=(50, 100, 150, 300), n_repeats=5):
    """Smaller C = stronger regularization.
    Expectation: strong regularization helps more when n is small."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for n in sizes:
        if n > len(y):
            continue
        for C in Cs:
            scores = []
            for _ in range(n_repeats):
                idx = rng.choice(len(y), n, replace=False)
                m = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)
                scores.append(cross_val_score(m, X[idx], y[idx], cv=5).mean())
            rows.append({"n": n, "C": C,
                         "accuracy": round(float(np.mean(scores)), 3)})
    return pd.DataFrame(rows).pivot(index="n", columns="C", values="accuracy")


# ============================================================
# EXP 6 — PCA
# ============================================================

def pca_comparison(X, y, components=(2, 3, 5, None)):
    """None = no PCA, all original features."""
    rows = []
    for k in components:
        Xk = X if k is None else PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(X)
        var = (1.0 if k is None else
               PCA(n_components=k, random_state=RANDOM_STATE).fit(X).explained_variance_ratio_.sum())
        for name in models():
            cv = cross_val_score(models()[name], Xk, y, cv=5)
            rows.append({"components": "all" if k is None else k,
                         "variance_kept": round(float(var), 3),
                         "model": name,
                         "accuracy": round(float(cv.mean()), 3)})
    return pd.DataFrame(rows)


# ============================================================
# EXP 7 & 8 — OUR OWN DATA
# ============================================================

FEATURES = ["Sleep (hours)", "Exercise (min)", "Screen time (hours)",
            "Study (hours)", "Caffeine (cups)"]
TARGET = "Stress (1-10)"


def labels_absolute(df, cutoff=6):
    """High stress = 6 or above, same cutoff for everyone."""
    return (df[TARGET] >= cutoff).astype(int).values


def labels_calibrated(df, person_col="Person"):
    """High stress = above THAT PERSON's own average."""
    z = df.groupby(person_col)[TARGET].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() else 1))
    return (z > 0).astype(int).values


def calibration_experiment(df, person_col="Person"):
    X = StandardScaler().fit_transform(df[FEATURES].values)
    rows = []
    for scheme, y in [("absolute cutoff", labels_absolute(df)),
                      ("per-person calibrated", labels_calibrated(df, person_col))]:
        for name in models():
            cv = cross_val_score(models()[name], X, y, cv=5)
            rows.append({"labelling": scheme, "model": name,
                         "accuracy": round(float(cv.mean()), 3),
                         "std": round(float(cv.std()), 3)})
    return pd.DataFrame(rows)


def day_types(df, k_range=(2, 3, 4, 5)):
    """Unsupervised: do our days fall into natural groups?
    Silhouette score says how well separated the clusters are."""
    X = StandardScaler().fit_transform(df[FEATURES].values)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k,
                     "silhouette": round(float(silhouette_score(X, km.labels_)), 3),
                     "inertia": round(float(km.inertia_), 1)})
    return pd.DataFrame(rows)


def profile_clusters(df, k=3):
    """What does each cluster actually look like? Average each feature per cluster."""
    X = StandardScaler().fit_transform(df[FEATURES].values)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    out = df[FEATURES + [TARGET]].copy()
    out["cluster"] = km.labels_
    return out.groupby("cluster").mean().round(2)
