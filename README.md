# Data Quality and Model Performance

Comparing classifier behaviour on a clinical benchmark and a self-collected daily log.

**CSI 4170 / 5170 — Machine Learning, Fall 2026 · Oakland University**
Jing Yee · Christina Carvalho · Pablo Avila

---

## Research question

Does classifier performance degrade measurably when the same pipeline is applied to a
small, self-reported dataset rather than a large, curated benchmark — and how much of any
gap can be recovered through data refinement?

## Approach

Two datasets are processed through an **identical pipeline**. Only the input differs.

| | Benchmark | Self-collected |
|---|---|---|
| Source | Wisconsin Diagnostic Breast Cancer (scikit-learn) | our own daily logs |
| Rows | 569 | ~150 |
| Features | 30 numeric | 5 numeric |
| Target | malignant / benign | high / low stress |
| Collection | clinical imaging | self-reported, once per day |

Each dataset is run through a **cumulative refinement ladder**, with metrics recorded at
every stage so the contribution of each step is measured rather than assumed:

```
stage 0  baseline (raw features, default settings)
stage 1  + feature scaling
stage 2  + outlier removal (training data only)
stage 3  + feature selection
stage 4  + hyperparameter tuning
```

## Algorithms

Naive Bayes · Logistic Regression · Support Vector Machine (RBF)

## Repository layout

```
code/      pipeline.py, experiments.py, refinement.py
data/      the self-collected log (anonymised) + its data dictionary
results/   output tables and figures
docs/      phase documents, architecture diagram
```

## Running it

```python
exec(open("code/pipeline.py").read())
exec(open("code/refinement.py").read())

X, y = load_benchmark()
refinement_ladder(X, y, "Benchmark (Wisconsin breast cancer)", select_k=15)
```

No download is required for the benchmark — it ships with scikit-learn.

## Data availability

The self-collected dataset is in `data/`. Contributors are identified by participant code
(P1, P2, P3) rather than by name. See `data/LICENSE-DATA.md` for licensing.

## Reproducibility

`random_state = 42` throughout. Every split, subsample and model is seeded, so a clean run
reproduces the reported numbers exactly.
