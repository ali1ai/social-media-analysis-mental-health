"""Metrics, bootstrap confidence intervals and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


def classification_metrics(y_true, y_pred, proba=None, labels=None) -> dict:
    """Headline metrics. Macro-F1 is primary because classes are imbalanced
    and the minority classes (e.g. Suicidal) are the ones that matter most."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    if proba is not None and labels is not None:
        out["log_loss"] = log_loss(y_true, proba, labels=labels)
        out["ece"] = expected_calibration_error(y_true, proba, labels)
        out["brier"] = multiclass_brier(y_true, proba, labels)
    return {k: float(v) for k, v in out.items()}


def bootstrap_ci(
    y_true, y_pred, metric=None, n_boot: int = 1000, alpha: float = 0.05, random_state: int = 42
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a metric on a fixed test set.

    Captures test-set sampling uncertainty only (not training variability).
    """
    metric = metric or (lambda a, b: f1_score(a, b, average="macro"))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(random_state)
    n = y_true.size
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_pred)), float(lo), float(hi)


def _one_hot(y_true, labels) -> np.ndarray:
    idx = {c: i for i, c in enumerate(labels)}
    y = np.asarray(y_true)
    oh = np.zeros((y.size, len(labels)))
    oh[np.arange(y.size), [idx[v] for v in y]] = 1.0
    return oh


def multiclass_brier(y_true, proba, labels) -> float:
    return float(np.mean(np.sum((np.asarray(proba) - _one_hot(y_true, labels)) ** 2, axis=1)))


def reliability_table(y_true, proba, labels, n_bins: int = 15) -> pd.DataFrame:
    """Top-label (confidence) reliability bins."""
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = np.asarray(labels)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append(
                {
                    "bin_low": edges[b],
                    "bin_high": edges[b + 1],
                    "n": int(m.sum()),
                    "confidence": float(conf[m].mean()),
                    "accuracy": float(correct[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, proba, labels, n_bins: int = 15) -> float:
    tab = reliability_table(y_true, proba, labels, n_bins)
    if tab.empty:
        return float("nan")
    w = tab["n"] / tab["n"].sum()
    return float((w * (tab["accuracy"] - tab["confidence"]).abs()).sum())
