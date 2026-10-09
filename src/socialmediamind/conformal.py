"""Split conformal prediction sets for multi-class classifiers.

Given a held-out calibration set that is exchangeable with future data, the
returned prediction sets contain the true label with probability >= 1 - alpha
(marginal guarantee). The class-conditional ("Mondrian") variant gives the
same guarantee *within each class*, which matters here: a model can hit 90%
coverage overall while under-covering the rare, high-stakes Suicidal class.

Two non-conformity scores are provided:

* ``"lac"`` - least-ambiguous set-valued classifier, s(x, y) = 1 - p_y(x).
  Smallest average sets, but sizes adapt poorly to hard inputs.
* ``"aps"`` - adaptive prediction sets, s(x, y) = probability mass of all
  classes ranked *above* y plus U * p_y with U ~ Uniform(0, 1) (Romano et al.,
  2020). More adaptive: hard inputs get bigger sets, easy ones smaller. The
  randomisation matters: without it, a confident model piles every class's
  cumulative mass up near 1.0 and the sets balloon. A fixed seed keeps
  results reproducible.

Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification" (2021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORES = ("lac", "aps")


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    if n == 0:
        return np.inf  # no calibration data -> include everything (conservative)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return np.inf  # too few points for this alpha -> trivial (full) sets
    return float(np.quantile(scores, level, method="higher"))


def score_matrix(proba: np.ndarray, score: str = "lac", u: np.ndarray | None = None) -> np.ndarray:
    """Non-conformity score s(x, k) for every sample and every candidate class.

    For APS, ``u`` is a per-sample uniform draw; ``None`` gives the
    deterministic (u = 1) variant.
    """
    proba = np.asarray(proba, dtype=float)
    if score == "lac":
        return 1.0 - proba
    if score == "aps":
        order = np.argsort(-proba, axis=1, kind="stable")
        sorted_p = np.take_along_axis(proba, order, axis=1)
        cum_before = np.cumsum(sorted_p, axis=1) - sorted_p
        u = np.ones((proba.shape[0], 1)) if u is None else np.asarray(u, dtype=float).reshape(-1, 1)
        out = np.empty_like(proba)
        np.put_along_axis(out, order, cum_before + u * sorted_p, axis=1)
        return out
    raise ValueError(f"score must be one of {SCORES}")


class SplitConformalClassifier:
    def __init__(
        self,
        labels,
        alpha: float = 0.10,
        class_conditional: bool = False,
        score: str = "lac",
        randomized: bool = True,
        random_state: int = 0,
    ):
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")
        self.labels = np.asarray(labels)
        self.alpha = alpha
        self.class_conditional = class_conditional
        self.score = score
        self.randomized = randomized
        self.random_state = random_state

    def _u(self, n: int, offset: int) -> np.ndarray | None:
        if self.score != "aps" or not self.randomized:
            return None
        return np.random.default_rng(self.random_state + offset).uniform(size=n)

    def _y_idx(self, y):
        idx = {c: i for i, c in enumerate(self.labels)}
        return np.array([idx[v] for v in np.asarray(y)])

    def fit(self, proba_cal, y_cal):
        y_idx = self._y_idx(y_cal)
        s = score_matrix(proba_cal, self.score, self._u(y_idx.size, 0))[np.arange(y_idx.size), y_idx]
        if self.class_conditional:
            self.qhat_ = np.array(
                [_conformal_quantile(s[y_idx == k], self.alpha) for k in range(len(self.labels))]
            )
        else:
            self.qhat_ = np.full(len(self.labels), _conformal_quantile(s, self.alpha))
        self.n_cal_per_class_ = np.bincount(y_idx, minlength=len(self.labels))
        return self

    def predict_sets(self, proba) -> np.ndarray:
        """Boolean matrix (n_samples, n_classes): True = label in the set."""
        proba = np.asarray(proba)
        return score_matrix(proba, self.score, self._u(proba.shape[0], 1)) <= self.qhat_[None, :]

    def sets_as_labels(self, proba) -> list[list[str]]:
        return [[str(c) for c in self.labels[row]] for row in self.predict_sets(proba)]


def coverage_report(sets: np.ndarray, y_true, labels) -> pd.DataFrame:
    """Per-class and overall empirical coverage and mean set size."""
    labels = np.asarray(labels)
    idx = {c: i for i, c in enumerate(labels)}
    y_idx = np.array([idx[v] for v in np.asarray(y_true)])
    covered = sets[np.arange(y_idx.size), y_idx]
    size = sets.sum(axis=1)
    rows = []
    for k in range(len(labels)):
        m = y_idx == k
        rows.append(
            {
                "class": str(labels[k]),
                "n": int(m.sum()),
                "coverage": float(covered[m].mean()) if m.any() else np.nan,
                "mean_set_size": float(size[m].mean()) if m.any() else np.nan,
            }
        )
    rows.append(
        {"class": "ALL", "n": int(y_idx.size), "coverage": float(covered.mean()), "mean_set_size": float(size.mean())}
    )
    return pd.DataFrame(rows)
