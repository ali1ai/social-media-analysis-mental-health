"""Temperature scaling (Guo et al., 2017) - a one-parameter, accuracy-preserving
post-hoc calibration method. Implemented directly so it does not depend on
version-specific sklearn APIs for calibrating pre-fitted models."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


class TemperatureScaler:
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def _y_idx(self, y):
        idx = {c: i for i, c in enumerate(self.labels)}
        return np.array([idx[v] for v in np.asarray(y)])

    def fit(self, logits, y):
        logits = np.asarray(logits, dtype=float)
        y_idx = self._y_idx(y)

        def nll(log_t):
            lp = log_softmax(logits / np.exp(log_t), axis=1)
            return -lp[np.arange(y_idx.size), y_idx].mean()

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        self.temperature_ = float(np.exp(res.x))
        return self

    def predict_proba(self, logits):
        return softmax(np.asarray(logits, dtype=float) / self.temperature_, axis=1)
