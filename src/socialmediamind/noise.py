"""Label-noise detection with confident learning (Northcutt, Jiang & Chuang, 2021).

Implemented from the paper's definitions rather than via a library, so every
step is visible:

1. Get *out-of-fold* predicted probabilities for every training post (each post
   is scored by a model that never saw it).
2. Per-class threshold t_j = mean self-confidence of posts labelled j.
3. A post is "confidently" in class j if p_j >= t_j; among such classes pick
   the most probable. Counting (given label, confident label) pairs gives the
   *confident joint*, an estimate of the label-noise matrix.
4. Posts whose confident label differs from the given label are flagged.

Flagged posts are *candidates* for review, not proof of mislabelling: in this
dataset many are genuinely ambiguous between neighbouring communities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_class_thresholds(proba: np.ndarray, y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    t = np.empty(n_classes)
    for j in range(n_classes):
        m = y_idx == j
        t[j] = proba[m, j].mean() if m.any() else 1.0
    return t


def find_label_issues(proba_oof, y, labels) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-sample table, confident joint).

    The per-sample table has: given label, confident label (or NaN if no class
    clears its threshold), self-confidence p_given, and ``is_issue``.
    """
    proba = np.asarray(proba_oof, dtype=float)
    labels = np.asarray(labels)
    idx = {c: i for i, c in enumerate(labels)}
    y_idx = np.array([idx[v] for v in np.asarray(y)])
    k = len(labels)

    t = per_class_thresholds(proba, y_idx, k)
    above = proba >= t[None, :]
    masked = np.where(above, proba, -np.inf)
    has_conf = above.any(axis=1)
    conf_idx = np.where(has_conf, masked.argmax(axis=1), -1)

    joint = np.zeros((k, k), dtype=int)
    for g, c in zip(y_idx[has_conf], conf_idx[has_conf]):
        joint[g, c] += 1

    issue = has_conf & (conf_idx != y_idx)
    table = pd.DataFrame(
        {
            "given": labels[y_idx],
            "confident_label": np.where(has_conf, labels[np.clip(conf_idx, 0, None)], None),
            "self_confidence": proba[np.arange(y_idx.size), y_idx],
            "is_issue": issue,
        }
    )
    joint_df = pd.DataFrame(joint, index=pd.Index(labels, name="given"), columns=pd.Index(labels, name="confident"))
    return table, joint_df


def estimated_noise_rate(joint: pd.DataFrame) -> pd.Series:
    """Per given-label share of confidently-counted posts that look like another class."""
    j = joint.to_numpy()
    total = j.sum(axis=1)
    off = total - np.diag(j)
    return pd.Series(np.divide(off, total, out=np.zeros(len(total)), where=total > 0), index=joint.index)
