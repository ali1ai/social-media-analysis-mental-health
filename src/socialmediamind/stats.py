"""Non-parametric group comparisons with effect sizes and FDR control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), order preserved."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def cliffs_delta(x, y) -> float:
    """Cliff's delta: P(X>Y) - P(X<Y). Robust, distribution-free effect size.

    Computed via ranks in O(n log n), so it scales to tens of thousands of posts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return np.nan
    ranks = stats.rankdata(np.concatenate([x, y]))
    rx = ranks[:nx].sum()
    u = rx - nx * (nx + 1) / 2  # Mann-Whitney U for x (ties count 0.5)
    return float(2 * u / (nx * ny) - 1)


def magnitude_cliffs(d: float) -> str:
    """Romano et al. (2006) thresholds."""
    a = abs(d)
    if np.isnan(a):
        return "n/a"
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def kruskal_by_group(df: pd.DataFrame, features: list[str], group_col: str) -> pd.DataFrame:
    """Kruskal-Wallis H per feature, epsilon-squared effect size, BH q-values.

    epsilon^2 = H / (n - 1); ~0.01 small, ~0.08 medium, ~0.26 large (Tomczak & Tomczak, 2014).
    """
    rows = []
    groups = [g for _, g in df.groupby(group_col)]
    n = len(df)
    for f in features:
        samples = [g[f].dropna().to_numpy() for g in groups]
        samples = [s for s in samples if s.size > 0]
        if len(samples) < 2:
            continue
        try:
            h, p = stats.kruskal(*samples)
        except ValueError:  # all values identical
            h, p = 0.0, 1.0
        rows.append({"feature": f, "H": h, "p_value": p, "epsilon_sq": h / (n - 1)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["q_value_bh"] = benjamini_hochberg(res["p_value"])
        res = res.sort_values("epsilon_sq", ascending=False).reset_index(drop=True)
    return res


def cliffs_vs_reference(
    df: pd.DataFrame, features: list[str], group_col: str, reference: str
) -> pd.DataFrame:
    """Cliff's delta of every group against a reference group, per feature."""
    ref = df[df[group_col] == reference]
    out = {}
    for g, sub in df.groupby(group_col):
        if g == reference:
            continue
        out[g] = {f: cliffs_delta(sub[f], ref[f]) for f in features}
    return pd.DataFrame(out)
