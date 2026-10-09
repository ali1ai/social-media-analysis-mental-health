"""Reproducible, notebook-free training run.

    python -m socialmediamind.train                       # Kaggle download
    python -m socialmediamind.train --data path/to.csv    # local CSV
    smm-train --data data/synthetic_fixture.csv --fast    # CI smoke run

Mirrors the notebook's protocol: clean and de-duplicate -> stratified
train/calibration/test split -> cross-validated choice between word and
word+char TF-IDF logistic regression -> randomised C search -> temperature
scaling (calibration half A) -> class-conditional conformal (half B) ->
single test evaluation. Writes the model bundle and a metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split

from .calibration import TemperatureScaler
from .conformal import SplitConformalClassifier, coverage_report
from .data import assert_no_overlap, clean_and_deduplicate, load_raw, make_splits, standardise_schema
from .evaluation import bootstrap_ci, classification_metrics
from .inference import SocialMediaMindModel
from .models import model_ladder


def run(
    data: str | None = None,
    out_dir: str = "models",
    report_dir: str = "reports",
    alpha: float = 0.10,
    score: str = "lac",
    fast: bool = False,
    random_state: int = 42,
) -> dict:
    t0 = time.time()
    df, cleaning = clean_and_deduplicate(standardise_schema(load_raw(data)))
    s = make_splits(df, random_state=random_state)
    assert_no_overlap(s.X_train, s.X_cal, s.X_test)
    print(f"[data] {cleaning.n_final:,} unique posts | splits {s.sizes()}")

    folds = 3 if fast else 5
    cv = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    ladder = model_ladder(random_state, include_xgb=False)
    families = ["4 Word TF-IDF + LogReg", "5 Word+Char TF-IDF + LogReg"]
    cv_scores = {
        f: float(cross_val_score(ladder[f], s.X_train, s.y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean())
        for f in families
    }
    family = max(cv_scores, key=cv_scores.get)
    print(f"[cv] {cv_scores} -> {family}")

    search = RandomizedSearchCV(
        ladder[family],
        {"clf__C": loguniform(0.5, 30)},
        n_iter=3 if fast else 10,
        scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
    ).fit(s.X_train, s.y_train)
    pipe = search.best_estimator_
    labels = [str(c) for c in pipe.classes_]
    print(f"[search] best C = {search.best_params_['clf__C']:.3f} (cv macro-F1 {search.best_score_:.4f})")

    X_a, X_b, y_a, y_b = train_test_split(
        s.X_cal, s.y_cal, test_size=0.5, stratify=s.y_cal, random_state=random_state
    )
    ts = TemperatureScaler(labels).fit(pipe.decision_function(X_a), y_a)
    cp = SplitConformalClassifier(labels, alpha=alpha, class_conditional=True, score=score, random_state=random_state).fit(
        ts.predict_proba(pipe.decision_function(X_b)), y_b
    )

    model = SocialMediaMindModel(
        pipeline=pipe, temperature=ts.temperature_, conformal=cp, labels=labels,
        metadata={"family": family, "best_params": {k: float(v) for k, v in search.best_params_.items()}},
    )
    proba = model.predict_proba(s.X_test)
    y_pred = np.asarray(labels)[proba.argmax(1)]
    point, lo, hi = bootstrap_ci(s.y_test, y_pred, n_boot=200 if fast else 1000, random_state=random_state)
    cov = coverage_report(model.prediction_sets(proba), s.y_test, labels).set_index("class")

    metrics = {
        "family": family,
        "cv_macro_f1": cv_scores,
        "temperature": ts.temperature_,
        "test": {**classification_metrics(s.y_test, y_pred, proba, labels), "macro_f1_ci95": [lo, hi]},
        "conformal": {
            "alpha": alpha, "score": score,
            "coverage": float(cov.loc["ALL", "coverage"]),
            "mean_set_size": float(cov.loc["ALL", "mean_set_size"]),
            "min_class_coverage": float(cov.drop("ALL")["coverage"].min()),
        },
        "cleaning": cleaning.to_dict(),
        "seconds": round(time.time() - t0, 1),
    }
    model.metadata["test_macro_f1"] = point
    path = model.save(Path(out_dir) / "socialmediamind_bundle.joblib")
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    (Path(report_dir) / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[test] macro-F1 {point:.4f} (95% CI {lo:.4f}-{hi:.4f}) | conformal coverage "
          f"{metrics['conformal']['coverage']:.3f}, mean set {metrics['conformal']['mean_set_size']:.2f}")
    print(f"[saved] {path}")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smm-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV path (default: SMM_DATA_PATH or Kaggle download)")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--score", choices=["lac", "aps"], default="lac")
    ap.add_argument("--fast", action="store_true", help="fewer folds/iterations (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    run(a.data, a.out_dir, a.report_dir, a.alpha, a.score, a.fast, a.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
