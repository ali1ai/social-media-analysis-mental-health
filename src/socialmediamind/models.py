"""Model ladder and explanation helpers.

The ladder exists to show *incremental* value: every step must beat the one
before it on cross-validated macro-F1, otherwise the extra complexity isn't
justified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import StylometricTransformer, tfidf_union


def logistic(C: float = 4.0, random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=3000,
        class_weight="balanced",
        # lbfgs converges in tens of iterations on sparse TF-IDF here; saga failed
        # to converge after 3000 epochs in benchmarking (100x slower).
        solver="lbfgs",
        random_state=random_state,
    )


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Lets estimators that need integer targets (XGBoost) accept string labels
    and balances classes with inverse-frequency sample weights."""

    def __init__(self, estimator=None, balance: bool = True):
        self.estimator = estimator
        self.balance = balance

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        y_enc = self.encoder_.transform(y)
        weights = None
        if self.balance:
            counts = np.bincount(y_enc)
            weights = (len(y_enc) / (len(counts) * counts))[y_enc]
        self.estimator_ = clone(self.estimator).fit(X, y_enc, sample_weight=weights)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.encoder_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))


def model_ladder(random_state: int = 42, include_xgb: bool = True) -> dict[str, Pipeline]:
    ladder = {
        "0 Majority class": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "1 Stylometric + LogReg": Pipeline(
            [
                ("style", StylometricTransformer()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
    }
    if include_xgb:
        from xgboost import XGBClassifier

        ladder["2 Stylometric + XGBoost"] = Pipeline(
            [
                ("style", StylometricTransformer()),
                (
                    "clf",
                    LabelEncodedClassifier(
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=5,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            tree_method="hist",
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        )
    ladder.update(
        {
            "3 Word TF-IDF + ComplementNB": Pipeline(
                [("tfidf", tfidf_union(use_char=False)), ("clf", ComplementNB(alpha=0.3))]
            ),
            "4 Word TF-IDF + LogReg": Pipeline(
                [("tfidf", tfidf_union(use_char=False)), ("clf", logistic(random_state=random_state))]
            ),
            "5 Word+Char TF-IDF + LogReg": Pipeline(
                [("tfidf", tfidf_union(use_char=True)), ("clf", logistic(random_state=random_state))]
            ),
        }
    )
    return ladder


def top_terms_per_class(pipe: Pipeline, n: int = 15, word_only: bool = True) -> pd.DataFrame:
    """Highest positive coefficients per class for a TF-IDF + linear pipeline.

    Coefficients are on sublinear-TF-IDF features, so magnitudes are
    comparable within a model. They describe what the model relies on, not
    why people write what they write.
    """
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    coefs = clf.coef_
    mask = np.array([nm.startswith("word__") or "__" not in nm for nm in names]) if word_only else None
    out = {}
    for k, cls in enumerate(clf.classes_):
        c = coefs[k].copy()
        if mask is not None:
            c = np.where(mask, c, -np.inf)
        top = np.argsort(c)[::-1][:n]
        out[cls] = [names[i].split("__", 1)[-1] for i in top]
    return pd.DataFrame(out)


def explain_text(pipe: Pipeline, text: str, n: int = 10) -> pd.DataFrame:
    """Exact additive contributions (coef x feature value) for the predicted
    class of a linear model. For a linear model this coincides with SHAP
    under a zero baseline, but is cheaper and exact."""
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    x = vec.transform([text])
    proba = clf.predict_proba(x)[0]
    k = int(np.argmax(proba))
    contrib = x.multiply(clf.coef_[k]).tocsr()
    names = vec.get_feature_names_out()
    idx = contrib.indices
    vals = contrib.data
    order = np.argsort(np.abs(vals))[::-1][:n]
    return pd.DataFrame(
        {
            "predicted_class": clf.classes_[k],
            "feature": [names[idx[i]] for i in order],
            "contribution": vals[order],
        }
    )
