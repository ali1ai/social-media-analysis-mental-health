"""End-to-end smoke test of the model ladder on synthetic data."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_synthetic_fixture import make

from socialmediamind.data import clean_and_deduplicate, make_splits, standardise_schema
from socialmediamind.models import explain_text, model_ladder, top_terms_per_class


def test_ladder_fits_and_text_models_beat_majority():
    df, _ = clean_and_deduplicate(standardise_schema(make(1500, seed=3)))
    s = make_splits(df)
    ladder = model_ladder(include_xgb=True)
    scores = {}
    for name, pipe in ladder.items():
        pipe.fit(s.X_train, s.y_train)
        scores[name] = np.mean(pipe.predict(s.X_test) == s.y_test)
    assert scores["5 Word+Char TF-IDF + LogReg"] > scores["0 Majority class"] + 0.2

    best = ladder["4 Word TF-IDF + LogReg"]
    terms = top_terms_per_class(best, n=5)
    assert terms.shape == (5, s.y_train.nunique())
    expl = explain_text(best, "i feel so hopeless and empty")
    assert len(expl) > 0
