import numpy as np
import pandas as pd
import pytest

from socialmediamind.calibration import TemperatureScaler
from socialmediamind.conformal import SplitConformalClassifier, coverage_report, score_matrix
from socialmediamind.data import (
    assert_no_overlap,
    clean_and_deduplicate,
    make_splits,
    normalise_for_dedup,
    standardise_schema,
)
from socialmediamind.evaluation import bootstrap_ci, expected_calibration_error
from socialmediamind.features import STYLOMETRIC_FEATURES, stylometric_frame
from socialmediamind.noise import estimated_noise_rate, find_label_issues
from socialmediamind.stats import benjamini_hochberg, cliffs_delta


# ---------------------------------------------------------------- data
def test_schema_drops_index_column():
    raw = pd.DataFrame({"Unnamed: 0": [0, 1], "statement": ["a", "b"], "status": ["Normal", "Stress"]})
    out = standardise_schema(raw)
    assert list(out.columns) == ["text", "label"]  # row index can never become a feature/target


def test_normalise_for_dedup_collapses_case_punct_urls():
    assert normalise_for_dedup("I'm  SAD!! http://x.co") == normalise_for_dedup("i m sad")


def test_clean_removes_dupes_conflicts_and_empty():
    df = pd.DataFrame(
        {
            "text": ["hello world", "Hello, world!", "conflict", "conflict", None, "   ", "unique"],
            "label": ["Normal", "Normal", "Stress", "Anxiety", "Normal", "Normal", "Bipolar"],
        }
    )
    out, rep = clean_and_deduplicate(df)
    assert sorted(out["text"]) == ["hello world", "unique"]
    assert rep.n_missing_text == 1
    assert rep.n_empty_text == 1
    assert rep.n_conflicting_texts == 1
    assert rep.n_exact_duplicates_removed == 1


def test_splits_are_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"text": [f"post number {i}" for i in range(700)], "label": rng.choice(["A", "B", "C"], 700, p=[.6, .3, .1])}
    )
    s = make_splits(df)
    assert_no_overlap(s.X_train, s.X_cal, s.X_test)
    share = lambda y: (y == "C").mean()
    assert abs(share(s.y_test) - share(df["label"])) < 0.03


def test_overlap_detected():
    with pytest.raises(AssertionError):
        assert_no_overlap(pd.Series(["Same post"]), pd.Series(["same POST!"]))


# ---------------------------------------------------------------- features
def test_stylometric_rates_are_length_normalised():
    f = stylometric_frame(["I always fail. I never win!", ""])
    assert list(f.columns) == STYLOMETRIC_FEATURES
    assert f.loc[0, "n_words"] == 6
    assert f.loc[0, "first_person_rate"] == pytest.approx(2 / 6)
    assert f.loc[0, "absolutist_rate"] == pytest.approx(2 / 6)
    assert f.loc[1].isna().sum() == 0  # empty text must not produce NaN/inf


# ---------------------------------------------------------------- stats
def test_bh_matches_known_values():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(q, [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)


def test_cliffs_delta_extremes_and_ties():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert cliffs_delta([1, 1], [1, 1]) == 0.0


# ---------------------------------------------------------------- evaluation
def test_bootstrap_ci_brackets_point_estimate():
    y = np.array(["a", "b"] * 100)
    p = y.copy()
    p[:20] = "b"
    point, lo, hi = bootstrap_ci(y, p, n_boot=200)
    assert lo <= point <= hi


def test_ece_zero_for_perfectly_calibrated_onehot():
    labels = ["a", "b"]
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(["a", "b"], proba, labels) == pytest.approx(0.0)


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b", "c"])
    y_idx = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y_idx] += 1.0
    logits *= 6  # make the model wildly overconfident
    ts = TemperatureScaler(labels).fit(logits, labels[y_idx])
    assert ts.temperature_ > 2.0


# ---------------------------------------------------------------- conformal
@pytest.mark.parametrize("score", ["lac", "aps"])
@pytest.mark.parametrize("class_conditional", [False, True])
def test_conformal_coverage_guarantee(class_conditional, score):
    rng = np.random.default_rng(1)
    labels = np.array(["a", "b", "c", "d"])
    n = 30000
    y = rng.choice(4, n, p=[.5, .3, .15, .05])
    logits = rng.normal(size=(n, 4))
    logits[np.arange(n), y] += 1.2
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    half = n // 2
    cp = SplitConformalClassifier(labels, alpha=0.1, class_conditional=class_conditional, score=score)
    cp.fit(proba[:half], labels[y[:half]])
    rep = coverage_report(cp.predict_sets(proba[half:]), labels[y[half:]], labels)
    overall = rep.loc[rep["class"] == "ALL", "coverage"].item()
    assert overall >= 0.88
    if class_conditional:
        assert (rep["coverage"] >= 0.85).all()


def test_aps_scores_are_cumulative_mass():
    proba = np.array([[0.6, 0.3, 0.1]])
    np.testing.assert_allclose(score_matrix(proba, "aps"), [[0.6, 0.9, 1.0]])            # deterministic
    np.testing.assert_allclose(score_matrix(proba, "aps", u=[0.5]), [[0.3, 0.75, 0.95]])  # randomised
    np.testing.assert_allclose(score_matrix(proba, "lac"), [[0.4, 0.7, 0.9]])


def test_too_few_calibration_points_gives_full_sets():
    cp = SplitConformalClassifier(["a", "b"], alpha=0.1, class_conditional=True)
    cp.fit(np.array([[0.9, 0.1], [0.8, 0.2]]), ["a", "a"])  # class "b" unseen, "a" has n=2
    assert cp.predict_sets(np.array([[0.5, 0.5]])).all()


# ---------------------------------------------------------------- label noise
def test_confident_learning_recovers_flipped_labels():
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b", "c"])
    n = 3000
    true = rng.integers(0, 3, n)
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), true] += 3.0
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    given = true.copy()
    flip = rng.random(n) < 0.10
    given[flip] = (true[flip] + 1) % 3
    table, joint = find_label_issues(proba, labels[given], labels)
    flagged = table["is_issue"].to_numpy()
    precision = (flagged & flip).sum() / flagged.sum()
    recall = (flagged & flip).sum() / flip.sum()
    # Confident learning is precision-oriented: it flags only when another class clears its own
    # per-class threshold, so flips the model is itself unsure about are (correctly) left alone.
    assert precision > 0.9 and recall > 0.7
    assert joint.to_numpy().sum() <= n
    rates = estimated_noise_rate(joint)
    assert ((rates > 0.05) & (rates < 0.2)).all()


def test_randomized_aps_keeps_sets_small_for_confident_model():
    rng = np.random.default_rng(2)
    labels = np.array(list("abcdefg"))
    n = 6000
    y = rng.integers(0, 7, n)
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += 8.0  # very confident, mostly right
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    half = n // 2
    rand = SplitConformalClassifier(labels, 0.1, True, "aps", randomized=True).fit(proba[:half], labels[y[:half]])
    det = SplitConformalClassifier(labels, 0.1, True, "aps", randomized=False).fit(proba[:half], labels[y[:half]])
    size_rand = rand.predict_sets(proba[half:]).sum(1).mean()
    size_det = det.predict_sets(proba[half:]).sum(1).mean()
    assert size_rand < 1.5 < size_det
