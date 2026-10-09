"""Interpretable stylometric features and the TF-IDF text pipeline."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

_WORD = re.compile(r"\b\w+\b")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "im", "i'm", "ive", "i've"}
ABSOLUTIST = {
    # Absolutist language has been linked to depression/suicidal ideation in
    # forum text (Al-Mosaiwi & Johnstone, 2018). Used here descriptively only.
    "always", "never", "nothing", "everything", "completely", "totally",
    "entire", "constantly", "definitely", "forever", "whole", "every",
}

STYLOMETRIC_FEATURES = [
    "n_chars",
    "n_words",
    "n_sentences",
    "avg_word_len",
    "type_token_ratio",
    "first_person_rate",
    "absolutist_rate",
    "question_rate",
    "exclamation_rate",
    "uppercase_ratio",
    "punctuation_ratio",
    "digit_ratio",
]


def stylometric_frame(texts: pd.Series | list[str]) -> pd.DataFrame:
    """Length, lexical-diversity and pronoun/absolutist rates per post.

    Rates are normalised by word count so they are not just proxies for
    post length.
    """
    rows = []
    for t in texts:
        t = str(t)
        words = [w.lower() for w in _WORD.findall(t)]
        n_words = len(words)
        n_chars = len(t)
        denom_w = max(n_words, 1)
        denom_c = max(n_chars, 1)
        rows.append(
            {
                "n_chars": n_chars,
                "n_words": n_words,
                "n_sentences": max(len(re.findall(r"[.!?]+", t)), 1),
                "avg_word_len": (sum(map(len, words)) / denom_w) if words else 0.0,
                "type_token_ratio": (len(set(words)) / denom_w) if words else 0.0,
                "first_person_rate": sum(w in FIRST_PERSON for w in words) / denom_w,
                "absolutist_rate": sum(w in ABSOLUTIST for w in words) / denom_w,
                "question_rate": t.count("?") / denom_w,
                "exclamation_rate": t.count("!") / denom_w,
                "uppercase_ratio": sum(c.isupper() for c in t) / denom_c,
                "punctuation_ratio": len(re.findall(r"[^\w\s]", t)) / denom_c,
                "digit_ratio": sum(c.isdigit() for c in t) / denom_c,
            }
        )
    return pd.DataFrame(rows, columns=STYLOMETRIC_FEATURES)


class StylometricTransformer(BaseEstimator, TransformerMixin):
    """sklearn wrapper so stylometric features can live inside a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return stylometric_frame(X).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(STYLOMETRIC_FEATURES, dtype=object)


def tfidf_union(
    word_ngrams: tuple[int, int] = (1, 2),
    char_ngrams: tuple[int, int] = (3, 5),
    max_word_features: int | None = 100_000,
    max_char_features: int | None = 150_000,
    min_df: int = 2,
    use_char: bool = True,
) -> FeatureUnion | TfidfVectorizer:
    """Word (+ optional char_wb) TF-IDF. Char n-grams are robust to the
    misspellings and creative spelling common in social-media text."""
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=word_ngrams,
        min_df=min_df,
        max_df=0.95,
        max_features=max_word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w']*\b",
    )
    if not use_char:
        return word
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=char_ngrams,
        min_df=min_df,
        max_features=max_char_features,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
