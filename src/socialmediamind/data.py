"""Data loading, auditing, de-duplication and leakage-safe splitting."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

KAGGLE_DATASET_ID = "suchintikasarkar/sentiment-analysis-for-mental-health"
TEXT_COL = "text"
LABEL_COL = "label"

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")


LOCAL_CANDIDATES = ("Combined Data.csv", "data/Combined Data.csv", "../data/Combined Data.csv")


def _in_colab() -> bool:
    import sys

    return "google.colab" in sys.modules


def find_data_file(path: str | os.PathLike | None = None) -> Path:
    """Locate the dataset CSV without asking for any account details.

    Resolution order:
      1. explicit ``path`` argument
      2. ``SMM_DATA_PATH`` environment variable
      3. a local ``Combined Data.csv`` (working dir, ``data/`` or ``../data/``)
      4. public Kaggle download via ``kagglehub`` (works anonymously for this
         public dataset; no Kaggle login is required)
      5. in Google Colab only: an upload dialog for the CSV
    """
    explicit = path or os.environ.get("SMM_DATA_PATH")
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        return p

    for cand in LOCAL_CANDIDATES:
        if Path(cand).exists():
            return Path(cand)

    try:
        import kagglehub

        download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
        csvs = sorted(download_dir.rglob("*.csv"))
        if csvs:
            return csvs[0]
        print(f"[data] Kaggle download finished but no CSV was found in {download_dir}.")
    except Exception as e:  # noqa: BLE001 - any failure (no network, rate limit, missing lib) -> fallback
        print(f"[data] Automatic Kaggle download did not work ({type(e).__name__}: {str(e)[:150]}).")

    if _in_colab():
        from google.colab import files  # type: ignore[import-not-found]

        print("[data] Please upload 'Combined Data.csv' (download it from "
              f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_ID}).")
        uploaded = files.upload()
        csvs = [name for name in uploaded if name.lower().endswith(".csv")]
        if csvs:
            return Path(csvs[0])

    raise FileNotFoundError(
        "Could not find the dataset. Download 'Combined Data.csv' from "
        f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_ID} and either place it next to the "
        "notebook / in data/, or set the SMM_DATA_PATH environment variable to its path."
    )


def load_raw(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load the raw CSV (see :func:`find_data_file` for where it is looked up)."""
    csv = find_data_file(path)
    print(f"[data] loading {csv}")
    return pd.read_csv(csv)


def standardise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Kaggle schema (``statement``, ``status``) onto ``text``/``label``.

    The CSV's unnamed index column is dropped explicitly: it is a row number,
    carries no information about the author, and must never be used as a
    feature or target.
    """
    out = df.copy()
    out = out.loc[:, ~out.columns.astype(str).str.startswith("Unnamed")]
    rename = {"statement": TEXT_COL, "status": LABEL_COL}
    out = out.rename(columns=rename)
    missing = {TEXT_COL, LABEL_COL} - set(out.columns)
    if missing:
        raise KeyError(f"Expected columns not found: {sorted(missing)}")
    return out[[TEXT_COL, LABEL_COL]]


def normalise_for_dedup(text: str) -> str:
    """Canonical form used only to detect duplicates (not for modelling)."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _URL.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return _WS.sub(" ", text).strip()


@dataclass
class CleaningReport:
    n_raw: int = 0
    n_missing_text: int = 0
    n_empty_text: int = 0
    n_exact_duplicates_removed: int = 0
    n_conflicting_texts: int = 0
    n_conflicting_rows_removed: int = 0
    n_final: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_and_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove missing/empty posts, collapse duplicates, drop label conflicts.

    Duplicates matter twice in this dataset: identical posts that land in both
    train and test inflate every metric (memorisation looks like skill), and
    identical posts with *different* labels are irreducible label noise.
    """
    rep = CleaningReport(n_raw=len(df))
    out = df.copy()

    missing = out[TEXT_COL].isna()
    rep.n_missing_text = int(missing.sum())
    out = out.loc[~missing]

    out[TEXT_COL] = out[TEXT_COL].astype(str)
    out["_key"] = out[TEXT_COL].map(normalise_for_dedup)
    empty = out["_key"].eq("")
    rep.n_empty_text = int(empty.sum())
    out = out.loc[~empty]

    labels_per_key = out.groupby("_key")[LABEL_COL].nunique()
    conflicting = labels_per_key[labels_per_key > 1].index
    rep.n_conflicting_texts = len(conflicting)
    conflict_mask = out["_key"].isin(conflicting)
    rep.n_conflicting_rows_removed = int(conflict_mask.sum())
    out = out.loc[~conflict_mask]

    before = len(out)
    out = out.drop_duplicates(subset="_key", keep="first")
    rep.n_exact_duplicates_removed = int(before - len(out))

    out = out.drop(columns="_key").reset_index(drop=True)
    rep.n_final = len(out)
    return out, rep


@dataclass
class Splits:
    X_train: pd.Series
    y_train: pd.Series
    X_cal: pd.Series
    y_cal: pd.Series
    X_test: pd.Series
    y_test: pd.Series

    def sizes(self) -> dict:
        return {"train": len(self.X_train), "calibration": len(self.X_cal), "test": len(self.X_test)}


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    cal_size: float = 0.15,
    random_state: int = 42,
) -> Splits:
    """Stratified train / calibration / test split.

    * train        - model fitting and cross-validated model selection
    * calibration  - probability calibration and conformal quantiles only
    * test         - touched once, for the final report
    """
    X, y = df[TEXT_COL], df[LABEL_COL]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_cal = cal_size / (1.0 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=rel_cal, stratify=y_rest, random_state=random_state
    )
    return Splits(X_train, y_train, X_cal, y_cal, X_test, y_test)


def assert_no_overlap(*parts: pd.Series) -> None:
    """Fail loudly if any normalised post appears in more than one split."""
    seen: set[str] = set()
    for part in parts:
        keys = set(part.map(normalise_for_dedup))
        overlap = seen & keys
        if overlap:
            raise AssertionError(f"{len(overlap)} posts leak across splits")
        seen |= keys
