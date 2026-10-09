"""Deployable bundle: text pipeline + temperature + conformal quantiles.

    from socialmediamind.inference import SocialMediaMindModel
    model = SocialMediaMindModel.load("models/socialmediamind_bundle.joblib")
    model.predict(["I can't stop worrying about everything"])

Research use only - see the model card. ``defer_to_human`` is True whenever
the conformal set is not a single label, a principled abstention signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .calibration import TemperatureScaler
from .conformal import SplitConformalClassifier

DISCLAIMER = (
    "Research/portfolio model. Not a diagnostic or screening tool; do not use it to make decisions about people."
)


@dataclass
class SocialMediaMindModel:
    pipeline: object
    temperature: float
    conformal: SplitConformalClassifier
    labels: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return self.conformal.alpha

    # ------------------------------------------------------------- inference
    def predict_proba(self, texts) -> np.ndarray:
        scaler = TemperatureScaler(self.labels)
        scaler.temperature_ = self.temperature
        return scaler.predict_proba(self.pipeline.decision_function(pd.Series(list(texts), dtype=str)))

    def prediction_sets(self, proba: np.ndarray) -> np.ndarray:
        return self.conformal.predict_sets(proba)

    def predict(self, texts) -> list[dict]:
        p = self.predict_proba(texts)
        sets = self.prediction_sets(p)
        labels = np.asarray(self.labels)
        return [
            {
                "top_label": str(labels[row.argmax()]),
                "confidence": round(float(row.max()), 4),
                "prediction_set": [str(c) for c in labels[s]],
                "defer_to_human": bool(s.sum() != 1),
            }
            for row, s in zip(p, sets)
        ]

    # ------------------------------------------------------------- persistence
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("package_version", __version__)
        self.metadata.setdefault("saved_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path) -> SocialMediaMindModel:
        obj = joblib.load(path)
        if not isinstance(obj, SocialMediaMindModel):
            raise TypeError(f"{path} does not contain a SocialMediaMindModel")
        return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smm-predict", description="Score texts with a trained SocialMediaMind bundle.")
    ap.add_argument("texts", nargs="*", help="texts to score (or use --file / stdin)")
    ap.add_argument("--model", default="models/socialmediamind_bundle.joblib")
    ap.add_argument("--file", help="text file, one post per line")
    args = ap.parse_args(argv)

    texts = list(args.texts)
    if args.file:
        texts += [ln.rstrip("\n") for ln in Path(args.file).read_text().splitlines() if ln.strip()]
    if not texts and not sys.stdin.isatty():
        texts += [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    if not texts:
        ap.error("no input texts")

    print(f"# {DISCLAIMER}", file=sys.stderr)
    model = SocialMediaMindModel.load(args.model)
    for text, pred in zip(texts, model.predict(texts)):
        print(json.dumps({"text": text[:80], **pred}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
