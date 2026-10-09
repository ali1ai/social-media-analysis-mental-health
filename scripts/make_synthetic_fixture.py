"""Generate a small synthetic CSV with the same schema as the Kaggle dataset.

Used for CI and smoke-testing the notebook end to end without Kaggle access.
The text is template-generated and NOT real user data; results on it mean
nothing about mental health.

    python scripts/make_synthetic_fixture.py data/synthetic_fixture.csv --n 4000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

CLASS_SHARES = {
    "Normal": 0.31, "Depression": 0.29, "Suicidal": 0.20, "Anxiety": 0.07,
    "Bipolar": 0.05, "Stress": 0.05, "Personality disorder": 0.03,
}
VOCAB = {
    "Normal": "weekend game coffee friends movie lunch weather music trip work fun dinner",
    "Depression": "empty sad hopeless tired numb alone worthless crying bed nothing heavy dark",
    "Suicidal": "end die goodbye pain cant anymore escape tired over burden nothing gone",
    "Anxiety": "worried panic heart racing nervous fear restless overthinking breath shaking",
    "Stress": "deadline pressure exams workload overwhelmed boss bills busy tense pressure",
    "Bipolar": "manic energy mood swings crash sleepless impulsive high low episode meds",
    "Personality disorder": "abandonment identity unstable relationships split anger mask fear empty",
}
SHARED = ["i", "me", "my", "the", "and", "to", "a", "feel", "today", "really", "just", "like", "know", "think", "time", "people", "life"]


def make(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(CLASS_SHARES), size=n, p=list(CLASS_SHARES.values()))
    texts = []
    for lab in labels:
        own = VOCAB[lab].split()
        other = VOCAB[rng.choice(list(VOCAB))].split()
        length = int(rng.integers(5, 60))
        words = []
        for _ in range(length):
            r = rng.random()
            pool = own if r < 0.25 else other if r < 0.35 else SHARED
            words.append(str(rng.choice(pool)))
        t = " ".join(words).capitalize() + rng.choice([".", "!", "?", "..."])
        texts.append(t)
    df = pd.DataFrame({"statement": texts, "status": labels})
    # Inject the data-quality problems seen in the real dataset.
    dup = df.sample(frac=0.04, random_state=seed)
    conflict = df.sample(n=max(n // 200, 2), random_state=seed + 1).copy()
    conflict["status"] = "Normal"
    df = pd.concat([df, dup, conflict], ignore_index=True)
    df.loc[df.sample(n=max(n // 300, 1), random_state=seed + 2).index, "statement"] = np.nan
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "Unnamed: 0", range(len(df)))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.n, a.seed).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
