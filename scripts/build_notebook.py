"""Build notebooks/SocialMediaMind.ipynb from source.

The notebook is generated (not hand-edited) so that it always embeds an exact,
checksum-verified copy of src/socialmediamind/. That lets it run standalone in
Google Colab or Kaggle with no git clone and no GitHub/Kaggle login.

    python scripts/build_notebook.py            # regenerate after changing src/ or this file

tests/test_notebook_sync.py fails if the embedded copy drifts from src/.
"""

import base64
import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "socialmediamind"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "notebooks" / "SocialMediaMind.ipynb"


def embedded_package() -> tuple[str, str]:
    """Deterministic (byte-identical across rebuilds) gzipped tar of the package."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for f in sorted(PKG.glob("*.py")):
            data = f.read_bytes()
            info = tarfile.TarInfo(f"socialmediamind/{f.name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0) as g:
        g.write(raw.getvalue())
    blob = gz.getvalue()
    return base64.b64encode(blob).decode(), hashlib.sha256(blob).hexdigest()


cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================
md(r"""
# SocialMediaMind

## Interpretable, calibrated and uncertainty-aware detection of mental-health signals in social-media text

**How to run:** *Runtime → Run all*. The notebook works on its own in Google Colab, Kaggle or locally. It does **not** need a GitHub account, a password, or a Kaggle login. A GPU runtime is optional; it only enables the transformer baseline in §10.2.

This notebook asks how much a classical, fully inspectable NLP model can learn about **self-labelled mental-health categories** from the language of social-media posts. It then asks a second question: how far should anyone trust that model's individual predictions?

The emphasis is on the parts of applied ML that usually get skipped:

| Concern | What this notebook does |
|---|---|
| **Data quality** | Audits missing, empty, duplicated and *contradictorily-labelled* posts; de-duplicates **before** splitting |
| **Leakage** | Disjoint train / calibration / test splits with an automated overlap check, plus a demonstration of how much duplicate leakage inflates scores |
| **Honest baselines** | A model ladder from a majority-class baseline to word + character TF-IDF, compared by cross-validated **macro-F1** |
| **Statistics** | Non-parametric tests with effect sizes (ε², Cliff's δ) and Benjamini–Hochberg FDR control |
| **Reliability** | Temperature scaling, reliability diagrams, ECE and Brier score |
| **Uncertainty** | Split-conformal **prediction sets** (LAC and adaptive APS scores) with marginal and class-conditional (Mondrian) coverage guarantees |
| **Label quality** | Confident learning on out-of-fold probabilities to estimate the label-noise matrix and flag likely mislabelled posts |
| **Deep-learning baseline** | Optional fine-tuned transformer (DistilRoBERTa, runs automatically on a GPU runtime) passed through the *same* calibration and conformal layer, with a paired-bootstrap comparison |
| **Explainability** | Per-class term weights, exact local attributions and SHAP on an interpretable stylometric model |
| **Robustness** | Error analysis, confusion structure and performance slices by post length |
| **Unsupervised structure** | LSA + k-means topic structure with internal validation and stability analysis |
| **Reproducibility** | Seeds, versioned artefacts, a tested `src/` package and an auto-generated model card |

> ### ⚠️ Responsible use
> This is an exploratory research and portfolio project. **It is not a diagnostic or screening tool**, and it must not be used to make decisions about individuals.
> * Labels come from the communities the posts were collected from (e.g. the subreddit they were posted in), so they are **not clinical diagnoses**. The model learns *how people write in those communities*, not who has a condition.
> * Associations reported here are **not causal**.
> * The *Suicidal* class means real crisis content exists in this data. Raw post text is **hidden by default** (`SHOW_TEXT_EXAMPLES = False`).
>
> If you or someone you know is struggling, please contact local emergency services or a crisis line (e.g. **988** in the US, **116 123** Samaritans in the UK/IE, or [findahelpline.com](https://findahelpline.com)).
""")

md(r"""
## 0. Research questions

* **RQ1 (association).** Do simple, interpretable stylometric properties of a post differ across categories? Examples include length, lexical diversity, first-person pronouns and absolutist words. How large are the differences, rather than just whether they are "significant"?
* **RQ2 (prediction).** How much does modelling the actual *words* add over a majority baseline and over stylometry alone, measured by macro-F1 on unseen, de-duplicated posts?
* **RQ3 (reliability).** Are the model's probabilities calibrated? Can we produce prediction **sets** with a guaranteed error rate, including for the rare, high-stakes classes?
* **RQ4 (explanation and failure modes).** Which features drive predictions, which categories are confused with each other, and where does the model fail?
* **RQ5 (structure).** Without labels, does the language organise into stable topical clusters, and how do those relate to the labels?
""")

# =====================================================================
md("## 1. Environment and configuration")

SETUP_CELL = r"""
#@title Setup: run this first (no GitHub or Kaggle login needed)
# Makes the `socialmediamind` helper package importable, in this order:
#   1. the repository's src/ folder, if this notebook is inside a clone of the repo
#   2. otherwise the copy embedded at the bottom of this cell (gzipped + base64,
#      SHA-256 verified). It is preferred over any pip-installed version because it
#      is guaranteed to match this exact notebook. It is generated from src/ by scripts/build_notebook.py,
#      and tests/test_notebook_sync.py checks it is identical to src/.
import base64, hashlib, importlib.util, io, os, subprocess, sys, tarfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

# Third-party libraries: install only what is missing (Colab already has all of them).
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
            "xgboost": "xgboost", "shap": "shap", "matplotlib": "matplotlib", "joblib": "joblib",
            "kagglehub": "kagglehub", "jinja2": "jinja2"}
missing = [pip_name for mod, pip_name in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing:", ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _setup_package():
    for root in (Path.cwd(), Path.cwd().parent):
        if (root / "src" / "socialmediamind" / "__init__.py").exists():
            sys.path.insert(0, str(root / "src"))
            return root, "repository src/"
    blob = base64.b64decode(EMBEDDED_PACKAGE)
    if hashlib.sha256(blob).hexdigest() != EMBEDDED_SHA256:
        raise RuntimeError("Embedded package is corrupted; re-download the notebook.")
    target = Path.cwd() / "_smm_embedded"
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        try:
            tar.extractall(target, filter="data")
        except TypeError:  # Python < 3.10.12 has no extraction filters
            tar.extractall(target)
    sys.path.insert(0, str(target))
    for name in [m for m in sys.modules if m == "socialmediamind" or m.startswith("socialmediamind.")]:
        del sys.modules[name]  # never mix with a previously imported/installed version
    return Path.cwd(), "embedded copy"


EMBEDDED_SHA256 = "__SHA__"
EMBEDDED_PACKAGE = "__BLOB__"

ROOT, PKG_SOURCE = _setup_package()
import socialmediamind  # noqa: E402

print(f"socialmediamind {socialmediamind.__version__} loaded from {PKG_SOURCE} | outputs -> {ROOT}")
"""
_blob, _sha = embedded_package()
code(SETUP_CELL.replace("__SHA__", _sha).replace("__BLOB__", _blob))
cells[-1]["metadata"]["cellView"] = "form"  # Colab: show the title, hide the long code

code(r"""
import json, math, platform, random, time, warnings
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy, sklearn, xgboost, shap

from scipy.stats import loguniform
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score, classification_report, confusion_matrix, f1_score,
    normalized_mutual_info_score, silhouette_score,
)
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.preprocessing import Normalizer

from socialmediamind import LABEL_ORDER
from socialmediamind.calibration import TemperatureScaler
from socialmediamind.conformal import SplitConformalClassifier, coverage_report
from socialmediamind.data import (
    LABEL_COL, TEXT_COL, assert_no_overlap, clean_and_deduplicate, load_raw,
    make_splits, normalise_for_dedup, standardise_schema,
)
from socialmediamind.evaluation import bootstrap_ci, classification_metrics, reliability_table
from socialmediamind.features import STYLOMETRIC_FEATURES, stylometric_frame
from socialmediamind.inference import SocialMediaMindModel
from socialmediamind.models import explain_text, model_ladder, top_terms_per_class
from socialmediamind.noise import estimated_noise_rate, find_label_issues
from socialmediamind.stats import cliffs_vs_reference, kruskal_by_group

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- configuration ----------------
RANDOM_STATE = 42
FAST_MODE = os.environ.get("SMM_FAST", "0") == "1"   # quick smoke run (CI / synthetic data)
SHOW_TEXT_EXAMPLES = False                           # keep sensitive post text out of saved outputs
CV_FOLDS = 3 if FAST_MODE else 5
SEARCH_ITER = 3 if FAST_MODE else 12
TUNE_SAMPLE = 4_000 if FAST_MODE else 20_000         # subsample for hyper-parameter search speed
N_BOOT = 200 if FAST_MODE else 1_000
ALPHA = 0.10                                         # conformal miscoverage -> 90% sets
CLUSTER_SAMPLE = 3_000 if FAST_MODE else 15_000
# Transformer baseline: "auto" = run only when a CUDA GPU is available (Colab: Runtime → Change runtime type → GPU)
SMM_TRANSFORMER = os.environ.get("SMM_TRANSFORMER", "auto")
TRANSFORMER_NAME = "distilroberta-base"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

REPORT_DIR, MODEL_DIR, FIG_DIR = ROOT / "reports", ROOT / "models", ROOT / "reports" / "figures"
for p in (REPORT_DIR, MODEL_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (fixed order -> colour follows the class, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})

def savefig(name):
    plt.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)

ENV = {
    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
    "xgboost": xgboost.__version__, "shap": shap.__version__,
    "fast_mode": FAST_MODE, "random_state": RANDOM_STATE,
}
pd.Series(ENV, name="value").to_frame()
""")

# =====================================================================
md(r"""
## 2. Data acquisition and provenance

**Source:** Kaggle, [`suchintikasarkar/sentiment-analysis-for-mental-health`](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health). This is a compilation of several public datasets (largely Reddit and Twitter posts), each post tagged with one of seven statuses.

What the labels are, and are not:
* They reflect the **source community or collection context**, not a clinician's assessment.
* A post in a depression community can be supportive advice rather than a first-person disclosure, so **label noise is expected**.
* The compilation draws on several sources, and *Normal* posts may come from different platforms than the clinical categories. Some of what the model learns may be **platform or style differences** rather than mental-health language. This confound is discussed again in §13.

**Getting the data needs no login.** The loader looks, in order, for:
1. a path in the `SMM_DATA_PATH` environment variable
2. a local `Combined Data.csv` (next to the notebook or in `data/`)
3. an anonymous public download through `kagglehub`
4. in Colab only, if the download is blocked, an **upload button** for the CSV, which you can get from the Kaggle page above
""")

code(r"""
df_raw = load_raw()
print("Raw shape:", df_raw.shape)
print("Raw columns:", list(df_raw.columns))

df = standardise_schema(df_raw)   # -> columns: text, label ; the CSV row-index column is dropped on purpose
df.head(3).assign(text=lambda d: d[TEXT_COL].str.slice(0, 60) + "…" if SHOW_TEXT_EXAMPLES else "[hidden]")
""")

md(r"""
> **Why the index column is dropped explicitly:** `Unnamed: 0` is just the CSV row number. The original data is ordered by label, so the row number is strongly correlated with the label. Any model that sees it, as a feature *or* as a target, will look impressive while learning nothing about language.
""")

# =====================================================================
md("## 3. Data audit")

code(r"""
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_n": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(3),
    "unique_n": df.nunique(),
})
display(audit)

keys = df[TEXT_COL].dropna().map(normalise_for_dedup)
dup_share = keys.duplicated(keep=False).mean()
conflicts = (
    df.dropna(subset=[TEXT_COL]).assign(_k=keys)
      .groupby("_k")[LABEL_COL].nunique().gt(1).sum()
)
print(f"Posts that share their (normalised) text with another post: {dup_share:.2%}")
print(f"Distinct texts carrying more than one label (label conflicts): {conflicts:,}")

labels_present = [c for c in LABEL_ORDER if c in set(df[LABEL_COL])] + sorted(set(df[LABEL_COL]) - set(LABEL_ORDER))
COLOR = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(labels_present)}

counts = df[LABEL_COL].value_counts().reindex(labels_present)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(counts.index[::-1], counts.values[::-1], color=[COLOR[c] for c in counts.index[::-1]], height=0.6)
for y, v in enumerate(counts.values[::-1]):
    ax.text(v, y, f"  {v:,} ({v / counts.sum():.1%})", va="center", fontsize=9)
ax.set_title("Class distribution (raw)"); ax.set_xlabel("posts"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, counts.max() * 1.25)
savefig("class_distribution"); plt.show()
""")

md(r"""
The classes are **imbalanced**: the smallest is roughly 15 times rarer than the largest. Accuracy would therefore reward a model that ignores the rare classes, so **macro-F1** is the primary metric throughout, with balanced accuracy alongside it.
""")

# =====================================================================
md(r"""
## 4. Cleaning, de-duplication and leakage-safe splits

Order matters here, so the steps run in this sequence:
1. Drop missing posts, and posts that are empty after normalisation.
2. Drop every copy of a text that appears with **conflicting labels**. These are irreducible noise: no model can be right on both.
3. Collapse remaining duplicates (case, punctuation and URL-insensitive) to one copy.
4. **Only then** split into stratified **train (70%) / calibration (15%) / test (15%)**.

| Split | Used for |
|---|---|
| train | fitting and *all* model selection (cross-validation, hyper-parameter search) |
| calibration | temperature scaling (half A) and conformal quantiles (half B), never for training |
| test | touched **once**, for the final report |
""")

code(r"""
df_clean, cleaning = clean_and_deduplicate(df)
display(pd.Series(cleaning.to_dict(), name="count").drop("notes").to_frame())

splits = make_splits(df_clean, test_size=0.15, cal_size=0.15, random_state=RANDOM_STATE)
assert_no_overlap(splits.X_train, splits.X_cal, splits.X_test)   # raises if any post leaks across splits
print("Split sizes:", splits.sizes(), "| overlap check passed ✓")

X_train, y_train = splits.X_train, splits.y_train
X_cal, y_cal = splits.X_cal, splits.y_cal
X_test, y_test = splits.X_test, splits.y_test

pd.DataFrame({
    "train": y_train.value_counts(normalize=True),
    "calibration": y_cal.value_counts(normalize=True),
    "test": y_test.value_counts(normalize=True),
}).reindex(labels_present).style.format("{:.1%}")
""")

md(r"""
### 4.1 How much would duplicate leakage have inflated the results?

This is a controlled demonstration. The *same* model is trained twice:
* **(a)** on a naive random split of the raw data, which still contains duplicates
* **(b)** on the de-duplicated split above

Any gap between the two scores comes from memorised duplicates, not from skill.
""")

code(r"""
naive = df.dropna(subset=[TEXT_COL])
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
    naive[TEXT_COL], naive[LABEL_COL], test_size=0.15, stratify=naive[LABEL_COL], random_state=RANDOM_STATE
)
leaked = Xn_te.map(normalise_for_dedup).isin(set(Xn_tr.map(normalise_for_dedup))).mean()

probe = model_ladder(RANDOM_STATE, include_xgb=False)["4 Word TF-IDF + LogReg"]
f1_naive = f1_score(yn_te, probe.fit(Xn_tr, yn_tr).predict(Xn_te), average="macro")
f1_clean = f1_score(y_test, probe.fit(X_train, y_train).predict(X_test), average="macro")

leak_demo = pd.DataFrame({
    "test posts with a duplicate in train": [f"{leaked:.1%}", "0.0%"],
    "macro-F1": [round(f1_naive, 4), round(f1_clean, 4)],
}, index=["(a) naive split, duplicates kept", "(b) de-duplicated split (used below)"])
leak_demo
""")

# =====================================================================
md(r"""
## 5. RQ1: Do interpretable writing-style features differ by category?

Twelve **stylometric** features are computed for each post. Rates are normalised by word count so they are not just proxies for length. Two of them have prior literature behind them:
* **First-person singular pronouns.** Higher use of "I/me/my" has repeatedly been associated with depressive language (e.g. Rude et al., 2004).
* **Absolutist words** such as *always*, *never* and *completely*. These are elevated in anxiety, depression and suicidal-ideation forums (Al-Mosaiwi & Johnstone, 2018).

**Statistical design.** Post lengths and rates are heavily skewed, so the analysis uses **Kruskal–Wallis** tests (non-parametric) across all classes and reports effect size **ε²**. A follow-up gives **Cliff's δ** for each class against *Normal*. Benjamini–Hochberg controls the false-discovery rate across the 12 tests.

With tens of thousands of posts, almost *any* difference is "significant", so the **effect sizes** are what to read. All of this runs on the **training split only**.
""")

code(r"""
style_train = stylometric_frame(X_train).assign(label=y_train.to_numpy())

kw = kruskal_by_group(style_train, STYLOMETRIC_FEATURES, "label")
kw["magnitude"] = pd.cut(kw["epsilon_sq"], [-np.inf, 0.01, 0.08, 0.26, np.inf], labels=["negligible", "small", "medium", "large"])
display(kw.style.format({"H": "{:,.0f}", "p_value": "{:.2e}", "q_value_bh": "{:.2e}", "epsilon_sq": "{:.3f}"}))

reference = "Normal" if "Normal" in labels_present else labels_present[0]
delta = cliffs_vs_reference(style_train, STYLOMETRIC_FEATURES, "label", reference)
delta = delta[[c for c in labels_present if c in delta.columns]]

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(delta.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(delta.shape[1]), delta.columns, rotation=30, ha="right")
ax.set_yticks(range(delta.shape[0]), delta.index)
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        v = delta.iat[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.5 else "#0b0b0b")
ax.grid(False)
fig.colorbar(im, ax=ax, label=f"Cliff's δ vs {reference}  (+ = higher than {reference})")
ax.set_title(f"Stylometric differences vs '{reference}' (train split)")
savefig("cliffs_delta_heatmap"); plt.show()
""")

md(r"""
**How to read this.** |δ| < 0.15 is negligible, about 0.33 medium and over 0.47 large (Romano et al., 2006). Expect clinical categories to differ from *Normal* mainly in **length** and **first-person rate**.

Part of the length difference may be a *source* artefact (§2): short posts from one platform against long posts from another, rather than anything about mental health. This is exactly why the effect sizes are only a description of this dataset and say nothing about people in general.
""")

# =====================================================================
md(r"""
## 6. RQ2: The model ladder

Each rung must earn its place by beating the one below it under **stratified k-fold cross-validation on the training split**:

| # | Model | What it tests |
|---|---|---|
| 0 | Majority class | the floor any model must beat |
| 1 | Stylometric + logistic regression | can *style alone* (12 features) separate classes? |
| 2 | Stylometric + XGBoost | do non-linear interactions of style help? |
| 3 | Word TF-IDF + Complement NB | classic strong text baseline for imbalanced data |
| 4 | Word TF-IDF + logistic regression | discriminative linear model on words |
| 5 | Word + char TF-IDF + logistic regression | adds robustness to misspellings and slang |

The final family is chosen **from these CV results** (§7), not decided in advance.

Linear models use `class_weight="balanced"` and XGBoost uses inverse-frequency sample weights, so the rare classes are not ignored. Every vectoriser lives **inside** the pipeline, so it is re-fitted within each fold and no vocabulary leaks from validation folds.
""")

code(r"""
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
ladder = model_ladder(RANDOM_STATE, include_xgb=True)

rows = []
for name, pipe in ladder.items():
    t0 = time.time()
    res = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={"macro_f1": "f1_macro", "balanced_acc": "balanced_accuracy", "accuracy": "accuracy"},
        n_jobs=1 if "XGBoost" in name else -1,
    )
    rows.append({
        "model": name,
        "macro_f1": res["test_macro_f1"].mean(), "macro_f1_sd": res["test_macro_f1"].std(),
        "balanced_acc": res["test_balanced_acc"].mean(), "accuracy": res["test_accuracy"].mean(),
        "fit_s_per_fold": res["fit_time"].mean(),
    })
    print(f"{name:<34} macro-F1 {rows[-1]['macro_f1']:.4f} ± {rows[-1]['macro_f1_sd']:.4f}   ({time.time() - t0:.0f}s)")

cv_table = pd.DataFrame(rows).set_index("model")
cv_table.to_csv(REPORT_DIR / "cv_model_ladder.csv")

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(cv_table.index[::-1], cv_table["macro_f1"][::-1], xerr=cv_table["macro_f1_sd"][::-1],
        color=PALETTE[0], height=0.55, error_kw={"elinewidth": 1})
for y, (v, sd) in enumerate(zip(cv_table["macro_f1"][::-1], cv_table["macro_f1_sd"][::-1])):
    ax.text(v + sd + 0.015, y, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 1); ax.set_xlabel(f"cross-validated macro-F1 ({CV_FOLDS}-fold, mean ± sd)")
ax.set_title("Model ladder"); ax.grid(axis="y", visible=False)
savefig("model_ladder"); plt.show()
cv_table.style.format("{:.4f}").format({"fit_s_per_fold": "{:.1f}"})
""")

md(r"""
**Reading the ladder.** The gap between rungs 0, 1 and 2 shows how much signal *style alone* carries. The jump to rungs 3 to 5 shows how much the *words* add. If word + char does not clearly beat word-only, given the fold-to-fold spread, the simpler model is the defensible choice.
""")

# =====================================================================
md(r"""
## 7. Hyper-parameter search

The search tunes whichever **linear TF-IDF family won the ladder** (rung 4 or 5). Linear models are kept for the final model because they give exact explanations and clean logits for calibration, and the gap to non-linear alternatives is small. The search is **randomised**. It samples the regularisation strength log-uniformly and tries alternative n-gram ranges. For speed the search runs on a stratified subsample of the training split, and the winning configuration is then **re-fitted on the full training split**. The test set remains untouched.
""")

code(r"""
X_tune, _, y_tune, _ = (
    train_test_split(X_train, y_train, train_size=TUNE_SAMPLE, stratify=y_train, random_state=RANDOM_STATE)
    if len(X_train) > TUNE_SAMPLE else (X_train, None, y_train, None)
)

linear_rungs = ["4 Word TF-IDF + LogReg", "5 Word+Char TF-IDF + LogReg"]
FINAL_FAMILY = cv_table.loc[linear_rungs, "macro_f1"].idxmax()
USE_CHAR = FINAL_FAMILY.startswith("5")
print("Tuning family:", FINAL_FAMILY)

prefix = "tfidf__word__" if USE_CHAR else "tfidf__"
param_dist = {
    "clf__C": loguniform(0.5, 30),
    f"{prefix}ngram_range": [(1, 1), (1, 2), (1, 3)],
    f"{prefix}min_df": [1, 2, 3],
}
if USE_CHAR:
    param_dist["tfidf__char__ngram_range"] = [(2, 4), (3, 5)]

search = RandomizedSearchCV(
    model_ladder(RANDOM_STATE, include_xgb=False)[FINAL_FAMILY],
    param_distributions=param_dist,
    n_iter=SEARCH_ITER, scoring="f1_macro",
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE, n_jobs=-1, refit=False,
)
search.fit(X_tune, y_tune)

search_results = (
    pd.DataFrame(search.cv_results_)
      .sort_values("rank_test_score")
      [["params", "mean_test_score", "std_test_score"]]
)
display(search_results.head(5))
BEST_PARAMS = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
print("Best params:", BEST_PARAMS)

final_model = model_ladder(RANDOM_STATE, include_xgb=False)[FINAL_FAMILY].set_params(**BEST_PARAMS)
t0 = time.time(); final_model.fit(X_train, y_train)
print(f"Final model refit on {len(X_train):,} posts in {time.time() - t0:.0f}s")
LABELS = list(final_model.classes_)
""")

# =====================================================================
md(r"""
## 8. RQ3a: Probability calibration

A model that says "90% Suicidal" should be right about 90% of the time when it says so. Regularised logistic regression on high-dimensional TF-IDF is often **miscalibrated**, and class weighting makes this worse.

The fix is **temperature scaling** (Guo et al., 2017). It learns a single scalar *T* that divides the logits. This leaves every predicted class unchanged, so accuracy and F1 are unaffected, while the confidence becomes realistic. *T* is fitted on **calibration half A** only.
""")

code(r"""
X_calA, X_calB, y_calA, y_calB = train_test_split(
    X_cal, y_cal, test_size=0.5, stratify=y_cal, random_state=RANDOM_STATE
)

ts = TemperatureScaler(LABELS).fit(final_model.decision_function(X_calA), y_calA)
print(f"Fitted temperature T = {ts.temperature_:.3f}  (T > 1 → model was over-confident, T < 1 → under-confident)")

logits_test = final_model.decision_function(X_test)
proba_raw = final_model.predict_proba(X_test)
proba_cal = ts.predict_proba(logits_test)

calib = pd.DataFrame({
    "uncalibrated": classification_metrics(y_test, final_model.predict(X_test), proba_raw, LABELS),
    "temperature-scaled": classification_metrics(y_test, np.array(LABELS)[proba_cal.argmax(1)], proba_cal, LABELS),
}).loc[["log_loss", "brier", "ece", "macro_f1"]]
display(calib.style.format("{:.4f}"))

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", lw=1, label="perfect calibration")
for name, p, c in [("uncalibrated", proba_raw, PALETTE[1]), ("temperature-scaled", proba_cal, PALETTE[0])]:
    rt = reliability_table(y_test, p, LABELS, n_bins=12)
    ax.plot(rt["confidence"], rt["accuracy"], marker="o", ms=5, lw=2, color=c, label=name)
ax.set_xlabel("predicted confidence (top class)"); ax.set_ylabel("observed accuracy")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title("Reliability diagram (test)"); ax.legend(frameon=False)
savefig("reliability_diagram"); plt.show()
""")

# =====================================================================
md(r"""
## 9. RQ2 result: Final evaluation on the held-out test set

This is the **only** time the test set is used. The macro-F1 interval is a 95% **percentile bootstrap** over test posts. It captures test-set sampling uncertainty, but not training variability.
""")

code(r"""
y_pred = np.array(LABELS)[proba_cal.argmax(1)]
FINAL_NAME = f"★ Final: tuned {FINAL_FAMILY[2:]} (calibrated)"

# Re-fit the two key reference rungs on the full training split for context
ref_models = {k: ladder[k] for k in ["0 Majority class", "2 Stylometric + XGBoost"]}
test_rows = {}
for name, m in ref_models.items():
    m.fit(X_train, y_train)
    test_rows[name] = classification_metrics(y_test, m.predict(X_test))
test_rows[FINAL_NAME] = classification_metrics(y_test, y_pred, proba_cal, LABELS)

test_table = pd.DataFrame(test_rows).T
point, lo, hi = bootstrap_ci(y_test, y_pred, n_boot=N_BOOT, random_state=RANDOM_STATE)
display(test_table.style.format("{:.4f}", na_rep="–"))
print(f"Final model macro-F1 = {point:.4f}  (95% bootstrap CI {lo:.4f} – {hi:.4f}, n_test = {len(y_test):,})")

report = pd.DataFrame(classification_report(y_test, y_pred, labels=labels_present, output_dict=True)).T
display(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:,.0f}"}))
""")

code(r"""
cm = confusion_matrix(y_test, y_pred, labels=labels_present, normalize="true")
fig, ax = plt.subplots(figsize=(7.5, 6.3))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels_present)), labels_present, rotation=35, ha="right")
ax.set_yticks(range(len(labels_present)), labels_present)
for i in range(len(labels_present)):
    for j in range(len(labels_present)):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if cm[i, j] > 0.55 else "#0b0b0b")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
ax.set_title("Confusion matrix (test, row-normalised = recall)")
fig.colorbar(im, ax=ax, fraction=0.046)
savefig("confusion_matrix"); plt.show()
""")

md(r"""
**What to look for.** The off-diagonal mass is the interesting part:
* **Depression ↔ Suicidal** confusion is expected. The two categories genuinely overlap in language, and many posts in one community could plausibly sit in the other.
* **Stress ↔ Anxiety** is a similar pair.

These confusions are partly **label ambiguity**, not just model failure. That motivates the prediction *sets* in the next section, which can say "Depression *or* Suicidal" rather than forcing a single guess.
""")

# =====================================================================
md(r"""
## 10. RQ3b: Conformal prediction sets (distribution-free uncertainty)

A single label hides uncertainty. **Split-conformal prediction** turns any probabilistic classifier into a *set-valued* one. Each prediction is a set of labels that contains the true label with probability **≥ 1 − α**. This guarantee needs only one assumption: that the calibration and test posts are exchangeable. No modelling assumptions are required.

* **Score:** LAC, defined as $s(x, y) = 1 - \hat p_y(x)$, computed on **calibration half B** (independent of the half used for temperature).
* **Marginal** conformal guarantees 90% coverage *on average across all posts*. A model can meet that while **under-covering a rare class**.
* **Class-conditional (Mondrian)** conformal computes a quantile *per class*, so the guarantee holds **within every class**. Here that includes *Suicidal*, the class where a miss is most costly.
""")

code(r"""
proba_calB = ts.predict_proba(final_model.decision_function(X_calB))

cp_marg = SplitConformalClassifier(LABELS, alpha=ALPHA, class_conditional=False).fit(proba_calB, y_calB)
cp_mond = SplitConformalClassifier(LABELS, alpha=ALPHA, class_conditional=True).fit(proba_calB, y_calB)

cov_marg = coverage_report(cp_marg.predict_sets(proba_cal), y_test, LABELS).set_index("class")
cov_mond = coverage_report(cp_mond.predict_sets(proba_cal), y_test, LABELS).set_index("class")
coverage = pd.concat({"marginal": cov_marg, "class-conditional": cov_mond}, axis=1)
coverage.to_csv(REPORT_DIR / "conformal_coverage.csv")
display(coverage.style.format("{:.3f}").format("{:,.0f}", subset=[("marginal", "n"), ("class-conditional", "n")]))

order = [c for c in labels_present if c in cov_marg.index] + ["ALL"]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(x - w / 2, cov_marg.loc[order, "coverage"], w, color=PALETTE[1], label="marginal")
ax.bar(x + w / 2, cov_mond.loc[order, "coverage"], w, color=PALETTE[0], label="class-conditional (Mondrian)")
ax.axhline(1 - ALPHA, ls="--", lw=1, color="#0b0b0b", label=f"target {1 - ALPHA:.0%}")
ax.set_xticks(x, order, rotation=30, ha="right"); ax.set_ylim(0.5, 1.02)
ax.set_ylabel("empirical coverage (test)"); ax.set_title("Conformal coverage by class", pad=34)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3); ax.grid(axis="x", visible=False)
savefig("conformal_coverage"); plt.show()
""")

code(r"""
sets = cp_mond.predict_sets(proba_cal)
set_size = sets.sum(1)
print(f"Mondrian 90% sets — singletons: {(set_size == 1).mean():.1%} | "
      f"mean size: {set_size.mean():.2f} | empty: {(set_size == 0).mean():.1%}")

# Which label combinations does the model hedge between most often?
combos = pd.Series(["+".join(np.array(LABELS)[row]) for row in sets[set_size == 2]]).value_counts().head(8)
display(combos.rename("count").to_frame("posts with a 2-label set"))
""")

md(r"""
**Interpretation.** Per-class coverage on the test set is itself a *finite-sample estimate*. For a class with a few hundred test posts, expect it to land within a few points of 90% rather than exactly on it, and the per-class quantile is only as reliable as the number of calibration posts in that class. Look for the pattern: marginal sets typically **under-cover the rare classes**, and Mondrian pulls them back toward the target.

Singleton sets are cases where the model is reliably confident. Two-label sets show *where* the ambiguity lives, and they tend to match the confusion pairs above. In a hypothetical human-in-the-loop setting, the set size is a principled **triage signal**: large sets mean "defer to a human".
""")

md(r"""
### 10.1 LAC vs adaptive prediction sets (APS)

LAC produces the *smallest* sets on average, but its set size reacts weakly to how hard an input is. **APS** (Romano et al., 2020) uses a different score: the probability mass ranked above the true label, plus a random fraction of the true label's own probability. Hard, ambiguous posts get larger sets and easy ones smaller. The randomisation is essential, not cosmetic. With a confident model, the deterministic variant piles every class's cumulative mass up near 1.0 and produces bloated sets; `tests/test_core.py` checks this directly. A fixed seed keeps it reproducible.

A good uncertainty signal should give **larger sets when the top-1 prediction is wrong**. The table measures exactly that.
""")

code(r"""
cp_aps = SplitConformalClassifier(LABELS, alpha=ALPHA, class_conditional=True, score="aps",
                                  randomized=True, random_state=RANDOM_STATE).fit(proba_calB, y_calB)
top1_wrong = y_pred != y_test.to_numpy()

def set_summary(name, S):
    cov = coverage_report(S, y_test, LABELS).set_index("class")
    size = S.sum(1)
    return {
        "method": name,
        "coverage": cov.loc["ALL", "coverage"],
        "worst-class coverage": cov.drop("ALL")["coverage"].min(),
        "mean set size": size.mean(),
        "singletons": (size == 1).mean(),
        "size | top-1 correct": size[~top1_wrong].mean(),
        "size | top-1 wrong": size[top1_wrong].mean() if top1_wrong.any() else np.nan,
    }

set_methods = pd.DataFrame([
    set_summary("LAC, marginal", cp_marg.predict_sets(proba_cal)),
    set_summary("LAC, class-conditional", cp_mond.predict_sets(proba_cal)),
    set_summary("APS, class-conditional", cp_aps.predict_sets(proba_cal)),
]).set_index("method")
set_methods.to_csv(REPORT_DIR / "conformal_methods.csv")
set_methods.style.format("{:.3f}").format("{:.1%}", subset=["singletons"])
""")

md(r"""
**Reading it.** All three methods should sit near 90% overall coverage. That is the guarantee. Differences show up in *how* the sets spend their size: class-conditional variants protect the worst class, and APS trades a larger average set for stronger adaptivity (a bigger gap between the "correct" and "wrong" columns). The deployed bundle uses class-conditional LAC as the compact default. `smm-train --score aps` switches to APS.
""")

md(r"""
### 10.2 Optional: fine-tuned transformer baseline

Is the transparent linear model leaving accuracy on the table? This section fine-tunes **DistilRoBERTa** for 2 epochs, using class-weighted cross-entropy, AdamW, linear warm-up and mixed precision. It then sends the model through the **identical** temperature-scaling and Mondrian-conformal layer, using the same calibration halves.

The comparison uses a **paired bootstrap** on the same test posts, so the confidence interval is for the *difference* in macro-F1.

It runs automatically on a **GPU runtime** (about 10–15 min on a Colab T4) and is skipped otherwise. Set `SMM_TRANSFORMER=1` to force it, or `0` to disable it.
""")

code(r"""
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
RUN_TRANSFORMER = HAS_GPU if SMM_TRANSFORMER == "auto" else SMM_TRANSFORMER == "1"
transformer_summary = None

if RUN_TRANSFORMER:
    from socialmediamind.transformer import finetune

    t0 = time.time()
    tf_model = finetune(X_train, y_train, LABELS, model_name=TRANSFORMER_NAME, epochs=1 if FAST_MODE else 2,
                        batch_size=32, max_length=256, seed=RANDOM_STATE, log_every=200)
    tf_minutes = (time.time() - t0) / 60
    tf_ts = TemperatureScaler(LABELS).fit(tf_model.decision_function(X_calA), y_calA)
    tf_proba = tf_ts.predict_proba(tf_model.decision_function(X_test))
    tf_pred = np.array(LABELS)[tf_proba.argmax(1)]
    tf_cp = SplitConformalClassifier(LABELS, alpha=ALPHA, class_conditional=True).fit(
        tf_ts.predict_proba(tf_model.decision_function(X_calB)), y_calB)
    tf_cov = coverage_report(tf_cp.predict_sets(tf_proba), y_test, LABELS).set_index("class")

    # paired bootstrap: same resampled test posts for both models
    rng_b = np.random.default_rng(RANDOM_STATE)
    yt = y_test.to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        i = rng_b.integers(0, len(yt), len(yt))
        diffs.append(f1_score(yt[i], tf_pred[i], average="macro") - f1_score(yt[i], y_pred[i], average="macro"))
    d_lo, d_hi = np.quantile(diffs, [0.025, 0.975])

    lin_m = classification_metrics(y_test, y_pred, proba_cal, LABELS)
    tf_m = classification_metrics(y_test, tf_pred, tf_proba, LABELS)
    comparison = pd.DataFrame({
        "linear (final)": {**lin_m, "conformal mean set size": cov_mond.loc["ALL", "mean_set_size"],
                           "worst-class coverage": cov_mond.drop("ALL")["coverage"].min()},
        TRANSFORMER_NAME: {**tf_m, "conformal mean set size": tf_cov.loc["ALL", "mean_set_size"],
                           "worst-class coverage": tf_cov.drop("ALL")["coverage"].min()},
    })
    display(comparison.style.format("{:.4f}"))
    print(f"Δ macro-F1 (transformer − linear) = {tf_m['macro_f1'] - lin_m['macro_f1']:+.4f} "
          f"[95% paired-bootstrap CI {d_lo:+.4f}, {d_hi:+.4f}] | fine-tuning took {tf_minutes:.1f} min")
    transformer_summary = {"model": TRANSFORMER_NAME, "minutes": tf_minutes, "temperature": tf_ts.temperature_,
                           **{f"test_{k}": v for k, v in tf_m.items()},
                           "delta_macro_f1": tf_m["macro_f1"] - lin_m["macro_f1"], "delta_ci95": [d_lo, d_hi],
                           "conformal_mean_set_size": float(tf_cov.loc["ALL", "mean_set_size"])}
else:
    print("Transformer baseline skipped (no GPU detected). Switch to a GPU runtime or set SMM_TRANSFORMER=1.")
""")

md(r"""
**How to judge the trade-off.** If the CI for Δ excludes 0 by a meaningful margin, the transformer is genuinely better, and that gain is weighed against losing *exact* explanations, longer training and roughly 100× slower inference. If the CI straddles 0, the linear model is the defensible choice.

The conformal layer is model-agnostic, so both models keep the same coverage guarantee. A better model shows up as **smaller sets**, not as higher coverage.
""")

# =====================================================================
md(r"""
## 11. RQ4: Explainability

The section looks at the model at three levels:
1. **Global (text model):** the highest-weighted word n-grams per class.
2. **Local (text model):** *exact* additive attributions for single predictions (coefficient × TF-IDF value). For a linear model this is the SHAP value under a zero baseline, but cheaper and exact.
3. **SHAP (stylometric XGBoost):** how the 12 interpretable style features drive a non-linear model.

Attributions explain the **model**, not the people who wrote the posts.
""")

code(r"""
top_terms = top_terms_per_class(final_model, n=12)
top_terms = top_terms[[c for c in labels_present if c in top_terms.columns]]
top_terms.to_csv(REPORT_DIR / "top_terms_per_class.csv", index=False)
top_terms
""")

code(r"""
rng = np.random.default_rng(RANDOM_STATE)
correct_idx = np.flatnonzero(y_pred == y_test.to_numpy())
wrong_idx = np.flatnonzero(y_pred != y_test.to_numpy())
for kind, pool in [("correct", correct_idx), ("misclassified", wrong_idx)]:
    if len(pool) == 0:
        continue
    i = int(rng.choice(pool))
    text = X_test.iloc[i]
    print(f"\n--- {kind}: true = {y_test.iloc[i]} | predicted = {y_pred[i]} | "
          f"90% set = {[str(l) for l in np.array(LABELS)[sets[i]]]} | words = {len(text.split())}")
    if SHOW_TEXT_EXAMPLES:
        print(text[:300] + ("…" if len(text) > 300 else ""))
    display(explain_text(final_model, text, n=10).style.format({"contribution": "{:+.3f}"}))
""")

code(r"""
xgb_pipe = ref_models["2 Stylometric + XGBoost"]        # already fitted on the training split
xgb_est = xgb_pipe.named_steps["clf"].estimator_
style_test = stylometric_frame(X_test)
sample = style_test.sample(min(1_500, len(style_test)), random_state=RANDOM_STATE)

explainer = shap.TreeExplainer(xgb_est)
sv = explainer.shap_values(sample)
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # -> (n, features, classes)
xgb_classes = list(xgb_pipe.named_steps["clf"].classes_)

imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=STYLOMETRIC_FEATURES, columns=xgb_classes)
imp = imp[[c for c in labels_present if c in imp.columns]]
imp = imp.loc[imp.sum(1).sort_values().index]

fig, ax = plt.subplots(figsize=(8.5, 5))
left = np.zeros(len(imp))
for c in imp.columns:
    ax.barh(imp.index, imp[c], left=left, color=COLOR[c], label=c, height=0.62, edgecolor="white", linewidth=1)
    left += imp[c].to_numpy()
ax.set_xlabel("mean |SHAP value| (log-odds), stacked by class"); ax.grid(axis="y", visible=False)
ax.set_title("Stylometric XGBoost: global SHAP importance"); ax.legend(frameon=False, fontsize=8, loc="lower right")
savefig("shap_stylometric_importance"); plt.show()

focus = "Suicidal" if "Suicidal" in xgb_classes else xgb_classes[0]
shap.summary_plot(sv[:, :, xgb_classes.index(focus)], sample, show=False, max_display=12)
plt.title(f"SHAP beeswarm, class = {focus}"); plt.tight_layout()
savefig("shap_beeswarm_focus_class"); plt.show()
""")

# =====================================================================
md(r"""
## 12. RQ4: Error analysis and robustness slices

The dataset has **no demographic attributes**, so a fairness evaluation across protected groups is *not possible*, and none is claimed. Two checks can still be made:
* **Length slices.** Does the model degrade on very short posts (little evidence) or very long ones?
* **Confident errors.** Mistakes made with high confidence are the most dangerous kind and often reveal label noise.
""")

code(r"""
lengths = X_test.str.split().str.len().to_numpy()
bins = pd.qcut(lengths, q=5, duplicates="drop")
slice_rows = []
for b in bins.categories:
    m = np.asarray(bins == b)
    slice_rows.append({
        "length bucket (words)": str(b), "n": int(m.sum()),
        "macro_f1": f1_score(y_test[m], y_pred[m], average="macro"),
        "accuracy": float((y_test[m].to_numpy() == y_pred[m]).mean()),
        "mean 90% set size": float(set_size[m].mean()),
    })
slices = pd.DataFrame(slice_rows)
display(slices.style.format({"macro_f1": "{:.3f}", "accuracy": "{:.3f}", "mean 90% set size": "{:.2f}"}))

conf = proba_cal.max(1)
errors = pd.DataFrame({"true": y_test.to_numpy(), "pred": y_pred, "confidence": conf, "words": lengths})
errors = errors[errors["true"] != errors["pred"]]
print(f"Errors: {len(errors):,} of {len(y_test):,} | errors with confidence > 0.9: {(errors['confidence'] > 0.9).sum():,}")
display(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(8).rename("count").to_frame())
""")

md(r"""
**Typical findings to discuss.** Very short posts carry little evidence: they show lower F1 and larger conformal sets, which is the correct behaviour. Many high-confidence "errors" turn out to be plausibly mislabelled, or genuinely ambiguous between two communities. (Set `SHOW_TEXT_EXAMPLES = True` locally to inspect them.)
""")

md(r"""
### 12.1 How noisy are the labels? (confident learning)

Some of those confident errors may be the *label's* fault, not the model's. **Confident learning** (Northcutt et al., 2021) estimates this directly:
1. Score every training post with a model that **never saw it** (out-of-fold predictions).
2. Compute per-class confidence thresholds.
3. Count (given label → confidently-predicted label) pairs. This gives the *confident joint*, an estimate of the label-noise matrix.

Flagged posts are *candidates* for review, not proven errors. The final model is **not** changed based on this analysis. The pruning experiment below is reported only to show how sensitive the results are to label noise.
""")

code(r"""
noise_probe = model_ladder(RANDOM_STATE, include_xgb=False)["4 Word TF-IDF + LogReg"]
proba_oof = cross_val_predict(noise_probe, X_train, y_train, method="predict_proba", n_jobs=-1,
                              cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
issues, joint = find_label_issues(proba_oof, y_train, LABELS)
noise_rates = estimated_noise_rate(joint).reindex(labels_present)
print(f"Flagged as likely label issues: {issues['is_issue'].sum():,} of {len(issues):,} training posts "
      f"({issues['is_issue'].mean():.1%})")

J = joint.reindex(index=labels_present, columns=labels_present)
Jn = J.div(J.sum(1), axis=0)
fig, ax = plt.subplots(figsize=(7.5, 6.3))
im = ax.imshow(Jn.to_numpy(), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels_present)), labels_present, rotation=35, ha="right")
ax.set_yticks(range(len(labels_present)), labels_present)
for i in range(len(labels_present)):
    for j in range(len(labels_present)):
        v = Jn.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 else "#0b0b0b")
ax.set_xlabel("confidently predicted label"); ax.set_ylabel("given label"); ax.grid(False)
ax.set_title("Confident joint (row-normalised): estimated label-noise matrix")
fig.colorbar(im, ax=ax, fraction=0.046)
savefig("label_noise_confident_joint"); plt.show()

# Sensitivity: same final configuration trained without the flagged posts
keep = ~issues["is_issue"].to_numpy()
pruned_model = clone(final_model).fit(X_train[keep], y_train[keep])
f1_pruned = f1_score(y_test, pruned_model.predict(X_test), average="macro")
noise_table = pd.DataFrame({
    "training posts": [len(X_train), int(keep.sum())],
    "test macro-F1": [f1_score(y_test, y_pred, average="macro"), f1_pruned],
}, index=["final model (all training posts)", "same config, flagged posts removed"])
display(noise_rates.rename("estimated noise rate").to_frame().style.format("{:.1%}"))
noise_table.style.format({"test macro-F1": "{:.4f}", "training posts": "{:,}"})
""")

md(r"""
**Caveat when reading the pruning result.** The test labels carry the *same* kind of noise. Removing ambiguous training posts can therefore look neutral or even harmful on a noisy test set while still producing a cleaner model. The estimated noise matrix is the more trustworthy output here: it shows which category boundaries are blurry *in the data itself*.
""")

# =====================================================================
md(r"""
## 13. RQ5: Unsupervised structure (topics without labels)

Does the language organise itself into stable clusters before any labels are used?

**Pipeline:** word TF-IDF → **LSA** (truncated SVD, 100 dimensions) → L2-normalise → **k-means** (cosine geometry).

The number of clusters *k* is chosen with **internal** criteria: silhouette, plus stability measured by the adjusted Rand index (ARI) across random restarts. Only *after* choosing *k* are the clusters compared with the labels (NMI and ARI). This keeps the analysis genuinely unsupervised.
""")

code(r"""
Xc, _, yc, _ = (
    train_test_split(X_train, y_train, train_size=CLUSTER_SAMPLE, stratify=y_train, random_state=RANDOM_STATE)
    if len(X_train) > CLUSTER_SAMPLE else (X_train, None, y_train, None)
)
vec_c = TfidfVectorizer(min_df=3, max_df=0.5, stop_words="english", sublinear_tf=True, max_features=30_000)
Tc = vec_c.fit_transform(Xc)
svd = TruncatedSVD(n_components=min(100, Tc.shape[1] - 1), random_state=RANDOM_STATE)
Z = Normalizer(copy=False).fit_transform(svd.fit_transform(Tc))
print(f"LSA on {len(Xc):,} posts | {svd.n_components} components explain {svd.explained_variance_ratio_.sum():.1%} of variance")

k_rows, labelings = [], {}
for k in range(3, 11):
    runs = [KMeans(k, n_init=5, random_state=RANDOM_STATE + s).fit_predict(Z) for s in range(5)]
    labelings[k] = runs[0]
    ari_stab = np.mean([adjusted_rand_score(runs[0], r) for r in runs[1:]])
    k_rows.append({
        "k": k,
        "silhouette": silhouette_score(Z, runs[0], sample_size=min(5_000, len(Z)), random_state=RANDOM_STATE),
        "stability_ARI": ari_stab,
    })
k_table = pd.DataFrame(k_rows).set_index("k")
k_table["score"] = k_table["silhouette"].rank(pct=True) + k_table["stability_ARI"].rank(pct=True)
BEST_K = int(k_table["score"].idxmax())
display(k_table.style.format("{:.3f}"))
print("Selected k =", BEST_K, "(internal criteria only)")
""")

code(r"""
clusters = labelings[BEST_K]
centroids = svd.inverse_transform(KMeans(BEST_K, n_init=5, random_state=RANDOM_STATE).fit(Z).cluster_centers_)
terms = vec_c.get_feature_names_out()
cluster_terms = pd.DataFrame({f"cluster {c}": terms[np.argsort(centroids[c])[::-1][:10]] for c in range(BEST_K)})
display(cluster_terms)

print(f"Agreement with labels — NMI: {normalized_mutual_info_score(yc, clusters):.3f} | ARI: {adjusted_rand_score(yc, clusters):.3f}")
ct = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(yc.to_numpy(), name="label"), normalize="index")
ct = ct[[c for c in labels_present if c in ct.columns]]
display(ct.style.format("{:.0%}").background_gradient(cmap="Blues", axis=None))
""")

md(r"""
**Interpretation.** Clusters usually capture **topics** (school or work stress, relationships, medication, everyday chatter) more than diagnostic categories. Moderate NMI means the labels are *partly* topical and *partly* something finer.

That is consistent with the confusion structure in §9, and it is a useful caution: a high-scoring classifier may be leaning on topic and source cues rather than on markers of mental state.
""")

# =====================================================================
md("## 14. Artefacts, model card and inference API")

code(r"""
smm_model = SocialMediaMindModel(
    pipeline=final_model, temperature=ts.temperature_, conformal=cp_mond,
    labels=[str(l) for l in LABELS],
    metadata={"family": FINAL_FAMILY, "best_params": BEST_PARAMS, "test_macro_f1": point, "environment": ENV},
)
bundle_path = smm_model.save(MODEL_DIR / "socialmediamind_bundle.joblib")
reloaded = SocialMediaMindModel.load(bundle_path)          # round-trip check
reloaded.predict(["Had a great time hiking with friends this weekend!",
                  "I can't stop worrying about everything, my heart keeps racing at night."])
""")

code(r"""
metrics = {
    "dataset": {"source": "kaggle:suchintikasarkar/sentiment-analysis-for-mental-health",
                "cleaning": cleaning.to_dict(), "splits": splits.sizes()},
    "leakage_demo": {"naive_macro_f1": f1_naive, "dedup_macro_f1": f1_clean, "naive_test_dup_share": float(leaked)},
    "cv_model_ladder": cv_table.round(4).to_dict(orient="index"),
    "best_params": {k: (list(v) if isinstance(v, tuple) else float(v) if isinstance(v, float) else v)
                    for k, v in BEST_PARAMS.items()},
    "test": {**test_table.loc[FINAL_NAME].dropna().to_dict(),
             "macro_f1_ci95": [lo, hi]},
    "calibration": {"temperature": ts.temperature_, **calib["temperature-scaled"].to_dict()},
    "conformal": {"alpha": ALPHA, "mondrian_overall_coverage": float(cov_mond.loc["ALL", "coverage"]),
                  "mondrian_mean_set_size": float(cov_mond.loc["ALL", "mean_set_size"]),
                  "min_class_coverage_marginal": float(cov_marg.drop("ALL")["coverage"].min()),
                  "min_class_coverage_mondrian": float(cov_mond.drop("ALL")["coverage"].min())},
    "conformal_methods": set_methods.round(4).to_dict(orient="index"),
    "label_noise": {"flagged_share": float(issues["is_issue"].mean()),
                    "noise_rate_by_class": noise_rates.round(4).to_dict(),
                    "test_macro_f1_pruned": float(f1_pruned)},
    "transformer": transformer_summary,
    "clustering": {"k": BEST_K, "nmi_vs_labels": float(normalized_mutual_info_score(yc, clusters))},
    "environment": ENV,
}
(REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

per_class = report.loc[[c for c in labels_present if c in report.index], ["precision", "recall", "f1-score", "support"]]
per_class_md = "| class | precision | recall | F1 | support |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support']):,} |"
    for c, r in per_class.iterrows()
)
card = f'''# Model card: SocialMediaMind text classifier

*Auto-generated by `notebooks/SocialMediaMind.ipynb` on {smm_model.metadata["saved_utc"]}.*

## Model
{FINAL_FAMILY[2:]} → multinomial logistic regression (class-balanced), temperature-scaled
(T = {ts.temperature_:.3f}), with class-conditional split-conformal prediction sets at {1 - ALPHA:.0%} coverage.
Hyper-parameters: `{BEST_PARAMS}`.

## Intended use
Research, teaching and portfolio demonstration of trustworthy-ML practice on text.
**Out of scope:** diagnosis, screening, triage of real people, content moderation decisions,
or any decision affecting an individual.

## Data
Kaggle `suchintikasarkar/sentiment-analysis-for-mental-health` (public social-media posts; labels reflect
source community, not clinical assessment). After cleaning: {cleaning.n_final:,} unique posts
({cleaning.n_exact_duplicates_removed:,} duplicates and {cleaning.n_conflicting_rows_removed:,} label-conflicting rows removed).
Splits: {splits.sizes()}.

## Performance (held-out test, n = {len(y_test):,})
* Macro-F1 **{point:.3f}** (95% bootstrap CI {lo:.3f}–{hi:.3f})
* Balanced accuracy {test_table.iloc[-1]["balanced_accuracy"]:.3f}, accuracy {test_table.iloc[-1]["accuracy"]:.3f}
* ECE after temperature scaling {calib.loc["ece", "temperature-scaled"]:.3f} (before: {calib.loc["ece", "uncalibrated"]:.3f})
* Conformal sets: coverage {cov_mond.loc["ALL", "coverage"]:.3f}, mean size {cov_mond.loc["ALL", "mean_set_size"]:.2f};
  lowest per-class coverage {cov_mond.drop("ALL")["coverage"].min():.3f}

{per_class_md}

## Additional analyses
* Confident learning flags {issues["is_issue"].mean():.1%} of training posts as likely label issues; the estimated
  per-class noise rate is highest for **{noise_rates.idxmax()}** ({noise_rates.max():.1%}).
* Transformer baseline: {"not run (no GPU)" if transformer_summary is None else
  f'{TRANSFORMER_NAME} macro-F1 {transformer_summary["test_macro_f1"]:.3f}, Δ vs linear {transformer_summary["delta_macro_f1"]:+.3f} (95% CI {transformer_summary["delta_ci95"][0]:+.3f} to {transformer_summary["delta_ci95"][1]:+.3f})'}.

## Usage
```bash
smm-predict --model models/socialmediamind_bundle.joblib "text to score"
```

## Limitations and risks
* Labels are noisy proxies; the model can exploit **source and platform artefacts** (e.g. length, style).
* No demographic data, so **fairness across groups is unassessed**.
* English-only, single time period; expect degradation under distribution shift. Conformal guarantees
  assume exchangeability and **do not hold** under such shift.
* False negatives on crisis content are possible; this model must never be the only safeguard.
'''
(REPORT_DIR / "MODEL_CARD.md").write_text(card)
print("Wrote:", *sorted(p.relative_to(ROOT).as_posix() for p in list(REPORT_DIR.glob("*.*")) + list(MODEL_DIR.glob("*.*"))), sep="\n  ")
""")

# =====================================================================
md(r"""
## 15. Summary, limitations and next steps

The final cell prints the headline numbers; `reports/metrics.json` and `reports/MODEL_CARD.md` hold the full record.

### What the evidence supports
* Interpretable stylometric features differ across categories with measurable effect sizes (RQ1). They carry real but limited predictive signal (ladder rungs 1 and 2).
* Word and character n-grams add substantial signal over style alone (RQ2). The gain is measured with leakage controlled, and §4.1 quantifies how much leakage would otherwise inflate it.
* Temperature scaling repairs miscalibration without changing a single prediction. Class-conditional conformal sets restore the coverage guarantee **inside** every class, including rare ones (RQ3).
* Errors concentrate in semantically adjacent pairs (Depression/Suicidal, Anxiety/Stress), and both clustering and the conformal sets point to the same ambiguity (RQ4, RQ5). Confident learning independently finds the same blurry boundaries in the labels themselves.
* The transformer comparison (when run) shows whether added model capacity buys anything once leakage, calibration and uncertainty are all held to the same standard.

### What it does *not* support
* Any clinical, diagnostic or causal claim.
* Generalisation to other platforms, languages, time periods, or to individuals.
* Any statement about fairness across demographic groups (no such data).

### Next steps
1. **Source-aware evaluation.** The public CSV does not record which source dataset each post came from. Recovering it would allow a train-on-one-source / test-on-another evaluation that directly probes the platform-artefact concern.
2. **Conformal risk control** to bound the *Suicidal* false-negative rate at a chosen level, rather than coverage in general.
3. **Domain-specific transformers** (e.g. MentalRoBERTa, gated on Hugging Face) and longer fine-tuning, assessed with the same paired-bootstrap protocol.
4. **Temporal and cross-platform shift tests**, plus weighted or adaptive conformal methods that stay valid under shift.
""")

code(r"""
print("=" * 72, "\nSOCIALMEDIAMIND — HEADLINE RESULTS\n" + "=" * 72)
print(f"Unique posts after cleaning : {cleaning.n_final:,}  (from {cleaning.n_raw:,} raw)")
print(f"Leakage demo                : naive macro-F1 {f1_naive:.3f} vs de-duplicated {f1_clean:.3f}")
print(f"Best CV rung                : {cv_table['macro_f1'].idxmax()}  ({cv_table['macro_f1'].max():.3f})")
print(f"Test macro-F1 (final)       : {point:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
print(f"ECE  raw → calibrated       : {calib.loc['ece', 'uncalibrated']:.3f} → {calib.loc['ece', 'temperature-scaled']:.3f}  (T = {ts.temperature_:.2f})")
print(f"Conformal 90% (Mondrian)    : coverage {cov_mond.loc['ALL', 'coverage']:.3f}, mean set size {cov_mond.loc['ALL', 'mean_set_size']:.2f}, "
      f"worst class {cov_mond.drop('ALL')['coverage'].min():.3f} (marginal worst {cov_marg.drop('ALL')['coverage'].min():.3f})")
print(f"APS vs LAC (Mondrian)        : mean set {set_methods.loc['APS, class-conditional', 'mean set size']:.2f} vs "
      f"{set_methods.loc['LAC, class-conditional', 'mean set size']:.2f}")
print(f"Label noise (confident lrn.) : {issues['is_issue'].mean():.1%} of training posts flagged")
if transformer_summary:
    print(f"Transformer Δ macro-F1      : {transformer_summary['delta_macro_f1']:+.3f} "
          f"[{transformer_summary['delta_ci95'][0]:+.3f}, {transformer_summary['delta_ci95'][1]:+.3f}]")
print(f"Topic clusters              : k = {BEST_K}, NMI vs labels {normalized_mutual_info_score(yc, clusters):.3f}")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells, embedded package sha256 {_sha[:12]}…)")
