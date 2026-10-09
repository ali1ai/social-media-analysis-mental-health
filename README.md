# SocialMediaMind

**Interpretable, calibrated, uncertainty-aware text classification of mental-health-related social-media posts.**

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> ⚠️ **Research and portfolio project, not a clinical tool.** Dataset labels reflect categories assigned in the source dataset, not clinical diagnoses. Do not use this model to diagnose, screen, triage, or make decisions about people. If you need support, call or text **988** in the US, call or text **9-8-8 in Canada** ([official Canadian service](https://988.ca/)), call **116 123** in the UK or Ireland, or visit [findahelpline.com](https://findahelpline.com/) to find a local service. If someone is in immediate danger, contact local emergency services.

Most text-classification projects stop at a performance score. SocialMediaMind treats that score as the start of the evaluation and asks five further questions:

1. **Is the score real?** Duplicate and conflicting-text records are addressed before splitting. Automated checks verify that normalized text does not overlap across splits, and a controlled experiment measures how much a naive evaluation can overstate performance.
2. **Is the complexity earned?** A model ladder compares a majority baseline, stylometric models, word TF-IDF, and word-plus-character TF-IDF using cross-validated **macro-F1**. A fine-tuned DistilRoBERTa comparison is optional; it was **not evaluated in the recorded run** because the notebook skipped it when no CUDA GPU was detected.
3. **Can the probabilities be trusted?** Temperature scaling is evaluated with reliability metrics, including Expected Calibration Error (ECE) and Brier score.
4. **How uncertain is each prediction?** Split-conformal prediction sets are evaluated using LAC and randomized APS scores, with marginal and class-conditional (Mondrian) variants. The nominal target is 90%; observed coverage is reported explicitly. Coverage depends on the method's assumptions and is not a guarantee for an individual prediction or under arbitrary distribution shift.
5. **How trustworthy are the labels?** Confident learning estimates potential label issues and flags examples for review. These flags are model-based candidates, not confirmed annotation errors.

## Results

Results below are from the recorded notebook run. The full metrics record is written to [`reports/metrics.json`](reports/metrics.json), and the automatically generated model card is available at [`reports/MODEL_CARD.md`](reports/MODEL_CARD.md).

| Measure | Recorded result |
|---|---:|
| Unique posts after cleaning | **50,940** (from 53,043 raw rows) |
| Leakage demonstration: naive vs. de-duplicated Macro-F1 | **0.7666 vs. 0.7445** (Δ +0.0221) |
| Best model-ladder CV Macro-F1 | **0.758** (Word + Character TF-IDF + Logistic Regression) |
| Final held-out test Macro-F1 | **0.7574** (95% bootstrap CI: **0.7428–0.7716**) |
| Final held-out test accuracy | **0.8009** |
| Final held-out test balanced accuracy | **0.7722** |
| ECE before → after temperature scaling | **0.0225 → 0.0168** (T = 0.891; calibration evaluation split) |
| Mondrian conformal: observed coverage / mean set size | **0.911 / 1.42** at a nominal 90% target |
| Lowest observed class coverage: Mondrian / marginal conformal | **0.891 / 0.739** |
| Mean set size: class-conditional APS vs. LAC | **1.64 vs. 1.42** |
| Training posts flagged as potential label issues | **8.7%** |
| Pruning sensitivity: Macro-F1 before → after removing flagged rows | **0.7574 → 0.7570** (Δ −0.0005; no improvement) |
| Transformer comparison | **Not evaluated** in the recorded run |
| Unsupervised clustering | **k = 5**, NMI = 0.263, ARI = 0.286; separation was weak |

The naive and de-duplicated leakage scores come from different evaluation samples, so the difference demonstrates the potential impact of duplicate leakage but should not be interpreted as an isolated causal estimate. ECE values above are from the calibration evaluation split, not the held-out test set. Conformal figures are observed empirical results; theoretical coverage relies on the relevant exchangeability assumptions.

Figures are saved to `reports/figures/`, and the model card documents intended use, evaluation, and limitations.

## What's inside

| Section | Techniques |
|---|---|
| Data audit and cleaning | Normalized de-duplication, removal of conflicting-text rows, stratified train/calibration/test split, automated overlap assertion, quantified leakage demonstration |
| Statistics (RQ1) | Kruskal–Wallis tests, epsilon-squared effect sizes, Cliff's delta vs. Normal, Benjamini–Hochberg FDR correction; stylometric features including first-person and absolutist-word rates |
| Modelling (RQ2) | Majority baseline → stylometric Logistic Regression/XGBoost → TF-IDF ComplementNB/Logistic Regression → word + character TF-IDF; stratified cross-validation and randomized hyperparameter search; optional DistilRoBERTa fine-tuning |
| Calibration and uncertainty (RQ3) | Temperature scaling, reliability plots, ECE, Brier score; split-conformal sets using LAC and randomized APS, marginal and Mondrian variants; set-size analysis |
| Explainability (RQ4) | Per-class n-gram weights, local linear-model feature contributions, SHAP analysis for the stylometric XGBoost model |
| Error analysis and data quality | Confusion matrix, high-confidence errors, text-length slices, confident-learning label-issue estimates and a pruning sensitivity experiment |
| Unsupervised analysis (RQ5) | TF-IDF, LSA, K-means, silhouette and restart-stability analysis, post-selection NMI/ARI comparison with existing labels |
| Engineering | `src/` package, unit/statistical/end-to-end tests, notebook generation with embedded package code, `smm-train` / `smm-predict` CLIs, CI, serialized model bundle, automated model card |

## Repository layout

```text
├── notebooks/SocialMediaMind.ipynb     # Full analysis; run top to bottom
├── src/socialmediamind/
│   ├── data.py                         # Loading, cleaning, de-duplication, leakage-safe splits
│   ├── features.py                     # Stylometric features and TF-IDF pipelines
│   ├── stats.py                        # Kruskal–Wallis, Cliff's delta, BH-FDR
│   ├── models.py                       # Model ladder and explanation helpers
│   ├── calibration.py                  # Temperature scaling
│   ├── conformal.py                    # LAC / randomized APS; marginal / Mondrian sets
│   ├── noise.py                        # Confident learning and label-issue estimation
│   ├── evaluation.py                   # Metrics, bootstrap CIs, ECE, reliability tables
│   ├── transformer.py                  # Optional DistilRoBERTa fine-tuning (PyTorch)
│   ├── inference.py                    # SocialMediaMindModel bundle and prediction CLI
│   └── train.py                        # Notebook-free training and training CLI
├── tests/                              # Unit, statistical and end-to-end tests
├── scripts/build_notebook.py            # Builds a notebook with embedded src/ code
├── scripts/make_synthetic_fixture.py    # Synthetic fixture for CI and smoke tests
├── docs/AUDIT_v01.md                   # Audit of the earlier version
├── docs/PUBLISHING.md                   # GitHub publishing notes
├── Makefile                            # install / test / smoke / train / notebook
└── reports/, models/                   # Generated reports and model artefacts
```

## Quick start

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload `notebooks/SocialMediaMind.ipynb`. After the repository is published, you can also open the notebook from GitHub.
2. Optionally select **Runtime → Change runtime type → T4 GPU** to attempt the transformer experiment.
3. Choose **Runtime → Run all**.

The notebook includes its helper package and attempts to download the public dataset. A GitHub account is not required to upload and run the notebook in Colab. If the dataset download is blocked, use the notebook's upload option with `Combined Data.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health). Transformer fine-tuning is optional and can be slow or unavailable depending on the runtime.

### Local environment

```bash
cd socialmediamind
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell alternative:
# .venv\Scripts\Activate.ps1

make install       # pip install -e ".[data,dev]"
make test          # linting and tests
make smoke         # pipeline on synthetic data; no Kaggle dataset needed
make train         # train on real data and write models/ + reports/
```

To use a local dataset copy, place `Combined Data.csv` in `data/` or set `SMM_DATA_PATH` to its path.

| Environment variable | Effect |
|---|---|
| `SMM_DATA_PATH` | Use a local CSV instead of downloading the dataset |
| `SMM_FAST=1` | Reduce folds, iterations, and bootstrap draws for smoke runs |
| `SMM_TRANSFORMER` | `auto` runs only when a GPU is available; `1` attempts to force execution; `0` skips it |

## Using the trained model

The notebook and training workflow save a serialized model bundle under `models/`. If the CLI is installed, a prediction can be requested with:

```bash
smm-predict "I can't stop worrying; my heart keeps racing at night."
```

The Python interface is:

```python
from socialmediamind.inference import SocialMediaMindModel

model = SocialMediaMindModel.load("models/socialmediamind_bundle.joblib")
predictions = model.predict([
    "Had a great time hiking with friends this weekend!"
])
print(predictions)
```

The current abstention rule can flag non-singleton conformal prediction sets for human review. This is an application-level uncertainty signal, **not** a guarantee that singleton predictions are correct or that crisis content will be detected safely. Do not use the model as a clinical or crisis-response system.

## Data

The project uses [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health), a public dataset of social-media posts with seven source-provided labels: **Normal, Depression, Suicidal, Anxiety, Stress, Bipolar**, and **Personality disorder**. These are dataset labels, not clinically verified diagnoses. Follow the dataset's license and terms. Raw data should not be committed to the repository, and public notebook outputs should avoid exposing raw posts (`SHOW_TEXT_EXAMPLES = False`).

## Limitations

- **Not clinically validated.** Labels are noisy source-dataset categories and must not be interpreted as diagnoses or verified mental-health states.
- **Potential artifacts.** The model may learn text length, writing style, usernames, URLs, or other dataset-specific patterns rather than robust semantic signals.
- **Uneven class performance.** Aggregate scores obscure class-level differences. Depression/Suicidal confusion remains a prominent source of errors.
- **Fairness unassessed.** Demographic attributes were unavailable or not evaluated, so fairness across demographic groups cannot be established.
- **Limited external validity.** Results are from one dataset. Generalization to other platforms, languages, populations, or time periods has not been demonstrated.
- **Conformal assumptions.** Empirical coverage is not an individual-level guarantee. Distribution shift can invalidate the assumptions underlying standard split-conformal coverage.
- **No transformer result in this run.** The optional transformer experiment was skipped when a CUDA GPU was not detected; no comparison against the linear model is claimed.
- **Privacy and safety.** Avoid publishing identifiable or sensitive post examples. False negatives are possible, and this model must not be the sole safeguard for crisis-related content.

## Roadmap

- [x] Leakage-aware evaluation, model ladder, calibration, and conformal sets (LAC and APS; marginal and Mondrian)
- [x] Confident-learning label-issue estimation and pruning sensitivity experiment
- [x] Model bundle, model card, and headline metrics export
- [x] Notebook workflow and repository test/CI infrastructure
- [ ] Run and evaluate the transformer baseline on a suitable GPU runtime
- [ ] Evaluate cross-source generalization if reliable per-post source metadata can be obtained
- [ ] Investigate conformal risk control for a pre-defined false-negative risk target on the Suicidal label
- [ ] Evaluate temporal/cross-platform shift and methods designed for distribution shift

## References

- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.*
- Romano, Y., Sesia, M., & Candès, E. (2020). *Classification with Valid and Adaptive Coverage.* NeurIPS.
- Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels.* Journal of Artificial Intelligence Research.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Al-Mosaiwi, M., & Johnstone, T. (2018). *In an Absolute State: Elevated Use of Absolutist Words Is a Marker Specific to Anxiety, Depression, and Suicidal Ideation.* Clinical Psychological Science.
- Rude, S., Gortner, E.-M., & Pennebaker, J. (2004). *Language use of depressed and depression-vulnerable college students.* Cognition & Emotion.
- Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006). *Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys?*
- Mitchell, M., et al. (2019). *Model Cards for Model Reporting.* FAT*.