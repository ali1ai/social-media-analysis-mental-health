# Audit of `SocialMediaMind_Master_v01.ipynb` and what changed in v0.2

This audit is kept in the repo on purpose. Finding and fixing your own methodological errors is a core ML skill, and this record shows the reasoning behind every design choice in v0.2.

## Critical issues (results were invalid)

| # | Issue | Evidence in v01 | Consequence | Fix in v0.2 |
|---|---|---|---|---|
| 1 | **The target was the CSV row number.** | The code loads `suchintikasarkar/sentiment-analysis-for-mental-health` (columns `Unnamed: 0, statement, status`). The automatic target builder then picked the only numeric column, `unnamed_0`, z-scored it, and named it `mental_health_strain_index`. | The "strain index" was uniformly distributed from −1.73 to 1.73: a rescaled row index. Every downstream analysis measured nothing about mental health. | The index column is dropped in `standardise_schema`. The real label (`status`, 7 classes) is the target, and a unit test asserts the index can never survive. |
| 2 | **Label leakage.** | `status` (the true label) and raw `statement` stayed in `X` as one-hot "categorical" features (`Categorical: 2`). | The data is sorted by label, so `status` predicts row position. Random Forest R² = 0.70 was the model recovering file order. | Model inputs are text only. Labels never enter feature matrices. |
| 3 | **Dataset / narrative mismatch.** | Markdown, research questions and §6, §7, §14, §15 describe a *survey* (usage hours, platforms, demographics). None of those columns exist. | Platform tests, usage features, subgroup and ablation sections silently produced empty tables while the narrative claimed results. | The project is reframed around the data actually used: text classification. Research questions are rewritten so every one is answerable. |
| 4 | **Not reproducible end to end.** | `NameError: name 'Pipeline' is not defined` at the tuning cell (stale kernel). Several outputs don't match their code (e.g. cell 4B prints text the code no longer contains). | Nobody can rerun it to get the shown results. | Executed top-to-bottom in CI (`nbconvert --execute`) on a synthetic fixture. Logic lives in a tested `src/` package. |
| 5 | **Duplicates not handled before splitting.** | v01's own audit table shows 51,073 unique statements among 52,681 non-missing rows: about 1.6k exact duplicate rows (more after case and punctuation normalisation), some with conflicting labels. | Duplicates across train and test inflate metrics through memorisation. | De-duplication and conflict removal happen before splitting, plus an automated overlap assertion and a quantified leakage demo (§4.1). |

## Methodological issues

* **The "explicit, auditable target" claim was contradicted by the code.** §4 promised to stop rather than guess, and §5 then guessed automatically from keyword lists. v0.2 has no guessing: the schema is validated and the code fails loudly.
* **Cronbach's α on a one-item scale**, reported in a reliability table. This is meaningless and was removed.
* **"Behavioural" clustering on punctuation counts.** Clustering post length and comma counts is not behavioural segmentation, and k = 2 simply split long posts from short ones. It is replaced by LSA topic clustering with stability analysis and post-hoc comparison to the labels.
* **Regression metrics on what is really a classification problem.** The task is now framed as 7-class classification, with **macro-F1** as the primary metric because of the ~15× class imbalance.
* **Conformal interval was a single global radius** for regression. It is replaced by conformal *prediction sets* for classification, with **class-conditional (Mondrian)** coverage so rare classes are protected.
* **Calibration was not assessed.** Temperature scaling, reliability diagrams, ECE and Brier score are added.
* **Statistics.** Welch t-test and Cohen's d on skewed count data are replaced by Kruskal–Wallis with ε², Cliff's δ and BH-FDR, with the emphasis on effect sizes given large n.
* **Responsible-AI section claimed subgroup analysis** without any demographic data. v0.2 states plainly that fairness is unassessable, and uses length slices as a robustness check instead.
* **Sensitive content in outputs.** v01 printed raw mental-health posts into the saved notebook. v0.2 hides text by default (`SHOW_TEXT_EXAMPLES = False`).

## Engineering issues

* Monolithic 400-line cells, duplicated imports, and `adjusted_rand_score` imported mid-notebook. These are split into small cells backed by a `src/` package.
* No tests. There are now 25 tests: unit tests, statistical tests of the conformal coverage guarantee, a confident-learning recovery test, a transformer training-loop test, an end-to-end train → save → load → CLI test, and notebook-sync tests.
* Under-specified solver choice. In benchmarking, `saga` failed to converge after 3000 epochs (228 s) while `lbfgs` converged in 35 iterations (2.4 s). The `lbfgs` choice is documented in code.
* No model card, environment capture or versioned artefacts. All three are now generated automatically.

## Beyond fixing v01: additions in v0.2

* **Randomised APS conformal sets** alongside LAC. Implementing deterministic APS first produced bloated sets (about 4.8 labels on average) with a confident model. Switching to the randomised score from Romano et al. (2020) brought that to about 1.3, with the best worst-class coverage and clearly larger sets on wrong predictions. A regression test pins this down.
* **Confident learning** (implemented from the paper) to estimate the label-noise matrix, with a pruning sensitivity check.
* **Optional DistilRoBERTa baseline** sent through the same calibration and conformal layer, compared by paired bootstrap.
* **Self-contained notebook.** An earlier v0.2 draft cloned the repo from GitHub in its first cell, which failed (and prompted for a password) before the repo existed. The notebook now embeds a checksum-verified copy of `src/` and runs in Colab with no login. A test keeps that copy in sync, and CI runs the notebook outside the repo to prove it.
* **Packaged inference** (`SocialMediaMindModel`, `smm-predict`), **reproducible training** (`smm-train`), Makefile, and multi-version CI that executes the notebook.
