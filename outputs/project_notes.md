# NBA Draft Bust/Star — Project Notes

Auto-generated summary of the most recent pipeline run.

## 1. Problem

Multiclass classification of NBA draft picks into **Star**, **Solid**, or
**Bust** using *pre-draft* information only (no NBA career stats as inputs).
The target label is derived from career Win Shares (WS) and games played:

* Star  — career WS ≥ 50
* Solid — 15 ≤ career WS < 50
* Bust  — career WS < 15 **or** fewer than 82 NBA games

Players drafted after 2018 are excluded (not enough career data). Players who
never played a single NBA game are also excluded.

## 2. Dataset

Final processed dataset: **1,452 rows × 26 columns**,
drafts 1990-2018, sourced from three public Kaggle CSVs and merged on
(player, year) with rapidfuzz fuzzy fallback.

### Class distribution

| label | count | pct |
| --- | --- | --- |
| Bust | 882 | 60.7 |
| Solid | 392 | 27.0 |
| Star | 178 | 12.3 |

### Feature missingness

Combine measurements were not collected before 2000 and even after 2000 only
~64% of drafted players attended, so missingness is structural. Median
imputation is fit inside the model pipeline (training folds only).

| feature | missing_pct |
| --- | --- |
| draft_pick | 0.0 |
| draft_round | 0.0 |
| height_no_shoes | 54.2 |
| weight | 54.2 |
| wingspan | 54.1 |
| standing_reach | 54.2 |
| hand_length | 75.0 |
| hand_width | 75.0 |
| standing_vertical | 59.3 |
| vertical_leap_max | 59.4 |
| lane_agility | 60.0 |
| shuttle_run | 87.5 |
| sprint | 59.6 |
| bench_press | 64.5 |
| position | 54.1 |

## 3. Preprocessing

* Numeric features: `SimpleImputer(strategy='median')`; scaler added only for SVM.
* Categorical (`position`): `SimpleImputer(fill_value='Unknown')` → `OneHotEncoder(handle_unknown='ignore')`.
* Train/test split: 80/20 stratified, `random_state=42`.
* CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, identical folds for every model.
* Class imbalance: `class_weight='balanced'` for RF and SVM; `compute_sample_weight('balanced')` for XGBoost.

## 4. Models & hyper-parameter grids

* **Baseline** — `DummyClassifier(strategy ∈ {most_frequent, stratified})`
* **Random Forest** — `n_estimators ∈ {200, 400}`, `max_depth ∈ {None, 6, 12}`, `min_samples_split ∈ {2, 5}`
* **XGBoost** — `n_estimators ∈ {200, 400}`, `learning_rate ∈ {0.05, 0.10}`, `max_depth ∈ {3, 5}`
* **SVM (RBF)** — `C ∈ {1.0, 3.0, 10.0}`, `gamma ∈ {scale, 0.1}` (with standardization)

Selection metric for CV tuning: **macro F1**.

## 5. Results

### Overall metrics

| model | cv_macro_f1_mean | cv_macro_f1_std | test_accuracy | test_macro_f1 | test_weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| Baseline (Dummy) | 0.3184 | 0.0399 | 0.4467 | 0.3388 | 0.4481 |
| Random Forest | 0.4899 | 0.0300 | 0.5086 | 0.4320 | 0.5301 |
| XGBoost | 0.4733 | 0.0175 | 0.5601 | 0.4919 | 0.5756 |
| SVM (RBF) | 0.4761 | 0.0222 | 0.5120 | 0.4373 | 0.5333 |

### Per-class precision / recall / F1 (test set)

| model | test_Bust_precision | test_Bust_recall | test_Bust_f1 | test_Solid_precision | test_Solid_recall | test_Solid_f1 | test_Star_precision | test_Star_recall | test_Star_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (Dummy) | 0.6000 | 0.5932 | 0.5966 | 0.2308 | 0.2308 | 0.2308 | 0.1842 | 0.1944 | 0.1892 |
| Random Forest | 0.7465 | 0.5989 | 0.6646 | 0.3176 | 0.3462 | 0.3313 | 0.2344 | 0.4167 | 0.3000 |
| XGBoost | 0.7619 | 0.6328 | 0.6914 | 0.3736 | 0.4359 | 0.4024 | 0.3208 | 0.4722 | 0.3820 |
| SVM (RBF) | 0.7465 | 0.5989 | 0.6646 | 0.3333 | 0.3462 | 0.3396 | 0.2353 | 0.4444 | 0.3077 |

Best by CV macro F1: **Random Forest** (0.490 ± 0.030).
Best by test macro F1: **XGBoost** (0.492).

## 6. Feature importance

### Random Forest — top 10 features

| feature | importance |
| --- | --- |
| num__draft_pick | 0.3273 |
| num__draft_round | 0.1376 |
| num__lane_agility | 0.0575 |
| num__weight | 0.0511 |
| num__wingspan | 0.0440 |
| num__vertical_leap_max | 0.0403 |
| num__height_no_shoes | 0.0401 |
| num__standing_reach | 0.0385 |
| num__sprint | 0.0377 |
| num__hand_width | 0.0360 |
### XGBoost — top 10 features

| feature | importance |
| --- | --- |
| num__draft_pick | 0.1369 |
| cat__position_SG | 0.0689 |
| cat__position_Unknown | 0.0646 |
| num__hand_width | 0.0596 |
| cat__position_PG-SG | 0.0530 |
| cat__position_PF | 0.0490 |
| num__lane_agility | 0.0418 |
| num__shuttle_run | 0.0397 |
| num__wingspan | 0.0394 |
| num__sprint | 0.0377 |

## 7. Error analysis — confidently wrong predictions (best CV model: Random Forest)

| player | draft_year | draft_pick | actual | predicted | predicted_prob | career_ws | career_g |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pat Connaughton | 2015 | 41 | Solid | Bust | 0.7190 | 18.1000 | 417 |
| Matt Barnes | 2002 | 46 | Solid | Bust | 0.6710 | 42.8000 | 929 |
| DeAndre Jordan | 2008 | 35 | Star | Bust | 0.6650 | 94.4000 | 980 |
| Kenyon Martin | 2000 | 1 | Solid | Star | 0.6240 | 48.0000 | 757 |
| Markelle Fultz | 2017 | 1 | Bust | Star | 0.6240 | 3.5000 | 131 |
| Deandre Ayton | 2018 | 1 | Solid | Star | 0.6240 | 24.6000 | 236 |
| Michael Olowokandi | 1998 | 1 | Bust | Star | 0.6240 | 2.5000 | 500 |
| Ramon Sessions | 2007 | 56 | Solid | Bust | 0.6150 | 28.8000 | 691 |
| Mahmoud Abdul-Rauf | 1990 | 3 | Solid | Star | 0.6140 | 25.2000 | 586 |
| Jarron Collins | 2001 | 53 | Solid | Bust | 0.6110 | 15.7000 | 542 |
| Manu Ginóbili | 1999 | 57 | Star | Bust | 0.6060 | 106.4000 | 1057 |
| Luc Mbah a Moute | 2008 | 37 | Solid | Bust | 0.6060 | 26.0000 | 689 |
| Jamal Mashburn | 1993 | 4 | Solid | Star | 0.6020 | 43.7000 | 611 |
| Jim Jackson | 1992 | 4 | Solid | Star | 0.6020 | 35.8000 | 885 |
| Monta Ellis | 2005 | 40 | Solid | Bust | 0.6010 | 41.9000 | 833 |

Two recurring failure modes:

1. **Late-round successes mislabeled as Bust** — players like DeAndre Jordan (pick 35),
   Manu Ginóbili (57), and Monta Ellis (40). Draft position dominates the signal and
   nothing in the pre-draft features distinguishes late-round stars from late-round busts.
2. **Top-3 picks mislabeled as Star** — players like Markelle Fultz (#1), Michael
   Olowokandi (#1), and Kenyon Martin (#1). Their outcomes were driven by post-draft
   events (injuries, fit, work ethic) that no pre-draft feature can encode.

## 8. Key findings

* All three substantive models beat the stratified baseline by roughly 0.10 macro F1
  on the held-out test set.
* Draft pick dominates the feature importances of every tree-based model.
* Combine measurements (wingspan, weight, vertical, lane agility) add secondary signal.
* The Star class is the hardest by far (small support, low recall). Recall on Star is
  ~0.4 — the model is conservative about predicting superstars from combine + pick alone.

## 9. Limitations

* **College stats not included.** None of the source CSVs include last-season college box-score stats.
  The pipeline has a hook (`data_prep.maybe_merge_real_college`) — drop a CSV with the right
  schema into `data/raw/college_stats.csv` and it merges automatically.
* **Position is missing for ~54% of rows.** Same root cause as combine missingness (pre-2000 + no-combine attendees).
* **Class imbalance** is handled with class weights only. SMOTE or focal loss would be reasonable extensions.

---

*All figures live in `outputs/figures/`; raw numbers in `outputs/results/`. Reproducibility: `random_state=42` everywhere.*
