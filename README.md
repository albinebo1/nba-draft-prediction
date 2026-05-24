# NBA Draft Bust/Star Prediction

A multiclass classification project: given **pre-draft** information (draft
position, NBA Combine measurements, position), predict whether a player will
become a **Star**, **Solid** contributor, or **Bust** over their NBA career.

Everything is reproducible with `random_state=42` baked into every module.

## Project layout

```
nba_draft_project/
├── data/
│   ├── raw/                       # source CSVs (3 Kaggle datasets)
│   └── processed/
│       └── nba_draft_processed.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_prep.py               # load, merge, label
│   ├── features.py                # ColumnTransformer + split helper
│   ├── models.py                  # Dummy / RF / XGBoost / SVM factories
│   ├── eda.py                     # all EDA plots (module form)
│   ├── evaluate.py                # CV, GridSearchCV, metrics, plots
│   └── report.py                  # auto-generates outputs/report_notes.md
├── outputs/
│   ├── figures/                   # all PNGs (EDA + per-model)
│   ├── results/                   # results_summary.csv + misclassifications
│   └── report_notes.md            # auto-generated report stub
├── requirements.txt
├── README.md
└── main.py                        # run the whole pipeline
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Data

Three Kaggle datasets, downloaded once into `data/raw/`:

| Dataset | What it gives us |
|---|---|
| `benwieland/nba-draft-data` | 1990-2021 draft + career WS/BPM/VORP (used to backfill missing career stats) |
| `mattop/nba-draft-basketball-player-data-19892021` | 1989-2021 draft + career stats (the primary draft table) |
| `marcusfern/nba-draft-combine` | 2000-2026 combine measurements (HGT, WGT, WNGSPN, STNDVERT, etc.) |

The combine CSV uses `Last, First` for player names; we flip and fuzzy-match
(`rapidfuzz.WRatio ≥ 88`) against the draft table.

## Target label

```
Star  : career WS ≥ 50
Solid : 15 ≤ career WS < 50
Bust  : career WS < 15  OR  career G < 82
```

Players drafted after 2018 are excluded (insufficient career data); players
who never played in the NBA are excluded entirely.

## Features (pre-draft only — no leakage)

* `draft_pick`, `draft_round`
* `position` (from combine; one-hot encoded; missing → `"Unknown"`)
* Combine measurements: `height_no_shoes`, `weight`, `wingspan`,
  `standing_reach`, `hand_length`, `hand_width`, `standing_vertical`,
  `vertical_leap_max`, `lane_agility`, `shuttle_run`, `sprint`, `bench_press`

Missing combine values get **median imputation fit on training folds only**.

> Last-season college box-score stats would be a useful additional signal
> but are not present in any of the three source CSVs, so they're left out
> of the baseline run. A hook is in place: drop a `data/raw/college_stats.csv`
> with the right schema and the pipeline will pick it up automatically.

## Run it

End-to-end:

```bash
python main.py
```

This executes (in order):

1. `data_prep.build_dataset()` — produces `data/processed/nba_draft_processed.csv`
2. `eda.run_eda()` — writes 5 EDA PNGs to `outputs/figures/`
3. `evaluate.run_full_evaluation()` — trains all 4 models with stratified
   5-fold CV + GridSearchCV, writes confusion matrices, learning curve,
   feature importances, `results_summary.csv`, and misclassification table.
4. `report.write_report_notes()` — writes `outputs/report_notes.md`.

Run individual stages from the project root:

```bash
python src/data_prep.py
python src/eda.py
python src/evaluate.py
```

## Models

| Model | Tuned over |
|---|---|
| **Baseline** `DummyClassifier` | `strategy ∈ {most_frequent, stratified}` |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` |
| **XGBoost** | `n_estimators`, `learning_rate`, `max_depth` |
| **SVM (RBF)** | `C`, `gamma` (with `StandardScaler`) |

All wrapped in an `sklearn.pipeline.Pipeline` so imputation/scaling fits on
each CV fold independently. Selection metric: **macro F1**.

## What you get

* **Figures** (`outputs/figures/`):
  `eda_*.png`, `confusion_*.png`, `feature_importance_*.png`,
  `learning_curve_<best>.png`
* **Numbers** (`outputs/results/`):
  `results_summary.csv` (all models × all metrics),
  `misclassifications_<best>.csv` (top 15 confidently wrong predictions —
  great for the report)
* **Project notes** (`outputs/project_notes.md`):
  dataset summary, preprocessing decisions, hyper-parameter grids, final
  results, feature importances, error analysis.

## Notes & caveats

* The combine dataset starts in 2000, so 1990-1999 picks have **all combine
  features missing** (median-imputed). This is structural, not a bug.
* Only ~64% of 1990-2018 draft picks match a combine row even with fuzzy
  matching — historically many prospects skipped the combine.
* The Star class is small (~12% of rows). Recall on Star is the headline
  weakness of every model. `class_weight='balanced'` is used where the
  estimator supports it.

## Data attribution

* `benwieland/nba-draft-data` — Kaggle
* `mattop/nba-draft-basketball-player-data-19892021` — Kaggle
* `marcusfern/nba-draft-combine` — Kaggle

All three are derived from Basketball-Reference / NBA.com.
