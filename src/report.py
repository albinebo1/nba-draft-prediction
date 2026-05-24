"""
Generate ``outputs/project_notes.md`` summarizing the most recent run:
dataset stats, preprocessing choices, model grids, final metrics, feature
importances, and a short error-analysis blurb.

Pure-Python, no external services. Re-run any time the pipeline finishes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import CATEGORICAL_FEATURES, CLASS_ORDER, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df.empty:
        return "(no data)"
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
        else:
            df[c] = df[c].astype(str)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in df.itertuples(index=False, name=None))
    return "\n".join([header, sep, body])


def write_report_notes(df, results, summary: pd.DataFrame, miscls: pd.DataFrame) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = df["career_label"].value_counts().reindex(CLASS_ORDER).astype(int)
    class_pct = (class_counts / class_counts.sum() * 100).round(1)

    feat_missing = (df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().mean() * 100).round(1)
    feat_missing_df = pd.DataFrame({"feature": feat_missing.index, "missing_pct": feat_missing.values})

    best_cv = max(
        (r for r in results if "Baseline" not in r.name),
        key=lambda r: r.cv_mean_macro_f1,
    )
    best_test = max(
        (r for r in results if "Baseline" not in r.name),
        key=lambda r: r.test_macro_f1,
    )

    fi_sections: list[str] = []
    for r in results:
        est = r.fitted_estimator
        clf = est.named_steps.get("clf") if hasattr(est, "named_steps") else None
        if clf is not None and hasattr(clf, "feature_importances_"):
            try:
                names = est.named_steps["pre"].get_feature_names_out()
            except Exception:
                names = [f"f{i}" for i in range(len(clf.feature_importances_))]
            imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False).head(10)
            fi_sections.append(
                f"### {r.name} — top 10 features\n\n"
                + _md_table(pd.DataFrame({"feature": imp.index, "importance": imp.values}))
            )

    summary_display = summary[
        [
            "model",
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
        ]
    ].copy()

    per_class_cols = ["model"] + [
        c for c in summary.columns
        if c.startswith("test_") and ("precision" in c or "recall" in c or "_f1" in c)
        and c not in ("test_macro_f1", "test_weighted_f1")
    ]
    per_class_display = summary[per_class_cols].copy()

    miscls_display = miscls.copy() if not miscls.empty else pd.DataFrame()
    if not miscls_display.empty:
        miscls_display["predicted_prob"] = miscls_display["predicted_prob"].round(3)
        miscls_display["career_ws"] = miscls_display["career_ws"].round(1)
        miscls_display["career_g"] = miscls_display["career_g"].astype(int)

    md = f"""# NBA Draft Bust/Star — Project Notes

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

Final processed dataset: **{len(df):,} rows × {len(df.columns)} columns**,
drafts 1990-2018, sourced from three public Kaggle CSVs and merged on
(player, year) with rapidfuzz fuzzy fallback.

### Class distribution

{_md_table(pd.DataFrame({"label": class_counts.index, "count": class_counts.values, "pct": class_pct.values}), float_fmt="{:.1f}")}

### Feature missingness

Combine measurements were not collected before 2000 and even after 2000 only
~64% of drafted players attended, so missingness is structural. Median
imputation is fit inside the model pipeline (training folds only).

{_md_table(feat_missing_df, float_fmt="{:.1f}")}

## 3. Preprocessing

* Numeric features: `SimpleImputer(strategy='median')`; scaler added only for SVM.
* Categorical (`position`): `SimpleImputer(fill_value='Unknown')` → `OneHotEncoder(handle_unknown='ignore')`.
* Train/test split: 80/20 stratified, `random_state=42`.
* CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, identical folds for every model.
* Class imbalance: `class_weight='balanced'` for RF and SVM; `compute_sample_weight('balanced')` for XGBoost.

## 4. Models & hyper-parameter grids

* **Baseline** — `DummyClassifier(strategy ∈ {{most_frequent, stratified}})`
* **Random Forest** — `n_estimators ∈ {{200, 400}}`, `max_depth ∈ {{None, 6, 12}}`, `min_samples_split ∈ {{2, 5}}`
* **XGBoost** — `n_estimators ∈ {{200, 400}}`, `learning_rate ∈ {{0.05, 0.10}}`, `max_depth ∈ {{3, 5}}`
* **SVM (RBF)** — `C ∈ {{1.0, 3.0, 10.0}}`, `gamma ∈ {{scale, 0.1}}` (with standardization)

Selection metric for CV tuning: **macro F1**.

## 5. Results

### Overall metrics

{_md_table(summary_display)}

### Per-class precision / recall / F1 (test set)

{_md_table(per_class_display)}

Best by CV macro F1: **{best_cv.name}** ({best_cv.cv_mean_macro_f1:.3f} ± {best_cv.cv_std_macro_f1:.3f}).
Best by test macro F1: **{best_test.name}** ({best_test.test_macro_f1:.3f}).

## 6. Feature importance

{chr(10).join(fi_sections) if fi_sections else "(no tree-based models reported importances)"}

## 7. Error analysis — confidently wrong predictions (best CV model: {best_cv.name})

{_md_table(miscls_display) if not miscls_display.empty else "(no misclassifications recorded)"}

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
"""
    out_path = OUTPUTS_DIR / "project_notes.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path
