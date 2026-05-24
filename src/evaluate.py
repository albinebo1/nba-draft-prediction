"""
Evaluation utilities for the NBA Draft Bust/Star task.

What this module does:
  * 5-fold StratifiedKFold cross-validation with a fixed seed so every model
    sees the same folds.
  * GridSearchCV for hyper-parameter selection per model (scoring=macro F1).
  * Final hold-out test set evaluation (untouched until the end).
  * Per-model metrics: accuracy, macro F1, weighted F1, per-class precision /
    recall / F1.
  * Confusion matrices saved as PNG figures in ``outputs/figures``.
  * Learning curve for the best model.
  * Feature importance plots for tree models.
  * Misclassification analysis — top "confidently wrong" predictions.
  * Results summary CSV written to ``outputs/results/results_summary.csv``.

XGBoost needs integer labels, so a LabelEncoder is fit on the training set
and applied before any pipeline fit. We carry the original string labels in
parallel for human-readable reports.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend - we never show, only save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder

from features import CLASS_ORDER, FEATURE_COLUMNS, load_full, load_xy, train_test_split_stratified
from models import HAS_XGBOOST, all_models

RANDOM_STATE = 42
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Silence the noisy sklearn UndefinedMetric warnings for empty classes in a
# fold during CV. We handle the metrics explicitly with zero_division=0.
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    best_params: dict
    cv_mean_macro_f1: float
    cv_std_macro_f1: float
    cv_mean_acc: float
    cv_std_acc: float
    test_macro_f1: float
    test_weighted_f1: float
    test_accuracy: float
    test_per_class: pd.DataFrame
    fitted_estimator: object  # the refit best estimator on the full train set
    label_encoder: Optional[LabelEncoder]


def _needs_label_encoding(name: str) -> bool:
    """Only XGBoost needs integer labels in our setup."""
    return name == "XGBoost"


def _encode_if_needed(name: str, y) -> tuple[np.ndarray, Optional[LabelEncoder]]:
    if _needs_label_encoding(name):
        enc = LabelEncoder()
        enc.classes_ = np.array(CLASS_ORDER)  # fix order so 0=Bust, 1=Solid, 2=Star
        y_int = np.array([list(CLASS_ORDER).index(v) for v in y])
        return y_int, enc
    return np.asarray(y), None


def _decode(name: str, y_pred, enc: Optional[LabelEncoder]):
    if enc is None:
        return np.asarray(y_pred)
    return np.array([CLASS_ORDER[int(v)] for v in y_pred])


# ---------------------------------------------------------------------------
# Per-model train + evaluate
# ---------------------------------------------------------------------------

def grid_search_and_score(
    name: str,
    pipeline,
    grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    """Run GridSearchCV with stratified 5-fold CV, then score on the test set."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    y_train_enc, enc = _encode_if_needed(name, y_train)
    y_test_enc, _ = _encode_if_needed(name, y_test)

    print(f"\n[fit] {name}: GridSearchCV over {sum(1 for _ in _grid_iter(grid))} combinations × {N_SPLITS} folds")
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    # XGBoost has no `class_weight` argument; we replicate the 'balanced' behavior
    # via per-sample weights so it competes on equal footing with RF and SVM.
    fit_params: dict = {}
    if name == "XGBoost":
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_enc)
        fit_params["clf__sample_weight"] = sample_weight
    gs.fit(X_train, y_train_enc, **fit_params)
    best = gs.best_estimator_

    # CV stats on the winning param combo
    cv_df = pd.DataFrame(gs.cv_results_)
    best_idx = gs.best_index_
    cv_mean_f1 = float(cv_df.loc[best_idx, "mean_test_score"])
    cv_std_f1 = float(cv_df.loc[best_idx, "std_test_score"])

    # Re-run a separate CV pass for accuracy to report alongside f1
    from sklearn.model_selection import cross_val_score

    acc_scores = cross_val_score(best, X_train, y_train_enc, cv=cv, scoring="accuracy", n_jobs=-1)

    # Test set evaluation
    y_pred_enc = best.predict(X_test)
    y_pred = _decode(name, y_pred_enc, enc)
    y_true = np.asarray(y_test)

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            cls: [report[cls]["precision"], report[cls]["recall"], report[cls]["f1-score"], report[cls]["support"]]
            for cls in CLASS_ORDER
        },
        index=["precision", "recall", "f1", "support"],
    ).T

    return ModelResult(
        name=name,
        best_params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        cv_mean_macro_f1=cv_mean_f1,
        cv_std_macro_f1=cv_std_f1,
        cv_mean_acc=float(acc_scores.mean()),
        cv_std_acc=float(acc_scores.std()),
        test_macro_f1=test_macro_f1,
        test_weighted_f1=test_weighted_f1,
        test_accuracy=test_acc,
        test_per_class=per_class,
        fitted_estimator=best,
        label_encoder=enc,
    )


def _grid_iter(grid: dict):
    if not grid:
        yield {}
        return
    keys = list(grid)
    from itertools import product

    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(result: ModelResult, X_test, y_test) -> Path:
    """Save a confusion matrix PNG for the given fitted model."""
    y_pred_enc = result.fitted_estimator.predict(X_test)
    y_pred = _decode(result.name, y_pred_enc, result.label_encoder)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[0],
    )
    axes[0].set_title(f"{result.name} — counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[1],
    )
    axes[1].set_title(f"{result.name} — row-normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    safe_name = result.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out = FIG_DIR / f"confusion_{safe_name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_learning_curve(result: ModelResult, X_train, y_train) -> Path:
    """Save a learning curve PNG for the given model."""
    y_train_enc, _ = _encode_if_needed(result.name, y_train)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    sizes = np.linspace(0.2, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        result.fitted_estimator,
        X_train,
        y_train_enc,
        train_sizes=sizes,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train", color="C0")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15, color="C0",
    )
    ax.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV", color="C1")
    ax.fill_between(
        train_sizes,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.15, color="C1",
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Learning curve — {result.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    safe = result.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out = FIG_DIR / f"learning_curve_{safe}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_feature_importance(result: ModelResult) -> Optional[Path]:
    """Save feature importance plot if the underlying estimator supports it."""
    est = result.fitted_estimator
    clf = est.named_steps["clf"] if hasattr(est, "named_steps") else est
    if not hasattr(clf, "feature_importances_"):
        return None

    # Get post-preprocessing feature names
    pre = est.named_steps["pre"]
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(len(clf.feature_importances_))]

    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
    top = imp.tail(15)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    top.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance — {result.name} (top 15)")
    plt.tight_layout()
    safe = result.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out = FIG_DIR / f"feature_importance_{safe}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Misclassification analysis
# ---------------------------------------------------------------------------

def misclassification_report(
    result: ModelResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_idx: np.ndarray,
    full_df: pd.DataFrame,
    top_k: int = 15,
) -> pd.DataFrame:
    """Return the top-K most confidently wrong predictions on the test set."""
    est = result.fitted_estimator
    if not hasattr(est, "predict_proba"):
        return pd.DataFrame()

    proba = est.predict_proba(X_test)
    y_pred_enc = est.predict(X_test)
    y_pred = _decode(result.name, y_pred_enc, result.label_encoder)
    y_true = np.asarray(y_test)

    # Class order from the underlying estimator (XGBoost uses 0/1/2)
    if _needs_label_encoding(result.name):
        proba_classes = [CLASS_ORDER[int(c)] for c in est.named_steps["clf"].classes_]
    else:
        proba_classes = list(est.named_steps["clf"].classes_)

    rows = []
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        if pred == true:
            continue
        pred_prob = proba[i, proba_classes.index(pred)]
        original_row = full_df.iloc[test_idx[i]]
        rows.append(
            {
                "player": original_row["player"],
                "draft_year": int(original_row["draft_year"]),
                "draft_pick": int(original_row["draft_pick"]),
                "actual": true,
                "predicted": pred,
                "predicted_prob": float(pred_prob),
                "career_ws": float(original_row["career_ws"]),
                "career_g": float(original_row["career_g"]),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("predicted_prob", ascending=False).head(top_k).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_evaluation() -> tuple[list[ModelResult], pd.DataFrame, pd.DataFrame]:
    """Train + evaluate every model in the registry. Returns the per-model
    results, the summary DataFrame, and the misclassification table for the
    best model."""
    full_df = load_full()
    X, y = load_xy()

    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    test_idx = X_test.index.to_numpy()

    print(f"[split] train={len(X_train)}  test={len(X_test)}")
    print(f"[split] train class dist:\n{y_train.value_counts()}")
    print(f"[split] test class dist:\n{y_test.value_counts()}")

    results: list[ModelResult] = []
    for name, (pipeline, grid, _scale) in all_models().items():
        result = grid_search_and_score(name, pipeline, grid, X_train, y_train, X_test, y_test)
        results.append(result)
        save_confusion_matrix(result, X_test, y_test)
        save_feature_importance(result)
        print(f"  ✓ {name}: CV macro F1 = {result.cv_mean_macro_f1:.3f} ± {result.cv_std_macro_f1:.3f} | test macro F1 = {result.test_macro_f1:.3f}")

    # Summary table
    summary_rows = []
    for r in results:
        row = {
            "model": r.name,
            "best_params": str(r.best_params),
            "cv_macro_f1_mean": round(r.cv_mean_macro_f1, 4),
            "cv_macro_f1_std": round(r.cv_std_macro_f1, 4),
            "cv_acc_mean": round(r.cv_mean_acc, 4),
            "cv_acc_std": round(r.cv_std_acc, 4),
            "test_accuracy": round(r.test_accuracy, 4),
            "test_macro_f1": round(r.test_macro_f1, 4),
            "test_weighted_f1": round(r.test_weighted_f1, 4),
        }
        # per-class metrics
        for cls in CLASS_ORDER:
            row[f"test_{cls}_precision"] = round(r.test_per_class.at[cls, "precision"], 4)
            row[f"test_{cls}_recall"] = round(r.test_per_class.at[cls, "recall"], 4)
            row[f"test_{cls}_f1"] = round(r.test_per_class.at[cls, "f1"], 4)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "results_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[save] {summary_path}")

    # Pick the model that did best in cross-validation (not on the test set —
    # that would be a soft form of test-set tuning) for the diagnostic plots.
    best = max(
        (r for r in results if "Baseline" not in r.name),
        key=lambda r: r.cv_mean_macro_f1,
    )
    print(f"\n[best] {best.name} (CV macro F1 = {best.cv_mean_macro_f1:.3f}, test macro F1 = {best.test_macro_f1:.3f})")
    save_learning_curve(best, X_train, y_train)

    miscls = misclassification_report(best, X_test, y_test, test_idx, full_df, top_k=15)
    if not miscls.empty:
        safe = best.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        miscls_path = RESULTS_DIR / f"misclassifications_{safe}.csv"
        miscls.to_csv(miscls_path, index=False)
        print(f"[save] {miscls_path}")

    return results, summary, miscls


if __name__ == "__main__":
    run_full_evaluation()
