"""
Model definitions for the NBA Draft Bust/Star task.

Each ``build_*`` factory returns a tuple:
    (sklearn Pipeline, param_grid for GridSearchCV, scale_required: bool)

The Pipeline always wraps the project's ``make_preprocessor`` so cross-validation
fits imputation/scaling on each training fold (no leakage).

Models:
  - DummyClassifier (baseline)             — most-frequent / stratified
  - RandomForestClassifier                 — substantive model 1
  - GradientBoostingClassifier or XGBoost  — substantive model 2 (auto-detects xgboost)
  - SVC (RBF kernel, with scaling)         — bonus model 3
"""
from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from features import make_preprocessor

RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(estimator, scale: bool) -> Pipeline:
    """Wrap an estimator with the project preprocessor in a Pipeline."""
    return Pipeline(
        [
            ("pre", make_preprocessor(scale=scale)),
            ("clf", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def build_baseline() -> tuple[Pipeline, dict[str, list[Any]], bool]:
    """DummyClassifier baseline. Tune the strategy so we can compare both
    stratified and most_frequent variants against the substantive models."""
    pipe = _wrap(DummyClassifier(random_state=RANDOM_STATE), scale=False)
    grid = {"clf__strategy": ["most_frequent", "stratified"]}
    return pipe, grid, False


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def build_random_forest() -> tuple[Pipeline, dict[str, list[Any]], bool]:
    pipe = _wrap(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",  # help with class imbalance (~60% Bust)
            n_jobs=-1,
        ),
        scale=False,
    )
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 6, 12],
        "clf__min_samples_split": [2, 5],
    }
    return pipe, grid, False


# ---------------------------------------------------------------------------
# Gradient Boosting (XGBoost if available)
# ---------------------------------------------------------------------------

def build_boosting() -> tuple[Pipeline, dict[str, list[Any]], bool]:
    if HAS_XGBOOST:
        # XGBoost needs integer labels, but our pipeline ships strings. We
        # handle this in evaluate.py by encoding before fit.
        clf = XGBClassifier(
            random_state=RANDOM_STATE,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        grid = {
            "clf__n_estimators": [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
        }
    else:  # pragma: no cover — fallback path
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
        grid = {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
        }
    pipe = _wrap(clf, scale=False)
    return pipe, grid, False


# ---------------------------------------------------------------------------
# SVM (RBF)
# ---------------------------------------------------------------------------

def build_svm() -> tuple[Pipeline, dict[str, list[Any]], bool]:
    pipe = _wrap(
        SVC(
            kernel="rbf",
            probability=True,  # needed for misclassification analysis
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        scale=True,
    )
    grid = {
        "clf__C": [1.0, 3.0, 10.0],
        "clf__gamma": ["scale", 0.1],
    }
    return pipe, grid, True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "Baseline (Dummy)":      build_baseline,
    "Random Forest":         build_random_forest,
    "XGBoost" if HAS_XGBOOST else "Gradient Boosting": build_boosting,
    "SVM (RBF)":             build_svm,
}


def all_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]], bool]]:
    """Return {name: (pipeline, param_grid, scale_required)} for every model."""
    return {name: factory() for name, factory in MODEL_REGISTRY.items()}


if __name__ == "__main__":
    for name, factory in MODEL_REGISTRY.items():
        pipe, grid, scale = factory()
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"{name:25s}  grid size = {n_combos:3d}  scale={scale}")
