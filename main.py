"""
End-to-end pipeline for the NBA Draft Bust/Star prediction project.

Running ``python main.py`` will:
  1. Build the processed dataset (load Kaggle CSVs → merge → label → save).
  2. Run EDA plots (class distribution, scatter, boxplots, correlation,
     missingness).
  3. Train every model with stratified 5-fold CV + GridSearchCV.
  4. Score on the held-out test set.
  5. Save all figures and CSVs under ``outputs/``.
  6. Write a Markdown report stub at ``outputs/report_notes.md``.
  7. Print a final summary table.

Everything is keyed off ``RANDOM_STATE=42`` for reproducibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_prep import build_dataset  # noqa: E402
from eda import run_eda  # noqa: E402
from evaluate import run_full_evaluation  # noqa: E402
from report import write_report_notes  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("STAGE 1 — Build processed dataset")
    print("=" * 70)
    df = build_dataset()

    print("\n" + "=" * 70)
    print("STAGE 2 — Exploratory Data Analysis (saving figures)")
    print("=" * 70)
    run_eda(df)

    print("\n" + "=" * 70)
    print("STAGE 3 — Train + evaluate all models")
    print("=" * 70)
    results, summary, miscls = run_full_evaluation()

    print("\n" + "=" * 70)
    print("STAGE 4 — Write project notes")
    print("=" * 70)
    report_path = write_report_notes(df, results, summary, miscls)
    print(f"[save] {report_path}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    cols = [
        "model",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    print(summary[cols].to_string(index=False))

    print("\nDone. See outputs/figures and outputs/results for artifacts.")


if __name__ == "__main__":
    main()
