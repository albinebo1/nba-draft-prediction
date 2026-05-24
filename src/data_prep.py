"""
Data acquisition and preparation for NBA Draft Bust/Star prediction.

Data sources (all real, in ``data/raw/``):
  * archive1/draft-data-20-years.csv          (Kaggle: benwieland/nba-draft-data)
  * archive2/nbaplayersdraft.csv              (Kaggle: mattop/nba-draft-basketball-player-data-19892021)
  * archive3/Draft Combine - Kaggle.csv       (Kaggle: marcusfern/nba-draft-combine)

Pipeline:
  1. Load mattop (the broader / cleaner draft table) as the base.
  2. Fill any career-stat gaps from benwieland (same schema, mostly overlapping).
  3. Merge with combine measurements on (player, year) - exact match first,
     fuzzy fallback via rapidfuzz for names that diverge slightly
     (suffixes like "Jr.", diacritics, etc.).
  4. Filter to drafts 1990-2018 (per spec: post-2018 excluded; pre-1990 dropped
     to keep the universe consistent with the original spec window).
  5. Build the Star/Solid/Bust label from career WS + games played.
  6. Drop players with zero NBA games (spec exclusion).
  7. Write ``data/processed/nba_draft_processed.csv``.

Pre-draft FEATURES (no leakage):
  draft_pick, draft_round, position, height_no_shoes, weight, wingspan,
  standing_reach, hand_length, hand_width, standing_vertical,
  vertical_leap_max, lane_agility, shuttle_run, sprint, bench_press.
TARGET: career_label in {Star, Solid, Bust}.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DRAFT_PRIMARY = RAW_DIR / "archive2" / "nbaplayersdraft.csv"
DRAFT_SECONDARY = RAW_DIR / "archive1" / "draft-data-20-years.csv"
COMBINE_CSV = RAW_DIR / "archive3" / "Draft Combine - Kaggle.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRAFT_YEAR_START = 1990
DRAFT_YEAR_END = 2018  # players drafted after this lack enough career data

LABEL_STAR_WS = 50
LABEL_SOLID_WS = 15
LABEL_MIN_G = 82

RANDOM_STATE = 42

# Rename combine columns to project-standard names.
COMBINE_RENAME = {
    "YEAR": "year",
    "PLAYER": "player_combine_raw",
    "POS": "position",
    "HGT": "height_no_shoes",
    "WGT": "weight",
    "WNGSPN": "wingspan",
    "STNDRCH": "standing_reach",
    "HANDL": "hand_length",
    "HANDW": "hand_width",
    "STNDVERT": "standing_vertical",
    "LPVERT": "vertical_leap_max",
    "LANE": "lane_agility",
    "SHUTTLE": "shuttle_run",
    "SPRINT": "sprint",
    "BENCH": "bench_press",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _flip_lastfirst(name: object) -> str:
    """Convert 'Last, First' (combine format) to 'First Last'."""
    if not isinstance(name, str) or "," not in name:
        return str(name).strip()
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def _normalize_name(name: object) -> str:
    """Lowercase, strip punctuation/suffixes for matching only."""
    s = str(name).lower().strip()
    s = re.sub(r"[.‘’']", "", s)  # remove dots and curly quotes
    s = re.sub(r"\s+(jr|sr|ii|iii|iv)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_draft_primary() -> pd.DataFrame:
    """Load the mattop draft table; rename to project conventions."""
    df = pd.read_csv(DRAFT_PRIMARY)
    df = df.rename(
        columns={
            "year": "draft_year",
            "overall_pick": "draft_pick",
            "team": "team",
            "player": "player",
            "college": "college",
            "years_active": "career_years",
            "games": "career_g",
            "win_shares": "career_ws",
            "win_shares_per_48_minutes": "career_ws48",
            "box_plus_minus": "career_bpm",
            "value_over_replacement": "career_vorp",
        }
    )
    keep = [
        "draft_year",
        "draft_pick",
        "team",
        "player",
        "college",
        "career_years",
        "career_g",
        "career_ws",
        "career_ws48",
        "career_bpm",
        "career_vorp",
    ]
    df = df[keep].copy()
    df["player"] = df["player"].astype(str).str.strip()
    df["college"] = df["college"].fillna("Unknown").astype(str).str.strip()
    return df


def load_draft_secondary() -> pd.DataFrame:
    """Load benwieland — used only to backfill missing career WS/G."""
    df = pd.read_csv(DRAFT_SECONDARY)
    df = df.rename(
        columns={
            "DraftYear": "draft_year",
            "Pk": "draft_pick",
            "Player": "player",
            "G": "career_g",
            "WS": "career_ws",
            "Yrs": "career_years",
        }
    )
    df = df[["draft_year", "draft_pick", "player", "career_g", "career_ws", "career_years"]]
    df["player"] = df["player"].astype(str).str.strip()
    return df


def load_combine() -> pd.DataFrame:
    """Load combine measurements, normalize player name to 'First Last'."""
    df = pd.read_csv(COMBINE_CSV)
    df = df.rename(columns=COMBINE_RENAME)
    df["player"] = df["player_combine_raw"].map(_flip_lastfirst)
    df = df.drop(columns=["player_combine_raw"])

    # Drop columns we don't use as features (BMI, BF, BAR, PAN, PBHGT, PDHGT
    # are derived/secondary measurements with heavy missingness).
    drop_cols = ["BMI", "BF", "BAR", "PAN", "PBHGT", "PDHGT"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_combine(draft: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Merge combine data into draft on (player, year). Exact match first,
    then a single per-year fuzzy pass for unmatched rows (score >= 88)."""
    from rapidfuzz import process, fuzz

    draft = draft.copy()
    combine = combine.copy()
    draft["_key"] = draft["player"].map(_normalize_name)
    combine["_key"] = combine["player"].map(_normalize_name)
    combine = combine.rename(columns={"year": "draft_year"})

    combine_feature_cols = [
        "position",
        "height_no_shoes",
        "weight",
        "wingspan",
        "standing_reach",
        "hand_length",
        "hand_width",
        "standing_vertical",
        "vertical_leap_max",
        "lane_agility",
        "shuttle_run",
        "sprint",
        "bench_press",
    ]

    merged = draft.merge(
        combine[["_key", "draft_year", *combine_feature_cols]],
        on=["_key", "draft_year"],
        how="left",
    )

    # Fuzzy fallback per year, only for rows missing a position (= unmatched).
    unmatched_idx = merged.index[merged["position"].isna()]
    fixed = 0
    for yr in merged.loc[unmatched_idx, "draft_year"].unique():
        pool = combine[combine["draft_year"] == yr]
        if pool.empty:
            continue
        pool_keys = pool["_key"].tolist()
        for idx in unmatched_idx:
            if merged.at[idx, "draft_year"] != yr:
                continue
            target = merged.at[idx, "_key"]
            hit = process.extractOne(target, pool_keys, scorer=fuzz.WRatio, score_cutoff=88)
            if hit is None:
                continue
            src = pool.iloc[hit[2]]
            for c in combine_feature_cols:
                merged.at[idx, c] = src[c]
            fixed += 1

    merged = merged.drop(columns=["_key"])
    print(f"[merge] combine: {(merged['position'].notna()).sum()}/{len(merged)} matched (exact + {fixed} fuzzy)")
    return merged


def backfill_career_from_secondary(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Fill missing career_ws / career_g in primary with secondary values."""
    sec = secondary.copy()
    sec["_key"] = sec["player"].map(_normalize_name)
    p = primary.copy()
    p["_key"] = p["player"].map(_normalize_name)

    merged = p.merge(
        sec[["_key", "draft_year", "career_ws", "career_g", "career_years"]].rename(
            columns={
                "career_ws": "career_ws_sec",
                "career_g": "career_g_sec",
                "career_years": "career_years_sec",
            }
        ),
        on=["_key", "draft_year"],
        how="left",
    )
    merged["career_ws"] = merged["career_ws"].combine_first(merged["career_ws_sec"])
    merged["career_g"] = merged["career_g"].combine_first(merged["career_g_sec"])
    merged["career_years"] = merged["career_years"].combine_first(merged["career_years_sec"])
    merged = merged.drop(columns=["career_ws_sec", "career_g_sec", "career_years_sec", "_key"])
    return merged


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["career_g"] = df["career_g"].fillna(0)
    df["career_ws"] = df["career_ws"].fillna(0)

    def _label(row) -> str:
        if row["career_g"] < LABEL_MIN_G or row["career_ws"] < LABEL_SOLID_WS:
            return "Bust"
        if row["career_ws"] >= LABEL_STAR_WS:
            return "Star"
        return "Solid"

    df["career_label"] = df.apply(_label, axis=1)
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["draft_round"] = np.where(df["draft_pick"] <= 30, 1, 2).astype(int)
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def maybe_merge_real_college(df: pd.DataFrame) -> pd.DataFrame:
    """Optional override: if ``data/raw/college_stats.csv`` exists, merge it
    onto the draft table by (player, draft_year). Expected schema:
    ``player,draft_year,college_ppg,college_rpg,college_apg,college_spg,
       college_bpg,college_fg_pct,college_three_pct,college_ft_pct,age_at_draft``

    Match strategy: exact normalized name + year, then rapidfuzz fallback
    at WRatio >= 88. Silently no-op if the file is absent.
    """
    csv_path = RAW_DIR / "college_stats.csv"
    if not csv_path.exists():
        return df

    print(f"[merge] real college stats from {csv_path}")
    real = pd.read_csv(csv_path)
    real["_key"] = real["player"].astype(str).map(_normalize_name)
    df = df.copy()
    df["_key"] = df["player"].map(_normalize_name)

    feature_cols = [c for c in real.columns if c not in ("player", "draft_year", "_key")]
    merged = df.merge(
        real[["_key", "draft_year", *feature_cols]],
        on=["_key", "draft_year"],
        how="left",
        suffixes=("", "_real"),
    )

    # Fuzzy fallback per year for unmatched rows
    try:
        from rapidfuzz import process, fuzz

        unmatched = merged[merged[feature_cols[0]].isna()].index
        for yr in merged.loc[unmatched, "draft_year"].unique():
            pool = real[real["draft_year"] == yr]
            if pool.empty:
                continue
            keys = pool["_key"].tolist()
            for idx in unmatched:
                if merged.at[idx, "draft_year"] != yr:
                    continue
                hit = process.extractOne(merged.at[idx, "_key"], keys, scorer=fuzz.WRatio, score_cutoff=88)
                if hit is None:
                    continue
                src = pool.iloc[hit[2]]
                for c in feature_cols:
                    merged.at[idx, c] = src[c]
    except ImportError:
        pass

    return merged.drop(columns=["_key"])


def build_dataset(verbose: bool = True) -> pd.DataFrame:
    """Load, merge, label, save. Returns the processed DataFrame."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[load] reading raw CSVs")
    primary = load_draft_primary()
    secondary = load_draft_secondary()
    combine = load_combine()

    if verbose:
        print(f"[load] primary draft: {primary.shape}, secondary: {secondary.shape}, combine: {combine.shape}")

    # Window: 1990-2018
    primary = primary[(primary["draft_year"] >= DRAFT_YEAR_START) & (primary["draft_year"] <= DRAFT_YEAR_END)].copy()
    if verbose:
        print(f"[filter] {len(primary)} picks in {DRAFT_YEAR_START}-{DRAFT_YEAR_END}")

    primary = backfill_career_from_secondary(primary, secondary)
    merged = merge_combine(primary, combine)
    merged = maybe_merge_real_college(merged)
    merged = add_derived(merged)
    labeled = assign_labels(merged)

    # Spec: exclude players who never played in the NBA (G == 0).
    before = len(labeled)
    labeled = labeled[labeled["career_g"] > 0].copy()
    if verbose:
        print(f"[exclude] dropped {before - len(labeled)} players with 0 NBA games")

    out_path = PROCESSED_DIR / "nba_draft_processed.csv"
    labeled.to_csv(out_path, index=False)
    if verbose:
        print(f"\n[done] wrote {len(labeled)} rows to {out_path}")
        print("[done] class distribution:")
        print(labeled["career_label"].value_counts())
        print(f"\n[done] columns ({len(labeled.columns)}):")
        print(list(labeled.columns))

        # Missingness summary for feature columns
        feature_cols = [
            "draft_pick", "draft_round", "position",
            "height_no_shoes", "weight", "wingspan", "standing_reach",
            "hand_length", "hand_width", "standing_vertical",
            "vertical_leap_max", "lane_agility", "shuttle_run",
            "sprint", "bench_press",
        ]
        missing = (labeled[feature_cols].isna().mean() * 100).round(1)
        print("\n[done] feature missingness (%):")
        print(missing.to_string())

    return labeled


if __name__ == "__main__":
    build_dataset()
