"""
diagnose_train_serve_gap.py
===========================
Explain why the pitcher stats look great on the training holdout (~1.3 MAE)
but worse on the live 2026 Season Accuracy page (~2.0 MAE).

It computes the SAME per-stat MAE three different ways so we can localise the
gap instead of guessing:

  1. HOLDOUT      — the trained model scored on the chronological test split of
                    pitcher_game_data.csv (this reproduces the number the
                    trainer printed, e.g. K ≈ 1.3).
  2. RAW-ON-2026  — the SAME trained model scored directly on 2026 completed
                    starts, using the clean as-of-game feature rows from
                    pitcher_game_data.csv (NO blend / floors / park /
                    calibration / context layers).
  3. LIVE-2026    — the fully post-processed projection that's actually graded
                    on the website, read straight from 2026_player_accuracy.csv
                    (proj_* vs actual_*).

How to read the result
----------------------
  • RAW-ON-2026 ≈ HOLDOUT, but LIVE much higher
        → the model is fine; the post-processing stack (60/40 league blend,
          K/IP floors, park, display calibration, weather/umpire/market) is
          adding the error. Fix the serving stack, not the model.

  • RAW-ON-2026 ≈ LIVE (both well above HOLDOUT)
        → the model itself is worse on real 2026 data than the holdout implied
          → feature skew (serving stale/as-of-last-start rows) or leakage in
          the holdout (season-to-date features peeking at the future).

Usage
-----
    python diagnose_train_serve_gap.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import hitterspitchers_train as hpt  # reuse chronological_split + constants

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODELS_DIR = HERE / "models"
ACC_CSV = HERE / "2026_player_accuracy.csv"
PITCHER_CSV = DATA_DIR / "pitcher_game_data.csv"

# (label, target column in pitcher_game_data, model pickle, accuracy proj col, accuracy actual col)
PITCHER_STATS = [
    ("Strikeouts",    "K",  "pitcher_K.pkl",  "proj_strikeouts",   "actual_strikeouts"),
    ("Walks",         "BB", "pitcher_BB.pkl", "proj_walks",        "actual_walks"),
    ("Hits Allowed",  "H",  "pitcher_H.pkl",  "proj_hits_allowed", "actual_hits_allowed"),
    ("Innings Pitched","IP","pitcher_IP.pkl", "proj_ip",           "actual_ip"),
]


def _load_bundle(fname: str):
    p = MODELS_DIR / fname
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _estimator(bundle):
    """The deployed bundles use either 'pipeline' (legacy trainer) or 'model'
    (two-stage trainer) for the fitted estimator."""
    return bundle.get("pipeline") or bundle.get("model")


def _model_mae(bundle, frame: pd.DataFrame, target_col: str):
    """MAE of the deployed model prediction vs the actual target column."""
    est = _estimator(bundle)
    feats = bundle["features"]
    work = frame.dropna(subset=[target_col]).copy()
    if work.empty or est is None:
        return np.nan, 0
    X = work.reindex(columns=feats)              # missing cols → NaN → imputer handles
    y = pd.to_numeric(work[target_col], errors="coerce").values
    pred = np.asarray(est.predict(X), dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() == 0:
        return np.nan, 0
    return float(mean_absolute_error(y[mask], pred[mask])), int(mask.sum())


def _fresh_oos_mae(train_df, test_df, target_col: str):
    """Train a FRESH model on the first-75% split and score the unseen last-25%.

    This is the honest out-of-sample number: unlike the deployed bundle (which
    may have been refit on all data, making the 'holdout' in-sample), this model
    never sees the test rows. If this lands near the LIVE number, the model's
    true skill ≈ live and the low deployed-holdout MAE was an in-sample artifact.
    """
    feats = hpt.select_features(train_df, hpt.PITCHER_FEATURES)
    tr = train_df.dropna(subset=[target_col])
    te = test_df.dropna(subset=[target_col])
    if tr.empty or te.empty or not feats:
        return np.nan
    pipe = hpt.build_sklearn_model("xgb", target_name=target_col)
    pipe.fit(tr[feats], pd.to_numeric(tr[target_col], errors="coerce"))
    pred = np.asarray(pipe.predict(te[feats]), dtype=float)
    y = pd.to_numeric(te[target_col], errors="coerce").values
    mask = np.isfinite(y) & np.isfinite(pred)
    return float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() else np.nan


def main() -> None:
    if not PITCHER_CSV.exists():
        print(f"⚠️  {PITCHER_CSV} not found"); return
    if not ACC_CSV.exists():
        print(f"⚠️  {ACC_CSV} not found"); return

    df = pd.read_csv(PITCHER_CSV, low_memory=False)
    date_col = "game_date" if "game_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Match the training filter: actual starters only.
    if "is_actual_starter" in df.columns:
        df = df[pd.to_numeric(df["is_actual_starter"], errors="coerce").fillna(0) == 1].copy()

    # 1) Recreate the trainer's chronological holdout split.
    train_df, test_df = hpt.chronological_split(df, date_col)   # default TRAIN_FRAC

    # 2) 2026-only completed starts (clean as-of-game features).
    df_2026 = df[df[date_col].dt.year == 2026].copy()

    # 3) Live graded pitcher rows.
    acc = pd.read_csv(ACC_CSV, low_memory=False)
    acc = acc[acc["player_type"].astype(str).str.lower() == "pitcher"].copy()
    if "played" in acc.columns:
        acc = acc[acc["played"].astype(str).str.lower().isin(["true", "1", "1.0"])].copy()

    print(f"\nRows  —  holdout test: {len(test_df):,}   2026 starts: {len(df_2026):,}   "
          f"live graded pitcher rows: {len(acc):,}\n")

    header = (f"{'Stat':<16}{'DEPLOYED':>11}{'FRESH-OOS':>11}"
              f"{'RAW-2026':>11}{'LIVE':>9}   verdict")
    print(header)
    print("-" * len(header))

    for label, tgt, pkl, proj_col, act_col in PITCHER_STATS:
        bundle = _load_bundle(pkl)
        if bundle is None:
            print(f"{label:<16}{'(no model: ' + pkl + ')':>40}")
            continue

        deployed_mae, _ = (_model_mae(bundle, test_df, tgt) if tgt in test_df.columns else (np.nan, 0))
        raw26_mae,    _ = (_model_mae(bundle, df_2026, tgt) if tgt in df_2026.columns else (np.nan, 0))
        fresh_oos = (_fresh_oos_mae(train_df, test_df, tgt)
                     if tgt in train_df.columns else np.nan)

        # Live MAE straight from the graded projection log.
        live_mae = np.nan
        if proj_col in acc.columns and act_col in acc.columns:
            sub = acc.dropna(subset=[proj_col, act_col])
            if not sub.empty:
                live_mae = float(mean_absolute_error(
                    pd.to_numeric(sub[act_col], errors="coerce"),
                    pd.to_numeric(sub[proj_col], errors="coerce")))

        # Verdict from the HONEST out-of-sample number vs live.
        verdict = ""
        if np.isfinite(fresh_oos) and np.isfinite(live_mae) and np.isfinite(deployed_mae):
            if fresh_oos - deployed_mae > 0.4:
                # honest OOS is much worse than the deployed "holdout" → that
                # holdout was in-sample (model refit on all data) → overfit.
                if abs(fresh_oos - live_mae) <= 0.35:
                    verdict = "OVERFIT: deployed-holdout was in-sample; honest skill ≈ LIVE"
                else:
                    verdict = "overfit + serving gap"
            elif abs(deployed_mae - live_mae) <= 0.35:
                verdict = "no gap (consistent)"
            else:
                verdict = "post-processing/serving adds the gap"

        def fmt(x):
            return f"{x:.3f}" if np.isfinite(x) else "  n/a"
        print(f"{label:<16}{fmt(deployed_mae):>11}{fmt(fresh_oos):>11}"
              f"{fmt(raw26_mae):>11}{fmt(live_mae):>9}   {verdict}")

    print("\nLegend:")
    print("  DEPLOYED  = the model pickle on disk, scored on the test split (in-sample if it was refit on all data)")
    print("  FRESH-OOS = a NEW model trained on the first 75%, scored on the unseen last 25% — the HONEST number")
    print("  RAW-2026  = deployed model on 2026 starts, clean features, no post-processing")
    print("  LIVE      = post-processed projection graded on the site (proj vs actual)")
    print("\n  FRESH-OOS ≈ LIVE  &  >> DEPLOYED  → the model overfits; the low 'holdout' was in-sample. LIVE is real skill.")
    print("  FRESH-OOS still tiny             → genuine leakage to hunt down.")
    print("  FRESH-OOS ≈ DEPLOYED but LIVE high → serving/post-processing is the culprit.")


if __name__ == "__main__":
    main()
