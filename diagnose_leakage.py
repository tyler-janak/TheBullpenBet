"""
diagnose_leakage.py
===================
Find out WHY the pitcher models score ~0.7 MAE out-of-sample on the stored
feature rows when no honest strikeout model can beat ~1.4-2.0. Either the
stored training rows leak the outcome, or the features are legitimately that
strong. This decides the whole charge-grade strategy.

Three independent checks per target (K, BB, H, IP):

  1. CORRELATIONS — Pearson |corr| of every feature with the target. A noisy
     per-game count should NOT correlate > ~0.85 with any strictly pre-game
     feature. Anything that high is a prime leak suspect.

  2. NAIVE BASELINE — MAE from just predicting the trailing season average
     (the *_std feature) directly, scored out-of-sample. This is the honest
     floor: if the model can only see trailing info, it can't beat this by
     much. (Naive K_std → K should be ~1.8-2.2.)

  3. ABLATION — fit a fresh model on the first 75% and score the unseen 25%,
     once with ALL features and once with the high-correlation suspects
     REMOVED. If the clean MAE jumps toward ~2.0, those features were the leak.

How to read it
--------------
  • Naive ~2.0, full-model ~0.7, clean (suspects removed) ~2.0
        → LEAKAGE confirmed in the listed features. Remove them from the
          feature pool, retrain, and the honest baseline is ~2.0.
  • Naive ~0.7
        → the *_std trailing column itself is contaminated (built without a
          proper shift) — the data build leaks; rebuild the feature tables.
  • Naive ~2.0, full-model ~2.0
        → no leak; the model's honest skill is ~2.0 and we improve from there.

Usage
-----
    python diagnose_leakage.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import hitterspitchers_train as hpt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PITCHER_CSV = HERE / "data" / "pitcher_game_data.csv"

# (target column, the trailing-average feature that is its honest naive baseline)
TARGETS = [("K", "K_std"), ("BB", "BB_std"), ("H", "H_std"), ("IP", "IP_std")]

CORR_FLAG = 0.85   # |corr| above this with a noisy count = leak suspect


def main() -> None:
    if not PITCHER_CSV.exists():
        print(f"⚠️  {PITCHER_CSV} not found"); return

    df = pd.read_csv(PITCHER_CSV, low_memory=False)
    date_col = "game_date" if "game_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "is_actual_starter" in df.columns:
        df = df[pd.to_numeric(df["is_actual_starter"], errors="coerce").fillna(0) == 1].copy()

    train_df, test_df = hpt.chronological_split(df, date_col)
    feat_pool = [f for f in hpt.PITCHER_FEATURES if f in df.columns]
    print(f"Rows: {len(df):,}  (train {len(train_df):,} / test {len(test_df):,})   "
          f"feature pool: {len(feat_pool)}\n")

    for tgt, naive_feat in TARGETS:
        if tgt not in df.columns:
            print(f"== {tgt}: not in data ==\n"); continue
        print("=" * 70)
        print(f"TARGET: {tgt}")
        print("=" * 70)

        # ── 1. correlations ──
        y = pd.to_numeric(df[tgt], errors="coerce")
        corrs = {}
        for f in feat_pool:
            x = pd.to_numeric(df[f], errors="coerce")
            if x.notna().sum() > 100 and x.std(skipna=True) > 1e-9:
                corrs[f] = abs(x.corr(y))
        corr_s = pd.Series(corrs).sort_values(ascending=False)
        suspects = corr_s[corr_s > CORR_FLAG].index.tolist()
        print("  Top feature |corr| with target:")
        for f, c in corr_s.head(12).items():
            flag = "  <-- LEAK SUSPECT" if c > CORR_FLAG else ""
            print(f"      {c:5.3f}  {f}{flag}")

        # ── 2. naive trailing-average baseline (out-of-sample) ──
        naive_mae = np.nan
        if naive_feat in test_df.columns:
            te = test_df.dropna(subset=[tgt, naive_feat])
            if not te.empty:
                naive_mae = float(mean_absolute_error(
                    pd.to_numeric(te[tgt], errors="coerce"),
                    pd.to_numeric(te[naive_feat], errors="coerce")))

        # ── 3. fresh-OOS ablation: all features vs suspects removed ──
        def _oos(feats):
            tr = train_df.dropna(subset=[tgt])
            te = test_df.dropna(subset=[tgt])
            if tr.empty or te.empty or not feats:
                return np.nan
            pipe = hpt.build_sklearn_model("xgb", target_name=tgt)
            pipe.fit(tr[feats], pd.to_numeric(tr[tgt], errors="coerce"))
            pred = np.asarray(pipe.predict(te[feats]), dtype=float)
            yv = pd.to_numeric(te[tgt], errors="coerce").values
            m = np.isfinite(yv) & np.isfinite(pred)
            return float(mean_absolute_error(yv[m], pred[m])) if m.sum() else np.nan

        full_oos  = _oos(feat_pool)

        # Targeted ablation: remove the NO-WINDOW context/platoon features
        # (team_*_rate_vs_hand and pitcher_*_rate_vs_hand_{R,L} with no
        # _last/_std suffix). Those carry the current game's realized rate
        # (leak); the trailing _last/_std versions are kept.
        no_window = [f for f in feat_pool
                     if "vs_hand" in f and "_last" not in f and "_std" not in f]
        clean_pool = [f for f in feat_pool if f not in no_window]
        clean_oos = _oos(clean_pool)

        print(f"\n  Naive baseline (predict {naive_feat} directly, OOS): {naive_mae:.3f}")
        print(f"  Fresh model OOS, ALL features:                      {full_oos:.3f}")
        print(f"  Fresh model OOS, no-window vs_hand REMOVED:         {clean_oos:.3f}")
        print(f"      removed {len(no_window)}: {', '.join(no_window)}")
        print()


if __name__ == "__main__":
    main()
