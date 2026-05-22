"""
train_pitcher_xhits.py
======================
Smoothed-target hits-allowed model — separates skill from BABIP variance.

Why this exists
---------------
The two-stage pitcher model's H9 head has MAE 2.32 in holdout (≈ 1.4 H per
5.5-IP start). That looks reasonable until you realize industry hits-allowed
MAE is ~1.3 per game. The gap is BABIP noise: roughly 30% of a hit outcome
is sequence luck the model can't predict from any feature.

Sportsbooks beat this by modelling **expected hits** rather than raw hits.
Two common shapes:

  1. xH = BIP × league_BABIP × park_BABIP_factor
        (deterministic; needs ball-in-play count per pitch)
  2. xH = α · actual_H + (1-α) · career_H_avg_pre_game
        (target smoothing; works with whatever features we already have)

We use (2) because it's a target-engineering tweak with no new data plumbing:
the trainer reads the same `pitcher_game_data.csv`, but the target it fits
is a shrinkage average that dampens BABIP outliers. The trained model learns
the *signal* component, not the noise.

At inference, the new `xH9` prediction can either replace the two-stage `H9`
prediction or blend with it. We default to a 50/50 blend in `apply_xhits()`
which is called from `score_pitchers`.

Output
------
    models/pitcher_xH9.pkl

Coexists with the existing `pitcher_H9.pkl` from the two-stage trainer. Both
are loaded by `hitterspitchers_today.py`; the scoring code blends.

Usage
-----
    python train_pitcher_xhits.py
    python train_pitcher_xhits.py --alpha 0.7 --eval-holdout 0.2
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _make_xgb() -> XGBRegressor:
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed — pip install xgboost")
    return XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.6, reg_lambda=2.0,   # heavier regularization than direct H9
        objective="reg:squarederror",
        random_state=42, tree_method="hist", n_jobs=-1,
    )


# Same feature pool as the two-stage trainer
XH_FEATURE_POOL: list[str] = [
    "K_rate_std", "BB_rate_std", "HR_rate_std", "H_rate_std",
    "IP_std", "avg_velocity_std", "avg_spin_std",
    "K_rate_last5", "BB_rate_last5", "HR_rate_last5", "H_rate_last5", "IP_last5",
    "K_rate_last10", "BB_rate_last10", "HR_rate_last10", "H_rate_last10", "IP_last10",
    "K_rate_last21", "BB_rate_last21", "HR_rate_last21", "H_rate_last21", "IP_last21",
    "K_rate_last30", "BB_rate_last30", "HR_rate_last30", "H_rate_last30", "IP_last30",
    "K_last5", "K_last10", "BB_last5", "BB_last10",
    "H_last5", "H_last10", "HR_last5", "HR_last10",
    "BF_std", "outs_std", "pitches_std",
    "BF_last5", "outs_last5", "pitches_last5",
    "BF_last10", "outs_last10", "pitches_last10",
    "BF_per_IP_std", "pitches_per_BF_std", "pitches_per_IP_std",
    "max_ip_last5", "max_ip_last10",
    "days_rest", "starter_pct_last5", "starter_pct_last10",
    "team_k_rate_vs_hand", "team_k_rate_vs_hand_last10",
    "team_bb_rate_vs_hand", "team_bb_rate_vs_hand_last10",
    "team_hr_rate_vs_hand", "team_hr_rate_vs_hand_last10",
    "team_h_rate_vs_hand", "team_h_rate_vs_hand_last10",
    "park_factor",
    # Pitch-quality features — particularly hard-hit/EV/barrel for hits-allowed
    "whiff_rate_std", "whiff_rate_last10",
    "csw_pct_std", "csw_pct_last10",
    "zone_pct_std", "zone_pct_last10",
    "hard_hit_pct_std", "hard_hit_pct_last10",
    "barrel_pct_std", "barrel_pct_last10",
    "avg_ev_allowed_std", "avg_ev_allowed_last10",
    "fb_velo_std", "fb_velo_last10",
]


def build_xh_target(df: pd.DataFrame, alpha: float = 0.6) -> pd.DataFrame:
    """
    Add an `xH9` column = α · actual_H9 + (1-α) · career_H9_avg_pre_game.

    `career_H9_avg_pre_game` is the pitcher's season-to-date hits-allowed
    rate computed only from games strictly before the row's `game_date`,
    so there's no leakage. Pitchers with no prior history use league mean.
    """
    work = df.copy()
    if "IP" not in work.columns or "H" not in work.columns:
        raise ValueError("xH target needs H and IP columns")

    pitcher_col = next((c for c in ("pitcher", "pitcher_id", "player_id") if c in work.columns), None)
    if pitcher_col is None:
        raise ValueError("xH target needs a pitcher id column (`pitcher`)")
    date_col = next((c for c in ("game_date", "date") if c in work.columns), None)
    if date_col is None:
        raise ValueError("xH target needs a game_date column")

    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values([pitcher_col, date_col]).reset_index(drop=True)

    ip = pd.to_numeric(work["IP"], errors="coerce")
    h  = pd.to_numeric(work["H"],  errors="coerce")
    ip_floor = ip.clip(lower=0.1)
    work["_H9_actual"] = (h / ip_floor) * 9.0
    work["_H9_actual"] = work["_H9_actual"].clip(lower=0, upper=27)

    # Per-pitcher running mean of historical H9 — shifted by 1 game so the
    # current row is excluded (no leakage).
    grp = work.groupby(pitcher_col)["_H9_actual"]
    work["_H9_career"] = grp.transform(lambda s: s.shift(1).expanding().mean())

    league_mean_h9 = float(work["_H9_actual"].mean())
    work["_H9_career"] = work["_H9_career"].fillna(league_mean_h9)

    work["xH9"] = alpha * work["_H9_actual"] + (1.0 - alpha) * work["_H9_career"]
    return work


def train_xhits(
    pitcher_csv: Path = DATA_DIR / "pitcher_game_data.csv",
    alpha: float = 0.6,
    min_ip: float = 0.5,
    eval_holdout: float = 0.20,
) -> dict:
    print(f"Loading {pitcher_csv} …")
    df = pd.read_csv(pitcher_csv, low_memory=False)
    df = df[pd.to_numeric(df.get("IP"), errors="coerce") >= min_ip].copy()
    print(f"  {len(df):,} pitcher games after IP >= {min_ip} filter")

    df = build_xh_target(df, alpha=alpha)
    print(f"  Built xH9 target: α={alpha:.2f}  range "
          f"{df['xH9'].min():.2f}…{df['xH9'].max():.2f}  mean {df['xH9'].mean():.2f}")

    feats = [c for c in XH_FEATURE_POOL if c in df.columns]
    print(f"  Feature pool: {len(feats)}/{len(XH_FEATURE_POOL)} present")
    X = df[feats].copy()
    y = df["xH9"].astype(float).values

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("xgb", _make_xgb())])

    if eval_holdout and 0 < eval_holdout < 1:
        date_col = next((c for c in ("game_date", "date") if c in df.columns), None)
        order = pd.to_datetime(df[date_col], errors="coerce").sort_values().index
        X = X.loc[order]
        y = df["xH9"].astype(float).loc[order].values
        cut = int(len(df) * (1 - eval_holdout))
        pipe.fit(X.iloc[:cut], y[:cut])
        preds = pipe.predict(X.iloc[cut:])
        mae = float(mean_absolute_error(y[cut:], preds))
        print(f"  xH9  n={len(df):>5}  features={len(feats):>2}  MAE(holdout)={mae:.3f}")
    else:
        pipe.fit(X, y)
        preds = pipe.predict(X)
        mae = float(mean_absolute_error(y, preds))
        print(f"  xH9  n={len(df):>5}  features={len(feats):>2}  MAE(in-sample)={mae:.3f}")

    bundle = {
        "model": pipe, "features": feats, "target": "xH9",
        "kind": "pitcher", "transform": "identity",
        "alpha": alpha, "n_rows": int(len(df)), "mae": mae,
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    out_path = MODELS_DIR / "pitcher_xH9.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"  → {out_path.name}")
    return bundle


def load_xhits_model(models_dir: Path = MODELS_DIR) -> dict | None:
    p = models_dir / "pitcher_xH9.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def predict_xhits(feature_row: pd.Series, ip: float, bundle: dict | None) -> float | None:
    """Predict expected H (not H9) for one pitcher game given the bundle + IP."""
    if bundle is None:
        return None
    feats = bundle["features"]
    X = pd.DataFrame([feature_row]).reindex(columns=feats)
    xh9 = float(bundle["model"].predict(X)[0])
    xh9 = max(0.0, xh9)
    return xh9 * (max(0.1, ip) / 9.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitcher-csv", default=str(DATA_DIR / "pitcher_game_data.csv"))
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Weight on actual H9 vs career avg (0.6 = mostly actual, "
                         "0.3 = heavily smoothed)")
    ap.add_argument("--min-ip", type=float, default=0.5)
    ap.add_argument("--eval-holdout", type=float, default=0.20)
    args = ap.parse_args()
    train_xhits(Path(args.pitcher_csv), alpha=args.alpha,
                min_ip=args.min_ip, eval_holdout=args.eval_holdout)


if __name__ == "__main__":
    main()
