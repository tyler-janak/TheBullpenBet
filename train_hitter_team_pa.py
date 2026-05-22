"""
train_hitter_team_pa.py
=======================
Hitter projection via team-PA decomposition.

Why this exists
---------------
The current hitter model predicts PA per hitter directly. Per-hitter PA is
extremely noisy (substitutions, pinch-hits, blowout pulls) so MAE sits at
~0.98 — almost a full plate appearance off, every hitter, every game. That
error dominates every downstream counting stat (hits = PA × rate, K = PA ×
rate, etc.).

Sportsbooks model PA top-down: project team_PA from pace + opposing-starter
expected IP + bullpen quality, then split across the 9 lineup spots using
empirical lineup-position multipliers. That cuts variance in half because
team_PA varies way less than per-hitter PA — a team scores 36–42 PA in 95%
of games.

Architecture
------------
Stage 1: TEAM_PA model
    Input features per (team, game):
        - team_obp_std, team_obp_last10
        - opp_starter_ip_proj  (from train_pitcher_two_stage IP head)
        - opp_starter_k_rate, opp_starter_bb_rate
        - park K-factor (pace proxy)
    Target: team_PA_in_game (sum of all 9 hitters' PA)

Stage 2: per-PA rate models (already exist in legacy trainer)
    h_per_pa, hr_per_pa, k_per_pa, bb_per_pa, runs_per_pa, rbi_per_pa
    These fit per-hitter using the existing rolling rate features. Output is
    a clean "per opportunity" rate divorced from PA noise.

Inference (`predict_hitter_via_team_pa`):
    1. Predict team_PA for the game
    2. Split via lineup-spot multiplier (empirical, see LINEUP_SPOT_PCT)
    3. Multiply per-PA rates by the hitter's allocated PA share

Output models (./models/):
    team_pa.pkl                — stage 1
    hitter_h_per_pa.pkl        — stage 2 (one per stat)
    hitter_hr_per_pa.pkl
    hitter_k_per_pa.pkl
    hitter_bb_per_pa.pkl
    hitter_runs_per_pa.pkl
    hitter_rbi_per_pa.pkl

These coexist with the legacy `hitter_PA.pkl`, `hitter_H.pkl`, etc.
`hitterspitchers_today.py` can switch via `predict_hitter_via_team_pa()`.

Usage
-----
    python train_hitter_team_pa.py
    python train_hitter_team_pa.py --eval-holdout 0.2
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


# Empirical fraction of team PA that goes to each lineup spot (1-9).
# 3-year MLB average; sums to 1.0.
LINEUP_SPOT_PCT = {
    1: 0.122,   # leadoff — most PA
    2: 0.118,
    3: 0.115,
    4: 0.112,
    5: 0.110,
    6: 0.107,
    7: 0.105,
    8: 0.103,
    9: 0.108,   # ↑ slight: NL DH/pitcher rule changed; still less than top
}


# Stage 1 feature pool — what drives team PA per game.
# The data builder's naming convention uses `_std` for season-to-date avgs.
# After `enrich_team_features.py` runs we also have explicit team-level
# OBP and team-PA-per-game features on each hitter row.
TEAM_PA_FEATURE_POOL: list[str] = [
    # opposing starter (already on hitter rows from enrich_hitter_with_opp_starter)
    "opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_h_rate", "opp_sp_hr_rate",
    "opp_sp_k_rate_last10", "opp_sp_bb_rate_last10",
    "opp_sp_k_rate_std", "opp_sp_bb_rate_std",
    "opp_sp_ip_std",  # season-to-date IP average — the trainer's "ip_avg" hook
    "opp_sp_ip_last10",
    # team offense vs handedness (already on hitter rows)
    "team_h_rate_vs_hand", "team_bb_rate_vs_hand", "team_k_rate_vs_hand",
    "team_h_rate_vs_hand_last10", "team_bb_rate_vs_hand_last10",
    # team-level OBP + PA-per-game rolling features (from enrich_team_features.py)
    "team_obp_std", "team_obp_last10",
    "team_pa_avg_std", "team_pa_avg_last10",
    # park (pace proxy — pitcher-friendly parks have shorter games / fewer PA)
    "park_factor",
]


# Per-PA rate feature pool — same as legacy hitter feature list
HITTER_RATE_FEATURE_POOL: list[str] = [
    "h_rate_std", "hr_rate_std", "bb_rate_std", "k_rate_std",
    "h_rate_last5", "hr_rate_last5", "bb_rate_last5", "k_rate_last5",
    "h_rate_last10", "hr_rate_last10", "bb_rate_last10", "k_rate_last10",
    "h_rate_last21", "hr_rate_last21", "bb_rate_last21", "k_rate_last21",
    "h_rate_last30", "hr_rate_last30", "bb_rate_last30", "k_rate_last30",
    # batted-ball quality (drives BABIP)
    "barrel_proxy_std", "hard_hit_proxy_std", "sweet_spot_proxy_std",
    "barrel_proxy_last10", "hard_hit_proxy_last10",
    # platoon
    "hitter_h_rate_vs_hand_R", "hitter_hr_rate_vs_hand_R",
    "hitter_bb_rate_vs_hand_R", "hitter_k_rate_vs_hand_R",
    "hitter_h_rate_vs_hand_L", "hitter_hr_rate_vs_hand_L",
    "hitter_bb_rate_vs_hand_L", "hitter_k_rate_vs_hand_L",
    # opposing pitcher rates vs hitter handedness
    "opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_h_rate", "opp_sp_hr_rate",
    # lineup spot
    "lineup_spot",
    # park
    "park_factor",
]


def _make_xgb(
    n_estimators: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.04,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.5,
    reg_lambda: float = 1.5,
    seed: int = 42,
) -> XGBRegressor:
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed — pip install xgboost")
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Stage 1: team-PA model
# ---------------------------------------------------------------------------
def build_team_pa_targets(hitter_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hitter rows to (team, game_date) → team_PA_in_game."""
    if "team" not in hitter_df.columns or "game_date" not in hitter_df.columns:
        raise ValueError("hitter_game_data needs `team` and `game_date` columns")
    if "PA" not in hitter_df.columns:
        raise ValueError("hitter_game_data needs `PA` column")

    pa = pd.to_numeric(hitter_df["PA"], errors="coerce").fillna(0)
    grp = hitter_df.assign(_pa=pa).groupby(["team", "game_date"], as_index=False)["_pa"].sum()
    grp = grp.rename(columns={"_pa": "team_PA_in_game"})

    # Carry forward team-level features by joining on the average of each
    # team-game's available rolling features. We pick one row per (team, date)
    # and use its features as the team's snapshot.
    keep_cols = ["team", "game_date"] + [c for c in TEAM_PA_FEATURE_POOL if c in hitter_df.columns]
    snap = (
        hitter_df[keep_cols]
        .drop_duplicates(subset=["team", "game_date"])
    )
    return grp.merge(snap, on=["team", "game_date"], how="left")


def train_team_pa(
    hitter_csv: Path = DATA_DIR / "hitter_game_data.csv",
    eval_holdout: float = 0.0,
) -> dict:
    print(f"Loading {hitter_csv} …")
    df = pd.read_csv(hitter_csv, low_memory=False)
    print(f"  {len(df):,} hitter game rows")

    team_df = build_team_pa_targets(df)
    team_df = team_df[(team_df["team_PA_in_game"] > 20) & (team_df["team_PA_in_game"] < 70)]
    print(f"  Team-game rows after sanity filter: {len(team_df):,}")

    feats = [c for c in TEAM_PA_FEATURE_POOL if c in team_df.columns]
    print(f"  team-PA features available: {len(feats)}/{len(TEAM_PA_FEATURE_POOL)}")
    if len(feats) < 3:
        print("  ⚠️  Too few team-PA features in the rebuilt CSV; rebuild data with the team-PA snapshot fields first.")

    X = team_df[feats].copy()
    y = team_df["team_PA_in_game"].astype(float).values

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("xgb", _make_xgb()),
    ])

    if eval_holdout and 0 < eval_holdout < 1:
        order = pd.to_datetime(team_df["game_date"], errors="coerce").sort_values().index
        X = X.loc[order]
        y = team_df["team_PA_in_game"].astype(float).loc[order].values
        cut = int(len(team_df) * (1 - eval_holdout))
        pipe.fit(X.iloc[:cut], y[:cut])
        preds = pipe.predict(X.iloc[cut:])
        mae = float(mean_absolute_error(y[cut:], preds))
        print(f"  team_PA  n={len(team_df):>5}  features={len(feats):>2}  MAE(holdout)={mae:.2f}")
    else:
        pipe.fit(X, y)
        preds = pipe.predict(X)
        mae = float(mean_absolute_error(y, preds))
        print(f"  team_PA  n={len(team_df):>5}  features={len(feats):>2}  MAE(in-sample)={mae:.2f}")

    bundle = {
        "model": pipe,
        "features": feats,
        "target": "team_PA",
        "kind": "team",
        "transform": "identity",
        "n_rows": int(len(team_df)),
        "mae": mae,
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    out_path = MODELS_DIR / "team_pa.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"  → {out_path.name}")
    return bundle


# ---------------------------------------------------------------------------
# Stage 2: per-PA rate models
# ---------------------------------------------------------------------------
def _attach_per_pa_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add h_per_pa, hr_per_pa, k_per_pa, bb_per_pa, runs_per_pa, rbi_per_pa."""
    out = df.copy()
    pa = pd.to_numeric(out.get("PA"), errors="coerce")
    pa_floor = pa.clip(lower=1.0)
    for raw, rate_col in [
        ("H", "h_per_pa"), ("HR", "hr_per_pa"),
        ("K", "k_per_pa"), ("BB", "bb_per_pa"),
        ("R", "runs_per_pa"), ("RBI", "rbi_per_pa"),
    ]:
        if raw in out.columns:
            out[rate_col] = pd.to_numeric(out[raw], errors="coerce") / pa_floor
            out[rate_col] = out[rate_col].clip(lower=0, upper=1.0)
    return out


def train_rate_models(
    hitter_csv: Path = DATA_DIR / "hitter_game_data.csv",
    eval_holdout: float = 0.0,
) -> dict:
    print(f"\nLoading {hitter_csv} for per-PA rate models …")
    df = pd.read_csv(hitter_csv, low_memory=False)
    df = df[pd.to_numeric(df.get("PA"), errors="coerce") >= 1].copy()
    df = _attach_per_pa_targets(df)
    print(f"  {len(df):,} hitter game rows after PA >= 1 filter")

    feats_pool = [c for c in HITTER_RATE_FEATURE_POOL if c in df.columns]
    print(f"  rate-feature pool: {len(feats_pool)}/{len(HITTER_RATE_FEATURE_POOL)}")

    targets = ["h_per_pa", "hr_per_pa", "k_per_pa", "bb_per_pa", "runs_per_pa", "rbi_per_pa"]
    out: dict[str, dict] = {}

    for tgt in targets:
        if tgt not in df.columns:
            print(f"  {tgt:<14} MISSING in CSV — skip")
            continue
        sub = df[feats_pool + [tgt]].dropna(subset=[tgt])
        if sub.empty:
            print(f"  {tgt:<14} no rows after dropna — skip")
            continue
        X = sub[feats_pool].copy()
        y = sub[tgt].astype(float).values
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("xgb", _make_xgb())])

        if eval_holdout and 0 < eval_holdout < 1:
            order = pd.to_datetime(df.loc[sub.index, "game_date"], errors="coerce").sort_values().index
            X = X.loc[order]
            y = sub[tgt].astype(float).loc[order].values
            cut = int(len(sub) * (1 - eval_holdout))
            pipe.fit(X.iloc[:cut], y[:cut])
            preds = pipe.predict(X.iloc[cut:])
            mae = float(mean_absolute_error(y[cut:], preds))
        else:
            pipe.fit(X, y)
            preds = pipe.predict(X)
            mae = float(mean_absolute_error(y, preds))
        print(f"  {tgt:<14} n={len(sub):>5}  features={len(feats_pool):>2}  MAE={mae:.4f}")

        bundle = {
            "model": pipe,
            "features": feats_pool,
            "target": tgt,
            "kind": "hitter",
            "transform": "identity",
            "n_rows": int(len(sub)),
            "mae": mae,
            "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
        out_path = MODELS_DIR / f"hitter_{tgt}.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(bundle, fh)
        out[tgt] = bundle

    return out


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def predict_hitter_via_team_pa(
    feature_row: pd.Series,
    lineup_spot: int,
    team_pa_predicted: float,
    rate_models: dict,
) -> dict:
    """
    Score one hitter using the team-PA decomposition.

    `team_pa_predicted` should already have been computed once per (team, game)
    via the team_pa.pkl model — passing it in saves us re-predicting it for
    every hitter on the team.
    """
    pa_share = LINEUP_SPOT_PCT.get(int(lineup_spot or 9), 0.111)
    hitter_pa = float(team_pa_predicted) * pa_share

    out = {"PA": hitter_pa}
    for raw, rate_col in [
        ("H", "h_per_pa"), ("HR", "hr_per_pa"),
        ("K", "k_per_pa"), ("BB", "bb_per_pa"),
        ("R", "runs_per_pa"), ("RBI", "rbi_per_pa"),
    ]:
        bundle = rate_models.get(rate_col)
        if bundle is None:
            continue
        feats = bundle["features"]
        X = pd.DataFrame([feature_row]).reindex(columns=feats)
        rate = float(bundle["model"].predict(X)[0])
        rate = max(0.0, min(1.0, rate))
        out[raw] = rate * hitter_pa
    return out


def load_team_pa_models(models_dir: Path = MODELS_DIR) -> dict:
    """Return ({"team_pa": bundle}, {"h_per_pa": bundle, ...})."""
    team_p = models_dir / "team_pa.pkl"
    team_bundle = None
    if team_p.exists():
        with open(team_p, "rb") as fh:
            team_bundle = pickle.load(fh)

    rate_models: dict[str, dict] = {}
    for tgt in ("h_per_pa", "hr_per_pa", "k_per_pa", "bb_per_pa", "runs_per_pa", "rbi_per_pa"):
        p = models_dir / f"hitter_{tgt}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            rate_models[tgt] = pickle.load(fh)

    return team_bundle, rate_models


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hitter-csv", default=str(DATA_DIR / "hitter_game_data.csv"),
                    help="Path to per-game hitter feature table.")
    ap.add_argument("--eval-holdout", type=float, default=0.20,
                    help="Holdout fraction for honest MAE eval (0 disables).")
    ap.add_argument("--skip-rate-models", action="store_true",
                    help="Only train team_pa.pkl, skip the per-PA rate models.")
    args = ap.parse_args()

    train_team_pa(Path(args.hitter_csv), eval_holdout=args.eval_holdout)
    if not args.skip_rate_models:
        train_rate_models(Path(args.hitter_csv), eval_holdout=args.eval_holdout)


if __name__ == "__main__":
    main()
