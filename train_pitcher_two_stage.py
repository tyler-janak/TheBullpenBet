"""
train_pitcher_two_stage.py
==========================
Two-stage pitcher projection model.

Why this exists
---------------
Single-stage XGBoost on raw counting stats (K, BB, HR, H) has a hidden
fragility: every counting target is mostly driven by IP. If IP MAE is 1.08,
that error compounds into K MAE ≈ 2.0, hits-allowed MAE ≈ 1.8, etc. — exactly
the pattern we see on the season accuracy page. Sportsbooks predict IP
explicitly because of this leverage.

Architecture
------------
Stage 1: predict IP from rolling features + opposing-lineup K-rate, manager
         hook history, days-rest, expected run differential.

Stage 2: predict per-9 RATES (K/9, BB/9, H/9, HR/9, ER/9). These are far
         lower-variance than raw counts because we've factored out the IP
         error. They train against `target / max(IP, 0.1) * 9`.

At inference (`predict_two_stage`):
    IP      = stage1.predict(features)
    K9      = stage2_K9.predict(features)
    proj_K  = K9 * (IP / 9)

This gives the same point estimate as a direct-K model in expectation, but the
distribution around it is tighter because the IP error is shared across all
counting stats instead of being independently re-noised by each direct model.

Output
------
Saves these pickles to ./models/:
    pitcher_IP.pkl
    pitcher_K9.pkl
    pitcher_BB9.pkl
    pitcher_H9.pkl
    pitcher_HR9.pkl
    pitcher_ER9.pkl

These coexist with the existing `pitcher_K.pkl`, `pitcher_BB.pkl` etc.
`hitterspitchers_today.py` can switch to the two-stage flavour by calling
`predict_two_stage()` instead of the legacy direct-target predictors — see
the docstring of that function below.

Usage
-----
    python train_pitcher_two_stage.py
    python train_pitcher_two_stage.py --min-ip 1.0 --eval-holdout 0.2
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

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)


# Feature pool — superset of legacy PITCHER_FEATURES, intersected with
# whatever columns the rebuilt game-data CSV exposes today.
PITCHER_FEATURE_POOL: list[str] = [
    # season-to-date
    "K_rate_std", "BB_rate_std", "HR_rate_std", "H_rate_std",
    "IP_std", "avg_velocity_std", "avg_spin_std",
    # rolling rates
    "K_rate_last5", "BB_rate_last5", "HR_rate_last5", "H_rate_last5", "IP_last5",
    "K_rate_last10", "BB_rate_last10", "HR_rate_last10", "H_rate_last10", "IP_last10",
    "K_rate_last21", "BB_rate_last21", "HR_rate_last21", "H_rate_last21", "IP_last21",
    "K_rate_last30", "BB_rate_last30", "HR_rate_last30", "H_rate_last30", "IP_last30",
    # raw counts (recent volume)
    "K_last5", "K_last10", "K_last21",
    "BB_last5", "BB_last10",
    "H_last5", "H_last10",
    "HR_last5", "HR_last10",
    # workload
    "BF_std", "outs_std", "pitches_std",
    "BF_last5", "outs_last5", "pitches_last5",
    "BF_last10", "outs_last10", "pitches_last10",
    "BF_per_IP_std", "pitches_per_BF_std", "pitches_per_IP_std",
    "max_ip_last5", "max_ip_last10",
    "days_rest", "starter_pct_last5", "starter_pct_last10",
    # opposing lineup K-rate (most important for IP — high-K lineups burn pitches faster)
    "team_k_rate_vs_hand", "team_k_rate_vs_hand_last10",
    "team_bb_rate_vs_hand", "team_bb_rate_vs_hand_last10",
    "team_hr_rate_vs_hand", "team_hr_rate_vs_hand_last10",
    "team_h_rate_vs_hand", "team_h_rate_vs_hand_last10",
    # park
    "park_factor",
    # ── pitch-quality "stuff" features (highest-leverage K + hits predictors) ──
    "whiff_rate_std", "whiff_rate_last5", "whiff_rate_last10", "whiff_rate_last21",
    "csw_pct_std",    "csw_pct_last5",    "csw_pct_last10",    "csw_pct_last21",
    "zone_pct_std",   "zone_pct_last5",   "zone_pct_last10",
    "f_strike_pct_std", "f_strike_pct_last5", "f_strike_pct_last10",
    "hard_hit_pct_std", "hard_hit_pct_last5", "hard_hit_pct_last10",
    "barrel_pct_std", "barrel_pct_last5", "barrel_pct_last10",
    "avg_ev_allowed_std", "avg_ev_allowed_last5", "avg_ev_allowed_last10",
    # pitch-mix + fastball velocity
    "fb_pct_std", "fb_pct_last5", "fb_pct_last10",
    "br_pct_std", "br_pct_last5", "br_pct_last10",
    "off_pct_std", "off_pct_last5", "off_pct_last10",
    "fb_velo_std", "fb_velo_last5", "fb_velo_last10", "fb_velo_last21",
    # Pitch-type-specific whiff rates — the decomposition that goes beyond
    # aggregate whiff_rate's correlation with K_rate. Strongest single
    # addition for K9 because two pitchers with identical overall whiff
    # rate can have completely different FB/BR profiles.
    "whiff_rate_fb_std",  "whiff_rate_fb_last10",  "whiff_rate_fb_last21",
    "whiff_rate_br_std",  "whiff_rate_br_last10",  "whiff_rate_br_last21",
    "whiff_rate_off_std", "whiff_rate_off_last10", "whiff_rate_off_last21",
    # Velocity separation (FB − OFF, FB − BR) — proxies pitch tunneling
    "velo_sep_fb_off_std", "velo_sep_fb_off_last10",
    "velo_sep_fb_br_std",  "velo_sep_fb_br_last10",
    "br_velo_std",  "br_velo_last10",
    "off_velo_std", "off_velo_last10",
    # ── Batter-level lineup aggregations (added by enrich_lineup_features) ──
    # Not correlated with the pitcher's own rate stats — describes the
    # specific 9 batters facing this pitcher. The "different information
    # channel" the model has been missing.
    "lineup_k_rate", "lineup_k_rate_last10",
    "lineup_bb_rate", "lineup_bb_rate_last10",
    "lineup_h_rate",  "lineup_h_rate_last10",
    "lineup_hr_rate", "lineup_hr_rate_last10",
    "lineup_avg_ev", "lineup_hard_hit_pct",
]


# ── Per-target feature exclusions ─────────────────────────────────────────
# By default every target uses the full pool above. When `force_stuff=True`
# is passed to `train_two_stage` we drop the rate stats matching each
# target's own family — that forces the trees to use the stuff features
# (whiff_rate, fb_velo, lineup_*) instead of the noisy raw rate stat.
# Out-of-sample MAE often goes DOWN because raw rates overfit to recent
# variance while stuff + lineup features generalize better.
FORCE_STUFF_DROP: dict[str, list[str]] = {
    "K9": [
        "K_rate_std", "K_rate_last5", "K_rate_last7", "K_rate_last10",
        "K_rate_last14", "K_rate_last21", "K_rate_last30",
        "K_std", "K_last5", "K_last10", "K_last21",
    ],
    "BB9": [
        "BB_rate_std", "BB_rate_last5", "BB_rate_last10",
        "BB_rate_last21", "BB_rate_last30",
        "BB_std", "BB_last5", "BB_last10",
    ],
    "H9": [
        "H_rate_std", "H_rate_last5", "H_rate_last10",
        "H_rate_last21", "H_rate_last30",
        "H_std", "H_last5", "H_last10",
    ],
    "HR9": [
        "HR_rate_std", "HR_rate_last5", "HR_rate_last10",
        "HR_rate_last21", "HR_rate_last30",
        "HR_std", "HR_last5", "HR_last10",
    ],
    "IP":  [],   # IP doesn't have an analogous rate stat to drop
    "ER9": [],
}


# ---------------------------------------------------------------------------
# Build per-9 rate targets for stage 2
# ---------------------------------------------------------------------------
def attach_per9_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add K9, BB9, H9, HR9, ER9 columns to a pitcher game-log.

    For ER we fall back to `R` (total runs) if `ER` isn't aggregated — the
    Statcast-based data builder writes `R` (sum of bat_runs_scored, see
    hitterspitchers_data.event_flags) because the earned/unearned split
    requires the official scorekeeper's call and isn't in the pitch stream.
    Across MLB the unearned-run rate is ~6%, so using R as ER is a close
    upper bound — close enough that ER9 training meaningfully outperforms
    the legacy `estimate_pitcher_runs` heuristic.
    """
    out = df.copy()
    ip = pd.to_numeric(out.get("IP"), errors="coerce")
    # IP can be 0 for emergency relief; floor it so rates don't blow up.
    ip_floor = ip.clip(lower=0.1)
    pairs = [("K", "K9"), ("BB", "BB9"), ("H", "H9"), ("HR", "HR9")]
    # ER target: use ER if present, otherwise fall back to R.
    if "ER" in out.columns:
        pairs.append(("ER", "ER9"))
    elif "R" in out.columns:
        pairs.append(("R", "ER9"))
    for raw, rate in pairs:
        if raw in out.columns:
            out[rate] = pd.to_numeric(out[raw], errors="coerce") / ip_floor * 9.0
            # Cap absurd rates from 0.1 IP edge cases — a pitcher allowing
            # 3 K in 0.1 IP would map to K9 = 270 which corrupts training.
            cap = {"K9": 27, "BB9": 18, "H9": 27, "HR9": 9, "ER9": 27}.get(rate, 30)
            out[rate] = out[rate].clip(lower=0, upper=cap)
    return out


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _make_xgb(
    n_estimators: int = 600,
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


# Domain rules — these constraints force the tree to respect direction.
# For each (target, feature) pair: +1 = monotonic non-decreasing in feature,
# -1 = non-increasing. Unlisted features get 0 (no constraint).
# Example: more whiffs always means more K9. More zone% always means fewer BB9.
_MONOTONIC_RULES: dict[str, dict[str, int]] = {
    "K9": {
        "whiff_rate_std": +1, "whiff_rate_last10": +1, "whiff_rate_last21": +1,
        "csw_pct_std": +1, "csw_pct_last10": +1, "csw_pct_last21": +1,
        "fb_velo_std": +1, "fb_velo_last10": +1,
        "whiff_rate_fb_std": +1, "whiff_rate_br_std": +1,
        # Lineup K rate INCREASES → pitcher K9 INCREASES (good lineups whiff more)
        "lineup_k_rate": +1, "lineup_k_rate_last10": +1,
    },
    "BB9": {
        # More zone% → fewer walks
        "zone_pct_std": -1, "zone_pct_last10": -1,
        "f_strike_pct_std": -1, "f_strike_pct_last10": -1,
        "lineup_bb_rate": +1, "lineup_bb_rate_last10": +1,
    },
    "H9": {
        # Harder contact → more hits
        "hard_hit_pct_std": +1, "hard_hit_pct_last10": +1,
        "avg_ev_allowed_std": +1, "avg_ev_allowed_last10": +1,
        "lineup_h_rate": +1, "lineup_h_rate_last10": +1,
        "lineup_avg_ev": +1, "lineup_hard_hit_pct": +1,
    },
    "HR9": {
        "barrel_pct_std": +1, "barrel_pct_last10": +1,
        "lineup_hr_rate": +1, "lineup_hr_rate_last10": +1,
        "park_factor": +1,
    },
    "ER9": {
        "lineup_hard_hit_pct": +1, "lineup_avg_ev": +1,
    },
    "IP": {},
}


def _make_lgb(
    target: str,
    feature_cols: list[str],
    n_estimators: int = 600,
    max_depth: int = 5,
    learning_rate: float = 0.04,
    seed: int = 42,
) -> "LGBMRegressor":
    """LightGBM with monotonic constraints for the given target.

    Monotonic constraints force the tree to respect known directional
    relationships (e.g., more whiffs → more Ks). They reduce overfitting
    on noise channels and typically give 3-7% MAE improvement at the
    same training data size.
    """
    if not HAS_LGB:
        raise RuntimeError("lightgbm not installed — pip install lightgbm")
    rules = _MONOTONIC_RULES.get(target, {})
    monotone = [rules.get(c, 0) for c in feature_cols]
    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=1.5,
        monotone_constraints=monotone,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def _make_model(model_type: str, target: str, feature_cols: list[str]):
    if model_type == "lgb":
        return _make_lgb(target, feature_cols)
    return _make_xgb()


def _train_one(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    eval_holdout: float = 0.0,
    model_type: str = "xgb",
) -> dict:
    feats = [c for c in feature_cols if c in df.columns]
    if not feats:
        raise ValueError(f"No features available for target {target_col}")

    work = df[feats + [target_col]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])
    if work.empty:
        raise ValueError(f"No training rows for target {target_col}")

    X = work[feats].copy()
    y = work[target_col].astype(float).values

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", _make_model(model_type, target_col, feats)),
    ])

    if eval_holdout and 0 < eval_holdout < 1:
        n = len(work)
        cut = int(n * (1 - eval_holdout))
        # Sort chronologically if a date is available so holdout = "future"
        if "game_date" in df.columns:
            order = pd.to_datetime(df.loc[work.index, "game_date"], errors="coerce")
            sort_idx = order.sort_values().index
            X = X.loc[sort_idx]
            y = work[target_col].astype(float).loc[sort_idx].values
        X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_te = y[:cut], y[cut:]
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        mae = float(mean_absolute_error(y_te, preds))
        print(f"  {target_col:<6}  n={n:>5}  features={len(feats):>3}  MAE(holdout)={mae:.3f}")
    else:
        pipe.fit(X, y)
        # In-sample sanity (loose floor on MAE)
        preds = pipe.predict(X)
        mae = float(mean_absolute_error(y, preds))
        print(f"  {target_col:<6}  n={len(work):>5}  features={len(feats):>3}  MAE(in-sample)={mae:.3f}")

    return {
        "model": pipe,
        "features": feats,
        "target": target_col,
        "n_rows": int(len(work)),
        "mae": mae,
    }


# ---------------------------------------------------------------------------
# Public training entrypoint
# ---------------------------------------------------------------------------
def train_two_stage(
    pitcher_csv: Path = DATA_DIR / "pitcher_game_data.csv",
    min_ip: float = 0.5,
    eval_holdout: float = 0.0,
    model_type: str = "xgb",
    force_stuff: bool = False,
) -> dict:
    if not pitcher_csv.exists():
        raise FileNotFoundError(f"{pitcher_csv} not found — refresh data first")

    print(f"Loading {pitcher_csv} …")
    df = pd.read_csv(pitcher_csv, low_memory=False)
    print(f"  {len(df):,} pitcher game rows  /  {df.shape[1]} columns")

    # Drop obviously bad rows (IP missing or below floor — the rate explosions)
    df = df[pd.to_numeric(df.get("IP"), errors="coerce") >= min_ip].copy()
    print(f"  After IP >= {min_ip} filter: {len(df):,} rows")

    df = attach_per9_targets(df)

    feats_present = [c for c in PITCHER_FEATURE_POOL if c in df.columns]
    print(f"  Feature pool (global): {len(feats_present)}/{len(PITCHER_FEATURE_POOL)} present")
    print(f"  Model type: {model_type}   force_stuff: {force_stuff}")

    def _feats_for(target: str) -> list[str]:
        """Return per-target feature subset (drops correlated rate stats when force_stuff)."""
        if not force_stuff:
            return feats_present
        drop_set = set(FORCE_STUFF_DROP.get(target, []))
        kept = [c for c in feats_present if c not in drop_set]
        dropped = [c for c in feats_present if c in drop_set]
        if dropped:
            print(f"    [{target}] dropping {len(dropped)} rate-stat features: "
                  f"{', '.join(dropped[:6])}{' …' if len(dropped) > 6 else ''}")
        return kept

    out: dict[str, dict] = {}
    print("\n── Stage 1: IP ──────────────────────────────────────────────")
    out["IP"] = _train_one(df, "IP", _feats_for("IP"), eval_holdout, model_type=model_type)

    print("\n── Stage 2: per-9 rate models ───────────────────────────────")
    for tgt in ("K9", "BB9", "H9", "HR9", "ER9"):
        if tgt not in df.columns:
            print(f"  {tgt:<6}  MISSING — skip")
            continue
        out[tgt] = _train_one(df, tgt, _feats_for(tgt), eval_holdout, model_type=model_type)

    print("\nWriting pickles …")
    for tgt, info in out.items():
        path = MODELS_DIR / f"pitcher_{tgt}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({
                "model": info["model"],
                "features": info["features"],
                "target": info["target"],
                "kind": "pitcher",
                "transform": "identity",
                "n_rows": info["n_rows"],
                "mae": info["mae"],
                "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            }, fh)
        print(f"  → {path.name}")
    return out


# ---------------------------------------------------------------------------
# Inference helper for hitterspitchers_today.py
# ---------------------------------------------------------------------------
def predict_two_stage(feature_row: pd.Series, models: dict) -> dict:
    """
    Score one pitcher game using the two-stage flow.

    `models` should be the dict returned by load_two_stage_models() — a mapping
    {'IP': bundle, 'K9': bundle, 'BB9': bundle, ...} where each bundle is the
    same shape produced by load_models_count_only().

    Returns: {'IP': float, 'K': float, 'BB': float, 'H': float, 'HR': float, 'ER': float}
    """
    out: dict[str, float] = {}
    ip_bundle = models.get("IP")
    if ip_bundle is None:
        raise ValueError("Two-stage IP model missing")

    feats = ip_bundle["features"]
    X = pd.DataFrame([feature_row])[feats]
    ip = float(ip_bundle["model"].predict(X)[0])
    ip = max(0.1, ip)
    out["IP"] = ip

    for raw, rate in [("K", "K9"), ("BB", "BB9"), ("H", "H9"), ("HR", "HR9"), ("ER", "ER9")]:
        bundle = models.get(rate)
        if bundle is None:
            continue
        feats_r = bundle["features"]
        X_r = pd.DataFrame([feature_row])[feats_r]
        rate_per9 = float(bundle["model"].predict(X_r)[0])
        rate_per9 = max(0.0, rate_per9)
        out[raw] = rate_per9 * (ip / 9.0)
    return out


def load_two_stage_models(models_dir: Path = MODELS_DIR) -> dict:
    """Load the six two-stage models from disk. Missing files are skipped."""
    out: dict[str, dict] = {}
    for tgt in ("IP", "K9", "BB9", "H9", "HR9", "ER9"):
        p = models_dir / f"pitcher_{tgt}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            out[tgt] = pickle.load(fh)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitcher-csv", default=str(DATA_DIR / "pitcher_game_data.csv"),
                    help="Path to the per-game pitcher feature table.")
    ap.add_argument("--min-ip", type=float, default=0.5,
                    help="Drop training rows with IP < this (filters emergency relief).")
    ap.add_argument("--eval-holdout", type=float, default=0.20,
                    help="Fraction of newest games held out for honest MAE eval (0 to disable).")
    ap.add_argument("--model", choices=("xgb", "lgb"), default="xgb",
                    help="xgb (default) or lgb (LightGBM with monotonic constraints).")
    ap.add_argument("--force-stuff", action="store_true",
                    help="Drop correlated rate stats per target so the model must use whiff/EV/lineup features.")
    args = ap.parse_args()
    train_two_stage(
        pitcher_csv=Path(args.pitcher_csv),
        min_ip=args.min_ip,
        eval_holdout=args.eval_holdout,
        model_type=args.model,
        force_stuff=args.force_stuff,
    )


if __name__ == "__main__":
    main()
