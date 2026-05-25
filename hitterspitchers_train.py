"""
hitterspitchers_train.py
========================
Train pitcher and hitter models from the built game-level tables.
"""

import argparse
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from model_calibration import CalibratedRegressor, fit_linear_calibration

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — skipping XGBoost models. pip install xgboost")

warnings.filterwarnings("ignore")


LOGIT_EPS = 1e-6


def _logit_transform(y):
    arr = np.asarray(y, dtype=float)
    arr = np.clip(arr, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(arr / (1.0 - arr))


def _inv_logit_transform(z):
    arr = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def clean_hitter_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for c in ["PA", "H", "HR", "BB", "K",
              "h_rate", "hr_rate", "bb_rate", "k_rate"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # PA sanity: 1-7 per game is realistic
    if "PA" in work.columns:
        work = work[work["PA"].between(1, 7, inclusive="both") | work["PA"].isna()].copy()

    # Count sanity: non-negative, at most as large as PA
    if {"H", "PA"}.issubset(work.columns):
        work = work[(work["H"] <= work["PA"]) | work["H"].isna() | work["PA"].isna()].copy()
        work = work[(work["H"] >= 0) | work["H"].isna()].copy()
    if {"HR", "H"}.issubset(work.columns):
        work = work[(work["HR"] <= work["H"]) | work["HR"].isna() | work["H"].isna()].copy()
        work = work[(work["HR"] >= 0) | work["HR"].isna()].copy()
    for c in ["BB", "K"]:
        if c in work.columns and "PA" in work.columns:
            work = work[(work[c] <= work["PA"]) | work[c].isna() | work["PA"].isna()].copy()
            work = work[(work[c] >= 0) | work[c].isna()].copy()

    # Rate sanity (kept for feature columns that are still present)
    for c in ["h_rate", "hr_rate", "bb_rate", "k_rate"]:
        if c in work.columns:
            work = work[work[c].between(0, 1, inclusive="both") | work[c].isna()].copy()

    if {"hr_rate", "h_rate"}.issubset(work.columns):
        work = work[(work["hr_rate"] <= work["h_rate"]) | work["hr_rate"].isna() | work["h_rate"].isna()].copy()

    if {"h_rate", "bb_rate"}.issubset(work.columns):
        on_base = pd.to_numeric(work["h_rate"], errors="coerce") + pd.to_numeric(work["bb_rate"], errors="coerce")
        work = work[(on_base <= 0.75) | on_base.isna()].copy()

    return work


def report_hitter_feature_coverage(df: pd.DataFrame):
    cols = ["opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_hr_rate", "opp_sp_h_rate"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return
    coverage = float(df[present].notna().all(axis=1).mean()) if len(df) else float("nan")
    print(f"  Hitter rows with opponent-starter core features: {coverage:.1%}")




# Predict counting stats directly — reduces compounding variance vs rate→count
PITCHER_TARGETS = ["K", "BB", "HR", "H", "IP"]
HITTER_TARGETS  = ["H", "HR", "BB", "K", "PA"]

PITCHER_FEATURES = [
    # ── season-to-date rates (stable baselines) ──────────────────────────
    "K_rate_std", "BB_rate_std", "HR_rate_std", "H_rate_std",
    "IP_std", "avg_velocity_std", "avg_spin_std",

    # ── rate rolling windows ──────────────────────────────────────────────
    "K_rate_last5",  "BB_rate_last5",  "HR_rate_last5",  "H_rate_last5",  "IP_last5",
    "K_rate_last7",  "BB_rate_last7",  "HR_rate_last7",  "H_rate_last7",  "IP_last7",
    "K_rate_last10", "BB_rate_last10", "HR_rate_last10", "H_rate_last10", "IP_last10",
    "K_rate_last14", "BB_rate_last14", "HR_rate_last14", "H_rate_last14", "IP_last14",
    "K_rate_last21", "BB_rate_last21", "HR_rate_last21", "H_rate_last21", "IP_last21",
    "K_rate_last30", "BB_rate_last30", "HR_rate_last30", "H_rate_last30", "IP_last30",
    "avg_velocity_last5", "avg_velocity_last7", "avg_velocity_last10",

    # ── RAW COUNT rolling windows (direct target history) ─────────────────
    "K_std",   "K_last5",  "K_last7",  "K_last10", "K_last14", "K_last21", "K_last30",
    "BB_std",  "BB_last5", "BB_last7", "BB_last10","BB_last14","BB_last21","BB_last30",
    "HR_std",  "HR_last5", "HR_last7", "HR_last10","HR_last14","HR_last21","HR_last30",
    "H_std",   "H_last5",  "H_last7",  "H_last10", "H_last14", "H_last21", "H_last30",

    # ── workload / usage ──────────────────────────────────────────────────
    "BF_std", "outs_std", "pitches_std",
    "BF_last3", "outs_last3", "pitches_last3",
    "BF_last5", "outs_last5", "pitches_last5",
    "BF_last7", "outs_last7", "pitches_last7",
    "BF_last10", "outs_last10", "pitches_last10",
    "BF_last14", "outs_last14", "pitches_last14",
    "BF_last21", "outs_last21", "pitches_last21",
    "BF_last30", "outs_last30", "pitches_last30",

    "BF_per_IP_std", "pitches_per_BF_std", "pitches_per_IP_std",
    "BF_per_IP_last5", "pitches_per_BF_last5", "pitches_per_IP_last5",
    "BF_per_IP_last10", "pitches_per_BF_last10", "pitches_per_IP_last10",

    "max_ip_last5", "max_ip_last10",
    "days_rest",
    "starter_pct_last5",
    "starter_pct_last10",

    # ── platoon splits ────────────────────────────────────────────────────
    # NO-WINDOW platoon splits removed — they hold the CURRENT game's per-hand
    # rate (leakage; see diagnose_leakage.py — corr ~0.5-0.6 with the target,
    # and they're overwritten with trailing values at serving so the model
    # falls apart live). Only the trailing _last5/_last10/_std windows are kept.
    "pitcher_k_rate_vs_hand_last5_R",  "pitcher_bb_rate_vs_hand_last5_R",
    "pitcher_hr_rate_vs_hand_last5_R", "pitcher_h_rate_vs_hand_last5_R",
    "pitcher_k_rate_vs_hand_last5_L",  "pitcher_bb_rate_vs_hand_last5_L",
    "pitcher_hr_rate_vs_hand_last5_L", "pitcher_h_rate_vs_hand_last5_L",

    "pitcher_k_rate_vs_hand_last10_R",  "pitcher_bb_rate_vs_hand_last10_R",
    "pitcher_hr_rate_vs_hand_last10_R", "pitcher_h_rate_vs_hand_last10_R",
    "pitcher_k_rate_vs_hand_last10_L",  "pitcher_bb_rate_vs_hand_last10_L",
    "pitcher_hr_rate_vs_hand_last10_L", "pitcher_h_rate_vs_hand_last10_L",

    "pitcher_k_rate_vs_hand_std_R",  "pitcher_bb_rate_vs_hand_std_R",
    "pitcher_hr_rate_vs_hand_std_R", "pitcher_h_rate_vs_hand_std_R",
    "pitcher_k_rate_vs_hand_std_L",  "pitcher_bb_rate_vs_hand_std_L",
    "pitcher_hr_rate_vs_hand_std_L", "pitcher_h_rate_vs_hand_std_L",

    # ── opponent team context ─────────────────────────────────────────────
    # NO-WINDOW team context removed — it's the CURRENT game's realized
    # opponent rate (leakage; corr ~0.7-0.8 with the target). Trailing kept.
    "team_k_rate_vs_hand_last5",  "team_bb_rate_vs_hand_last5",
    "team_hr_rate_vs_hand_last5", "team_h_rate_vs_hand_last5",
    "team_k_rate_vs_hand_last7",  "team_bb_rate_vs_hand_last7",
    "team_hr_rate_vs_hand_last7", "team_h_rate_vs_hand_last7",
    "team_k_rate_vs_hand_last10", "team_bb_rate_vs_hand_last10",
    "team_hr_rate_vs_hand_last10","team_h_rate_vs_hand_last10",
    "team_k_rate_vs_hand_last14", "team_bb_rate_vs_hand_last14",
    "team_hr_rate_vs_hand_last14","team_h_rate_vs_hand_last14",
    "team_k_rate_vs_hand_std",    "team_bb_rate_vs_hand_std",
    "team_hr_rate_vs_hand_std",   "team_h_rate_vs_hand_std",

    # ── true-talent (empirical-Bayes shrunk) + log5 lineup matchup ──
    # Honest, leakage-free signal from enrich_truetalent.py: the pitcher's
    # sample-size-regressed rates, the opposing lineup's shrunk rates, and the
    # two combined via the log5 odds-ratio. This is the legitimate version of
    # the opponent signal the leaked team_*_rate_vs_hand columns were faking.
    "p_tt_k", "p_tt_bb", "p_tt_h", "p_tt_hr",
    "lineup_tt_k", "lineup_tt_bb", "lineup_tt_h", "lineup_tt_hr",
    "matchup_k", "matchup_bb", "matchup_h", "matchup_hr",

    "park_factor",
]

HITTER_FEATURES = [
    # ── season-to-date rates (stable baselines) ──────────────────────────
    "h_rate_std", "hr_rate_std", "bb_rate_std", "k_rate_std",
    "PA_std", "avg_EV_std", "max_EV_std", "avg_LA_std", "avg_direction_std",

    # ── rate rolling windows ──────────────────────────────────────────────
    "h_rate_last5",  "hr_rate_last5",  "bb_rate_last5",  "k_rate_last5",  "PA_last5",
    "h_rate_last7",  "hr_rate_last7",  "bb_rate_last7",  "k_rate_last7",  "PA_last7",
    "h_rate_last10", "hr_rate_last10", "bb_rate_last10", "k_rate_last10", "PA_last10",
    "h_rate_last14", "hr_rate_last14", "bb_rate_last14", "k_rate_last14", "PA_last14",
    "h_rate_last21", "hr_rate_last21", "bb_rate_last21", "k_rate_last21", "PA_last21",
    "h_rate_last30", "hr_rate_last30", "bb_rate_last30", "k_rate_last30", "PA_last30",
    "avg_EV_last5", "max_EV_last5", "avg_LA_last5",
    "avg_EV_last10", "max_EV_last10", "avg_LA_last10",

    # ── RAW COUNT rolling windows (direct target history) ─────────────────
    "H_std",  "H_last5",  "H_last7",  "H_last10", "H_last14", "H_last21", "H_last30",
    "HR_std", "HR_last5", "HR_last7", "HR_last10","HR_last14","HR_last21","HR_last30",
    "BB_std", "BB_last5", "BB_last7", "BB_last10","BB_last14","BB_last21","BB_last30",
    "K_std",  "K_last5",  "K_last7",  "K_last10", "K_last14", "K_last21", "K_last30",

    # ── PA convenience features ───────────────────────────────────────────
    "PA_last3",
    "max_hr_rate_last10", "max_h_rate_last10",
    "days_since_game",

    # ── batted-ball quality ───────────────────────────────────────────────
    "barrel_proxy_std", "hard_hit_proxy_std", "sweet_spot_proxy_std", "blast_proxy_std",
    "ev_la_interaction_std", "ev_spread_std",
    "times_on_base_rate_std", "xbh_proxy_rate_std",

    "barrel_proxy_last5", "hard_hit_proxy_last5", "sweet_spot_proxy_last5", "blast_proxy_last5",
    "ev_la_interaction_last5", "ev_spread_last5",
    "times_on_base_rate_last5", "xbh_proxy_rate_last5",

    "barrel_proxy_last10", "hard_hit_proxy_last10", "sweet_spot_proxy_last10", "blast_proxy_last10",
    "ev_la_interaction_last10", "ev_spread_last10",
    "times_on_base_rate_last10", "xbh_proxy_rate_last10",

    # ── platoon splits ────────────────────────────────────────────────────
    # NO-WINDOW platoon splits removed — current game's per-hand rate
    # (leakage, same pattern as the pitcher side). Trailing windows kept.
    "hitter_h_rate_vs_hand_last5_R",  "hitter_hr_rate_vs_hand_last5_R",
    "hitter_bb_rate_vs_hand_last5_R", "hitter_k_rate_vs_hand_last5_R",
    "hitter_h_rate_vs_hand_last5_L",  "hitter_hr_rate_vs_hand_last5_L",
    "hitter_bb_rate_vs_hand_last5_L", "hitter_k_rate_vs_hand_last5_L",

    "hitter_h_rate_vs_hand_last10_R",  "hitter_hr_rate_vs_hand_last10_R",
    "hitter_bb_rate_vs_hand_last10_R", "hitter_k_rate_vs_hand_last10_R",
    "hitter_h_rate_vs_hand_last10_L",  "hitter_hr_rate_vs_hand_last10_L",
    "hitter_bb_rate_vs_hand_last10_L", "hitter_k_rate_vs_hand_last10_L",

    "hitter_h_rate_vs_hand_std_R",  "hitter_hr_rate_vs_hand_std_R",
    "hitter_bb_rate_vs_hand_std_R", "hitter_k_rate_vs_hand_std_R",
    "hitter_h_rate_vs_hand_std_L",  "hitter_hr_rate_vs_hand_std_L",
    "hitter_bb_rate_vs_hand_std_L", "hitter_k_rate_vs_hand_std_L",

    # ── opponent pitching team context ────────────────────────────────────
    # NO-WINDOW opponent-pitching context removed — current game's realized
    # rate (leakage). Trailing versions kept.
    "team_allowed_k_rate_vs_hand_last5",  "team_allowed_bb_rate_vs_hand_last5",
    "team_allowed_hr_rate_vs_hand_last5", "team_allowed_h_rate_vs_hand_last5",
    "team_allowed_k_rate_vs_hand_last7",  "team_allowed_bb_rate_vs_hand_last7",
    "team_allowed_hr_rate_vs_hand_last7", "team_allowed_h_rate_vs_hand_last7",
    "team_allowed_k_rate_vs_hand_last10", "team_allowed_bb_rate_vs_hand_last10",
    "team_allowed_hr_rate_vs_hand_last10","team_allowed_h_rate_vs_hand_last10",
    "team_allowed_k_rate_vs_hand_last14", "team_allowed_bb_rate_vs_hand_last14",
    "team_allowed_hr_rate_vs_hand_last14","team_allowed_h_rate_vs_hand_last14",
    "team_allowed_k_rate_vs_hand_std",    "team_allowed_bb_rate_vs_hand_std",
    "team_allowed_hr_rate_vs_hand_std",   "team_allowed_h_rate_vs_hand_std",

    # ── opposing starter ──────────────────────────────────────────────────
    "opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_hr_rate", "opp_sp_h_rate", "opp_sp_ip",
    "opp_sp_k_rate_last5",  "opp_sp_bb_rate_last5",  "opp_sp_hr_rate_last5",
    "opp_sp_h_rate_last5",  "opp_sp_ip_last5",
    "opp_sp_k_rate_last10", "opp_sp_bb_rate_last10", "opp_sp_hr_rate_last10",
    "opp_sp_h_rate_last10", "opp_sp_ip_last10",
    "opp_sp_k_rate_std",    "opp_sp_bb_rate_std",    "opp_sp_hr_rate_std",
    "opp_sp_h_rate_std",    "opp_sp_ip_std",

    # ── hitter true-talent (empirical-Bayes shrunk per-PA rates) ──
    "h_tt_k", "h_tt_bb", "h_tt_h", "h_tt_hr",

    "park_factor",
]

TRAIN_FRAC = 0.75
CV_FOLDS = 5
RANDOM_STATE = 42

# Defaults for the new tuning + calibration behaviour. Both can be toggled
# from the CLI (--no-tune / --no-calibrate / --tune-iter).
TUNE_ITER_DEFAULT = 24      # randomized-search samples per (target, xgb)
CALIB_FRAC = 0.15           # tail of the train window held out to fit a + b

# Randomized-search space for XGBoost. Centred on the regularised defaults the
# project already uses (shallow trees, strong reg) — with ~1-5k pitcher rows
# and ~45k hitter rows, deep/under-regularised trees overfit hot/cold streaks,
# so the grid deliberately keeps depth low and reg high.
XGB_TUNE_GRID = {
    "n_estimators":    [300, 400, 600, 800],
    "max_depth":       [2, 3, 4, 5],
    "learning_rate":   [0.02, 0.03, 0.04, 0.05],
    "subsample":       [0.7, 0.8, 0.9],
    "colsample_bytree":[0.6, 0.7, 0.8],
    "reg_alpha":       [0.0, 0.5, 1.0, 2.0],
    "reg_lambda":      [1.0, 2.0, 3.0, 5.0],
    "min_child_weight":[5, 10, 20, 30],
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _tune_xgb(X_train, y_train, target: str, n_iter: int = TUNE_ITER_DEFAULT,
              seed: int = RANDOM_STATE) -> dict:
    """Randomized hyperparameter search for an XGB model on one target.

    Uses a CHRONOLOGICAL inner split of the (already date-sorted) training
    window — the last 20% of train rows act as the validation fold. Random
    K-fold is intentionally avoided: these are game logs, and shuffling lets
    a row's future leak into its own training set, producing optimistic params
    that fall apart out-of-sample.

    Returns the best parameter dict (empty if tuning can't run, in which case
    build_sklearn_model falls back to its hand-tuned defaults).
    """
    if not (HAS_XGB and target is not None):
        return {}
    n = len(X_train)
    cut = int(n * 0.8)
    if cut < 60 or (n - cut) < 25:
        return {}

    X_tr, X_val = X_train.iloc[:cut], X_train.iloc[cut:]
    y_tr, y_val = y_train.iloc[:cut], y_train.iloc[cut:]

    rng = random.Random(seed)
    keys = list(XGB_TUNE_GRID)
    best_params: dict = {}
    best_mae = float("inf")
    for _ in range(max(1, n_iter)):
        params = {k: rng.choice(XGB_TUNE_GRID[k]) for k in keys}
        try:
            m = build_sklearn_model("xgb", target_name=target, params=params)
            m.fit(X_tr, y_tr)
            mae = float(mean_absolute_error(y_val, m.predict(X_val)))
        except Exception:
            continue
        if mae < best_mae:
            best_mae, best_params = mae, params
    if best_params:
        print(f"      [tune {target}] inner-val MAE={best_mae:.4f}  "
              f"depth={best_params['max_depth']} n_est={best_params['n_estimators']} "
              f"lr={best_params['learning_rate']} reg_l={best_params['reg_lambda']}")
    return best_params


def chronological_split(df: pd.DataFrame, date_col: str, frac: float = TRAIN_FRAC):
    work = df.copy()
    work = work[work[date_col].notna()].sort_values(date_col).reset_index(drop=True)

    unique_dates = sorted(pd.to_datetime(work[date_col]).dt.normalize().unique())
    if len(unique_dates) < 2:
        return work.copy(), work.iloc[0:0].copy()

    cutoff_idx = max(1, int(len(unique_dates) * frac))
    cutoff_idx = min(cutoff_idx, len(unique_dates) - 1)

    train_dates = set(unique_dates[:cutoff_idx])
    test_dates = set(unique_dates[cutoff_idx:])

    train_df = work[pd.to_datetime(work[date_col]).dt.normalize().isin(train_dates)].copy()
    test_df = work[pd.to_datetime(work[date_col]).dt.normalize().isin(test_dates)].copy()
    return train_df, test_df


def select_features(df: pd.DataFrame, candidate_features: list) -> list:
    present = [f for f in candidate_features if f in df.columns]
    good = [f for f in present if df[f].notna().sum() > 50]
    return good


def build_sklearn_model(model_type: str = "rf", target_name: str | None = None,
                        params: dict | None = None) -> Pipeline:
    """Build an (imputer → model) pipeline, optionally target-transformed.

    `params`, when given, overrides the default XGBoost hyperparameters — this
    is how the randomized search in `_tune_xgb` evaluates candidate configs and
    how the winning config is rebuilt for the final fit. It's ignored for rf/nn.
    """
    imputer = SimpleImputer(strategy="median")

    if model_type == "rf":
        if target_name == "PA":
            # PA is relatively smooth — slightly deeper tree is OK
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=20,
                max_features=0.6,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            # Count targets (H, HR, BB, K, IP): stay shallow to avoid
            # memorising hot/cold streaks from short rolling windows.
            # With ~1-5k training rows, max_depth > 5 leads to severe overfit.
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=4,
                min_samples_leaf=25,
                max_features=0.5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        base = Pipeline([("imputer", imputer), ("model", model)])

    elif model_type == "xgb" and HAS_XGB:
        if target_name == "PA":
            xgb_kwargs = dict(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_weight=20,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            )
        else:
            # Strong regularisation: forces regression toward mean,
            # prevents learning "hot streak → predict high counts".
            xgb_kwargs = dict(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.65,
                reg_alpha=1.0,
                reg_lambda=3.0,
                min_child_weight=30,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            )
        # Tuned hyperparameters (from _tune_xgb) override the hand-picked
        # defaults above. random_state / verbosity / n_jobs are never tuned.
        if params:
            xgb_kwargs.update({k: v for k, v in params.items()
                               if k not in ("random_state", "verbosity", "n_jobs")})
        model = XGBRegressor(**xgb_kwargs)
        base = Pipeline([("imputer", imputer), ("model", model)])

    elif model_type == "nn":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
        base = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Rate targets (legacy, kept for backward compat if rates are ever needed)
    if target_name in {"h_rate", "hr_rate", "bb_rate", "k_rate",
                       "K_rate", "BB_rate", "HR_rate", "H_rate"}:
        return TransformedTargetRegressor(
            regressor=base,
            func=_logit_transform,
            inverse_func=_inv_logit_transform,
            check_inverse=False,
        )

    # Counting-stat targets — non-negative integers, log1p keeps them ≥ 0
    if target_name in {"H", "HR", "BB", "K",           # hitter counts
                       "PA",                            # plate appearances
                       "IP",                            # innings pitched
                       "BF", "outs", "pitches"}:        # pitcher counts
        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    return base


def feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    fitted = pipeline
    if isinstance(fitted, CalibratedRegressor):
        fitted = fitted.base
    if isinstance(fitted, TransformedTargetRegressor):
        fitted = fitted.regressor_

    if hasattr(fitted, "named_steps"):
        model = fitted.named_steps.get("model")
    else:
        model = fitted

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame()


def validate_pitcher_training_data(df: pd.DataFrame):
    work = df.copy()

    if "IP" not in work.columns:
        raise ValueError("Pitcher training data is missing IP column.")

    work["IP"] = pd.to_numeric(work["IP"], errors="coerce")
    mean_ip = float(work["IP"].mean())
    median_ip = float(work["IP"].median())

    print("\nPitcher training data check:")
    print(f"  Rows:      {len(work):,}")
    print(f"  Mean IP:   {mean_ip:.3f}")
    print(f"  Median IP: {median_ip:.3f}")

    if "is_actual_starter" in work.columns:
        starter_rate = float(pd.to_numeric(work["is_actual_starter"], errors="coerce").fillna(0).mean())
        print(f"  Starter flag mean: {starter_rate:.3f}")

    if mean_ip < 3.5:
        raise ValueError(
            f"Pitcher training data still looks relief-heavy (mean IP={mean_ip:.3f}). "
            "Rebuild pitcher_game_data.csv before training."
        )


def train_one_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list,
    target: str,
    model_type: str = "rf",
    tune: bool = False,
    n_iter: int = TUNE_ITER_DEFAULT,
    calibrate: bool = False,
    preset_params: dict | None = None,
) -> dict:
    if target not in train_df.columns:
        print(f"    [skip] '{target}' not in data")
        return {}

    feats = select_features(train_df, features)
    if not feats:
        print(f"    [skip] No usable features for '{target}'")
        return {}

    train_work = train_df[feats + [target]].copy()
    test_work = test_df[feats + [target]].copy()

    train_work = train_work[train_work[target].notna()].copy()
    test_work = test_work[test_work[target].notna()].copy()

    if train_work.empty or test_work.empty:
        print(f"    [skip] '{target}' has empty train/test after dropping missing targets")
        return {}

    X_train = train_work[feats]
    y_train = train_work[target]

    X_test = test_work[feats]
    y_test = test_work[target]

    # ── 1. Hyperparameter selection (xgb only) ───────────────────────────
    # Priority: caller-supplied preset_params (e.g. the daily refit reusing
    # hyperparameters tuned in an earlier offline run) → otherwise a fresh
    # randomized search when tune=True → otherwise the hand-tuned defaults.
    # train_df arrives chronologically sorted from chronological_split, so the
    # inner validation fold inside _tune_xgb is a genuine "future" hold-out.
    best_params: dict = {}
    if model_type == "xgb" and HAS_XGB:
        if preset_params:
            best_params = dict(preset_params)
            print(f"      [preset {target}] reusing tuned params "
                  f"(depth={best_params.get('max_depth')}, n_est={best_params.get('n_estimators')})")
        elif tune:
            best_params = _tune_xgb(X_train, y_train, target, n_iter=n_iter)

    # ── 2. Prediction calibration ────────────────────────────────────────
    # Fit a + b on the most-recent CALIB_FRAC of the training window, using a
    # model trained only on the earlier rows (so the calibration slice is
    # unseen). The (a, b) is then baked into the final model via
    # CalibratedRegressor. Falls back to identity (0, 1) if data is thin.
    calib_a, calib_b = 0.0, 1.0
    floor = 0.0  # all targets (counts / IP / PA) are non-negative
    if calibrate and len(X_train) >= 200:
        k = int(len(X_train) * (1.0 - CALIB_FRAC))
        if k >= 60 and (len(X_train) - k) >= 30:
            cal_fit = build_sklearn_model(model_type, target_name=target, params=best_params)
            cal_fit.fit(X_train.iloc[:k], y_train.iloc[:k])
            cal_pred = cal_fit.predict(X_train.iloc[k:])
            calib_a, calib_b = fit_linear_calibration(y_train.iloc[k:], cal_pred)

    # ── 3. Final fit on the full training window, then wrap with calibration
    pipe = build_sklearn_model(model_type, target_name=target, params=best_params)
    pipe.fit(X_train, y_train)
    model = CalibratedRegressor(pipe, a=calib_a, b=calib_b, floor=floor)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    # Raw (pre-calibration) test MAE, for transparency in the log
    test_pred_raw = np.clip(np.asarray(pipe.predict(X_test), dtype=float), floor, None)
    test_mae_raw = float(mean_absolute_error(y_test, test_pred_raw))

    cv_n = min(CV_FOLDS, len(X_train))
    if cv_n >= 3:
        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv_n,
            scoring="neg_root_mean_squared_error"
        )
        cv_rmse_mean = float(-cv_scores.mean())
        cv_rmse_std = float(cv_scores.std())
    else:
        cv_rmse_mean = np.nan
        cv_rmse_std = np.nan

    calibrated = (calib_a, calib_b) != (0.0, 1.0)
    metrics = {
        "target": target,
        "model": model_type,
        "train_rmse": rmse(y_train, train_pred),
        "test_rmse": rmse(y_test, test_pred),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_mae_raw": test_mae_raw,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": feats,
        "tuned": bool(best_params),
        "calibrated": calibrated,
        "calib_a": calib_a,
        "calib_b": calib_b,
        "best_params": best_params,
        "test_actual_mean": float(np.mean(y_test)),
        "test_pred_mean": float(np.mean(test_pred)),
        "test_actual_std": float(np.std(y_test)),
        "test_pred_std": float(np.std(test_pred)),
    }

    imp_df = feature_importance(pipe, feats)

    cal_note = ""
    if calibrated:
        cal_note = f"  cal:{test_mae_raw:.4f}→{metrics['test_mae']:.4f}(a={calib_a:.2f},b={calib_b:.2f})"
    print(
        f"    {target:<22} | {model_type.upper():<3} | "
        f"test RMSE={metrics['test_rmse']:.4f}  "
        f"MAE={metrics['test_mae']:.4f}  "
        f"cv={metrics['cv_rmse_mean']:.4f}±{metrics['cv_rmse_std']:.4f}  "
        f"pred_mean={metrics['test_pred_mean']:.4f}"
        f"{'  [tuned]' if best_params else ''}{cal_note}"
    )

    return {
        "pipeline": model,          # calibration baked in → scores correctly at inference
        "raw_pipeline": pipe,       # uncalibrated, kept for diagnostics
        "metrics": metrics,
        "importance": imp_df,
        "features": feats,
        "calibration": {"a": calib_a, "b": calib_b},
    }


def train_pitcher_models(df: pd.DataFrame, model_dir: Path,
                         tune: bool = False, n_iter: int = TUNE_ITER_DEFAULT,
                         calibrate: bool = False,
                         preset_params: dict | None = None) -> dict:
    print("\n" + "=" * 60)
    print("PITCHER MODELS")
    print("=" * 60)

    date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    if not date_col:
        raise ValueError("No date column found in pitcher data")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "is_actual_starter" in df.columns:
        df = df[pd.to_numeric(df["is_actual_starter"], errors="coerce").fillna(0) == 1].copy()

    train_df, test_df = chronological_split(df, date_col)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    all_results = {}
    all_metrics = []
    model_types = ["rf"] + (["xgb"] if HAS_XGB else []) + ["nn"]

    for target in PITCHER_TARGETS:
        print(f"\n  Target: {target}")
        best = None

        preset = (preset_params or {}).get(target)
        for mt in model_types:
            res = train_one_target(train_df, test_df, PITCHER_FEATURES, target, mt,
                                   tune=tune, n_iter=n_iter, calibrate=calibrate,
                                   preset_params=preset)
            if not res:
                continue

            all_metrics.append(res["metrics"])

            if best is None or res["metrics"]["test_rmse"] < best["metrics"]["test_rmse"]:
                best = res

        if best:
            all_results[target] = best
            with open(model_dir / f"pitcher_{target}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "pipeline": best["pipeline"],
                        "features": best["features"],
                        "model_type": best["metrics"].get("model"),
                        "calibration": best.get("calibration"),
                        "best_params": best["metrics"].get("best_params"),
                    },
                    f,
                )

    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        metrics_df.to_csv(model_dir / "pitcher_metrics.csv", index=False)

    imp_frames = []
    for target, res in all_results.items():
        if not res["importance"].empty:
            tmp = res["importance"].copy()
            tmp["target"] = target
            imp_frames.append(tmp)
    if imp_frames:
        pd.concat(imp_frames, ignore_index=True).to_csv(model_dir / "pitcher_importance.csv", index=False)

    print(f"\n  Pitcher models saved to {model_dir}")
    return all_results


def train_hitter_models(df: pd.DataFrame, model_dir: Path,
                        tune: bool = False, n_iter: int = TUNE_ITER_DEFAULT,
                        calibrate: bool = False,
                        preset_params: dict | None = None) -> dict:
    print("\n" + "=" * 60)
    print("HITTER MODELS")
    print("=" * 60)

    date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    if not date_col:
        raise ValueError("No date column found in hitter data")

    df = clean_hitter_training_rows(df)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    train_df, test_df = chronological_split(df, date_col)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    all_results = {}
    all_metrics = []
    model_types = ["rf"] + (["xgb"] if HAS_XGB else []) + ["nn"]

    for target in HITTER_TARGETS:
        print(f"\n  Target: {target}")
        best = None

        preset = (preset_params or {}).get(target)
        for mt in model_types:
            res = train_one_target(train_df, test_df, HITTER_FEATURES, target, mt,
                                   tune=tune, n_iter=n_iter, calibrate=calibrate,
                                   preset_params=preset)
            if not res:
                continue

            all_metrics.append(res["metrics"])

            if best is None or res["metrics"]["test_rmse"] < best["metrics"]["test_rmse"]:
                best = res

        if best:
            all_results[target] = best
            with open(model_dir / f"hitter_{target}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "pipeline": best["pipeline"],
                        "features": best["features"],
                        "model_type": best["metrics"].get("model"),
                        "calibration": best.get("calibration"),
                        "best_params": best["metrics"].get("best_params"),
                    },
                    f,
                )

    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        metrics_df.to_csv(model_dir / "hitter_metrics.csv", index=False)

    imp_frames = []
    for target, res in all_results.items():
        if not res["importance"].empty:
            tmp = res["importance"].copy()
            tmp["target"] = target
            imp_frames.append(tmp)
    if imp_frames:
        pd.concat(imp_frames, ignore_index=True).to_csv(model_dir / "hitter_importance.csv", index=False)

    print(f"\n  Hitter models saved to {model_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train MLB hitter + pitcher projection models")
    parser.add_argument("--pitcher-data", default="data/pitcher_game_data.csv", help="Pitcher game-level CSV")
    parser.add_argument("--hitter-data", default="data/hitter_game_data.csv", help="Hitter game-level CSV")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--no-tune", action="store_true",
                        help="Disable XGBoost hyperparameter search (on by default).")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Disable post-hoc linear prediction calibration (on by default).")
    parser.add_argument("--tune-iter", type=int, default=TUNE_ITER_DEFAULT,
                        help="Randomized-search samples per tuned target.")
    args = parser.parse_args()

    tune = not args.no_tune
    calibrate = not args.no_calibrate
    print(f"Tuning: {'ON' if tune else 'OFF'}  (iter={args.tune_iter})   "
          f"Calibration: {'ON' if calibrate else 'OFF'}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pitcher_df = pd.read_csv(args.pitcher_data, low_memory=False)
    hitter_df = pd.read_csv(args.hitter_data, low_memory=False)

    validate_pitcher_training_data(pitcher_df)

    train_pitcher_models(pitcher_df, model_dir,
                         tune=tune, n_iter=args.tune_iter, calibrate=calibrate)
    train_hitter_models(hitter_df, model_dir,
                        tune=tune, n_iter=args.tune_iter, calibrate=calibrate)

    print("\nDone.")


if __name__ == "__main__":
    main()