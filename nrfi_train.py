"""
nrfi_train.py
=============
Train a binary NRFI (No Run First Inning) classifier.

Feature priority (highest signal first):
  1. Full-game pitcher stats (K%, BB%, H%, HR%) — stable, large samples
  2. Full-game team offense (runs/game, K%, OBP)
  3. First-inning pitcher rates (noisy but inning-specific)
  4. First-inning team batting rates
  5. Park factor

Target: nrfi = 1 if neither team scored in the 1st inning, 0 otherwise

Output:
  models/nrfi_model.pkl    — best pipeline {"pipeline", "features", "threshold"}
  models/nrfi_metrics.csv  — hold-out metrics
  models/nrfi_importance.csv — feature importances

Usage:
    python nrfi_train.py
    python nrfi_train.py --data data/nrfi_game_data.csv --model-dir models
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — skipping XGB. pip install xgboost")

warnings.filterwarnings("ignore")

WINDOWS_FULL = [5, 10]
WINDOWS_1ST  = [5, 10]


# ── feature catalogue ─────────────────────────────────────────────────────────

def _sp_full_game_features(prefix: str) -> list[str]:
    """Full-game rolling stats for one starter — PRIMARY signal."""
    stats = ["K_rate", "BB_rate", "H_rate", "HR_rate", "IP"]
    feats = []
    for stat in stats:
        feats.append(f"{prefix}fg_{stat}_std")
        for w in WINDOWS_FULL:
            feats.append(f"{prefix}fg_{stat}_last{w}")
    feats.append(f"{prefix}fg_days_rest")
    return feats


def _sp_1st_inning_features(prefix: str) -> list[str]:
    """First-inning rolling stats for one starter — secondary (noisy) signal."""
    stats = ["k1_rate", "bb1_rate", "runs1_per_pa"]
    feats = []
    for stat in stats:
        feats.append(f"{prefix}{stat}_std")
        for w in WINDOWS_1ST:
            feats.append(f"{prefix}{stat}_last{w}")
    return feats


def _team_full_game_features(prefix: str) -> list[str]:
    """Full-game team offensive stats — stable team quality signal."""
    stats = ["fg_k_pct", "fg_ob_pct", "fg_runs_pg"]
    feats = []
    for stat in stats:
        feats.append(f"{prefix}bat_fg_{stat}_std")
        for w in WINDOWS_FULL:
            feats.append(f"{prefix}bat_fg_{stat}_last{w}")
    return feats


def _team_1st_inning_features(prefix: str) -> list[str]:
    """First-inning team batting — inning-specific but small samples."""
    stats = ["ob1_rate", "k1_bat_rate", "runs1_scored"]
    feats = []
    for stat in stats:
        feats.append(f"{prefix}bat_{stat}_std")
        for w in WINDOWS_1ST:
            feats.append(f"{prefix}bat_{stat}_last{w}")
    return feats


CANDIDATE_FEATURES = (
    # Pitcher full-game (highest signal)
    _sp_full_game_features("away_sp_")
    + _sp_full_game_features("home_sp_")
    # Team full-game offense
    + _team_full_game_features("away_")
    + _team_full_game_features("home_")
    # Pitcher 1st-inning (secondary)
    + _sp_1st_inning_features("away_sp_")
    + _sp_1st_inning_features("home_sp_")
    # Team 1st-inning batting (secondary)
    + _team_1st_inning_features("away_")
    + _team_1st_inning_features("home_")
    # Game context
    + ["park_factor"]
)


# ── utilities ─────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame, date_col: str, frac: float = 0.80):
    """Use more data for training since signal is weak — 80/20 split."""
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = sorted(pd.to_datetime(df[date_col]).dt.normalize().unique())
    if len(dates) < 2:
        return df, df.iloc[0:0]
    cut = max(1, int(len(dates) * frac))
    cut = min(cut, len(dates) - 1)
    train_dates = set(dates[:cut])
    train = df[pd.to_datetime(df[date_col]).dt.normalize().isin(train_dates)].copy()
    test  = df[~pd.to_datetime(df[date_col]).dt.normalize().isin(train_dates)].copy()
    return train, test


def select_features(df: pd.DataFrame, candidates: list[str],
                    min_fill_rate: float = 0.30) -> list[str]:
    """Keep features with at least min_fill_rate non-null values."""
    n = len(df)
    min_count = max(30, int(n * min_fill_rate))
    return [f for f in candidates if f in df.columns and df[f].notna().sum() >= min_count]


def feature_importances(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    step = pipeline.named_steps.get("model") or pipeline.named_steps.get("clf")
    if step is None:
        return pd.DataFrame()
    base = step
    for attr in ["base_estimator", "estimator"]:
        if hasattr(base, attr):
            base = getattr(base, attr)
    if hasattr(base, "calibrated_classifiers_"):
        try:
            base = base.calibrated_classifiers_[0].estimator
        except Exception:
            pass
    if hasattr(base, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": base.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    if hasattr(base, "coef_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(base.coef_[0]),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def evaluate(y_true, y_prob, threshold=0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "brier":     brier_score_loss(y_true, y_prob),
        "log_loss":  log_loss(y_true, y_prob),
        "nrfi_rate": float(y_true.mean()),
        "n":         int(len(y_true)),
    }


# ── model builders ────────────────────────────────────────────────────────────

def build_pipeline(model_type: str) -> Pipeline:
    imputer = SimpleImputer(strategy="median")

    if model_type == "lr":
        # Logistic regression with L1 — best baseline for small datasets,
        # auto-selects features, less prone to overfitting than trees
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.05,
            class_weight="balanced", max_iter=1000, random_state=42,
        )
        return Pipeline([("imputer", imputer), ("scaler", StandardScaler()), ("model", clf)])

    if model_type == "rf":
        clf = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=15,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ), cv=3, method="isotonic",
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.7, min_samples_leaf=15, random_state=42,
        )
    elif model_type == "xgb" and HAS_XGB:
        clf = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=300, max_depth=2, learning_rate=0.04,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=2.0,
                eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1,
            ), cv=3, method="isotonic",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("imputer", imputer), ("model", clf)])


# ── training ──────────────────────────────────────────────────────────────────

def train(data_path: str, model_dir: Path):
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} rows from {data_path}")

    date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("No date column found.")
    if "nrfi" not in df.columns:
        raise ValueError("'nrfi' target column not found.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna() & df["nrfi"].notna()].copy()

    train_df, test_df = chronological_split(df, date_col, frac=0.80)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"  NRFI rate — train: {train_df['nrfi'].mean():.3f}  test: {test_df['nrfi'].mean():.3f}")

    feats = select_features(df, CANDIDATE_FEATURES, min_fill_rate=0.25)
    if not feats:
        raise ValueError("No usable features found. Run nrfi_data.py first.")

    # Report feature group coverage
    fg_feats   = [f for f in feats if "fg_" in f]
    sp_feats   = [f for f in feats if "sp_" in f and "fg_" not in f]
    team_feats = [f for f in feats if "bat_" in f and "fg_" not in f]
    print(f"  Features: {len(feats)} total — {len(fg_feats)} full-game, "
          f"{len(sp_feats)} SP-1st-inning, {len(team_feats)} team-1st-inning")

    X_train = train_df[feats].copy()
    y_train = train_df["nrfi"].astype(int)
    X_test  = test_df[feats].copy()
    y_test  = test_df["nrfi"].astype(int)

    # LR first — it's the best baseline for NRFI (weak signal, many correlated features)
    model_types = ["lr", "gb", "rf"] + (["xgb"] if HAS_XGB else [])
    all_metrics = []
    best_pipe   = None
    best_auc    = -np.inf

    for mt in model_types:
        try:
            pipe = build_pipeline(mt)
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, y_prob)
            m["model_type"] = mt
            m["features"]   = len(feats)
            all_metrics.append(m)
            print(f"  {mt.upper():<4}  AUC={m['roc_auc']:.4f}  "
                  f"Acc={m['accuracy']:.4f}  Brier={m['brier']:.4f}")
            if m["roc_auc"] > best_auc:
                best_auc  = m["roc_auc"]
                best_pipe = pipe
                best_mt   = mt
        except Exception as e:
            print(f"  {mt.upper()} failed: {e}")

    if best_pipe is None:
        raise RuntimeError("All models failed to train.")

    print(f"\n  Best model: {best_mt.upper()}  AUC={best_auc:.4f}")

    # Optimal threshold
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
    thresholds  = np.linspace(0.35, 0.65, 61)
    best_thresh = max(
        thresholds,
        key=lambda t: accuracy_score(y_test, (y_prob_best >= t).astype(int))
    )
    print(f"  Optimal threshold={best_thresh:.2f}")

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "pipeline":  best_pipe,
        "features":  feats,
        "threshold": float(best_thresh),
        "model_type": best_mt,
    }
    with open(model_dir / "nrfi_model.pkl", "wb") as f:
        pickle.dump(save_obj, f)

    pd.DataFrame(all_metrics).to_csv(model_dir / "nrfi_metrics.csv", index=False)

    imp = feature_importances(best_pipe, feats)
    if not imp.empty:
        imp.to_csv(model_dir / "nrfi_importance.csv", index=False)
        print("\n  Top 15 features by importance:")
        print(imp.head(15).to_string(index=False))

    print(f"\n  Saved: {model_dir / 'nrfi_model.pkl'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NRFI binary classifier")
    parser.add_argument("--data",      default="data/nrfi_game_data.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    train(data_path=args.data, model_dir=Path(args.model_dir))
    print("\nNRFI training complete.")


if __name__ == "__main__":
    main()
