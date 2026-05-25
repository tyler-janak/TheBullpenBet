"""
enrich_truetalent.py
=====================
Adds two families of LEAKAGE-FREE features that are the highest-leverage,
honest signal for pitcher prop projection:

  1. Empirical-Bayes "true-talent" rates  (#1 from the roadmap)
     Each pitcher's and hitter's K/BB/H/HR rate, regressed toward the league
     mean in proportion to sample size:

         tt_rate = (cum_events + k * league_rate) / (cum_opportunities + k)

     where `cum_*` is the player's running total over ALL PRIOR games only
     (computed with .shift(1) so the current game is excluded), and `k` is the
     stabilization constant for that stat (≈ the opportunities at which the
     rate is half-regressed). This turns noisy 5-start rolling rates into
     stable, predictive inputs — the classic "Marcel beats fancy models" win.

  2. log5 lineup matchup  (#2 from the roadmap)
     For each pitcher-game, the opposing lineup's mean true-talent rate
     (using each hitter's PRE-GAME shrunk rate, so no leakage) is combined with
     the pitcher's own true-talent rate via the log5 / odds-ratio formula:

         matchup = (p*b/lg) / ( p*b/lg + (1-p)(1-b)/(1-lg) )

     This is the honest version of the opponent signal — the same idea as the
     leaked `team_k_rate_vs_hand`, but built from each side's trailing talent
     instead of the current game's realized rate.

Columns added
-------------
hitter_game_data.csv :  h_tt_k, h_tt_bb, h_tt_h, h_tt_hr      (per PA, trailing)
pitcher_game_data.csv:  p_tt_k, p_tt_bb, p_tt_h, p_tt_hr      (per BF, trailing)
                        lineup_tt_k, lineup_tt_bb, lineup_tt_h, lineup_tt_hr
                        matchup_k,  matchup_bb,  matchup_h,  matchup_hr

When this runs
--------------
After the per-game tables are built and AFTER enrich_team_features /
enrich_lineup_features. Idempotent: prior columns are dropped first.

Usage
-----
    python enrich_truetalent.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# Stabilization constants: opportunities (PA/BF) at which the rate is ~half
# regressed toward league. Standard sabermetric ballpark values.
STAB = {"k": 70.0, "bb": 120.0, "h": 60.0, "hr": 300.0}

HITTER_TT_COLS  = ["h_tt_k", "h_tt_bb", "h_tt_h", "h_tt_hr"]
PITCHER_TT_COLS = ["p_tt_k", "p_tt_bb", "p_tt_h", "p_tt_hr"]
LINEUP_COLS     = ["lineup_tt_k", "lineup_tt_bb", "lineup_tt_h", "lineup_tt_hr"]
MATCHUP_COLS    = ["matchup_k", "matchup_bb", "matchup_h", "matchup_hr"]

EPS = 1e-4


def _first_col(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _cum_shrunk(df: pd.DataFrame, group_col: str, num_col: str,
                den_col: str, k_const: float, league_rate: float) -> pd.Series:
    """Empirical-Bayes shrunk rate using ONLY prior games (shift(1))."""
    g = df.groupby(group_col, sort=False)
    cum_num = g[num_col].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().sum())
    cum_den = g[den_col].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().sum())
    cum_num = cum_num.fillna(0.0)
    cum_den = cum_den.fillna(0.0)
    rate = (cum_num + k_const * league_rate) / (cum_den + k_const)
    return rate.clip(EPS, 1 - EPS)


def _log5(p: pd.Series | np.ndarray, b: pd.Series | np.ndarray, lg: float):
    """Combine pitcher rate p and batter rate b relative to league lg."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    b = np.clip(np.asarray(b, dtype=float), EPS, 1 - EPS)
    lg = min(max(float(lg), EPS), 1 - EPS)
    num = (p * b) / lg
    den = num + ((1 - p) * (1 - b)) / (1 - lg)
    out = np.where(den > 0, num / den, lg)
    return np.clip(out, EPS, 1 - EPS)


def _league_rate(df: pd.DataFrame, num_col: str, den_col: str) -> float:
    n = pd.to_numeric(df.get(num_col), errors="coerce").sum()
    d = pd.to_numeric(df.get(den_col), errors="coerce").sum()
    return float(n / d) if d and d > 0 else 0.1


# ---------------------------------------------------------------------------
def add_hitter_truetalent(hdf: pd.DataFrame, leagues: dict) -> pd.DataFrame:
    out = hdf.drop(columns=[c for c in HITTER_TT_COLS if c in hdf.columns], errors="ignore").copy()
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce")
    bat_col = _first_col(out, ("batter", "batter_id", "mlb_id", "player_id"))
    if bat_col is None or "PA" not in out.columns:
        print("⚠️  hitter true-talent: missing batter id or PA column — skipping")
        return out
    out = out.sort_values([bat_col, "game_date"]).reset_index(drop=True)
    for stat, raw in [("k", "K"), ("bb", "BB"), ("h", "H"), ("hr", "HR")]:
        if raw not in out.columns:
            continue
        out[f"h_tt_{stat}"] = _cum_shrunk(out, bat_col, raw, "PA",
                                          STAB[stat], leagues[stat])
    return out


def add_pitcher_truetalent(pdf: pd.DataFrame, leagues: dict) -> pd.DataFrame:
    out = pdf.drop(columns=[c for c in PITCHER_TT_COLS if c in pdf.columns], errors="ignore").copy()
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce")
    p_col = _first_col(out, ("pitcher", "pitcher_id", "mlb_id", "player_id"))
    den_col = "BF" if "BF" in out.columns else ("PA" if "PA" in out.columns else None)
    if p_col is None or den_col is None:
        print("⚠️  pitcher true-talent: missing pitcher id or BF column — skipping")
        return out
    out = out.sort_values([p_col, "game_date"]).reset_index(drop=True)
    for stat, raw in [("k", "K"), ("bb", "BB"), ("h", "H"), ("hr", "HR")]:
        if raw not in out.columns:
            continue
        out[f"p_tt_{stat}"] = _cum_shrunk(out, p_col, raw, den_col,
                                          STAB[stat], leagues[stat])
    return out


def add_lineup_matchup(pdf: pd.DataFrame, hdf: pd.DataFrame, leagues: dict) -> pd.DataFrame:
    """Aggregate the opposing lineup's trailing true-talent rates and log5 them
    with the pitcher's own trailing rate. Uses the same (game_date, opponent)
    matching as enrich_lineup_features."""
    out = pdf.drop(columns=[c for c in LINEUP_COLS + MATCHUP_COLS if c in pdf.columns],
                   errors="ignore").copy()

    p_team_col = _first_col(out, ("team", "pitcher_team"))
    h_opp_col = _first_col(hdf, ("opponent", "opponent_team", "pitcher_team"))
    if p_team_col is None or h_opp_col is None:
        print("⚠️  lineup matchup: missing team/opponent columns — skipping")
        for c in LINEUP_COLS + MATCHUP_COLS:
            out[c] = np.nan
        return out

    h = hdf.copy()
    h["game_date"] = pd.to_datetime(h.get("game_date"), errors="coerce")
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce")
    if h_opp_col != "opponent":
        h = h.rename(columns={h_opp_col: "opponent"})
    if p_team_col != "team":
        out = out.rename(columns={p_team_col: "team"})

    have_tt = [s for s in ("k", "bb", "h", "hr") if f"h_tt_{s}" in h.columns]
    if not have_tt:
        print("⚠️  lineup matchup: hitter true-talent columns missing — run add_hitter_truetalent first")
        for c in LINEUP_COLS + MATCHUP_COLS:
            out[c] = np.nan
        return out

    groups = h.groupby(["game_date", "opponent"], sort=False)

    lineup_vals: dict[str, list] = {s: [] for s in ("k", "bb", "h", "hr")}
    for _, prow in out.iterrows():
        key = (prow.get("game_date"), prow.get("team"))
        try:
            sub = groups.get_group(key)
        except KeyError:
            for s in lineup_vals:
                lineup_vals[s].append(np.nan)
            continue
        if "PA" in sub.columns and len(sub) > 9:
            sub = sub.nlargest(9, "PA")
        for s in ("k", "bb", "h", "hr"):
            col = f"h_tt_{s}"
            if col in sub.columns:
                v = pd.to_numeric(sub[col], errors="coerce")
                lineup_vals[s].append(float(v.mean()) if v.notna().any() else np.nan)
            else:
                lineup_vals[s].append(np.nan)

    for s in ("k", "bb", "h", "hr"):
        out[f"lineup_tt_{s}"] = lineup_vals[s]
        p_col = f"p_tt_{s}"
        if p_col in out.columns:
            out[f"matchup_{s}"] = _log5(out[p_col],
                                        pd.Series(lineup_vals[s], index=out.index),
                                        leagues[s])
        else:
            out[f"matchup_{s}"] = np.nan
    return out


def enrich(pitcher_csv: Path = DATA_DIR / "pitcher_game_data.csv",
           hitter_csv: Path = DATA_DIR / "hitter_game_data.csv",
           write_back: bool = True) -> None:
    if not pitcher_csv.exists() or not hitter_csv.exists():
        print(f"⚠️  Missing input — pitcher: {pitcher_csv.exists()}, hitter: {hitter_csv.exists()}")
        return

    print(f"\n── True-talent + log5 matchup enrichment ──")
    pdf = pd.read_csv(pitcher_csv, low_memory=False)
    hdf = pd.read_csv(hitter_csv, low_memory=False)

    # League per-PA rates (from hitter table — the natural per-opportunity base)
    leagues = {
        "k":  _league_rate(hdf, "K", "PA"),
        "bb": _league_rate(hdf, "BB", "PA"),
        "h":  _league_rate(hdf, "H", "PA"),
        "hr": _league_rate(hdf, "HR", "PA"),
    }
    print(f"  League per-PA rates: " + ", ".join(f"{k}={v:.3f}" for k, v in leagues.items()))

    hdf = add_hitter_truetalent(hdf, leagues)
    pdf = add_pitcher_truetalent(pdf, leagues)
    pdf = add_lineup_matchup(pdf, hdf, leagues)

    for c in MATCHUP_COLS:
        if c in pdf.columns:
            cov = pdf[c].notna().mean()
            print(f"  {c}: coverage {cov:.1%}  mean {pd.to_numeric(pdf[c], errors='coerce').mean():.3f}")

    if write_back:
        hdf.to_csv(hitter_csv, index=False, float_format="%.4f")
        pdf.to_csv(pitcher_csv, index=False, float_format="%.4f")
        print(f"  Wrote {hitter_csv.name} and {pitcher_csv.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitcher-csv", default=str(DATA_DIR / "pitcher_game_data.csv"))
    ap.add_argument("--hitter-csv", default=str(DATA_DIR / "hitter_game_data.csv"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    enrich(Path(args.pitcher_csv), Path(args.hitter_csv), write_back=not args.no_write)


if __name__ == "__main__":
    main()
