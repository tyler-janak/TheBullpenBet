"""
enrich_lineup_features.py
=========================
Adds per-pitcher-game lineup-level aggregation columns to
`data/pitcher_game_data.csv`.

Why this exists
---------------
The current pitcher features include `team_k_rate_vs_hand_last10` — the
entire opposing team's K-rate against the pitcher's hand. That's the team
average, which is useful but coarse: pitchers face a specific 9-batter
lineup each night, and there's a huge spread between a Judge-Soto-Stanton
top of order and a Maldonado-Wong-Ruiz bottom. Aggregating the **actual
lineup that faced this pitcher** gives a sharper signal — and crucially,
one that's NOT correlated with the pitcher's own K_rate, so XGBoost can't
substitute it.

What gets added
---------------
For each pitcher-game in pitcher_game_data.csv, we look up the 9 batters
who faced that pitcher in that game (from hitter_game_data.csv, matched by
team + game_date + opp_pitcher_name), pull each batter's pre-game
season-to-date and last-10 rates vs the pitcher's handedness, then aggregate:

    lineup_k_rate              — mean of hitter K-rate vs pitcher hand
    lineup_k_rate_last10       — mean of hitter last-10 K-rate vs pitcher hand
    lineup_bb_rate             — mean of hitter walk rate vs pitcher hand
    lineup_bb_rate_last10
    lineup_h_rate              — mean of hitter hit rate vs pitcher hand
    lineup_h_rate_last10
    lineup_hr_rate             — mean of hitter HR rate vs pitcher hand
    lineup_hr_rate_last10
    lineup_avg_ev              — mean of hitter avg EV (contact quality)
    lineup_hard_hit_pct        — mean of hitter hard-hit rate
    lineup_n_batters           — how many batters we actually matched (≤9)

The lineup_* columns are explicitly NOT a copy or correlate of any pitcher
feature — they describe the opponent. This is the information channel that
unlocked the most gain in industry models.

When this runs
--------------
After `refresh_full_history.py` builds the per-game tables, before
training. Idempotent: re-runs drop the prior lineup_* columns first.

Usage
-----
    python enrich_lineup_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

LINEUP_COLS = [
    "lineup_k_rate", "lineup_k_rate_last10",
    "lineup_bb_rate", "lineup_bb_rate_last10",
    "lineup_h_rate",  "lineup_h_rate_last10",
    "lineup_hr_rate", "lineup_hr_rate_last10",
    "lineup_avg_ev",  "lineup_hard_hit_pct",
    "lineup_n_batters",
]


def _pick_hand_col(df: pd.DataFrame, base: str, hand: str) -> pd.Series:
    """Return df[base + '_' + hand] series; falls back to 'R' if 'L' missing."""
    col = f"{base}_{hand}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    other = "L" if hand == "R" else "R"
    fallback = f"{base}_{other}"
    if fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def enrich(
    pitcher_csv: Path = DATA_DIR / "pitcher_game_data.csv",
    hitter_csv: Path = DATA_DIR / "hitter_game_data.csv",
    write_back: bool = True,
) -> pd.DataFrame:
    if not pitcher_csv.exists() or not hitter_csv.exists():
        print(f"⚠️  Missing input — pitcher: {pitcher_csv.exists()}, hitter: {hitter_csv.exists()}")
        return pd.DataFrame()

    print(f"\n── Enriching lineup features in {pitcher_csv.name} ──")
    pdf = pd.read_csv(pitcher_csv, low_memory=False)
    hdf = pd.read_csv(hitter_csv,  low_memory=False)

    # Drop prior enrichment columns so re-runs are idempotent
    drop = [c for c in LINEUP_COLS if c in pdf.columns]
    if drop:
        pdf = pdf.drop(columns=drop)

    pdf["game_date"] = pd.to_datetime(pdf["game_date"], errors="coerce")
    hdf["game_date"] = pd.to_datetime(hdf["game_date"], errors="coerce")

    # We need: hitter rows where opponent == pitcher's team AND game_date matches.
    # The hitter row's `opp_pitcher_name` doesn't always match pitcher_name from
    # the pitcher table because of name normalization quirks, so we match by
    # (game_date, opponent_team=pitcher's team). For a given pitcher-game this
    # gives us all 9 hitters on the opposing offense for that day.
    #
    # Column names vary across data builds — pitcher rows use `team` for the
    # pitcher's own team, but the opponent column may be `opponent` or
    # `opponent_team` depending on the builder version. Same on the hitter side.
    pitcher_team_col     = next((c for c in ("team", "pitcher_team")       if c in pdf.columns), None)
    # The hitter table's OPPONENT (from the hitter's POV = the pitcher's team)
    # may be named `opponent`, `opponent_team`, or — most commonly in this
    # build — `pitcher_team` because every hitter row records who they faced.
    hitter_opponent_col  = next((c for c in ("opponent", "opponent_team", "pitcher_team") if c in hdf.columns), None)

    if pitcher_team_col is None:
        print(f"⚠️  pitcher_game_data missing team column "
              f"(have: {[c for c in pdf.columns if 'team' in c.lower()][:5]})")
        return pdf
    if hitter_opponent_col is None:
        print(f"⚠️  hitter_game_data missing opponent column "
              f"(have: {[c for c in hdf.columns if 'opp' in c.lower() or 'team' in c.lower()][:5]})")
        return pdf

    print(f"  Using pitcher team column: {pitcher_team_col}  "
          f"hitter opponent column: {hitter_opponent_col}")
    # Standardize for downstream matching: rename to canonical names locally
    if pitcher_team_col != "team":
        pdf = pdf.rename(columns={pitcher_team_col: "team"})
    if hitter_opponent_col != "opponent":
        hdf = hdf.rename(columns={hitter_opponent_col: "opponent"})

    # Sanity: hitter's `opponent` is who the hitter was FACING (the pitcher's team).
    # For pitcher P on team T playing date D, we want hitters where
    # hdf.opponent == T AND hdf.game_date == D.
    # That's the 9-batter lineup that faced P.

    # Pre-compute the per-hand columns we'll pull from the hitter rows.
    # We do this once and split downstream by pitcher hand for performance.
    base_cols = {
        "k":   "hitter_k_rate_vs_hand",
        "bb":  "hitter_bb_rate_vs_hand",
        "h":   "hitter_h_rate_vs_hand",
        "hr":  "hitter_hr_rate_vs_hand",
        "k10": "hitter_k_rate_vs_hand_last10",
        "bb10": "hitter_bb_rate_vs_hand_last10",
        "h10":  "hitter_h_rate_vs_hand_last10",
        "hr10": "hitter_hr_rate_vs_hand_last10",
    }

    print(f"  Iterating {len(pdf):,} pitcher games × ~9 batters each "
          f"({len(hdf):,} hitter rows available)…")

    # Group hitter rows by (game_date, opponent_team) → list of hitter rows.
    # opponent here = the team the hitters are FACING, which is the pitcher's team.
    hitter_groups = hdf.groupby(["game_date", "opponent"], sort=False)

    enrich_rows = []
    matched_total = 0
    for idx, prow in pdf.iterrows():
        pteam = prow.get("team")
        pdate = prow.get("game_date")
        phand = str(prow.get("pitcher_hand", "")).strip().upper() or "R"
        if phand not in ("R", "L"):
            phand = "R"

        try:
            sub = hitter_groups.get_group((pdate, pteam))
        except KeyError:
            enrich_rows.append({c: np.nan for c in LINEUP_COLS})
            continue

        if sub.empty:
            enrich_rows.append({c: np.nan for c in LINEUP_COLS})
            continue

        # Limit to the top 9 by PA — guards against pinch-hitter spam blowing
        # up the lineup average.
        if "PA" in sub.columns and len(sub) > 9:
            sub = sub.nlargest(9, "PA")

        matched_total += 1

        out_row = {}
        for key, base in [
            ("lineup_k_rate",         "hitter_k_rate_vs_hand"),
            ("lineup_k_rate_last10",  "hitter_k_rate_vs_hand_last10"),
            ("lineup_bb_rate",        "hitter_bb_rate_vs_hand"),
            ("lineup_bb_rate_last10", "hitter_bb_rate_vs_hand_last10"),
            ("lineup_h_rate",         "hitter_h_rate_vs_hand"),
            ("lineup_h_rate_last10",  "hitter_h_rate_vs_hand_last10"),
            ("lineup_hr_rate",        "hitter_hr_rate_vs_hand"),
            ("lineup_hr_rate_last10", "hitter_hr_rate_vs_hand_last10"),
        ]:
            series = _pick_hand_col(sub, base, phand)
            out_row[key] = float(series.mean()) if series.notna().any() else np.nan

        # Contact quality (handedness-agnostic; barrel/EV aren't reliably split)
        if "avg_EV" in sub.columns:
            ev_vals = pd.to_numeric(sub["avg_EV"], errors="coerce")
            out_row["lineup_avg_ev"] = float(ev_vals.mean()) if ev_vals.notna().any() else np.nan
        else:
            out_row["lineup_avg_ev"] = np.nan

        # hard_hit_proxy_std is the per-hitter hard-hit rate already computed
        if "hard_hit_proxy_std" in sub.columns:
            hh = pd.to_numeric(sub["hard_hit_proxy_std"], errors="coerce")
            out_row["lineup_hard_hit_pct"] = float(hh.mean()) if hh.notna().any() else np.nan
        else:
            out_row["lineup_hard_hit_pct"] = np.nan

        out_row["lineup_n_batters"] = int(len(sub))
        enrich_rows.append(out_row)

    enriched = pd.DataFrame(enrich_rows, index=pdf.index)
    pdf = pd.concat([pdf, enriched], axis=1)

    print(f"  Matched {matched_total:,}/{len(pdf):,} pitcher games to a lineup "
          f"({100*matched_total/max(1,len(pdf)):.1f}%)")
    print(f"  Coverage on lineup_k_rate: {pdf['lineup_k_rate'].notna().mean():.1%}")
    print(f"  Coverage on lineup_k_rate_last10: {pdf['lineup_k_rate_last10'].notna().mean():.1%}")
    print(f"  Mean lineup_k_rate: {pdf['lineup_k_rate'].mean():.3f}")

    if write_back:
        # %.4f keeps the in-place rewrite under GitHub's 100 MB limit.
        pdf.to_csv(pitcher_csv, index=False, float_format="%.4f")
        print(f"  Wrote {len(pdf):,} rows back → {pitcher_csv}")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitcher-csv", default=str(DATA_DIR / "pitcher_game_data.csv"))
    ap.add_argument("--hitter-csv",  default=str(DATA_DIR / "hitter_game_data.csv"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    enrich(Path(args.pitcher_csv), Path(args.hitter_csv), write_back=not args.no_write)


if __name__ == "__main__":
    main()
