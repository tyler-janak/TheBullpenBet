"""
enrich_team_features.py
=======================
Post-processing pass over `data/hitter_game_data.csv` that adds the
missing team-level context columns the team-PA hitter trainer expects:

    team_obp_std           — team's season-to-date on-base rate
    team_obp_last10        — rolling 10-game on-base rate
    team_pa_avg_std        — team's season-to-date avg PA per game
    team_pa_avg_last10     — rolling 10-game avg PA per game
    opp_sp_ip_avg          — opposing starter's season-to-date IP avg
                             (just an alias for the existing _std column —
                             named differently in the trainer's pool)

Why this is a separate script
-----------------------------
`hitterspitchers_data.py` builds the per-hitter game log directly from
Statcast pitch data. Adding team-level rolling aggregates inside that
builder would entangle the per-pitch logic with team-level groupbys.
A post-processing pass on the already-emitted CSV is cleaner: we read the
hitter table, compute team-game aggregates, join back, write the enriched
CSV. Run idempotently.

When this runs
--------------
Wired into `daily_update.py` immediately AFTER the Statcast data refresh.
Whenever `data/hitter_game_data.csv` is rebuilt we enrich it in-place,
then the team-PA trainer / inference can read the new columns.

Usage
-----
    python enrich_team_features.py
    python enrich_team_features.py --hitter-csv data/hitter_game_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


# ---------------------------------------------------------------------------
# Team aggregations
# ---------------------------------------------------------------------------
def _team_game_table(hitter_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game_date): summed PA / H / BB and the OBP component."""
    cols_needed = {"team", "game_date", "PA", "H", "BB"}
    missing = cols_needed - set(hitter_df.columns)
    if missing:
        raise ValueError(f"hitter_game_data.csv missing columns: {missing}")

    work = hitter_df.copy()
    for c in ("PA", "H", "BB"):
        work[c] = pd.to_numeric(work[c], errors="coerce")

    team_game = (
        work.groupby(["team", "game_date"], as_index=False)
            .agg(team_PA=("PA", "sum"),
                 team_H=("H", "sum"),
                 team_BB=("BB", "sum"))
    )
    # OBP ≈ (H + BB) / (PA). We don't have HBP at the team-game level here
    # (it's not aggregated separately in the per-hitter file), so this is
    # OBP-without-HBP — typically ~0.5% lower than true OBP, immaterial for
    # use as a feature.
    team_game["team_OBP"] = (
        (team_game["team_H"] + team_game["team_BB"]).clip(lower=0)
        / team_game["team_PA"].clip(lower=1)
    )
    team_game["game_date"] = pd.to_datetime(team_game["game_date"], errors="coerce")
    team_game = team_game.sort_values(["team", "game_date"]).reset_index(drop=True)
    return team_game


def _rolling_features(team_game: pd.DataFrame) -> pd.DataFrame:
    """Add team_obp_std/last10 and team_pa_avg_std/last10 with no leakage."""
    work = team_game.copy()

    def _shift_expanding_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().mean()

    def _shift_rolling_mean(s: pd.Series, w: int) -> pd.Series:
        return s.shift(1).rolling(w, min_periods=1).mean()

    work["team_obp_std"]    = work.groupby("team")["team_OBP"].transform(_shift_expanding_mean)
    work["team_obp_last10"] = work.groupby("team")["team_OBP"].transform(lambda s: _shift_rolling_mean(s, 10))
    work["team_pa_avg_std"]    = work.groupby("team")["team_PA"].transform(_shift_expanding_mean)
    work["team_pa_avg_last10"] = work.groupby("team")["team_PA"].transform(lambda s: _shift_rolling_mean(s, 10))

    return work[[
        "team", "game_date",
        "team_obp_std", "team_obp_last10",
        "team_pa_avg_std", "team_pa_avg_last10",
    ]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def enrich(
    hitter_csv: Path = DATA_DIR / "hitter_game_data.csv",
    write_back: bool = True,
) -> pd.DataFrame:
    if not hitter_csv.exists():
        print(f"⚠️  {hitter_csv} not found — skipping team feature enrichment.")
        return pd.DataFrame()

    print(f"\n── Enriching team features in {hitter_csv.name} ──")
    df = pd.read_csv(hitter_csv, low_memory=False)
    n_before = len(df)
    cols_before = set(df.columns)

    # Drop any pre-existing enrichment columns so re-runs are idempotent
    drop_cols = [c for c in (
        "team_obp_std", "team_obp_last10",
        "team_pa_avg_std", "team_pa_avg_last10",
        "opp_sp_ip_avg",
    ) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    team_game = _team_game_table(df)
    rolling = _rolling_features(team_game)
    df = df.merge(rolling, on=["team", "game_date"], how="left")

    # opp_sp_ip_avg — pure alias for opp_sp_ip_std if present, so the trainer
    # finds it under whichever name it looks for. We populate both.
    if "opp_sp_ip_std" in df.columns and "opp_sp_ip_avg" not in df.columns:
        df["opp_sp_ip_avg"] = df["opp_sp_ip_std"]

    added = [c for c in df.columns if c not in cols_before]
    print(f"  Added columns: {added}")
    print(f"  team_obp_std coverage: {df['team_obp_std'].notna().mean():.1%}")
    print(f"  team_pa_avg_std coverage: {df['team_pa_avg_std'].notna().mean():.1%}")

    assert len(df) == n_before, "enrichment changed row count — bug!"

    if write_back:
        # %.4f keeps the in-place rewrite under GitHub's 100 MB limit.
        df.to_csv(hitter_csv, index=False, float_format="%.4f")
        print(f"  Wrote {len(df):,} rows back → {hitter_csv}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hitter-csv", default=str(DATA_DIR / "hitter_game_data.csv"))
    ap.add_argument("--no-write", action="store_true",
                    help="Compute features but don't overwrite the CSV (for inspection).")
    args = ap.parse_args()
    enrich(Path(args.hitter_csv), write_back=not args.no_write)


if __name__ == "__main__":
    main()
