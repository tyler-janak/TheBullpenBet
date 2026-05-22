"""
refresh_2026_data.py
====================
Pulls fresh 2026 Statcast pitch-level data via pybaseball, then rebuilds the
four feature CSVs the projection model reads from:

    data/hitter_game_data.csv
    data/pitcher_game_data.csv
    data/team_batting_hand_context.csv
    data/team_pitching_hand_context.csv

It is INCREMENTAL — it only pulls Statcast for dates newer than what's already
in the local pitch_data_2026.csv cache, then concats. The full feature build
runs against the combined cache so rolling stats (last_10, last_20, etc.)
are computed correctly across the whole season.

Run:
    python refresh_2026_data.py
    python refresh_2026_data.py --start 2026-03-01 --end 2026-04-29
    python refresh_2026_data.py --rebuild        # force a full re-pull

Outputs:
    pitch_data_2026.csv          (the local cache — gitignored, big)
    data/hitter_game_data.csv    (committed — features for projections)
    data/pitcher_game_data.csv   (committed)
    data/team_batting_hand_context.csv
    data/team_pitching_hand_context.csv

Notes
-----
* pybaseball.statcast() is rate-limited and chunked. A full season pull
  is several minutes; an incremental "since yesterday" pull is fast.
* If pybaseball isn't installed or the network is flaky, the script
  prints a warning and exits 0 — the daily pipeline will continue with
  whatever data files already exist on disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
SEASON_START = "2026-03-01"   # cover spring training + reg season
SEASON_END_FALLBACK = "2026-11-15"  # max date we'd ever pull through

CACHE_PATH = Path("pitch_data_2026.csv")
DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Statcast fetch (pybaseball)
# ---------------------------------------------------------------------------
def _try_import_pybaseball():
    try:
        from pybaseball import statcast
        return statcast
    except ImportError as e:
        print(f"⚠️  pybaseball not available: {e}")
        print("   Install with:  pip install pybaseball")
        return None


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _latest_cached_date() -> str | None:
    """Read the cache (if any) and return the max game_date as 'YYYY-MM-DD'."""
    if not CACHE_PATH.exists():
        return None
    try:
        # Read only the game_date column for speed.
        df = pd.read_csv(CACHE_PATH, usecols=["game_date"], low_memory=False)
        if df.empty:
            return None
        return pd.to_datetime(df["game_date"], errors="coerce").max().strftime("%Y-%m-%d")
    except Exception as e:
        print(f"⚠️  Couldn't read existing cache ({e}) — will treat as empty.")
        return None


def fetch_statcast_range(start: str, end: str, statcast_fn) -> pd.DataFrame:
    """Wrap pybaseball.statcast() so a single failure doesn't abort the run."""
    print(f"  → pulling Statcast for {start} → {end} (this can take a few minutes)…")
    try:
        df = statcast_fn(start_dt=start, end_dt=end)
    except Exception as e:
        print(f"⚠️  pybaseball error: {e}")
        traceback.print_exc()
        return pd.DataFrame()
    if df is None or df.empty:
        print("   (no rows returned)")
        return pd.DataFrame()
    print(f"   pulled {len(df):,} pitches")
    return df


def refresh_pitch_cache(start: str | None = None,
                        end: str | None = None,
                        rebuild: bool = False) -> Path | None:
    """
    Update pitch_data_2026.csv with any missing dates. Returns the cache path
    if data is now available, otherwise None.
    """
    statcast_fn = _try_import_pybaseball()
    if statcast_fn is None:
        return CACHE_PATH if CACHE_PATH.exists() else None

    end = end or _today_et()
    if rebuild or not CACHE_PATH.exists():
        pull_start = start or SEASON_START
        new_pitches = fetch_statcast_range(pull_start, end, statcast_fn)
        if new_pitches.empty:
            return CACHE_PATH if CACHE_PATH.exists() else None
        new_pitches.to_csv(CACHE_PATH, index=False)
        print(f"   wrote {CACHE_PATH} ({len(new_pitches):,} rows)")
        return CACHE_PATH

    # Incremental: pull only what's missing.
    latest = _latest_cached_date()
    if latest is None:
        pull_start = start or SEASON_START
    else:
        # +1 day so we don't re-pull the final cached day.
        pull_start = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    if pull_start > end:
        print(f"   cache already current through {latest} — nothing to fetch.")
        return CACHE_PATH

    new_pitches = fetch_statcast_range(pull_start, end, statcast_fn)
    if new_pitches.empty:
        return CACHE_PATH

    existing = pd.read_csv(CACHE_PATH, low_memory=False)
    combined = pd.concat([existing, new_pitches], ignore_index=True, sort=False)
    # Statcast occasionally returns dup pitches at chunk boundaries — dedupe
    # on the natural primary key (game_pk + at_bat + pitch_number).
    dedupe_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in combined.columns]
    if dedupe_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
        print(f"   deduped {before - len(combined):,} duplicate pitches")
    combined.to_csv(CACHE_PATH, index=False)
    print(f"   wrote {CACHE_PATH} ({len(combined):,} total rows)")
    return CACHE_PATH


# ---------------------------------------------------------------------------
# Feature build (delegates to hitterspitchers_data.py)
# ---------------------------------------------------------------------------
def build_features(pitch_csv: Path) -> bool:
    """Run hitterspitchers_data feature builder on the pitch cache."""
    try:
        import hitterspitchers_data as hpd
    except Exception as e:
        print(f"⚠️  Couldn't import hitterspitchers_data: {e}")
        return False

    print(f"\n── Building per-game features from {pitch_csv} ──")
    DATA_DIR.mkdir(exist_ok=True)

    df = hpd.load_data(str(pitch_csv))
    df = hpd.event_flags(df)
    df = hpd.mark_actual_starters(df)

    park_factors = hpd.load_park_factors()

    team_batting_hand_ctx = hpd.build_team_batting_hand_context(df)
    team_pitching_hand_ctx = hpd.build_team_pitching_hand_context(df)

    pitcher_df = hpd.build_pitcher_games(df, team_batting_hand_ctx)
    hitter_df = hpd.build_hitter_games(df, team_pitching_hand_ctx)
    hitter_df = hpd.enrich_hitter_with_opp_starter(hitter_df, pitcher_df)

    pitcher_df = hpd.merge_park_factors(pitcher_df, park_factors)
    hitter_df = hpd.merge_park_factors(hitter_df, park_factors)

    # float_format="%.4f" keeps the committed feature tables under GitHub's
    # 100 MB limit (~195 float columns; full repr blows hitter_game_data.csv
    # to 150 MB+). int64 ID columns are unaffected.
    pitcher_df.to_csv(DATA_DIR / "pitcher_game_data.csv", index=False, float_format="%.4f")
    hitter_df.to_csv(DATA_DIR / "hitter_game_data.csv", index=False, float_format="%.4f")
    team_batting_hand_ctx.to_csv(DATA_DIR / "team_batting_hand_context.csv", index=False, float_format="%.4f")
    team_pitching_hand_ctx.to_csv(DATA_DIR / "team_pitching_hand_context.csv", index=False, float_format="%.4f")

    print("\nWrote:")
    print(f"  {DATA_DIR/'pitcher_game_data.csv'}  ({len(pitcher_df):,} rows)")
    print(f"  {DATA_DIR/'hitter_game_data.csv'}   ({len(hitter_df):,} rows)")
    print(f"  {DATA_DIR/'team_batting_hand_context.csv'}")
    print(f"  {DATA_DIR/'team_pitching_hand_context.csv'}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def refresh(start: str | None = None,
            end: str | None = None,
            rebuild: bool = False,
            skip_features: bool = False) -> bool:
    print(f"\n========== Refreshing 2026 player history ==========")
    cache = refresh_pitch_cache(start=start, end=end, rebuild=rebuild)
    if cache is None or not cache.exists():
        print("⚠️  No 2026 pitch data available — leaving existing data/*.csv untouched.")
        return False

    if skip_features:
        print("Skipping feature build (--skip-features).")
        return True

    return build_features(cache)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=None,
                   help=f"First date YYYY-MM-DD (default {SEASON_START} on first run; otherwise resume from cache)")
    p.add_argument("--end", default=None, help="Last date YYYY-MM-DD (default: today ET)")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore the existing cache and re-pull from --start.")
    p.add_argument("--skip-features", action="store_true",
                   help="Only refresh the pitch cache; skip the feature CSV build.")
    args = p.parse_args()

    ok = refresh(start=args.start, end=args.end,
                 rebuild=args.rebuild, skip_features=args.skip_features)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
