"""
refresh_full_history.py
=======================
Multi-season Statcast refresh: pulls 2025 (full season) + 2026 (year-to-date)
into a unified pitch-level cache, then rebuilds the per-game feature tables
on the combined data.

Why this exists
---------------
`refresh_2026_data.py` only sees the partial 2026 season. As of mid-May that's
~700 pitcher games — too few for the two-stage / xHits / team-PA models to
generalize well. 2025 is a full season (~5000 pitcher games, ~30000 hitter
games) and the underlying baseball game hasn't changed in any way that
materially breaks transferability, so combining the two years gives the
trainers ~7-8× more data.

What it produces
----------------
Same four CSVs as the 2026 refresh, but built from the union of two seasons:

    data/pitcher_game_data.csv      (~5700 rows once 2025 + 2026 are combined)
    data/hitter_game_data.csv       (~33000 rows)
    data/team_batting_hand_context.csv
    data/team_pitching_hand_context.csv

Caches each season's pitch data in its own file so re-pulls are incremental:

    pitch_data_2025.csv   (only ever pulled once unless --rebuild-2025)
    pitch_data_2026.csv   (incremental, updated every cron tick)
    pitch_data_combined.csv (re-built every run; not gitignored — wait, IS gitignored)

Rolling-window features (last_5, last_10, etc.) span both seasons in this
build. That's correct: a pitcher's "last 10 starts" in April 2026 legitimately
spans Sept 2025 + April 2026. Season-to-date `_std` features get reset per
season inside the feature builder so they remain interpretable.

Usage
-----
    python refresh_full_history.py
    python refresh_full_history.py --rebuild-2025     # force re-pull 2025
    python refresh_full_history.py --skip-2026         # skip the 2026 refresh
                                                       # (training-only run)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

CACHE_2025 = Path("pitch_data_2025.csv")
CACHE_2026 = Path("pitch_data_2026.csv")
CACHE_COMBINED = Path("pitch_data_combined.csv")

DATA_DIR = Path("data")

SEASON_2025_START = "2025-03-01"
SEASON_2025_END   = "2025-11-01"   # past final WS game
SEASON_2026_START = "2026-03-01"


# ---------------------------------------------------------------------------
# Statcast pull helpers
# ---------------------------------------------------------------------------
def _try_import_statcast():
    try:
        from pybaseball import statcast
        return statcast
    except ImportError as e:
        print(f"⚠️  pybaseball not available: {e}")
        return None


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _fetch_range(start: str, end: str, statcast_fn) -> pd.DataFrame:
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


def _latest_cached_date(cache: Path) -> str | None:
    if not cache.exists():
        return None
    try:
        df = pd.read_csv(cache, usecols=["game_date"], low_memory=False)
        if df.empty:
            return None
        return pd.to_datetime(df["game_date"], errors="coerce").max().strftime("%Y-%m-%d")
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    dedupe_cols = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in df.columns]
    if dedupe_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
        if before != len(df):
            print(f"   deduped {before - len(df):,} duplicate pitches")
    df.to_csv(path, index=False)
    print(f"   wrote {path} ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# 2025 — pull once, then cached forever
# ---------------------------------------------------------------------------
def refresh_2025(rebuild: bool = False) -> Path | None:
    """Pull the full 2025 season if we don't already have it cached."""
    statcast = _try_import_statcast()
    if statcast is None:
        return CACHE_2025 if CACHE_2025.exists() else None

    if CACHE_2025.exists() and not rebuild:
        latest = _latest_cached_date(CACHE_2025)
        if latest and latest >= SEASON_2025_END:
            print(f"   2025 cache complete (through {latest}) — skipping pull.")
            return CACHE_2025
        # Cache exists but is incomplete — top it up
        next_start = (pd.to_datetime(latest) + timedelta(days=1)).strftime("%Y-%m-%d") if latest else SEASON_2025_START
        new = _fetch_range(next_start, SEASON_2025_END, statcast)
        if not new.empty:
            existing = pd.read_csv(CACHE_2025, low_memory=False)
            combined = pd.concat([existing, new], ignore_index=True, sort=False)
            _save_cache(combined, CACHE_2025)
        return CACHE_2025

    # Fresh pull (rebuild or no cache)
    df = _fetch_range(SEASON_2025_START, SEASON_2025_END, statcast)
    if df.empty:
        return CACHE_2025 if CACHE_2025.exists() else None
    _save_cache(df, CACHE_2025)
    return CACHE_2025


# ---------------------------------------------------------------------------
# 2026 — delegate to the existing incremental refresh logic
# ---------------------------------------------------------------------------
def refresh_2026(rebuild: bool = False, end: str | None = None) -> Path | None:
    try:
        from refresh_2026_data import refresh_pitch_cache
        return refresh_pitch_cache(start=None, end=end, rebuild=rebuild)
    except Exception as e:
        print(f"⚠️  2026 refresh failed: {e}")
        return CACHE_2026 if CACHE_2026.exists() else None


# ---------------------------------------------------------------------------
# Concat 2025 + 2026 into a single pitch cache
# ---------------------------------------------------------------------------
def combine_caches() -> Path | None:
    parts: list[pd.DataFrame] = []
    for cache in (CACHE_2025, CACHE_2026):
        if cache.exists():
            print(f"   reading {cache}…")
            try:
                parts.append(pd.read_csv(cache, low_memory=False))
            except Exception as e:
                print(f"⚠️  couldn't read {cache}: {e}")
    if not parts:
        return None
    combined = pd.concat(parts, ignore_index=True, sort=False)
    dedupe_cols = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in combined.columns]
    if dedupe_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
        if before != len(combined):
            print(f"   deduped {before - len(combined):,} duplicate pitches across seasons")
    _save_cache(combined, CACHE_COMBINED)
    return CACHE_COMBINED


# ---------------------------------------------------------------------------
# Feature build over the combined cache
# ---------------------------------------------------------------------------
def build_features(pitch_csv: Path) -> bool:
    try:
        import hitterspitchers_data as hpd
    except Exception as e:
        print(f"⚠️  couldn't import hitterspitchers_data: {e}")
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

    # Report year split so we know the trainer sample sizes
    if "game_date" in pitcher_df.columns:
        pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"], errors="coerce")
        by_year = pitcher_df["game_date"].dt.year.value_counts().sort_index()
        print(f"\n  Pitcher games by season: {by_year.to_dict()}")
    if "game_date" in hitter_df.columns:
        hitter_df["game_date"] = pd.to_datetime(hitter_df["game_date"], errors="coerce")
        by_year_h = hitter_df["game_date"].dt.year.value_counts().sort_index()
        print(f"  Hitter games by season:  {by_year_h.to_dict()}")

    print("\nWrote:")
    print(f"  {DATA_DIR/'pitcher_game_data.csv'}  ({len(pitcher_df):,} rows)")
    print(f"  {DATA_DIR/'hitter_game_data.csv'}   ({len(hitter_df):,} rows)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def refresh(
    rebuild_2025: bool = False,
    skip_2026: bool = False,
    end_2026: str | None = None,
    skip_features: bool = False,
) -> bool:
    print(f"\n========== Multi-season Statcast refresh ==========")
    print(f"   2025 cache: {CACHE_2025}  (exists={CACHE_2025.exists()})")
    print(f"   2026 cache: {CACHE_2026}  (exists={CACHE_2026.exists()})")

    print("\n── 2025 ────────────────────────────────────────")
    refresh_2025(rebuild=rebuild_2025)

    if not skip_2026:
        print("\n── 2026 ────────────────────────────────────────")
        refresh_2026(end=end_2026)
    else:
        print("\n── 2026 ────────────────────────────────────────")
        print("   (skipped)")

    print("\n── Combining ───────────────────────────────────")
    combined = combine_caches()
    if combined is None:
        print("⚠️  No pitch data available for either season.")
        return False

    if skip_features:
        print("Skipping feature build (--skip-features).")
        return True

    ok = build_features(combined)

    # Run team-feature enrichment on top of the multi-season build
    if ok:
        try:
            from enrich_team_features import enrich
            enrich()
        except Exception as e:
            print(f"⚠️  team-feature enrichment failed: {e}")

        # Lineup-feature enrichment — adds lineup_k_rate / lineup_bb_rate /
        # lineup_avg_ev / etc. to pitcher_game_data.csv. Needs hitter rows
        # built, so it runs AFTER the team feature pass.
        try:
            from enrich_lineup_features import enrich as enrich_lineup
            enrich_lineup()
        except Exception as e:
            print(f"⚠️  lineup-feature enrichment failed: {e}")

    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-2025", action="store_true",
                   help="Force a fresh re-pull of the full 2025 season.")
    p.add_argument("--skip-2026", action="store_true",
                   help="Skip the 2026 refresh — useful for retraining-only runs.")
    p.add_argument("--end-2026", default=None,
                   help="Last date YYYY-MM-DD for the 2026 pull (default: today ET).")
    p.add_argument("--skip-features", action="store_true",
                   help="Only refresh caches; skip the per-game feature build.")
    args = p.parse_args()

    ok = refresh(
        rebuild_2025=args.rebuild_2025,
        skip_2026=args.skip_2026,
        end_2026=args.end_2026,
        skip_features=args.skip_features,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
