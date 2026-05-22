"""
backfill_player_predictions.py
==============================
Generates a player-projection snapshot for every date in the 2026 season
that doesn't already have one, then grades them against MLB box scores.

The previous version of this file delegated to ``run_projections(date)``,
which scrapes Rotowire for lineups. Rotowire only carries today/tomorrow's
lineups, so past-date runs returned empty snapshots and the grader had
nothing to grade — this is why the season had only ~5 days of data.

This version pulls the **actual lineup** from each game's MLB Stats API
boxscore (the same source the grader uses), builds the lineup DataFrame
in the shape ``score_pitchers`` / ``score_hitters`` expect, and runs the
trained models against those features. That gives us a real per-game
projection for every player who actually batted / pitched on that date,
which the grader can then join to actual outcomes.

Usage
-----
    python backfill_player_predictions.py               # full season backfill + grade
    python backfill_player_predictions.py --diagnose    # just report what's missing
    python backfill_player_predictions.py --start 2026-03-25 --end 2026-04-28
    python backfill_player_predictions.py --force       # re-run every date
    python backfill_player_predictions.py --no-grade    # only build snapshots
    python backfill_player_predictions.py --verbose     # full traceback per failure

Output
------
For each completed date we write ``outputs/hitterspitchers_<date>.csv`` in
the same column shape as ``hitterspitchers_today.csv`` so the grader can
join to actuals using ``mlb_id`` + ``game_date``.
"""

from __future__ import annotations

# Suppress sklearn's `joblib.delayed` UserWarning before any sklearn-using
# import. The warning fires once per predict() call — during a 37-date
# backfill that's literally tens of thousands of warnings, drowning the
# real progress output. See hitterspitchers_today.py header for details.
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")

OUT_DIR = Path("outputs")
SEASON_START_DEFAULT = "2026-03-25"

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

FINISHED_STATES = {"Final", "Game Over", "Completed Early", "Completed"}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def _date_range(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        return []
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _snapshot_path(date_str: str) -> Path:
    return OUT_DIR / f"hitterspitchers_{date_str}.csv"


def _snapshot_status(date_str: str) -> str:
    p = _snapshot_path(date_str)
    if not p.exists():
        return "missing"
    try:
        sz = p.stat().st_size
    except OSError:
        return "missing"
    return "present_empty" if sz < 64 else "present_with_rows"


# ---------------------------------------------------------------------------
# MLB Stats API — fetch actual lineups from past-game box scores
# ---------------------------------------------------------------------------
def _fetch_finished_games(date_str: str, timeout: int = 30) -> list[dict]:
    """Return [{game_pk, away_id, home_id, away_abbr, home_abbr, away_pid, home_pid, status}]."""
    try:
        r = requests.get(SCHEDULE_URL, params={
            "sportId": 1, "date": date_str,
            "hydrate": "probablePitcher,team",
        }, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [schedule] {date_str}: {e}")
        return []

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = (g.get("status") or {}).get("detailedState", "")
            if status not in FINISHED_STATES:
                continue
            teams = g.get("teams", {}) or {}
            home = teams.get("home", {}) or {}
            away = teams.get("away", {}) or {}
            home_team = home.get("team", {}) or {}
            away_team = away.get("team", {}) or {}
            home_pp = home.get("probablePitcher", {}) or {}
            away_pp = away.get("probablePitcher", {}) or {}
            games.append({
                "game_pk": g.get("gamePk"),
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "home_abbr": home_team.get("abbreviation"),
                "away_abbr": away_team.get("abbreviation"),
                "home_probable_pitcher_id": home_pp.get("id"),
                "home_probable_pitcher": home_pp.get("fullName"),
                "away_probable_pitcher_id": away_pp.get("id"),
                "away_probable_pitcher": away_pp.get("fullName"),
                "status": status,
            })
    return games


def _fetch_boxscore_lineups(game_pk: int, timeout: int = 30) -> dict:
    """
    Return:
      {
        'home': {'team_abbr', 'starting_pitcher_id', 'starting_pitcher_name',
                 'lineup': [{mlb_id, name, lineup_spot, pos, hand}], ...},
        'away': {...},
      }
    Only batters who actually batted (battingOrder set) are returned, in order.
    Starting pitcher = the pitcher who recorded the first out, identified by
    'gameStatus.isCurrentBatter == False AND stats.pitching.gamesStarted == 1'
    in the per-player block — but since the API exposes a 'startingPitcher'
    via the 'teams.<side>.pitchers[0]', we use that.
    """
    try:
        r = requests.get(BOXSCORE_URL.format(game_pk=game_pk), timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [box]      {game_pk}: {e}")
        return {}

    out = {}
    for side in ("home", "away"):
        block = (data.get("teams") or {}).get(side) or {}
        team = (block.get("team") or {}).get("abbreviation", "")
        players = block.get("players") or {}
        pitchers_order = block.get("pitchers") or []
        starting_pitcher_id = pitchers_order[0] if pitchers_order else None

        # Build the lineup ordered by battingOrder.
        lineup_rows = []
        for pid_key, p in players.items():
            person = p.get("person") or {}
            mlb_id = person.get("id")
            name = person.get("fullName")
            pos = (p.get("position") or {}).get("abbreviation", "")
            batting_side = (p.get("batSide") or {}).get("code")  # 'L' / 'R' / 'S'
            # battingOrder is "100" / "200" / etc. for actual lineup spots,
            # plus offsets (e.g. "101") for substitutions. We want the
            # starters: those whose battingOrder ends with "00".
            bo = p.get("battingOrder")
            if bo is None:
                continue
            try:
                bo_int = int(bo)
            except (TypeError, ValueError):
                continue
            if bo_int % 100 != 0:
                continue   # this row is a substitute, not a starter
            lineup_spot = bo_int // 100
            stats_batting = (p.get("stats") or {}).get("batting") or {}
            had_pa = bool(stats_batting.get("plateAppearances")) or bool(stats_batting.get("atBats"))
            if not had_pa:
                continue
            lineup_rows.append({
                "mlb_id": mlb_id,
                "name": name,
                "lineup_spot": lineup_spot,
                "pos": pos,
                "batter_hand": batting_side,
            })

        # Find the actual starting pitcher's name (preferred over probable).
        sp_name = None
        if starting_pitcher_id is not None:
            sp_block = players.get(f"ID{starting_pitcher_id}") or {}
            sp_name = (sp_block.get("person") or {}).get("fullName")

        out[side] = {
            "team_abbr": team,
            "starting_pitcher_id": starting_pitcher_id,
            "starting_pitcher_name": sp_name,
            "lineup": sorted(lineup_rows, key=lambda x: x["lineup_spot"]),
        }
    return out


# ---------------------------------------------------------------------------
# Build the input DataFrames score_pitchers / score_hitters expect, but
# from past-date actual lineups instead of Rotowire scrapes.
# ---------------------------------------------------------------------------
def _build_past_inputs(date_str: str, sleep_seconds: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For the given past date, returns (pitchers_today_df, hitters_today_df)
    matching the column shape that score_pitchers / score_hitters expect.
    """
    games = _fetch_finished_games(date_str)
    if not games:
        return pd.DataFrame(), pd.DataFrame()

    pitcher_rows = []
    hitter_rows = []

    for g in games:
        box = _fetch_boxscore_lineups(int(g["game_pk"]))
        if not box:
            continue

        # Both starting pitchers
        for side, opp_side in (("home", "away"), ("away", "home")):
            sb = box.get(side) or {}
            opp_sb = box.get(opp_side) or {}
            sp_id = sb.get("starting_pitcher_id")
            sp_name = sb.get("starting_pitcher_name")
            if sp_id and sp_name:
                pitcher_rows.append({
                    "player_type": "pitcher",
                    "team": sb.get("team_abbr"),
                    "opponent": opp_sb.get("team_abbr"),
                    "player_name": sp_name,
                    "mlb_id": sp_id,
                    "is_actual_starter": True,
                })

            # The hitters in this team's lineup.
            for h in (sb.get("lineup") or []):
                hitter_rows.append({
                    "player_type": "hitter",
                    "team": sb.get("team_abbr"),
                    "opponent": opp_sb.get("team_abbr"),
                    "player_name": h["name"],
                    "mlb_id": h["mlb_id"],
                    "lineup_spot": h["lineup_spot"],
                    "pos": h["pos"],
                    "lineup_status": "Confirmed",  # box scores are by definition the actual lineup
                    "norm_name": (h["name"] or "").lower(),
                    "roster_name": h["name"],
                })

        time.sleep(sleep_seconds)

    return pd.DataFrame(pitcher_rows), pd.DataFrame(hitter_rows)


# ---------------------------------------------------------------------------
# Core: project a single past date using the trained models
# ---------------------------------------------------------------------------
def _project_past_date(date_str: str, sleep_seconds: float = 0.2) -> Optional[pd.DataFrame]:
    """
    Run the trained pitcher + hitter projection models against the actual
    lineups that played on `date_str`. Returns a DataFrame in the same
    shape as outputs/hitterspitchers_today.csv, or None on full failure.
    """
    # Lazy import — these modules pull in heavy deps, only do it when needed.
    import hitterspitchers_today as hpt

    pitchers_today, hitters_today = _build_past_inputs(date_str, sleep_seconds=sleep_seconds)
    if pitchers_today.empty and hitters_today.empty:
        return None

    target_ts = pd.Timestamp(date_str)

    pitcher_game_df = pd.read_csv(hpt.DATA_DIR / "pitcher_game_data.csv", low_memory=False)
    hitter_game_df = pd.read_csv(hpt.DATA_DIR / "hitter_game_data.csv", low_memory=False)
    team_batting_ctx = pd.read_csv(hpt.DATA_DIR / "team_batting_hand_context.csv", low_memory=False)
    team_pitching_ctx = pd.read_csv(hpt.DATA_DIR / "team_pitching_hand_context.csv", low_memory=False)

    league_means = hpt.infer_league_means(pitcher_game_df, hitter_game_df)
    # V3 overhaul renamed `load_models` → `load_models_count_only`. Fall back
    # gracefully so this also works against older versions of the module.
    _load = getattr(hpt, "load_models_count_only", None) or getattr(hpt, "load_models", None)
    if _load is None:
        raise AttributeError("hitterspitchers_today exposes neither load_models_count_only nor load_models")
    pitcher_models = _load("pitcher", hpt.PITCHER_TARGETS)
    hitter_models  = _load("hitter",  hpt.HITTER_TARGETS)

    # Hybrid models: loaded if available, score_* falls back to legacy if not.
    two_stage_pitcher = None
    try:
        if getattr(hpt, "load_two_stage_models", None) is not None:
            two_stage_pitcher = hpt.load_two_stage_models()
    except Exception as e:
        print(f"  [two-stage pitcher] {date_str} load failed: {type(e).__name__}: {e}")
        two_stage_pitcher = None

    team_pa_bundle, rate_models = None, None
    try:
        if getattr(hpt, "load_team_pa_models", None) is not None:
            team_pa_bundle, rate_models = hpt.load_team_pa_models()
    except Exception as e:
        print(f"  [team-PA hitter] {date_str} load failed: {type(e).__name__}: {e}")
        team_pa_bundle, rate_models = None, None

    pitcher_proj = pd.DataFrame()
    hitter_proj = pd.DataFrame()
    if not pitchers_today.empty:
        try:
            pitcher_proj = hpt.score_pitchers(
                pitchers_today, pitcher_game_df, pitcher_models,
                team_batting_ctx, target_ts, league_means,
                two_stage_models=two_stage_pitcher,
            )
        except Exception as e:
            print(f"  [score_pitchers] {date_str}: {type(e).__name__}: {e}")

    if not hitters_today.empty:
        try:
            hitter_proj = hpt.score_hitters(
                hitters_today, pitchers_today, hitter_game_df, pitcher_game_df,
                hitter_models, team_pitching_ctx, target_ts, league_means,
                team_pa_bundle=team_pa_bundle,
                rate_models=rate_models,
            )
        except Exception as e:
            print(f"  [score_hitters] {date_str}: {type(e).__name__}: {e}")

    if pitcher_proj.empty and hitter_proj.empty:
        return None

    out = pd.concat([pitcher_proj, hitter_proj], ignore_index=True, sort=False)
    out["game_date"] = date_str
    return out


# ---------------------------------------------------------------------------
# Diagnose mode — just report status, no work
# ---------------------------------------------------------------------------
def diagnose(start: str = SEASON_START_DEFAULT, end: str | None = None) -> None:
    end = end or _yesterday_et()
    dates = _date_range(start, end)
    if not dates:
        print("Empty date range — nothing to diagnose.")
        return

    counts = Counter()
    rows = []
    for d in dates:
        status = _snapshot_status(d)
        counts[status] += 1
        rows.append((d, status))

    print(f"\n=== Snapshot diagnosis for {dates[0]} → {dates[-1]} ({len(dates)} dates) ===")
    print(f"  with rows : {counts['present_with_rows']:>3}")
    print(f"  empty     : {counts['present_empty']:>3}")
    print(f"  missing   : {counts['missing']:>3}\n")

    bad = [(d, s) for d, s in rows if s != "present_with_rows"]
    if not bad:
        print("✓ All dates have populated snapshots.")
        return
    print("Dates that need work:")
    for d, s in bad:
        marker = "✗" if s == "missing" else "⚠"
        print(f"  {marker} {d}  [{s}]")


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------
def backfill(
    start: str = SEASON_START_DEFAULT,
    end: str | None = None,
    force: bool = False,
    grade: bool = True,
    sleep_between_dates: float = 0.0,
    verbose: bool = False,
) -> None:
    end = end or _yesterday_et()
    dates = _date_range(start, end)
    if not dates:
        print("Empty date range — nothing to do.")
        return

    print(f"\n=== Backfilling player projections for {len(dates)} date(s): {dates[0]} → {dates[-1]} ===")
    print(f"    force={force}  grade={grade}  verbose={verbose}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_built = 0
    n_skipped = 0
    n_empty = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []
    error_types: Counter = Counter()

    for i, date_str in enumerate(dates, start=1):
        snap = _snapshot_path(date_str)
        if snap.exists() and _snapshot_status(date_str) == "present_with_rows" and not force:
            n_skipped += 1
            print(f"[{i:>3}/{len(dates)}] {date_str}  ✓ skipped (snapshot already populated)")
            continue
        try:
            print(f"[{i:>3}/{len(dates)}] {date_str}  → fetching actual lineups + projecting…")
            df = _project_past_date(date_str, sleep_seconds=0.15)
            if df is None or df.empty:
                snap.write_text("")
                n_empty += 1
                print(f"             ⚠ empty (no finished games / no lineups)")
            else:
                df.to_csv(snap, index=False)
                n_built += 1
                print(f"             ✓ built ({len(df):,} player-games)")
        except Exception as e:
            n_failed += 1
            err_type = type(e).__name__
            err_msg = str(e)[:200]
            error_types[err_type] += 1
            failures.append((date_str, f"{err_type}: {err_msg}"))
            print(f"             ✗ failed: {err_type}: {err_msg}")
            if verbose:
                traceback.print_exc()
        if sleep_between_dates:
            time.sleep(sleep_between_dates)

    print(
        f"\n=== Backfill summary ===\n"
        f"  built   : {n_built}\n"
        f"  skipped : {n_skipped}\n"
        f"  empty   : {n_empty}\n"
        f"  failed  : {n_failed}\n"
        f"  total   : {len(dates)} dates"
    )

    coverage = n_built + n_skipped
    print(f"\nCoverage: {coverage}/{len(dates)} dates have populated snapshots.")

    if failures:
        print("\nFailure breakdown by error type:")
        for et, c in error_types.most_common():
            print(f"  {et}: {c}")
        print("\nFailed dates (first 20):")
        for d, err in failures[:20]:
            print(f"  {d}  →  {err}")
        if len(failures) > 20:
            print(f"  …and {len(failures) - 20} more")

    if not grade:
        print("\nGrading skipped (--no-grade).")
        return

    print("\n── Grading all snapshots against MLB box scores ───────────────")
    from grade_player_predictions import grade_player_predictions
    grade_player_predictions(
        snapshots_dir=str(OUT_DIR),
        output_file="2026_player_accuracy.csv",
        season_start=start,
    )
    print("✅ Backfill + grade complete.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=SEASON_START_DEFAULT,
                   help=f"First date YYYY-MM-DD (default {SEASON_START_DEFAULT})")
    p.add_argument("--end", default=None,
                   help="Last date YYYY-MM-DD (default: yesterday ET)")
    p.add_argument("--diagnose", action="store_true",
                   help="Just report which dates have populated snapshots — don't run anything.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate snapshots even if one already exists")
    p.add_argument("--no-grade", dest="grade", action="store_false", default=True,
                   help="Only build snapshots; skip the grade step")
    p.add_argument("--verbose", action="store_true",
                   help="Print full traceback for each failed date")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Seconds to sleep between dates")
    args = p.parse_args()

    if args.diagnose:
        diagnose(start=args.start, end=args.end)
        return

    backfill(
        start=args.start,
        end=args.end,
        force=args.force,
        grade=args.grade,
        sleep_between_dates=args.sleep,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
