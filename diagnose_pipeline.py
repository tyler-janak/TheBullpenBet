"""
diagnose_pipeline.py
====================
Run this from the website folder to find out exactly what's broken in
the daily pipeline. It checks every prerequisite the cron depends on
and prints PASS / FAIL per check, then runs a single past date through
the backfill loop verbosely so you can see exactly where it's failing.

Usage
-----
    python diagnose_pipeline.py                       # full health check
    python diagnose_pipeline.py --test-date 2026-04-25  # also exercise that date

Exit code is 0 if everything looks healthy, non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def _yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    line = f"  {icon}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


def warn(label: str, detail: str = "") -> None:
    line = f"  {WARN}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


# ---------------------------------------------------------------------------
# 1. Files & directories
# ---------------------------------------------------------------------------
def check_files() -> bool:
    header("Step 1: file & directory presence")
    ok = True
    must_exist = [
        ("daily_update.py", True),
        ("daily_mlb_model_runner.py", True),
        ("hitterspitchers_today.py", True),
        ("hitterspitchers_data.py", True),
        ("backfill_player_predictions.py", True),
        ("grade_player_predictions.py", True),
        ("refresh_2026_data.py", True),
        ("calibrate_projections.py", True),
        ("server.py", True),
        ("index.html", True),
        ("requirements.txt", True),
        ("data/hitter_game_data.csv", True),
        ("data/pitcher_game_data.csv", True),
        ("data/team_batting_hand_context.csv", True),
        ("data/team_pitching_hand_context.csv", True),
        ("betting_model.pkl", True),
        ("2025_model_data.csv", True),
    ]
    for path, required in must_exist:
        p = HERE / path
        present = p.exists()
        ok = check(f"{path}", present, "" if present else "MISSING") and ok
    return ok


# ---------------------------------------------------------------------------
# 2. Python imports
# ---------------------------------------------------------------------------
def check_imports() -> bool:
    header("Step 2: Python module imports")
    ok = True
    sys.path.insert(0, str(HERE))

    deps = [
        ("pandas", True),
        ("numpy", True),
        ("requests", True),
        ("fastapi", True),
        ("xgboost", True),
        ("sklearn", True),
        ("pybaseball", True),
        ("bs4", True),    # beautifulsoup4
    ]
    for name, required in deps:
        try:
            importlib.import_module(name)
            check(f"import {name}", True)
        except Exception as e:
            ok = check(f"import {name}", False, f"{type(e).__name__}: {e}") and ok

    own = [
        "daily_mlb_model_runner",
        "hitterspitchers_today",
        "hitterspitchers_data",
        "backfill_player_predictions",
        "grade_player_predictions",
        "refresh_2026_data",
        "calibrate_projections",
    ]
    for mod in own:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as e:
            ok = check(f"import {mod}", False, f"{type(e).__name__}: {str(e)[:120]}") and ok

    return ok


# ---------------------------------------------------------------------------
# 3. Data freshness
# ---------------------------------------------------------------------------
def check_data_freshness() -> bool:
    header("Step 3: data freshness")
    ok = True

    for path in ("data/hitter_game_data.csv", "data/pitcher_game_data.csv"):
        p = HERE / path
        if not p.exists():
            check(path, False, "missing")
            ok = False
            continue
        try:
            df = pd.read_csv(p, usecols=["game_date"], low_memory=False)
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            n_2026 = int((df["game_date"].dt.year == 2026).sum())
            latest = df["game_date"].max()
            latest_str = latest.strftime("%Y-%m-%d") if pd.notna(latest) else "—"
            has_2026 = n_2026 > 0
            check(f"{path}: latest game_date = {latest_str}, 2026 rows = {n_2026:,}",
                  has_2026,
                  "" if has_2026
                  else "ZERO 2026 rows — refresh_2026_data.py hasn't successfully run yet. "
                       "Past-date projections will be based on 2025-only history.")
            if not has_2026:
                ok = False
        except Exception as e:
            check(f"{path}", False, f"{type(e).__name__}: {e}")
            ok = False

    # Today's projection — is the live alias actually fresh?
    today_path = HERE / "outputs" / "hitterspitchers_today.csv"
    today = _today_et()
    if not today_path.exists():
        check(f"outputs/hitterspitchers_today.csv", False, "missing — daily pipeline hasn't generated today's projection")
        ok = False
    else:
        try:
            df = pd.read_csv(today_path, usecols=["game_date"], low_memory=False)
            file_dates = set(df["game_date"].dropna().astype(str).str[:10].unique().tolist())
            is_fresh = today in file_dates
            check(f"outputs/hitterspitchers_today.csv carries today's date ({today})",
                  is_fresh,
                  f"file has dates {sorted(file_dates)[:3]} — stale")
            if not is_fresh:
                ok = False
        except Exception as e:
            check(f"outputs/hitterspitchers_today.csv", False, f"{type(e).__name__}: {e}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# 4. Past-date snapshot coverage
# ---------------------------------------------------------------------------
def check_snapshot_coverage(season_start: str = "2026-03-25") -> bool:
    header("Step 4: past-date snapshot coverage")
    out_dir = HERE / "outputs"
    if not out_dir.exists():
        check("outputs/ directory", False, "missing")
        return False

    end = _yesterday_et()
    s = datetime.strptime(season_start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        warn("Season hasn't started yet — nothing to backfill")
        return True

    expected_dates = []
    cur = s
    while cur <= e:
        expected_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    populated = 0
    empty = 0
    missing = 0
    sample_missing: list[str] = []
    sample_empty: list[str] = []
    for d in expected_dates:
        p = out_dir / f"hitterspitchers_{d}.csv"
        if not p.exists():
            missing += 1
            if len(sample_missing) < 5:
                sample_missing.append(d)
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            missing += 1
            continue
        if sz < 64:
            empty += 1
            if len(sample_empty) < 5:
                sample_empty.append(d)
        else:
            populated += 1

    n = len(expected_dates)
    coverage_pct = 100.0 * populated / max(1, n)
    print(f"  Coverage: {populated}/{n} dates have populated snapshots ({coverage_pct:.1f}%)")
    print(f"  Empty markers: {empty}")
    print(f"  Missing: {missing}")
    if sample_missing:
        print(f"  Example missing dates: {', '.join(sample_missing)}")
    if sample_empty:
        print(f"  Example empty dates: {', '.join(sample_empty)}")

    if populated == 0 and n > 1:
        warn("No past-date snapshots exist. Run:")
        print(f"      python backfill_player_predictions.py --start {season_start}")
        return False
    return True


# ---------------------------------------------------------------------------
# 5. Player accuracy log row counts
# ---------------------------------------------------------------------------
def check_player_accuracy_log() -> bool:
    header("Step 5: player accuracy log")
    p = HERE / "2026_player_accuracy.csv"
    if not p.exists():
        check("2026_player_accuracy.csv", False, "missing — grader hasn't run yet")
        return False
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        check("2026_player_accuracy.csv readable", False, f"{type(e).__name__}: {e}")
        return False
    n = len(df)
    if n == 0:
        check("2026_player_accuracy.csv has rows", False, "0 rows logged")
        return False
    n_played = 0
    n_pitchers = 0
    n_hitters = 0
    if "played" in df.columns:
        n_played = int(df["played"].astype(str).str.lower().isin({"true","1","1.0"}).sum())
    if "player_type" in df.columns:
        n_pitchers = int((df["player_type"].astype(str).str.lower() == "pitcher").sum())
        n_hitters = int((df["player_type"].astype(str).str.lower() == "hitter").sum())
    check(f"2026_player_accuracy.csv: {n:,} rows, {n_played:,} graded ({n_hitters:,} hitter / {n_pitchers:,} pitcher)",
          n_played > 0,
          "" if n_played > 0 else "no graded rows yet")
    return n_played > 0


# ---------------------------------------------------------------------------
# 6. Test backfill on a single date verbosely
# ---------------------------------------------------------------------------
def test_backfill_one_date(date_str: str) -> bool:
    header(f"Step 6: dry-run backfill for {date_str}")
    sys.path.insert(0, str(HERE))

    try:
        import backfill_player_predictions as bpp
    except Exception as e:
        check(f"import backfill_player_predictions", False, f"{type(e).__name__}: {e}")
        return False

    print(f"\n  Calling _build_past_inputs('{date_str}')…")
    try:
        pitchers_today, hitters_today = bpp._build_past_inputs(date_str, sleep_seconds=0.15)
        print(f"  → pitchers_today: {len(pitchers_today)} rows")
        print(f"  → hitters_today : {len(hitters_today)} rows")
        if pitchers_today.empty and hitters_today.empty:
            warn("MLB Stats API returned no finished games / lineups for this date")
            return False
        # Show the first row of each so the user can sanity-check schema.
        if not pitchers_today.empty:
            print("\n  First pitcher row:")
            for k, v in pitchers_today.iloc[0].items():
                print(f"    {k:20s} {v}")
        if not hitters_today.empty:
            print("\n  First hitter row:")
            for k, v in hitters_today.iloc[0].items():
                print(f"    {k:20s} {v}")
    except Exception as e:
        check("_build_past_inputs", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    print(f"\n  Calling _project_past_date('{date_str}')…")
    try:
        df = bpp._project_past_date(date_str, sleep_seconds=0.15)
        if df is None or df.empty:
            warn("Projections came back empty. The score functions may have rejected the input shape, OR data/hitter_game_data.csv has no usable history for the players who batted that day.")
            return False
        print(f"  → produced {len(df):,} player-game projections")
        print(f"  → first few rows:")
        cols_to_show = [c for c in ["player_type","player_name","team","opponent",
                                     "proj_pa","proj_hits","proj_strikeouts",
                                     "proj_ip","proj_strikeouts","proj_walks"]
                         if c in df.columns]
        print(df[cols_to_show].head().to_string(index=False))
    except Exception as e:
        check("_project_past_date", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-date", default=None,
                   help="Optional past date to dry-run through the backfill (e.g. 2026-04-25)")
    p.add_argument("--season-start", default="2026-03-25")
    args = p.parse_args()

    print(f"\n  Diagnostic pass for {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"  Working directory: {HERE}")

    results = []
    results.append(("files",         check_files()))
    results.append(("imports",       check_imports()))
    results.append(("data freshness", check_data_freshness()))
    results.append(("snapshots",     check_snapshot_coverage(season_start=args.season_start)))
    results.append(("accuracy log",  check_player_accuracy_log()))

    if args.test_date:
        results.append((f"backfill {args.test_date}", test_backfill_one_date(args.test_date)))

    header("Summary")
    all_ok = True
    for name, ok in results:
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✅ All checks passed.")
        return 0

    print("\n❌ One or more checks failed. Most common fixes:")
    print("   1. If the data refresh check failed, run:")
    print("        python refresh_2026_data.py")
    print("   2. If snapshot coverage is empty, run:")
    print("        python backfill_player_predictions.py --start 2026-03-25 --verbose")
    print("   3. If today's projection is stale, run:")
    print("        python -c \"from hitterspitchers_today import run_projections; "
          "run_projections()\"")
    print("   4. Then re-run this diagnostic.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
