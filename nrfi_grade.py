"""
nrfi_grade.py
=============
Grade past NRFI predictions against the MLB Stats API.

For every game in 2026_nrfi_picks.csv whose first inning has been played,
fetch the linescore via the MLB Stats API and record:
  - first_inning_runs_home, first_inning_runs_away
  - actual_nrfi (True if both are 0)
  - hit_yes (lean was YES and actual NRFI happened)
  - hit_no  (lean was NO  and actual NRFI did NOT happen)
  - correct (lean was YES and NRFI=True, OR lean was NO and NRFI=False)

Outputs
-------
2026_nrfi_accuracy.csv  — full pick log with grading columns added.

Usage
-----
    python nrfi_grade.py
    python nrfi_grade.py --picks-file 2026_nrfi_picks.csv --output 2026_nrfi_accuracy.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LINESCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"

FINISHED_STATES = {"Final", "Game Over", "Completed Early", "Completed"}
# We can grade NRFI as soon as the 1st inning ends — even mid-game.
GRADEABLE_STATES = FINISHED_STATES | {"In Progress", "Manager challenge"}


def _yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def _games_for_date(date_str: str, timeout: int = 30) -> list[dict]:
    """Schedule lookup — returns [{game_pk, away_abbr, home_abbr, status}]."""
    try:
        r = requests.get(SCHEDULE_URL, params={
            "sportId": 1, "date": date_str, "hydrate": "team",
        }, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [schedule] {date_str}: {e}")
        return []
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {}) or {}
            home = (teams.get("home", {}) or {}).get("team", {}) or {}
            away = (teams.get("away", {}) or {}).get("team", {}) or {}
            games.append({
                "game_pk": g.get("gamePk"),
                "home_abbr": home.get("abbreviation"),
                "away_abbr": away.get("abbreviation"),
                "status": (g.get("status") or {}).get("detailedState", ""),
            })
    return games


def _linescore_first_inning(game_pk: int, timeout: int = 30) -> dict | None:
    """Pull the game's linescore and extract 1st-inning runs for each side."""
    try:
        r = requests.get(LINESCORE_URL.format(game_pk=game_pk), timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [linescore] {game_pk}: {e}")
        return None
    innings = data.get("innings") or []
    if not innings:
        return None
    first = innings[0]
    home_runs = ((first.get("home") or {}).get("runs"))
    away_runs = ((first.get("away") or {}).get("runs"))
    if home_runs is None or away_runs is None:
        return None
    return {"home_runs_1st": int(home_runs), "away_runs_1st": int(away_runs)}


def _normalize_team_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([str(a or "").strip(), str(b or "").strip()]))


def grade_nrfi_picks(picks_file: str = "2026_nrfi_picks.csv",
                     output_file: str = "2026_nrfi_accuracy.csv",
                     sleep_seconds: float = 0.2) -> pd.DataFrame:
    picks_path = Path(picks_file)
    if not picks_path.exists():
        print(f"⚠️  {picks_path} doesn't exist yet — nothing to grade.")
        return pd.DataFrame()

    df = pd.read_csv(picks_path, low_memory=False)
    if df.empty:
        print(f"⚠️  {picks_path} is empty.")
        return df

    df["game_date"] = df["game_date"].astype(str).str[:10]

    # Pull schedule for every unique date in the picks log (cached per date).
    dates = sorted(df["game_date"].dropna().unique().tolist())
    print(f"Grading NRFI across {len(dates)} date(s)…")

    schedule_by_date: dict[str, list[dict]] = {}
    for d in dates:
        schedule_by_date[d] = _games_for_date(d)

    # Build a (date, team_pair) → game_pk lookup
    pk_lookup: dict[tuple, dict] = {}
    for d, games in schedule_by_date.items():
        for g in games:
            if g["home_abbr"] and g["away_abbr"] and g["game_pk"]:
                pair = _normalize_team_pair(g["home_abbr"], g["away_abbr"])
                pk_lookup[(d, pair)] = g

    # For each pick, find its game_pk + linescore + grade
    home_runs_1st = []
    away_runs_1st = []
    actual_nrfi = []
    correct = []
    statuses = []

    fetched: dict[int, dict] = {}
    for _, row in df.iterrows():
        d = str(row.get("game_date", ""))[:10]
        pair = _normalize_team_pair(row.get("team_a"), row.get("team_b"))
        info = pk_lookup.get((d, pair))
        if not info:
            home_runs_1st.append(None); away_runs_1st.append(None)
            actual_nrfi.append(None); correct.append(None); statuses.append("not_found")
            continue
        gpk = int(info["game_pk"])
        statuses.append(info.get("status", ""))
        if info.get("status") not in GRADEABLE_STATES:
            home_runs_1st.append(None); away_runs_1st.append(None)
            actual_nrfi.append(None); correct.append(None)
            continue
        if gpk not in fetched:
            fetched[gpk] = _linescore_first_inning(gpk) or {}
            time.sleep(sleep_seconds)
        ls = fetched[gpk]
        if not ls or "home_runs_1st" not in ls:
            home_runs_1st.append(None); away_runs_1st.append(None)
            actual_nrfi.append(None); correct.append(None)
            continue
        h = ls["home_runs_1st"]; a = ls["away_runs_1st"]
        nrfi = (h == 0 and a == 0)
        lean = str(row.get("lean", "")).upper()
        is_correct = None
        if lean == "YES":
            is_correct = bool(nrfi)
        elif lean == "NO":
            is_correct = not bool(nrfi)
        # PASS picks aren't graded for accuracy (we didn't lean either way)

        home_runs_1st.append(h); away_runs_1st.append(a)
        actual_nrfi.append(bool(nrfi)); correct.append(is_correct)

    df["home_runs_1st"] = home_runs_1st
    df["away_runs_1st"] = away_runs_1st
    df["actual_nrfi"] = actual_nrfi
    df["correct"] = correct
    df["status"] = statuses

    n_total = len(df)
    n_graded = int(df["correct"].notna().sum())
    n_yes_picks = int((df["lean"].astype(str).str.upper() == "YES").sum())
    n_no_picks = int((df["lean"].astype(str).str.upper() == "NO").sum())
    n_correct = int(df["correct"].sum()) if n_graded else 0

    df.to_csv(output_file, index=False)
    print(f"\nWrote {n_total} rows ({n_graded} graded, {n_correct} correct) → {output_file}")
    print(f"  YES picks: {n_yes_picks}  NO picks: {n_no_picks}  PASS: {n_total - n_yes_picks - n_no_picks}")
    if n_graded:
        accuracy = n_correct / n_graded * 100
        print(f"  Lean accuracy on graded picks: {accuracy:.1f}%")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks-file", default="2026_nrfi_picks.csv")
    p.add_argument("--output", default="2026_nrfi_accuracy.csv")
    args = p.parse_args()
    grade_nrfi_picks(picks_file=args.picks_file, output_file=args.output)


if __name__ == "__main__":
    main()
