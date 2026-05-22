"""
grade_player_predictions.py
===========================
Joins each day's saved player projections (outputs/hitterspitchers_<date>.csv)
to actual MLB box-score results pulled from the MLB Stats API, and writes the
graded rows to ``2026_player_accuracy.csv``.

This is the player-side analogue of ``grade_saved_picks`` in
``daily_mlb_model_runner.py``. It is idempotent — it always rebuilds the log
from every dated snapshot it can find, so re-running it on later dates just
fills in any newly-completed games.

Public entrypoint
-----------------
    grade_player_predictions(
        snapshots_dir="outputs",
        output_file="2026_player_accuracy.csv",
        season_start="2026-03-25",
    )

Output columns
--------------
    game_date, game_pk, player_type, player_name, mlb_id, team, opponent,
    proj_pa, proj_hits, proj_hr, proj_strikeouts, proj_walks, proj_runs,
    proj_rbi, proj_ip, proj_hits_allowed, proj_runs_allowed, lineup_status,
    confidence, confidence_score,
    actual_pa, actual_hits, actual_hr, actual_strikeouts, actual_walks,
    actual_runs, actual_rbi, actual_ip, actual_hits_allowed,
    actual_runs_allowed, status, played
"""

from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

DATE_FROM_NAME = re.compile(r"hitterspitchers_(\d{4}-\d{2}-\d{2})\.csv$")


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------
def _games_finished_for_date(date_str: str, timeout: int = 30) -> pd.DataFrame:
    """Return a frame of (game_pk, status) for games on a given date."""
    try:
        r = requests.get(
            SCHEDULE_URL,
            params={"sportId": 1, "date": date_str},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [schedule] {date_str}: {e}")
        return pd.DataFrame(columns=["game_pk", "status"])

    rows = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            rows.append({
                "game_pk": g.get("gamePk"),
                "status": (g.get("status") or {}).get("detailedState", ""),
            })
    return pd.DataFrame(rows)


def _ip_to_outs(ip_val) -> Optional[float]:
    """Convert MLB Stats API innings-pitched string ('5.2' = 5⅔ IP) to outs."""
    if ip_val is None or pd.isna(ip_val):
        return None
    try:
        s = str(ip_val).strip()
        if not s:
            return None
        whole, _, frac = s.partition(".")
        outs = int(whole) * 3
        if frac:
            outs += int(frac[0])  # only first digit matters: .0 .1 .2
        return float(outs)
    except (ValueError, TypeError):
        return None


def _outs_to_ip(outs) -> Optional[float]:
    """Convert outs back to a float IP for display (e.g. 17 outs -> 5.667)."""
    if outs is None or pd.isna(outs):
        return None
    try:
        return float(outs) / 3.0
    except (ValueError, TypeError):
        return None


def _box_for_game(game_pk: int, timeout: int = 30) -> dict:
    """Fetch a game's boxscore. Returns {'hitters': [...], 'pitchers': [...]}."""
    try:
        r = requests.get(BOXSCORE_URL.format(game_pk=game_pk), timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [box] {game_pk}: {e}")
        return {"hitters": [], "pitchers": []}

    hitters: list[dict] = []
    pitchers: list[dict] = []

    for side in ("home", "away"):
        team_block = (data.get("teams") or {}).get(side) or {}
        players = team_block.get("players") or {}
        for _pid, pdata in players.items():
            person = pdata.get("person") or {}
            mlb_id = person.get("id")
            name = person.get("fullName")
            stats = pdata.get("stats") or {}

            batting = stats.get("batting") or {}
            pitching = stats.get("pitching") or {}

            # Hitter row — only count if they actually batted.
            pa = batting.get("plateAppearances")
            ab = batting.get("atBats")
            if (pa not in (None, 0)) or (ab not in (None, 0)):
                hitters.append({
                    "mlb_id": mlb_id,
                    "player_name": name,
                    "actual_pa": batting.get("plateAppearances"),
                    "actual_hits": batting.get("hits"),
                    "actual_hr": batting.get("homeRuns"),
                    "actual_strikeouts_h": batting.get("strikeOuts"),
                    "actual_walks_h": batting.get("baseOnBalls"),
                    "actual_runs": batting.get("runs"),
                    "actual_rbi": batting.get("rbi"),
                })

            # Pitcher row — only if they recorded outs.
            outs_pitched = _ip_to_outs(pitching.get("inningsPitched"))
            if outs_pitched and outs_pitched > 0:
                pitchers.append({
                    "mlb_id": mlb_id,
                    "player_name": name,
                    "actual_outs": outs_pitched,
                    "actual_ip": _outs_to_ip(outs_pitched),
                    "actual_hits_allowed": pitching.get("hits"),
                    "actual_runs_allowed": pitching.get("earnedRuns"),
                    "actual_strikeouts_p": pitching.get("strikeOuts"),
                    "actual_walks_p": pitching.get("baseOnBalls"),
                })

    return {"hitters": hitters, "pitchers": pitchers}


# ---------------------------------------------------------------------------
# Snapshot loaders
# ---------------------------------------------------------------------------
PROJ_COLUMNS_KEEP = [
    "player_type", "player_name", "mlb_id", "team", "opponent",
    "proj_pa", "proj_hits", "proj_hr", "proj_strikeouts", "proj_walks",
    "proj_runs", "proj_rbi",
    "proj_ip", "proj_hits_allowed", "proj_runs_allowed",
    "lineup_status", "lineup_spot", "confidence", "confidence_score",
    "used_fallback",
]


def _load_snapshots(snapshots_dir: Path, season_start: str) -> pd.DataFrame:
    """Read every dated player-projection snapshot we have on disk."""
    paths = sorted(glob.glob(str(snapshots_dir / "hitterspitchers_*.csv")))
    frames: list[pd.DataFrame] = []
    season_start_ts = pd.to_datetime(season_start)

    for p in paths:
        m = DATE_FROM_NAME.search(os.path.basename(p))
        if not m:
            continue
        date_str = m.group(1)
        ts = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(ts) or ts < season_start_ts:
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"  [snap] could not read {p}: {e}")
            continue
        df["snapshot_date"] = date_str
        # Some snapshots may not have game_date — fall back to filename date.
        if "game_date" not in df.columns:
            df["game_date"] = date_str
        else:
            df["game_date"] = df["game_date"].fillna(date_str)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    keep = [c for c in PROJ_COLUMNS_KEEP if c in out.columns]
    out = out[["snapshot_date", "game_date"] + keep].copy()
    return out


# ---------------------------------------------------------------------------
# Box-score fetch by date
# ---------------------------------------------------------------------------
def _fetch_actuals_for_dates(dates: list[str], sleep_seconds: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each date, fetch the schedule, then one boxscore per finished game.
    Returns (hitter_actuals_df, pitcher_actuals_df, status_df).
    """
    hitter_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    status_rows: list[dict] = []

    finished_states = {
        "Final", "Game Over", "Completed Early", "Completed",
    }

    for date_str in dates:
        sched = _games_finished_for_date(date_str)
        if sched.empty:
            continue
        for _, row in sched.iterrows():
            gp = row["game_pk"]
            st = row["status"]
            status_rows.append({"game_date": date_str, "game_pk": gp, "status": st})
            if st not in finished_states:
                continue
            box = _box_for_game(int(gp))
            for h in box["hitters"]:
                h["game_date"] = date_str
                h["game_pk"] = gp
                hitter_rows.append(h)
            for p in box["pitchers"]:
                p["game_date"] = date_str
                p["game_pk"] = gp
                pitcher_rows.append(p)
            time.sleep(sleep_seconds)

    return (
        pd.DataFrame(hitter_rows),
        pd.DataFrame(pitcher_rows),
        pd.DataFrame(status_rows),
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def grade_player_predictions(
    snapshots_dir: str | Path = "outputs",
    output_file: str | Path = "2026_player_accuracy.csv",
    season_start: str = "2026-03-25",
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    snapshots_dir = Path(snapshots_dir)
    output_file = Path(output_file)

    snaps = _load_snapshots(snapshots_dir, season_start=season_start)
    if snaps.empty:
        print("No dated player-projection snapshots found yet.")
        # Still write an empty file so the API has something to read.
        if not output_file.exists():
            pd.DataFrame().to_csv(output_file, index=False)
        return pd.DataFrame()

    # We only want one row per (player, game_date) — pick the snapshot
    # taken on the same day as the game (most accurate projection set).
    snaps["matches_game_day"] = snaps["snapshot_date"] == snaps["game_date"]
    snaps = snaps.sort_values(
        ["mlb_id", "game_date", "matches_game_day", "snapshot_date"],
        ascending=[True, True, False, False],
    )
    snaps = snaps.drop_duplicates(subset=["mlb_id", "game_date"], keep="first")
    snaps = snaps.drop(columns=["matches_game_day"])

    dates = sorted(snaps["game_date"].dropna().unique().tolist())
    print(f"Grading player projections across {len(dates)} date(s)…")

    hitters_act, pitchers_act, status_df = _fetch_actuals_for_dates(
        dates, sleep_seconds=sleep_seconds,
    )

    # Split projections by player_type for the merges.
    pitchers_proj = snaps[snaps["player_type"].astype(str).str.lower() == "pitcher"].copy()
    hitters_proj = snaps[snaps["player_type"].astype(str).str.lower() == "hitter"].copy()

    # ── HITTERS merge ────────────────────────────────────────────────────
    if not hitters_proj.empty:
        if hitters_act.empty:
            hitters_proj = hitters_proj.assign(
                game_pk=np.nan,
                actual_pa=np.nan, actual_hits=np.nan, actual_hr=np.nan,
                actual_strikeouts=np.nan, actual_walks=np.nan,
                actual_runs=np.nan, actual_rbi=np.nan, played=False,
            )
            hitters_graded = hitters_proj
        else:
            hitters_act = hitters_act.rename(columns={
                "actual_strikeouts_h": "actual_strikeouts",
                "actual_walks_h": "actual_walks",
            })
            # Use mlb_id as the join key.
            hitters_proj["mlb_id"] = pd.to_numeric(hitters_proj["mlb_id"], errors="coerce")
            hitters_act["mlb_id"] = pd.to_numeric(hitters_act["mlb_id"], errors="coerce")
            hitters_graded = hitters_proj.merge(
                hitters_act[[
                    "mlb_id", "game_date", "game_pk",
                    "actual_pa", "actual_hits", "actual_hr",
                    "actual_strikeouts", "actual_walks",
                    "actual_runs", "actual_rbi",
                ]],
                on=["mlb_id", "game_date"],
                how="left",
            )
            hitters_graded["played"] = hitters_graded["actual_pa"].notna()
    else:
        hitters_graded = pd.DataFrame()

    # ── PITCHERS merge ───────────────────────────────────────────────────
    if not pitchers_proj.empty:
        if pitchers_act.empty:
            pitchers_proj = pitchers_proj.assign(
                game_pk=np.nan,
                actual_ip=np.nan, actual_outs=np.nan,
                actual_strikeouts=np.nan, actual_walks=np.nan,
                actual_hits_allowed=np.nan, actual_runs_allowed=np.nan,
                played=False,
            )
            pitchers_graded = pitchers_proj
        else:
            pitchers_act = pitchers_act.rename(columns={
                "actual_strikeouts_p": "actual_strikeouts",
                "actual_walks_p": "actual_walks",
            })
            pitchers_proj["mlb_id"] = pd.to_numeric(pitchers_proj["mlb_id"], errors="coerce")
            pitchers_act["mlb_id"] = pd.to_numeric(pitchers_act["mlb_id"], errors="coerce")
            pitchers_graded = pitchers_proj.merge(
                pitchers_act[[
                    "mlb_id", "game_date", "game_pk",
                    "actual_ip", "actual_outs",
                    "actual_strikeouts", "actual_walks",
                    "actual_hits_allowed", "actual_runs_allowed",
                ]],
                on=["mlb_id", "game_date"],
                how="left",
            )
            pitchers_graded["played"] = pitchers_graded["actual_ip"].notna()
    else:
        pitchers_graded = pd.DataFrame()

    if not status_df.empty:
        status_df = status_df.drop_duplicates(subset=["game_pk"])

    graded = pd.concat([hitters_graded, pitchers_graded], ignore_index=True, sort=False)
    if graded.empty:
        print("Nothing to grade.")
        graded.to_csv(output_file, index=False)
        return graded

    if not status_df.empty and "game_pk" in graded.columns:
        graded = graded.merge(
            status_df.rename(columns={"game_date": "_sd"})[["game_pk", "status"]],
            on="game_pk", how="left",
        )
    else:
        graded["status"] = np.nan

    # Keep a stable column ordering for downstream consumers.
    preferred = [
        "game_date", "snapshot_date", "game_pk", "player_type", "player_name",
        "mlb_id", "team", "opponent",
        "proj_pa", "proj_hits", "proj_hr", "proj_strikeouts", "proj_walks",
        "proj_runs", "proj_rbi",
        "proj_ip", "proj_hits_allowed", "proj_runs_allowed",
        "actual_pa", "actual_hits", "actual_hr",
        "actual_strikeouts", "actual_walks", "actual_runs", "actual_rbi",
        "actual_ip", "actual_outs", "actual_hits_allowed", "actual_runs_allowed",
        "lineup_status", "lineup_spot", "confidence", "confidence_score",
        "used_fallback", "status", "played",
    ]
    cols = [c for c in preferred if c in graded.columns]
    extras = [c for c in graded.columns if c not in cols]
    graded = graded[cols + extras]
    graded = graded.sort_values(["game_date", "player_type", "player_name"], na_position="last")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    graded.to_csv(output_file, index=False)

    n_played = int(graded["played"].fillna(False).sum())
    print(f"Wrote {len(graded):,} rows ({n_played:,} graded) → {output_file}")
    return graded


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--snapshots-dir", default="outputs")
    p.add_argument("--output", default="2026_player_accuracy.csv")
    p.add_argument("--season-start", default="2026-03-25")
    args = p.parse_args()

    grade_player_predictions(
        snapshots_dir=args.snapshots_dir,
        output_file=args.output,
        season_start=args.season_start,
    )
