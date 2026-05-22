"""
nrfi.py
=======
"No Run First Inning" (NRFI) projections.

Concept
-------
NRFI = neither team scores in the 1st inning. P(NRFI) = P(home blanks 1st) × P(away blanks 1st).

Each side's "blank the 1st" probability is built from:
  - The opposing starter's first-inning suppression (their K rate against the
    top of the order)
  - The opposing top-5 hitters' OBP and SLG (the part of the lineup that
    always bats in the first)

We model the expected runs allowed in the 1st inning as a Poisson process
and return P(0 runs in 1st) = exp(−lambda). Lambda is built from the
opposing top-5 hitters' base-runner expectancy minus the starter's K credit:

    lambda = max(0.05,
                 OBP_WEIGHT * top5_obp * (1 + SLG_WEIGHT * top5_slg)
                 - K_CREDIT * pitcher_k_rate * OBP_WEIGHT)

The OBP_WEIGHT factor (~3) reflects ~3 batters faced in a clean half-inning.

This is intentionally not a deep-learning model — it's a transparent
heuristic with three knobs (`OBP_WEIGHT`, `SLG_WEIGHT`, `K_CREDIT`) you can
tune as you accumulate graded NRFI outcomes. Once you have enough graded
data this can be swapped for a trained classifier without changing the
output schema, so the API + UI keep working.

Outputs
-------
For each game today:
  - team_a, team_b, starters, NRFI probability
  - Per-side breakdown (pitcher + opposing top-5 numbers)
  - "Lean" — YES if P(NRFI) ≥ 0.55, NO if ≤ 0.45, otherwise PASS
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
HITTER_GAMES = DATA_DIR / "hitter_game_data.csv"
PITCHER_GAMES = DATA_DIR / "pitcher_game_data.csv"
TODAY_PROJ = OUT_DIR / "hitterspitchers_today.csv"

# Tunable weights — start with reasonable defaults
OBP_WEIGHT = 3.0   # ≈ batters faced in a clean half-inning
SLG_WEIGHT = 0.5   # how much SLG inflates expected damage given a runner
K_CREDIT  = 1.0    # full credit for each pitcher K (removes a hitter's chance)

LEAN_YES_THRESHOLD = 0.55
LEAN_NO_THRESHOLD  = 0.45

# League average half-inning run rate ≈ 0.55 — used as a fallback when an
# input is missing so the projection doesn't blow up.
LEAGUE_AVG_HALF_INNING_LAMBDA = 0.55


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v, default: float | None = None) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _today_str() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _hitter_rate_summary(hitter_df: pd.DataFrame, batter_id, target_date: pd.Timestamp) -> dict:
    """OBP-ish and SLG-ish rates over the last 20 games before target_date."""
    if batter_id is None or pd.isna(batter_id):
        return {"obp": None, "slg": None, "games": 0}
    sub = hitter_df[
        (hitter_df["batter"] == batter_id)
        & (hitter_df["game_date"] < target_date)
    ]
    if sub.empty:
        return {"obp": None, "slg": None, "games": 0}
    sub = sub.sort_values("game_date").tail(20)
    pa = pd.to_numeric(sub["PA"], errors="coerce").fillna(0).sum()
    h  = pd.to_numeric(sub["H"],  errors="coerce").fillna(0).sum()
    bb = pd.to_numeric(sub["BB"], errors="coerce").fillna(0).sum()
    hr = pd.to_numeric(sub["HR"], errors="coerce").fillna(0).sum()
    if pa <= 0:
        return {"obp": None, "slg": None, "games": int(len(sub))}
    obp = (h + bb) / pa
    # SLG approximation: only HR is a known XBH; the rest of H is treated as 1B.
    ab = max(pa - bb, 1)
    tb = (h - hr) + 4 * hr
    slg = tb / ab
    return {"obp": float(obp), "slg": float(slg), "games": int(len(sub))}


def _pitcher_first_inning_strength(pitcher_df: pd.DataFrame, pitcher_id, target_date: pd.Timestamp) -> dict:
    """K rate over the starter's last 10 starts before target_date."""
    if pitcher_id is None or pd.isna(pitcher_id):
        return {"k_rate": None, "starts": 0}
    sub = pitcher_df[
        (pitcher_df["pitcher"] == pitcher_id)
        & (pitcher_df["game_date"] < target_date)
    ]
    if sub.empty:
        return {"k_rate": None, "starts": 0}
    sub = sub.sort_values("game_date").tail(10)
    bf = pd.to_numeric(sub["BF"], errors="coerce").fillna(0).sum()
    k  = pd.to_numeric(sub["K"],  errors="coerce").fillna(0).sum()
    if bf <= 0:
        return {"k_rate": None, "starts": int(len(sub))}
    return {"k_rate": float(k / bf), "starts": int(len(sub))}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
@dataclass
class SideNRFI:
    """One side's contribution to NRFI (e.g. 'home blanks 1st')."""
    pitcher_name: str
    pitcher_k_rate: Optional[float]
    top5_obp: Optional[float]
    top5_slg: Optional[float]
    runs_lambda: float
    p_zero: float


def _side_p_zero(opposing_pitcher: dict, opposing_top5: list[dict]) -> SideNRFI:
    """
    Compute P(this side blanks the 1st). The 'side' here is the *defense* —
    i.e. the home team's pitcher facing the away team's top-5 lineup.
    """
    k_rate = opposing_pitcher.get("k_rate")
    obps = [h.get("obp") for h in opposing_top5 if h.get("obp") is not None]
    slgs = [h.get("slg") for h in opposing_top5 if h.get("slg") is not None]

    avg_obp = float(np.mean(obps)) if obps else None
    avg_slg = float(np.mean(slgs)) if slgs else None

    # If we don't have enough data, fall back to league-average lambda.
    if avg_obp is None or k_rate is None:
        lam = LEAGUE_AVG_HALF_INNING_LAMBDA
    else:
        baserunner_term = OBP_WEIGHT * avg_obp
        slg_inflation = 1 + (SLG_WEIGHT * (avg_slg or 0.40))
        damage_potential = baserunner_term * slg_inflation
        k_subtractor = K_CREDIT * k_rate * OBP_WEIGHT
        lam = max(0.05, damage_potential - k_subtractor)

    p_zero = math.exp(-lam)
    return SideNRFI(
        pitcher_name=opposing_pitcher.get("name", "—"),
        pitcher_k_rate=k_rate,
        top5_obp=avg_obp,
        top5_slg=avg_slg,
        runs_lambda=lam,
        p_zero=p_zero,
    )


def _today_projections() -> pd.DataFrame:
    """Read today's hitter+pitcher projection table."""
    if not TODAY_PROJ.exists():
        return pd.DataFrame()
    return pd.read_csv(TODAY_PROJ, low_memory=False)


def _games_today(proj: pd.DataFrame) -> list[dict]:
    """
    Pull (team_a, team_b) pairs plus each side's pitcher and top-5 batters
    from today's projection table. Each game appears once.
    """
    if proj.empty or "team" not in proj.columns or "opponent" not in proj.columns:
        return []

    pitchers = proj[proj["player_type"].astype(str).str.lower() == "pitcher"]
    hitters  = proj[proj["player_type"].astype(str).str.lower() == "hitter"]

    games: list[dict] = []
    seen_pairs: set = set()
    for _, p in pitchers.iterrows():
        team = str(p.get("team", ""))
        opp  = str(p.get("opponent", ""))
        if not team or not opp:
            continue
        # Canonicalize the matchup so we don't add it twice (once per starter).
        pair = tuple(sorted([team, opp]))
        if pair in seen_pairs:
            # Already added this matchup; enrich it with the second pitcher.
            for g in games:
                if {g["team_a"], g["team_b"]} == set(pair):
                    if team == pair[0] and g.get("pitcher_a") is None:
                        g["pitcher_a"] = {"name": p.get("player_name", ""),
                                          "mlb_id": _safe_float(p.get("mlb_id"), None),
                                          "team": team}
                    elif team == pair[1] and g.get("pitcher_b") is None:
                        g["pitcher_b"] = {"name": p.get("player_name", ""),
                                          "mlb_id": _safe_float(p.get("mlb_id"), None),
                                          "team": team}
                    break
            continue
        seen_pairs.add(pair)
        games.append({
            "team_a": pair[0],
            "team_b": pair[1],
            "pitcher_a": {"name": p.get("player_name", ""),
                          "mlb_id": _safe_float(p.get("mlb_id"), None),
                          "team": team} if team == pair[0] else None,
            "pitcher_b": {"name": p.get("player_name", ""),
                          "mlb_id": _safe_float(p.get("mlb_id"), None),
                          "team": team} if team == pair[1] else None,
        })

    # Attach top-5 hitters (by lineup_spot) for each team.
    for g in games:
        for key, team in (("top5_a", g["team_a"]), ("top5_b", g["team_b"])):
            tdf = hitters[hitters["team"].astype(str) == team].copy()
            if "lineup_spot" in tdf.columns:
                tdf["lineup_spot"] = pd.to_numeric(tdf["lineup_spot"], errors="coerce")
                tdf = tdf.sort_values("lineup_spot")
            g[key] = [
                {"player_name": h.get("player_name", ""),
                 "mlb_id": _safe_float(h.get("mlb_id"), None),
                 "lineup_spot": _safe_float(h.get("lineup_spot"), None)}
                for _, h in tdf.head(5).iterrows()
            ]
    return games


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_nrfi_for_today(target_date: str | None = None) -> list[dict]:
    """
    Build NRFI projections for every game in today's (or target_date's)
    projection table. Returns a list of dicts ready to be JSON-serialized
    by the API endpoint.
    """
    proj = _today_projections()
    if proj.empty:
        return []

    if target_date:
        td = pd.Timestamp(target_date)
    else:
        td = pd.Timestamp(_today_str())

    # Load history tables once
    hdf = pd.DataFrame()
    pdf = pd.DataFrame()
    if HITTER_GAMES.exists():
        hdf = pd.read_csv(HITTER_GAMES, usecols=["batter", "game_date", "PA", "H", "HR", "BB"], low_memory=False)
        hdf["game_date"] = pd.to_datetime(hdf["game_date"], errors="coerce")
        hdf["batter"] = pd.to_numeric(hdf["batter"], errors="coerce")
    if PITCHER_GAMES.exists():
        pdf = pd.read_csv(PITCHER_GAMES, usecols=["pitcher", "game_date", "BF", "K"], low_memory=False)
        pdf["game_date"] = pd.to_datetime(pdf["game_date"], errors="coerce")
        pdf["pitcher"] = pd.to_numeric(pdf["pitcher"], errors="coerce")

    out: list[dict] = []
    target_str = str(td.date())
    for game in _games_today(proj):
        pa = game.get("pitcher_a") or {}
        pb = game.get("pitcher_b") or {}

        pa_strength = _pitcher_first_inning_strength(pdf, pa.get("mlb_id"), td) if pa.get("mlb_id") is not None else {"k_rate": None, "starts": 0}
        pb_strength = _pitcher_first_inning_strength(pdf, pb.get("mlb_id"), td) if pb.get("mlb_id") is not None else {"k_rate": None, "starts": 0}
        pa_full = {**pa, **pa_strength}
        pb_full = {**pb, **pb_strength}

        def _enrich(top5):
            return [
                {**h, **_hitter_rate_summary(hdf, h.get("mlb_id"), td)}
                for h in top5
            ]
        top5_a_full = _enrich(game.get("top5_a", []))
        top5_b_full = _enrich(game.get("top5_b", []))

        # Pitcher A faces team B's lineup; pitcher B faces team A's lineup
        side_a = _side_p_zero(pa_full, top5_b_full)
        side_b = _side_p_zero(pb_full, top5_a_full)

        p_nrfi = side_a.p_zero * side_b.p_zero
        if p_nrfi >= LEAN_YES_THRESHOLD:
            lean = "YES"
        elif p_nrfi <= LEAN_NO_THRESHOLD:
            lean = "NO"
        else:
            lean = "PASS"

        out.append({
            "game_date": target_str,
            "team_a": game["team_a"],
            "team_b": game["team_b"],
            "p_nrfi": round(p_nrfi, 3),
            "lean": lean,
            "side_a": {
                "pitcher_name": side_a.pitcher_name,
                "pitcher_team": pa.get("team", game["team_a"]),
                "pitcher_k_rate": round(side_a.pitcher_k_rate, 3) if side_a.pitcher_k_rate is not None else None,
                "opposing_top5_obp": round(side_a.top5_obp, 3) if side_a.top5_obp is not None else None,
                "opposing_top5_slg": round(side_a.top5_slg, 3) if side_a.top5_slg is not None else None,
                "runs_lambda": round(side_a.runs_lambda, 3),
                "p_zero": round(side_a.p_zero, 3),
            },
            "side_b": {
                "pitcher_name": side_b.pitcher_name,
                "pitcher_team": pb.get("team", game["team_b"]),
                "pitcher_k_rate": round(side_b.pitcher_k_rate, 3) if side_b.pitcher_k_rate is not None else None,
                "opposing_top5_obp": round(side_b.top5_obp, 3) if side_b.top5_obp is not None else None,
                "opposing_top5_slg": round(side_b.top5_slg, 3) if side_b.top5_slg is not None else None,
                "runs_lambda": round(side_b.runs_lambda, 3),
                "p_zero": round(side_b.p_zero, 3),
            },
        })

    out.sort(key=lambda g: g["p_nrfi"], reverse=True)
    return out


def save_today_nrfi_log(target_date: str | None = None,
                        log_path: str = "2026_nrfi_picks.csv") -> int:
    """
    Append today's NRFI projections to a persistent pick log so the grader
    can compare them to actual first-inning outcomes later.

    Idempotent — if today's date is already in the log, those rows are
    replaced with fresh values rather than duplicated.
    """
    rows = compute_nrfi_for_today(target_date)
    if not rows:
        return 0
    new_df = pd.DataFrame(rows)
    new_df["picked_at"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    log = Path(log_path)
    if log.exists():
        try:
            existing = pd.read_csv(log, low_memory=False)
            # Drop any prior rows for the same (game_date, team_a, team_b)
            today_mask = (
                (existing["game_date"] == new_df["game_date"].iloc[0])
                & existing["team_a"].isin(new_df["team_a"])
                & existing["team_b"].isin(new_df["team_b"])
            )
            existing = existing[~today_mask].copy()
            combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(log, index=False)
    return len(new_df)


if __name__ == "__main__":
    import json
    rows = compute_nrfi_for_today()
    print(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} games processed.")
