"""
park_factors.py
===============
Multiplicative ballpark adjustments applied to projections at scoring time.

Why this exists
---------------
Two hitters with identical underlying skill will produce wildly different box
scores depending on the park: Coors Field plays as ~1.18× league-average runs;
T-Mobile and Citi Field suppress hits/HR by 5-10%. The XGBoost models train on
each player's own history (which is partly park-dependent) but inference uses
the *upcoming* park, which the model doesn't see directly. Layering park
factors on top recovers a 3-8% projection lift that the model misses.

Source data
-----------
Park factors below are 3-year averages (2022-2024) blended from FanGraphs
Guts! and Baseball Savant. They cover the four stat families our props use:

    * RUNS    — affects proj_runs, proj_runs_allowed, proj_rbi
    * HR      — affects proj_hr (home-run rate)
    * HITS    — affects proj_hits, proj_hits_allowed (BABIP-adjacent)
    * K       — affects proj_strikeouts (umpire-zone proxy by park)

A factor of 1.00 means league-neutral; 1.10 means 10% above league average for
that stat in that park. Multiplying a projection by the factor for the player's
HOME (when batting at home) or AWAY (opponent's park) destination gives the
adjusted projection.

Usage
-----
    from park_factors import apply_park_factors
    df = apply_park_factors(df, kind="hitter")     # operates on home_team / opponent
    df = apply_park_factors(df, kind="pitcher")

Both `home_team` and `opponent` columns are expected on the projections df —
the function infers which park the game is played in (hitter's `home_team` if
they're at home, else `opponent`'s park).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Static factor table — keys are MLB team abbreviations matching our scraper.
# Rows: (RUNS, HR, HITS, K) factor, 1.00 = neutral.
# ---------------------------------------------------------------------------
# Lightweight 3-year avg (2022-24) from public FanGraphs/Savant — keep in sync
# with `data/park_factors.csv` if we ever want to load dynamically.
PARK_FACTORS: dict[str, dict[str, float]] = {
    # --- Hitter-friendly extreme ---
    "COL": {"runs": 1.18, "hr": 1.12, "hits": 1.10, "k": 0.94},   # Coors
    # --- Hitter-friendly tier ---
    "CIN": {"runs": 1.07, "hr": 1.13, "hits": 1.02, "k": 0.97},   # GABP
    "BOS": {"runs": 1.06, "hr": 0.96, "hits": 1.07, "k": 0.95},   # Fenway (singles + doubles park)
    "TEX": {"runs": 1.05, "hr": 1.06, "hits": 1.02, "k": 0.97},   # Globe Life
    "PHI": {"runs": 1.04, "hr": 1.10, "hits": 1.01, "k": 0.97},   # Citizens Bank
    "AZ":  {"runs": 1.03, "hr": 1.04, "hits": 1.02, "k": 0.99},   # Chase
    "CHC": {"runs": 1.03, "hr": 1.03, "hits": 1.02, "k": 0.97},   # Wrigley (wind-dependent)
    "BAL": {"runs": 1.02, "hr": 1.06, "hits": 1.01, "k": 0.98},   # Camden (post-LF wall)
    "MIN": {"runs": 1.02, "hr": 1.02, "hits": 1.01, "k": 0.99},   # Target
    # --- Neutral cluster ---
    "ATL": {"runs": 1.01, "hr": 1.02, "hits": 1.00, "k": 0.99},
    "TOR": {"runs": 1.01, "hr": 1.04, "hits": 1.00, "k": 0.99},
    "MIL": {"runs": 1.00, "hr": 1.04, "hits": 0.99, "k": 0.99},
    "STL": {"runs": 1.00, "hr": 0.97, "hits": 1.01, "k": 0.99},
    "WSH": {"runs": 1.00, "hr": 1.01, "hits": 1.00, "k": 1.00},
    "HOU": {"runs": 0.99, "hr": 1.04, "hits": 0.99, "k": 1.00},
    "ATH": {"runs": 0.99, "hr": 0.95, "hits": 1.00, "k": 1.01},
    "OAK": {"runs": 0.99, "hr": 0.95, "hits": 1.00, "k": 1.01},   # legacy
    "SF":  {"runs": 0.99, "hr": 0.94, "hits": 1.01, "k": 0.99},   # Oracle (HR suppressor, BABIP+)
    "KC":  {"runs": 0.99, "hr": 0.94, "hits": 1.02, "k": 0.99},   # large OF
    "LAA": {"runs": 0.99, "hr": 0.99, "hits": 1.00, "k": 1.00},
    "CHW": {"runs": 0.99, "hr": 1.05, "hits": 0.98, "k": 1.01},   # Rate Field
    "CWS": {"runs": 0.99, "hr": 1.05, "hits": 0.98, "k": 1.01},   # alias
    "NYY": {"runs": 0.99, "hr": 1.10, "hits": 0.97, "k": 1.01},   # short porch HR↑, hits↓
    # --- Pitcher-friendly tier ---
    "TB":  {"runs": 0.97, "hr": 0.96, "hits": 0.99, "k": 1.02},   # Tropicana (dome, K-friendly)
    "CLE": {"runs": 0.97, "hr": 0.95, "hits": 0.99, "k": 1.01},   # Progressive
    "DET": {"runs": 0.96, "hr": 0.92, "hits": 1.00, "k": 1.01},   # Comerica (deep gaps)
    "LAD": {"runs": 0.96, "hr": 1.00, "hits": 0.97, "k": 1.02},   # Dodger (marine layer)
    "SD":  {"runs": 0.95, "hr": 0.93, "hits": 0.97, "k": 1.02},   # Petco
    "PIT": {"runs": 0.95, "hr": 0.92, "hits": 0.99, "k": 1.01},   # PNC
    "MIA": {"runs": 0.95, "hr": 0.91, "hits": 0.99, "k": 1.02},   # loanDepot
    "NYM": {"runs": 0.95, "hr": 0.93, "hits": 0.97, "k": 1.02},   # Citi
    "SEA": {"runs": 0.94, "hr": 0.92, "hits": 0.97, "k": 1.03},   # T-Mobile (most pitcher-friendly)
}

# Which projection columns get scaled by which factor
HITTER_SCALE = {
    "proj_hits":        "hits",
    "proj_hr":          "hr",
    "proj_total_bases": "hr",     # TB heavily HR-driven; HR factor approximates
    "proj_runs":        "runs",
    "proj_rbi":         "runs",
    "proj_strikeouts":  "k",
}
PITCHER_SCALE = {
    "proj_hits_allowed":  "hits",
    "proj_runs_allowed":  "runs",
    "proj_strikeouts":    "k",
    # IP and walks have weak park signal — leave alone
}

# Don't shift > this fraction in either direction. Coors is the only park that
# pushes 1.18; everything else stays in [0.91, 1.13]. Cap is a safety bound
# for the rare unknown-team case.
_MIN_FACTOR = 0.85
_MAX_FACTOR = 1.20


def get_park_factor(team: str | None, stat_family: str) -> float:
    """Return the multiplier for a stat family at a team's HOME park."""
    if not team:
        return 1.0
    block = PARK_FACTORS.get(str(team).upper())
    if not block:
        return 1.0
    f = float(block.get(stat_family, 1.0))
    return float(np.clip(f, _MIN_FACTOR, _MAX_FACTOR))


def _resolve_park_team(row: pd.Series) -> str | None:
    """
    Return the team abbreviation of the park where this game is played.
    For hitters: home_team is their team if they're at home, else opponent.
    Our projection schema stores `home_team` as the actual home team of the
    game, so just use that directly when present.
    """
    for col in ("home_team", "park_team", "venue_team"):
        if col in row and pd.notna(row.get(col)):
            return str(row[col]).upper()
    return None


def apply_park_factors(
    df: pd.DataFrame,
    kind: str,
    inplace: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Multiply each projection column by the park factor of the game's host park.
    `kind` is 'hitter' or 'pitcher' — selects which projection columns to
    scale. Rows with no resolvable park team are left unchanged.
    """
    if df is None or df.empty:
        return df
    out = df if inplace else df.copy()

    scale_map = HITTER_SCALE if kind == "hitter" else PITCHER_SCALE
    cols_present = [c for c in scale_map.keys() if c in out.columns]
    if not cols_present:
        return out

    if "home_team" not in out.columns:
        # Without a resolvable host team, we can't apply factors safely.
        if verbose:
            print(f"  park_factors: skipped {kind} — no home_team column")
        return out

    parks = out["home_team"].astype(str).str.upper()

    for col in cols_present:
        family = scale_map[col]
        factors = parks.map(lambda t: get_park_factor(t, family))
        if verbose:
            avg_factor = float(factors.mean())
            print(f"  park_factors: {kind}.{col} family={family}  avg×{avg_factor:.3f}")
        out[col] = pd.to_numeric(out[col], errors="coerce") * factors
        # Keep counting stats non-negative
        out[col] = out[col].clip(lower=0.0)

    return out


# ---------------------------------------------------------------------------
# CLI: dump the factor table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'TEAM':<5} {'RUNS':>6} {'HR':>6} {'HITS':>6} {'K':>6}")
    for team in sorted(PARK_FACTORS.keys()):
        b = PARK_FACTORS[team]
        print(f"{team:<5} {b['runs']:>6.2f} {b['hr']:>6.2f} {b['hits']:>6.2f} {b['k']:>6.2f}")
