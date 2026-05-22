"""
weather_features.py
===================
Park-level weather → multiplicative adjustments on HR / hits / runs.

Why this exists
---------------
Weather is the single largest unmodeled factor in run scoring:
  * 10°F temperature ↑ → HR rate ↑ ~3-5%
  * Wind blowing out 10mph → HR ↑ ~10-15% in HR-friendly parks
  * Wind blowing in 10mph → HR ↓ ~10-15%
  * High humidity → ball travels less (counterintuitive, but Coors humidor
    showed this: humid balls are heavier)
  * Low air density (altitude + heat) → everything carries farther

These effects compound with park factor in a way the per-pitcher / per-hitter
models can't see — the same matchup in Denver at 95°F with the wind out
plays totally differently from Denver at 55°F with the wind in.

Source
------
open-meteo.com — free, no API key required. We hit the forecast endpoint
with each park's lat/lon and the game's local start time.

Pipeline
--------
1. `fetch_today_weather(games)` — per-game weather snapshot from open-meteo
   at first-pitch local time. Returns DataFrame keyed by `game_pk`.
2. `compute_factors(weather_df)` — converts raw weather to multiplicative
   factors per stat family (HR, hits, runs) using empirical sensitivities
   from public research (FanGraphs Park Factors, Baseball Savant studies).
3. `apply_weather_factors(proj_df, factors_df)` — multiplies the relevant
   projection columns at scoring time.

All factors clip to [0.85, 1.20] so any weird API response can't blow up a
projection.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
ET = ZoneInfo("America/New_York")

# Cap how far weather can shift a projection. Even Coors at 95°F with a
# 20mph tailwind doesn't change HR rate more than ~25%; runs maybe 15%.
_MIN_FACTOR = 0.85
_MAX_FACTOR = 1.20


# ---------------------------------------------------------------------------
# Static ballpark coordinates + orientation (degrees from CF to home plate)
# Orientation lets us project wind direction onto out-to-CF, which is what
# actually matters for HR carry.
# ---------------------------------------------------------------------------
# Format: team_abbr → (lat, lon, cf_bearing_deg, is_dome)
# `cf_bearing_deg` is the compass bearing from home plate toward CF; we use
# it to decompose wind into a "blowing out" component. is_dome=True means
# we hard-set all weather factors to 1.00 since wind/temp don't matter.
PARKS: dict[str, dict] = {
    "ARI": {"lat": 33.4453, "lon": -112.0667, "cf_bearing": 23,  "dome": True},   # Chase (retractable, usually closed)
    "AZ":  {"lat": 33.4453, "lon": -112.0667, "cf_bearing": 23,  "dome": True},
    "ATL": {"lat": 33.8907, "lon": -84.4677,  "cf_bearing": 138, "dome": False},  # Truist
    "BAL": {"lat": 39.2839, "lon": -76.6217,  "cf_bearing": 36,  "dome": False},  # Camden
    "BOS": {"lat": 42.3467, "lon": -71.0972,  "cf_bearing": 47,  "dome": False},  # Fenway
    "CHC": {"lat": 41.9484, "lon": -87.6553,  "cf_bearing": 36,  "dome": False},  # Wrigley
    "CWS": {"lat": 41.8299, "lon": -87.6338,  "cf_bearing": 40,  "dome": False},  # Rate Field
    "CHW": {"lat": 41.8299, "lon": -87.6338,  "cf_bearing": 40,  "dome": False},
    "CIN": {"lat": 39.0975, "lon": -84.5066,  "cf_bearing": 24,  "dome": False},  # GABP
    "CLE": {"lat": 41.4962, "lon": -81.6852,  "cf_bearing": 14,  "dome": False},  # Progressive
    "COL": {"lat": 39.7561, "lon": -104.9942, "cf_bearing": 0,   "dome": False},  # Coors
    "DET": {"lat": 42.3390, "lon": -83.0485,  "cf_bearing": 142, "dome": False},  # Comerica
    "HOU": {"lat": 29.7570, "lon": -95.3554,  "cf_bearing": 18,  "dome": True},   # Minute Maid (retractable)
    "KC":  {"lat": 39.0517, "lon": -94.4803,  "cf_bearing": 49,  "dome": False},  # Kauffman
    "LAA": {"lat": 33.8003, "lon": -117.8827, "cf_bearing": 53,  "dome": False},  # Angel Stadium
    "LAD": {"lat": 34.0739, "lon": -118.2400, "cf_bearing": 0,   "dome": False},  # Dodger
    "MIA": {"lat": 25.7781, "lon": -80.2197,  "cf_bearing": 60,  "dome": True},   # loanDepot (retractable)
    "MIL": {"lat": 43.0280, "lon": -87.9712,  "cf_bearing": 19,  "dome": True},   # American Family (retractable)
    "MIN": {"lat": 44.9817, "lon": -93.2776,  "cf_bearing": 95,  "dome": False},  # Target
    "NYM": {"lat": 40.7571, "lon": -73.8458,  "cf_bearing": 23,  "dome": False},  # Citi
    "NYY": {"lat": 40.8296, "lon": -73.9262,  "cf_bearing": 67,  "dome": False},  # Yankee Stadium
    "ATH": {"lat": 37.7516, "lon": -122.2005, "cf_bearing": 60,  "dome": False},  # Coliseum (placeholder)
    "OAK": {"lat": 37.7516, "lon": -122.2005, "cf_bearing": 60,  "dome": False},
    "PHI": {"lat": 39.9061, "lon": -75.1665,  "cf_bearing": 36,  "dome": False},  # Citizens Bank
    "PIT": {"lat": 40.4469, "lon": -80.0058,  "cf_bearing": 116, "dome": False},  # PNC
    "SD":  {"lat": 32.7073, "lon": -117.1567, "cf_bearing": 0,   "dome": False},  # Petco
    "SF":  {"lat": 37.7786, "lon": -122.3893, "cf_bearing": 92,  "dome": False},  # Oracle
    "SEA": {"lat": 47.5914, "lon": -122.3325, "cf_bearing": 28,  "dome": True},   # T-Mobile (retractable)
    "STL": {"lat": 38.6226, "lon": -90.1928,  "cf_bearing": 73,  "dome": False},  # Busch
    "TB":  {"lat": 27.7682, "lon": -82.6534,  "cf_bearing": 50,  "dome": True},   # Tropicana
    "TEX": {"lat": 32.7473, "lon": -97.0843,  "cf_bearing": 31,  "dome": True},   # Globe Life (retractable)
    "TOR": {"lat": 43.6414, "lon": -79.3894,  "cf_bearing": 0,   "dome": True},   # Rogers Centre (retractable)
    "WSH": {"lat": 38.8730, "lon": -77.0074,  "cf_bearing": 31,  "dome": False},  # Nationals Park
}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _fetch_park_weather(lat: float, lon: float, when: datetime,
                        timeout: int = 8) -> Optional[dict]:
    """Return the hourly weather snapshot closest to `when` for one park."""
    try:
        r = requests.get(OPEN_METEO_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
            "forecast_days": 2,
        }, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [weather] fetch failed for ({lat:.2f},{lon:.2f}): {e}")
        return None

    hourly = (data or {}).get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None

    target_iso = when.strftime("%Y-%m-%dT%H:00")
    # Find the closest hour string in the response
    idx = 0
    for i, t in enumerate(times):
        if t >= target_iso:
            idx = i
            break
    return {
        "temp_f":    (hourly.get("temperature_2m") or [None])[idx],
        "humidity":  (hourly.get("relative_humidity_2m") or [None])[idx],
        "wind_mph":  (hourly.get("wind_speed_10m") or [None])[idx],
        "wind_dir":  (hourly.get("wind_direction_10m") or [None])[idx],
        "pressure":  (hourly.get("surface_pressure") or [None])[idx],
        "valid_at":  times[idx] if idx < len(times) else None,
    }


def fetch_today_weather(games: pd.DataFrame, default_local_hour: int = 19) -> pd.DataFrame:
    """
    Return {game_pk, home_team, weather…} for each game. Closed-roof parks
    get a sentinel "dome=True" row that downstream factor math zeroes out.

    `games` must have columns: game_pk, home_team, commence_time (ISO) or
    a local game-start hint. If commence_time is missing we use 7pm local.
    """
    rows: list[dict] = []
    if games is None or games.empty:
        return pd.DataFrame()

    for _, g in games.iterrows():
        home = str(g.get("home_team", "")).upper()
        park = PARKS.get(home)
        if park is None:
            rows.append({"game_pk": g.get("game_pk"), "home_team": home,
                         "dome": True, "skipped": True})
            continue
        if park.get("dome"):
            rows.append({"game_pk": g.get("game_pk"), "home_team": home,
                         "dome": True, "skipped": False})
            continue

        ct = g.get("commence_time")
        if pd.notna(ct):
            try:
                when = pd.to_datetime(ct).to_pydatetime()
            except Exception:
                when = datetime.now().replace(hour=default_local_hour, minute=0)
        else:
            when = datetime.now().replace(hour=default_local_hour, minute=0)

        w = _fetch_park_weather(park["lat"], park["lon"], when)
        if w is None:
            rows.append({"game_pk": g.get("game_pk"), "home_team": home,
                         "dome": False, "skipped": True})
            continue

        rows.append({
            "game_pk": g.get("game_pk"), "home_team": home,
            "dome": False, "skipped": False,
            "cf_bearing": park["cf_bearing"],
            **w,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convert weather → multiplicative factors
# ---------------------------------------------------------------------------
def _wind_out_component(wind_speed_mph: float, wind_from_deg: float,
                        cf_bearing_deg: float) -> float:
    """
    Return the signed wind component along the home-plate→CF axis.
    Positive = blowing OUT toward CF (helps HR/hits), negative = blowing IN.

    `wind_from_deg` is meteorological "direction wind is coming FROM" (0=N,
    90=E, etc.). The wind is blowing TOWARD `(wind_from + 180) % 360`.
    """
    blowing_to = (wind_from_deg + 180.0) % 360.0
    diff = (blowing_to - cf_bearing_deg + 540.0) % 360.0 - 180.0   # [-180, 180]
    return float(wind_speed_mph) * math.cos(math.radians(diff))


def compute_factors(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per game with `hr_factor`, `hits_factor`, `runs_factor`.
    Domes / missing-data rows get 1.0 for every factor.
    """
    if weather is None or weather.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, w in weather.iterrows():
        gpk = w.get("game_pk")
        home = w.get("home_team")
        if w.get("dome") or w.get("skipped"):
            rows.append({"game_pk": gpk, "home_team": home,
                         "hr_factor": 1.0, "hits_factor": 1.0, "runs_factor": 1.0})
            continue

        try:
            temp = float(w["temp_f"])
            wind = float(w["wind_mph"])
            wdir = float(w["wind_dir"])
            cf   = float(w["cf_bearing"])
        except (TypeError, ValueError, KeyError):
            rows.append({"game_pk": gpk, "home_team": home,
                         "hr_factor": 1.0, "hits_factor": 1.0, "runs_factor": 1.0})
            continue

        # Temperature effect (centered at 72°F): +1% HR per 4°F above, capped
        temp_delta = (temp - 72.0) / 4.0
        hr_temp = 1.0 + 0.01 * temp_delta

        # Wind effect — only the out-to-CF component matters for HR carry
        wind_out = _wind_out_component(wind, wdir, cf)
        hr_wind = 1.0 + 0.012 * wind_out   # ~12% per 10 mph out-to-CF wind

        # Hits get a smaller effect (mostly via fly-ball BABIP)
        hits_wind = 1.0 + 0.004 * wind_out
        hits_temp = 1.0 + 0.003 * temp_delta

        # Runs combine the two roughly geometrically
        hr_factor   = float(np.clip(hr_temp * hr_wind,     _MIN_FACTOR, _MAX_FACTOR))
        hits_factor = float(np.clip(hits_temp * hits_wind, _MIN_FACTOR, _MAX_FACTOR))
        runs_factor = float(np.clip(0.6 * hr_factor + 0.4 * hits_factor,
                                    _MIN_FACTOR, _MAX_FACTOR))

        rows.append({
            "game_pk": gpk, "home_team": home,
            "hr_factor": hr_factor,
            "hits_factor": hits_factor,
            "runs_factor": runs_factor,
            "temp_f": temp, "wind_mph": wind, "wind_out_mph": wind_out,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Apply at scoring time
# ---------------------------------------------------------------------------
def apply_weather_factors(
    proj_df: pd.DataFrame,
    factors: pd.DataFrame,
    kind: str,            # "pitcher" or "hitter"
    *,
    home_team_col: str = "home_team",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Multiply HR / hits / runs projection columns by the weather factor for
    the game's host park. Joins on `home_team_col`; rows with no factor row
    are left unchanged.
    """
    if proj_df is None or proj_df.empty or factors is None or factors.empty:
        return proj_df
    if home_team_col not in proj_df.columns:
        return proj_df

    out = proj_df.copy()

    # Join factors by home_team
    fact_min = factors[["home_team", "hr_factor", "hits_factor", "runs_factor"]].copy()
    out = out.merge(fact_min, on=home_team_col, how="left", suffixes=("", "_wx"))
    for c in ("hr_factor", "hits_factor", "runs_factor"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(1.0)

    scale_map = {
        "hitter": {
            "proj_hr":   "hr_factor",
            "proj_hits": "hits_factor",
            "proj_runs": "runs_factor",
            "proj_rbi":  "runs_factor",
        },
        "pitcher": {
            "proj_hr_allowed":     "hr_factor",   # may not exist
            "proj_hits_allowed":   "hits_factor",
            "proj_runs_allowed":   "runs_factor",
        },
    }.get(kind, {})

    for col, fac_col in scale_map.items():
        if col not in out.columns:
            continue
        before_mean = pd.to_numeric(out[col], errors="coerce").mean()
        out[col] = pd.to_numeric(out[col], errors="coerce") * out[fac_col]
        out[col] = out[col].clip(lower=0)
        after_mean = pd.to_numeric(out[col], errors="coerce").mean()
        if verbose:
            print(f"  weather: {kind}.{col}: avg {before_mean:.2f} → {after_mean:.2f}")

    out = out.drop(columns=["hr_factor", "hits_factor", "runs_factor"], errors="ignore")
    return out


# ---------------------------------------------------------------------------
# CLI: fetch + cache today's factors
# ---------------------------------------------------------------------------
def write_today_factors(games: pd.DataFrame,
                         out_path: Path | None = None) -> Path:
    out_path = out_path or DATA_DIR / "weather_today.csv"
    w = fetch_today_weather(games)
    f = compute_factors(w)
    f.to_csv(out_path, index=False)
    print(f"  Wrote {len(f)} weather-factor rows → {out_path}")
    return out_path


if __name__ == "__main__":
    print("This module is a library. Import `fetch_today_weather` + "
          "`compute_factors` + `apply_weather_factors` from the daily pipeline.")
