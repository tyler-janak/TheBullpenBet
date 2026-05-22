"""
props_fetch.py
==============
Pulls today's player-prop lines from The Odds API, joins them to the local
projections (outputs/hitterspitchers_today.csv), and writes:

    outputs/today_props_raw.csv         — raw fetched props (gitignored)
    outputs/today_props_with_ev.csv     — props + edge / EV / score (committed)
    2026_props_log.csv                  — append-only history of every prop seen
                                          (used for CLV tracking and grading)

The Odds API charges per request. To stay within the free tier (~500/month),
we fetch player props once per cron call only at the configured "primary"
ticks (defaults to the 5pm ET cron when lineups are mostly confirmed). The
3 AM and 11 AM cron ticks skip the props pull.

Public functions
----------------
    fetch_today_props(api_key, sportsbooks=("draftkings","fanduel"))
    compute_edge_today()
    log_clv_open(props_df)   — appended to 2026_props_log.csv with stage="open"
    log_clv_close(props_df)  — appended with stage="close"
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent

OUT_DIR = HERE / "outputs"
RAW_PATH        = OUT_DIR / "today_props_raw.csv"
EDGE_PATH       = OUT_DIR / "today_props_with_ev.csv"
PROPS_LOG_PATH  = HERE / "2026_props_log.csv"
# Prefer the raw (uncalibrated) projections for prop math — calibration is a
# display-time bias correction and shouldn't cascade into the edge engine,
# or every Over above a sportsbook line gets mechanically flagged as VALUE.
# Falls back to the regular (possibly calibrated) CSV if the raw copy isn't
# present (e.g. before the first calibration run of the season).
TODAY_PROJ_PATH_RAW = OUT_DIR / "hitterspitchers_today_raw.csv"
TODAY_PROJ_PATH_CAL = OUT_DIR / "hitterspitchers_today.csv"
TODAY_PROJ_PATH     = TODAY_PROJ_PATH_RAW   # default for compute_edge_today below

ODDS_API_KEY_DEFAULT = os.environ.get(
    "ODDS_API_KEY",
    "afa28350c34fba9f318ecd7ae4e21b63",   # same key the game-pick pipeline uses
)
SPORT = "baseball_mlb"
DEFAULT_BOOKS = ("draftkings", "fanduel")

# Markets our engine supports (must match keys in props_engine.MARKET_CONFIG).
MARKETS = [
    "pitcher_strikeouts",
    "pitcher_walks",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
    "pitcher_outs",
    "batter_hits",
    "batter_home_runs",
    "batter_total_bases",
    "batter_strikeouts",
    "batter_walks",
    "batter_rbis",
    "batter_runs_scored",
]

EVENTS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
EVENT_ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{{event_id}}/odds"


# ---------------------------------------------------------------------------
# The Odds API helpers
# ---------------------------------------------------------------------------
def _today_str() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _fetch_today_event_ids(api_key: str, timeout: int = 30) -> list[dict]:
    """Return [{id, home_team, away_team, commence_time}] for today's slate."""
    try:
        r = requests.get(EVENTS_URL, params={
            "apiKey": api_key,
            "dateFormat": "iso",
        }, timeout=timeout)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"⚠️  Odds API events fetch failed: {e}")
        return []

    today = _today_str()
    out = []
    for e in events:
        ct = (e.get("commence_time") or "")[:10]
        if ct == today:
            out.append({
                "id": e.get("id"),
                "home_team": e.get("home_team"),
                "away_team": e.get("away_team"),
                "commence_time": e.get("commence_time"),
            })
    return out


def _fetch_event_props(api_key: str, event_id: str,
                       markets: Iterable[str] = MARKETS,
                       bookmakers: Iterable[str] = DEFAULT_BOOKS,
                       timeout: int = 30) -> dict:
    """Pull props for one event. Returns the raw JSON or {}."""
    try:
        r = requests.get(EVENT_ODDS_URL.format(event_id=event_id), params={
            "apiKey": api_key,
            "markets": ",".join(markets),
            "bookmakers": ",".join(bookmakers),
            "oddsFormat": "american",
        }, timeout=timeout)
        if r.status_code == 422:
            # Bookmakers may not offer all markets — this isn't fatal
            return {}
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [props] {event_id}: {e}")
        return {}


def _normalize_event_props(raw: dict) -> list[dict]:
    """
    Flatten the API response into a row-per-(player, market, line, sportsbook).
    Each market's outcomes pair as (Over, Under) per player_name × line.
    """
    rows: list[dict] = []
    home = raw.get("home_team")
    away = raw.get("away_team")
    commence = raw.get("commence_time")
    bookmakers = raw.get("bookmakers") or []
    for bk in bookmakers:
        sportsbook = bk.get("key")
        for market in (bk.get("markets") or []):
            mkt = market.get("key")
            outcomes = market.get("outcomes") or []
            # Group by (player_name, line) so we can pair Over/Under
            grouped: dict[tuple, dict] = {}
            for o in outcomes:
                player = o.get("description") or o.get("name")  # API uses "description" for player on prop markets
                point = o.get("point")
                side = (o.get("name") or "").strip().lower()
                price = o.get("price")
                if player is None or point is None:
                    continue
                key = (str(player), float(point))
                grouped.setdefault(key, {"player_name": str(player), "line": float(point)})
                if side == "over":
                    grouped[key]["over_odds"] = int(price) if price is not None else None
                elif side == "under":
                    grouped[key]["under_odds"] = int(price) if price is not None else None
            for key, vals in grouped.items():
                if "over_odds" not in vals or "under_odds" not in vals:
                    continue
                rows.append({
                    "fetched_at": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"),
                    "game_date": (commence or "")[:10],
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence,
                    "sportsbook": sportsbook,
                    "market": mkt,
                    "player_name": vals["player_name"],
                    "line": vals["line"],
                    "over_odds": vals.get("over_odds"),
                    "under_odds": vals.get("under_odds"),
                })
    return rows


# ---------------------------------------------------------------------------
# Public: fetch today's props + persist
# ---------------------------------------------------------------------------
def fetch_today_props(api_key: str | None = None,
                      bookmakers: Iterable[str] = DEFAULT_BOOKS,
                      sleep_seconds: float = 0.4) -> pd.DataFrame:
    """
    Pull every supported player-prop market for every game on today's slate
    from the configured sportsbooks. Returns a DataFrame and writes
    outputs/today_props_raw.csv.
    """
    key = api_key or ODDS_API_KEY_DEFAULT
    print("\n── Fetching today's player props ────────────────────────────")
    events = _fetch_today_event_ids(key)
    print(f"  {len(events)} event(s) on today's slate")
    if not events:
        return pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for i, ev in enumerate(events, start=1):
        raw = _fetch_event_props(key, ev["id"], MARKETS, bookmakers)
        if raw:
            rows = _normalize_event_props(raw)
            all_rows.extend(rows)
            print(f"  [{i:>2}/{len(events)}] {ev['away_team']} @ {ev['home_team']}  → {len(rows)} prop rows")
        else:
            print(f"  [{i:>2}/{len(events)}] {ev['away_team']} @ {ev['home_team']}  → no props")
        time.sleep(sleep_seconds)

    if not all_rows:
        print("  No props fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_PATH, index=False)
    print(f"  Wrote {len(df):,} prop rows → {RAW_PATH}")
    return df


# ---------------------------------------------------------------------------
# Edge computation entry point — joins props to projections
# ---------------------------------------------------------------------------
def _name_to_mlb_id_map(proj_df: pd.DataFrame) -> dict[str, int]:
    """Build a normalized-name → mlb_id index from the projection table."""
    out: dict[str, int] = {}
    if proj_df.empty or "player_name" not in proj_df.columns:
        return out
    for _, r in proj_df.iterrows():
        try:
            mid = int(float(r["mlb_id"])) if pd.notna(r.get("mlb_id")) else None
        except (TypeError, ValueError):
            mid = None
        if mid is None:
            continue
        nm = str(r.get("player_name", "")).strip().lower()
        if nm:
            out[nm] = mid
    return out


def _attach_mlb_ids(props_df: pd.DataFrame, proj_df: pd.DataFrame) -> pd.DataFrame:
    """Add mlb_id column to props_df by name lookup against projections."""
    name_map = _name_to_mlb_id_map(proj_df)
    if not name_map:
        props_df["mlb_id"] = None
        return props_df
    props_df = props_df.copy()
    props_df["mlb_id"] = props_df["player_name"].astype(str).str.strip().str.lower().map(name_map)
    return props_df


def compute_edge_today() -> pd.DataFrame:
    """
    Read raw props + today's projections, run props_engine.add_edge_columns,
    and write outputs/today_props_with_ev.csv. Also appends every prop seen
    to 2026_props_log.csv (stage='open' since this is the first pull of the day).
    """
    if not RAW_PATH.exists():
        print(f"⚠️  {RAW_PATH} not found — run fetch_today_props() first")
        return pd.DataFrame()

    # Prefer raw projections (uncalibrated) for prop math; fall back to the
    # calibrated/display CSV only if the raw copy hasn't been written yet.
    if TODAY_PROJ_PATH_RAW.exists():
        proj_path = TODAY_PROJ_PATH_RAW
        print(f"  Using RAW projections for prop math: {proj_path.name}")
    elif TODAY_PROJ_PATH_CAL.exists():
        proj_path = TODAY_PROJ_PATH_CAL
        print(f"  ⚠️  Raw projections not found — falling back to calibrated CSV {proj_path.name}.")
        print(f"  ⚠️  Edge math will be biased toward Overs until calibration writes a raw copy.")
    else:
        print(f"⚠️  No projections file found at {TODAY_PROJ_PATH_RAW} or {TODAY_PROJ_PATH_CAL}")
        return pd.DataFrame()

    props = pd.read_csv(RAW_PATH, low_memory=False)
    proj = pd.read_csv(proj_path, low_memory=False)
    if props.empty or proj.empty:
        return pd.DataFrame()

    # Synthesize batter total_bases from PA / hits / HR if needed.
    # Approximation: TB = (hits − HR) × 1 + HR × 4 (treating non-HR hits as singles).
    if "proj_total_bases" not in proj.columns:
        if "proj_hits" in proj.columns and "proj_hr" in proj.columns:
            h  = pd.to_numeric(proj["proj_hits"], errors="coerce").fillna(0)
            hr = pd.to_numeric(proj["proj_hr"],   errors="coerce").fillna(0)
            proj["proj_total_bases"] = (h - hr).clip(lower=0) + hr * 4

    # Synthesize pitcher proj_outs from proj_ip if absent (1 IP = 3 outs).
    if "proj_outs" not in proj.columns and "proj_ip" in proj.columns:
        proj["proj_outs"] = pd.to_numeric(proj["proj_ip"], errors="coerce") * 3.0

    props = _attach_mlb_ids(props, proj)

    from props_engine import add_edge_columns
    edged = add_edge_columns(props, proj)
    if edged.empty:
        print("⚠️  No props matched a projection — nothing to write.")
        return edged

    edged.to_csv(EDGE_PATH, index=False)
    print(f"  Wrote {len(edged):,} edged rows → {EDGE_PATH}")
    print(f"  Top 5 by score:")
    cols = [c for c in ["player_name","market","line","sportsbook","side","ev","score","flag"] if c in edged.columns]
    print(edged[cols].head(5).to_string(index=False))

    # Append to season log (CLV stage = open)
    log_clv_open(edged)
    return edged


# ---------------------------------------------------------------------------
# CLV tracking
# ---------------------------------------------------------------------------
def _append_to_props_log(df: pd.DataFrame, stage: str) -> None:
    if df is None or df.empty:
        return
    snap = df.copy()
    snap["clv_stage"] = stage   # "open" or "close"
    snap["logged_at"] = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    if PROPS_LOG_PATH.exists():
        try:
            existing = pd.read_csv(PROPS_LOG_PATH, low_memory=False)
            combined = pd.concat([existing, snap], ignore_index=True, sort=False)
        except Exception as e:
            print(f"⚠️  could not read {PROPS_LOG_PATH} ({e}) — overwriting.")
            combined = snap
    else:
        combined = snap
    combined.to_csv(PROPS_LOG_PATH, index=False)


def log_clv_open(props_df: pd.DataFrame) -> None:
    """Snapshot of opening lines (first time we saw the prop today)."""
    _append_to_props_log(props_df, stage="open")


def log_clv_close(props_df: pd.DataFrame) -> None:
    """Snapshot of closing lines (final pull just before first pitch)."""
    _append_to_props_log(props_df, stage="close")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fetch-only", action="store_true", help="Only pull from Odds API; don't compute edge.")
    p.add_argument("--edge-only",  action="store_true", help="Only compute edge using existing today_props_raw.csv.")
    p.add_argument("--api-key", default=ODDS_API_KEY_DEFAULT)
    p.add_argument("--books", default=",".join(DEFAULT_BOOKS),
                   help="Comma-separated sportsbook keys (e.g. draftkings,fanduel)")
    args = p.parse_args()

    if not args.edge_only:
        fetch_today_props(api_key=args.api_key, bookmakers=args.books.split(","))
    if not args.fetch_only:
        compute_edge_today()


if __name__ == "__main__":
    main()
