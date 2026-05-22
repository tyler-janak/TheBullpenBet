"""
fanduel_props.py
================
Scrape FanDuel player prop lines via The Odds API.

Fetches over/under lines for:
  Hitters : hits, home_runs, rbis, runs_scored, stolen_bases
  Pitchers: strikeouts, hits_allowed, walks, outs

Outputs
-------
outputs/fanduel_props_today.csv
    One row per player-prop with columns:
        player_name, norm_name, market, line,
        over_odds, under_odds, player_type, game_date

Usage
-----
    python fanduel_props.py --api-key YOUR_KEY
    python fanduel_props.py --api-key YOUR_KEY --date 2026-05-03
"""

import argparse
import re
import time
import unicodedata
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

OUT_DIR = Path("outputs")

# ── Odds-API market keys ─────────────────────────────────────────────────────
HITTER_MARKETS = [
    "batter_hits",
    "batter_home_runs",
    "batter_rbis",
    "batter_runs_scored",
    "batter_stolen_bases",
    "batter_strikeouts",
]

PITCHER_MARKETS = [
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_outs",
    "pitcher_earned_runs",
]

# Map Odds API market key → friendly stat name used in projection columns
MARKET_STAT_MAP = {
    "batter_hits":         "hits",
    "batter_home_runs":    "hr",
    "batter_rbis":         "rbi",
    "batter_runs_scored":  "runs",
    "batter_stolen_bases": "sb",
    "batter_strikeouts":   "strikeouts",
    "pitcher_strikeouts":  "strikeouts",
    "pitcher_hits_allowed":"hits_allowed",
    "pitcher_walks":       "walks",
    "pitcher_outs":        "outs",
    "pitcher_earned_runs": "er",
}


# ── Name normalisation ────────────────────────────────────────────────────────
def normalize_name(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Odds API calls ────────────────────────────────────────────────────────────
def get_event_ids(api_key: str, target_date: str) -> list[dict]:
    """Return list of {eventId, home_team, away_team, commence_time} for today."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
        "daysFrom": 2,          # include today + tomorrow to catch late-night UTC cutoffs
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    events = r.json()
    target_dt = pd.to_datetime(target_date).date()

    out = []
    for ev in events:
        ct = pd.to_datetime(ev.get("commence_time"))
        if ct.date() == target_dt:
            out.append({
                "eventId": ev["id"],
                "home_team": ev.get("home_team", ""),
                "away_team": ev.get("away_team", ""),
                "commence_time": ev.get("commence_time"),
            })

    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  Events found for {target_date}: {len(out)}  |  API credits remaining: {remaining}")
    return out


def get_event_props(api_key: str, event_id: str, markets: list[str]) -> list[dict]:
    """Fetch player props for one event from FanDuel."""
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    params = {
        "apiKey":       api_key,
        "regions":      "us",
        "markets":      ",".join(markets),
        "bookmakers":   "fanduel",
        "oddsFormat":   "american",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 422:
        # event may not have props yet
        return []
    r.raise_for_status()
    return r.json().get("bookmakers", [])


# ── Parse bookmaker response ──────────────────────────────────────────────────
def parse_bookmaker_props(bookmakers: list[dict], game_date: str) -> pd.DataFrame:
    rows = []
    for book in bookmakers:
        if book.get("key") != "fanduel":
            continue
        for market in book.get("markets", []):
            mkt_key = market.get("key", "")
            stat = MARKET_STAT_MAP.get(mkt_key)
            if stat is None:
                continue

            # Determine player type from market key prefix
            player_type = "pitcher" if mkt_key.startswith("pitcher") else "hitter"

            # Outcomes come as pairs: Over X @ odds, Under X @ odds
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}

            over = outcomes.get("Over")
            under = outcomes.get("Under")

            if over is None and under is None:
                continue

            line = float(over.get("point", under.get("point", np.nan))) if (over or under) else np.nan
            over_odds  = int(over["price"])  if over  else np.nan
            under_odds = int(under["price"]) if under else np.nan

            player_name = (over or under).get("description", "")

            rows.append({
                "player_name":  player_name,
                "norm_name":    normalize_name(player_name),
                "market":       stat,
                "line":         line,
                "over_odds":    over_odds,
                "under_odds":   under_odds,
                "player_type":  player_type,
                "game_date":    game_date,
            })

    return pd.DataFrame(rows)


# ── Main fetch function ───────────────────────────────────────────────────────
def fetch_fanduel_props(api_key: str, target_date: str | None = None) -> pd.DataFrame:
    if target_date is None:
        target_date = str(date.today())

    all_markets = HITTER_MARKETS + PITCHER_MARKETS

    print(f"\nFetching FanDuel props for {target_date}...")
    events = get_event_ids(api_key, target_date)

    if not events:
        print("  No events found.")
        return pd.DataFrame()

    frames = []
    for i, ev in enumerate(events):
        event_id = ev["eventId"]
        print(f"  [{i+1}/{len(events)}] {ev['away_team']} @ {ev['home_team']}", end=" ... ")
        try:
            bookmakers = get_event_props(api_key, event_id, all_markets)
            df = parse_bookmaker_props(bookmakers, target_date)
            if not df.empty:
                df["home_team_api"] = ev["home_team"]
                df["away_team_api"] = ev["away_team"]
                frames.append(df)
                print(f"{len(df)} props")
            else:
                print("0 props")
        except Exception as exc:
            print(f"ERROR: {exc}")
        # Be nice to the API — 0.25 s between calls
        time.sleep(0.25)

    if not frames:
        print("  No props returned from FanDuel.")
        return pd.DataFrame()

    props = pd.concat(frames, ignore_index=True)
    props = props.drop_duplicates(subset=["player_name", "market"])
    print(f"\n  Total prop lines: {len(props)}")
    return props


# ── Pivot to wide format for easy merging ─────────────────────────────────────
def pivot_props_wide(props: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per player, columns:
        player_name, norm_name, player_type,
        fd_hits_line, fd_hits_over_odds, fd_hits_under_odds,
        fd_hr_line, ...  etc.
    """
    if props.empty:
        return pd.DataFrame()

    frames = []
    for stat, grp in props.groupby("market"):
        sub = grp[["player_name", "norm_name", "player_type",
                    "line", "over_odds", "under_odds"]].copy()
        sub = sub.rename(columns={
            "line":       f"fd_{stat}_line",
            "over_odds":  f"fd_{stat}_over_odds",
            "under_odds": f"fd_{stat}_under_odds",
        })
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    wide = frames[0]
    for f in frames[1:]:
        wide = wide.merge(
            f[["norm_name"] + [c for c in f.columns if c.startswith("fd_")]],
            on="norm_name",
            how="outer",
        )

    return wide


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch FanDuel player props via The Odds API")
    parser.add_argument("--api-key", required=True, help="The Odds API key")
    parser.add_argument("--date",    default=None,   help="Target date YYYY-MM-DD (default=today)")
    args = parser.parse_args()

    target_date = args.date or str(date.today())

    props_long = fetch_fanduel_props(api_key=args.api_key, target_date=target_date)
    props_wide = pivot_props_wide(props_long)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_path = OUT_DIR / "fanduel_props_today_long.csv"
    wide_path = OUT_DIR / "fanduel_props_today.csv"

    props_long.to_csv(long_path, index=False)
    props_wide.to_csv(wide_path, index=False)

    print(f"\nSaved:")
    print(f"  {long_path}  ({len(props_long)} rows)")
    print(f"  {wide_path}  ({len(props_wide)} rows, wide format)")


if __name__ == "__main__":
    main()
