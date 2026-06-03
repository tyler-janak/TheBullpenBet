"""
oddsapi_fetch.py
=================
Pull moneyline and player-prop odds from The Odds API (the one you already
use), shape them into market dicts that market_model.score_slate() consumes,
and write the recommended picks + a CLV-ready log.

What this gives you
-------------------
- Mainline MLB moneylines (h2h) across regions us, us2, eu, uk so the
  consensus has Pinnacle / Circa / BetOnline alongside FanDuel / DraftKings.
- Player props (pitcher K/BB/H/outs, batter H/HR/TB/K/BB/R/RBI by default)
  scored the same way.
- Output:
    outputs/market_consensus_picks_today.csv  - this run's +EV bets
    2026_market_consensus_log.csv             - every pick ever surfaced
                                                (the snapshot needed for CLV)

No new credentials — uses the ODDS_API_KEY env var if set, otherwise falls
back to the same key already hard-coded in daily_mlb_model_runner.run().

Usage
-----
    python oddsapi_fetch.py
    python oddsapi_fetch.py --my-book DraftKings --bankroll 5000 --min-edge 0.015
    python oddsapi_fetch.py --skip-props
    python oddsapi_fetch.py --prop-markets pitcher_strikeouts,batter_total_bases

Note on API credits
-------------------
Each per-event prop call costs one API credit per event x N regions. To save
quota, trim --prop-markets or run --skip-props on most ticks and only pull
full props once a day.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

import market_model as mm

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)
PICKS_CSV = OUT_DIR / "market_consensus_picks_today.csv"
LOG_CSV   = HERE / "2026_market_consensus_log.csv"

# Same key already used in daily_mlb_model_runner.run(). ODDS_API_KEY env var
# overrides it (matches how props_fetch.py reads the env var).
DEFAULT_KEY = "afa28350c34fba9f318ecd7ae4e21b63"

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

# The Odds API bookmaker key -> canonical name in market_model's weight tables.
# Anything unmapped is silently dropped (so a new book joining the API can't
# accidentally enter the consensus with the default 0.02 weight).
BOOK_KEY_MAP = {
    "pinnacle":       "Pinnacle",
    "circasports":    "Circa",
    "circa":          "Circa",
    "betonlineag":    "BetOnline",
    "betonline":      "BetOnline",
    "betcris":        "BetCRIS",
    "fanduel":        "FanDuel",
    "draftkings":     "DraftKings",
    "betmgm":         "BetMGM",
    "williamhill_us": "Caesars",
    "caesars":        "Caesars",
    "betrivers":      "BetRivers",
    "bet365":         "Bet365",
    "pointsbetus":    "PointsBet",
    "novig":          "Novig",
    "prophetx":       "ProphetX",
}

# Default MLB prop markets to pull. The Odds API requires these to be passed
# as a comma list to the per-event odds endpoint.
DEFAULT_PROP_MARKETS_MLB = [
    "pitcher_strikeouts",
    "pitcher_walks",
    "pitcher_hits_allowed",
    "pitcher_outs",
    "batter_hits",
    "batter_home_runs",
    "batter_total_bases",
    "batter_strikeouts",
    "batter_walks",
    "batter_runs_scored",
    "batter_rbis",
]

MAINLINE_TAG = "mlb_ml"
PROPS_TAG    = "player_prop_mlb"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_key(cli_key: str | None) -> str:
    return cli_key or os.environ.get("ODDS_API_KEY") or DEFAULT_KEY


def _http_get(url: str, params: dict, retries: int = 2, sleep: float = 1.0):
    """GET with light retries on 429/5xx. Raises on persistent failure."""
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=25)
        except requests.RequestException as e:
            last_err = e
            time.sleep(sleep * (i + 1))
            continue
        if r.status_code == 200:
            return r.json(), r.headers
        if r.status_code in (429, 500, 502, 503, 504) and i < retries:
            time.sleep(sleep * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"odds api request failed: {url} ({last_err})")


def _map_book(odds_api_key: str) -> str | None:
    return BOOK_KEY_MAP.get((odds_api_key or "").lower())


# ---------------------------------------------------------------------------
# Mainline (h2h moneyline)
# ---------------------------------------------------------------------------

def fetch_mainline_markets(api_key: str,
                           sport_key: str = "baseball_mlb",
                           regions: str = "us,us2,eu,uk",
                           pregame_only: bool = True,
                           min_minutes_to_start: int = 5) -> list[dict]:
    """Pull h2h odds for every upcoming event and shape into market dicts.

    `pregame_only=True` (default) filters out games that have already started —
    in-progress games return LIVE odds, and live prices on a fast book mixed
    with stale prices on a slow book produces nonsense consensus (e.g. one
    book still posting a 2-run dog as +630 while sharps have re-priced the
    game as 50/50 because the dog is winning 5-0)."""
    import datetime as _dt
    url = f"{ODDS_API_BASE}/{sport_key}/odds"
    params = {
        "apiKey":      api_key,
        "regions":     regions,
        "markets":     "h2h",
        "oddsFormat":  "american",
        "dateFormat":  "iso",
    }
    events, _ = _http_get(url, params)

    now_utc = _dt.datetime.now(_dt.timezone.utc)
    cutoff  = now_utc + _dt.timedelta(minutes=min_minutes_to_start)

    out: list[dict] = []
    skipped_live = 0
    for ev in events:
        away = ev.get("away_team")
        home = ev.get("home_team")
        if not (away and home):
            continue
        if pregame_only:
            ct = ev.get("commence_time")
            try:
                start = _dt.datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            except Exception:
                start = None
            if start is not None and start < cutoff:
                skipped_live += 1
                continue
        books: dict[str, dict] = {}
        for bm in ev.get("bookmakers", []):
            cname = _map_book(bm.get("key"))
            if not cname:
                continue
            for mk in bm.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                prices: dict[str, float] = {}
                for o in mk.get("outcomes", []):
                    name = o.get("name")
                    if name == away:
                        prices["away"] = o.get("price")
                    elif name == home:
                        prices["home"] = o.get("price")
                if "away" in prices and "home" in prices:
                    books[cname] = prices
        if not books:
            continue
        out.append({
            "market_id":       f"{sport_key}:{ev.get('id')}:h2h",
            "kind":            "moneyline",
            "sides":           ["away", "home"],
            "market_type_tag": MAINLINE_TAG,
            "commence_time":   ev.get("commence_time"),
            "matchup":         f"{away} @ {home}",
            "books":           books,
        })
    if pregame_only and skipped_live:
        print(f"  (skipped {skipped_live} in-progress / past games — live odds excluded)")
    return out


# ---------------------------------------------------------------------------
# Player props (per-event call)
# ---------------------------------------------------------------------------

def _list_events(api_key: str, sport_key: str) -> list[dict]:
    """Cheap event list (does not return odds)."""
    url = f"{ODDS_API_BASE}/{sport_key}/events"
    events, _ = _http_get(url, {"apiKey": api_key, "dateFormat": "iso"})
    return events


def fetch_prop_markets(api_key: str,
                       sport_key: str = "baseball_mlb",
                       regions: str = "us,us2,eu,uk",
                       prop_markets: list[str] | None = None,
                       pregame_only: bool = True,
                       min_minutes_to_start: int = 5,
                       verbose: bool = False) -> list[dict]:
    """Per-event call for player props. Each event is one API request whose
    cost scales with the number of regions + markets included. Keep the
    `prop_markets` list tight in production.

    `pregame_only=True` skips events that have already started — saves the
    per-event API credit AND avoids the live-vs-stale consensus contamination
    that produced the bogus +630 Rockies pick on mainlines."""
    import datetime as _dt
    props = prop_markets or DEFAULT_PROP_MARKETS_MLB
    events = _list_events(api_key, sport_key)

    now_utc = _dt.datetime.now(_dt.timezone.utc)
    cutoff  = now_utc + _dt.timedelta(minutes=min_minutes_to_start)

    out: list[dict] = []
    skipped_live = 0
    for ev in events:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        if pregame_only:
            ct = ev.get("commence_time")
            try:
                start = _dt.datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            except Exception:
                start = None
            if start is not None and start < cutoff:
                skipped_live += 1
                continue
        url = f"{ODDS_API_BASE}/{sport_key}/events/{ev_id}/odds"
        params = {
            "apiKey":     api_key,
            "regions":    regions,
            "markets":    ",".join(props),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        try:
            payload, _ = _http_get(url, params)
        except Exception as e:
            if verbose:
                print(f"  [props] {ev.get('away_team')} @ {ev.get('home_team')}: {e}")
            continue

        # Aggregate into (market_key, player, point) -> {book: {over, under}}
        by_key: dict[tuple, dict[str, dict[str, float]]] = {}
        for bm in payload.get("bookmakers", []):
            cname = _map_book(bm.get("key"))
            if not cname:
                continue
            for mk in bm.get("markets", []):
                mk_key = mk.get("key")
                if mk_key not in props:
                    continue
                for o in mk.get("outcomes", []):
                    side = (o.get("name") or "").lower()           # "over" / "under"
                    # The Odds API puts player name in "description" for props
                    player = o.get("description") or o.get("participant") or ""
                    point = o.get("point")
                    price = o.get("price")
                    if side not in ("over", "under") or not player or price is None:
                        continue
                    k = (mk_key, player, point)
                    by_key.setdefault(k, {}).setdefault(cname, {})[side] = price

        for (mk_key, player, point), per_book in by_key.items():
            cleaned = {b: p for b, p in per_book.items() if "over" in p and "under" in p}
            if not cleaned:
                continue
            out.append({
                "market_id":       f"{sport_key}:{ev_id}:{mk_key}:{player}:{point}",
                "kind":            "player_prop",
                "sides":           ["over", "under"],
                "market_type_tag": PROPS_TAG,
                "commence_time":   ev.get("commence_time"),
                "matchup":         f"{ev.get('away_team')} @ {ev.get('home_team')}",
                "player":          player,
                "market_key":      mk_key,
                "line":            point,
                "books":           cleaned,
            })
    if pregame_only and skipped_live:
        print(f"  (skipped {skipped_live} in-progress / past events — saved API credits)")
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(api_key: str | None = None,
        my_book: str = "FanDuel",
        bankroll: float = 10_000,
        min_edge: float = 0.02,
        skip_mainline: bool = False,
        skip_props: bool = False,
        prop_markets: list[str] | None = None,
        regions: str = "us,us2,eu,uk",
        sport_key: str = "baseball_mlb",
        pregame_only: bool = True,
        verbose: bool = True) -> pd.DataFrame:
    key = _get_key(api_key)
    markets: list[dict] = []

    if not skip_mainline:
        if verbose:
            print(f"Fetching mainline (h2h moneyline) markets "
                  f"({'pregame only' if pregame_only else 'pregame + live'}) …")
        try:
            ml = fetch_mainline_markets(key, sport_key, regions=regions,
                                        pregame_only=pregame_only)
            if verbose:
                print(f"  pulled {len(ml)} moneyline markets")
            markets.extend(ml)
        except Exception as e:
            print(f"  mainline fetch failed: {e}")

    if not skip_props:
        if verbose:
            print(f"Fetching player-prop markets "
                  f"({'pregame only' if pregame_only else 'pregame + live'}) …")
        try:
            pr = fetch_prop_markets(key, sport_key, regions=regions,
                                    prop_markets=prop_markets,
                                    pregame_only=pregame_only, verbose=verbose)
            if verbose:
                print(f"  pulled {len(pr)} prop markets")
            markets.extend(pr)
        except Exception as e:
            print(f"  props fetch failed: {e}")

    if not markets:
        print("No markets to score.")
        return pd.DataFrame()

    picks = mm.score_slate(markets, my_book=my_book, bankroll=bankroll,
                           min_edge_pct=min_edge)
    if picks.empty:
        print(f"No +EV bets found at {my_book} (min_edge_pct={min_edge}).")
        return picks

    # Carry matchup/player labels through for the log + CLV grading.
    meta = pd.DataFrame([{
        "market_id":      m["market_id"],
        "matchup":        m.get("matchup"),
        "player":         m.get("player"),
        "market_key":     m.get("market_key"),
        "line":           m.get("line"),
        "commence_time":  m.get("commence_time"),
    } for m in markets])
    picks = picks.merge(meta, on="market_id", how="left")
    picks["snapshot_ts"] = pd.Timestamp.utcnow().isoformat()

    picks.to_csv(PICKS_CSV, index=False)
    print(f"Wrote {len(picks)} picks -> {PICKS_CSV}")

    try:
        if LOG_CSV.exists():
            prev = pd.read_csv(LOG_CSV, low_memory=False)
            pd.concat([prev, picks], ignore_index=True).to_csv(LOG_CSV, index=False)
        else:
            picks.to_csv(LOG_CSV, index=False)
        print(f"Appended to log -> {LOG_CSV}")
    except Exception as e:
        print(f"  log append failed: {e}")

    return picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-key", default=None,
                    help="Defaults to ODDS_API_KEY env var, then to the hardcoded key.")
    ap.add_argument("--my-book", default="FanDuel",
                    help="The book you'll actually place the bet at.")
    ap.add_argument("--bankroll", type=float, default=10_000)
    ap.add_argument("--min-edge", type=float, default=0.02,
                    help="Minimum EV-per-dollar to surface a pick (e.g. 0.02 = 2%%).")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--regions", default="us,us2,eu,uk",
                    help="Regions to query — broader = more books in consensus.")
    ap.add_argument("--skip-mainline", action="store_true")
    ap.add_argument("--skip-props", action="store_true")
    ap.add_argument("--prop-markets", default=None,
                    help="Comma-separated Odds-API prop market keys (overrides defaults).")
    ap.add_argument("--include-live", action="store_true",
                    help="Include in-progress games. OFF by default — mixing live odds "
                         "(fast books) with stale lookahead-style odds (slow books) "
                         "produces nonsense consensus.")
    args = ap.parse_args()

    prop_markets = None
    if args.prop_markets:
        prop_markets = [s.strip() for s in args.prop_markets.split(",") if s.strip()]

    df = run(api_key=args.api_key, my_book=args.my_book, bankroll=args.bankroll,
             min_edge=args.min_edge, sport_key=args.sport, regions=args.regions,
             skip_mainline=args.skip_mainline, skip_props=args.skip_props,
             prop_markets=prop_markets, pregame_only=not args.include_live)
    if df is None or df.empty:
        return
    cols = ["matchup", "player", "market_key", "line", "side",
            "my_book", "my_price", "fair_price", "edge_pct",
            "stake", "multiplier", "plus_ev_vs_all"]
    keep = [c for c in cols if c in df.columns]
    print("\nTop 20 picks by recommended stake:")
    print(df[keep].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
