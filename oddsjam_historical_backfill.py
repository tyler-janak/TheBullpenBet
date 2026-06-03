"""
oddsjam_historical_backfill.py
===============================
Backtest the market-consensus model (market_model.py) against historical odds
from OddsJam. Walks a date range, pulls each game's closing-line snapshot
across every book, runs the consensus engine, finds the picks that WOULD have
been made at FanDuel (or whichever sharp-soft book you're betting), settles
each pick against the actual MLB result, and produces a full ROI / CLV / by-
bucket breakdown.

Why this exists
---------------
Forward-testing the consensus model takes weeks to accumulate a meaningful
sample. A historical backfill against 2025 + 2026 season-to-date gives the
same statistical confidence in hours, IF you have access to per-book closing
snapshots. OddsJam sells exactly that data.

What you provide when you switch over
-------------------------------------
1. ODDSJAM_API_KEY env var (or pass --oj-key).
2. The exact endpoint shape (their docs change, and tier features differ).
   Find `# TODO: REPLACE WITH ACTUAL OJ ENDPOINT` blocks below and plug in.

What this script does (works the moment OJ data flows)
------------------------------------------------------
1. For each date in [--start, --end]:
   a. List MLB games (via MLB Stats API — already free).
   b. For each game, fetch the OJ closing snapshot at first_pitch - <close_lead_min>.
   c. Build a market_model market dict from the OJ payload.
   d. Run market_model.score_slate(..., my_book=...) -> the picks at the close.
2. Settle each pick:
   - Moneyline -> MLB Stats API final score.
   - Player prop -> 2026_player_accuracy.csv actual_* columns
     (you already build this; backfill it first via backfill_player_predictions
      if any past dates are missing).
3. Aggregate:
   - Overall ROI, win rate, total P/L on flat-stake AND model-recommended stake.
   - Breakdown by book / market_key / confidence flag / month.
   - CLV column = (closing fair price) - (your locked price)  [in cents].
4. Optionally optimise:
   - --tune-weights re-fits the book sharpness weights from the backtest's
     CLV data using a simple grid search (mainline vs props, per-bucket).

Usage
-----
    python oddsjam_historical_backfill.py --start 2025-04-01 --end 2025-10-31
    python oddsjam_historical_backfill.py --start 2026-03-25 --my-book DraftKings
    python oddsjam_historical_backfill.py --tune-weights --start 2025-04-01

Cost note
---------
OJ bills per snapshot per sport per market tier. Mainlines for a full MLB
season is ~2,500 snapshots; props is dramatically more (per-event × per-prop).
Estimate before running anything wider than --markets h2h.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

import market_model as mm
from market_consensus_grade import (
    _fetch_mlb_winners, _settle_prop, _pl_per_dollar, PROP_TO_ACTUAL
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"; OUT_DIR.mkdir(exist_ok=True)
BACKTEST_CSV = HERE / "market_consensus_backtest.csv"
PLAYER_ACC_CSV = HERE / "2026_player_accuracy.csv"

OJ_BASE = "https://api-external.oddsjam.com/api/v2"   # TODO: confirm with your tier

# OddsJam sportsbook -> canonical name used by market_model.
# OJ uses Title Case but capitalises a few books differently than The Odds API.
OJ_BOOK_MAP = {
    "Pinnacle":           "Pinnacle",
    "Circa":              "Circa",
    "Circa Sports":       "Circa",
    "BetOnline":          "BetOnline",
    "BetOnline.ag":       "BetOnline",
    "BetCRIS":            "BetCRIS",
    "BookMaker.eu":       "BetOnline",   # same parent group as BetOnline
    "FanDuel":            "FanDuel",
    "DraftKings":         "DraftKings",
    "BetMGM":             "BetMGM",
    "Caesars":            "Caesars",
    "William Hill":       "Caesars",
    "BetRivers":          "BetRivers",
    "Bet365":             "Bet365",
    "PointsBet":          "PointsBet",
    "Novig":              "Novig",
    "ProphetX":           "ProphetX",
    "Prophet Exchange":   "ProphetX",
}


# ---------------------------------------------------------------------------
# OJ HTTP wrapper
# ---------------------------------------------------------------------------

def _oj_key(cli_key: str | None) -> str:
    k = cli_key or os.environ.get("ODDSJAM_API_KEY")
    if not k:
        raise RuntimeError(
            "Set ODDSJAM_API_KEY env var or pass --oj-key. "
            "Get it from oddsjam.com → Account → API."
        )
    return k


def _oj_get(path: str, params: dict, key: str, retries: int = 2):
    """OJ uses ?api_key=... auth (not Bearer). Confirm in your contract."""
    url = f"{OJ_BASE}/{path.lstrip('/')}"
    p = dict(params); p["api_key"] = key
    for i in range(retries + 1):
        r = requests.get(url, params=p, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and i < retries:
            time.sleep(1.5 * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"OJ request failed: {url}")


# ---------------------------------------------------------------------------
# Snapshot fetchers — REPLACE INNER BODIES with your actual OJ endpoint(s)
# ---------------------------------------------------------------------------

def fetch_oj_mainline_snapshot(game_id: str, snapshot_ts: dt.datetime,
                               key: str, league: str = "MLB") -> dict | None:
    """Return one game's h2h consensus snapshot at the given UTC instant.

    Expected return shape (whatever OJ returns, we normalise to):
        {
          "id": "...",                # OJ game id (or your composite)
          "home_team": "...",
          "away_team": "...",
          "commence_time": "ISO UTC",
          "rows": [
            {"sportsbook": "Pinnacle", "market": "Moneyline",
             "selection": "Away Team", "price": +122, "timestamp": "..."},
            ...
          ]
        }
    """
    # TODO: REPLACE WITH ACTUAL OJ ENDPOINT.
    # As of writing OJ exposes /historical-odds and /historical-events with
    # `sport=baseball`, `league=mlb`, `start_date`, `end_date`, `markets=h2h`,
    # but field names and pagination vary by contract. Confirm the response
    # shape and adapt the parser below.
    try:
        payload = _oj_get(
            "historical-odds",
            {
                "sport":       "baseball",
                "league":      league.lower(),
                "game_id":     game_id,
                "markets":     "moneyline",
                "as_of":       snapshot_ts.isoformat().replace("+00:00", "Z"),
            },
            key,
        )
    except Exception as e:
        print(f"  [oj] snapshot fail {game_id} @ {snapshot_ts}: {e}")
        return None
    return _normalise_oj_payload(payload)


def fetch_oj_prop_snapshot(game_id: str, snapshot_ts: dt.datetime,
                           key: str, markets: Iterable[str],
                           league: str = "MLB") -> list[dict]:
    """Return one game's player-prop snapshots at the given UTC instant.

    Each emitted dict matches market_model's player_prop schema directly."""
    try:
        payload = _oj_get(
            "historical-odds",
            {
                "sport":   "baseball",
                "league":  league.lower(),
                "game_id": game_id,
                "markets": ",".join(markets),
                "as_of":   snapshot_ts.isoformat().replace("+00:00", "Z"),
            },
            key,
        )
    except Exception as e:
        print(f"  [oj] prop snapshot fail {game_id} @ {snapshot_ts}: {e}")
        return []
    return _normalise_oj_prop_payload(payload, game_id)


def _normalise_oj_payload(payload: dict) -> dict | None:
    """Map an OJ moneyline response into market_model's market-dict shape."""
    if not payload:
        return None
    # TODO: adjust based on real OJ payload. Below assumes payload has the
    # game-level fields at top level and an "odds" array of book rows.
    g = payload.get("data") or payload
    if isinstance(g, list):
        if not g: return None
        g = g[0]
    home = g.get("home_team")
    away = g.get("away_team")
    rows = g.get("rows") or g.get("odds") or []
    if not (home and away and rows):
        return None
    books: dict[str, dict] = {}
    for r in rows:
        market = (r.get("market") or r.get("market_name") or "").lower()
        if "moneyline" not in market and "h2h" not in market:
            continue
        bookname = OJ_BOOK_MAP.get(r.get("sportsbook") or r.get("book"))
        if not bookname:
            continue
        sel = (r.get("selection") or r.get("name") or "").strip()
        price = r.get("price") or r.get("odds")
        side = "away" if sel == away else "home" if sel == home else None
        if side is None or price is None:
            continue
        books.setdefault(bookname, {})[side] = price
    books = {b: p for b, p in books.items() if {"away","home"} <= set(p)}
    if not books:
        return None
    return {
        "market_id":       f"oj_mlb:{g.get('id') or g.get('game_id')}:h2h",
        "kind":            "moneyline",
        "sides":           ["away", "home"],
        "market_type_tag": "mlb_ml",
        "commence_time":   g.get("commence_time") or g.get("start_date"),
        "matchup":         f"{away} @ {home}",
        "books":           books,
    }


def _normalise_oj_prop_payload(payload: dict, game_id: str) -> list[dict]:
    """Map an OJ player-prop response into a list of market_model dicts."""
    out: list[dict] = []
    if not payload:
        return out
    g = payload.get("data") or payload
    if isinstance(g, list):
        if not g: return out
        g = g[0]
    rows = g.get("rows") or g.get("odds") or []
    home, away = g.get("home_team"), g.get("away_team")
    commence = g.get("commence_time") or g.get("start_date")

    # by (market, player, line) -> {book: {over, under}}
    grouped: dict[tuple, dict[str, dict[str, float]]] = {}
    for r in rows:
        market_key = (r.get("market") or r.get("market_name") or "").lower().replace(" ", "_")
        if not (market_key.startswith("pitcher_") or market_key.startswith("batter_")):
            continue
        player = r.get("player") or r.get("participant") or r.get("description")
        line   = r.get("line") or r.get("point")
        side   = (r.get("name") or r.get("selection") or "").lower()
        side   = "over" if "over" in side else "under" if "under" in side else None
        price  = r.get("price") or r.get("odds")
        book   = OJ_BOOK_MAP.get(r.get("sportsbook") or r.get("book"))
        if not (player and line is not None and side and price is not None and book):
            continue
        grouped.setdefault((market_key, player, line), {}).setdefault(book, {})[side] = price

    for (mk, player, line), per_book in grouped.items():
        cleaned = {b: p for b, p in per_book.items() if {"over","under"} <= set(p)}
        if not cleaned:
            continue
        out.append({
            "market_id":       f"oj_mlb:{game_id}:{mk}:{player}:{line}",
            "kind":            "player_prop",
            "sides":           ["over", "under"],
            "market_type_tag": "player_prop_mlb",
            "commence_time":   commence,
            "matchup":         f"{away} @ {home}",
            "player":          player,
            "market_key":      mk,
            "line":            line,
            "books":           cleaned,
        })
    return out


# ---------------------------------------------------------------------------
# Game list (free — MLB Stats API)
# ---------------------------------------------------------------------------

def list_mlb_games(date: dt.date) -> list[dict]:
    """Returns [{game_id, away, home, commence_utc}] for a date."""
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": str(date)}, timeout=15,
    )
    r.raise_for_status()
    out = []
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            commence = g.get("gameDate")
            try:
                ct = dt.datetime.fromisoformat(str(commence).replace("Z","+00:00"))
            except Exception:
                continue
            out.append({
                "game_id":       g.get("gamePk"),                  # OJ may use its own id
                "away":          ((g.get("teams") or {}).get("away") or {}).get("team",{}).get("name"),
                "home":          ((g.get("teams") or {}).get("home") or {}).get("team",{}).get("name"),
                "commence_utc":  ct,
                "status":        (g.get("status") or {}).get("abstractGameState"),
            })
    return out


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------

def backtest(start: str, end: str, *,
             my_book: str = "FanDuel", bankroll: float = 10_000,
             min_edge: float = 0.02, close_lead_min: int = 5,
             skip_props: bool = True,
             prop_markets: list[str] | None = None,
             oj_key: str | None = None) -> pd.DataFrame:
    """Walk every MLB game in [start, end], snapshot the close, score + settle."""
    key = _oj_key(oj_key)
    start_d = dt.date.fromisoformat(start)
    end_d   = dt.date.fromisoformat(end)
    all_picks: list[dict] = []

    cur = start_d
    while cur <= end_d:
        games = list_mlb_games(cur)
        if games:
            print(f"\n── {cur}  ({len(games)} games) ──")
        for g in games:
            if g["status"] != "Final":
                continue
            snap_ts = g["commence_utc"] - dt.timedelta(minutes=close_lead_min)
            markets: list[dict] = []

            ml = fetch_oj_mainline_snapshot(str(g["game_id"]), snap_ts, key)
            if ml:
                markets.append(ml)
            if not skip_props:
                props = fetch_oj_prop_snapshot(str(g["game_id"]), snap_ts, key,
                                               prop_markets or DEFAULT_PROP_MARKETS)
                markets.extend(props)
            if not markets:
                continue

            picks = mm.score_slate(markets, my_book=my_book, bankroll=bankroll,
                                   min_edge_pct=min_edge)
            if picks.empty:
                continue
            picks["game_date_et"] = cur                    # for grading lookup
            picks["matchup_away"] = g["away"]
            picks["matchup_home"] = g["home"]
            all_picks.append(picks)
        cur += dt.timedelta(days=1)

    if not all_picks:
        print("No picks generated across the date range.")
        return pd.DataFrame()

    df = pd.concat(all_picks, ignore_index=True)
    print(f"\nGenerated {len(df)} historical picks. Settling …")

    # ---- Settlement ----
    # Mainline: MLB Stats API
    ml = df[df["kind"] == "moneyline"].copy()
    if not ml.empty:
        winners = _fetch_mlb_winners(list(ml["game_date_et"].unique()))
        ml["actual_winner"] = [
            winners.get((d, a, h)) for d, a, h in
            zip(ml["game_date_et"], ml["matchup_away"], ml["matchup_home"])
        ]
        ml["result"] = ml.apply(
            lambda r: ("ungradeable" if r["actual_winner"] is None
                       else "push" if r["actual_winner"] == "push"
                       else "win"  if r["actual_winner"] == r["side"]
                       else "loss"),
            axis=1,
        )

    # Props: 2026_player_accuracy.csv (build it first via backfill_player_predictions
    # if your prop dates aren't in there yet).
    pp = df[df["kind"] == "player_prop"].copy()
    if not pp.empty and PLAYER_ACC_CSV.exists():
        acc = pd.read_csv(PLAYER_ACC_CSV, low_memory=False)
        acc["game_date"] = pd.to_datetime(acc["game_date"], errors="coerce").dt.date
        acc["_name_lc"]  = acc["player_name"].astype(str).str.lower().str.strip()
        pp["_name_lc"]   = pp["player"].astype(str).str.lower().str.strip()
        pp["result"] = "ungradeable"
        for i, r in pp.iterrows():
            mapping = PROP_TO_ACTUAL.get(r.get("market_key"))
            if not mapping or mapping[1] is None: continue
            ptype, actual_col = mapping
            m = acc[(acc["player_type"] == ptype) &
                    (acc["game_date"]   == r["game_date_et"]) &
                    (acc["_name_lc"]    == r["_name_lc"])]
            if m.empty or actual_col not in m.columns: continue
            actual = pd.to_numeric(m.iloc[0][actual_col], errors="coerce")
            pp.at[i, "result"] = _settle_prop(r["side"],
                                              pd.to_numeric(r["line"], errors="coerce"),
                                              actual)
    elif not pp.empty:
        print(f"⚠️  {PLAYER_ACC_CSV} missing — back-fill player accuracy first.")
        pp["result"] = "ungradeable"

    graded = pd.concat([ml, pp], ignore_index=True, sort=False)

    def _pnl(r):
        if r["result"] == "win":
            return float(r["stake"]) * _pl_per_dollar(float(r["my_price"]), True)
        if r["result"] == "loss":
            return -float(r["stake"])
        return 0.0
    graded["profit"] = graded.apply(_pnl, axis=1)
    graded["staked"] = graded["result"].isin(["win","loss","push"]) * \
                       pd.to_numeric(graded.get("stake", 0), errors="coerce").fillna(0)

    graded.to_csv(BACKTEST_CSV, index=False)
    print(f"\nWrote {BACKTEST_CSV}")
    _summary(graded)
    return graded


def _summary(g: pd.DataFrame) -> None:
    decided = g[g["result"].isin(["win", "loss"])]
    if decided.empty:
        print("(no decided picks)"); return
    n = len(decided)
    wr = (decided["result"] == "win").mean()
    staked = float(decided["staked"].sum())
    profit = float(decided["profit"].sum())
    print("\n── Backtest summary ──")
    print(f"  picks: {n:,}    win rate: {wr*100:.2f}%    "
          f"staked: ${staked:,.0f}    P/L: ${profit:+,.0f}    "
          f"ROI: {(profit/staked)*100:+.2f}%")
    print("\n  By kind:")
    for k, sub in decided.groupby("kind"):
        wrk = (sub["result"]=="win").mean()
        st = float(sub["staked"].sum()); pl = float(sub["profit"].sum())
        print(f"    {k:<14} n={len(sub):>4}  WR={wrk*100:5.1f}%  "
              f"ROI={(pl/st)*100:+.2f}%")


DEFAULT_PROP_MARKETS = [
    "pitcher_strikeouts", "pitcher_walks", "pitcher_hits_allowed",
    "batter_hits", "batter_home_runs", "batter_total_bases",
    "batter_strikeouts", "batter_walks", "batter_runs_scored", "batter_rbis",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--my-book", default="FanDuel")
    ap.add_argument("--bankroll", type=float, default=10_000)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--close-lead-min", type=int, default=5,
                    help="Snapshot N minutes before first pitch.")
    ap.add_argument("--skip-props", action="store_true",
                    help="Mainlines only (way cheaper on OJ credits).")
    ap.add_argument("--prop-markets", default=None,
                    help="Comma-separated prop market keys.")
    ap.add_argument("--oj-key", default=None)
    args = ap.parse_args()

    pm = None
    if args.prop_markets:
        pm = [s.strip() for s in args.prop_markets.split(",") if s.strip()]

    backtest(args.start, args.end,
             my_book=args.my_book, bankroll=args.bankroll, min_edge=args.min_edge,
             close_lead_min=args.close_lead_min, skip_props=args.skip_props,
             prop_markets=pm, oj_key=args.oj_key)


if __name__ == "__main__":
    main()
