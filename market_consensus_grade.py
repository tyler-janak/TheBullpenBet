"""
market_consensus_grade.py
==========================
Settle the picks logged by oddsapi_fetch.py against actual game results, then
emit ROI / win rate / per-bucket breakdowns. This is the forward-test grader
for the market-consensus model.

Inputs
------
- 2026_market_consensus_log.csv     (written by oddsapi_fetch.py)
- 2026_player_accuracy.csv          (built by grade_player_predictions.py)
                                    -> actual_* columns by (player_name, game_date)
- MLB Stats API                     -> game-level scores for moneyline grading

Outputs
-------
- 2026_market_consensus_graded.csv  (every gradable pick, win/loss/push, profit)
- printed summary: overall ROI, win rate, edges by market_key/book/side

What's NOT here yet
-------------------
- CLV (closing-line value). Needs a separate "close snapshot" cron tick that
  fires ~5 min before first pitch and records the consensus fair price + your
  book's price at the close. Then CLV = pre_close_price - close_price per pick.
- Historical backtest. Needs The Odds API historical endpoint, which requires
  a tier upgrade. Once you have it, see the TODO at the bottom for the call.

Usage
-----
    python market_consensus_grade.py
    python market_consensus_grade.py --since 2026-05-01 --min-stake 5
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
LOG_CSV          = HERE / "2026_market_consensus_log.csv"
PLAYER_ACC_CSV   = HERE / "2026_player_accuracy.csv"
GRADED_CSV       = HERE / "2026_market_consensus_graded.csv"

# Map an Odds-API prop market key to (player_type, accuracy CSV actual column).
# None means we don't have a clean column for it in 2026_player_accuracy.csv
# (e.g. total bases requires 1B/2B/3B splits we don't store) — those skip.
PROP_TO_ACTUAL = {
    "pitcher_strikeouts":  ("pitcher", "actual_strikeouts"),
    "pitcher_walks":       ("pitcher", "actual_walks"),
    "pitcher_hits_allowed":("pitcher", "actual_hits_allowed"),
    "pitcher_outs":        ("pitcher", "actual_outs"),
    "batter_hits":         ("hitter",  "actual_hits"),
    "batter_home_runs":    ("hitter",  "actual_hr"),
    "batter_strikeouts":   ("hitter",  "actual_strikeouts"),
    "batter_walks":        ("hitter",  "actual_walks"),
    "batter_runs_scored":  ("hitter",  "actual_runs"),
    "batter_rbis":         ("hitter",  "actual_rbi"),
    "batter_total_bases":  ("hitter",  None),
    "pitcher_record_a_win":("pitcher", None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pl_per_dollar(american_odds: float, won: bool) -> float:
    """Return profit per $1 staked, ignoring push (handle separately)."""
    if not won:
        return -1.0
    if american_odds >= 0:
        return american_odds / 100.0
    return 100.0 / abs(american_odds)


def _split_matchup(s: str) -> tuple[str | None, str | None]:
    if not isinstance(s, str) or " @ " not in s:
        return None, None
    a, h = s.split(" @ ", 1)
    return a.strip(), h.strip()


def _fetch_mlb_winners(dates: list[dt.date]) -> dict[tuple, str]:
    """For a list of ET dates, hit MLB Stats API and return
    {(date, away_team, home_team) -> 'away' | 'home' | 'push'} for final games."""
    out: dict[tuple, str] = {}
    for d in sorted(set(dates)):
        if d is None:
            continue
        url = "https://statsapi.mlb.com/api/v1/schedule"
        try:
            r = requests.get(url, params={"sportId": 1, "date": str(d)}, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            print(f"  [mlb-api] {d}: {e}")
            continue
        for day in payload.get("dates", []):
            for g in day.get("games", []):
                state = (g.get("status") or {}).get("abstractGameState", "")
                if state != "Final":
                    continue
                away = ((g.get("teams") or {}).get("away") or {}).get("team", {}).get("name")
                home = ((g.get("teams") or {}).get("home") or {}).get("team", {}).get("name")
                a_score = ((g.get("teams") or {}).get("away") or {}).get("score")
                h_score = ((g.get("teams") or {}).get("home") or {}).get("score")
                if not (away and home and a_score is not None and h_score is not None):
                    continue
                if a_score > h_score:
                    winner = "away"
                elif h_score > a_score:
                    winner = "home"
                else:
                    winner = "push"            # MLB doesn't tie, but defensive
                out[(d, away, home)] = winner
    return out


def _settle_prop(side: str, line: float, actual: float) -> str:
    """Return 'win' / 'loss' / 'push'."""
    if pd.isna(actual) or pd.isna(line):
        return "ungradeable"
    if float(actual) == float(line):
        return "push"
    actual_side = "over" if float(actual) > float(line) else "under"
    return "win" if actual_side == side else "loss"


# ---------------------------------------------------------------------------
# Main grading
# ---------------------------------------------------------------------------

def grade(log_csv: Path = LOG_CSV,
          player_acc_csv: Path = PLAYER_ACC_CSV,
          out_csv: Path = GRADED_CSV,
          since: str | None = None,
          min_stake: float = 0.0,
          grade_buffer_hours: int = 4) -> pd.DataFrame:
    if not log_csv.exists():
        print(f"⚠️  No log yet at {log_csv} — run oddsapi_fetch.py a few times first.")
        return pd.DataFrame()

    df = pd.read_csv(log_csv, low_memory=False)
    if df.empty:
        print("Log is empty.")
        return pd.DataFrame()

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True, errors="coerce")
    df["commence_dt"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    df["game_date_et"] = df["commence_dt"].dt.tz_convert("America/New_York").dt.date

    # Keep only the most recent snapshot per (market_id, side) so we don't
    # double-count when oddsapi_fetch ran multiple times for the same pick.
    df = (df.sort_values("snapshot_ts")
            .groupby(["market_id", "side"], as_index=False)
            .tail(1)
            .reset_index(drop=True))

    if since:
        cutoff_dt = pd.to_datetime(since, utc=True)
        df = df[df["commence_dt"] >= cutoff_dt]

    if min_stake > 0 and "stake" in df.columns:
        df = df[pd.to_numeric(df["stake"], errors="coerce").fillna(0) >= min_stake]

    # Only grade games that finished long enough ago to be reliably final.
    now_utc = pd.Timestamp.utcnow()
    settle_cutoff = now_utc - pd.Timedelta(hours=grade_buffer_hours)
    df_to_grade = df[df["commence_dt"] < settle_cutoff].copy()
    not_yet = len(df) - len(df_to_grade)
    if not_yet:
        print(f"  ({not_yet} pick(s) not yet gradable — game in future / too recent)")
    if df_to_grade.empty:
        print("Nothing to grade yet.")
        return pd.DataFrame()

    # Split matchup -> away/home for mainline lookup
    df_to_grade[["matchup_away", "matchup_home"]] = df_to_grade["matchup"].apply(
        lambda s: pd.Series(_split_matchup(s)))

    # ---------- Mainline (moneyline) ----------
    ml = df_to_grade[df_to_grade["kind"] == "moneyline"].copy()
    if not ml.empty:
        print(f"Grading {len(ml)} moneyline picks "
              f"({ml['game_date_et'].nunique()} dates) via MLB Stats API …")
        winners = _fetch_mlb_winners(list(ml["game_date_et"].unique()))
        ml["actual_winner"] = [
            winners.get((d, a, h)) for d, a, h in
            zip(ml["game_date_et"], ml["matchup_away"], ml["matchup_home"])
        ]
        ml["result"] = ml.apply(
            lambda r: ("ungradeable" if r["actual_winner"] is None
                       else "push" if r["actual_winner"] == "push"
                       else "win" if r["actual_winner"] == r["side"]
                       else "loss"),
            axis=1,
        )
    else:
        ml = pd.DataFrame()

    # ---------- Player props ----------
    pp = df_to_grade[df_to_grade["kind"] == "player_prop"].copy()
    if not pp.empty and player_acc_csv.exists():
        print(f"Grading {len(pp)} prop picks against {player_acc_csv.name} …")
        acc = pd.read_csv(player_acc_csv, low_memory=False)
        acc["game_date"] = pd.to_datetime(acc["game_date"], errors="coerce").dt.date
        acc["_name_lc"]  = acc["player_name"].astype(str).str.lower().str.strip()
        pp["_name_lc"]   = pp["player"].astype(str).str.lower().str.strip()
        pp["result"]   = "ungradeable"
        pp["actual_value"] = pd.NA
        for i, r in pp.iterrows():
            mk = r.get("market_key")
            mapping = PROP_TO_ACTUAL.get(mk)
            if not mapping or mapping[1] is None:
                continue                       # not supported (e.g. total_bases)
            ptype, actual_col = mapping
            match = acc[(acc["player_type"] == ptype) &
                        (acc["game_date"]   == r["game_date_et"]) &
                        (acc["_name_lc"]    == r["_name_lc"])]
            if match.empty or actual_col not in match.columns:
                continue
            actual = pd.to_numeric(match.iloc[0][actual_col], errors="coerce")
            line = pd.to_numeric(r.get("line"), errors="coerce")
            pp.at[i, "actual_value"] = actual
            pp.at[i, "result"] = _settle_prop(r["side"], line, actual)
    elif not pp.empty:
        print(f"⚠️  {player_acc_csv} missing — skipping {len(pp)} prop picks.")
        pp["result"] = "ungradeable"

    graded = pd.concat([ml, pp], ignore_index=True, sort=False)
    if graded.empty:
        print("No picks gradable.")
        return graded

    # P/L per pick
    def _pnl(row):
        if row["result"] == "win":
            return float(row.get("stake", 0)) * _pl_per_dollar(float(row["my_price"]), True)
        if row["result"] == "loss":
            return -float(row.get("stake", 0))
        return 0.0          # push or ungradeable
    graded["profit"] = graded.apply(_pnl, axis=1)
    graded["staked"] = graded["result"].isin(["win", "loss", "push"]) * pd.to_numeric(
        graded.get("stake", 0), errors="coerce").fillna(0)

    graded.to_csv(out_csv, index=False)
    print(f"Wrote {len(graded)} graded picks -> {out_csv}")

    _print_summary(graded)
    return graded


def _print_summary(g: pd.DataFrame) -> None:
    gradable = g[g["result"].isin(["win", "loss", "push"])].copy()
    if gradable.empty:
        print("(no gradable picks yet — all in 'ungradeable')")
        return

    def _block(df, label):
        n = len(df)
        wins   = (df["result"] == "win").sum()
        losses = (df["result"] == "loss").sum()
        pushes = (df["result"] == "push").sum()
        decided = wins + losses
        wr = wins / decided if decided else float("nan")
        staked = float(pd.to_numeric(df.get("staked", 0), errors="coerce").sum())
        profit = float(pd.to_numeric(df.get("profit", 0), errors="coerce").sum())
        roi = (profit / staked) if staked else float("nan")
        print(f"  {label:<26}  n={n:>4}  W-L-P={wins}-{losses}-{pushes}  "
              f"WR={wr*100:5.1f}%  staked=${staked:,.0f}  P/L=${profit:+,.0f}  ROI={roi*100:+5.2f}%")

    print("\n── Summary ──")
    _block(gradable, "ALL gradable picks")
    print("\n  By kind:")
    for k, sub in gradable.groupby("kind"):
        _block(sub, k)
    print("\n  By book:")
    for b, sub in gradable.groupby("my_book"):
        _block(sub, b)
    if "plus_ev_vs_all" in gradable.columns:
        print("\n  Confidence split:")
        _block(gradable[gradable["plus_ev_vs_all"] == True],  "plus_ev_vs_all=True")
        _block(gradable[gradable["plus_ev_vs_all"] == False], "plus_ev_vs_all=False")
    if "market_key" in gradable.columns and gradable["kind"].eq("player_prop").any():
        print("\n  By prop market:")
        for mk, sub in gradable[gradable["kind"] == "player_prop"].groupby("market_key"):
            _block(sub, str(mk))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log",        default=str(LOG_CSV))
    ap.add_argument("--player-acc", default=str(PLAYER_ACC_CSV))
    ap.add_argument("--out",        default=str(GRADED_CSV))
    ap.add_argument("--since",      default=None,
                    help="Only grade picks for games on/after this date (YYYY-MM-DD).")
    ap.add_argument("--min-stake",  type=float, default=0.0,
                    help="Skip picks with recommended stake below this.")
    ap.add_argument("--grade-buffer-hours", type=int, default=4,
                    help="Wait this many hours after first pitch before grading "
                         "(makes sure the box-score is final).")
    args = ap.parse_args()

    grade(log_csv=Path(args.log),
          player_acc_csv=Path(args.player_acc),
          out_csv=Path(args.out),
          since=args.since,
          min_stake=args.min_stake,
          grade_buffer_hours=args.grade_buffer_hours)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# TODO (when you upgrade to The Odds API historical tier):
# Build market_consensus_backtest.py that, for each MLB game in 2025+2026:
#   1. Call /v4/historical/sports/baseball_mlb/odds?date=<T-5min before first pitch>
#      with regions=us,us2,eu,uk and markets=h2h (and props per-event).
#   2. Pass the response through oddsapi_fetch._build_market dicts.
#   3. Run market_model.score_slate(my_book="FanDuel", bankroll=10000) on it.
#   4. Grade each pick the same way grade() does above.
#   5. Emit a season-level ROI / CLV report.
# Cost: ~10 credits per historical snapshot per region per market. Budget
# accordingly before you fire it.
# ----------------------------------------------------------------------------
