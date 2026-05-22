"""
props_grade.py
==============
Grades the historical props log (2026_props_log.csv) against actual outcomes
from the player accuracy log (2026_player_accuracy.csv) and computes:

  - Per-pick result (HIT / MISS / PUSH)
  - Profit per $1 stake (using the price logged at fetch time)
  - Closing line value (CLV) — for any prop with both an "open" and "close"
    stage row, the line and price movement is recorded
  - Season-level summary (W-L, ROI, CLV-positive %)

Outputs
-------
2026_props_accuracy.csv  — graded picks (one row per Value-flagged pick)
2026_props_clv.csv       — CLV deltas (one row per prop that has both stages)

Usage
-----
    python props_grade.py
    python props_grade.py --picks-file 2026_props_log.csv
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROPS_LOG_PATH      = HERE / "2026_props_log.csv"
PLAYER_ACC_PATH     = HERE / "2026_player_accuracy.csv"
PROPS_ACC_PATH      = HERE / "2026_props_accuracy.csv"
PROPS_CLV_PATH      = HERE / "2026_props_clv.csv"

# Map prop market → (player_type, actual_column_in_player_accuracy_log)
ACTUAL_COL_MAP: dict[str, tuple[str, str]] = {
    "pitcher_strikeouts":   ("pitcher", "actual_strikeouts"),
    "pitcher_walks":        ("pitcher", "actual_walks"),
    "pitcher_hits_allowed": ("pitcher", "actual_hits_allowed"),
    "pitcher_earned_runs":  ("pitcher", "actual_runs_allowed"),
    "pitcher_outs":         ("pitcher", "actual_outs"),    # raw outs (IP × 3 if absent)
    "batter_hits":          ("hitter",  "actual_hits"),
    "batter_home_runs":     ("hitter",  "actual_hr"),
    "batter_strikeouts":    ("hitter",  "actual_strikeouts"),
    "batter_walks":         ("hitter",  "actual_walks"),
    "batter_rbis":          ("hitter",  "actual_rbi"),
    "batter_runs_scored":   ("hitter",  "actual_runs"),
    # batter_total_bases is synthesized below from H + HR
}


def _norm(s: str | float) -> str:
    return str(s or "").strip().lower()


def _decimal_payout(american) -> Optional[float]:
    if american is None or pd.isna(american):
        return None
    a = float(american)
    if a > 0:
        return a / 100.0
    if a < 0:
        return 100.0 / abs(a)
    return None


def _profit_for_pick(side: str, line: float, actual: float, odds) -> Optional[float]:
    """Profit on a 1u stake. Returns None if push or actual missing."""
    if actual is None or pd.isna(actual):
        return None
    payout = _decimal_payout(odds)
    if payout is None:
        return None
    if abs(actual - line) < 1e-9:
        return 0.0   # push
    if side == "OVER":
        return payout if actual > line else -1.0
    if side == "UNDER":
        return payout if actual < line else -1.0
    return None


def _result_label(side: str, line: float, actual: float) -> str:
    if actual is None or pd.isna(actual):
        return "PENDING"
    if abs(actual - line) < 1e-9:
        return "PUSH"
    if side == "OVER":
        return "HIT" if actual > line else "MISS"
    if side == "UNDER":
        return "HIT" if actual < line else "MISS"
    return "—"


def _build_actuals_lookup(player_acc: pd.DataFrame) -> dict[tuple, dict]:
    """
    Index actuals by (game_date, mlb_id, player_type).
    Each value contains the columns score_pitchers/score_hitters wrote +
    a synthesized actual_total_bases.
    """
    if player_acc.empty:
        return {}
    df = player_acc.copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["player_type"] = df["player_type"].astype(str).str.lower()

    # Synthesize actual total bases (H − HR singles + HR×4)
    if "actual_hits" in df.columns and "actual_hr" in df.columns:
        h  = pd.to_numeric(df["actual_hits"], errors="coerce")
        hr = pd.to_numeric(df["actual_hr"], errors="coerce")
        df["actual_total_bases"] = (h - hr).clip(lower=0) + hr * 4
    # actual_outs from IP if missing
    if "actual_outs" not in df.columns and "actual_ip" in df.columns:
        df["actual_outs"] = pd.to_numeric(df["actual_ip"], errors="coerce") * 3.0

    out: dict[tuple, dict] = {}
    for _, r in df.iterrows():
        gd = r.get("game_date")
        try:
            mid = int(float(r["mlb_id"])) if pd.notna(r.get("mlb_id")) else None
        except (TypeError, ValueError):
            mid = None
        if not gd or mid is None:
            continue
        out[(gd, mid, str(r["player_type"]))] = r.to_dict()
    return out


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def grade_props(
    props_log_path: Path = PROPS_LOG_PATH,
    player_acc_path: Path = PLAYER_ACC_PATH,
    output_path: Path = PROPS_ACC_PATH,
) -> pd.DataFrame:
    if not props_log_path.exists():
        print(f"⚠️  {props_log_path} doesn't exist — nothing to grade.")
        return pd.DataFrame()
    if not player_acc_path.exists():
        print(f"⚠️  {player_acc_path} doesn't exist — can't look up actuals.")
        return pd.DataFrame()

    log = pd.read_csv(props_log_path, low_memory=False)
    if log.empty:
        return log

    # Only grade Value-flagged picks — that's our betting universe.
    if "flag" in log.columns:
        log["flag"] = log["flag"].astype(str).str.upper()
        # We keep the "open" stage row for each value pick (it's what was bet).
        if "clv_stage" in log.columns:
            log = log[(log["flag"] == "VALUE") & (log["clv_stage"].astype(str).str.lower() == "open")]
        else:
            log = log[log["flag"] == "VALUE"]
    if log.empty:
        print("No Value-flagged picks in the log to grade.")
        return log

    player_acc = pd.read_csv(player_acc_path, low_memory=False)
    actuals = _build_actuals_lookup(player_acc)

    out_rows: list[dict] = []
    for _, p in log.iterrows():
        market = str(p.get("market", ""))
        cfg = ACTUAL_COL_MAP.get(market)
        actual_col = "actual_total_bases" if market == "batter_total_bases" else (cfg[1] if cfg else None)
        kind = "hitter" if market.startswith("batter_") else "pitcher"
        if cfg is None and market != "batter_total_bases":
            continue

        gd = str(p.get("game_date", ""))[:10]
        try:
            mid = int(float(p["mlb_id"])) if pd.notna(p.get("mlb_id")) else None
        except (TypeError, ValueError):
            mid = None
        if not gd or mid is None or actual_col is None:
            continue
        actual_row = actuals.get((gd, mid, kind))
        if not actual_row:
            continue
        actual_val = actual_row.get(actual_col)
        actual_val = float(actual_val) if pd.notna(actual_val) else None

        side = str(p.get("side", "")).upper()
        line = float(p.get("line", 0))
        odds = p.get("over_odds") if side == "OVER" else p.get("under_odds")
        profit = _profit_for_pick(side, line, actual_val, odds)
        result = _result_label(side, line, actual_val)

        out_rows.append({
            "game_date": gd,
            "player_name": p.get("player_name"),
            "mlb_id": mid,
            "kind": kind,
            "market": market,
            "line": line,
            "side": side,
            "ev": p.get("ev"),
            "score": p.get("score"),
            "sportsbook": p.get("sportsbook"),
            "odds": odds,
            "proj_value": p.get("proj_value"),
            "actual_value": actual_val,
            "result": result,
            "profit_units": profit,
        })

    if not out_rows:
        print("No graded prop picks (no actuals matched).")
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    # Running totals
    out = out.sort_values(["game_date", "market"]).reset_index(drop=True)
    settled = out[out["profit_units"].notna()].copy()
    n_total   = len(settled)
    n_hit     = int((settled["result"] == "HIT").sum())
    n_miss    = int((settled["result"] == "MISS").sum())
    n_push    = int((settled["result"] == "PUSH").sum())
    units     = float(settled["profit_units"].fillna(0).sum())
    roi       = (units / n_total * 100.0) if n_total else 0.0

    out.to_csv(output_path, index=False)
    print(f"\nGraded {len(out):,} prop picks ({n_total} settled): "
          f"{n_hit}-{n_miss}-{n_push}  ·  {units:+.2f}u  ·  ROI {roi:+.1f}%")
    print(f"Wrote → {output_path}")
    return out


# ---------------------------------------------------------------------------
# CLV tracking
# ---------------------------------------------------------------------------
def compute_clv(
    props_log_path: Path = PROPS_LOG_PATH,
    output_path: Path = PROPS_CLV_PATH,
) -> pd.DataFrame:
    """
    For every prop that has BOTH a clv_stage='open' and clv_stage='close' row,
    record line/price movement. Positive CLV = the line moved your way after
    you logged the open price.
    """
    if not props_log_path.exists():
        print(f"⚠️  {props_log_path} doesn't exist.")
        return pd.DataFrame()
    log = pd.read_csv(props_log_path, low_memory=False)
    if "clv_stage" not in log.columns or log.empty:
        return pd.DataFrame()

    log["clv_stage"] = log["clv_stage"].astype(str).str.lower()
    key_cols = ["game_date", "player_name", "market", "sportsbook"]
    opens  = log[log["clv_stage"] == "open" ].copy()
    closes = log[log["clv_stage"] == "close"].copy()
    if opens.empty or closes.empty:
        print("Need both open and close stage rows to compute CLV — none yet.")
        return pd.DataFrame()

    # Inner join on key. Drop dupes per key keeping latest row of each stage.
    opens  = opens.sort_values("logged_at").drop_duplicates(subset=key_cols, keep="last")
    closes = closes.sort_values("logged_at").drop_duplicates(subset=key_cols, keep="last")
    merged = opens.merge(closes, on=key_cols, how="inner", suffixes=("_open", "_close"))

    # Compute deltas
    def _safe_diff(a, b):
        try:
            return float(b) - float(a)
        except (TypeError, ValueError):
            return None

    rows = []
    for _, r in merged.iterrows():
        side = str(r.get("side_open", "")).upper()
        line_open = r.get("line_open")
        line_close = r.get("line_close")
        # Take the price for whichever side we picked
        if side == "OVER":
            odds_open  = r.get("over_odds_open")
            odds_close = r.get("over_odds_close")
        else:
            odds_open  = r.get("under_odds_open")
            odds_close = r.get("under_odds_close")
        line_delta = _safe_diff(line_open, line_close)
        odds_delta = _safe_diff(odds_open, odds_close)

        # CLV positive if line moved against your side (you got better number)
        # OVER 5.5 → close opened 6.5 = line moved +1, BAD for over (line shifted away).
        # OVER 5.5 → close 5.0 = -0.5, GOOD for over. So clv_line = -line_delta * sign(side)
        sign = 1 if side == "OVER" else -1
        # If you took OVER and the line dropped, that's positive CLV.
        clv_line = (-(line_delta) * sign) if line_delta is not None else None

        rows.append({
            "game_date": r["game_date"],
            "player_name": r["player_name"],
            "market": r["market"],
            "sportsbook": r["sportsbook"],
            "side": side,
            "line_open": line_open,
            "line_close": line_close,
            "line_delta": line_delta,
            "odds_open": odds_open,
            "odds_close": odds_close,
            "odds_delta": odds_delta,
            "clv_line": clv_line,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["game_date", "player_name"]).reset_index(drop=True)
    out.to_csv(output_path, index=False)

    settled = out[out["clv_line"].notna()]
    pos_clv = int((settled["clv_line"] > 0).sum()) if len(settled) else 0
    pct = (pos_clv / len(settled) * 100.0) if len(settled) else 0.0
    print(f"\nCLV: {len(settled)} pairs analyzed  ·  positive-CLV rate {pct:.1f}%")
    print(f"Wrote → {output_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks-file", default=str(PROPS_LOG_PATH))
    p.add_argument("--player-acc", default=str(PLAYER_ACC_PATH))
    p.add_argument("--output", default=str(PROPS_ACC_PATH))
    p.add_argument("--clv-only", action="store_true")
    args = p.parse_args()

    if not args.clv_only:
        grade_props(
            props_log_path=Path(args.picks_file),
            player_acc_path=Path(args.player_acc),
            output_path=Path(args.output),
        )
    compute_clv(props_log_path=Path(args.picks_file))


if __name__ == "__main__":
    main()
