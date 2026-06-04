"""
print_all_stats.py
==================
One-stop dashboard for every accuracy / grading metric the pipeline produces.

Reads (whatever exists):
  models/pitcher_metrics.csv          - training holdout MAE per target/model
  models/hitter_metrics.csv
  2026_player_accuracy.csv            - live projection MAE per stat
  2026_picks_accuracy.csv             - game moneyline pick W-L / ROI
  2026_nrfi_accuracy.csv              - NRFI W-L / ROI
  2026_props_accuracy.csv             - projection-based prop grade (if present)
  2026_props_clv.csv                  - projection-based CLV
  2026_market_consensus_graded.csv    - market-consensus W-L / ROI

Prints a unified table per section so you can see at a glance which lens is
performing, which stats are above/below sportsbook benchmarks, and where the
edge (or lack of it) is hiding.

Usage
-----
    python print_all_stats.py
    python print_all_stats.py --since 2026-05-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Sportsbook benchmark MAEs for each pitcher stat (rough industry numbers —
# what a competitive book / sharp model achieves on closing-line graded props).
BENCHMARK_MAE = {
    "strikeouts":     (1.5, 1.7),   # green if <= 1.5, yellow if <= 1.7
    "walks":          (1.0, 1.2),
    "hits_allowed":   (1.5, 1.8),
    "ip":             (0.9, 1.1),
    "runs_allowed":   (1.4, 1.7),
    "hits":           (0.6, 0.7),
    "hr":             (0.25, 0.35),
    "runs":           (0.55, 0.7),
    "rbi":            (0.55, 0.7),
    "pa":             (0.6, 0.8),
}

C_GREEN = "\033[92m"; C_YEL = "\033[93m"; C_RED = "\033[91m"; C_DIM = "\033[2m"; C_END = "\033[0m"


def _color(val: float, bench: tuple[float, float] | None) -> str:
    if bench is None or not np.isfinite(val):
        return f"{val:6.3f}"
    g, y = bench
    if val <= g:   return f"{C_GREEN}{val:6.3f}{C_END}"
    if val <= y:   return f"{C_YEL}{val:6.3f}{C_END}"
    return f"{C_RED}{val:6.3f}{C_END}"


def _read(name: str) -> pd.DataFrame | None:
    p = HERE / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"  [warn] couldn't read {name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
def section_training_holdout() -> None:
    print(f"\n{'='*72}\nTRAINING HOLDOUT MAE  (what the trainer printed)\n{'='*72}")
    for which, fn in [("PITCHER", "models/pitcher_metrics.csv"),
                      ("HITTER",  "models/hitter_metrics.csv")]:
        df = _read(fn)
        if df is None or df.empty:
            print(f"  {which:<8} (no {fn})"); continue
        # Keep the best model per target (smallest test_rmse)
        best = (df.sort_values("test_rmse")
                  .groupby("target", as_index=False)
                  .head(1))
        print(f"\n  {which}:  (target / best-model / test MAE / test RMSE / n_test / cal? / tuned?)")
        for _, r in best.iterrows():
            tag = ""
            if "calibrated" in r and r.get("calibrated"):  tag += " cal"
            if "tuned" in r and r.get("tuned"):            tag += " tune"
            print(f"      {r['target']:<6}  {str(r.get('model','?')).upper():<3}  "
                  f"MAE={r.get('test_mae', float('nan')):.3f}  "
                  f"RMSE={r.get('test_rmse', float('nan')):.3f}  "
                  f"n={int(r.get('n_test', 0)):>4}{C_DIM}{tag}{C_END}")


# ─────────────────────────────────────────────────────────────────────────
def section_player_accuracy(since: str | None) -> None:
    df = _read("2026_player_accuracy.csv")
    if df is None or df.empty:
        print("\n(no 2026_player_accuracy.csv — skipping live projection MAE)")
        return
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if since:
        df = df[df["game_date"] >= pd.to_datetime(since)]
    if df.empty:
        print(f"\n(no rows in player accuracy since {since})"); return

    print(f"\n{'='*72}\nLIVE PROJECTION MAE  (graded vs MLB box scores"
          f"{' — since '+since if since else ''})\n{'='*72}")

    # Pitcher stats
    p = df[df["player_type"] == "pitcher"].copy()
    if not p.empty:
        print(f"\n  PITCHER stats  (n={len(p):,} pitcher-game rows)")
        for stat, proj_col, act_col, bench_key in [
            ("Strikeouts",   "proj_strikeouts",  "actual_strikeouts",  "strikeouts"),
            ("Walks",        "proj_walks",       "actual_walks",       "walks"),
            ("Hits Allowed", "proj_hits_allowed","actual_hits_allowed","hits_allowed"),
            ("IP",           "proj_ip",          "actual_ip",          "ip"),
            ("Runs Allowed", "proj_runs_allowed","actual_runs_allowed","runs_allowed"),
        ]:
            if proj_col not in p or act_col not in p:
                continue
            sub = p.dropna(subset=[proj_col, act_col])
            if sub.empty: continue
            err = (pd.to_numeric(sub[proj_col]) - pd.to_numeric(sub[act_col])).abs()
            mae = float(err.mean())
            tol = 1.5 if stat in ("Strikeouts", "Hits Allowed", "Runs Allowed") else 1.0
            within = float((err <= tol).mean())
            bench = BENCHMARK_MAE.get(bench_key)
            bench_str = f"  (book ≤ {bench[0]:.1f})" if bench else ""
            print(f"      {stat:<14}  MAE={_color(mae, bench)}  "
                  f"within ±{tol}: {within*100:5.1f}%  n={len(sub):>4}{bench_str}")

    # Hitter stats
    h = df[df["player_type"] == "hitter"].copy()
    if not h.empty:
        print(f"\n  HITTER stats  (n={len(h):,} hitter-game rows)")
        for stat, proj_col, act_col, bench_key in [
            ("Hits",   "proj_hits",       "actual_hits",       "hits"),
            ("HR",     "proj_hr",         "actual_hr",         "hr"),
            ("Runs",   "proj_runs",       "actual_runs",       "runs"),
            ("RBI",    "proj_rbi",        "actual_rbi",        "rbi"),
            ("Walks",  "proj_walks",      "actual_walks",      "walks"),
            ("K",      "proj_strikeouts", "actual_strikeouts", "strikeouts"),
            ("PA",     "proj_pa",         "actual_pa",         "pa"),
        ]:
            if proj_col not in h or act_col not in h:
                continue
            sub = h.dropna(subset=[proj_col, act_col])
            if sub.empty: continue
            err = (pd.to_numeric(sub[proj_col]) - pd.to_numeric(sub[act_col])).abs()
            mae = float(err.mean())
            tol = 1.0 if stat in ("Hits", "Runs", "RBI", "Walks", "PA") else 0.5
            within = float((err <= tol).mean())
            bench = BENCHMARK_MAE.get(bench_key)
            bench_str = f"  (book ≤ {bench[0]:.2f})" if bench else ""
            print(f"      {stat:<6}  MAE={_color(mae, bench)}  "
                  f"within ±{tol}: {within*100:5.1f}%  n={len(sub):>4}{bench_str}")


# ─────────────────────────────────────────────────────────────────────────
def _winrate_block(df: pd.DataFrame, label: str, *,
                   correct_col: str = "correct",
                   stake_col: str | None = None,
                   profit_col: str | None = None) -> None:
    n = len(df)
    if n == 0:
        print(f"  {label:<28}  (empty)"); return
    wins = int((df[correct_col] == True).sum()) if correct_col in df else 0
    losses = int((df[correct_col] == False).sum()) if correct_col in df else 0
    decided = wins + losses
    wr = wins / decided if decided else float("nan")
    extras = ""
    if stake_col and profit_col and stake_col in df and profit_col in df:
        s = pd.to_numeric(df[stake_col], errors="coerce").sum()
        p = pd.to_numeric(df[profit_col], errors="coerce").sum()
        roi = (p / s) if s else float("nan")
        extras = f"  staked=${s:,.0f}  P/L=${p:+,.0f}  ROI={roi*100:+.2f}%"
    print(f"  {label:<28}  n={n:>4}  W-L={wins}-{losses}  WR={wr*100:5.1f}%{extras}")


# ─────────────────────────────────────────────────────────────────────────
def section_picks(since: str | None) -> None:
    print(f"\n{'='*72}\nGAME / NRFI / PROP PICKS  (W-L, ROI where stake is tracked)\n{'='*72}")

    # Game moneyline
    df = _read("2026_picks_accuracy.csv")
    if df is not None and not df.empty:
        if "game_date" in df:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            if since: df = df[df["game_date"] >= pd.to_datetime(since)]
        print("\n  GAME MONEYLINE")
        _winrate_block(df, "All graded picks",
                       correct_col=("correct" if "correct" in df else "result"))
    else:
        print("\n  (no 2026_picks_accuracy.csv)")

    # NRFI
    df = _read("2026_nrfi_accuracy.csv")
    if df is not None and not df.empty:
        if "game_date" in df:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            if since: df = df[df["game_date"] >= pd.to_datetime(since)]
        print("\n  NRFI")
        cc = "correct" if "correct" in df else ("result" if "result" in df else None)
        if cc:
            _winrate_block(df, "All graded picks", correct_col=cc)

    # Projection-based props
    df = _read("2026_props_accuracy.csv")
    if df is not None and not df.empty:
        if "game_date" in df:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            if since: df = df[df["game_date"] >= pd.to_datetime(since)]
        print("\n  PROJECTION-BASED PROPS")
        cc = "correct" if "correct" in df else ("result" if "result" in df else None)
        if cc:
            _winrate_block(df, "All graded picks", correct_col=cc,
                           stake_col="stake", profit_col="profit")
            if "market" in df.columns:
                for m, sub in df.groupby("market"):
                    _winrate_block(sub, f"  by market: {m}", correct_col=cc,
                                   stake_col="stake", profit_col="profit")

    # CLV (projection-based)
    df = _read("2026_props_clv.csv")
    if df is not None and not df.empty and "clv_cents" in df.columns:
        clv = pd.to_numeric(df["clv_cents"], errors="coerce").dropna()
        if len(clv):
            pos = float((clv > 0).mean())
            print(f"\n  PROJECTION PROPS CLV   n={len(clv):>4}  mean={clv.mean():+.1f}c  "
                  f"median={clv.median():+.1f}c  positive={pos*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────
def section_market_consensus(since: str | None) -> None:
    df = _read("2026_market_consensus_graded.csv")
    if df is None or df.empty:
        print("\n(no 2026_market_consensus_graded.csv — run market_consensus_grade.py)")
        return
    if "game_date_et" in df:
        df["game_date_et"] = pd.to_datetime(df["game_date_et"], errors="coerce")
        if since: df = df[df["game_date_et"] >= pd.to_datetime(since)]

    decided = df[df["result"].isin(["win", "loss"])].copy()
    print(f"\n{'='*72}\nMARKET-CONSENSUS PICKS  (Video-1 approach)\n{'='*72}")
    if decided.empty:
        print("  (no decided picks yet)"); return

    n = len(decided)
    wins = int((decided["result"] == "win").sum())
    losses = int((decided["result"] == "loss").sum())
    wr = wins / n if n else float("nan")
    staked = float(pd.to_numeric(decided["staked"], errors="coerce").sum())
    profit = float(pd.to_numeric(decided["profit"], errors="coerce").sum())
    roi = (profit / staked) if staked else float("nan")
    print(f"\n  Overall   n={n:>4}  W-L={wins}-{losses}  WR={wr*100:5.1f}%  "
          f"staked=${staked:,.0f}  P/L=${profit:+,.0f}  ROI={roi*100:+.2f}%")

    print("\n  By kind:")
    for k, sub in decided.groupby("kind"):
        s = float(pd.to_numeric(sub["staked"], errors="coerce").sum())
        p = float(pd.to_numeric(sub["profit"], errors="coerce").sum())
        rk = (p / s) if s else float("nan")
        wrk = (sub["result"]=="win").mean()
        print(f"      {k:<14}  n={len(sub):>4}  WR={wrk*100:5.1f}%  "
              f"staked=${s:,.0f}  P/L=${p:+,.0f}  ROI={rk*100:+.2f}%")

    if "my_book" in decided:
        print("\n  By book:")
        for b, sub in decided.groupby("my_book"):
            s = float(pd.to_numeric(sub["staked"], errors="coerce").sum())
            p = float(pd.to_numeric(sub["profit"], errors="coerce").sum())
            rk = (p / s) if s else float("nan")
            wrk = (sub["result"]=="win").mean()
            print(f"      {b:<14}  n={len(sub):>4}  WR={wrk*100:5.1f}%  "
                  f"staked=${s:,.0f}  P/L=${p:+,.0f}  ROI={rk*100:+.2f}%")

    if "plus_ev_vs_all" in decided.columns:
        print("\n  Confidence split:")
        for flag, label in [(True, "plus_ev_vs_all=True "), (False, "plus_ev_vs_all=False")]:
            sub = decided[decided["plus_ev_vs_all"] == flag]
            if len(sub):
                s = float(pd.to_numeric(sub["staked"], errors="coerce").sum())
                p = float(pd.to_numeric(sub["profit"], errors="coerce").sum())
                rk = (p / s) if s else float("nan")
                wrk = (sub["result"]=="win").mean()
                print(f"      {label}  n={len(sub):>4}  WR={wrk*100:5.1f}%  "
                      f"staked=${s:,.0f}  P/L=${p:+,.0f}  ROI={rk*100:+.2f}%")


# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=None,
                    help="Only include picks/games on/after this date (YYYY-MM-DD).")
    args = ap.parse_args()

    section_training_holdout()
    section_player_accuracy(args.since)
    section_picks(args.since)
    section_market_consensus(args.since)
    print(f"\n{'='*72}\nDone.\n{'='*72}\n")


if __name__ == "__main__":
    main()
