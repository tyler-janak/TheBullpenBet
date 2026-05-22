"""
refresh_today.py
================
Quick manual fix for when the cron has failed and the live site is showing
stale projections. Just regenerates today's projection (skipping the
slow backfill) and applies bias calibration.

Use this when:
  - The Player Predictions page shows "STALE — projection dated <yesterday>"
  - You don't have time to wait for the next cron tick
  - The full daily_update.py is hitting a transient failure (API rate
    limit, etc.) that the targeted run won't trigger

What this does:
  1. Generates today's hitter/pitcher projections
     → writes outputs/hitterspitchers_today.csv (the live front-end alias)
     → writes outputs/hitterspitchers_<today>.csv (the dated snapshot)
  2. Applies the most recent bias calibration if calibration.json exists
  3. Optionally commits and pushes if --commit is passed (you must have
     gh / git credentials set up)

Usage
-----
    python refresh_today.py                         # regenerate + calibrate
    python refresh_today.py --commit                # also git commit + push
    python refresh_today.py --date 2026-04-30       # force a specific date

Skips entirely:
    - Game pipeline (game_picks_accuracy.csv won't update)
    - Past-date backfill (use backfill_player_predictions.py for that)
    - Grading (use the daily cron — needs box scores from completed games)
    - 2026 Statcast refresh (use refresh_2026_data.py for that)
"""

from __future__ import annotations

# Suppress sklearn warnings before any sklearn-using import.
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=None,
                   help="YYYY-MM-DD (default: today ET)")
    p.add_argument("--commit", action="store_true",
                   help="Also git add + commit + push the outputs after refreshing")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip applying bias calibration (writes raw model output)")
    args = p.parse_args()

    target = args.date or datetime.now(ET).strftime("%Y-%m-%d")
    print(f"\n========== Manual refresh for {target} ==========")

    # 1) Generate today's projection
    print("\n── Step 1/2: regenerate today's player projections ─────────")
    try:
        from hitterspitchers_today import run_projections
        run_projections(target, write_today_alias=True)
    except Exception as e:
        print(f"❌  Today's projection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2

    # 2) Apply calibration (best-effort)
    if not args.no_calibrate:
        print("\n── Step 2/2: apply bias calibration ────────────────────────")
        try:
            from calibrate_projections import (
                compute_calibration, save_calibration,
                calibrate_today_csv, _print_calibration,
            )
            cal = compute_calibration(min_n=30)
            _print_calibration(cal)
            save_calibration(cal)
            calibrate_today_csv(cal=cal)
        except Exception as e:
            print(f"⚠️  Calibration step failed (non-fatal): {type(e).__name__}: {e}")
    else:
        print("\n── Step 2/2: skipped calibration (--no-calibrate) ──────────")

    # 3) Optional commit + push
    if args.commit:
        print("\n── Committing fresh outputs to git ──────────────────────────")
        snap = Path("outputs") / f"hitterspitchers_{target}.csv"
        live = Path("outputs") / "hitterspitchers_today.csv"
        cal_path = Path("calibration.json")
        files = [str(p) for p in (live, snap, cal_path) if p.exists()]
        if not files:
            print("⚠️  No output files to commit.")
        else:
            run(["git", "add"] + files)
            commit_msg = f"manual refresh — {target}"
            run(["git", "commit", "-m", commit_msg])
            run(["git", "push"])

    print(f"\n✅ Manual refresh complete for {target}.")
    print("   Live alias:    outputs/hitterspitchers_today.csv")
    print(f"   Dated snapshot: outputs/hitterspitchers_{target}.csv")
    if not args.commit:
        print("\n   To deploy the fresh data, run:")
        print(f"     git add outputs/ calibration.json && "
              f"git commit -m 'manual refresh — {target}' && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
