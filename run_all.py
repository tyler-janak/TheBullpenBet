"""
run_all.py
==========
Master orchestration script — train all models and/or generate all daily predictions.

Steps (in order)
----------------
TRAINING phase  (--train-only or default):
  1. Build hitter/pitcher game data     python hitterspitchers_data.py
  2. Build NRFI game data               python nrfi_data.py --input <pitch_csv>
  3. Train hitter/pitcher models        python hitterspitchers_train.py
  4. Train NRFI model                   python nrfi_train.py

PREDICTION phase  (--predict-only or default):
  5. Run daily moneyline model          python daily_update.py
  6. Fetch FanDuel props                python fanduel_props.py --api-key <key>
  7. Generate player projections        python hitterspitchers_today.py
  8. Generate NRFI predictions          python nrfi_today.py

Usage
-----
    # Run everything (train + predict):
    python run_all.py

    # Train only (rebuild models from historical data):
    python run_all.py --train-only

    # Predict only (assumes models exist, just generate today's outputs):
    python run_all.py --predict-only

    # Pass a specific pitch data CSV for training:
    python run_all.py --pitch-csv data/pitch_data_2025.csv

    # Pass FanDuel / Odds API key:
    python run_all.py --odds-api-key YOUR_KEY

    # Skip a step by name:
    python run_all.py --skip moneyline --skip fanduel

    # Predict for a specific date:
    python run_all.py --predict-only --date 2026-05-03
"""

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
PYTHON = sys.executable   # same interpreter that invoked this script


def _run(label: str, cmd: list[str], skip_set: set[str]) -> bool:
    """
    Run a subprocess command.  Returns True on success, False on failure.
    Skips gracefully if *label* is in skip_set.
    """
    if label in skip_set:
        print(f"\n[SKIP] {label}")
        return True

    print(f"\n{'='*60}")
    print(f"[RUN ] {label}")
    print(f"       {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"[OK  ] {label}  ({elapsed:.1f}s)")
        return True
    else:
        print(f"[FAIL] {label} exited with code {result.returncode}  ({elapsed:.1f}s)")
        return False


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def run_training(pitch_csv: str, skip: set[str]) -> bool:
    ok = True

    # Step 1 — build hitter/pitcher game data
    ok &= _run(
        "hitterspitchers_data",
        [PYTHON, "hitterspitchers_data.py"],
        skip,
    )

    # Step 2 — build NRFI game data (requires raw pitch CSV)
    nrfi_data_cmd = [PYTHON, "nrfi_data.py", "--input", pitch_csv]
    ok &= _run("nrfi_data", nrfi_data_cmd, skip)

    # Step 3 — train hitter/pitcher models
    ok &= _run(
        "hitterspitchers_train",
        [PYTHON, "hitterspitchers_train.py"],
        skip,
    )

    # Step 4 — train NRFI model
    ok &= _run("nrfi_train", [PYTHON, "nrfi_train.py"], skip)

    return ok


def run_predictions(target_date: str, odds_api_key: str | None, skip: set[str]) -> bool:
    ok = True

    # Step 5 — moneyline model (daily_update.py)
    ok &= _run("moneyline", [PYTHON, "daily_update.py"], skip)

    # Step 6 — FanDuel props
    if odds_api_key:
        fanduel_cmd = [
            PYTHON, "fanduel_props.py",
            "--api-key", odds_api_key,
            "--date",    target_date,
        ]
        ok &= _run("fanduel", fanduel_cmd, skip)
    else:
        print("\n[SKIP] fanduel  (no --odds-api-key provided; pass one to fetch live props)")

    # Step 7 — player projections
    ok &= _run(
        "hitterspitchers_today",
        [PYTHON, "hitterspitchers_today.py", "--date", target_date],
        skip,
    )

    # Step 8 — NRFI predictions
    ok &= _run(
        "nrfi_today",
        [PYTHON, "nrfi_today.py", "--date", target_date],
        skip,
    )

    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all MLB Edge model training and/or daily predictions."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="Only run the training phase (steps 1-4). Do not generate today's predictions.",
    )
    mode.add_argument(
        "--predict-only",
        action="store_true",
        help="Only run the prediction phase (steps 5-8). Skip training.",
    )
    parser.add_argument(
        "--pitch-csv",
        default="data/pitch_data.csv",
        help="Path to the raw Statcast pitch-level CSV used by nrfi_data.py "
             "(default: data/pitch_data.csv)",
    )
    parser.add_argument(
        "--odds-api-key",
        default=None,
        help="The Odds API key for fetching FanDuel props. "
             "If omitted the fanduel step is skipped.",
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Target date for predictions (YYYY-MM-DD, default=today).",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="STEP",
        help=(
            "Skip a named step. Can be repeated. "
            "Valid names: hitterspitchers_data, nrfi_data, hitterspitchers_train, "
            "nrfi_train, moneyline, fanduel, hitterspitchers_today, nrfi_today"
        ),
    )
    args = parser.parse_args()

    skip_set = set(args.skip)
    do_train   = not args.predict_only
    do_predict = not args.train_only

    print(f"\nMLB Edge — run_all.py")
    print(f"  Date       : {args.date}")
    print(f"  Train phase: {'yes' if do_train   else 'no'}")
    print(f"  Pred phase : {'yes' if do_predict else 'no'}")
    if skip_set:
        print(f"  Skipping   : {', '.join(sorted(skip_set))}")

    results = []

    if do_train:
        ok = run_training(pitch_csv=args.pitch_csv, skip=skip_set)
        results.append(("Training", ok))

    if do_predict:
        ok = run_predictions(
            target_date=args.date,
            odds_api_key=args.odds_api_key,
            skip=skip_set,
        )
        results.append(("Predictions", ok))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for phase, ok in results:
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"  {phase:<14} {status}")
        all_ok &= ok

    if all_ok:
        print("\n✅  All steps completed successfully.")
        sys.exit(0)
    else:
        print("\n❌  One or more steps failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
