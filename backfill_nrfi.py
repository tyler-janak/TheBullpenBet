"""
backfill_nrfi.py
================
Backfill NRFI predictions for past dates this season.

For each past date not already in 2026_nrfi_picks.csv, calls run_nrfi()
and appends the result rows to the picks log.  Idempotent — dates that
are already present in the picks log are skipped unless --force is given.

Usage
-----
    python backfill_nrfi.py --start 2026-03-25
    python backfill_nrfi.py --start 2026-03-25 --end 2026-04-30
    python backfill_nrfi.py --start 2026-03-25 --force
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


# ── helpers ───────────────────────────────────────────────────────────────────

def _yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def _date_range(start: str, end: str) -> list[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out: list[str] = []
    while s <= e:
        out.append(str(s))
        s += timedelta(days=1)
    return out


def _picks_dates(picks_file: str) -> set[str]:
    """Return the set of game_date strings already in the picks log."""
    p = Path(picks_file)
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p, usecols=["game_date"], low_memory=False)
        return set(df["game_date"].astype(str).str[:10].unique())
    except Exception:
        return set()


def append_nrfi_picks(results: pd.DataFrame, picks_file: str) -> None:
    """
    Append NRFI run_nrfi() results to the picks log CSV.
    Deduplicates by (game_date, team pair) — keep last so today's run
    overwrites a stale backfill row for the same game.
    """
    if results is None or results.empty:
        return

    to_append = results.copy()

    # Ensure grading-compatible columns exist
    if "team_a" not in to_append.columns:
        to_append["team_a"] = to_append.get("away_team", "")
    if "team_b" not in to_append.columns:
        to_append["team_b"] = to_append.get("home_team", "")
    if "lean" not in to_append.columns:
        to_append["lean"] = to_append["pick"].map(
            {"NRFI": "YES", "YRFI": "NO"}
        ).fillna("PASS")

    p = Path(picks_file)
    if p.exists():
        try:
            existing = pd.read_csv(p, low_memory=False)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, to_append], ignore_index=True)
    else:
        combined = to_append.copy()

    # Dedup key: date + sorted team pair (order-insensitive)
    def _key(r):
        d  = str(r.get("game_date", ""))[:10]
        ta = str(r.get("team_a", ""))
        tb = str(r.get("team_b", ""))
        return f"{d}|{min(ta, tb)}|{max(ta, tb)}"

    combined["_dedup_key"] = combined.apply(_key, axis=1)
    combined = (
        combined
        .drop_duplicates("_dedup_key", keep="last")
        .drop(columns=["_dedup_key"])
        .sort_values(["game_date", "team_a", "team_b"], na_position="last")
    )

    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(picks_file, index=False)


# ── main backfill ─────────────────────────────────────────────────────────────

def backfill_nrfi(
    start: str = "2026-03-25",
    end: str | None = None,
    picks_file: str = "2026_nrfi_picks.csv",
    force: bool = False,
    sleep_seconds: float = 0.5,
    verbose: bool = True,
) -> None:
    """
    Backfill NRFI predictions from `start` through `end` (default: yesterday ET).

    Parameters
    ----------
    start        : First date to consider.
    end          : Last date to consider (inclusive).  Default = yesterday ET.
    picks_file   : Path to the picks-log CSV to append/update.
    force        : If True, re-run all dates even if already in the log.
    sleep_seconds: Seconds to sleep between API calls (avoids rate limits).
    verbose      : Print progress.
    """
    from nrfi_today import run_nrfi  # import here to avoid circular deps

    end = end or _yesterday_et()
    all_dates   = _date_range(start, end)
    existing    = _picks_dates(picks_file) if not force else set()
    dates_to_run = [d for d in all_dates if d not in existing]

    if verbose:
        print(f"  NRFI backfill: {len(dates_to_run)} date(s) to process "
              f"(of {len(all_dates)} total, {len(existing)} already in log)")

    if not dates_to_run:
        if verbose:
            print("  Nothing to backfill.")
        return

    for i, d in enumerate(dates_to_run):
        try:
            results = run_nrfi(d)
            if results is not None and not results.empty:
                append_nrfi_picks(results, picks_file)
                if verbose:
                    print(f"  [{i+1}/{len(dates_to_run)}] {d}: {len(results)} game(s) appended")
            else:
                if verbose:
                    print(f"  [{i+1}/{len(dates_to_run)}] {d}: no games / no predictions")
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{len(dates_to_run)}] {d}: failed — {e}")

        if i < len(dates_to_run) - 1:
            time.sleep(sleep_seconds)

    if verbose:
        print("  NRFI backfill complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2026-03-25", help="First date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=None,         help="Last date (YYYY-MM-DD), default=yesterday")
    parser.add_argument("--picks-file", default="2026_nrfi_picks.csv")
    parser.add_argument("--force", action="store_true", help="Re-run all dates even if already logged")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    backfill_nrfi(
        start=args.start,
        end=args.end,
        picks_file=args.picks_file,
        force=args.force,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
