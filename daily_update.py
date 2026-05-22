"""
daily_update.py
===============
Single entry point for the daily MLB pipeline. Run this once per cron tick
and it will:

    0. Refresh 2026 Statcast pitch data via pybaseball and rebuild the four
       per-game feature CSVs in data/ (incremental — only pulls new dates).
    1. Backfill any missing game-pick rows for completed dates.
    2. Run today's game model (predictions + EV + bet log).
    3. Re-grade the season pick log against final scores.
    4. Backfill any missing NRFI predictions for past dates.
    5. Generate today's NRFI predictions and append to the picks log.
    6. Grade past NRFI picks against MLB first-inning linescores.
    7. Backfill any missing dated player-projection snapshots from past
       dates (uses MLB Stats API for actual lineups — no Rotowire scraping
       so this works for any past date).
    8. Generate today's hitter / pitcher projections (for the live site).
    9. Grade every player snapshot against MLB box scores → rebuild
       2026_player_accuracy.csv.
   10. Recompute bias calibration from the graded log and apply it to
       today's projection (post-hoc fix for the systematic PA / IP
       under-projection observed in the 2025-data-only era).

Steps 0, 4–6, and 7–9 are wrapped in their own try/except so a failure in
any sub-pipeline never blocks the others.  The data refresh is also
non-blocking — if pybaseball is unavailable or the network is flaky, the
pipeline continues with whatever data files already exist on disk.

Outputs touched (must be committed by the cron workflow):
    - data/hitter_game_data.csv
    - data/pitcher_game_data.csv
    - data/team_batting_hand_context.csv
    - data/team_pitching_hand_context.csv
    - 2026_picks_accuracy.csv
    - 2026_player_accuracy.csv
    - 2026_nrfi_picks.csv
    - 2026_nrfi_accuracy.csv
    - outputs/today_predictions_with_ev*.csv
    - outputs/today_bets_to_make*.csv
    - outputs/hitterspitchers_today.csv
    - outputs/hitterspitchers_<date>.csv  (one per past date)
    - outputs/nrfi_today.csv
"""

import os
# Suppress sklearn's `joblib.delayed → sklearn.utils.parallel.delayed`
# UserWarning that fires once per predict() call (joblib subprocess
# workers inherit PYTHONWARNINGS at startup). Must come before any
# sklearn import — see hitterspitchers_today.py for the long version.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from daily_mlb_model_runner import backfill_season, grade_saved_picks, run

# Always run "today" in Eastern Time. GitHub Actions runners are UTC, but
# we want our day to roll over on ET so we never grade tomorrow before
# today's late West Coast games have actually finished.
ET = ZoneInfo("America/New_York")

SEASON_START     = "2026-03-25"
PICKS_FILE       = "2026_picks_accuracy.csv"
PLAYER_ACC_FILE  = "2026_player_accuracy.csv"
NRFI_PICKS_FILE  = "2026_nrfi_picks.csv"
NRFI_ACC_FILE    = "2026_nrfi_accuracy.csv"
MODELS_DIR       = Path("models")


def _load_preset_params(prefix: str, targets: list[str]) -> dict:
    """Read previously-tuned XGB hyperparameters out of existing model pickles.

    Lets the daily refit reuse the expensive hyperparameter search from an
    earlier offline `hitterspitchers_train.py` run instead of re-searching on
    every cron tick — we refit on fresh data + recalibrate, but keep the tuned
    depth / n_estimators / regularisation. Returns {target: params}.
    """
    out: dict[str, dict] = {}
    for t in targets:
        p = MODELS_DIR / f"{prefix}_{t}.pkl"
        if not p.exists():
            continue
        try:
            with open(p, "rb") as fh:
                bundle = pickle.load(fh)
            bp = bundle.get("best_params") if isinstance(bundle, dict) else None
            if bp:
                out[t] = bp
        except Exception:
            continue
    return out


def _retrain_player_models(tune: bool) -> None:
    """Retrain the player model stack on the freshly-refreshed feature tables.

    Calibration is always on (it's cheap and removes systematic bias). Tuning
    reuses persisted hyperparameters when available; a full randomized search
    only runs when `tune=True` AND no preset exists for a target. The two-stage
    pitcher, xHits, and team-PA hitter models are refit on all rows so today's
    projection AND the past-date backfill score off current-data models.
    """
    import pandas as pd
    import hitterspitchers_train as hpt

    MODELS_DIR.mkdir(exist_ok=True)
    pitcher_df = pd.read_csv("data/pitcher_game_data.csv", low_memory=False)
    hitter_df  = pd.read_csv("data/hitter_game_data.csv",  low_memory=False)

    hpt.validate_pitcher_training_data(pitcher_df)

    pitcher_presets = _load_preset_params("pitcher", hpt.PITCHER_TARGETS)
    hitter_presets  = _load_preset_params("hitter",  hpt.HITTER_TARGETS)

    hpt.train_pitcher_models(pitcher_df, MODELS_DIR, tune=tune, calibrate=True,
                             preset_params=pitcher_presets)
    hpt.train_hitter_models(hitter_df, MODELS_DIR, tune=tune, calibrate=True,
                            preset_params=hitter_presets)

    # Advanced decomposition stack (two-stage per-9 / xHits / team-PA). These
    # are NOT used by run_projections anymore — the direct counting-stat models
    # above score better overall, so USE_DECOMPOSITION_MODELS is False in
    # hitterspitchers_today.py. We skip retraining them by default to save cron
    # time; set BULLPEN_TRAIN_DECOMP=1 to keep them fresh (e.g. if you flip the
    # projection flag back on).
    if os.environ.get("BULLPEN_TRAIN_DECOMP", "0") == "1":
        try:
            from train_pitcher_two_stage import train_two_stage
            train_two_stage(eval_holdout=0.0)
        except Exception as e:
            print(f"⚠️  two-stage pitcher retrain failed: {e}")
        try:
            from train_pitcher_xhits import train_xhits
            train_xhits(eval_holdout=0.0)
        except Exception as e:
            print(f"⚠️  xHits retrain failed: {e}")
        try:
            from train_hitter_team_pa import train_team_pa, train_rate_models
            train_team_pa(eval_holdout=0.0)
            train_rate_models(eval_holdout=0.0)
        except Exception as e:
            print(f"⚠️  team-PA retrain failed: {e}")


def main():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"\n========== Daily update for {today} (ET) ==========\n")

    # ── DATA REFRESH (Statcast → per-game features) ─────────────────────
    # 0) Pull any new 2026 pitch-level Statcast data via pybaseball and
    #    rebuild data/hitter_game_data.csv + data/pitcher_game_data.csv +
    #    the team hand-context CSVs. Incremental — only new dates are
    #    pulled, the full feature build runs on the combined cache.
    try:
        from refresh_2026_data import refresh as refresh_2026_data
        print("── Refreshing 2026 player history from Statcast ────────────")
        refresh_2026_data(
            start=None,         # resume from cache; first run uses 2026-03-01
            end=None,           # default = today ET
            rebuild=False,
            skip_features=False,
        )
    except Exception as e:
        # Non-blocking — projections will fall back to whatever's already in
        # data/*.csv if the Statcast pull fails (network / pybaseball issue).
        print(f"⚠️  2026 data refresh failed: {e}")

    # 0a) Post-process hitter_game_data.csv with team-level rolling features
    # (team_obp_*, team_pa_avg_*, opp_sp_ip_avg alias). Idempotent — re-runs
    # drop the prior enrichment columns before recomputing.
    try:
        from enrich_team_features import enrich as _enrich_team_features
        _enrich_team_features()
    except Exception as e:
        print(f"⚠️  team-feature enrichment failed: {e}")

    # 0b) Add batter-level lineup aggregations to pitcher_game_data.csv
    # (lineup_k_rate / lineup_bb_rate / lineup_avg_ev / …). The two-stage
    # pitcher model trains on these, so they must be present before retrain.
    try:
        from enrich_lineup_features import enrich as _enrich_lineup_features
        _enrich_lineup_features()
    except Exception as e:
        print(f"⚠️  lineup-feature enrichment failed: {e}")

    # 0c) RETRAIN the player models on the freshly-refreshed data so that both
    # today's projection (step 8) and the past-date backfill (step 7) score
    # off current-data, tuned + calibrated models. By default this runs only on
    # the early-morning (3 AM ET) cron tick so the midday/evening lineup-update
    # ticks stay fast; override with BULLPEN_RETRAIN=force / skip. A full
    # hyperparameter search runs only when BULLPEN_TUNE=1 (otherwise the refit
    # reuses the hyperparameters tuned in the last offline run).
    retrain_mode = os.environ.get("BULLPEN_RETRAIN", "auto").lower()
    do_retrain = (
        retrain_mode == "force"
        or (retrain_mode == "auto" and datetime.now(ET).hour < 9)
    )
    if retrain_mode == "skip":
        do_retrain = False
    if do_retrain:
        tune = os.environ.get("BULLPEN_TUNE", "0") == "1"
        try:
            print(f"\n── Retraining player models (tune={'ON' if tune else 'reuse-preset'}, "
                  f"calibrate=ON) ───")
            _retrain_player_models(tune=tune)
        except Exception as e:
            # Non-blocking — if retrain fails we fall back to the committed
            # pickles and projections still run.
            print(f"⚠️  player model retrain failed (using existing models): {e}")
    else:
        print("   (skipping model retrain this tick — set BULLPEN_RETRAIN=force to override)")

    # ── GAME PIPELINE ────────────────────────────────────────────────────
    # 1) Backfill any missing completed dates through yesterday.
    backfill_season(
        season_start=SEASON_START,
        model_path="betting_model.pkl",
        history_path="2025_model_data.csv",
        picks_file=PICKS_FILE,
        sleep_seconds=0.3,
    )

    # 2) Run today's slate and save today's picks / outputs.
    run(
        date=today,
        odds_api_key="afa28350c34fba9f318ecd7ae4e21b63",
        model_path="betting_model.pkl",
        history_path="2025_model_data.csv",
        min_ev=0.02,
        save_today_csv=True,
        save_pick_log=True,
        picks_file=PICKS_FILE,
    )

    # 3) Re-grade the whole pick log so the season accuracy file always has
    #    fresh actual_winner / correct values for completed games.
    grade_saved_picks(
        picks_file=PICKS_FILE,
        output_file=PICKS_FILE,
    )

    # ── NRFI PIPELINE ────────────────────────────────────────────────────
    # 4) Backfill any past dates not yet in the NRFI picks log.  run_nrfi()
    #    already filters features to < target_date, so this is safe for any
    #    historical date once nrfi_game_data.csv and pitcher_game_data.csv
    #    have been built.
    try:
        from backfill_nrfi import backfill_nrfi
        print("\n── Backfilling past NRFI predictions ───────────────────────")
        backfill_nrfi(
            start=SEASON_START,
            end=None,           # auto = yesterday ET
            picks_file=NRFI_PICKS_FILE,
            force=False,        # skip dates already in the log
            sleep_seconds=0.4,
            verbose=True,
        )
    except Exception as e:
        print(f"⚠️  NRFI backfill failed: {e}")

    # 5) Generate TODAY's NRFI predictions and append to the picks log.
    try:
        from nrfi_today import run_nrfi
        from backfill_nrfi import append_nrfi_picks
        print("\n── Running today's NRFI predictions ────────────────────────")
        nrfi_results = run_nrfi(today)
        if nrfi_results is not None and not nrfi_results.empty:
            append_nrfi_picks(nrfi_results, NRFI_PICKS_FILE)
            print(f"  Appended {len(nrfi_results)} NRFI row(s) to {NRFI_PICKS_FILE}")
    except Exception as e:
        print(f"⚠️  Today's NRFI predictions failed: {e}")

    # 6) Grade every NRFI pick in the log against the MLB Stats API linescore
    #    and rebuild 2026_nrfi_accuracy.csv.
    try:
        from nrfi_grade import grade_nrfi_picks
        print("\n── Grading NRFI predictions vs MLB linescores ──────────────")
        grade_nrfi_picks(
            picks_file=NRFI_PICKS_FILE,
            output_file=NRFI_ACC_FILE,
        )
    except Exception as e:
        print(f"⚠️  NRFI grading failed: {e}")

    # ── PLAYER PIPELINE ──────────────────────────────────────────────────
    # 7) Backfill any missing past-date snapshots FIRST. The backfill loop
    #    calls run_projections(date, write_today_alias=False) for each past
    #    date, which only writes outputs/hitterspitchers_<date>.csv (no
    #    overwriting of the live "today" alias). Idempotent — dates that
    #    already have a populated snapshot are skipped.
    try:
        from backfill_player_predictions import backfill as backfill_player_predictions
        print("\n── Backfilling past-date player snapshots ──────────────────")
        backfill_player_predictions(
            start=SEASON_START,
            end=None,           # auto = yesterday ET
            force=False,        # skip dates that already have a populated snapshot
            grade=False,        # we'll grade once at the end after today's run too
            verbose=False,
        )
    except Exception as e:
        # Backfill is non-blocking.
        print(f"⚠️  Player backfill failed: {e}")

    # 8) Generate TODAY's hitter / pitcher projections last so the live
    #    "today" alias (outputs/hitterspitchers_today.csv) reflects the
    #    current slate — not whichever past date the backfill ended on.
    try:
        from hitterspitchers_today import run_projections
        print("\n── Running today's player projections ──────────────────────")
        run_projections(today)
    except Exception as e:
        print(f"⚠️  Today's player projections failed: {e}")

    # 9) Grade every snapshot (past + today) against MLB box scores and
    #    rebuild 2026_player_accuracy.csv.
    try:
        from grade_player_predictions import grade_player_predictions
        print("\n── Grading player projections vs MLB box scores ────────────")
        grade_player_predictions(
            snapshots_dir="outputs",
            output_file=PLAYER_ACC_FILE,
            season_start=SEASON_START,
        )
    except Exception as e:
        # Grading is non-blocking — game accuracy must still update even
        # if box-score endpoints are slow or rate-limited.
        print(f"⚠️  Player grading failed: {e}")

    # 10) Rebuild calibration from the freshly-graded log, then apply it
    #    to the live "today" projection so users see bias-corrected
    #    numbers. The corrections are conservative — they only fire if
    #    we have ≥30 graded games for that stat AND |bias| ≥ 0.05.
    try:
        from calibrate_projections import (
            compute_calibration, save_calibration,
            calibrate_today_csv, _print_calibration,
        )
        print("\n── Rebuilding bias calibration from accuracy log ───────────")
        cal = compute_calibration(min_n=30)
        _print_calibration(cal)
        save_calibration(cal)
        print("\n── Applying calibration to today's projection ──────────────")
        calibrate_today_csv(cal=cal)
    except Exception as e:
        print(f"⚠️  Calibration step failed: {e}")

    # ── PROPS / EDGE ENGINE ─────────────────────────────────────────────
    # 11) Pull today's player-prop lines from The Odds API, join to the
    #     freshly-calibrated projections, compute edge/EV per side, and
    #     write outputs/today_props_with_ev.csv. Also appends each prop
    #     to 2026_props_log.csv with stage='open' so the close-line
    #     snapshot at 5pm and the post-game grader can find them.
    try:
        from props_fetch import fetch_today_props, compute_edge_today, log_clv_close
        print("\n── Fetching today's player props from The Odds API ─────────")
        props_df = fetch_today_props()
        edged = None
        if props_df is not None and not props_df.empty:
            edged = compute_edge_today()
            # On the 5pm cron tick (lineups locked, ~first pitch), also stamp
            # the props as "close" so CLV gets both bookends.
            try:
                hour_et = datetime.now(ET).hour
                if hour_et >= 17 and edged is not None and not edged.empty:
                    log_clv_close(edged)
                    print("   logged close lines for CLV tracking")
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️  Props fetch / edge step failed: {e}")

    # 12) Grade past Value-flagged props against actual results from the
    #     player accuracy log. Builds 2026_props_accuracy.csv +
    #     2026_props_clv.csv summary.
    try:
        from props_grade import grade_props, compute_clv
        print("\n── Grading past prop picks vs actuals + computing CLV ──────")
        grade_props()
        compute_clv()
    except Exception as e:
        print(f"⚠️  Props grading step failed: {e}")

    print("\n✅ Daily update complete")


if __name__ == "__main__":
    main()
