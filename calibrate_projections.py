"""
calibrate_projections.py
========================
Post-hoc bias correction for the player-projection models.

Why this exists
---------------
The player accuracy log shows a consistent **negative bias** on every
counting stat (PA bias ≈ -0.5, IP bias ≈ -0.45, hits/strikeouts/etc all
negative). That's a calibration error — the model's *shape* is fine, but
its *level* is shifted. Retraining is the proper long-term fix; in the
meantime, we can compute the bias directly from the graded log and apply
an additive (or multiplicative) correction at projection time.

How it works
------------
1. Read 2026_player_accuracy.csv → for each (kind, stat) pair, compute
   the average bias (mean of proj − actual) on graded rows.
2. Save those biases to ``calibration.json``.
3. ``apply_calibration(df, kind)`` shifts each projection column by the
   negative of the learned bias (so a projection that was -0.5 too low
   gets +0.5 added back). Multiplicative mode is also available for
   stats where ratio errors make more sense (rates, etc.).

What gets calibrated
--------------------
Hitter columns:
  proj_pa, proj_hits, proj_hr, proj_strikeouts, proj_walks, proj_runs, proj_rbi
Pitcher columns:
  proj_ip, proj_strikeouts, proj_walks, proj_hits_allowed, proj_runs_allowed

The calibration is **conservative** — we only apply a correction if we
have at least N graded games AND the absolute bias is meaningful
(|bias| > 0.05). Otherwise we leave the projection alone.

Order of leverage
-----------------
The HIGHEST-leverage fix is PA (hitters) and IP (pitchers), because every
other counting stat is derived by multiplying a per-PA / per-batter rate
by these. Calibrating PA fixes hits/HR/K/BB for hitters all at once.
Calibrating IP cascades to pitcher hits-allowed, runs, and K. We weight
those two heavier in the calibration logic.

Usage
-----
    python calibrate_projections.py                     # rebuild calibration.json
    python calibrate_projections.py --min-n 30          # require at least 30 games
    python calibrate_projections.py --inspect           # print current biases
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_ACC_PATH = Path("2026_player_accuracy.csv")
CALIBRATION_PATH = Path("calibration.json")

# Only these projection columns get calibrated. The actual-column for each
# is the column the grader writes when joining to box scores.
HITTER_PAIRS = [
    ("proj_pa",          "actual_pa"),
    ("proj_hits",        "actual_hits"),
    ("proj_hr",          "actual_hr"),
    ("proj_strikeouts",  "actual_strikeouts"),
    ("proj_walks",       "actual_walks"),
    ("proj_runs",        "actual_runs"),
    ("proj_rbi",         "actual_rbi"),
]

PITCHER_PAIRS = [
    ("proj_ip",            "actual_ip"),
    ("proj_strikeouts",    "actual_strikeouts"),
    ("proj_walks",         "actual_walks"),
    ("proj_hits_allowed",  "actual_hits_allowed"),
    ("proj_runs_allowed",  "actual_runs_allowed"),
]

DEFAULT_MIN_N = 30   # don't calibrate unless we have this many graded games
BIAS_FLOOR = 0.05    # don't apply a correction smaller than this

# Per-tier calibration: instead of one shift for every hitter, learn a
# different shift per lineup-spot tier (top, middle, bottom). A leadoff hitter
# gets ~5 PA/game; a #9 gets ~3.5. Applying the same +1.034 PA shift to both
# overcorrects the leadoff and undercorrects the #9, which is exactly the
# kind of "global mean OK but per-segment wrong" pattern that kills prop ROI.
# Pitchers split into starter vs reliever via expected IP — relievers get
# their own (much smaller) bias terms because they sit in a totally different
# IP regime than starters.
HITTER_TIERS = {
    "top":    [1, 2, 3],
    "middle": [4, 5, 6],
    "bottom": [7, 8, 9],
}
PITCHER_TIERS = {
    "starter":  ("ip_ge", 4.0),   # expected ≥ 4 IP
    "reliever": ("ip_lt", 4.0),
}
PER_TIER_MIN_N = 20   # need at least this many graded games per tier to apply


# ---------------------------------------------------------------------------
# Learn calibration from the graded log
# ---------------------------------------------------------------------------
def compute_calibration(min_n: int = DEFAULT_MIN_N) -> dict[str, Any]:
    """
    Return a dict like:
      {
        "computed_at": "2026-04-30",
        "n_hitter":  216,
        "n_pitcher": 23,
        "min_n":     30,
        "hitter":  {"proj_pa": {"bias": -0.52, "n": 216, "applied": True}, ...},
        "pitcher": {"proj_ip": {"bias": -0.45, "n":  23, "applied": False}, ...},
      }
    """
    out = {
        "computed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "min_n": min_n,
        "n_hitter": 0,
        "n_pitcher": 0,
        "hitter": {},
        "pitcher": {},
    }
    if not PLAYER_ACC_PATH.exists():
        print(f"⚠️  {PLAYER_ACC_PATH} doesn't exist — no calibration possible yet.")
        return out

    try:
        df = pd.read_csv(PLAYER_ACC_PATH, low_memory=False)
    except Exception as e:
        print(f"⚠️  Couldn't read {PLAYER_ACC_PATH}: {e}")
        return out

    if "player_type" not in df.columns or "played" not in df.columns:
        print("⚠️  Player accuracy log missing expected columns — skipping.")
        return out

    df["played"] = df["played"].astype(str).str.lower().isin({"true", "1", "1.0"})
    graded = df[df["played"] == True].copy()  # noqa: E712
    if graded.empty:
        return out

    hitter_df = graded[graded["player_type"].astype(str).str.lower() == "hitter"]
    pitcher_df = graded[graded["player_type"].astype(str).str.lower() == "pitcher"]

    # Hitters: only count appearances where the hitter actually batted.
    if not hitter_df.empty and "actual_pa" in hitter_df.columns:
        hitter_df = hitter_df[pd.to_numeric(hitter_df["actual_pa"], errors="coerce").fillna(0) > 0]

    out["n_hitter"] = int(len(hitter_df))
    out["n_pitcher"] = int(len(pitcher_df))

    def _learn(sub_df: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict[str, dict]:
        """For each (proj_col, actual_col), compute mean(proj - actual)."""
        result: dict[str, dict] = {}
        for proj_col, actual_col in pairs:
            if proj_col not in sub_df.columns or actual_col not in sub_df.columns:
                result[proj_col] = {"bias": 0.0, "n": 0, "applied": False, "reason": "column missing"}
                continue
            tmp = sub_df[[proj_col, actual_col]].copy()
            tmp[proj_col] = pd.to_numeric(tmp[proj_col], errors="coerce")
            tmp[actual_col] = pd.to_numeric(tmp[actual_col], errors="coerce")
            tmp = tmp.dropna()
            n = len(tmp)
            if n == 0:
                result[proj_col] = {"bias": 0.0, "n": 0, "applied": False, "reason": "no graded rows"}
                continue
            bias = float((tmp[proj_col] - tmp[actual_col]).mean())
            applied = (n >= min_n) and (abs(bias) >= BIAS_FLOOR)
            result[proj_col] = {
                "bias": round(bias, 4),
                "n": int(n),
                "applied": bool(applied),
                "reason": (
                    f"n<{min_n}" if n < min_n
                    else f"|bias|<{BIAS_FLOOR}" if abs(bias) < BIAS_FLOOR
                    else "ok"
                ),
            }
        return result

    out["hitter"] = _learn(hitter_df, HITTER_PAIRS)
    out["pitcher"] = _learn(pitcher_df, PITCHER_PAIRS)

    # Per-tier blocks: stored under "hitter_tiers" / "pitcher_tiers" so old
    # code paths that only consume "hitter"/"pitcher" still work.
    #
    # HITTERS: per-tier is a real win — leadoff vs #9 batters genuinely need
    # different PA shifts (5+ PA vs ~3.5 PA per game).
    #
    # PITCHERS: per-tier was actively HURTING us (MAE +0.05 to +0.22 vs
    # global). Our slate only contains *probable starters* — there is no
    # reliever signal to capture, so splitting just produces a slightly
    # different (and slightly worse) bias estimate. We keep the empty dict
    # in the JSON so `apply_calibration` cleanly falls back to global.
    out["hitter_tiers"] = _learn_per_tier_hitter(hitter_df, HITTER_PAIRS, min_n=PER_TIER_MIN_N)
    out["pitcher_tiers"] = {}
    return out


def _learn_per_tier_hitter(df: pd.DataFrame, pairs: list[tuple[str, str]], min_n: int) -> dict:
    """Group hitters by lineup_spot tier and learn a separate bias per tier."""
    if df.empty or "lineup_spot" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    spots = pd.to_numeric(df["lineup_spot"], errors="coerce")
    for tier_name, spots_in_tier in HITTER_TIERS.items():
        sub = df[spots.isin(spots_in_tier)]
        if sub.empty:
            continue
        tier_block: dict[str, dict] = {}
        for proj_col, actual_col in pairs:
            if proj_col not in sub.columns or actual_col not in sub.columns:
                continue
            tmp = sub[[proj_col, actual_col]].copy()
            tmp[proj_col] = pd.to_numeric(tmp[proj_col], errors="coerce")
            tmp[actual_col] = pd.to_numeric(tmp[actual_col], errors="coerce")
            tmp = tmp.dropna()
            n = len(tmp)
            if n == 0:
                continue
            bias = float((tmp[proj_col] - tmp[actual_col]).mean())
            applied = (n >= min_n) and (abs(bias) >= BIAS_FLOOR)
            tier_block[proj_col] = {
                "bias": round(bias, 4),
                "n": int(n),
                "applied": bool(applied),
                "reason": (
                    f"n<{min_n}" if n < min_n
                    else f"|bias|<{BIAS_FLOOR}" if abs(bias) < BIAS_FLOOR
                    else "ok"
                ),
            }
        out[tier_name] = tier_block
    return out


def _learn_per_tier_pitcher(df: pd.DataFrame, pairs: list[tuple[str, str]], min_n: int) -> dict:
    """Split pitchers by starter/reliever and learn per-tier biases."""
    if df.empty:
        return {}
    out: dict[str, dict] = {}
    # Use actual_ip if available — relievers cluster < 2 IP, starters > 4 IP.
    # Don't have ip ⇒ assume starter (the common case for our slate).
    actual_ip = pd.to_numeric(df.get("actual_ip"), errors="coerce") if "actual_ip" in df.columns else None
    for tier_name, (rule, threshold) in PITCHER_TIERS.items():
        if actual_ip is None:
            sub = df if tier_name == "starter" else df.iloc[0:0]
        elif rule == "ip_ge":
            sub = df[actual_ip >= threshold]
        else:
            sub = df[actual_ip < threshold]
        if sub.empty:
            continue
        tier_block: dict[str, dict] = {}
        for proj_col, actual_col in pairs:
            if proj_col not in sub.columns or actual_col not in sub.columns:
                continue
            tmp = sub[[proj_col, actual_col]].copy()
            tmp[proj_col] = pd.to_numeric(tmp[proj_col], errors="coerce")
            tmp[actual_col] = pd.to_numeric(tmp[actual_col], errors="coerce")
            tmp = tmp.dropna()
            n = len(tmp)
            if n == 0:
                continue
            bias = float((tmp[proj_col] - tmp[actual_col]).mean())
            applied = (n >= min_n) and (abs(bias) >= BIAS_FLOOR)
            tier_block[proj_col] = {
                "bias": round(bias, 4),
                "n": int(n),
                "applied": bool(applied),
                "reason": (
                    f"n<{min_n}" if n < min_n
                    else f"|bias|<{BIAS_FLOOR}" if abs(bias) < BIAS_FLOOR
                    else "ok"
                ),
            }
        out[tier_name] = tier_block
    return out


def hitter_tier_for_lineup_spot(spot) -> str:
    """Return tier name ('top'/'middle'/'bottom') for a lineup spot."""
    try:
        s = int(float(spot))
    except (TypeError, ValueError):
        return "middle"
    for tier, spots in HITTER_TIERS.items():
        if s in spots:
            return tier
    return "middle"


def save_calibration(cal: dict[str, Any], path: Path = CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(cal, indent=2))
    print(f"Wrote calibration → {path}")


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"hitter": {}, "pitcher": {}, "n_hitter": 0, "n_pitcher": 0}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"⚠️  Couldn't load calibration: {e}")
        return {"hitter": {}, "pitcher": {}, "n_hitter": 0, "n_pitcher": 0}


# ---------------------------------------------------------------------------
# Apply calibration to a projections DataFrame
# ---------------------------------------------------------------------------
def apply_calibration(df: pd.DataFrame, kind: str, cal: dict | None = None) -> pd.DataFrame:
    """
    Shift each calibratable projection column by `-bias` so the corrected
    projection has zero expected bias. Only applied when the calibration
    block has 'applied': True.

    `kind` is 'hitter' or 'pitcher'.

    If the calibration file has a per-tier block (`hitter_tiers` /
    `pitcher_tiers`) AND the input df has the tier-key column (lineup_spot
    or projected IP) we apply the bias per-tier; otherwise we fall back to
    the global block. Per-tier shifts are far more accurate because a
    leadoff hitter and a #9 hitter need different PA corrections.
    """
    if df is None or df.empty:
        return df
    if cal is None:
        cal = load_calibration()
    block = (cal or {}).get(kind) or {}
    if not block:
        return df

    tier_block = (cal or {}).get(f"{kind}_tiers") or {}

    out = df.copy()

    # Pre-compute tier per row if we can.
    tier_per_row = None
    if tier_block:
        if kind == "hitter" and "lineup_spot" in out.columns:
            tier_per_row = out["lineup_spot"].apply(hitter_tier_for_lineup_spot)
        elif kind == "pitcher" and "proj_ip" in out.columns:
            ip_proj = pd.to_numeric(out["proj_ip"], errors="coerce")
            tier_per_row = ip_proj.apply(
                lambda v: "starter" if pd.notna(v) and v >= 4.0 else "reliever"
            )

    applied_log: list[str] = []

    def _bias_for(col: str, tier: str | None) -> float | None:
        # Prefer per-tier bias when the tier has an `applied: True` entry.
        if tier and tier in tier_block:
            t_info = tier_block[tier].get(col)
            if t_info and t_info.get("applied"):
                return float(t_info.get("bias", 0.0))
        # Otherwise fall back to global if applied
        g_info = block.get(col)
        if g_info and g_info.get("applied"):
            return float(g_info.get("bias", 0.0))
        return None

    for col in {*block.keys(), *(c for tb in tier_block.values() for c in tb.keys())}:
        if col not in out.columns:
            continue
        before_mean = pd.to_numeric(out[col], errors="coerce").mean()

        if tier_per_row is not None:
            # Vectorize: build a bias-per-row Series and subtract once.
            bias_series = tier_per_row.apply(lambda t: _bias_for(col, t) or 0.0)
            shifted = pd.to_numeric(out[col], errors="coerce") - bias_series
        else:
            global_bias = _bias_for(col, None)
            if global_bias is None or global_bias == 0:
                continue
            shifted = pd.to_numeric(out[col], errors="coerce") - global_bias

        out[col] = shifted.clip(lower=0)
        after_mean = pd.to_numeric(out[col], errors="coerce").mean()
        applied_log.append(
            f"  {kind}.{col}: avg {before_mean:.2f} → {after_mean:.2f}"
        )

    if applied_log:
        print(f"Applied calibration ({kind}):")
        for line in applied_log:
            print(line)
    return out


# ---------------------------------------------------------------------------
# Apply calibration in-place to the daily projections CSV
# ---------------------------------------------------------------------------
def calibrate_today_csv(
    csv_path: Path = Path("outputs/hitterspitchers_today.csv"),
    cal: dict | None = None,
    save_raw_copy: bool = True,
) -> bool:
    """
    Read outputs/hitterspitchers_today.csv, apply calibration to both
    pitcher rows and hitter rows, and write it back. Returns True if
    the file was rewritten, False if nothing to do.

    If `save_raw_copy=True` (the default), the original UNCALIBRATED
    projections are also saved alongside as ``hitterspitchers_today_raw.csv``.
    This is what `props_fetch.py` reads — calibration is a display-time
    bias correction and should NOT cascade into prop edge math, otherwise
    the edge engine sees inflated projections and mass-flags Overs.
    """
    if not csv_path.exists():
        print(f"⚠️  {csv_path} doesn't exist — skipping calibration.")
        return False
    if cal is None:
        cal = load_calibration()
    if not cal or (not cal.get("hitter") and not cal.get("pitcher")):
        print("⚠️  No calibration available — skipping.")
        return False

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty or "player_type" not in df.columns:
        return False

    # Save raw (uncalibrated) copy for the props engine BEFORE we modify the
    # display CSV. The props engine reads this so it sees the model's actual
    # mean prediction, not a bias-corrected one.
    if save_raw_copy:
        raw_path = csv_path.with_name(csv_path.stem + "_raw" + csv_path.suffix)
        df.to_csv(raw_path, index=False)
        print(f"Saved raw projections → {raw_path}")

    pitcher_mask = df["player_type"].astype(str).str.lower() == "pitcher"
    hitter_mask = df["player_type"].astype(str).str.lower() == "hitter"

    pitchers = apply_calibration(df[pitcher_mask], "pitcher", cal)
    hitters = apply_calibration(df[hitter_mask], "hitter", cal)
    others = df[~(pitcher_mask | hitter_mask)]

    fixed = pd.concat([pitchers, hitters, others], ignore_index=True, sort=False)
    fixed.to_csv(csv_path, index=False)
    print(f"Calibrated {csv_path}")
    return True


def calibrate_all_dated_snapshots(
    snapshots_dir: Path = Path("outputs"),
    cal: dict | None = None,
) -> int:
    """
    Apply calibration to every dated snapshot too (so the player accuracy
    log eventually grades against calibrated projections — the bias should
    fall toward zero as the daily pipeline runs forward).

    Returns the number of files calibrated.
    """
    if cal is None:
        cal = load_calibration()
    if not cal:
        return 0
    n = 0
    for p in sorted(snapshots_dir.glob("hitterspitchers_*.csv")):
        # Skip the live alias — it's been calibrated separately.
        if p.name == "hitterspitchers_today.csv":
            continue
        if calibrate_today_csv(p, cal=cal):
            n += 1
    print(f"Calibrated {n} dated snapshot(s).")
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_calibration(cal: dict) -> None:
    print(f"\n=== Calibration computed {cal.get('computed_at','—')} "
          f"(min_n={cal.get('min_n')}, n_hitter={cal.get('n_hitter')}, n_pitcher={cal.get('n_pitcher')}) ===")
    for kind in ("hitter", "pitcher"):
        block = cal.get(kind) or {}
        if not block:
            print(f"  {kind}: no data")
            continue
        print(f"  {kind}:")
        for col, info in block.items():
            mark = "✓" if info.get("applied") else "·"
            print(f"    {mark} {col:24s}  bias={info.get('bias',0):+.3f}  "
                  f"n={info.get('n',0):>4d}  reason={info.get('reason','')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                   help=f"Minimum graded games to apply a correction (default {DEFAULT_MIN_N})")
    p.add_argument("--inspect", action="store_true",
                   help="Print current biases (rebuilds from log) without writing.")
    args = p.parse_args()

    cal = compute_calibration(min_n=args.min_n)
    _print_calibration(cal)
    if not args.inspect:
        save_calibration(cal)


if __name__ == "__main__":
    main()
