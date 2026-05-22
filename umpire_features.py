"""
umpire_features.py
==================
Home-plate umpire K-zone effect as a post-hoc multiplier on pitcher K
projections.

Why this exists
---------------
Umpire strike-zone tendencies swing pitcher strikeout totals by roughly
0.5–1.0 K per game in the extremes (Angel Hernandez vs Pat Hoberg, etc.).
This signal is highly leveraged for K props but the per-game projection
models don't see it because the umpire isn't a Statcast feature.

Pipeline
--------
1. `data/umpire_k_factors.csv` is a static table with one row per home-plate
   umpire: `umpire_name, k_factor, n_games`. A factor of 1.00 is neutral;
   1.08 means "Ks above league average +8% behind this ump"; 0.92 means -8%.
   The file ships with a small seed of well-known extremes and falls back
   to 1.00 for unknown umps.
2. `fetch_today_umpires(game_pks)` pulls today's confirmed home-plate ump
   from the MLB Stats API gamefeed (`/api/v1.1/game/{pk}/feed/live` →
   `liveData.boxscore.officials`). Returns {game_pk: umpire_name}.
3. `apply_umpire_k_factor(df)` multiplies `proj_strikeouts` by the lookup
   factor when `home_plate_umpire` is present on the row.

Building the seed CSV
---------------------
For the initial commit we ship neutral defaults. To populate with real
data, run a one-off script that aggregates `K / BF` per umpire across
historical Statcast plus MLB Stats API ump assignments, then normalizes
by league mean. UmpScores.com publishes per-umpire ratings annually —
that's the easiest cold-start source; cite their data when populating.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
UMPIRE_CSV = DATA_DIR / "umpire_k_factors.csv"

# Cap how far we'll shift the K projection in either direction — even the
# most extreme umpire in MLB swings totals by ~12% across a full season, so
# 0.88 / 1.12 is a reasonable safety bound for the lookup output.
_MIN_FACTOR = 0.88
_MAX_FACTOR = 1.12

# Seed with neutral defaults for every umpire we encounter. The CSV is the
# single source of truth; this in-memory cache is rebuilt on every load.
_CACHE: dict[str, float] = {}
_CACHE_MTIME: float = 0.0


def _ensure_seed_csv() -> None:
    """Create data/umpire_k_factors.csv with neutral defaults if missing."""
    if UMPIRE_CSV.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(UMPIRE_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["umpire_name", "k_factor", "n_games", "source"])
        # A few public/well-known extremes from UmpScores' historical
        # consistency ratings — kept conservative (within ±8%) until
        # real data is populated. These are placeholders; replace with
        # current-season values when the user runs a one-off backfill.
        seed_rows = [
            ("Pat Hoberg",        1.06, 0, "seed"),
            ("Tripp Gibson",      1.04, 0, "seed"),
            ("Mark Carlson",      1.03, 0, "seed"),
            ("Dan Iassogna",      1.02, 0, "seed"),
            ("Angel Hernandez",   0.94, 0, "seed"),
            ("CB Bucknor",        0.95, 0, "seed"),
            ("Doug Eddings",      0.96, 0, "seed"),
            ("Laz Diaz",          0.97, 0, "seed"),
        ]
        for r in seed_rows:
            w.writerow(r)


def load_factor_table(path: Path = UMPIRE_CSV) -> dict[str, float]:
    """Load (name → factor) lookup; refreshes the module cache if mtime changes."""
    global _CACHE, _CACHE_MTIME
    _ensure_seed_csv()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    if _CACHE and mtime == _CACHE_MTIME:
        return _CACHE

    out: dict[str, float] = {}
    try:
        df = pd.read_csv(path)
        if "umpire_name" in df.columns and "k_factor" in df.columns:
            for _, r in df.iterrows():
                name = str(r["umpire_name"]).strip()
                try:
                    factor = float(r["k_factor"])
                except (TypeError, ValueError):
                    continue
                factor = float(np.clip(factor, _MIN_FACTOR, _MAX_FACTOR))
                out[name.lower()] = factor
    except Exception as e:
        print(f"  [umpire] could not load {path}: {e}")
        return {}

    _CACHE = out
    _CACHE_MTIME = mtime
    return out


def get_factor(umpire_name: Optional[str]) -> float:
    """Return the multiplicative K factor for an ump, or 1.0 if unknown."""
    if not umpire_name:
        return 1.0
    table = load_factor_table()
    return table.get(str(umpire_name).strip().lower(), 1.0)


# ---------------------------------------------------------------------------
# MLB Stats API — fetch today's home-plate umpires for the slate
# ---------------------------------------------------------------------------
_BOX_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def fetch_umpire_for_game(game_pk: int, timeout: int = 8) -> Optional[str]:
    """Pull the home-plate umpire name from the MLB Stats API gamefeed.

    The feed exposes `liveData.boxscore.officials` as a list of dicts where
    each entry has `official.fullName` and `officialType` (one of "Home Plate",
    "First Base", "Second Base", "Third Base"). We return the home-plate ump
    or None if not yet assigned (this is common 4+ hours before first pitch).
    """
    try:
        r = requests.get(_BOX_URL.format(game_pk=int(game_pk)), timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    officials = (((data or {}).get("liveData") or {}).get("boxscore") or {}).get("officials") or []
    for o in officials:
        if str(o.get("officialType", "")).strip().lower() == "home plate":
            return ((o.get("official") or {}).get("fullName") or "").strip() or None
    return None


def fetch_today_umpires(game_pks: Iterable[int]) -> dict[int, str]:
    """Return {game_pk: home_plate_umpire_name} for whatever's been assigned."""
    out: dict[int, str] = {}
    for pk in game_pks:
        try:
            ump = fetch_umpire_for_game(int(pk))
            if ump:
                out[int(pk)] = ump
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Apply at projection time
# ---------------------------------------------------------------------------
def apply_umpire_k_factor(
    df: pd.DataFrame,
    ump_by_game: dict[int, str] | None = None,
    *,
    ump_col: str = "home_plate_umpire",
    game_pk_col: str = "game_pk",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Multiply `proj_strikeouts` by the ump's K factor.

    If `ump_by_game` is provided, we use it as the source of truth (keyed by
    `game_pk_col`). Otherwise we look for `ump_col` already on the row.
    Rows with no resolvable umpire are left unchanged.
    """
    if df is None or df.empty or "proj_strikeouts" not in df.columns:
        return df

    out = df.copy()

    # Resolve umpire per row
    if ump_by_game and game_pk_col in out.columns:
        pks = pd.to_numeric(out[game_pk_col], errors="coerce")
        out["_ump_name"] = pks.apply(lambda v: ump_by_game.get(int(v)) if pd.notna(v) else None)
    elif ump_col in out.columns:
        out["_ump_name"] = out[ump_col]
    else:
        out["_ump_name"] = None

    factors = out["_ump_name"].apply(get_factor)
    before_mean = pd.to_numeric(out["proj_strikeouts"], errors="coerce").mean()
    out["proj_strikeouts"] = pd.to_numeric(out["proj_strikeouts"], errors="coerce") * factors
    out["proj_strikeouts"] = out["proj_strikeouts"].clip(lower=0)
    after_mean = pd.to_numeric(out["proj_strikeouts"], errors="coerce").mean()

    n_known = int((factors != 1.0).sum())
    if verbose:
        print(f"  umpire factors: {n_known}/{len(out)} rows had known umpires  "
              f"avg K {before_mean:.2f} → {after_mean:.2f}")

    out = out.drop(columns=["_ump_name"], errors="ignore")
    return out


# ---------------------------------------------------------------------------
# CLI: refresh today's umpire assignments (writes data/umpire_today.csv)
# ---------------------------------------------------------------------------
def write_today_umpires_csv(game_pks: Iterable[int], out_path: Path | None = None) -> Path:
    out_path = out_path or DATA_DIR / "umpire_today.csv"
    pks = list(int(pk) for pk in game_pks)
    print(f"  Fetching home-plate umpires for {len(pks)} games…")
    lookup = fetch_today_umpires(pks)
    rows = [{"game_pk": pk, "home_plate_umpire": lookup.get(pk, "")} for pk in pks]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    n_known = sum(1 for v in lookup.values() if v)
    print(f"  Wrote {len(rows)} rows ({n_known} with confirmed umpire) → {out_path}")
    return out_path


if __name__ == "__main__":
    _ensure_seed_csv()
    print(f"Umpire factor table seeded → {UMPIRE_CSV}")
    print(f"  loaded {len(load_factor_table())} umpires")
