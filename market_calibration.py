"""
market_calibration.py
=====================
Sportsbook market as a calibration layer for our model projections.

Why this exists
---------------
Sportsbooks bake in workload limits, injuries, bullpen plans, weather, sharp
action, and other context our standalone baseball model can't see. Treating
their lines as a Bayesian prior — and blending our raw model output toward
the market — gives us a *market-aware* final projection without losing the
model's edge on stats markets misprice.

This is NOT replacing the model with the market (that gives you zero edge).
It's:

    final_projection = (1 - β) · model_proj + β · market_implied_proj

where β is small (typically 0.10–0.25) — enough to absorb context the model
misses without erasing the model's information advantage.

Pipeline
--------
1. Fetch today's sportsbook lines for each prop market (we already do this
   for the props edge engine in `props_fetch.py`).
2. For each line/odds pair, invert the same distribution math used in
   `props_engine.py` to recover the **market-implied projected mean**
   (the `μ` such that P(over_line | μ) equals the no-vig probability).
3. Join market-implied μ to our model projection by (player, market).
4. Compute blended μ — heavier blend when:
       * the market line is from multiple books that agree (reliable)
       * the model–market gap is small (no info advantage)
       * the model's confidence on that player is low
5. Return augmented projection df with both `proj_<x>_raw` and
   `proj_<x>` (blended).

This is a **calibration layer**, not a replacement. The Edge tab is gone
but we still consume the same props_log data to extract market priors.

Usage
-----
    from market_calibration import (
        load_market_priors,
        apply_market_calibration,
    )

    priors = load_market_priors()                # reads outputs/today_props_raw.csv
    proj   = apply_market_calibration(proj, priors, beta=0.20)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom, norm
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
TODAY_PROPS_RAW = OUT_DIR / "today_props_raw.csv"


# ---------------------------------------------------------------------------
# Inverse distribution math: line + odds → market-implied mean μ
# ---------------------------------------------------------------------------
def _american_to_implied(odds: float) -> Optional[float]:
    if odds is None or pd.isna(odds):
        return None
    a = float(odds)
    return 1 / (1 + a / 100.0) if a > 0 else abs(a) / (abs(a) + 100.0)


def _no_vig(over_o: float, under_o: float) -> Optional[float]:
    po = _american_to_implied(over_o)
    pu = _american_to_implied(under_o)
    if po is None or pu is None or (po + pu) <= 0:
        return None
    return po / (po + pu)


def _solve_poisson_mu(p_over: float, line: float) -> Optional[float]:
    """Find μ such that P(X > line | X~Poisson(μ)) = p_over."""
    if p_over <= 0 or p_over >= 1:
        return None
    threshold = math.floor(line) + 1
    f = lambda mu: (1.0 - poisson.cdf(threshold - 1, mu)) - p_over
    try:
        return float(brentq(f, 1e-3, 50.0, xtol=1e-4))
    except (ValueError, RuntimeError):
        return None


def _solve_nb_mu(p_over: float, line: float, dispersion: float = 0.30) -> Optional[float]:
    if p_over <= 0 or p_over >= 1:
        return None
    threshold = math.floor(line) + 1

    def f(mu):
        n = max(1e-3, mu / max(1e-3, dispersion))
        p = n / (n + mu)
        return (1.0 - nbinom.cdf(threshold - 1, n, p)) - p_over

    try:
        return float(brentq(f, 1e-3, 30.0, xtol=1e-4))
    except (ValueError, RuntimeError):
        return None


def _solve_normal_mu(p_over: float, line: float, sigma: float = 1.0) -> Optional[float]:
    if p_over <= 0 or p_over >= 1:
        return None
    # P(X > line | N(μ, σ²)) = 1 - Φ((line - μ)/σ) = p_over
    #  →  (line - μ)/σ = Φ⁻¹(1 - p_over)
    z = norm.ppf(1.0 - p_over)
    return float(line - sigma * z)


# Same mapping the props engine uses — market → distribution + projection col
_MARKET_TO_PROJ_COL = {
    "pitcher_strikeouts":   ("proj_strikeouts",    "poisson", "pitcher"),
    "pitcher_walks":        ("proj_walks",         "poisson", "pitcher"),
    "pitcher_hits_allowed": ("proj_hits_allowed",  "poisson", "pitcher"),
    "pitcher_earned_runs":  ("proj_runs_allowed",  "poisson", "pitcher"),
    "pitcher_outs":         ("proj_outs",          "normal",  "pitcher"),
    "batter_hits":          ("proj_hits",          "nb",      "hitter"),
    "batter_home_runs":     ("proj_hr",            "poisson", "hitter"),
    "batter_total_bases":   ("proj_total_bases",   "nb",      "hitter"),
    "batter_strikeouts":    ("proj_strikeouts",    "poisson", "hitter"),
    "batter_walks":         ("proj_walks",         "poisson", "hitter"),
    "batter_rbis":          ("proj_rbi",           "poisson", "hitter"),
    "batter_runs_scored":   ("proj_runs",          "poisson", "hitter"),
}


def _market_implied_mu(market: str, line: float, over_o, under_o) -> Optional[float]:
    info = _MARKET_TO_PROJ_COL.get(market)
    if info is None:
        return None
    _, dist, _ = info
    p_over = _no_vig(over_o, under_o)
    if p_over is None:
        return None
    if dist == "poisson":
        return _solve_poisson_mu(p_over, float(line))
    if dist == "nb":
        return _solve_nb_mu(p_over, float(line))
    if dist == "normal":
        return _solve_normal_mu(p_over, float(line))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class MarketPrior:
    player_name: str
    market: str
    proj_col: str
    kind: str            # "pitcher" or "hitter"
    line: float
    market_mu: float     # market-implied projected mean
    n_books: int         # how many sportsbooks contributed
    mlb_id: Optional[int] = None


def load_market_priors(props_path: Path = TODAY_PROPS_RAW) -> list[MarketPrior]:
    """Read today's props CSV and invert each line into a market-implied μ."""
    if not props_path.exists():
        return []
    try:
        df = pd.read_csv(props_path, low_memory=False)
    except Exception:
        return []
    if df.empty:
        return []

    # Average across sportsbooks for the same (player, market, line) before
    # inverting — gives a cleaner consensus prior than picking one book.
    grp_cols = [c for c in ("player_name", "market", "line") if c in df.columns]
    if not grp_cols:
        return []

    consensus = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              over_odds=("over_odds", "mean"),
              under_odds=("under_odds", "mean"),
              n_books=("sportsbook", "nunique"),
              mlb_id=("mlb_id", "first") if "mlb_id" in df.columns else ("player_name", "first"),
          )
    )

    out: list[MarketPrior] = []
    for _, r in consensus.iterrows():
        market = str(r["market"])
        info = _MARKET_TO_PROJ_COL.get(market)
        if info is None:
            continue
        proj_col, _, kind = info
        mu = _market_implied_mu(market, float(r["line"]),
                                 r.get("over_odds"), r.get("under_odds"))
        if mu is None or not np.isfinite(mu):
            continue
        out.append(MarketPrior(
            player_name=str(r["player_name"]),
            market=market, proj_col=proj_col, kind=kind,
            line=float(r["line"]), market_mu=float(mu),
            n_books=int(r.get("n_books", 1) or 1),
            mlb_id=(int(r["mlb_id"]) if "mlb_id" in r and pd.notna(r["mlb_id"]) else None),
        ))
    return out


def apply_market_calibration(
    proj_df: pd.DataFrame,
    priors: list[MarketPrior],
    *,
    beta_base: float = 0.20,
    beta_cap: float = 0.40,
    confidence_col: str = "confidence_score",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Blend each `proj_<x>` column toward its market-implied value when a
    prior exists for that (player, market). β is dynamic:

        β = min(beta_cap, beta_base × n_books × low_confidence_bonus)

    More books = more confidence in the consensus. Lower model confidence
    grade = higher β (we lean on the market more for noisy projections).

    Returns a copy with two new columns per affected stat:
        proj_<x>_raw      — the model's pre-blend projection
        proj_<x>          — the blended (final) projection
        market_<x>_mu     — what the market implied
    """
    if proj_df is None or proj_df.empty or not priors:
        return proj_df

    out = proj_df.copy()
    by_player: dict[tuple, list[MarketPrior]] = {}
    for p in priors:
        key = (p.player_name.strip().lower(), p.kind)
        by_player.setdefault(key, []).append(p)

    touched: dict[str, int] = {}

    for idx, row in out.iterrows():
        name = str(row.get("player_name", "")).strip().lower()
        kind = str(row.get("player_type", row.get("kind", ""))).lower()
        priors_for = by_player.get((name, kind), [])
        if not priors_for:
            continue

        # Per-row confidence (0-100 grade). Default mid-band.
        conf = row.get(confidence_col)
        try:
            conf_g = float(conf)
        except (TypeError, ValueError):
            conf_g = 50.0
        low_conf_bonus = 1.0 + max(0.0, (50.0 - conf_g) / 50.0)   # up to 2x for grade ≤ 0

        for p in priors_for:
            if p.proj_col not in out.columns:
                continue
            raw_val = pd.to_numeric(out.at[idx, p.proj_col], errors="coerce")
            if pd.isna(raw_val) or raw_val < 0:
                continue

            # Dynamic β
            beta = min(beta_cap, beta_base * max(1, p.n_books) * low_conf_bonus / 2.0)
            blended = (1.0 - beta) * float(raw_val) + beta * float(p.market_mu)

            raw_col = f"{p.proj_col}_raw"
            mkt_col = f"market_{p.proj_col[len('proj_'):]}_mu" if p.proj_col.startswith("proj_") else f"market_{p.proj_col}_mu"
            if raw_col not in out.columns:
                out[raw_col] = out[p.proj_col]
            if mkt_col not in out.columns:
                out[mkt_col] = np.nan
            out.at[idx, raw_col] = float(raw_val)
            out.at[idx, mkt_col] = float(p.market_mu)
            out.at[idx, p.proj_col] = max(0.0, blended)
            touched[p.proj_col] = touched.get(p.proj_col, 0) + 1

    if verbose and touched:
        print("  Market calibration applied to:")
        for col, n in sorted(touched.items()):
            print(f"    {col}: {n} player rows")
    return out


# ---------------------------------------------------------------------------
# CLI: dump market priors for inspection
# ---------------------------------------------------------------------------
def main() -> None:
    priors = load_market_priors()
    if not priors:
        print("No market priors available — run props_fetch first.")
        return
    df = pd.DataFrame([{
        "player_name": p.player_name, "market": p.market, "line": p.line,
        "market_mu": round(p.market_mu, 2), "n_books": p.n_books,
    } for p in priors])
    df = df.sort_values(["market", "market_mu"], ascending=[True, False])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
