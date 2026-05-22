"""
props_engine.py
===============
Math layer for the player-prop "edge engine":

    1. Convert American odds → decimal payout, implied probability, no-vig fair price
    2. Convert a player projection (mean) → P(over line) using a stat-appropriate
       distribution (Poisson for K/BB/HR, Negative Binomial for hits, Normal for IP)
    3. Compute edge (model_p − no_vig_p), EV per $1 stake, and confidence-weighted
       score for ranking the best plays
    4. Decide a side (OVER / UNDER) and a flag (VALUE / PASS) per row

This module is pure math — no I/O, no config. It's called by props_fetch.py
which feeds it (line, odds) tuples and projections from
outputs/hitterspitchers_today.csv.

Design notes
------------
* Calibration: if calibration.json exists, projections are shifted by -bias
  before computing P(over). Same source of truth as the player-accuracy display.
* Confidence: the projection's 20-80 grade is folded into a `score` value
  (EV × confidence_weight) so the ranking surface promotes plays where the
  model has conviction, not just plays sitting on a noisy projection.
* Conservative tolerances: Poisson is appropriate for K and HR (rare events
  with mean < 10). Hits use Negative Binomial because a hitter's per-PA
  hit count is overdispersed relative to Poisson. IP uses Normal because
  it's a continuous quantity in roughly 3.0–7.5 IP range with ~1.0 std.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom, norm

HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = HERE / "calibration.json"

# EV thresholds — tightened after the 30-38-0 / -22.6% ROI diagnostic showed
# we were mass-flagging shaky 4-8% EV picks where the model had structural
# bias. Real prop edges in efficient markets cluster at 3-15%; anything north
# of 30% is almost always a model error (missing injury, lineup change,
# starter swap, etc.) rather than a genuine inefficiency, so we route those
# to a separate REVIEW bucket instead of treating them as betting picks.
VALUE_EV_THRESHOLD  = 0.08   # +8% EV minimum to flag as VALUE
REVIEW_EV_THRESHOLD = 0.30   # +30% EV → flag as REVIEW (suspect projection)
PASS_EV_THRESHOLD   = 0.00   # below break-even = pass

# Confidence floor: a plain 50-grade projection gets weight 1.0; below 30
# the score is effectively zeroed out so Value flags don't fire on weak data.
def _confidence_weight(confidence_score: float | None) -> float:
    if confidence_score is None or pd.isna(confidence_score):
        return 0.6
    g = float(confidence_score)
    if g >= 60: return 1.10
    if g >= 50: return 1.00
    if g >= 40: return 0.80
    if g >= 30: return 0.55
    return 0.30


# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------
def american_to_decimal(american: float | int | None) -> Optional[float]:
    """Convert American odds → decimal (total payout per $1 incl. stake)."""
    if american is None or (isinstance(american, float) and math.isnan(american)):
        return None
    a = float(american)
    if a > 0:
        return 1 + a / 100.0
    if a < 0:
        return 1 + 100.0 / abs(a)
    return None


def american_to_implied_prob(american: float | int | None) -> Optional[float]:
    """American odds → vig-included implied probability."""
    d = american_to_decimal(american)
    return None if d is None else 1.0 / d


def decimal_payout(american: float | int | None) -> Optional[float]:
    """Profit per $1 stake (decimal_odds − 1)."""
    d = american_to_decimal(american)
    return None if d is None else d - 1.0


def no_vig_pair(over_american, under_american) -> tuple[Optional[float], Optional[float]]:
    """
    Strip the vig from a two-sided market. Returns (no_vig_p_over, no_vig_p_under)
    that sum to 1.0. None if either side is missing.
    """
    p_o = american_to_implied_prob(over_american)
    p_u = american_to_implied_prob(under_american)
    if p_o is None or p_u is None:
        return (None, None)
    total = p_o + p_u
    if total <= 0:
        return (None, None)
    return (p_o / total, p_u / total)


# ---------------------------------------------------------------------------
# Distribution-aware P(stat > line)
# ---------------------------------------------------------------------------
def prob_over_poisson(mu: float, line: float) -> float:
    """
    P(X > line) where X ~ Poisson(mu). Handles half-lines correctly:
      line=5.5: P(X >= 6) = 1 − P(X <= 5)
      line=5.0: P(X >= 6) = 1 − P(X <= 5)   (whole line: push at 5 not counted as win)
    """
    if mu is None or pd.isna(mu) or mu < 0:
        return 0.5
    mu = max(0.001, float(mu))
    # For half lines we need P(X >= floor(line)+1).
    # For whole lines, the over typically means strictly greater than the line,
    # with a push at exactly L → no profit, no loss. We treat it as >= L+1 for
    # EV purposes (slightly conservative; pushes are rare in MLB props anyway).
    threshold = math.floor(line) + 1
    # 1 - cdf(threshold - 1) == P(X >= threshold) for integer threshold
    p = 1.0 - poisson.cdf(threshold - 1, mu)
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def prob_over_neg_binom(mu: float, line: float, dispersion: float = 0.30) -> float:
    """
    P(X > line) where X ~ NegativeBinomial with mean=mu and dispersion (overdispersion)
    parameter. Used for hits — slightly overdispersed relative to Poisson because
    hitters' true rate varies across at-bats (matchup, count, fatigue).
    """
    if mu is None or pd.isna(mu) or mu < 0:
        return 0.5
    mu = max(0.001, float(mu))
    # Convert (mean, dispersion) parameterization to scipy's (n, p):
    #   variance = mu + mu^2 / n  →  n = mu / dispersion (with our convention)
    # Smaller `dispersion` = closer to Poisson.
    n = max(1e-3, mu / max(1e-3, dispersion))
    p = n / (n + mu)
    threshold = math.floor(line) + 1
    prob = 1.0 - nbinom.cdf(threshold - 1, n, p)
    return float(np.clip(prob, 1e-6, 1 - 1e-6))


def prob_over_normal(mu: float, line: float, sigma: float = 1.0) -> float:
    """
    P(X > line) for a continuous quantity (IP). sigma defaults to 1.0 IP, which
    matches the typical std observed in pitcher_game_data.
    """
    if mu is None or pd.isna(mu):
        return 0.5
    sigma = max(0.1, float(sigma))
    return float(np.clip(1.0 - norm.cdf(line, loc=float(mu), scale=sigma),
                         1e-6, 1 - 1e-6))


# Map sportsbook market codes → distribution + projection column
# This is the single source of truth for "how do we score this prop?"
MARKET_CONFIG: dict[str, dict] = {
    "pitcher_strikeouts":     {"proj_col": "proj_strikeouts",   "dist": "poisson", "kind": "pitcher"},
    "pitcher_walks":          {"proj_col": "proj_walks",        "dist": "poisson", "kind": "pitcher"},
    "pitcher_hits_allowed":   {"proj_col": "proj_hits_allowed", "dist": "poisson", "kind": "pitcher"},
    "pitcher_earned_runs":    {"proj_col": "proj_runs_allowed", "dist": "poisson", "kind": "pitcher"},
    "pitcher_outs":           {"proj_col": "proj_outs",         "dist": "normal",  "kind": "pitcher", "sigma": 3.0},
    # batter
    "batter_hits":            {"proj_col": "proj_hits",         "dist": "nb",      "kind": "hitter"},
    "batter_home_runs":       {"proj_col": "proj_hr",           "dist": "poisson", "kind": "hitter"},
    "batter_total_bases":     {"proj_col": "proj_total_bases",  "dist": "nb",      "kind": "hitter"},
    "batter_strikeouts":      {"proj_col": "proj_strikeouts",   "dist": "poisson", "kind": "hitter"},
    "batter_walks":           {"proj_col": "proj_walks",        "dist": "poisson", "kind": "hitter"},
    "batter_rbis":            {"proj_col": "proj_rbi",          "dist": "poisson", "kind": "hitter"},
    "batter_runs_scored":     {"proj_col": "proj_runs",         "dist": "poisson", "kind": "hitter"},
}


# ---------------------------------------------------------------------------
# Calibration loader (optional — applied to projections before scoring)
# ---------------------------------------------------------------------------
def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except Exception:
        return {}


def _apply_calibration(value: float, kind: str, proj_col: str, cal: dict) -> float:
    """Identity now — projection is expected to already be raw (uncalibrated).

    Calibration WAS being applied here too, on top of the calibrated CSV that
    `props_fetch` was reading. That stacked the bias correction twice and
    pushed every Over above its sportsbook line, mechanically inflating EVs
    and tanking ROI to ~−22% with a 44% Over hit rate. We now read the raw
    projection CSV in `props_fetch` and treat this function as a no-op so
    the engine never silently re-shifts the projection.
    """
    return float(value) if value is not None else value


# ---------------------------------------------------------------------------
# Edge & EV
# ---------------------------------------------------------------------------
@dataclass
class PropEdge:
    market: str
    line: float
    p_over_model: float
    p_over_no_vig: Optional[float]
    edge_over: Optional[float]
    edge_under: Optional[float]
    ev_over: Optional[float]
    ev_under: Optional[float]
    side: str          # OVER | UNDER | PASS
    ev: Optional[float]
    flag: str          # VALUE | PASS
    score: float       # EV × confidence weight
    confidence: float


def _prob_over(market: str, mu: float, line: float) -> float:
    cfg = MARKET_CONFIG.get(market)
    if not cfg:
        return 0.5
    dist = cfg.get("dist", "poisson")
    if dist == "poisson":
        return prob_over_poisson(mu, line)
    if dist == "nb":
        return prob_over_neg_binom(mu, line)
    if dist == "normal":
        return prob_over_normal(mu, line, sigma=cfg.get("sigma", 1.0))
    return prob_over_poisson(mu, line)


def compute_edge_for_prop(
    market: str,
    line: float,
    over_odds: float | int,
    under_odds: float | int,
    projection: float,
    confidence_score: float | None = None,
    cal: dict | None = None,
) -> PropEdge:
    """
    Score one prop. Returns the side (OVER/UNDER/PASS), expected value per
    $1 stake, edge versus no-vig, and a confidence-weighted ranking score.
    """
    cfg = MARKET_CONFIG.get(market) or {}
    if cal is None:
        cal = load_calibration()
    proj_col = cfg.get("proj_col", "")
    kind = cfg.get("kind", "hitter")

    # Apply calibration to projection (same bias correction the accuracy
    # display uses, so what users see and what we score agree).
    mu = _apply_calibration(projection, kind, proj_col, cal) if cal else projection

    p_over_model = _prob_over(market, mu, float(line))
    p_under_model = 1.0 - p_over_model

    nv_o, nv_u = no_vig_pair(over_odds, under_odds)
    edge_o = (p_over_model - nv_o)  if nv_o is not None else None
    edge_u = (p_under_model - nv_u) if nv_u is not None else None

    payout_o = decimal_payout(over_odds)
    payout_u = decimal_payout(under_odds)
    ev_o = (p_over_model * payout_o - (1 - p_over_model)) if payout_o is not None else None
    ev_u = (p_under_model * payout_u - (1 - p_under_model)) if payout_u is not None else None

    # Pick the side with the better EV; tie-break to OVER.
    side = "PASS"
    ev = None
    if ev_o is not None and ev_u is not None:
        if ev_o >= ev_u:
            side, ev = "OVER", ev_o
        else:
            side, ev = "UNDER", ev_u
    elif ev_o is not None:
        side, ev = "OVER", ev_o
    elif ev_u is not None:
        side, ev = "UNDER", ev_u

    # Three-bucket flag:
    #   PASS   — below the value threshold, don't bet
    #   VALUE  — genuine edge in the realistic 8–30% EV band
    #   REVIEW — >30% EV: almost always model error, not a real edge.
    #            Surfaced separately so the user can spot-check, but never
    #            auto-included in the bet recommendation list.
    flag = "PASS"
    if ev is not None:
        if ev >= REVIEW_EV_THRESHOLD:
            flag = "REVIEW"
        elif ev >= VALUE_EV_THRESHOLD:
            flag = "VALUE"

    cw = _confidence_weight(confidence_score)
    score = (ev or 0.0) * cw

    return PropEdge(
        market=market,
        line=float(line),
        p_over_model=p_over_model,
        p_over_no_vig=nv_o,
        edge_over=edge_o,
        edge_under=edge_u,
        ev_over=ev_o,
        ev_under=ev_u,
        side=side,
        ev=ev,
        flag=flag,
        score=score,
        confidence=cw,
    )


# ---------------------------------------------------------------------------
# DataFrame entry point — what props_fetch / daily_update calls
# ---------------------------------------------------------------------------
def add_edge_columns(props_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join props (per-line) with projections (per-player) and compute edge/EV
    columns. Returns a copy with these added:
      proj_value, p_over_model, p_over_no_vig, edge_over, edge_under,
      ev_over, ev_under, side, ev, flag, score, confidence_weight.
    """
    if props_df is None or props_df.empty or projections_df is None or projections_df.empty:
        return pd.DataFrame()

    # Index projections by mlb_id (preferred) and by player_name (fallback).
    proj_by_id: dict[int, pd.Series] = {}
    proj_by_name: dict[str, pd.Series] = {}
    if "mlb_id" in projections_df.columns:
        for _, r in projections_df.iterrows():
            try:
                k = int(float(r["mlb_id"]))
                proj_by_id[k] = r
            except (TypeError, ValueError):
                pass
    for _, r in projections_df.iterrows():
        nm = str(r.get("player_name", "")).strip().lower()
        if nm:
            proj_by_name[nm] = r

    cal = load_calibration()
    out_rows: list[dict] = []
    for _, p in props_df.iterrows():
        market = str(p.get("market", ""))
        cfg = MARKET_CONFIG.get(market)
        if not cfg:
            continue   # unsupported market

        # Try mlb_id match, then name match
        proj = None
        try:
            mid = int(float(p.get("mlb_id"))) if p.get("mlb_id") is not None else None
        except (TypeError, ValueError):
            mid = None
        if mid is not None and mid in proj_by_id:
            proj = proj_by_id[mid]
        else:
            nm = str(p.get("player_name", "")).strip().lower()
            if nm in proj_by_name:
                proj = proj_by_name[nm]
        if proj is None:
            continue   # no projection found, skip

        proj_col = cfg["proj_col"]
        proj_val = pd.to_numeric(proj.get(proj_col), errors="coerce")
        if pd.isna(proj_val):
            continue

        edge = compute_edge_for_prop(
            market=market,
            line=float(p["line"]),
            over_odds=p.get("over_odds"),
            under_odds=p.get("under_odds"),
            projection=float(proj_val),
            confidence_score=proj.get("confidence_score"),
            cal=cal,
        )

        row = dict(p)
        row.update({
            "kind": cfg["kind"],
            "proj_value": round(float(proj_val), 3),
            "p_over_model": round(edge.p_over_model, 4),
            "p_over_no_vig": round(edge.p_over_no_vig, 4) if edge.p_over_no_vig is not None else None,
            "edge_over": round(edge.edge_over, 4) if edge.edge_over is not None else None,
            "edge_under": round(edge.edge_under, 4) if edge.edge_under is not None else None,
            "ev_over": round(edge.ev_over, 4) if edge.ev_over is not None else None,
            "ev_under": round(edge.ev_under, 4) if edge.ev_under is not None else None,
            "side": edge.side,
            "ev": round(edge.ev, 4) if edge.ev is not None else None,
            "flag": edge.flag,
            "score": round(edge.score, 4),
            "confidence_weight": round(edge.confidence, 2),
            "confidence_score": int(proj.get("confidence_score") or 50),
        })
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)
    # Sort by score (EV × confidence) descending — best plays first
    if "score" in out.columns:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    # Quick self-test: 5.5 K line at -110/-110, model says 6.2 K
    e = compute_edge_for_prop(
        market="pitcher_strikeouts",
        line=5.5, over_odds=-110, under_odds=-110,
        projection=6.2, confidence_score=60,
    )
    print(f"P(over) model = {e.p_over_model:.3f}")
    print(f"P(over) no-vig = {e.p_over_no_vig:.3f}")
    print(f"Edge over = {e.edge_over:+.3f}")
    print(f"EV over = {e.ev_over:+.3f}  EV under = {e.ev_under:+.3f}")
    print(f"Side: {e.side}  EV: {e.ev:+.3f}  Flag: {e.flag}  Score: {e.score:+.3f}")
