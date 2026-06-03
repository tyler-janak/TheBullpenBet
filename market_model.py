"""
market_model.py
================
Market-consensus betting model for moneylines and player props.

What this is
------------
A second, complementary lens to the projection model. Instead of predicting
outcomes from box-score features, this asks: "What does the betting MARKET
think the fair price is, and where is one book offering a price that beats
that consensus?" Built directly on the approach laid out in the two videos:

  • Video 1 (quant-style, market-data first): the first input to any betting
    model should be other sportsbooks' odds. Devig each book to extract its
    no-juice fair price, weight by sharpness/liquidity, take the weighted
    average. If your bookable price beats the consensus, you have +EV. Size
    with half-Kelly multiplied by a stack of conviction factors (confidence,
    market activity, crossed/arb, book softness, market width, exchange
    liquidity).

  • Video 2 (NHL spreadsheet model): start simple, validate honestly. Track
    every bet, measure CLV (closing-line value) — the only thing that proves
    skill vs variance over a real sample. The output of this module slots
    directly into the 2026_props_log.csv / 2026_props_clv.csv plumbing.

Public API
----------
    american_to_prob(odds)              odds <-> probability conversions
    prob_to_american(p)
    devig_two_way(p_a, p_b)             power devig of one book's market
    devig_multiway(probs)
    consensus_fair_price(market)        weighted consensus across books
    detect_value(market, my_book, my_price, side)  -> ValueBet | None
    recommended_stake(value, bankroll)  half-Kelly × stacked multipliers
    score_slate(markets, my_book, bankroll)        -> DataFrame of bets

Input market dict (book-agnostic — adapts to OddsJam, The Odds API, etc.):
    {
      "market_id": "MLB:gp123:moneyline",
      "kind": "moneyline",                # or "spread" / "total" / "player_prop"
      "sides": ["away", "home"],          # or ["over", "under"]
      "market_type_tag": "mlb_ml",        # used for activity multiplier
      "is_crossed": False,                # set True if an arb exists elsewhere
      "books": {
        "Pinnacle":  {"away": +122, "home": -138,            "limit":  500},
        "Circa":     {"away": +125, "home": -145,            "limit": 1000},
        "BetOnline": {"away": +120, "home": -140,            "limit":  500},
        "FanDuel":   {"away": +136, "home": -160,            "limit": 1000},
        "Novig":     {"away": +130, "home": -135, "liq": 5600},
      },
    }

OddsJam integration
-------------------
OddsJam's Game Odds / Player Props REST responses can be mapped into this
shape — write a thin adapter in `oddsjam_fetch.py` that pulls each market's
per-book prices (and exchange liquidity where present) and emits the dict
above. Then call `score_slate(...)`.

Run as a script for a synthetic demo:
    python market_model.py
"""

from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

warnings.filterwarnings("ignore")

# Sanity defenses against bad upstream feeds (an international book mis-labeling
# the home/away side, a stale market with 100-cent juice, etc.). Tuned loose
# enough not to filter legitimate sharp markets, tight enough that a flipped
# feed can't poison the consensus.
BAD_OVERROUND_CENTS = 40.0      # drop books charging > 40c of juice (broken)
OUTLIER_PROB_DELTA  = 0.20      # drop books > 20pp off median fair prob
MAX_PLAUSIBLE_EDGE  = 0.50      # any "edge" > 50% is almost certainly bad data
SUSPICIOUS_EDGE     = 0.20      # 20-50% gets flagged for manual review
MIN_BOOKS_REQUIRED  = 2         # consensus needs at least 2 books to be real


# =============================================================================
# 1. ODDS  <->  PROBABILITY
# =============================================================================

def american_to_prob(odds: float) -> float:
    """American odds -> implied probability (includes the vig)."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return float("nan")
    if o == 0 or not math.isfinite(o):
        return float("nan")
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def prob_to_american(p: float) -> float:
    """No-vig probability -> American odds."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return float("nan")
    if not (0.0 < p < 1.0):
        return float("nan")
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def implied_edge(my_price: float, fair_prob: float) -> tuple[float, float]:
    """Return (edge_pct, edge_cents) for an offered American price vs a
    consensus fair probability. edge_pct = expected $ return per $1 staked."""
    if not math.isfinite(fair_prob) or fair_prob <= 0:
        return float("nan"), float("nan")
    if my_price >= 0:
        payout = my_price / 100.0 + 1.0          # $1 stake -> total return on win
    else:
        payout = 100.0 / abs(my_price) + 1.0
    ev_per_dollar = fair_prob * payout - 1.0
    fair_price = prob_to_american(fair_prob)
    edge_cents = float(fair_price - my_price) if math.isfinite(fair_price) else float("nan")
    return float(ev_per_dollar), edge_cents


# =============================================================================
# 2. DEVIGGING — pull the juice out of a single book's market
# =============================================================================

def _power_devig_solve(probs: list[float]) -> float:
    """Find k such that sum(p_i ** k) == 1. Bisection — converges in ~50 iter."""
    lo, hi = 0.5, 1.5
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if sum(p ** mid for p in probs) > 1.0:
            lo = mid                              # too low k -> sum too big
        else:
            hi = mid
    return (lo + hi) / 2.0


def devig_two_way(p_a: float, p_b: float, method: str = "power") -> tuple[float, float]:
    """Two-way market devig. Default 'power' is what sharp shops use — it
    preserves the favorite/dog asymmetry better than naive 'multiplicative'."""
    if not (p_a > 0 and p_b > 0):
        return float("nan"), float("nan")
    s = p_a + p_b
    if s <= 1.0:                                  # already no-vig (or +EV pair!)
        return p_a, p_b
    if method == "multiplicative":
        return p_a / s, p_b / s
    k = _power_devig_solve([p_a, p_b])
    return p_a ** k, p_b ** k


def devig_multiway(probs: list[float], method: str = "power") -> list[float]:
    """N-way devig (e.g. soccer 1X2, futures markets)."""
    probs = [p for p in probs if p > 0]
    if not probs:
        return []
    s = sum(probs)
    if s <= 1.0:
        return list(probs)
    if method == "multiplicative":
        return [p / s for p in probs]
    k = _power_devig_solve(probs)
    return [p ** k for p in probs]


def book_fair_probs(book_prices: dict[str, float], method: str = "power") -> dict[str, float]:
    """One book's per-side American prices -> no-vig fair probabilities."""
    sides = list(book_prices.keys())
    probs = [american_to_prob(book_prices[s]) for s in sides]
    if len(sides) == 2:
        a, b = devig_two_way(probs[0], probs[1], method=method)
        return {sides[0]: a, sides[1]: b}
    fair = devig_multiway(probs, method=method)
    return dict(zip(sides, fair))


def market_width_cents(book_prices: dict[str, float]) -> float:
    """Overround in 'cents' for a two-way market — roughly the bid/ask spread
    the book is charging in American-odds units. Tighter = sharper market."""
    probs = [american_to_prob(p) for p in book_prices.values()]
    if any(not math.isfinite(p) for p in probs):
        return float("nan")
    overround = sum(probs) - 1.0
    return float(max(0.0, overround) * 200.0)


# =============================================================================
# 3. BOOK SHARPNESS WEIGHTS  (Video 1: weight sharper books higher)
# =============================================================================
# Starting weights drawn directly from Video 1's framing. For MAINLINE markets
# the overseas sharps (Pinnacle, Circa, BetOnline) carry most of the weight;
# for PLAYER PROPS US books get a big bump because they take so much more
# action on props that they're effectively the price-setters there. EXCHANGES
# (Novig, ProphetX) are weighted PROPORTIONAL TO LIQUIDITY — they're only
# sharp when real money is sitting on the other side.
#
# These should ultimately be back-tested per sport/market — Video 1 mentioned
# that OddsJam does this empirically. Treat the numbers below as priors.

MAINLINE_WEIGHTS = {
    "Pinnacle":   0.30,
    "Circa":      0.25,
    "BetOnline":  0.15,
    "BetCRIS":    0.15,
    "Bet365":     0.05,
    "FanDuel":    0.05,
    "DraftKings": 0.03,
    "Caesars":    0.02,
}

PROPS_WEIGHTS = {
    "Pinnacle":   0.15,
    "Circa":      0.10,
    "BetOnline":  0.05,
    "FanDuel":    0.25,
    "DraftKings": 0.20,
    "Caesars":    0.15,
    "BetMGM":     0.10,
}

EXCHANGES = {"Novig", "ProphetX"}
EXCHANGE_LIQ_PER_WEIGHT = 1000.0       # $1000 of liquidity -> +5% raw weight
EXCHANGE_LIQ_STEP = 0.05
EXCHANGE_MAX_WEIGHT = 0.30             # cap so a huge exchange book can't dominate


def exchange_weight(liq_usd: float) -> float:
    """Scale an exchange's weight by how much money is sitting opposite this
    price. No liquidity -> zero weight; lots -> capped at 30%."""
    try:
        liq = float(liq_usd)
    except (TypeError, ValueError):
        return 0.0
    if liq <= 0:
        return 0.0
    raw = (liq / EXCHANGE_LIQ_PER_WEIGHT) * EXCHANGE_LIQ_STEP
    return float(min(raw, EXCHANGE_MAX_WEIGHT))


def book_weights_for(kind: str, books_present: Iterable[str],
                     exchange_liquidity: dict[str, float] | None = None) -> dict[str, float]:
    """Pick the right weight table for the market kind. Books we don't have a
    price from get zero. Exchanges are scaled by their visible liquidity."""
    base = PROPS_WEIGHTS if kind == "player_prop" else MAINLINE_WEIGHTS
    out: dict[str, float] = {}
    for b in books_present:
        if b in EXCHANGES:
            liq = (exchange_liquidity or {}).get(b, 0.0)
            out[b] = exchange_weight(liq)
        else:
            out[b] = base.get(b, 0.02)            # unknown -> small default
    return out


# =============================================================================
# 4. CONSENSUS FAIR PRICE
# =============================================================================

@dataclass
class ConsensusResult:
    fair_probs:        dict[str, float]                  # per-side no-vig prob
    fair_prices:       dict[str, float]                  # per-side American odds
    per_book_fair:     dict[str, dict[str, float]]       # devigged per-book probs
    weights:           dict[str, float]                  # normalized weights used
    overround_by_book: dict[str, float]                  # market width per book


def consensus_fair_price(market: dict) -> ConsensusResult:
    """Devig every book in the market, weight them, take the weighted average.
    Returns the consensus per-side no-vig probability + American price."""
    kind = market.get("kind", "moneyline")
    sides = list(market["sides"])
    books = market.get("books", {})

    per_book_fair: dict[str, dict[str, float]] = {}
    overround_by_book: dict[str, float] = {}
    for bname, bdata in books.items():
        prices_by_side = {s: bdata[s] for s in sides
                          if s in bdata and bdata[s] is not None}
        if len(prices_by_side) != len(sides):
            continue                              # need both sides to devig
        per_book_fair[bname] = book_fair_probs(prices_by_side)
        overround_by_book[bname] = market_width_cents(prices_by_side)

    # ── Sanity defense 1: drop books with absurd juice (broken feed) ──
    for bname in list(per_book_fair.keys()):
        over = overround_by_book.get(bname, float("nan"))
        if math.isfinite(over) and over > BAD_OVERROUND_CENTS:
            per_book_fair.pop(bname, None)

    # ── Sanity defense 2: drop books whose per-side fair prob is a wild
    # outlier vs the median. This catches the classic "international feed has
    # the home/away labels flipped on a US mainline" case — that one bad book
    # quietly inverts the side and the bogus side then dominates the average.
    if len(per_book_fair) >= 3:
        for s in sides:
            vals = [b.get(s, float("nan")) for b in per_book_fair.values()
                    if math.isfinite(b.get(s, float("nan")))]
            if len(vals) < 3:
                continue
            med = statistics.median(vals)
            for bname in list(per_book_fair.keys()):
                v = per_book_fair[bname].get(s, float("nan"))
                if math.isfinite(v) and abs(v - med) > OUTLIER_PROB_DELTA:
                    per_book_fair.pop(bname, None)

    exchange_liq = {b: float(books[b].get("liq", 0.0) or 0.0)
                    for b in books if b in EXCHANGES}

    weights = book_weights_for(kind, per_book_fair.keys(), exchange_liq)
    wsum = sum(weights.values())
    if wsum <= 0:
        return ConsensusResult({}, {}, per_book_fair, weights, overround_by_book)
    weights = {k: v / wsum for k, v in weights.items()}

    fair_probs = {s: 0.0 for s in sides}
    for bname, bfair in per_book_fair.items():
        w = weights.get(bname, 0.0)
        for s in sides:
            fair_probs[s] += w * bfair.get(s, 0.0)

    # Re-normalize tiny float drift; emit American prices.
    s_total = sum(fair_probs.values())
    if s_total > 0:
        fair_probs = {s: p / s_total for s, p in fair_probs.items()}
    fair_prices = {s: prob_to_american(p) for s, p in fair_probs.items()}
    return ConsensusResult(fair_probs, fair_prices, per_book_fair, weights, overround_by_book)


# =============================================================================
# 5. +EV DETECTION
# =============================================================================

@dataclass
class ValueBet:
    market_id:       str
    kind:            str
    side:            str
    my_book:         str
    my_price:        float
    fair_price:      float
    fair_prob:       float
    edge_pct:        float          # EV per $1 staked
    edge_cents:      float          # American-odds cents above fair
    plus_ev_vs_all:  bool           # +EV vs EVERY other book's devigged fair?
    is_crossed:      bool
    market_width:    float          # narrowest book's overround in cents
    exchange_liq:    float          # largest exchange liquidity on this side
    per_book_fair:   dict[str, dict[str, float]] = field(default_factory=dict)


def detect_value(market: dict, my_book: str, my_price: float, side: str,
                 min_edge_pct: float = 0.0) -> ValueBet | None:
    """Compare one offered price against consensus. Returns ValueBet if
    edge_pct >= min_edge_pct, else None."""
    cons = consensus_fair_price(market)
    if side not in cons.fair_probs:
        return None
    edge_pct, edge_cents = implied_edge(my_price, cons.fair_probs[side])
    if not math.isfinite(edge_pct) or edge_pct < min_edge_pct:
        return None

    # Confidence check (Video 1): is the offered price +EV against EVERY other
    # book's devigged fair price on this side? If yes, every market in the
    # world thinks this bet is good — double down (handled in sizing).
    my_offered_prob = american_to_prob(my_price)
    plus_ev_vs_all = True
    for bname, bfair in cons.per_book_fair.items():
        if bname == my_book:
            continue
        if my_offered_prob >= bfair.get(side, 0.0):
            # my_offered_prob >= bfair means this book thinks our side is
            # LESS likely than our offered price implies -> not +EV vs them.
            plus_ev_vs_all = False
            break

    widths = [w for w in cons.overround_by_book.values() if math.isfinite(w)]
    width = float(min(widths)) if widths else float("nan")

    # Best exchange liquidity opposing our side -> shows real money is sharp.
    exch_liq = 0.0
    for b in EXCHANGES:
        bd = market.get("books", {}).get(b)
        if bd:
            exch_liq = max(exch_liq, float(bd.get("liq", 0.0) or 0.0))

    return ValueBet(
        market_id=str(market.get("market_id", "")),
        kind=str(market.get("kind", "moneyline")),
        side=side,
        my_book=my_book,
        my_price=float(my_price),
        fair_price=float(cons.fair_prices[side]) if math.isfinite(cons.fair_prices[side]) else float("nan"),
        fair_prob=float(cons.fair_probs[side]),
        edge_pct=float(edge_pct),
        edge_cents=float(edge_cents),
        plus_ev_vs_all=bool(plus_ev_vs_all),
        is_crossed=bool(market.get("is_crossed", False)),
        market_width=width,
        exchange_liq=float(exch_liq),
        per_book_fair=cons.per_book_fair,
    )


# =============================================================================
# 6. BET SIZING — half-Kelly × the Video-1 multiplier stack
# =============================================================================

DEFAULT_KELLY_FRACTION = 0.5          # sharp standard for variance control

# Market activity (Video 1: bigger on liquid markets, smaller on thin ones).
MARKET_ACTIVITY = {
    "mlb_ml":          1.5,
    "mlb_total":       1.4,
    "mlb_run_line":    1.2,
    "nba_ml":          1.6,
    "nfl_ml":          1.6,
    "nhl_ml":          1.2,
    "tennis_w":        0.5,
    "wnba":            0.7,
    "player_prop_mlb": 1.0,
    "player_prop_nba": 1.2,
    "default":         1.0,
}

# Book softness for SIZING (separate from the fair-price weighting above).
# Soft books we can press; sharp/exchange books we taper.
BOOK_SOFTNESS = {
    "FanDuel":    1.25,
    "DraftKings": 1.20,
    "BetMGM":     1.15,
    "Caesars":    1.10,
    "BetRivers":  1.05,
    "Bet365":     1.00,
    "Pinnacle":   0.60,
    "Circa":      0.65,
    "BetOnline":  0.70,
    "Novig":      0.45,
    "ProphetX":   0.45,
}


def kelly_stake(prob: float, american_odds: float, bankroll: float,
                fraction: float = DEFAULT_KELLY_FRACTION) -> float:
    """Fractional-Kelly stake in dollars. Returns 0 if edge non-positive."""
    if not (0.0 < prob < 1.0):
        return 0.0
    b = (american_odds / 100.0) if american_odds >= 0 else (100.0 / abs(american_odds))
    edge = prob * (b + 1.0) - 1.0
    if edge <= 0:
        return 0.0
    kelly = edge / b
    return float(max(0.0, kelly * fraction * bankroll))


def market_activity_multiplier(market_type_tag: str | None) -> float:
    return MARKET_ACTIVITY.get(market_type_tag or "default", MARKET_ACTIVITY["default"])


def market_width_multiplier(width_cents: float) -> float:
    """Tighter market = sharper price = more confidence in our edge."""
    if not math.isfinite(width_cents):
        return 1.0
    if width_cents <=  8: return 1.5
    if width_cents <= 15: return 1.2
    if width_cents <= 25: return 1.0
    if width_cents <= 40: return 0.7
    return 0.5


def liquidity_multiplier(exch_liq: float) -> float:
    """More $$ on an exchange opposite our side = sharper counterparty
    confirming our price. Modest bump, capped."""
    if exch_liq <= 0:    return 1.0
    if exch_liq <  500:  return 1.05
    if exch_liq < 2000:  return 1.15
    if exch_liq < 5000:  return 1.25
    return 1.35


def recommended_stake(value: ValueBet, bankroll: float,
                      market_type_tag: str | None = None,
                      max_multiplier: float = 4.0,
                      min_multiplier: float = 0.25) -> dict:
    """Compose all Video-1 conviction multipliers on top of half-Kelly."""
    base = kelly_stake(value.fair_prob, value.my_price, bankroll)
    if base <= 0:
        return {"base_kelly": 0.0, "total_multiplier": 0.0, "stake": 0.0, "factors": {}}

    factors = {
        "confidence": 2.0 if value.plus_ev_vs_all else 1.0,
        "activity":   market_activity_multiplier(market_type_tag),
        "crossed":    1.5 if value.is_crossed else 1.0,
        "softness":   BOOK_SOFTNESS.get(value.my_book, 1.0),
        "width":      market_width_multiplier(value.market_width),
        "liquidity":  liquidity_multiplier(value.exchange_liq),
    }
    total = 1.0
    for v in factors.values():
        total *= float(v)
    total = float(max(min_multiplier, min(max_multiplier, total)))

    return {
        "base_kelly": round(base, 2),
        "total_multiplier": round(total, 3),
        "stake": round(base * total, 2),
        "factors": {k: round(float(v), 3) for k, v in factors.items()},
    }


# =============================================================================
# 7. SLATE PIPELINE
# =============================================================================

def score_slate(markets: list[dict], my_book: str, bankroll: float,
                min_edge_pct: float = 0.02,
                max_multiplier: float = 4.0) -> pd.DataFrame:
    """Run consensus + EV + sizing over a slate. Returns one row per +EV side
    offered at `my_book`, sorted by recommended stake descending."""
    rows: list[dict] = []
    for m in markets:
        my_prices = (m.get("books") or {}).get(my_book) or {}
        sides = m.get("sides", [])
        for side in sides:
            price = my_prices.get(side)
            if price is None:
                continue
            v = detect_value(m, my_book=my_book, my_price=price, side=side,
                             min_edge_pct=min_edge_pct)
            if v is None:
                continue
            # Hard cap: anything claiming a > 50% edge is upstream data error.
            if v.edge_pct > MAX_PLAUSIBLE_EDGE:
                continue
            # Need a real consensus, not a one-book "average".
            n_books = len(v.per_book_fair)
            if n_books < MIN_BOOKS_REQUIRED:
                continue
            stake = recommended_stake(v, bankroll,
                                      market_type_tag=m.get("market_type_tag"),
                                      max_multiplier=max_multiplier)
            # 20-50% edge -> flag for manual review (rare but legit on soft books)
            is_suspicious = v.edge_pct > SUSPICIOUS_EDGE
            rows.append({
                "market_id":      v.market_id,
                "kind":           v.kind,
                "side":           v.side,
                "my_book":        v.my_book,
                "my_price":       v.my_price,
                "fair_price":     round(v.fair_price, 1) if math.isfinite(v.fair_price) else None,
                "fair_prob":      round(v.fair_prob, 4),
                "edge_pct":       round(v.edge_pct, 4),
                "edge_cents":     round(v.edge_cents, 1) if math.isfinite(v.edge_cents) else None,
                "plus_ev_vs_all":      v.plus_ev_vs_all,
                "is_suspicious":       is_suspicious,
                "n_books_in_consensus":n_books,
                "is_crossed":          v.is_crossed,
                "market_width":        round(v.market_width, 1) if math.isfinite(v.market_width) else None,
                "exchange_liq":        round(v.exchange_liq, 0),
                "base_kelly":          stake["base_kelly"],
                "multiplier":     stake["total_multiplier"],
                "stake":          stake["stake"],
                **{f"x_{k}": v_ for k, v_ in stake["factors"].items()},
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("stake", ascending=False).reset_index(drop=True)


# =============================================================================
# 8. DEMO — runnable as a script
# =============================================================================

def _demo() -> None:
    sample = [
        {
            "market_id": "MLB:gp99999:moneyline",
            "kind": "moneyline",
            "sides": ["away", "home"],
            "market_type_tag": "mlb_ml",
            "books": {
                "Pinnacle":  {"away": +122, "home": -138, "limit":  500},
                "Circa":     {"away": +125, "home": -145, "limit": 1000},
                "BetOnline": {"away": +120, "home": -140, "limit":  500},
                "FanDuel":   {"away": +136, "home": -160, "limit": 1000},
                "DraftKings":{"away": +130, "home": -150, "limit": 1000},
                "Novig":     {"away": +132, "home": -134, "liq": 5600},
            },
            "is_crossed": False,
        },
        {
            "market_id": "MLB:gp99999:K_alcantara_ov_6.5",
            "kind": "player_prop",
            "sides": ["over", "under"],
            "market_type_tag": "player_prop_mlb",
            "books": {
                "Pinnacle":   {"over": -115, "under": -105},
                "FanDuel":    {"over": -105, "under": -115},
                "DraftKings": {"over": -110, "under": -110},
                "Caesars":    {"over": -108, "under": -112},
            },
        },
    ]
    df = score_slate(sample, my_book="FanDuel", bankroll=10_000, min_edge_pct=0.0)
    if df.empty:
        print("No +EV bets at FanDuel in the demo slate.")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    _demo()
