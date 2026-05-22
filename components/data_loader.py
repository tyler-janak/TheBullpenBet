"""
components/data_loader.py
=========================
Shared helpers for loading predictions, projections, and schedule data.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Helpers ───────────────────────────────────────────────────────────────────

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def fmt_odds(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    val = float(val)
    return f"+{int(val)}" if val > 0 else str(int(val))


def fmt_pct(val, decimals=1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def fmt_ev(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}%"


def fmt_time(ts_str) -> str:
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        eastern = dt.astimezone(ZoneInfo("America/New_York"))
        return eastern.strftime("%-I:%M %p ET")
    except Exception:
        return str(ts_str)


def team_key(home: str, away: str) -> str:
    return f"{away.upper()}@{home.upper()}"


def normalize_name(name) -> str:
    if name is None or pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── MLB Stats API ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_schedule(date: str) -> list[dict]:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher,team,linescore"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"Could not fetch schedule: {e}")
        return []

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})

            home_abbr = home.get("team", {}).get("abbreviation", "")
            away_abbr = away.get("team", {}).get("abbreviation", "")

            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            games.append({
                "game_pk": g.get("gamePk"),
                "commence_time": g.get("gameDate"),
                "home": home_abbr,
                "away": away_abbr,
                "home_name": home.get("team", {}).get("name", home_abbr),
                "away_name": away.get("team", {}).get("name", away_abbr),
                "homeStarter": home_pitcher.get("fullName", "TBD"),
                "awayStarter": away_pitcher.get("fullName", "TBD"),
                "home_starter_id": home_pitcher.get("id"),
                "away_starter_id": away_pitcher.get("id"),
                "status": g.get("status", {}).get("abstractGameState", ""),
            })
    return games


# ── Predictions CSV ────────────────────────────────────────────────────────────

def load_predictions(preds_path: Path) -> list[dict]:
    if not preds_path.exists():
        return []
    try:
        df = pd.read_csv(preds_path)
    except Exception:
        return []

    out = []
    for _, row in df.iterrows():
        home = str(row.get("home_team", "")).strip().upper()
        away = str(row.get("away_team", "")).strip().upper()
        home_prob = _safe_float(row.get("home_win_prob", 0.5))
        away_prob = _safe_float(row.get("away_win_prob", 0.5))
        if home_prob is None:
            home_prob = 0.5
        if away_prob is None:
            away_prob = 1 - home_prob

        predicted_winner_raw = str(row.get("predicted_winner", home)).strip().upper()
        winner_prob = home_prob if predicted_winner_raw == home else away_prob
        winner_odds = _safe_float(row.get("best_ev_odds") or row.get("home_ml" if predicted_winner_raw == home else "away_ml"))
        ev = _safe_float(row.get("best_ev"))
        home_ml = _safe_float(row.get("home_ml"))
        away_ml = _safe_float(row.get("away_ml"))
        bet_rec = bool(row.get("bet_recommended", False))

        out.append({
            "home": home,
            "away": away,
            "homeWinProb": home_prob,
            "awayWinProb": away_prob,
            "predictedWinner": predicted_winner_raw,
            "winnerProb": winner_prob,
            "winnerOdds": winner_odds,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "ev": ev,
            "betRecommended": bet_rec,
            "commenceTime": row.get("commence_time"),
        })
    return out


def _safe_float(val):
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except Exception:
        return None


# ── Projections CSV ────────────────────────────────────────────────────────────

def load_projections(proj_path: Path) -> pd.DataFrame:
    if not proj_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(proj_path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def get_pitcher_projs(proj_df: pd.DataFrame) -> pd.DataFrame:
    if proj_df.empty:
        return pd.DataFrame()
    return proj_df[proj_df["player_type"] == "pitcher"].copy()


def get_hitter_projs(proj_df: pd.DataFrame) -> pd.DataFrame:
    if proj_df.empty:
        return pd.DataFrame()
    return proj_df[proj_df["player_type"] == "hitter"].copy()


# ── Accuracy history ──────────────────────────────────────────────────────────

def load_accuracy(picks_file: Path) -> dict:
    empty = {"labels": [], "cumulative_accuracy": [], "overall_acc": None,
             "total_games": 0, "correct_games": 0}
    if not picks_file.exists():
        return empty

    try:
        df = pd.read_csv(picks_file, low_memory=False)
    except Exception:
        return empty

    if "actual_winner" not in df.columns or "predicted_winner" not in df.columns:
        return empty

    df = df.dropna(subset=["actual_winner", "predicted_winner"]).copy()
    if df.empty:
        return empty

    df["_date"] = pd.to_datetime(
        df.get("game_date", df.get("date", pd.NaT)), errors="coerce"
    )
    df = df.dropna(subset=["_date"]).sort_values("_date")
    df["_correct"] = df["predicted_winner"].str.strip().str.upper() == \
                     df["actual_winner"].str.strip().str.upper()

    df["running_correct"] = df["_correct"].astype(int).cumsum()
    df["game_number"] = range(1, len(df) + 1)
    df["cumulative_accuracy"] = df["running_correct"] / df["game_number"]

    total = len(df)
    correct = int(df["_correct"].astype(int).sum())
    acc = correct / total if total else None

    try:
        labels = df["_date"].dt.strftime("%-m/%-d").tolist()
    except Exception:
        labels = df["_date"].dt.strftime("%m/%d").tolist()

    return {
        "labels": labels,
        "cumulative_accuracy": df["cumulative_accuracy"].tolist(),
        "overall_acc": acc,
        "total_games": total,
        "correct_games": correct,
    }


import pandas as pd
df = pd.read_csv("outputs/hitterspitchers_today.csv")
h = df[df["player_type"] == "hitter"]
print(h[["player_name", "proj_pa", "proj_hits", "proj_hr", "proj_walks", "proj_strikeouts"]].head(10).to_string())