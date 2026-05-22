"""
nrfi_today.py
=============
Generate today's per-game NRFI probability predictions.

Steps
-----
1. Fetch today's MLB schedule (starters via MLB Stats API)
2. Load historical nrfi_game_data.csv to get each starter's rolling
   1st-inning feature snapshot (most recent game before today)
3. Load nrfi_model.pkl and predict NRFI probability for each game
4. Optionally pull FanDuel NRFI odds from the props long CSV
5. Save outputs/nrfi_today.csv

Usage
-----
    python nrfi_today.py
    python nrfi_today.py --date 2026-05-03
"""

import argparse
import pickle
import re
import unicodedata
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")
OUT_DIR   = Path("outputs")

ROLLING_WINDOWS      = [5, 10]
ROLLING_WINDOWS_FULL = [5, 10]

# Columns to pull from pitcher_game_data.csv for full-game features
PITCHER_GAME_STAT_COLS = [
    "K_rate_last5",  "K_rate_last10",  "K_rate_std",
    "BB_rate_last5", "BB_rate_last10", "BB_rate_std",
    "H_rate_last5",  "H_rate_last10",  "H_rate_std",
    "HR_rate_last5", "HR_rate_last10", "HR_rate_std",
    "IP_last5",      "IP_last10",      "IP_std",
    "days_rest",
]

TEAM_MAP = {
    "Arizona Diamondbacks": "AZ",  "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",         "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",      "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",     "Detroit Tigers": "DET",
    "Houston Astros": "HOU",       "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",   "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",        "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",      "New York Mets": "NYM",
    "New York Yankees": "NYY",     "Athletics": "ATH",
    "Philadelphia Phillies": "PHI","Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",      "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",     "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",        "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",    "Washington Nationals": "WSH",
    "Oakland Athletics": "ATH",
}


def team_to_abbr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    s = str(val).strip()
    return TEAM_MAP.get(s, s)


def normalize_name(name) -> str:
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── schedule ──────────────────────────────────────────────────────────────────

def fetch_schedule(target_date: str) -> list[dict]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={target_date}&hydrate=probablePitcher,team")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            away  = teams.get("away", {})
            home  = teams.get("home", {})
            away_t = away.get("team", {}) or {}
            home_t = home.get("team", {}) or {}
            away_p = away.get("probablePitcher", {}) or {}
            home_p = home.get("probablePitcher", {}) or {}

            games.append({
                "gamePk":           g.get("gamePk"),
                "game_date":        target_date,
                "away_team":        team_to_abbr(away_t.get("abbreviation")),
                "home_team":        team_to_abbr(home_t.get("abbreviation")),
                "away_full":        away_t.get("name"),
                "home_full":        home_t.get("name"),
                "away_sp_name":     away_p.get("fullName"),
                "away_sp_id":       away_p.get("id"),
                "home_sp_name":     home_p.get("fullName"),
                "home_sp_id":       home_p.get("id"),
            })
    return games


# ── feature lookup ────────────────────────────────────────────────────────────

def load_pitcher_game_df() -> pd.DataFrame:
    """Load pitcher_game_data.csv for full-game rolling stats (primary NRFI signal)."""
    path = DATA_DIR / "pitcher_game_data.csv"
    if not path.exists():
        return pd.DataFrame()
    cols_needed = ["pitcher", "game_date", "player_name", "pitcher_name", "team"] + PITCHER_GAME_STAT_COLS
    try:
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in header.columns if c in cols_needed]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        return df
    except Exception as e:
        print(f"  [warn] Could not load pitcher_game_data.csv: {e}")
        return pd.DataFrame()


def get_pitcher_fg_features(pitcher_game_df: pd.DataFrame,
                             target_date: pd.Timestamp,
                             sp_id=None, sp_name: str = "",
                             side_prefix: str = "away_sp_fg_") -> dict:
    """
    Look up full-game rolling stats for a pitcher from pitcher_game_data.csv.
    Returns dict with keys like 'away_sp_fg_K_rate_last5', etc.
    These are the highest-signal features for NRFI prediction.
    """
    if pitcher_game_df.empty:
        return {}

    tmp = pitcher_game_df[pitcher_game_df["game_date"] < target_date].copy()
    if tmp.empty:
        return {}

    row = pd.Series(dtype=object)

    # Match by pitcher ID
    if sp_id is not None and "pitcher" in tmp.columns:
        m = tmp[pd.to_numeric(tmp["pitcher"], errors="coerce") == float(sp_id)]
        if not m.empty:
            row = m.sort_values("game_date").iloc[-1]

    # Fall back to name match
    if row.empty and sp_name:
        norm = normalize_name(sp_name)
        for name_col in ["player_name", "pitcher_name"]:
            if name_col in tmp.columns:
                mask = tmp[name_col].fillna("").astype(str).apply(normalize_name) == norm
                if mask.any():
                    row = tmp[mask].sort_values("game_date").iloc[-1]
                    break

    if row.empty:
        return {}

    return {
        f"{side_prefix}{col}": row[col]
        for col in PITCHER_GAME_STAT_COLS
        if col in row.index and pd.notna(row[col])
    }


def get_sp_features(nrfi_df: pd.DataFrame,
                    target_date: pd.Timestamp,
                    sp_id=None, sp_name: str = "") -> pd.Series:
    """
    Return the most recent pre-game 1st-inning feature row for a starter.
    Tries pitcher ID first, then normalized name.
    """
    if nrfi_df.empty:
        return pd.Series(dtype=object)

    date_col = next((c for c in ["game_date", "date"] if c in nrfi_df.columns), None)
    if date_col is None:
        return pd.Series(dtype=object)

    tmp = nrfi_df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col] < target_date]
    if tmp.empty:
        return pd.Series(dtype=object)

    # Try by pitcher ID
    for id_col in ["away_sp_id", "home_sp_id"]:
        if id_col in tmp.columns and sp_id is not None:
            m = tmp[pd.to_numeric(tmp[id_col], errors="coerce") == float(sp_id)]
            if not m.empty:
                return m.sort_values(date_col).iloc[-1]

    # Try by name
    if sp_name:
        norm = normalize_name(sp_name)
        for name_col in ["away_sp_name", "home_sp_name",
                         "away_sp_pitcher_name", "home_sp_pitcher_name"]:
            if name_col in tmp.columns:
                mask = tmp[name_col].astype(str).apply(normalize_name) == norm
                if mask.any():
                    return tmp[mask].sort_values(date_col).iloc[-1]

    return pd.Series(dtype=object)


def get_team_bat_features(nrfi_df: pd.DataFrame,
                          target_date: pd.Timestamp,
                          team: str,
                          side: str) -> pd.Series:
    """
    Return most recent pre-game 1st-inning batting features for a team.
    side = 'away' or 'home'
    """
    prefix = f"{side}_bat_"
    if nrfi_df.empty:
        return pd.Series(dtype=object)

    date_col = next((c for c in ["game_date", "date"] if c in nrfi_df.columns), None)
    if date_col is None:
        return pd.Series(dtype=object)

    team_col = f"{side}_team"
    if team_col not in nrfi_df.columns:
        return pd.Series(dtype=object)

    tmp = nrfi_df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[(tmp[date_col] < target_date) & (tmp[team_col] == team)]
    if tmp.empty:
        return pd.Series(dtype=object)

    return tmp.sort_values(date_col).iloc[-1]


# ── feature assembly ──────────────────────────────────────────────────────────

def _prefix_rename(series: pd.Series, old_prefix: str, new_prefix: str) -> pd.Series:
    return series.rename(index={
        k: k.replace(old_prefix, new_prefix, 1)
        for k in series.index if k.startswith(old_prefix)
    })


def build_game_feature_row(
    game: dict,
    nrfi_df: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    target_date: pd.Timestamp,
    model_features: list[str],
) -> pd.Series:
    """
    Assemble one feature row for a single game.
    Sources (in priority order):
      1. pitcher_game_data.csv — full-game rolling stats (highest signal)
      2. nrfi_game_data.csv   — first-inning specific stats + team batting
    """
    row = pd.Series(dtype=float)

    # ── Full-game pitcher stats (primary signal) ──────────────────────────────
    away_fg = get_pitcher_fg_features(
        pitcher_game_df, target_date,
        sp_id=game.get("away_sp_id"), sp_name=game.get("away_sp_name", ""),
        side_prefix="away_sp_fg_",
    )
    row = pd.concat([row, pd.Series(away_fg)])

    home_fg = get_pitcher_fg_features(
        pitcher_game_df, target_date,
        sp_id=game.get("home_sp_id"), sp_name=game.get("home_sp_name", ""),
        side_prefix="home_sp_fg_",
    )
    row = pd.concat([row, pd.Series(home_fg)])

    # ── 1st-inning pitcher stats (from nrfi_game_data) ────────────────────────
    away_feat = get_sp_features(nrfi_df, target_date,
                                sp_id=game.get("away_sp_id"),
                                sp_name=game.get("away_sp_name", ""))
    for col in [c for c in (away_feat.index if not away_feat.empty else [])
                if "away_sp_" in c and "fg_" not in c and (
                    c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS)
                )]:
        row[col] = away_feat[col]

    home_feat = get_sp_features(nrfi_df, target_date,
                                sp_id=game.get("home_sp_id"),
                                sp_name=game.get("home_sp_name", ""))
    for col in [c for c in (home_feat.index if not home_feat.empty else [])
                if "home_sp_" in c and "fg_" not in c and (
                    c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS)
                )]:
        row[col] = home_feat[col]

    # ── Team batting features (1st-inning and full-game from nrfi_game_data) ──
    away_bat = get_team_bat_features(nrfi_df, target_date, game["away_team"], "away")
    for col in [c for c in (away_bat.index if not away_bat.empty else [])
                if "away_bat_" in c]:
        row[col] = away_bat[col]

    home_bat = get_team_bat_features(nrfi_df, target_date, game["home_team"], "home")
    for col in [c for c in (home_bat.index if not home_bat.empty else [])
                if "home_bat_" in c]:
        row[col] = home_bat[col]

    # Park factor
    if not home_feat.empty and "park_factor" in home_feat.index:
        row["park_factor"] = home_feat["park_factor"]

    # Ensure all model features exist
    for f in model_features:
        if f not in row.index:
            row[f] = np.nan

    return row


# ── load FanDuel NRFI odds ────────────────────────────────────────────────────

def load_fd_nrfi_odds() -> pd.DataFrame:
    """
    Look for NRFI lines in the FanDuel props long CSV.
    FanDuel sometimes lists NRFI as a team-level market.
    Returns empty DataFrame if not available.
    """
    path = OUT_DIR / "fanduel_props_today_long.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "market" not in df.columns:
        return pd.DataFrame()
    nrfi_rows = df[df["market"].str.lower().str.contains("nrfi|first.inning", na=False)]
    return nrfi_rows


# ── main runner ───────────────────────────────────────────────────────────────

def run_nrfi(target_date: str | None = None) -> pd.DataFrame:
    target_date = target_date or str(date.today())
    target_ts   = pd.Timestamp(target_date)

    print(f"\nGenerating NRFI predictions for: {target_date}")

    # Load model
    model_path = MODEL_DIR / "nrfi_model.pkl"
    if not model_path.exists():
        print("  [error] nrfi_model.pkl not found. Run: python nrfi_train.py")
        return pd.DataFrame()

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    pipeline  = model_obj["pipeline"]
    features  = model_obj["features"]
    threshold = model_obj.get("threshold", 0.50)
    print(f"  Model loaded | {len(features)} features | threshold={threshold:.2f}")

    # Load historical NRFI data
    nrfi_path = DATA_DIR / "nrfi_game_data.csv"
    if not nrfi_path.exists():
        print("  [error] data/nrfi_game_data.csv not found. Run: python nrfi_data.py")
        return pd.DataFrame()
    nrfi_df = pd.read_csv(nrfi_path, low_memory=False)
    if "game_date" in nrfi_df.columns:
        nrfi_df["game_date"] = pd.to_datetime(nrfi_df["game_date"], errors="coerce")

    # Fetch today's schedule
    schedule = fetch_schedule(target_date)
    print(f"  Games found: {len(schedule)}")

    fd_odds = load_fd_nrfi_odds()

    # Load full-game pitcher stats (primary NRFI signal)
    pitcher_game_df = load_pitcher_game_df()
    if pitcher_game_df.empty:
        print("  [warn] pitcher_game_data.csv not found — full-game features unavailable.")
        print("         Run: python hitterspitchers_data.py --input <pitch_csv>")
    else:
        print(f"  Pitcher game data loaded: {len(pitcher_game_df):,} rows")

    rows = []
    for game in schedule:
        if not game.get("away_team") or not game.get("home_team"):
            continue

        feat_row = build_game_feature_row(game, nrfi_df, pitcher_game_df, target_ts, features)

        X = pd.DataFrame([feat_row[features].to_dict()])
        try:
            nrfi_prob = float(pipeline.predict_proba(X)[0, 1])
        except Exception as e:
            print(f"  [{game['away_team']} @ {game['home_team']}] prediction failed: {e}")
            nrfi_prob = np.nan

        yrfi_prob = 1.0 - nrfi_prob if pd.notna(nrfi_prob) else np.nan
        nrfi_pick = "NRFI" if (pd.notna(nrfi_prob) and nrfi_prob >= threshold) else "YRFI"

        out_row = {
            "game_date":     target_date,
            "gamePk":        game.get("gamePk"),
            "away_team":     game["away_team"],
            "home_team":     game["home_team"],
            "team_a":        game["away_team"],   # alias for nrfi_grade.py
            "team_b":        game["home_team"],   # alias for nrfi_grade.py
            "away_full":     game.get("away_full"),
            "home_full":     game.get("home_full"),
            "away_sp":       game.get("away_sp_name") or "TBD",
            "home_sp":       game.get("home_sp_name") or "TBD",
            "nrfi_prob":     round(nrfi_prob, 4) if pd.notna(nrfi_prob) else None,
            "yrfi_prob":     round(yrfi_prob, 4) if pd.notna(yrfi_prob) else None,
            "pick":          nrfi_pick,
            "lean":          "YES" if nrfi_pick == "NRFI" else "NO",  # for nrfi_grade.py
            "threshold":     threshold,
        }
        rows.append(out_row)

    if not rows:
        print("  No predictions generated.")
        return pd.DataFrame()

    results = pd.DataFrame(rows).sort_values("nrfi_prob", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "nrfi_today.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(results)} games)")

    print("\n── NRFI Predictions ─────────────────────────────────────────────")
    print(results[["away_team", "home_team", "away_sp", "home_sp",
                   "nrfi_prob", "pick"]].to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate today's NRFI predictions")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default=today)")
    args = parser.parse_args()
    run_nrfi(args.date)


if __name__ == "__main__":
    main()
