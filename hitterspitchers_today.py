"""
hitterspitchers_today.py
========================
Daily hitter + pitcher projection runner.

Run:
    python hitterspitchers_today.py
    python hitterspitchers_today.py --date 2026-04-15
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
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")

# Hybrid model inference helpers. Imports are wrapped so the module still
# loads if either trainer file is removed or has a syntax error — the
# legacy direct-target flow remains the default fallback inside score_*.
try:
    from train_pitcher_two_stage import predict_two_stage, load_two_stage_models
    _HAS_TWO_STAGE_PITCHER = True
except Exception as _e:  # pragma: no cover
    predict_two_stage = None       # type: ignore
    load_two_stage_models = None   # type: ignore
    _HAS_TWO_STAGE_PITCHER = False

try:
    from train_hitter_team_pa import predict_hitter_via_team_pa, load_team_pa_models
    _HAS_TEAM_PA_HITTER = True
except Exception as _e:  # pragma: no cover
    predict_hitter_via_team_pa = None   # type: ignore
    load_team_pa_models = None          # type: ignore
    _HAS_TEAM_PA_HITTER = False

# Context-layer modules (umpire / weather / xhits / market). Each is opt-in:
# if its data isn't present (no umpire CSV, no weather pull, no xH model,
# no props_log), the apply step is a no-op and the legacy projection flows
# through unchanged.
try:
    from umpire_features import apply_umpire_k_factor as _apply_umpire
except Exception:
    _apply_umpire = None
try:
    from weather_features import (
        fetch_today_weather as _fetch_weather,
        compute_factors as _weather_factors,
        apply_weather_factors as _apply_weather,
    )
except Exception:
    _fetch_weather = _weather_factors = _apply_weather = None
try:
    from train_pitcher_xhits import load_xhits_model as _load_xh, predict_xhits as _predict_xh
except Exception:
    _load_xh = _predict_xh = None
try:
    from market_calibration import (
        load_market_priors as _load_market_priors,
        apply_market_calibration as _apply_market,
    )
except Exception:
    _load_market_priors = _apply_market = None

# Preferred: count-stat models trained by hitterspitchers_train.py
# Fallback:  legacy rate models (used until retraining is done)
PITCHER_TARGETS = ["K", "BB", "HR", "H", "IP"]

HITTER_TARGETS  = ["H", "HR", "BB", "K", "PA"]

TEAM_MAP = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    "Oakland Athletics": "ATH",
}

ALL_TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH", "AZ"
}

POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}

CONFIG = {
    "pitcher_models": PITCHER_TARGETS,
    "hitter_models": HITTER_TARGETS,
    "park_shrink_hits": 0.20,
    "park_shrink_hr": 0.30,
    "ip_floor": 3.0,
    "ip_ceiling": 8.5,
}

def safe_predict(model, feat):
    try:
        return predict_model(model, feat)
    except:
        return None

def run_engine(target_date=None, debug=False):
    return run_projections(target_date)

def save_output(df, path=OUT_DIR / "hitterspitchers_today.csv"):
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return path



def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_name(name) -> str:
    if name is None or pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # Strip name suffixes — intentionally excludes "v" because it is far too
    # common as an abbreviated first initial (V. Guerrero, V. Pasquantino, etc.)
    # and would silently collapse "V. Lastname" to just "Lastname", breaking
    # the two-token flexible matcher.
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_token_set(name) -> set[str]:
    return {tok for tok in normalize_name(name).split() if tok}



def names_match_flexible(a: str, b: str) -> bool:
    if not a or not b:
        return False

    ta = normalize_name(a).split()
    tb = normalize_name(b).split()

    if not ta or not tb:
        return False

    if ta == tb:
        return True

    # Standard abbreviated first name: "C. Correa" ↔ "Carlos Correa"
    # Also catches single-letter initials: "V. Guerrero" ↔ "Vladimir Guerrero"
    if len(ta) >= 2 and len(tb) >= 2:
        if ta[-1] == tb[-1] and ta[0][0] == tb[0][0]:
            return True

    # One side is just a last name (e.g. after stripping or scraping artifact)
    # and it matches the other's last name, with matching first initial.
    if len(ta) == 1 and len(tb) >= 2:
        if ta[0] == tb[-1]:
            return True
    if len(tb) == 1 and len(ta) >= 2:
        if tb[0] == ta[-1]:
            return True

    return False


def name_match_score(a, b) -> tuple[int, int, int]:
    """Return a comparable score tuple for fuzzy name matching.

    The tuple is `(last_name_match, common_token_count, first_initial_match)`
    so callers can do `score >= (1, 2, 1)` for "strict" or `>= (1, 2, 0)` for
    "lenient last+token-overlap" matches, matching the V2 scorer that the V3
    overhaul referenced but did not redefine.
    """
    ta = normalize_name(a).split()
    tb = normalize_name(b).split()
    if not ta or not tb:
        return (0, 0, 0)

    last_match = 1 if ta[-1] == tb[-1] else 0

    set_a = {tok for tok in ta if tok}
    set_b = {tok for tok in tb if tok}
    common = len(set_a & set_b)

    initial_match = 1 if (ta[0] and tb[0] and ta[0][0] == tb[0][0]) else 0

    return (last_match, common, initial_match)


def hitter_has_any_history(
    df: pd.DataFrame,
    id_col: str | None,
    name_col: str | None,
    *,
    player_id=None,
    player_name=None,
) -> bool:
    """True if the hitter game-log has at least one row for this player.

    Tries id match first, then exact normalized name, then the flexible
    abbreviation matcher. Used by `score_hitters` to decide whether to skip
    league-mean fallback projection for an unknown player.
    """
    if df is None or df.empty:
        return False

    if player_id is not None and id_col and id_col in df.columns:
        try:
            pid = int(player_id)
            if (pd.to_numeric(df[id_col], errors="coerce") == pid).any():
                return True
        except (TypeError, ValueError):
            pass

    if player_name and name_col and name_col in df.columns:
        target_norm = normalize_name(player_name)
        if target_norm:
            normed = df[name_col].astype(str).apply(normalize_name)
            if (normed == target_norm).any():
                return True
            if df[name_col].astype(str).apply(
                lambda x: names_match_flexible(player_name, x)
            ).any():
                return True

    return False


def team_to_abbr(value):
    if value is None or pd.isna(value):
        return value
    s = str(value).strip()
    if s in ALL_TEAM_ABBRS:
        return s
    return TEAM_MAP.get(s, s)



def safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def confidence_label(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def build_hitter_confidence_bundle(
    used_fallback: bool,
    used_starter_context: bool,
    lineup_status: str | None,
) -> dict:
    lineup_status = str(lineup_status or "").strip().lower()

    base = 88
    if used_fallback:
        base = 38
    else:
        if used_starter_context:
            base += 6
        else:
            base -= 8

    if "expected" in lineup_status:
        base -= 6
    elif "confirmed" in lineup_status:
        base += 2

    base = max(20, min(97, base))

    pa_score = base + 4
    hits_score = base
    hr_score = base - 8
    k_score = base - 3
    runs_score = base - 5
    rbi_score = base - 5
    hrrbi_score = base - 4
    bb_score = base - 2

    scores = {
        "confidence_score": base,
        "proj_pa_confidence_score": max(20, min(99, pa_score)),
        "proj_hits_confidence_score": max(20, min(99, hits_score)),
        "proj_hr_confidence_score": max(20, min(99, hr_score)),
        "proj_strikeouts_confidence_score": max(20, min(99, k_score)),
        "proj_runs_confidence_score": max(20, min(99, runs_score)),
        "proj_rbi_confidence_score": max(20, min(99, rbi_score)),
        "proj_hrrbi_confidence_score": max(20, min(99, hrrbi_score)),
        "proj_walks_confidence_score": max(20, min(99, bb_score)),
    }

    labels = {
        "confidence": confidence_label(scores["confidence_score"]),
        "proj_pa_confidence": confidence_label(scores["proj_pa_confidence_score"]),
        "proj_hits_confidence": confidence_label(scores["proj_hits_confidence_score"]),
        "proj_hr_confidence": confidence_label(scores["proj_hr_confidence_score"]),
        "proj_strikeouts_confidence": confidence_label(scores["proj_strikeouts_confidence_score"]),
        "proj_runs_confidence": confidence_label(scores["proj_runs_confidence_score"]),
        "proj_rbi_confidence": confidence_label(scores["proj_rbi_confidence_score"]),
        "proj_hrrbi_confidence": confidence_label(scores["proj_hrrbi_confidence_score"]),
        "proj_walks_confidence": confidence_label(scores["proj_walks_confidence_score"]),
    }

    return {**scores, **labels}


# --- target-transform helpers needed for loading transformed hitter pickles ---
def _clip_prob(y):
    y = np.asarray(y, dtype=float)
    return np.clip(y, 1e-6, 1 - 1e-6)


def _logit_transform(y):
    y = _clip_prob(y)
    return np.log(y / (1.0 - y))


def _logit_inverse(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _inv_logit_transform(z):
    return _logit_inverse(z)


def _log1p_transform(y):
    y = np.asarray(y, dtype=float)
    return np.log1p(np.clip(y, 0.0, None))


def _log1p_inverse(z):
    z = np.asarray(z, dtype=float)
    return np.expm1(z)


def _inv_log1p_transform(z):
    return _log1p_inverse(z)


def load_models_count_only(prefix: str, targets: list[str]) -> dict:
    models = {}

    for t in targets:
        path = MODEL_DIR / f"{prefix}_{t}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing required model: {path}")

        with open(path, "rb") as f:
            models[t] = pickle.load(f)

    print(f"[{prefix}] Loaded COUNT models: {targets}")
    return models


def predict_model(model_obj: dict, feature_row: pd.Series) -> float | None:
    pipeline = model_obj["pipeline"]
    features = model_obj["features"]
    if feature_row.empty:
        return None

    missing_cols = [c for c in features if c not in feature_row.index]
    if missing_cols:
        return None

    try:
        X = pd.DataFrame([feature_row[features].to_dict()])
        pred = float(pipeline.predict(X)[0])
        if pd.isna(pred):
            return None
        return pred
    except Exception:
        return None


def park_multiplier(factor_value, shrink=0.25):
    try:
        pf = float(factor_value)
    except Exception:
        pf = 100.0
    return 1.0 + shrink * ((pf / 100.0) - 1.0)


def infer_league_means(pitcher_game_df: pd.DataFrame, hitter_game_df: pd.DataFrame) -> dict:
    means = {}

    for stat in ["K_rate", "BB_rate", "HR_rate", "H_rate", "IP"]:
        if stat in pitcher_game_df.columns:
            means[f"pitcher_{stat}"] = float(pd.to_numeric(pitcher_game_df[stat], errors="coerce").mean())

    for stat in ["h_rate", "hr_rate", "bb_rate", "k_rate", "PA"]:
        if stat in hitter_game_df.columns:
            means[f"hitter_{stat}"] = float(pd.to_numeric(hitter_game_df[stat], errors="coerce").mean())

    defaults = {
        "pitcher_K_rate": 0.225,
        "pitcher_BB_rate": 0.085,
        "pitcher_HR_rate": 0.030,
        "pitcher_H_rate": 0.215,
        "pitcher_IP": 4.90,
        "hitter_h_rate": 0.205,
        "hitter_hr_rate": 0.032,
        "hitter_bb_rate": 0.082,
        "hitter_k_rate": 0.225,
        "hitter_PA": 3.95,
    }

    for k, v in defaults.items():
        if k not in means or pd.isna(means[k]):
            means[k] = v

    return means


def fetch_schedule(target_date: str) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={target_date}&hydrate=probablePitcher,team"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            away_team = away.get("team", {})
            home_team = home.get("team", {})
            away_pitcher = away.get("probablePitcher", {}) or {}
            home_pitcher = home.get("probablePitcher", {}) or {}

            games.append({
                "gamePk": g.get("gamePk"),
                "gameDate": g.get("gameDate"),
                "away_team_id": away_team.get("id"),
                "home_team_id": home_team.get("id"),
                "away": team_to_abbr(away_team.get("abbreviation")),
                "home": team_to_abbr(home_team.get("abbreviation")),
                "away_full": away_team.get("name"),
                "home_full": home_team.get("name"),
                "away_pitcher_name": away_pitcher.get("fullName"),
                "home_pitcher_name": home_pitcher.get("fullName"),
                "away_pitcher_id": away_pitcher.get("id"),
                "home_pitcher_id": home_pitcher.get("id"),
            })

    return sorted(games, key=lambda x: str(x.get("gameDate", "")))


def fetch_team_roster(team_id: int) -> list[dict]:
    if team_id is None:
        return []

    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    out = []
    for row in data.get("roster", []):
        person = row.get("person", {}) or {}
        out.append({
            "mlb_id": person.get("id"),
            "name": person.get("fullName"),
            "norm_name": normalize_name(person.get("fullName")),
        })
    return out


def build_roster_maps(schedule_games: list[dict]) -> dict:
    seen = {}
    for g in schedule_games:
        for team_abbr, team_id in [(g["away"], g["away_team_id"]), (g["home"], g["home_team_id"])]:
            if team_abbr not in seen:
                seen[team_abbr] = fetch_team_roster(team_id)
    return seen


def extract_text_lines_from_rotowire() -> list[str]:
    url = "https://www.rotowire.com/baseball/daily-lineups.php"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    return [x for x in lines if x]


def is_probable_name_line(line: str) -> bool:
    if not line:
        return False
    s = line.strip()

    if s in POS_SET:
        return False
    if s in ALL_TEAM_ABBRS:
        return False
    if s in {"Confirmed Lineup", "Expected Lineup", "Home Run Odds", "Starting Pitcher Intel"}:
        return False
    if re.fullmatch(r"[LRS]", s):
        return False
    if re.search(r"\bERA\b", s):
        return False
    if re.search(r"\d+-\d+", s):
        return False
    if "Precipitation" in s or "Wind" in s or "LINE " in s or "O/U " in s:
        return False
    if re.search(r"\bET\b", s):
        return False
    if "Umpire:" in s:
        return False

    return bool(re.search(r"[A-Za-z]", s))


def parse_lineup_block(lines: list[str], start_idx: int) -> tuple[list[dict], str]:
    status = lines[start_idx]
    players = []
    i = start_idx + 1

    while i < len(lines):
        line = lines[i].strip()

        if line in {"Home Run Odds", "Starting Pitcher Intel"} or line.startswith("Umpire:"):
            break

        if line in POS_SET:
            pos = line
            j = i + 1
            picked_name = None

            while j < len(lines):
                cand = lines[j].strip()
                if cand in POS_SET:
                    break
                if cand in {"Home Run Odds", "Starting Pitcher Intel"} or cand.startswith("Umpire:"):
                    break
                if is_probable_name_line(cand):
                    picked_name = cand
                    break
                j += 1

            if picked_name:
                players.append({
                    "lineup_spot": len(players) + 1,
                    "pos": pos,
                    "player_name": picked_name,
                    "lineup_status": status,
                })
                i = j + 1
                if len(players) >= 9:
                    break
                continue

        i += 1

    return players, status


def scrape_rotowire_lineups(schedule_games: list[dict]) -> pd.DataFrame:
    lines = extract_text_lines_from_rotowire()
    results = []
    cursor = 0

    for game in schedule_games:
        away = game["away"]
        home = game["home"]

        pair_idx = None
        for i in range(cursor, len(lines) - 1):
            if lines[i] == away and lines[i + 1] == home:
                pair_idx = i
                break

        if pair_idx is None:
            for i in range(0, len(lines) - 1):
                if lines[i] == away and lines[i + 1] == home:
                    pair_idx = i
                    break

        if pair_idx is None:
            continue

        next_pair_idx = len(lines)
        for j in range(pair_idx + 2, len(lines) - 1):
            if lines[j] in ALL_TEAM_ABBRS and lines[j + 1] in ALL_TEAM_ABBRS:
                next_pair_idx = j
                break

        seg = lines[pair_idx:next_pair_idx]
        status_idxs = [k for k, x in enumerate(seg) if x in {"Confirmed Lineup", "Expected Lineup"}]

        if len(status_idxs) < 2:
            cursor = next_pair_idx
            continue

        away_players, _ = parse_lineup_block(seg, status_idxs[0])
        home_players, _ = parse_lineup_block(seg, status_idxs[1])

        for p in away_players:
            results.append({
                "player_type": "hitter",
                "team": away,
                "opponent": home,
                "lineup_spot": p["lineup_spot"],
                "pos": p["pos"],
                "lineup_status": p["lineup_status"],
                "player_name": p["player_name"],
            })

        for p in home_players:
            results.append({
                "player_type": "hitter",
                "team": home,
                "opponent": away,
                "lineup_spot": p["lineup_spot"],
                "pos": p["pos"],
                "lineup_status": p["lineup_status"],
                "player_name": p["player_name"],
            })

        cursor = next_pair_idx

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out["norm_name"] = out["player_name"].apply(normalize_name)
    return out.drop_duplicates(subset=["team", "player_name", "lineup_spot"])


def map_lineups_to_rosters(lineups: pd.DataFrame, roster_maps: dict) -> pd.DataFrame:
    if lineups.empty:
        return lineups.copy()

    rows = []
    for _, r in lineups.iterrows():
        team = r["team"]
        roster = roster_maps.get(team, [])
        norm_name = r["norm_name"]

        match_id = None
        match_name = r["player_name"]

        # 1) exact normalized roster match
        for pl in roster:
            if pl["norm_name"] == norm_name:
                match_id = pl["mlb_id"]
                match_name = pl["name"]
                break

        # 2) flexible abbreviation roster match
        if match_id is None:
            for pl in roster:
                if names_match_flexible(r["player_name"], pl["name"]):
                    match_id = pl["mlb_id"]
                    match_name = pl["name"]
                    break

        # 3) scored fallback
        if match_id is None:
            target_tokens = norm_name.split()
            best = None
            best_key = (-1, -1, -1)

            for pl in roster:
                key = name_match_score(norm_name, pl.get("norm_name", ""))
                if key > best_key:
                    best_key = key
                    best = pl

            if best is not None and best_key >= (1, 2, 1):
                match_id = best["mlb_id"]
                match_name = best["name"]
            elif best is not None and len(target_tokens) >= 2 and best_key >= (1, 2, 0):
                match_id = best["mlb_id"]
                match_name = best["name"]

        row = r.to_dict()
        row["mlb_id"] = match_id
        row["roster_name"] = match_name
        rows.append(row)

    return pd.DataFrame(rows)


def get_pitcher_id_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["pitcher_id", "mlb_id", "player_id", "pitcher"])


def get_pitcher_name_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["pitcher_name", "pitcher", "player_name", "name"])


def get_hitter_id_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["batter_id", "mlb_id", "player_id", "batter"])


def get_hitter_name_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["batter_name", "batter", "player_name", "name"])


def get_date_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["game_date", "date"])


def get_latest_player_row(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    id_col: str | None,
    name_col: str | None,
    player_id=None,
    player_name=None,
) -> pd.Series:
    date_col = get_date_col(df)
    if date_col is None:
        return pd.Series(dtype=object)

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col] < target_date].copy()

    if tmp.empty:
        return pd.Series(dtype=object)

    # 1) exact ID match
    if id_col is not None and player_id is not None and id_col in tmp.columns:
        try:
            tmp_id = pd.to_numeric(tmp[id_col], errors="coerce")
            pid = float(player_id)
            match = tmp[tmp_id == pid].sort_values(date_col)
            if not match.empty:
                return match.iloc[-1]
        except Exception:
            pass

    # 2) name-based match
    if name_col is not None and player_name and name_col in tmp.columns:
        tmp["_norm_name"] = tmp[name_col].astype(str).apply(normalize_name)
        target_norm = normalize_name(player_name)

        # 2a) exact normalized match
        match = tmp[tmp["_norm_name"] == target_norm].sort_values(date_col)
        if not match.empty:
            return match.iloc[-1]

        # 2b) flexible abbreviation match
        flex_mask = tmp[name_col].astype(str).apply(lambda x: names_match_flexible(player_name, x))
        flex = tmp[flex_mask].sort_values(date_col)
        if not flex.empty:
            return flex.iloc[-1]

        # 2c) scored fuzzy fallback
        scores = tmp[name_col].astype(str).apply(lambda x: name_match_score(player_name, x))
        if len(scores):
            best_idx = scores.idxmax()
            best_key = scores.loc[best_idx]
            if best_key >= (1, 2, 1):
                return tmp.loc[[best_idx]].sort_values(date_col).iloc[-1]

    return pd.Series(dtype=object)


def get_today_team_context_row(
    ctx_df: pd.DataFrame,
    team: str,
    target_date: pd.Timestamp,
    hand_value: str,
    hand_col: str,
) -> pd.Series:
    if ctx_df.empty or hand_value not in {"R", "L"}:
        return pd.Series(dtype=object)

    date_col = get_date_col(ctx_df)
    if date_col is None or "team" not in ctx_df.columns or hand_col not in ctx_df.columns:
        return pd.Series(dtype=object)

    tmp = ctx_df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[
        (tmp["team"].astype(str) == str(team))
        & (tmp[hand_col].astype(str) == str(hand_value))
        & (tmp[date_col] < target_date)
    ].sort_values(date_col)

    if tmp.empty:
        return pd.Series(dtype=object)

    return tmp.iloc[-1]


def apply_context_overwrite(base_row: pd.Series, ctx_row: pd.Series, prefixes: list[str]) -> pd.Series:
    row = base_row.copy()
    if ctx_row.empty:
        return row

    for col in ctx_row.index:
        if any(col.startswith(p) for p in prefixes):
            row[col] = ctx_row[col]
    return row


def get_today_probable_pitcher_features(
    pitchers_today: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    target_date: pd.Timestamp,
    opponent_team: str,
) -> pd.Series:
    if pitchers_today.empty or pitcher_game_df.empty:
        return pd.Series(dtype=object)

    row = pitchers_today[pitchers_today["team"].astype(str) == str(opponent_team)]
    if row.empty:
        return pd.Series(dtype=object)
    row = row.iloc[0]

    p_id_col = get_pitcher_id_col(pitcher_game_df)
    p_name_col = get_pitcher_name_col(pitcher_game_df)

    return get_latest_player_row(
        pitcher_game_df,
        target_date=target_date,
        id_col=p_id_col,
        name_col=p_name_col,
        player_id=row.get("mlb_id"),
        player_name=row.get("player_name"),
    )


def apply_today_probable_pitcher_context(base_row: pd.Series, starter_row: pd.Series) -> pd.Series:
    row = base_row.copy()
    if starter_row.empty:
        return row

    mapping = {
        "pitcher_hand": "opp_sp_hand",
        "p_throws": "opp_sp_hand",
        "K_rate": "opp_sp_k_rate",
        "BB_rate": "opp_sp_bb_rate",
        "HR_rate": "opp_sp_hr_rate",
        "H_rate": "opp_sp_h_rate",
        "IP": "opp_sp_ip",
        "K_rate_last5": "opp_sp_k_rate_last5",
        "BB_rate_last5": "opp_sp_bb_rate_last5",
        "HR_rate_last5": "opp_sp_hr_rate_last5",
        "H_rate_last5": "opp_sp_h_rate_last5",
        "IP_last5": "opp_sp_ip_last5",
        "K_rate_last10": "opp_sp_k_rate_last10",
        "BB_rate_last10": "opp_sp_bb_rate_last10",
        "HR_rate_last10": "opp_sp_hr_rate_last10",
        "H_rate_last10": "opp_sp_h_rate_last10",
        "IP_last10": "opp_sp_ip_last10",
        "K_rate_std": "opp_sp_k_rate_std",
        "BB_rate_std": "opp_sp_bb_rate_std",
        "HR_rate_std": "opp_sp_hr_rate_std",
        "H_rate_std": "opp_sp_h_rate_std",
        "IP_std": "opp_sp_ip_std",
    }
    for src_col, dst_col in mapping.items():
        if src_col in starter_row.index and pd.notna(starter_row.get(src_col)):
            row[dst_col] = starter_row.get(src_col)
    return row


def build_today_pitchers(schedule_games: list[dict]) -> pd.DataFrame:
    rows = []
    for g in schedule_games:
        if g.get("away_pitcher_name"):
            rows.append({
                "player_type": "pitcher",
                "team": g["away"],
                "opponent": g["home"],
                "player_name": g["away_pitcher_name"],
                "mlb_id": g.get("away_pitcher_id"),
                "lineup_spot": np.nan,
                "pos": np.nan,
                "lineup_status": "Probable Starter",
                "gamePk": g["gamePk"],
            })
        if g.get("home_pitcher_name"):
            rows.append({
                "player_type": "pitcher",
                "team": g["home"],
                "opponent": g["away"],
                "player_name": g["home_pitcher_name"],
                "mlb_id": g.get("home_pitcher_id"),
                "lineup_spot": np.nan,
                "pos": np.nan,
                "lineup_status": "Probable Starter",
                "gamePk": g["gamePk"],
            })
    return pd.DataFrame(rows)


def estimate_pitcher_bf(ip: float, h_rate: float, bb_rate: float) -> float:
    outs = ip * 3.0
    non_out_rate = float(h_rate + bb_rate)
    non_out_rate = min(max(non_out_rate, 0.08), 0.42)
    out_rate = max(0.50, 1.0 - non_out_rate)
    return outs / out_rate


def estimate_pitcher_runs(hits: float, walks: float, hr: float) -> float:
    non_hr_hits = max(0.0, hits - hr)
    runs = 0.34 * non_hr_hits + 0.30 * walks + 0.95 * hr
    return max(0.0, runs)


def estimate_hitter_runs_rbi(hits: float, hr: float, bb: float, lineup_spot: int, pa: float) -> tuple[float, float]:
    lineup_spot = lineup_spot if lineup_spot in range(1, 10) else 5
    top_half_bonus = 1.05 if lineup_spot <= 5 else 0.92
    runs = (0.38 * hits + 0.18 * bb + 0.55 * hr) * top_half_bonus
    rbi = (0.34 * hits + 0.75 * hr) * (1.08 if 3 <= lineup_spot <= 6 else 0.95)
    runs = min(max(runs, 0.0), pa)
    rbi = min(max(rbi, 0.0), pa)
    return runs, rbi


# ─────────────────────────────────────────────────────────────────────────
# RAW MODEL MODE
# ─────────────────────────────────────────────────────────────────────────
# When True, score_pitchers / score_hitters emit the model's raw prediction
# verbatim (only a >=0 clamp) — NO league-baseline blend, IP/K/walk floors,
# realistic caps, or park multipliers — and run_projections skips the
# weather/umpire/market context layers, and daily_update skips display
# calibration. diagnose_train_serve_gap.py showed the post-processing/serving
# stack was the entire source of the ~0.7 → ~2.0 train→live MAE gap, so we
# surface the model output directly and let the grader measure it.
RAW_MODEL_ONLY = True


def score_pitchers(
    pitchers_today: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    pitcher_models: dict,
    team_batting_ctx: pd.DataFrame,
    target_date: pd.Timestamp,
    league_means: dict,
    two_stage_models: dict | None = None,
) -> pd.DataFrame:
    """
    Score pitchers for the slate. If `two_stage_models` is provided (loaded
    via `train_pitcher_two_stage.load_two_stage_models()`), we use the
    two-stage IP + per-9-rate models for IP/K/BB/H (cuts compounding error
    from IP miss) and keep the legacy direct model for HR (the direct model
    happened to win on HR in holdout, so the hybrid keeps it).
    Falls back per-pitcher to the legacy direct flow if two-stage scoring
    raises for any reason.
    """
    if pitchers_today.empty:
        return pd.DataFrame()

    p_id_col = get_pitcher_id_col(pitcher_game_df)
    p_name_col = get_pitcher_name_col(pitcher_game_df)

    rows = []

    # ─────────────────────────────────────────────
    # LEAGUE BASELINES (stabilizers)
    # ─────────────────────────────────────────────
    lg_k_per_ip   = league_means.get("pitcher_k_per_ip", 0.85)
    lg_bb_per_ip  = league_means.get("pitcher_bb_per_ip", 0.30)
    lg_h_per_ip   = league_means.get("pitcher_h_per_ip", 1.05)
    lg_hr_per_ip  = league_means.get("pitcher_hr_per_ip", 0.12)
    lg_r_per_ip   = league_means.get("pitcher_r_per_ip", 0.45)

    for _, row in pitchers_today.iterrows():

        base_feat = get_latest_player_row(
            pitcher_game_df,
            target_date=target_date,
            id_col=p_id_col,
            name_col=p_name_col,
            player_id=row.get("mlb_id"),
            player_name=row.get("player_name"),
        )

        if base_feat.empty:
            continue

        if "IP_std" in base_feat.index:
            try:
                if pd.notna(base_feat["IP_std"]) and float(base_feat["IP_std"]) < 2.5:
                    continue
            except Exception:
                pass

        pitcher_hand = str(base_feat.get("pitcher_hand", "")).strip().upper()
        if pitcher_hand not in {"R", "L"}:
            pitcher_hand = str(base_feat.get("p_throws", "")).strip().upper()

        ctx_row = get_today_team_context_row(
            team_batting_ctx,
            team=row["opponent"],
            target_date=target_date,
            hand_value=pitcher_hand,
            hand_col="pitcher_hand_split",
        )

        feat = apply_context_overwrite(
            base_feat,
            ctx_row,
            prefixes=[
                "team_k_rate_vs_hand",
                "team_bb_rate_vs_hand",
                "team_hr_rate_vs_hand",
                "team_h_rate_vs_hand",
            ],
        )

        # ─────────────────────────────────────────────
        # INNINGS PITCHED (ANCHOR) + RAW COUNT TARGETS
        # ─────────────────────────────────────────────
        # Hybrid scoring: two-stage model for IP/K/BB/H (cuts the compounding
        # error from a noisy IP estimate), direct model for HR (won in
        # holdout MAE). Any failure in the two-stage path falls back to the
        # full legacy direct-target flow so we never silently drop a starter.
        used_two_stage = False
        if two_stage_models and "IP" in two_stage_models:
            try:
                ts = predict_two_stage(feat, two_stage_models)
                ip = float(ts.get("IP", float("nan")))
                if pd.isna(ip):
                    raise ValueError("two-stage returned NaN IP")
                k_raw  = float(ts.get("K",  float("nan")))
                bb_raw = float(ts.get("BB", float("nan")))
                h_raw  = float(ts.get("H",  float("nan")))
                if any(pd.isna(v) for v in (k_raw, bb_raw, h_raw)):
                    raise ValueError("two-stage returned NaN for a counting target")
                # HR keeps the legacy direct model (hybrid)
                hr_raw = predict_model(pitcher_models["HR"], feat)
                if hr_raw is None:
                    raise ValueError("legacy HR model returned None")
                hr_raw = float(hr_raw)
                used_two_stage = True
            except Exception as e:
                print(f"  [score_pitchers] two-stage failed for "
                      f"{row.get('player_name','?')}: {type(e).__name__}: {e} — using legacy")
                used_two_stage = False

        if not used_two_stage:
            ip = predict_model(pitcher_models["IP"], feat)
            if ip is None:
                continue
            ip = float(ip)
            k_raw  = predict_model(pitcher_models["K"], feat)
            bb_raw = predict_model(pitcher_models["BB"], feat)
            h_raw  = predict_model(pitcher_models["H"], feat)
            hr_raw = predict_model(pitcher_models["HR"], feat)
            if any(v is None for v in [k_raw, bb_raw, h_raw, hr_raw]):
                continue
            k_raw  = float(k_raw)
            bb_raw = float(bb_raw)
            h_raw  = float(h_raw)
            hr_raw = float(hr_raw)

        if RAW_MODEL_ONLY:
            # Emit the raw model predictions verbatim (only a >=0 sanity clamp).
            # No IP floor, league blend, K/walk floors, park multipliers, or
            # caps — the grader measures the model directly.
            ip = max(0.0, ip)
            proj_strikeouts   = max(0.0, k_raw)
            proj_walks        = max(0.0, bb_raw)
            proj_hits_allowed = max(0.0, h_raw)
            proj_hr_allowed   = max(0.0, hr_raw)
            proj_runs_allowed = max(0.0, estimate_pitcher_runs(
                hits=proj_hits_allowed, walks=proj_walks, hr=proj_hr_allowed))
        else:
            # IP floor: keep at 3.0 for both flows. The earlier 1.0 floor for the
            # two-stage model was supposed to capture opener / early-hook outings
            # but the two-stage model (trained on 684 games) doesn't yet have
            # enough signal on those short outings — letting the prediction go to
            # 1.0 IP just introduced large misses on real 5+ IP starts. Bring back
            # the conservative 3.0 floor; we can relax again once the two-stage
            # model has 2000+ training games.
            ip = min(max(ip, 3.0), 8.5)

            # ─────────────────────────────────────────────
            # LEAGUE EXPECTED BASE (IP * rate)
            # ─────────────────────────────────────────────
            k_base  = ip * lg_k_per_ip
            bb_base = ip * lg_bb_per_ip
            h_base  = ip * lg_h_per_ip
            hr_base = ip * lg_hr_per_ip

            # ─────────────────────────────────────────────
            # BLENDED PROJECTIONS (MODEL + BASELINE)
            # ─────────────────────────────────────────────
            # 60/40 model:baseline blend for both flows. The 85/15 split for the
            # two-stage model assumed it was sample-rich enough to project on its
            # own; in practice the May-2026 training set is too small for that.
            # Revisit after a full season of two-stage training data.
            model_w = 0.60
            base_w  = 0.40
            proj_strikeouts   = model_w * k_raw  + base_w * k_base
            proj_walks        = model_w * bb_raw + base_w * bb_base
            proj_hits_allowed = model_w * h_raw  + base_w * h_base
            proj_hr_allowed   = model_w * hr_raw + base_w * hr_base

            # ─────────────────────────────────────────────
            # STRIKEOUT SAFETY FLOOR — back to 5.5 K/9
            # ─────────────────────────────────────────────
            # The looser 3.5 K/9 floor for the two-stage model allowed projections
            # that were too low for the empirical distribution. League-min K/9 is
            # ~5.5 for starters going 4+ IP; we floor there as a backstop and let
            # the model still drive within that constraint.
            proj_strikeouts = max(proj_strikeouts, ip * 0.55)

            # ─────────────────────────────────────────────
            # PARK FACTOR
            # ─────────────────────────────────────────────
            park_mult_hits = park_multiplier(base_feat.get("park_factor", 100.0), shrink=0.20)
            park_mult_hr   = park_multiplier(base_feat.get("park_factor", 100.0), shrink=0.30)

            proj_hits_allowed *= park_mult_hits
            proj_hr_allowed   *= park_mult_hr

            # ─────────────────────────────────────────────
            # REALISTIC CAPS
            # ─────────────────────────────────────────────
            proj_strikeouts   = min(proj_strikeouts, 15.0)
            proj_walks        = min(proj_walks, 7.0)
            proj_hits_allowed = min(proj_hits_allowed, 12.0)
            proj_hr_allowed   = min(proj_hr_allowed, 3.0)

            # ─────────────────────────────────────────────
            # WALK FLOOR — enforced for both flows
            # ─────────────────────────────────────────────
            # The two-stage BB9 model is too noisy on the small May-2026 training
            # set to reliably project control-artist outliers correctly. Bring
            # back the floor for both flows; once we have ~2000+ pitcher games of
            # training data the model can carry walks on its own.
            if ip >= 4.5:
                proj_walks = max(proj_walks, 0.60)
            if ip >= 5.5:
                proj_walks = max(proj_walks, 0.80)

            # ─────────────────────────────────────────────
            # RUNS ALLOWED MODEL
            # ─────────────────────────────────────────────
            proj_runs_allowed = max(0.0, estimate_pitcher_runs(
                hits=proj_hits_allowed,
                walks=proj_walks,
                hr=proj_hr_allowed,
            ))

        rows.append({
            "player_type": "pitcher",
            "team": row["team"],
            "opponent": row["opponent"],
            "player_name": row["player_name"],
            "mlb_id": row.get("mlb_id"),

            "lineup_spot": np.nan,
            "pos": np.nan,
            "lineup_status": row.get("lineup_status"),

            "proj_pa": np.nan,
            "proj_ip": round(ip, 2),
            "proj_outs": int(round(ip * 3)),

            "proj_runs_allowed": round(proj_runs_allowed, 2),
            "proj_hits_allowed": round(proj_hits_allowed, 2),
            "proj_strikeouts": round(proj_strikeouts, 2),
            "proj_walks": round(proj_walks, 2),

            "proj_hits": np.nan,
            "proj_runs": np.nan,
            "proj_rbi": np.nan,
            "proj_hrrbi": np.nan,
            "proj_hr": np.nan,
        })

    df = pd.DataFrame(rows)
    # Apply park factors as a multiplicative adjustment — small but free lift
    # the model misses because it trains on each pitcher's career mix of
    # parks but inference has to score them in TODAY's specific park.
    # Skipped in RAW_MODEL_ONLY mode so the output is the model verbatim.
    if not RAW_MODEL_ONLY:
        try:
            from park_factors import apply_park_factors
            df = apply_park_factors(df, kind="pitcher")
        except Exception as e:
            print(f"  [park_factors] pitcher: skipped ({type(e).__name__}: {e})")
    return df


def build_hitter_fallback_row(
    hitter_game_df: pd.DataFrame,
    team: str,
    opponent: str,
    lineup_spot,
    target_date: pd.Timestamp,
    batter_hand: str,
    team_pitching_ctx: pd.DataFrame,
    league_means: dict,
) -> pd.Series:
    base = pd.Series(dtype=object)
    work = hitter_game_df.copy()

    date_col = get_date_col(work)
    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work[work[date_col] < target_date].copy()

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()

    pools = []

    if "team" in work.columns:
        pools.append(work[work["team"].astype(str) == str(team)].copy())

    if "opponent" in work.columns:
        pools.append(work[work["opponent"].astype(str) == str(opponent)].copy())

    if "team" in work.columns and "opponent" in work.columns:
        pools.append(
            work[
                (work["team"].astype(str) == str(team)) &
                (work["opponent"].astype(str) == str(opponent))
            ].copy()
        )

    if "batter_hand" in work.columns and batter_hand in {"R", "L"}:
        pools.append(work[work["batter_hand"].astype(str) == batter_hand].copy())

    chosen = None
    for pool in pools[::-1]:
        if not pool.empty and len(pool) >= 15:
            chosen = pool
            break

    if chosen is None:
        chosen = work

    if numeric_cols:
        med = chosen[numeric_cols].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
        for col, val in med.items():
            base[col] = val

    base["team"] = team
    base["opponent"] = opponent
    base["game_date"] = target_date

    if batter_hand in {"R", "L"}:
        base["batter_hand"] = batter_hand

    lineup_spot = safe_int(lineup_spot)
    lineup_pa_defaults = {
        1: 4.75, 2: 4.60, 3: 4.45, 4: 4.35, 5: 4.20,
        6: 4.00, 7: 3.85, 8: 3.70, 9: 3.55
    }
    pa_prior = lineup_pa_defaults.get(lineup_spot, float(league_means.get("hitter_PA", 3.95)))

    for col in ["PA_std", "PA_last5", "PA_last10"]:
        base[col] = pa_prior

    for stat, mean_key in [
        ("h_rate", "hitter_h_rate"),
        ("hr_rate", "hitter_hr_rate"),
        ("bb_rate", "hitter_bb_rate"),
        ("k_rate", "hitter_k_rate"),
    ]:
        fallback_val = float(league_means.get(mean_key))
        for suffix in ["_std", "_last5", "_last10"]:
            col = f"{stat}{suffix}"
            if col not in base or pd.isna(base[col]):
                base[col] = fallback_val

    ctx_row = get_today_team_context_row(
        team_pitching_ctx,
        team=opponent,
        target_date=target_date,
        hand_value=batter_hand if batter_hand in {"R", "L"} else "R",
        hand_col="batter_hand_split",
    )
    base = apply_context_overwrite(
        base,
        ctx_row,
        prefixes=[
            "team_allowed_k_rate_vs_hand",
            "team_allowed_bb_rate_vs_hand",
            "team_allowed_hr_rate_vs_hand",
            "team_allowed_h_rate_vs_hand",
        ],
    )
    return base


def sanitize_hitter_projection(
    pa: float,
    h_rate: float,
    hr_rate: float,
    bb_rate: float,
    k_rate: float,
    lineup_spot: int | None,
    base_feat: pd.Series,
    league_means: dict,
) -> tuple[float, float, float, float, float]:
    pa = float(pa) if pd.notna(pa) else float(league_means.get("hitter_PA", 3.95))
    h_rate = float(h_rate) if pd.notna(h_rate) else float(league_means.get("hitter_h_rate", 0.205))
    hr_rate = float(hr_rate) if pd.notna(hr_rate) else float(league_means.get("hitter_hr_rate", 0.032))
    bb_rate = float(bb_rate) if pd.notna(bb_rate) else float(league_means.get("hitter_bb_rate", 0.082))
    k_rate = float(k_rate) if pd.notna(k_rate) else float(league_means.get("hitter_k_rate", 0.225))

    pa_floor = 2.2 if (lineup_spot and 1 <= lineup_spot <= 9) else 1.0
    pa_ceiling = 5.15 if (lineup_spot and 1 <= lineup_spot <= 9) else 4.5
    pa = min(max(pa, pa_floor), pa_ceiling)

    h_rate = min(max(h_rate, 0.02), 0.40)
    hr_rate = min(max(hr_rate, 0.0), 0.10)
    bb_rate = min(max(bb_rate, 0.0), 0.24)
    k_rate = min(max(k_rate, 0.0), 0.50)

    hr_rate = min(hr_rate, h_rate * 0.45)

    on_base_budget = h_rate + bb_rate
    if on_base_budget > 0.56:
        scale = 0.56 / max(on_base_budget, 1e-9)
        h_rate *= scale
        bb_rate *= scale
        hr_rate = min(hr_rate, h_rate * 0.45)

    return pa, h_rate, hr_rate, bb_rate, k_rate


def score_hitters(
    hitters_today: pd.DataFrame,
    hitters_today_pitchers: pd.DataFrame,
    hitter_game_df: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    hitter_models: dict,
    team_pitching_ctx: pd.DataFrame,
    target_date: pd.Timestamp,
    league_means: dict,
    team_pa_bundle: dict | None = None,
    rate_models: dict | None = None,
) -> pd.DataFrame:
    """
    Score the hitter slate. If `team_pa_bundle` and `rate_models` are both
    provided (loaded via `train_hitter_team_pa.load_team_pa_models()`), we
    use the team-PA decomposition (cuts per-hitter PA noise by predicting
    team_PA first, then per-PA rates × lineup-spot share). Falls back
    per-hitter to the legacy direct-target flow if either is missing or
    raises during scoring.
    """
    if hitters_today.empty:
        return pd.DataFrame()

    h_id_col = get_hitter_id_col(hitter_game_df)
    h_name_col = get_hitter_name_col(hitter_game_df)
    date_col = get_date_col(hitter_game_df)

    rows = []
    hitter_game_df = hitter_game_df.copy()

    # Per-team team_PA cache so we only call the stage-1 model once per team
    # in the slate (it's a team-level prediction; running it nine times per
    # team adds nothing but a sklearn call per row).
    _team_pa_cache: dict[str, float] = {}

    if h_name_col is not None and h_name_col in hitter_game_df.columns:
        hitter_game_df["_norm_name_lookup"] = hitter_game_df[h_name_col].apply(normalize_name)
    else:
        hitter_game_df["_norm_name_lookup"] = ""

    # ─────────────────────────────────────────────
    # LEAGUE BASELINES (STABILITY LAYER)
    # ─────────────────────────────────────────────
    lg_h_per_pa  = league_means.get("hitter_h_rate", 0.205)
    lg_hr_per_pa = league_means.get("hitter_hr_rate", 0.032)
    lg_bb_per_pa = league_means.get("hitter_bb_rate", 0.082)
    lg_k_per_pa  = league_means.get("hitter_k_rate", 0.225)

    lg_r_per_pa  = league_means.get("hitter_r_rate", 0.12)
    lg_rbi_per_pa = league_means.get("hitter_rbi_rate", 0.10)

    for _, row in hitters_today.iterrows():

        used_fallback = False
        used_starter_context = False

        lookup_name = row.get("roster_name") or row.get("player_name")

        any_history = hitter_has_any_history(
            hitter_game_df,
            h_id_col,
            h_name_col,
            player_id=row.get("mlb_id"),
            player_name=lookup_name,
        )

        base_feat = get_latest_player_row(
            hitter_game_df,
            target_date=target_date,
            id_col=h_id_col,
            name_col=h_name_col,
            player_id=row.get("mlb_id"),
            player_name=lookup_name,
        )

        # ---- batter hand ----
        batter_hand = ""
        if not base_feat.empty:
            batter_hand = str(base_feat.get("batter_hand", "")).strip().upper()
            if batter_hand not in {"R", "L"}:
                batter_hand = str(base_feat.get("stand", "")).strip().upper()

        if batter_hand not in {"R", "L"} and date_col is not None:
            nm = normalize_name(lookup_name)
            hand_match = hitter_game_df[hitter_game_df["_norm_name_lookup"] == nm].copy()
            if not hand_match.empty:
                hand_match = hand_match.sort_values(date_col)
                batter_hand = str(hand_match.iloc[-1].get("batter_hand", "")).strip().upper()

        # ---- fallback ----
        if base_feat.empty:
            used_fallback = True
            fallback_hand = batter_hand if batter_hand in {"R", "L"} else "R"

            base_feat = build_hitter_fallback_row(
                hitter_game_df=hitter_game_df.drop(columns=["_norm_name_lookup"], errors="ignore"),
                team=row["team"],
                opponent=row["opponent"],
                lineup_spot=row.get("lineup_spot"),
                target_date=target_date,
                batter_hand=fallback_hand,
                team_pitching_ctx=team_pitching_ctx,
                league_means=league_means,
            )
            batter_hand = fallback_hand

        hand_for_ctx = batter_hand if batter_hand in {"R", "L"} else "R"

        ctx_row = get_today_team_context_row(
            team_pitching_ctx,
            team=row["opponent"],
            target_date=target_date,
            hand_value=hand_for_ctx,
            hand_col="batter_hand_split",
        )

        feat = apply_context_overwrite(
            base_feat,
            ctx_row,
            prefixes=[
                "team_allowed_k_rate_vs_hand",
                "team_allowed_bb_rate_vs_hand",
                "team_allowed_hr_rate_vs_hand",
                "team_allowed_h_rate_vs_hand",
            ],
        )

        starter_row = get_today_probable_pitcher_features(
            hitters_today_pitchers,
            pitcher_game_df,
            target_date,
            row["opponent"],
        )

        if not starter_row.empty:
            used_starter_context = True

        feat = apply_today_probable_pitcher_context(feat, starter_row)

        lineup_spot = safe_int(row.get("lineup_spot"))

        # ─────────────────────────────────────────────
        # PA MODEL (ANCHOR) + RAW COUNT TARGETS
        # ─────────────────────────────────────────────
        # Hybrid scoring: if team-PA decomposition models are available,
        # predict team_PA once per team (cached) and decompose into
        # per-hitter PA via lineup-spot share × per-PA rates. Falls back
        # to the legacy direct-target flow if anything errors.
        used_team_pa = False
        if team_pa_bundle is not None and rate_models:
            try:
                team_key = str(row.get("team", ""))
                if team_key not in _team_pa_cache:
                    tp_feats = team_pa_bundle["features"]
                    Xt = pd.DataFrame([feat]).reindex(columns=tp_feats)
                    tp_pred = float(team_pa_bundle["model"].predict(Xt)[0])
                    # Sanity-clip team_PA to physical range: a team scores
                    # 30-48 PA per 9-inning game across the entire MLB
                    # distribution; outside this range is a feature artifact.
                    tp_pred = min(max(tp_pred, 30.0), 48.0)
                    _team_pa_cache[team_key] = tp_pred
                team_pa_predicted = _team_pa_cache[team_key]

                result = predict_hitter_via_team_pa(
                    feat,
                    lineup_spot=int(lineup_spot) if lineup_spot else 9,
                    team_pa_predicted=team_pa_predicted,
                    rate_models=rate_models,
                )
                pa = float(result.get("PA", float("nan")))
                if pd.isna(pa):
                    raise ValueError("team-PA returned NaN PA")
                # Map rate-model outputs into the local var names the rest
                # of the function uses. Missing rates fall back to None so
                # the league-baseline blend below can stabilise them.
                h_raw  = float(result["H"])  if "H"  in result else None
                hr_raw = float(result["HR"]) if "HR" in result else None
                bb_raw = float(result["BB"]) if "BB" in result else None
                k_raw  = float(result["K"])  if "K"  in result else None
                used_team_pa = True
            except Exception as e:
                print(f"  [score_hitters] team-PA failed for "
                      f"{lookup_name}: {type(e).__name__}: {e} — using legacy")
                used_team_pa = False

        if not used_team_pa:
            pa = predict_model(hitter_models["PA"], feat)
            pa = float(pa) if (pa is not None and pd.notna(pa)) else float(league_means.get("hitter_PA", 3.95))
            h_raw  = predict_model(hitter_models["H"], feat)
            hr_raw = predict_model(hitter_models["HR"], feat)
            bb_raw = predict_model(hitter_models["BB"], feat)
            k_raw  = predict_model(hitter_models["K"], feat)

        if RAW_MODEL_ONLY:
            # Emit raw model predictions verbatim (only a >=0 clamp); fall back
            # to the league base only when a model returned None. No PA
            # floor/ceiling, league blend, floors, or caps.
            pa   = max(0.0, pa)
            hits = max(0.0, float(h_raw))  if h_raw  is not None else pa * lg_h_per_pa
            hr   = max(0.0, float(hr_raw)) if hr_raw is not None else pa * lg_hr_per_pa
            bb   = max(0.0, float(bb_raw)) if bb_raw is not None else pa * lg_bb_per_pa
            k    = max(0.0, float(k_raw))  if k_raw  is not None else pa * lg_k_per_pa
        else:
            pa_floor   = 2.2 if (lineup_spot and 1 <= lineup_spot <= 9) else 1.0
            pa_ceiling = 5.0 if (lineup_spot and 1 <= lineup_spot <= 9) else 4.5
            pa = min(max(pa, pa_floor), pa_ceiling)

            # ─────────────────────────────────────────────
            # LEAGUE EXPECTED BASES
            # ─────────────────────────────────────────────
            h_base  = pa * lg_h_per_pa
            hr_base = pa * lg_hr_per_pa
            bb_base = pa * lg_bb_per_pa
            k_base  = pa * lg_k_per_pa

            # ─────────────────────────────────────────────
            # BLENDED PROJECTIONS
            # ─────────────────────────────────────────────
            hits = 0.60 * (float(h_raw)  if h_raw  is not None else h_base)  + 0.40 * h_base
            hr   = 0.60 * (float(hr_raw) if hr_raw is not None else hr_base) + 0.40 * hr_base
            bb   = 0.60 * (float(bb_raw) if bb_raw is not None else bb_base) + 0.40 * bb_base
            k    = 0.60 * (float(k_raw)  if k_raw  is not None else k_base)  + 0.40 * k_base

            # ─────────────────────────────────────────────
            # HARD FLOOR (PREVENT UNDERPROJECTION)
            # ─────────────────────────────────────────────
            hits = max(hits, pa * 0.12)
            hr   = max(hr, pa * 0.01)
            bb   = max(bb, pa * 0.03)
            k    = max(k, pa * 0.10)

            # ─────────────────────────────────────────────
            # REALISTIC CAPS
            # ─────────────────────────────────────────────
            hits = min(hits, pa * 0.42, 2.2)
            hr   = min(hr, hits * 0.40, 0.55)
            bb   = min(bb, pa * 0.28, 1.5)
            k    = min(k, pa * 0.65, 3.5)

        # ─────────────────────────────────────────────
        # RUNS / RBI
        # ─────────────────────────────────────────────
        runs, rbi = estimate_hitter_runs_rbi(
            hits=hits,
            hr=hr,
            bb=bb,
            lineup_spot=(lineup_spot or 5),
            pa=pa,
        )

        if used_fallback and not any_history:
            used_starter_context = False

        conf = build_hitter_confidence_bundle(
            used_fallback=used_fallback,
            used_starter_context=used_starter_context,
            lineup_status=row.get("lineup_status"),
        )

        rows.append({
            "player_type": "hitter",
            "team": row["team"],
            "opponent": row["opponent"],
            "player_name": lookup_name,
            "mlb_id": row.get("mlb_id"),
            "lineup_spot": lineup_spot,
            "pos": row.get("pos"),
            "lineup_status": row.get("lineup_status"),

            "proj_pa": round(pa, 2),
            "proj_ip": np.nan,
            "proj_outs": np.nan,

            "proj_runs_allowed": np.nan,
            "proj_hits_allowed": np.nan,

            "proj_strikeouts": round(k, 2),
            "proj_walks": round(bb, 2),
            "proj_hits": round(hits, 2),
            "proj_runs": round(runs, 2),
            "proj_rbi": round(rbi, 2),
            "proj_hrrbi": round(hits + runs + rbi, 2),
            "proj_hr": round(hr, 2),

            "used_fallback": used_fallback,
            **conf,
        })

    df = pd.DataFrame(rows)
    # Apply park factors as a multiplicative adjustment — Coors gives a real
    # +10% boost to hits/HR that the per-hitter model can't see at inference.
    # Skipped in RAW_MODEL_ONLY mode so the output is the model verbatim.
    if not RAW_MODEL_ONLY:
        try:
            from park_factors import apply_park_factors
            df = apply_park_factors(df, kind="hitter")
        except Exception as e:
            print(f"  [park_factors] hitter: skipped ({type(e).__name__}: {e})")
    return df


def load_fanduel_props() -> pd.DataFrame:
    """
    Load the wide-format FanDuel props CSV produced by fanduel_props.py.
    Returns empty DataFrame if file doesn't exist.
    """
    path = OUT_DIR / "fanduel_props_today.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "norm_name" not in df.columns and "player_name" in df.columns:
        df["norm_name"] = df["player_name"].apply(normalize_name)
    return df


def merge_fanduel_lines(proj_df: pd.DataFrame, props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join FanDuel prop lines onto projections by normalized player name.
    Adds fd_* columns without touching any existing projection columns.
    """
    if proj_df.empty or props_df.empty:
        return proj_df

    if "norm_name" not in proj_df.columns:
        proj_df = proj_df.copy()
        proj_df["norm_name"] = proj_df["player_name"].apply(normalize_name)

    fd_cols = [c for c in props_df.columns if c.startswith("fd_") or c == "norm_name"]
    props_slim = props_df[fd_cols].drop_duplicates(subset=["norm_name"])

    merged = proj_df.merge(props_slim, on="norm_name", how="left")
    # Drop helper column if it wasn't there before
    if "norm_name" not in proj_df.columns:
        merged = merged.drop(columns=["norm_name"], errors="ignore")
    return merged


def run_projections(target_date: str | None = None) -> pd.DataFrame:
    target_ts = pd.Timestamp(target_date) if target_date else pd.Timestamp(date.today())
    target_str = str(target_ts.date())

    print(f"\nGenerating projections for: {target_str}")

    pitcher_game_df = pd.read_csv(DATA_DIR / "pitcher_game_data.csv", low_memory=False)
    hitter_game_df = pd.read_csv(DATA_DIR / "hitter_game_data.csv", low_memory=False)
    team_batting_ctx = pd.read_csv(DATA_DIR / "team_batting_hand_context.csv", low_memory=False)
    team_pitching_ctx = pd.read_csv(DATA_DIR / "team_pitching_hand_context.csv", low_memory=False)

    # validate_pitcher_projection_data() was removed in the V3 overhaul; the
    # subsequent feature builders + score_* functions handle missing-data
    # cases themselves, so the standalone validator is no longer required.
    league_means = infer_league_means(pitcher_game_df, hitter_game_df)

    pitcher_models = load_models_count_only("pitcher", PITCHER_TARGETS)
    hitter_models   = load_models_count_only("hitter", HITTER_TARGETS)

    # ── Projection model selection ──────────────────────────────────────
    # Use the DIRECT counting-stat models — pitcher_K/BB/HR/H/IP and
    # hitter_H/HR/BB/K/PA, now hyperparameter-tuned + calibrated in
    # hitterspitchers_train.py — rather than the two-stage per-9 / team-PA
    # per-rate decompositions and the xHits per-9 blend. The direct models
    # predict the actual stat in one shot and score better overall on the
    # combined-season data, so the decomposition paths are turned OFF here.
    # Flip USE_DECOMPOSITION_MODELS back to True to restore the hybrid flow.
    USE_DECOMPOSITION_MODELS = False

    two_stage_pitcher = None
    if USE_DECOMPOSITION_MODELS and _HAS_TWO_STAGE_PITCHER and load_two_stage_models is not None:
        try:
            two_stage_pitcher = load_two_stage_models()
        except Exception as e:
            print(f"  [two-stage pitcher] load failed: {type(e).__name__}: {e}")
            two_stage_pitcher = None

    team_pa_bundle, rate_models = None, None
    if USE_DECOMPOSITION_MODELS and _HAS_TEAM_PA_HITTER and load_team_pa_models is not None:
        try:
            team_pa_bundle, rate_models = load_team_pa_models()
        except Exception as e:
            print(f"  [team-PA hitter] load failed: {type(e).__name__}: {e}")
            team_pa_bundle, rate_models = None, None

    if USE_DECOMPOSITION_MODELS:
        print(f"  Pitcher models loaded: {len(pitcher_models)}"
              f"{'  + two-stage [' + ','.join(two_stage_pitcher.keys()) + ']' if two_stage_pitcher else ''}")
        print(f"  Hitter  models loaded: {len(hitter_models)}"
              f"{'  + team-PA [' + ','.join(rate_models.keys()) + ']' if rate_models else ''}")
    else:
        print(f"  Pitcher models loaded: {len(pitcher_models)} (direct counting-stat models)")
        print(f"  Hitter  models loaded: {len(hitter_models)} (direct counting-stat models)")

    schedule_games = fetch_schedule(target_str)
    print(f"  Games found: {len(schedule_games)}")

    roster_maps = build_roster_maps(schedule_games)

    pitchers_today = build_today_pitchers(schedule_games)
    print(f"  Pitchers found: {len(pitchers_today)}")

    hitters_raw = scrape_rotowire_lineups(schedule_games)
    hitters_today = map_lineups_to_rosters(hitters_raw, roster_maps) if not hitters_raw.empty else hitters_raw
    print(f"  Hitters found:  {len(hitters_today)}")

    pitcher_proj = score_pitchers(
        pitchers_today,
        pitcher_game_df,
        pitcher_models,
        team_batting_ctx,
        target_ts,
        league_means,
        two_stage_models=two_stage_pitcher,
    )
    hitter_proj = score_hitters(
        hitters_today,
        pitchers_today,
        hitter_game_df,
        pitcher_game_df,
        hitter_models,
        team_pitching_ctx,
        target_ts,
        league_means,
        team_pa_bundle=team_pa_bundle,
        rate_models=rate_models,
    )

    print(f"  Pitchers projected: {len(pitcher_proj)}")
    print(f"  Pitchers dropped:   {max(0, len(pitchers_today) - len(pitcher_proj))}")
    print(f"  Hitters projected:  {len(hitter_proj)}")
    print(f"  Hitters dropped:    {max(0, len(hitters_today) - len(hitter_proj))}")

    # ─────────────────────────────────────────────────────────────────────
    # CONTEXT LAYER STACK — applied in order, each step is opt-in (no-op
    # if data missing). Pipeline order matches the architecture document:
    #   Raw model → Park (already inside score_*) → Weather → Umpire
    #     → xHits blend → Market calibration → Bias calibration (later)
    # ─────────────────────────────────────────────────────────────────────

    # Weather: pull once per slate, then apply to both projection DFs.
    if (not RAW_MODEL_ONLY) and _fetch_weather is not None and _apply_weather is not None:
        try:
            games_for_weather = pd.DataFrame([
                {"game_pk": gm.get("gamePk") or gm.get("game_pk"),
                 "home_team": team_to_abbr(gm.get("home_team") or gm.get("home")),
                 "commence_time": gm.get("gameDate")}
                for gm in schedule_games
            ])
            weather_raw = _fetch_weather(games_for_weather)
            weather_fx = _weather_factors(weather_raw)
            if not weather_fx.empty:
                # Pitchers need home_team for the join; pitcher_proj has `opponent`
                # for the away starter and `team` for the home starter, so we
                # synthesize home_team explicitly.
                if not pitcher_proj.empty:
                    pp = pitcher_proj.copy()
                    pp["home_team"] = pp.apply(
                        lambda r: r["team"] if r.get("team") and r.get("opponent")
                                  and pp[pp["opponent"] == r["team"]].shape[0] > 0
                                  else r.get("opponent", r["team"]), axis=1)
                    pitcher_proj = _apply_weather(pp, weather_fx, kind="pitcher", verbose=True)
                if not hitter_proj.empty:
                    hp = hitter_proj.copy()
                    hp["home_team"] = hp.apply(
                        lambda r: r["team"] if r.get("team") and r.get("opponent")
                                  and hp[hp["opponent"] == r["team"]].shape[0] > 0
                                  else r.get("opponent", r["team"]), axis=1)
                    hitter_proj = _apply_weather(hp, weather_fx, kind="hitter", verbose=True)
        except Exception as e:
            print(f"  [weather] skipped ({type(e).__name__}: {e})")

    # Umpire: K factor on pitcher_proj only (hitter K props also get the
    # benefit since the underlying hitter K projection is per-PA and the
    # ump effect mostly comes through the pitcher-side K rate).
    if (not RAW_MODEL_ONLY) and _apply_umpire is not None and not pitcher_proj.empty:
        try:
            pitcher_proj = _apply_umpire(pitcher_proj, verbose=True)
        except Exception as e:
            print(f"  [umpire] skipped ({type(e).__name__}: {e})")

    # xHits: blend the H-allowed projection with the smoothed-target xH9
    # prediction (BABIP-noise dampened). This is a per-9 rate model, so it's
    # disabled alongside the other decomposition models — the direct
    # pitcher_H model is used as-is. Gated on USE_DECOMPOSITION_MODELS.
    xh_bundle = _load_xh() if (USE_DECOMPOSITION_MODELS and _load_xh is not None) else None
    if xh_bundle is not None and not pitcher_proj.empty and "proj_ip" in pitcher_proj.columns:
        try:
            # Read the pitcher feature rows so we can re-score xH from the
            # same features the two-stage model used. We rejoin by mlb_id.
            pgd = pd.read_csv(DATA_DIR / "pitcher_game_data.csv", low_memory=False)
            id_col = get_pitcher_id_col(pgd)
            n_col = get_pitcher_name_col(pgd)
            xh_blend_rows = []
            for _, r in pitcher_proj.iterrows():
                base = get_latest_player_row(
                    pgd, target_date=target_ts, id_col=id_col, name_col=n_col,
                    player_id=r.get("mlb_id"), player_name=r.get("player_name"),
                )
                if base.empty:
                    xh_blend_rows.append(None)
                    continue
                xh = _predict_xh(base, float(r.get("proj_ip", 5.0)), xh_bundle)
                xh_blend_rows.append(xh)
            # 50/50 blend with the existing proj_hits_allowed (drops to 0% if xh missing)
            if any(v is not None for v in xh_blend_rows):
                pitcher_proj = pitcher_proj.copy()
                xh_series = pd.Series(xh_blend_rows, index=pitcher_proj.index)
                old_h = pd.to_numeric(pitcher_proj["proj_hits_allowed"], errors="coerce")
                blended = old_h.where(xh_series.isna(), 0.5 * old_h + 0.5 * xh_series)
                pitcher_proj["proj_hits_allowed_raw"] = old_h
                pitcher_proj["proj_hits_allowed"] = blended.clip(lower=0)
                print(f"  [xHits] blended {xh_series.notna().sum()}/{len(pitcher_proj)} pitcher rows")
        except Exception as e:
            print(f"  [xHits] skipped ({type(e).__name__}: {e})")

    # Market calibration: blend with sportsbook-implied projections.
    if (not RAW_MODEL_ONLY) and _load_market_priors is not None and _apply_market is not None:
        try:
            priors = _load_market_priors()
            if priors:
                if not pitcher_proj.empty:
                    pitcher_proj = _apply_market(pitcher_proj, priors,
                                                  beta_base=0.15, verbose=True)
                if not hitter_proj.empty:
                    hitter_proj = _apply_market(hitter_proj, priors,
                                                 beta_base=0.20, verbose=True)
        except Exception as e:
            print(f"  [market] skipped ({type(e).__name__}: {e})")

    fallback_count = 0
    if not hitter_proj.empty and "used_fallback" in hitter_proj.columns:
        fallback_count = int(
            pd.to_numeric(hitter_proj["used_fallback"], errors="coerce").fillna(0).sum()
        )
        print(f"  Hitters using fallback: {fallback_count}")

    out = pd.concat([pitcher_proj, hitter_proj], ignore_index=True, sort=False)

    # ── Merge FanDuel sportsbook lines ────────────────────────────────────
    fanduel_props = load_fanduel_props()
    if not fanduel_props.empty:
        out = merge_fanduel_lines(out, fanduel_props)
        fd_cols = [c for c in out.columns if c.startswith("fd_")]
        print(f"  FanDuel prop columns merged: {fd_cols}")
    else:
        print("  FanDuel props not found — run fanduel_props.py first (optional)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "hitterspitchers_today.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

    if not pitcher_proj.empty:
        print("\n── PITCHERS (all by projected strikeouts) ───────────────────")
        print(
            pitcher_proj.sort_values("proj_strikeouts", ascending=False)[
                [
                    "player_name", "team", "opponent", "proj_ip", "proj_outs",
                    "proj_runs_allowed", "proj_hits_allowed", "proj_strikeouts",
                    "proj_walks",
                ]
            ].to_string(index=False)
        )

    if not hitter_proj.empty:
        print("\n── HITTERS (top 20 by projected hits) ──────────────────────")
        print(
            hitter_proj.sort_values("proj_hits", ascending=False)[
                [
                    "player_name", "team", "opponent", "lineup_spot", "proj_pa",
                    "proj_hits", "proj_hr", "proj_strikeouts", "confidence"
                ]
            ].head(20).to_string(index=False)
        )

        if "used_fallback" in hitter_proj.columns:
            real_hitters = hitter_proj[hitter_proj["used_fallback"] == False].copy()
            fallback_hitters = hitter_proj[hitter_proj["used_fallback"] == True].copy()

            if not real_hitters.empty:
                print("\n── HITTERS (top 20 by projected hits, real matches) ───────")
                print(
                    real_hitters.sort_values("proj_hits", ascending=False)[
                        [
                            "player_name", "team", "opponent", "lineup_spot", "proj_pa",
                            "proj_hits", "proj_hr", "proj_strikeouts", "confidence"
                        ]
                    ].head(20).to_string(index=False)
                )

            if not fallback_hitters.empty:
                print("\n── HITTERS (fallback hitters) ─────────────────────────────")
                print(
                    fallback_hitters.sort_values("proj_hits", ascending=False)[
                        [
                            "player_name", "team", "opponent", "lineup_spot", "proj_pa",
                            "proj_hits", "proj_hr", "proj_strikeouts", "confidence"
                        ]
                    ].to_string(index=False)
                )

    return out

def print_summary(pitcher_df, hitter_df, out_path):
    print("\n==============================")
    print(" PROJECTION SUMMARY")
    print("==============================")
    print(f"Pitchers: {len(pitcher_df)}")
    print(f"Hitters : {len(hitter_df)}")
    print(f"Output  : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run daily hitter/pitcher MLB projections")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default=today)")
    args = parser.parse_args()

    run_projections(args.date)


if __name__ == "__main__":
    main()