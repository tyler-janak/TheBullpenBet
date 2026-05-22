"""
hitterspitchers_data.py
=======================
Builds pitcher / hitter game-level modeling tables from pitch-level Statcast data.

Outputs:
- data/pitcher_game_data.csv
- data/hitter_game_data.csv
- data/team_batting_hand_context.csv
- data/team_pitching_hand_context.csv

Run:
    python hitterspitchers_data.py --input pitch_data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COL = {
    "pitcher":        "pitcher",
    "batter":         "batter",
    "game_date":      "game_date",
    "pitcher_team":   "pitcher_team",
    "batter_team":    "batter_team",
    "opponent":       "opponent_team",
    "pitcher_hand":   "p_throws",
    "batter_hand":    "stand",
    "pitcher_name":   "player_name",
    "events":         "events",
    "description":    "description",
    "velocity":       "release_speed",
    "spin_rate":      "release_spin_rate",
    "pitch_type":     "pitch_type",
    "launch_speed":   "launch_speed",
    "launch_angle":   "launch_angle",
    "hit_direction":  "spray_angle",
    "inning_topbot":  "inning_topbot",
    "home_team":      "home_team",
    "away_team":      "away_team",
}

STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WALK_EVENTS = {"walk", "intent_walk"}
HR_EVENTS = {"home_run"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "force_out",
    "double_play", "triple_play", "fielders_choice_out",
    "sac_fly", "sac_bunt", "fielders_choice",
    "strikeout", "strikeout_double_play",
}

ROLLING_WINDOWS = [7, 10, 14, 21, 30]


def load_park_factors(filepath: str = "data/park_factors.csv") -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        print(f"  [warn] {filepath} not found — using neutral park factor (100)")
        return pd.DataFrame(columns=["team", "park_factor"])

    pf = pd.read_csv(path)
    pf.columns = [str(c).strip() for c in pf.columns]

    if "team" not in pf.columns or "park_factor" not in pf.columns:
        raise ValueError("park_factors.csv must have columns: team, park_factor")

    pf["team"] = pf["team"].astype(str).str.strip()
    pf["park_factor"] = pd.to_numeric(pf["park_factor"], errors="coerce").fillna(100.0)
    return pf[["team", "park_factor"]].copy()


def merge_park_factors(df: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    if park_factors.empty or "team" not in df.columns:
        out = df.copy()
        out["park_factor"] = 100.0
        return out

    out = df.merge(park_factors, on="team", how="left")
    out["park_factor"] = out["park_factor"].fillna(100.0)
    return out


def safe_rate(num, denom, default=np.nan):
    num = pd.to_numeric(num, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return np.where((denom > 0) & pd.notna(denom), num / denom, default)


def normalize_hand(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().upper()
    if v.startswith("R"):
        return "R"
    if v.startswith("L"):
        return "L"
    return np.nan


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def flatten_columns(columns) -> list[str]:
    out = []
    for col in columns:
        if isinstance(col, tuple):
            out.append("_".join([str(x) for x in col if str(x) != ""]).rstrip("_"))
        else:
            out.append(str(col))
    return out


NEEDED_COLS = {
    # COL mappings
    "pitcher", "batter", "game_date", "player_name",
    "p_throws", "stand", "events", "description",
    "release_speed", "release_spin_rate", "pitch_type",
    "launch_speed", "launch_angle", "spray_angle",
    "inning_topbot", "home_team", "away_team",
    # pitch-quality features (whiff/csw/zone/EV/hard-hit/barrel/pitch-mix)
    "zone", "balls", "strikes",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "pfx_x", "pfx_z",
    # derived / optional columns used in event_flags
    "post_bat_score", "bat_score",
    "pitcher_team", "batter_team", "opponent_team",
}

# Pitch type buckets for usage / velocity-by-type features.
# Statcast pitch_type codes; sweepers (SV) ride with sliders.
PITCH_TYPE_FB    = {"FF", "SI", "FC"}           # four-seam, sinker, cutter
PITCH_TYPE_BR    = {"SL", "CU", "KC", "CS", "SV"}  # slider, curve, sweeper
PITCH_TYPE_OFF   = {"CH", "FS", "FO", "SC"}     # change, splitter, forkball


def load_data(filepath: str) -> pd.DataFrame:
    print(f"Loading {filepath} ...")
    # Read only the header first to find which needed columns actually exist
    header = pd.read_csv(filepath, nrows=0, low_memory=False)
    header.columns = header.columns.str.strip().str.lower()
    usecols = [c for c in header.columns if c in NEEDED_COLS]
    print(f"  Loading {len(usecols)} of {len(header.columns)} columns ...")
    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    df.columns = df.columns.str.strip().str.lower()

    date_col = COL["game_date"]
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found. Check COL mapping.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    required = {COL["inning_topbot"], COL["home_team"], COL["away_team"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns needed to derive teams: {missing}")

    top_mask = df[COL["inning_topbot"]].astype(str).str.strip().str.lower().eq("top")

    df = df.assign(
        pitcher_team=np.where(top_mask, df[COL["home_team"]], df[COL["away_team"]]),
        batter_team=np.where(top_mask, df[COL["away_team"]], df[COL["home_team"]]),
    )
    df["opponent_team"] = df["batter_team"]

    return df.copy()


def event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ev = (
        df[COL["events"]].fillna("").astype(str).str.lower()
        if COL["events"] in df.columns
        else pd.Series("", index=df.index)
    )

    df["is_k"] = ev.isin(STRIKEOUT_EVENTS).astype(int)
    df["is_bb"] = ev.isin(WALK_EVENTS).astype(int)
    df["is_hr"] = ev.isin(HR_EVENTS).astype(int)
    df["is_hit"] = ev.isin(HIT_EVENTS).astype(int)
    df["is_out"] = ev.isin(OUT_EVENTS).astype(int)
    df["is_pa"] = ev.ne("").astype(int)

    # ── Pitch-quality flags (per-pitch, summed per game later) ─────────────
    # Statcast description vocabulary:
    #   swinging_strike, swinging_strike_blocked, foul_tip → whiff
    #   called_strike                                       → called strike
    #   foul, foul_bunt, hit_into_play, *_into_play_*       → swing
    #   ball, blocked_ball, pitchout, hit_by_pitch          → not a swing
    desc = (
        df["description"].fillna("").astype(str).str.lower()
        if "description" in df.columns
        else pd.Series("", index=df.index)
    )
    _WHIFFS  = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
    _CALLED  = {"called_strike"}
    _SWINGS  = _WHIFFS | {"foul", "foul_bunt", "hit_into_play",
                          "hit_into_play_no_out", "hit_into_play_score"}
    df["is_swing"]         = desc.isin(_SWINGS).astype(int)
    df["is_whiff"]         = desc.isin(_WHIFFS).astype(int)
    df["is_called_strike"] = desc.isin(_CALLED).astype(int)

    # First-pitch strike: ball+strike count is 0-0 at start of PA and the
    # outcome is anything resulting in 0-1 (called_strike, swinging_strike,
    # foul, foul_tip, or hit_into_play that ended the PA in the pitcher's
    # favor). We approximate as "0-0 pitch that wasn't a ball/HBP".
    if "balls" in df.columns and "strikes" in df.columns:
        b0 = pd.to_numeric(df["balls"],   errors="coerce").fillna(-1) == 0
        s0 = pd.to_numeric(df["strikes"], errors="coerce").fillna(-1) == 0
        first_pitch = b0 & s0
        df["is_first_pitch"]        = first_pitch.astype(int)
        df["is_first_pitch_strike"] = (first_pitch & ~desc.isin({"ball", "blocked_ball", "hit_by_pitch", "pitchout"})).astype(int)
    else:
        df["is_first_pitch"] = 0
        df["is_first_pitch_strike"] = 0

    # In-zone: Statcast `zone` ∈ {1..9} = strike zone, {11..14} = ball zones
    if "zone" in df.columns:
        zn = pd.to_numeric(df["zone"], errors="coerce")
        df["is_in_zone"] = ((zn >= 1) & (zn <= 9)).astype(int)
    else:
        df["is_in_zone"] = 0

    # Ball in play + quality-of-contact flags
    if "launch_speed" in df.columns:
        ev_speed = pd.to_numeric(df["launch_speed"], errors="coerce")
        df["is_bip"]       = ev_speed.notna().astype(int)
        df["is_hard_hit"]  = (ev_speed >= 95).fillna(False).astype(int)
        if "launch_angle" in df.columns:
            la = pd.to_numeric(df["launch_angle"], errors="coerce")
            # Approximate barrel definition (Statcast formal def is more
            # nuanced but this captures ~95% of barrels): EV ≥ 98 AND
            # LA within the sweet-spot band that scales with EV.
            df["is_barrel"] = ((ev_speed >= 98) & (la >= 26) & (la <= 30)).fillna(False).astype(int)
        else:
            df["is_barrel"] = 0
        # Keep the actual EV / LA per pitch for mean aggregations later.
        df["bip_ev"] = ev_speed.where(ev_speed.notna(), other=np.nan)
    else:
        df["is_bip"] = 0
        df["is_hard_hit"] = 0
        df["is_barrel"] = 0
        df["bip_ev"] = np.nan

    # Pitch-type bucket flags (for usage / velocity-by-type)
    if "pitch_type" in df.columns:
        pt = df["pitch_type"].fillna("").astype(str).str.upper()
        df["is_fb"]  = pt.isin(PITCH_TYPE_FB).astype(int)
        df["is_br"]  = pt.isin(PITCH_TYPE_BR).astype(int)
        df["is_off"] = pt.isin(PITCH_TYPE_OFF).astype(int)
        # Per-pitch velocity bucketed to fastball pitches only (NaN otherwise)
        # so a `mean` aggregation gives fastball-only avg velocity.
        if "release_speed" in df.columns:
            rs = pd.to_numeric(df["release_speed"], errors="coerce")
            df["fb_velocity"] = rs.where(df["is_fb"] == 1, other=np.nan)
            # Breaking ball + offspeed velocity for velocity-differential math
            df["br_velocity"]  = rs.where(df["is_br"]  == 1, other=np.nan)
            df["off_velocity"] = rs.where(df["is_off"] == 1, other=np.nan)
        else:
            df["fb_velocity"] = df["br_velocity"] = df["off_velocity"] = np.nan

        # Pitch-type × swing/whiff cross-flags. These decompose stuff in a
        # way the aggregate whiff_rate can't — two pitchers with identical
        # overall whiff_rate but completely different FB/BR profiles project
        # very differently against a fastball-heavy vs breaking-ball-heavy
        # lineup. This is the single highest-leverage K feature beyond the
        # raw rate stats.
        df["is_swing_fb"]  = (df["is_swing"] & df["is_fb"]).astype(int)
        df["is_swing_br"]  = (df["is_swing"] & df["is_br"]).astype(int)
        df["is_swing_off"] = (df["is_swing"] & df["is_off"]).astype(int)
        df["is_whiff_fb"]  = (df["is_whiff"] & df["is_fb"]).astype(int)
        df["is_whiff_br"]  = (df["is_whiff"] & df["is_br"]).astype(int)
        df["is_whiff_off"] = (df["is_whiff"] & df["is_off"]).astype(int)
    else:
        df["is_fb"] = df["is_br"] = df["is_off"] = 0
        df["fb_velocity"] = df["br_velocity"] = df["off_velocity"] = np.nan
        df["is_swing_fb"] = df["is_swing_br"] = df["is_swing_off"] = 0
        df["is_whiff_fb"] = df["is_whiff_br"] = df["is_whiff_off"] = 0

    if "post_bat_score" in df.columns and "bat_score" in df.columns:
        post_bat = pd.to_numeric(df["post_bat_score"], errors="coerce")
        bat_score = pd.to_numeric(df["bat_score"], errors="coerce")
        df["bat_runs_scored"] = (post_bat - bat_score).clip(lower=0).fillna(0)
    else:
        df["bat_runs_scored"] = np.nan

    if COL["pitcher_hand"] in df.columns:
        df[COL["pitcher_hand"]] = df[COL["pitcher_hand"]].apply(normalize_hand)

    if COL["batter_hand"] in df.columns:
        df[COL["batter_hand"]] = df[COL["batter_hand"]].apply(normalize_hand)

    return df


def mark_actual_starters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark actual starting pitcher for each team/game as the last pitcher
    that appears for that team in the pitch-level data.

    This is used because this dataset is compiled backwards, so the starter
    is the final pitcher entry for that team/game.
    """
    df = df.copy()

    pitcher_col = COL["pitcher"]
    date_col = COL["game_date"]
    team_col = COL["pitcher_team"]

    if any(c not in df.columns for c in [pitcher_col, date_col, team_col]):
        df["is_actual_starter"] = 0
        return df

    game_id_col = first_existing(df, ["game_pk", "game_id", "gamepk"])

    group_cols = [date_col, team_col]
    if game_id_col is not None and game_id_col in df.columns:
        group_cols = [game_id_col, team_col]

    temp = df.reset_index().rename(columns={"index": "_row_order"})

    starter_rows = (
        temp.sort_values("_row_order")
        .groupby(group_cols, as_index=False)
        .last()[group_cols + [pitcher_col]]
        .rename(columns={pitcher_col: "_starter_pitcher"})
    )

    temp = temp.merge(starter_rows, on=group_cols, how="left")
    temp["is_actual_starter"] = (temp[pitcher_col] == temp["_starter_pitcher"]).astype(int)

    temp = temp.drop(columns=["_starter_pitcher"])
    temp = temp.sort_values("_row_order").drop(columns=["_row_order"])
    return temp


def add_rolling(game_df: pd.DataFrame, player_col: str, stat_cols: list, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    game_df = game_df.sort_values([player_col, COL["game_date"]]).copy()

    for w in windows:
        for col in stat_cols:
            if col not in game_df.columns:
                continue
            game_df[f"{col}_last{w}"] = (
                game_df.groupby(player_col)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    return game_df


def season_to_date(game_df: pd.DataFrame, player_col: str, stat_cols: list) -> pd.DataFrame:
    game_df = game_df.sort_values([player_col, COL["game_date"]]).copy()

    for col in stat_cols:
        if col not in game_df.columns:
            continue
        game_df[f"{col}_std"] = (
            game_df.groupby(player_col)[col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    return game_df


def add_group_rolling(df: pd.DataFrame, group_col: str, stat_cols: list, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    df = df.sort_values([group_col, COL["game_date"]]).copy()

    for w in windows:
        for col in stat_cols:
            if col not in df.columns:
                continue
            df[f"{col}_last{w}"] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    return df


def add_group_std(df: pd.DataFrame, group_col: str, stat_cols: list) -> pd.DataFrame:
    df = df.sort_values([group_col, COL["game_date"]]).copy()

    for col in stat_cols:
        if col not in df.columns:
            continue
        df[f"{col}_std"] = (
            df.groupby(group_col)[col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    return df


def build_team_batting_hand_context(df: pd.DataFrame) -> pd.DataFrame:
    date_col = COL["game_date"]
    team_col = COL["batter_team"]
    p_hand_col = COL["pitcher_hand"]

    needed = [team_col, date_col, p_hand_col]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    temp = df[df[p_hand_col].isin(["R", "L"])].copy()

    grp = temp.groupby([team_col, date_col, p_hand_col]).agg(
        PA=("is_pa", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        HR=("is_hr", "sum"),
        H=("is_hit", "sum"),
    ).reset_index()

    grp["team_k_rate_vs_hand"] = safe_rate(grp["K"], grp["PA"])
    grp["team_bb_rate_vs_hand"] = safe_rate(grp["BB"], grp["PA"])
    grp["team_hr_rate_vs_hand"] = safe_rate(grp["HR"], grp["PA"])
    grp["team_h_rate_vs_hand"] = safe_rate(grp["H"], grp["PA"])

    grp["team_hand_key"] = grp[team_col].astype(str) + "_vs_" + grp[p_hand_col].astype(str)

    stat_cols = [
        "team_k_rate_vs_hand",
        "team_bb_rate_vs_hand",
        "team_hr_rate_vs_hand",
        "team_h_rate_vs_hand",
    ]
    grp = add_group_rolling(grp, "team_hand_key", stat_cols)
    grp = add_group_std(grp, "team_hand_key", stat_cols)

    return grp.rename(columns={team_col: "team", p_hand_col: "pitcher_hand_split"})


def build_team_pitching_hand_context(df: pd.DataFrame) -> pd.DataFrame:
    date_col = COL["game_date"]
    team_col = COL["pitcher_team"]
    b_hand_col = COL["batter_hand"]

    needed = [team_col, date_col, b_hand_col]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    temp = df[df[b_hand_col].isin(["R", "L"])].copy()

    grp = temp.groupby([team_col, date_col, b_hand_col]).agg(
        PA=("is_pa", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        HR=("is_hr", "sum"),
        H=("is_hit", "sum"),
    ).reset_index()

    grp["team_allowed_k_rate_vs_hand"] = safe_rate(grp["K"], grp["PA"])
    grp["team_allowed_bb_rate_vs_hand"] = safe_rate(grp["BB"], grp["PA"])
    grp["team_allowed_hr_rate_vs_hand"] = safe_rate(grp["HR"], grp["PA"])
    grp["team_allowed_h_rate_vs_hand"] = safe_rate(grp["H"], grp["PA"])

    grp["team_hand_key"] = grp[team_col].astype(str) + "_vs_" + grp[b_hand_col].astype(str)

    stat_cols = [
        "team_allowed_k_rate_vs_hand",
        "team_allowed_bb_rate_vs_hand",
        "team_allowed_hr_rate_vs_hand",
        "team_allowed_h_rate_vs_hand",
    ]
    grp = add_group_rolling(grp, "team_hand_key", stat_cols)
    grp = add_group_std(grp, "team_hand_key", stat_cols)

    return grp.rename(columns={team_col: "team", b_hand_col: "batter_hand_split"})


def build_pitcher_games(df: pd.DataFrame, team_batting_hand_ctx: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding pitcher game-level data ...")

    pitcher_col = COL["pitcher"]
    date_col = COL["game_date"]
    opponent_col = COL["opponent"]
    team_col = COL["pitcher_team"]
    hand_col = COL["pitcher_hand"]
    b_hand_col = COL["batter_hand"]
    pitch_name_col = COL["pitcher_name"]
    vel_col = COL["velocity"]
    spin_col = COL["spin_rate"]
    pitch_col = COL["pitch_type"]

    needed = [pitcher_col, date_col, opponent_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required pitcher grouping columns: {missing}")

    game_id_col = first_existing(df, ["game_pk", "game_id", "gamepk"])

    group_cols = [pitcher_col, date_col, opponent_col]
    if game_id_col is not None and game_id_col in df.columns:
        group_cols = [pitcher_col, game_id_col, date_col, opponent_col]

    grp = df.groupby(group_cols)

    agg = grp.agg(
        pitches=(pitcher_col, "count"),
        BF=("is_pa", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        HR=("is_hr", "sum"),
        H=("is_hit", "sum"),
        # R = total runs scored against this pitcher across the game. Statcast
        # does not natively expose ER (earned/unearned split needs the
        # scorekeeper's call), so we use total runs as the proxy — across MLB
        # the unearned-run rate is ~6%, so R is a close upper bound for ER.
        # `bat_runs_scored` is the per-pitch run-delta we computed in event_flags.
        R=("bat_runs_scored", "sum"),
        outs=("is_out", "sum"),
        avg_velocity=(vel_col, "mean"),
        avg_spin=(spin_col, "mean"),
        # ── pitch-quality aggregations (highest-leverage K + hits features) ──
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
        first_pitches=("is_first_pitch", "sum"),
        first_pitch_strikes=("is_first_pitch_strike", "sum"),
        in_zone=("is_in_zone", "sum"),
        bip=("is_bip", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        barrels=("is_barrel", "sum"),
        avg_ev_allowed=("bip_ev", "mean"),
        # ── pitch-mix + per-type velocity ────────────────────────────────────
        fb_pitches=("is_fb", "sum"),
        br_pitches=("is_br", "sum"),
        off_pitches=("is_off", "sum"),
        fb_velo=("fb_velocity", "mean"),
        br_velo=("br_velocity", "mean"),
        off_velo=("off_velocity", "mean"),
        # ── pitch-type-specific whiff (stuff decomposition) ─────────────────
        swings_fb=("is_swing_fb",  "sum"),
        swings_br=("is_swing_br",  "sum"),
        swings_off=("is_swing_off", "sum"),
        whiffs_fb=("is_whiff_fb",  "sum"),
        whiffs_br=("is_whiff_br",  "sum"),
        whiffs_off=("is_whiff_off", "sum"),
    ).reset_index()

    # Derive per-game rates from the summed flags. NaN-safe with epsilon
    # floors so 0-pitch edge cases don't blow up.
    eps = 1e-9
    agg["whiff_rate"]       = agg["whiffs"]               / agg["swings"].clip(lower=eps)
    agg["csw_pct"]          = (agg["whiffs"] + agg["called_strikes"]) / agg["pitches"].clip(lower=eps)
    agg["zone_pct"]         = agg["in_zone"]              / agg["pitches"].clip(lower=eps)
    agg["f_strike_pct"]     = agg["first_pitch_strikes"]  / agg["first_pitches"].clip(lower=eps)
    agg["hard_hit_pct"]     = agg["hard_hits"]            / agg["bip"].clip(lower=eps)
    agg["barrel_pct"]       = agg["barrels"]              / agg["bip"].clip(lower=eps)
    agg["fb_pct"]           = agg["fb_pitches"]           / agg["pitches"].clip(lower=eps)
    agg["br_pct"]           = agg["br_pitches"]           / agg["pitches"].clip(lower=eps)
    agg["off_pct"]          = agg["off_pitches"]          / agg["pitches"].clip(lower=eps)

    # Pitch-type-specific whiff rates. The decomposition matters because
    # K_rate already captures aggregate strike-out rate; what it can't see
    # is WHERE the whiffs come from (a ground-baller's slider vs a
    # fastball-only thrower) which projects very differently against the
    # next lineup's contact profile.
    agg["whiff_rate_fb"]    = agg["whiffs_fb"]   / agg["swings_fb"].clip(lower=eps)
    agg["whiff_rate_br"]    = agg["whiffs_br"]   / agg["swings_br"].clip(lower=eps)
    agg["whiff_rate_off"]   = agg["whiffs_off"]  / agg["swings_off"].clip(lower=eps)

    # Velocity differential (fastball − offspeed) — pitchers with a wider
    # velo gap between fastball and changeup miss bats more (tunnel effect).
    agg["velo_sep_fb_off"]  = agg["fb_velo"] - agg["off_velo"]
    agg["velo_sep_fb_br"]   = agg["fb_velo"] - agg["br_velo"]

    # Clip rates that can degenerate (very rare pitchers with 0 BIP, etc.)
    for c in ("whiff_rate","csw_pct","zone_pct","f_strike_pct",
              "hard_hit_pct","barrel_pct","fb_pct","br_pct","off_pct",
              "whiff_rate_fb","whiff_rate_br","whiff_rate_off"):
        agg[c] = agg[c].clip(lower=0, upper=1)

    if "is_actual_starter" in df.columns:
        starter_df = grp["is_actual_starter"].max().reset_index()
        agg = agg.merge(starter_df, on=group_cols, how="left")
        agg["is_actual_starter"] = agg["is_actual_starter"].fillna(0).astype(int)
    else:
        agg["is_actual_starter"] = 0

    if team_col in df.columns:
        team_df = grp[team_col].first().reset_index().rename(columns={team_col: "team"})
        agg = agg.merge(team_df, on=group_cols, how="left")

    if pitch_name_col in df.columns:
        name_df = grp[pitch_name_col].first().reset_index().rename(columns={pitch_name_col: "pitcher_name"})
        agg = agg.merge(name_df, on=group_cols, how="left")

    if hand_col in df.columns:
        hand_df = grp[hand_col].first().reset_index().rename(columns={hand_col: "pitcher_hand"})
        agg = agg.merge(hand_df, on=group_cols, how="left")

    if b_hand_col in df.columns:
        split_group_cols = group_cols + [b_hand_col]

        split_grp = (
            df[df[b_hand_col].isin(["R", "L"])]
            .groupby(split_group_cols)
            .agg(
                split_bf=("is_pa", "sum"),
                split_k=("is_k", "sum"),
                split_bb=("is_bb", "sum"),
                split_hr=("is_hr", "sum"),
                split_h=("is_hit", "sum"),
            )
            .reset_index()
        )

        split_grp["pitcher_k_rate_vs_hand"] = safe_rate(split_grp["split_k"], split_grp["split_bf"])
        split_grp["pitcher_bb_rate_vs_hand"] = safe_rate(split_grp["split_bb"], split_grp["split_bf"])
        split_grp["pitcher_hr_rate_vs_hand"] = safe_rate(split_grp["split_hr"], split_grp["split_bf"])
        split_grp["pitcher_h_rate_vs_hand"] = safe_rate(split_grp["split_h"], split_grp["split_bf"])

        split_grp["pitcher_hand_key"] = split_grp[pitcher_col].astype(str) + "_vs_" + split_grp[b_hand_col].astype(str)

        split_stat_cols = [
            "pitcher_k_rate_vs_hand",
            "pitcher_bb_rate_vs_hand",
            "pitcher_hr_rate_vs_hand",
            "pitcher_h_rate_vs_hand",
        ]
        split_grp = add_group_rolling(split_grp, "pitcher_hand_key", split_stat_cols)
        split_grp = add_group_std(split_grp, "pitcher_hand_key", split_stat_cols)

        split_value_cols = [
            c for c in [
                "pitcher_k_rate_vs_hand", "pitcher_bb_rate_vs_hand",
                "pitcher_hr_rate_vs_hand", "pitcher_h_rate_vs_hand",
                "pitcher_k_rate_vs_hand_last5", "pitcher_bb_rate_vs_hand_last5",
                "pitcher_hr_rate_vs_hand_last5", "pitcher_h_rate_vs_hand_last5",
                "pitcher_k_rate_vs_hand_last10", "pitcher_bb_rate_vs_hand_last10",
                "pitcher_hr_rate_vs_hand_last10", "pitcher_h_rate_vs_hand_last10",
                "pitcher_k_rate_vs_hand_std", "pitcher_bb_rate_vs_hand_std",
                "pitcher_hr_rate_vs_hand_std", "pitcher_h_rate_vs_hand_std",
            ] if c in split_grp.columns
        ]

        if split_value_cols:
            split_pivot = (
                split_grp.pivot_table(
                    index=group_cols,
                    columns=b_hand_col,
                    values=split_value_cols,
                    aggfunc="first",
                )
                .reset_index()
            )
            split_pivot.columns = flatten_columns(split_pivot.columns)
            agg = agg.merge(split_pivot, on=group_cols, how="left")

    agg["IP"] = agg["outs"] / 3.0
    agg["K_rate"] = safe_rate(agg["K"], agg["BF"])
    agg["BB_rate"] = safe_rate(agg["BB"], agg["BF"])
    agg["HR_rate"] = safe_rate(agg["HR"], agg["BF"])
    agg["H_rate"] = safe_rate(agg["H"], agg["BF"])

    agg["BF_per_IP"] = np.where(agg["IP"] > 0, agg["BF"] / agg["IP"], np.nan)
    agg["pitches_per_BF"] = np.where(agg["BF"] > 0, agg["pitches"] / agg["BF"], np.nan)
    agg["pitches_per_IP"] = np.where(agg["IP"] > 0, agg["pitches"] / agg["IP"], np.nan)

    if pitch_col in df.columns:
        pitch_mix = (
            df.groupby(group_cols + [pitch_col])
            .size()
            .reset_index(name="n")
        )
        pitch_total = pitch_mix.groupby(group_cols)["n"].transform("sum")
        pitch_mix["pct"] = np.where(pitch_total > 0, pitch_mix["n"] / pitch_total, np.nan)

        pivot = (
            pitch_mix.pivot_table(
                index=group_cols,
                columns=pitch_col,
                values="pct",
                fill_value=np.nan,
            )
            .reset_index()
        )

        pivot.columns = [f"pitch_pct_{c}" if c not in group_cols else c for c in pivot.columns]
        agg = agg.merge(pivot, on=group_cols, how="left")

    rate_cols = [
        "K_rate", "BB_rate", "HR_rate", "H_rate", "IP",
        "avg_velocity", "avg_spin",
        "BF", "outs", "pitches",
        # raw counting stats — used as direct model targets
        "K", "BB", "HR", "H",
        "BF_per_IP", "pitches_per_BF", "pitches_per_IP",
        # NEW: pitch-quality + pitch-mix features (highest-leverage signals)
        "whiff_rate", "csw_pct", "zone_pct", "f_strike_pct",
        "hard_hit_pct", "barrel_pct", "avg_ev_allowed",
        "fb_pct", "br_pct", "off_pct", "fb_velo",
        # Pitch-type-specific whiff (stuff decomposition) + velo separation
        "whiff_rate_fb", "whiff_rate_br", "whiff_rate_off",
        "br_velo", "off_velo", "velo_sep_fb_off", "velo_sep_fb_br",
    ]
    rate_cols = [c for c in rate_cols if c in agg.columns]
    agg = add_rolling(agg, pitcher_col, rate_cols)
    agg = season_to_date(agg, pitcher_col, rate_cols)

    agg = agg.sort_values([pitcher_col, date_col]).copy()

    agg["BF_last3"] = agg.groupby(pitcher_col)["BF"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    agg["outs_last3"] = agg.groupby(pitcher_col)["outs"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    agg["pitches_last3"] = agg.groupby(pitcher_col)["pitches"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    agg["max_ip_last5"] = agg.groupby(pitcher_col)["IP"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max())
    agg["max_ip_last10"] = agg.groupby(pitcher_col)["IP"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).max())
    agg["days_rest"] = agg.groupby(pitcher_col)[date_col].transform(lambda x: (x - x.shift(1)).dt.days)
    agg["starter_pct_last5"] = agg.groupby(pitcher_col)["is_actual_starter"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    agg["starter_pct_last10"] = agg.groupby(pitcher_col)["is_actual_starter"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    if not team_batting_hand_ctx.empty and "pitcher_hand" in agg.columns:
        merge_cols = [
            "team", date_col, "pitcher_hand_split",
            "team_k_rate_vs_hand", "team_bb_rate_vs_hand", "team_hr_rate_vs_hand", "team_h_rate_vs_hand",
            "team_k_rate_vs_hand_last5", "team_bb_rate_vs_hand_last5", "team_hr_rate_vs_hand_last5", "team_h_rate_vs_hand_last5",
            "team_k_rate_vs_hand_last10", "team_bb_rate_vs_hand_last10", "team_hr_rate_vs_hand_last10", "team_h_rate_vs_hand_last10",
            "team_k_rate_vs_hand_std", "team_bb_rate_vs_hand_std", "team_hr_rate_vs_hand_std", "team_h_rate_vs_hand_std",
        ]
        merge_cols = [c for c in merge_cols if c in team_batting_hand_ctx.columns]

        agg = agg.merge(
            team_batting_hand_ctx[merge_cols],
            left_on=[opponent_col, date_col, "pitcher_hand"],
            right_on=["team", date_col, "pitcher_hand_split"],
            how="left",
        ).drop(columns=["team_y", "pitcher_hand_split"], errors="ignore")

        if "team_x" in agg.columns:
            agg = agg.rename(columns={"team_x": "team"})

    agg = agg[agg["is_actual_starter"] == 1].copy()

    print("\nStarter-only pitcher dataset check:")
    print(f"  Rows: {len(agg):,}")
    if not agg.empty:
        print(f"  Mean IP:   {agg['IP'].mean():.3f}")
        print(f"  Median IP: {agg['IP'].median():.3f}")

    print(f"  Pitcher games: {len(agg):,} rows, {len(agg.columns)} columns")
    return agg


def build_hitter_games(df: pd.DataFrame, team_pitching_hand_ctx: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding hitter game-level data ...")

    batter_col = COL["batter"]
    date_col = COL["game_date"]
    team_col = COL["batter_team"]
    opponent_col = COL["pitcher_team"]
    opp_p_col = COL["pitcher_name"]
    hand_col = COL["batter_hand"]
    p_hand_col = COL["pitcher_hand"]
    ev_col = COL["launch_speed"]
    la_col = COL["launch_angle"]
    dir_col = COL["hit_direction"]

    batter_name_col = first_existing(
        df,
        ["batter_name", "batter_full_name", "batter_player_name", "batter_name_display"]
    )

    needed = [batter_col, date_col, opponent_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required hitter grouping columns: {missing}")

    game_id_col = first_existing(df, ["game_pk", "game_id", "gamepk"])

    group_cols = [batter_col, date_col, opponent_col]
    if game_id_col is not None and game_id_col in df.columns:
        group_cols = [batter_col, game_id_col, date_col, opponent_col]

    grp = df.groupby(group_cols)

    agg_dict = {
        "PA": ("is_pa", "sum"),
        "H": ("is_hit", "sum"),
        "HR": ("is_hr", "sum"),
        "BB": ("is_bb", "sum"),
        "K": ("is_k", "sum"),
        # RBI is approximated as the sum of `bat_runs_scored` over the
        # batter's PA — that's literally "runs that resulted from this
        # at-bat", which equals batter RBI on the vast majority of plays
        # (modern scorekeeping awards RBI for any run scored on a hit,
        # walk, HBP, sac, or productive out). Errors and wild pitches are
        # the only edge cases; the bias from those is < 5%.
        "RBI": ("bat_runs_scored", "sum"),
    }

    if ev_col in df.columns:
        agg_dict["avg_EV"] = (ev_col, "mean")
        agg_dict["max_EV"] = (ev_col, "max")
    if la_col in df.columns:
        agg_dict["avg_LA"] = (la_col, "mean")
    if dir_col in df.columns:
        agg_dict["avg_direction"] = (dir_col, "mean")

    agg = grp.agg(**agg_dict).reset_index()

    if team_col in df.columns:
        team_df = grp[team_col].first().reset_index().rename(columns={team_col: "team"})
        agg = agg.merge(team_df, on=group_cols, how="left")

    if batter_name_col and batter_name_col in df.columns:
        name_df = grp[batter_name_col].first().reset_index().rename(columns={batter_name_col: "batter_name"})
        agg = agg.merge(name_df, on=group_cols, how="left")

    if hand_col in df.columns:
        hand_df = grp[hand_col].first().reset_index().rename(columns={hand_col: "batter_hand"})
        agg = agg.merge(hand_df, on=group_cols, how="left")

    if opp_p_col in df.columns:
        opp_df = grp[opp_p_col].first().reset_index().rename(columns={opp_p_col: "opp_pitcher_name"})
        agg = agg.merge(opp_df, on=group_cols, how="left")

    if p_hand_col in df.columns:
        p_hand_df = grp[p_hand_col].first().reset_index().rename(columns={p_hand_col: "opp_pitcher_hand"})
        agg = agg.merge(p_hand_df, on=group_cols, how="left")

        split_grp = (
            df[df[p_hand_col].isin(["R", "L"])]
            .groupby(group_cols + [p_hand_col])
            .agg(
                split_pa=("is_pa", "sum"),
                split_h=("is_hit", "sum"),
                split_hr=("is_hr", "sum"),
                split_bb=("is_bb", "sum"),
                split_k=("is_k", "sum"),
            )
            .reset_index()
        )

        split_grp["hitter_h_rate_vs_hand"] = safe_rate(split_grp["split_h"], split_grp["split_pa"])
        split_grp["hitter_hr_rate_vs_hand"] = safe_rate(split_grp["split_hr"], split_grp["split_pa"])
        split_grp["hitter_bb_rate_vs_hand"] = safe_rate(split_grp["split_bb"], split_grp["split_pa"])
        split_grp["hitter_k_rate_vs_hand"] = safe_rate(split_grp["split_k"], split_grp["split_pa"])

        split_grp["hitter_hand_key"] = (
            split_grp[batter_col].astype(str) + "_vs_" + split_grp[p_hand_col].astype(str)
        )

        split_stat_cols = [
            "hitter_h_rate_vs_hand",
            "hitter_hr_rate_vs_hand",
            "hitter_bb_rate_vs_hand",
            "hitter_k_rate_vs_hand",
        ]
        split_grp = add_group_rolling(split_grp, "hitter_hand_key", split_stat_cols)
        split_grp = add_group_std(split_grp, "hitter_hand_key", split_stat_cols)

        split_value_cols = [
            c for c in [
                "hitter_h_rate_vs_hand", "hitter_hr_rate_vs_hand",
                "hitter_bb_rate_vs_hand", "hitter_k_rate_vs_hand",
                "hitter_h_rate_vs_hand_last5", "hitter_hr_rate_vs_hand_last5",
                "hitter_bb_rate_vs_hand_last5", "hitter_k_rate_vs_hand_last5",
                "hitter_h_rate_vs_hand_last10", "hitter_hr_rate_vs_hand_last10",
                "hitter_bb_rate_vs_hand_last10", "hitter_k_rate_vs_hand_last10",
                "hitter_h_rate_vs_hand_std", "hitter_hr_rate_vs_hand_std",
                "hitter_bb_rate_vs_hand_std", "hitter_k_rate_vs_hand_std",
            ] if c in split_grp.columns
        ]

        if split_value_cols:
            split_pivot = (
                split_grp.pivot_table(
                    index=group_cols,
                    columns=p_hand_col,
                    values=split_value_cols,
                    aggfunc="first",
                )
                .reset_index()
            )
            split_pivot.columns = flatten_columns(split_pivot.columns)
            agg = agg.merge(split_pivot, on=group_cols, how="left")

    for stat in ["H", "HR", "BB", "K"]:
        if stat in agg.columns and "PA" in agg.columns:
            agg[f"{stat.lower()}_rate"] = safe_rate(agg[stat], agg["PA"])

    if {"avg_EV", "avg_LA"}.issubset(set(agg.columns)):
        agg["barrel_proxy"] = (
            (pd.to_numeric(agg["avg_EV"], errors="coerce") >= 95) &
            (pd.to_numeric(agg["avg_LA"], errors="coerce").between(10, 30))
        ).astype(float)

        agg["hard_hit_proxy"] = (
            pd.to_numeric(agg["avg_EV"], errors="coerce") >= 92
        ).astype(float)

        agg["sweet_spot_proxy"] = (
            pd.to_numeric(agg["avg_LA"], errors="coerce").between(8, 32)
        ).astype(float)

        agg["blast_proxy"] = (
            (pd.to_numeric(agg["avg_EV"], errors="coerce") >= 98) &
            (pd.to_numeric(agg["avg_LA"], errors="coerce").between(18, 32))
        ).astype(float)

        agg["ev_la_interaction"] = (
            pd.to_numeric(agg["avg_EV"], errors="coerce").fillna(0) *
            pd.to_numeric(agg["avg_LA"], errors="coerce").fillna(0)
        )

    if "max_EV" in agg.columns and "avg_EV" in agg.columns:
        agg["ev_spread"] = (
            pd.to_numeric(agg["max_EV"], errors="coerce").fillna(0) -
            pd.to_numeric(agg["avg_EV"], errors="coerce").fillna(0)
        )

    if {"H", "BB", "HR", "PA"}.issubset(set(agg.columns)):
        agg["times_on_base_rate"] = safe_rate(agg["H"] + agg["BB"], agg["PA"])
        agg["xbh_proxy_rate"] = safe_rate(agg["HR"], agg["PA"])

    rate_cols = [
        "h_rate", "hr_rate", "bb_rate", "k_rate", "PA",
        # raw counting stats — used as direct model targets
        "H", "HR", "BB", "K",
        "avg_EV", "max_EV", "avg_LA", "avg_direction",
        "barrel_proxy", "hard_hit_proxy", "sweet_spot_proxy", "blast_proxy",
        "ev_la_interaction", "ev_spread", "times_on_base_rate", "xbh_proxy_rate",
    ]
    rate_cols = [c for c in rate_cols if c in agg.columns]

    agg = add_rolling(agg, batter_col, rate_cols)
    agg = season_to_date(agg, batter_col, rate_cols)

    agg = agg.sort_values([batter_col, date_col]).copy()
    agg["PA_last3"] = agg.groupby(batter_col)["PA"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    agg["PA_last7"] = agg.groupby(batter_col)["PA"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    agg["max_hr_rate_last10"] = agg.groupby(batter_col)["hr_rate"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).max()
    )
    agg["max_h_rate_last10"] = agg.groupby(batter_col)["h_rate"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).max()
    )
    agg["days_since_game"] = agg.groupby(batter_col)[date_col].transform(
        lambda x: (x - x.shift(1)).dt.days
    )

    if not team_pitching_hand_ctx.empty and "batter_hand" in agg.columns:
        merge_cols = [
            "team", date_col, "batter_hand_split",
            "team_allowed_k_rate_vs_hand", "team_allowed_bb_rate_vs_hand",
            "team_allowed_hr_rate_vs_hand", "team_allowed_h_rate_vs_hand",
            "team_allowed_k_rate_vs_hand_last5", "team_allowed_bb_rate_vs_hand_last5",
            "team_allowed_hr_rate_vs_hand_last5", "team_allowed_h_rate_vs_hand_last5",
            "team_allowed_k_rate_vs_hand_last10", "team_allowed_bb_rate_vs_hand_last10",
            "team_allowed_hr_rate_vs_hand_last10", "team_allowed_h_rate_vs_hand_last10",
            "team_allowed_k_rate_vs_hand_std", "team_allowed_bb_rate_vs_hand_std",
            "team_allowed_hr_rate_vs_hand_std", "team_allowed_h_rate_vs_hand_std",
        ]
        merge_cols = [c for c in merge_cols if c in team_pitching_hand_ctx.columns]

        agg = agg.merge(
            team_pitching_hand_ctx[merge_cols],
            left_on=[opponent_col, date_col, "batter_hand"],
            right_on=["team", date_col, "batter_hand_split"],
            how="left",
        ).drop(columns=["team_y", "batter_hand_split"], errors="ignore")

        if "team_x" in agg.columns:
            agg = agg.rename(columns={"team_x": "team"})

    print(f"  Hitter games: {len(agg):,} rows, {len(agg.columns)} columns")
    return agg




def build_team_recent_form(hitter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """Simple team recent-form table for website insights and optional modeling use."""
    frames = []
    date_col = COL["game_date"]

    if not hitter_df.empty and all(c in hitter_df.columns for c in ["team", date_col]):
        h = hitter_df.copy()
        keep = [c for c in ["team", date_col, "PA", "H", "HR", "BB", "K", "h_rate", "hr_rate", "bb_rate", "k_rate"] if c in h.columns]
        h = h[keep].copy()
        grp = h.groupby(["team", date_col], as_index=False).agg({
            **({"PA": "sum"} if "PA" in h.columns else {}),
            **({"H": "sum"} if "H" in h.columns else {}),
            **({"HR": "sum"} if "HR" in h.columns else {}),
            **({"BB": "sum"} if "BB" in h.columns else {}),
            **({"K": "sum"} if "K" in h.columns else {}),
            **({"h_rate": "mean"} if "h_rate" in h.columns else {}),
            **({"hr_rate": "mean"} if "hr_rate" in h.columns else {}),
            **({"bb_rate": "mean"} if "bb_rate" in h.columns else {}),
            **({"k_rate": "mean"} if "k_rate" in h.columns else {}),
        })
        rate_cols = [c for c in ["PA", "H", "HR", "BB", "K", "h_rate", "hr_rate", "bb_rate", "k_rate"] if c in grp.columns]
        grp = add_group_rolling(grp, "team", rate_cols)
        grp = add_group_std(grp, "team", rate_cols)
        frames.append(grp)

    if not pitcher_df.empty and all(c in pitcher_df.columns for c in ["team", date_col]):
        p = pitcher_df.copy()
        keep = [c for c in ["team", date_col, "IP", "H", "HR", "BB", "K", "K_rate", "BB_rate", "HR_rate", "H_rate"] if c in p.columns]
        p = p[keep].copy()
        grp = p.groupby(["team", date_col], as_index=False).agg({
            **({"IP": "sum"} if "IP" in p.columns else {}),
            **({"H": "sum"} if "H" in p.columns else {}),
            **({"HR": "sum"} if "HR" in p.columns else {}),
            **({"BB": "sum"} if "BB" in p.columns else {}),
            **({"K": "sum"} if "K" in p.columns else {}),
            **({"K_rate": "mean"} if "K_rate" in p.columns else {}),
            **({"BB_rate": "mean"} if "BB_rate" in p.columns else {}),
            **({"HR_rate": "mean"} if "HR_rate" in p.columns else {}),
            **({"H_rate": "mean"} if "H_rate" in p.columns else {}),
        })
        grp = grp.rename(columns={c: f"pitching_{c}" for c in grp.columns if c not in ["team", date_col]})
        rate_cols = [c for c in grp.columns if c not in ["team", date_col]]
        grp = add_group_rolling(grp, "team", rate_cols)
        grp = add_group_std(grp, "team", rate_cols)
        frames.append(grp)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for nxt in frames[1:]:
        out = out.merge(nxt, on=["team", date_col], how="outer")
    return out.sort_values(["team", date_col]).reset_index(drop=True)


def build_recent_games_export(df: pd.DataFrame, player_type: str) -> pd.DataFrame:
    date_col = COL["game_date"]
    if df.empty or date_col not in df.columns:
        return pd.DataFrame()
    keep = [c for c in df.columns if c in {date_col, "team", "opponent", "pitcher_name", "batter_name", "player_name", "K_rate", "BB_rate", "HR_rate", "H_rate", "IP", "PA", "h_rate", "hr_rate", "bb_rate", "k_rate", "park_factor"}]
    out = df[keep].copy()
    out["player_type"] = player_type
    return out.sort_values(date_col).reset_index(drop=True)

def enrich_hitter_with_opp_starter(hitter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge opposing starter features into hitter game rows.

    We match:
      hitter.team      -> starter.opponent
      hitter.pitcher_team/opponent_team/opponent -> starter.team
      hitter.game_date -> starter.game_date
      optional game_id/game_pk if present in both

    This version is defensive about column names so it works with your
    current hitter/pitcher build outputs.
    """
    if hitter_df.empty or pitcher_df.empty:
        return hitter_df.copy()

    out = hitter_df.copy()
    starters = pitcher_df.copy()

    date_col = COL["game_date"]
    game_id_col = first_existing(out, ["game_pk", "game_id", "gamepk"])

    # Figure out the hitter-side opponent column that actually exists
    hitter_opp_col = None
    for cand in ["pitcher_team", "opponent", "opponent_team"]:
        if cand in out.columns:
            hitter_opp_col = cand
            break

    # Figure out the pitcher-side team/opponent columns
    starter_team_col = "team" if "team" in starters.columns else None
    starter_opp_col = None
    for cand in ["opponent", "opponent_team", COL["opponent"]]:
        if cand in starters.columns:
            starter_opp_col = cand
            break

    if hitter_opp_col is None:
        raise ValueError(
            "Could not find hitter opponent column. Expected one of: "
            "['pitcher_team', 'opponent', 'opponent_team']"
        )

    if starter_team_col is None or starter_opp_col is None:
        raise ValueError(
            "Pitcher starter table is missing required team/opponent columns."
        )

    # Keep only the starter columns that actually exist
    starter_cols = [
        starter_team_col,
        starter_opp_col,
        date_col,
        "pitcher_name",
        "pitcher_hand",
        "K_rate",
        "BB_rate",
        "HR_rate",
        "H_rate",
        "IP",
        "K_rate_last5",
        "BB_rate_last5",
        "HR_rate_last5",
        "H_rate_last5",
        "IP_last5",
        "K_rate_last10",
        "BB_rate_last10",
        "HR_rate_last10",
        "H_rate_last10",
        "IP_last10",
        "K_rate_std",
        "BB_rate_std",
        "HR_rate_std",
        "H_rate_std",
        "IP_std",
    ]

    if game_id_col is not None and game_id_col in starters.columns:
        starter_cols.append(game_id_col)

    starter_cols = [c for c in starter_cols if c in starters.columns]
    starters = starters[starter_cols].copy()

    rename_map = {
        "pitcher_name": "opp_sp_name",
        "pitcher_hand": "opp_sp_hand",
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
        starter_team_col: "__starter_team",
        starter_opp_col: "__starter_opp",
    }
    starters = starters.rename(columns=rename_map)

    # Build merge keys using columns that definitely exist
    out["__hitter_team"] = out["team"]
    out["__hitter_opp"] = out[hitter_opp_col]

    left_on = ["__hitter_team", "__hitter_opp", date_col]
    right_on = ["__starter_opp", "__starter_team", date_col]

    if game_id_col is not None and game_id_col in out.columns and game_id_col in starters.columns:
        left_on.append(game_id_col)
        right_on.append(game_id_col)

    out = out.merge(
        starters,
        left_on=left_on,
        right_on=right_on,
        how="left",
        suffixes=("", "_oppsp"),
    )

    out = out.drop(
        columns=[
            "__hitter_team",
            "__hitter_opp",
            "__starter_team",
            "__starter_opp",
            "team_oppsp",
            "opponent_oppsp",
        ],
        errors="ignore",
    )

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input pitch-level CSV file")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)

    df = load_data(args.input)
    df = event_flags(df)
    df = mark_actual_starters(df)

    park_factors = load_park_factors()

    team_batting_hand_ctx = build_team_batting_hand_context(df)
    team_pitching_hand_ctx = build_team_pitching_hand_context(df)

    pitcher_df = build_pitcher_games(df, team_batting_hand_ctx)
    hitter_df = build_hitter_games(df, team_pitching_hand_ctx)
    hitter_df = enrich_hitter_with_opp_starter(hitter_df, pitcher_df)

    pitcher_df = merge_park_factors(pitcher_df, park_factors)
    hitter_df = merge_park_factors(hitter_df, park_factors)

    # float_format="%.4f" keeps these committed feature tables under GitHub's
    # 100 MB file limit (the hitter table has ~195 float columns; full repr
    # makes it 150 MB+, rounding to 4 decimals cuts it ~3x with no
    # modelling-relevant precision loss). int64 ID columns are unaffected.
    pitcher_df.to_csv("data/pitcher_game_data.csv", index=False, float_format="%.4f")
    hitter_df.to_csv("data/hitter_game_data.csv", index=False, float_format="%.4f")
    team_batting_hand_ctx.to_csv("data/team_batting_hand_context.csv", index=False, float_format="%.4f")
    team_pitching_hand_ctx.to_csv("data/team_pitching_hand_context.csv", index=False, float_format="%.4f")

    print("\nSaved:")
    print("  data/pitcher_game_data.csv")
    print("  data/hitter_game_data.csv")
    print("  data/team_batting_hand_context.csv")
    print("  data/team_pitching_hand_context.csv")


if __name__ == "__main__":
    main()
