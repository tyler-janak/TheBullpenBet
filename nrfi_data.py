"""
nrfi_data.py
============
Build first-inning game-level dataset for NRFI (No Run First Inning) modeling.

Features
--------
• Starting pitcher's FULL-GAME rolling stats (K%, BB%, H%, HR%) — primary signal, stable
• Starting pitcher's 1st-inning rolling stats — secondary signal, noisy but specific
• Batting team's full-game offensive stats (runs/game, K%, OBP) — team quality signal
• Batting team's 1st-inning production stats
• Park factor

Output
------
    data/nrfi_game_data.csv

Usage
-----
    python nrfi_data.py --input pitch_data.csv
    python nrfi_data.py --input pitch_data.csv --park-factors data/park_factors.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
ROLLING_WINDOWS_FULL = [5, 10]       # for full-game stats — keep compact
ROLLING_WINDOWS_1ST  = [5, 10]       # for first-inning stats — keep compact

# Columns to pull from pitcher_game_data.csv
PITCHER_GAME_COLS = [
    "pitcher", "game_date", "team",
    "K_rate", "BB_rate", "H_rate", "HR_rate", "IP",
    "K_rate_last5",  "K_rate_last10",  "K_rate_std",
    "BB_rate_last5", "BB_rate_last10", "BB_rate_std",
    "H_rate_last5",  "H_rate_last10",  "H_rate_std",
    "HR_rate_last5", "HR_rate_last10", "HR_rate_std",
    "IP_last5",      "IP_last10",      "IP_std",
    "days_rest",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_rate(num, denom, default=np.nan):
    num   = pd.to_numeric(num,   errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return np.where((denom > 0) & denom.notna(), num / denom, default)


def rolling_mean(series, w):
    """Shifted rolling mean (no lookahead)."""
    return series.shift(1).rolling(w, min_periods=1).mean()


def expanding_mean(series):
    """Shifted expanding mean — season-to-date."""
    return series.shift(1).expanding().mean()


def add_rolling_features(df: pd.DataFrame, group_col: str,
                         stat_cols: list[str],
                         windows: list[int] = ROLLING_WINDOWS_1ST) -> pd.DataFrame:
    df = df.sort_values([group_col, "game_date"]).copy()
    for w in windows:
        for col in stat_cols:
            if col not in df.columns:
                continue
            df[f"{col}_last{w}"] = (
                df.groupby(group_col)[col]
                .transform(lambda x, _w=w: rolling_mean(x, _w))
            )
    for col in stat_cols:
        if col not in df.columns:
            continue
        df[f"{col}_std"] = (
            df.groupby(group_col)[col]
            .transform(expanding_mean)
        )
    return df


# ── load & prep raw pitch data ────────────────────────────────────────────────

NEEDED_COLS = {
    "game_date", "inning_topbot", "home_team", "away_team",
    "pitcher_team", "batter_team", "pitcher", "batter",
    "player_name", "events", "description",
    "p_throws", "stand",
    "post_bat_score", "bat_score",
    "game_pk", "inning",
}


def load_and_prep(filepath: str) -> pd.DataFrame:
    print(f"Loading {filepath} ...")
    header = pd.read_csv(filepath, nrows=0, low_memory=False)
    header.columns = header.columns.str.strip().str.lower()
    usecols = [c for c in header.columns if c in NEEDED_COLS]
    print(f"  Loading {len(usecols)} of {len(header.columns)} columns ...")
    df = pd.read_csv(filepath, usecols=usecols, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    df.columns = df.columns.str.strip().str.lower()

    if "game_date" not in df.columns:
        raise ValueError("'game_date' column not found.")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "pitcher_team" not in df.columns:
        if "inning_topbot" in df.columns and "home_team" in df.columns and "away_team" in df.columns:
            top_mask = df["inning_topbot"].astype(str).str.strip().str.lower().eq("top")
            df["pitcher_team"] = np.where(top_mask, df["home_team"], df["away_team"])
            df["batter_team"]  = np.where(top_mask, df["away_team"], df["home_team"])

    ev = df["events"].fillna("").astype(str).str.lower() if "events" in df.columns else pd.Series("", index=df.index)
    df["is_k"]   = ev.isin({"strikeout", "strikeout_double_play"}).astype(int)
    df["is_bb"]  = ev.isin({"walk", "intent_walk"}).astype(int)
    df["is_hr"]  = ev.isin({"home_run"}).astype(int)
    df["is_hit"] = ev.isin({"single", "double", "triple", "home_run"}).astype(int)
    df["is_pa"]  = ev.ne("").astype(int)

    if "post_bat_score" in df.columns and "bat_score" in df.columns:
        df["runs_scored"] = (
            pd.to_numeric(df["post_bat_score"], errors="coerce")
            - pd.to_numeric(df["bat_score"], errors="coerce")
        ).clip(lower=0).fillna(0)
    else:
        df["runs_scored"] = 0.0

    if "game_pk" not in df.columns:
        df["game_pk"] = (
            df["game_date"].dt.strftime("%Y%m%d").fillna("0") + "_"
            + df.get("home_team", pd.Series("UNK", index=df.index)).fillna("UNK") + "_"
            + df.get("away_team", pd.Series("UNK", index=df.index)).fillna("UNK")
        )

    return df


# ── load pitcher_game_data.csv (full-game stats) ──────────────────────────────

def load_pitcher_game_data(path: str = "data/pitcher_game_data.csv") -> pd.DataFrame:
    """
    Load pre-computed pitcher game-level data (output of hitterspitchers_data.py).
    These full-game rolling stats are the primary NRFI predictor for pitcher quality.
    """
    p = Path(path)
    if not p.exists():
        print(f"  [warn] {path} not found — full-game pitcher features will be skipped.")
        print(f"         Run: python hitterspitchers_data.py --input <pitch_csv>")
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Normalize column names for team lookup
    if "team" not in df.columns and "pitcher_team" in df.columns:
        df = df.rename(columns={"pitcher_team": "team"})

    keep = [c for c in PITCHER_GAME_COLS if c in df.columns]
    if not keep:
        print("  [warn] pitcher_game_data.csv has no usable columns.")
        return pd.DataFrame()

    print(f"  Loaded pitcher_game_data.csv: {len(df):,} rows, {len(keep)} relevant columns")
    return df[keep].copy()


# ── build team full-game offense stats ───────────────────────────────────────

def build_team_full_game_offense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full-game team batting stats per game, with rolling features.
    Much more stable than 1st-inning-only stats.
    """
    if "batter_team" not in df.columns:
        return pd.DataFrame()

    game_id_col = "game_pk" if "game_pk" in df.columns else None
    group_cols = ["batter_team", "game_date"]
    if game_id_col:
        group_cols = ["batter_team", game_id_col, "game_date"]

    grp = df.groupby(group_cols).agg(
        fg_pa    = ("is_pa",       "sum"),
        fg_h     = ("is_hit",      "sum"),
        fg_bb    = ("is_bb",       "sum"),
        fg_k     = ("is_k",        "sum"),
        fg_hr    = ("is_hr",       "sum"),
        fg_runs  = ("runs_scored", "sum"),
    ).reset_index()

    grp["fg_k_pct"]   = safe_rate(grp["fg_k"],              grp["fg_pa"])
    grp["fg_ob_pct"]  = safe_rate(grp["fg_h"] + grp["fg_bb"], grp["fg_pa"])
    grp["fg_runs_pg"] = grp["fg_runs"]

    stat_cols = ["fg_k_pct", "fg_ob_pct", "fg_runs_pg"]
    grp = add_rolling_features(grp, "batter_team", stat_cols, windows=ROLLING_WINDOWS_FULL)

    print(f"  Team full-game offense rows: {len(grp):,}")
    return grp


# ── game outcomes (runs in 1st inning) ───────────────────────────────────────

def build_game_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if "inning" not in df.columns:
        raise ValueError("'inning' column not found in pitch data.")

    df1 = df[pd.to_numeric(df["inning"], errors="coerce") == 1].copy()
    if df1.empty:
        raise ValueError("No first-inning rows found.")

    top_mask = df1["inning_topbot"].astype(str).str.strip().str.lower().eq("top")

    top1 = (
        df1[top_mask].groupby("game_pk")["runs_scored"]
        .sum().reset_index()
        .rename(columns={"runs_scored": "away_runs_1st"})
    )
    bot1 = (
        df1[~top_mask].groupby("game_pk")["runs_scored"]
        .sum().reset_index()
        .rename(columns={"runs_scored": "home_runs_1st"})
    )

    meta_cols = [c for c in ["game_pk", "game_date", "home_team", "away_team"] if c in df.columns]
    meta = (
        df.groupby("game_pk")[meta_cols]
        .first().reset_index(drop=True)
        .drop_duplicates("game_pk")
    )

    games = meta.merge(top1, on="game_pk", how="left")
    games = games.merge(bot1, on="game_pk", how="left")
    games["away_runs_1st"] = games["away_runs_1st"].fillna(0)
    games["home_runs_1st"] = games["home_runs_1st"].fillna(0)
    games["nrfi"] = (
        (games["away_runs_1st"] == 0) & (games["home_runs_1st"] == 0)
    ).astype(int)

    print(f"  Game outcomes: {len(games):,} games | NRFI rate: {games['nrfi'].mean():.3f}")
    return games


# ── pitcher 1st-inning stats ──────────────────────────────────────────────────

def build_pitcher_1st_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "inning" not in df.columns:
        return pd.DataFrame()

    df1 = df[pd.to_numeric(df["inning"], errors="coerce") == 1].copy()
    if df1.empty:
        return pd.DataFrame()

    group_cols = ["pitcher", "game_pk", "game_date"]
    if "pitcher_team" in df1.columns:
        group_cols.append("pitcher_team")

    grp = df1.groupby(group_cols).agg(
        bf           = ("is_pa",       "sum"),
        k            = ("is_k",        "sum"),
        bb           = ("is_bb",       "sum"),
        h            = ("is_hit",      "sum"),
        hr           = ("is_hr",       "sum"),
        runs_allowed = ("runs_scored", "sum"),
    ).reset_index()

    if "player_name" in df1.columns:
        names = df1.groupby("pitcher")["player_name"].first().reset_index()
        grp = grp.merge(names, on="pitcher", how="left")

    grp["k1_rate"]      = safe_rate(grp["k"],            grp["bf"])
    grp["bb1_rate"]     = safe_rate(grp["bb"],           grp["bf"])
    grp["h1_rate"]      = safe_rate(grp["h"],            grp["bf"])
    grp["hr1_rate"]     = safe_rate(grp["hr"],           grp["bf"])
    grp["runs1_per_pa"] = safe_rate(grp["runs_allowed"], grp["bf"])

    stat_cols = ["k1_rate", "bb1_rate", "h1_rate", "runs1_per_pa"]
    grp = add_rolling_features(grp, "pitcher", stat_cols, windows=ROLLING_WINDOWS_1ST)

    print(f"  Pitcher 1st-inning rows: {len(grp):,}")
    return grp


# ── team batting 1st-inning stats ─────────────────────────────────────────────

def build_team_bat_1st_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "inning" not in df.columns or "batter_team" not in df.columns:
        return pd.DataFrame()

    df1 = df[pd.to_numeric(df["inning"], errors="coerce") == 1].copy()
    if df1.empty:
        return pd.DataFrame()

    grp = df1.groupby(["batter_team", "game_pk", "game_date"]).agg(
        pa          = ("is_pa",       "sum"),
        h           = ("is_hit",      "sum"),
        bb          = ("is_bb",       "sum"),
        k           = ("is_k",        "sum"),
        hr          = ("is_hr",       "sum"),
        runs_scored = ("runs_scored", "sum"),
    ).reset_index()

    grp["ob1_rate"]     = safe_rate(grp["h"] + grp["bb"], grp["pa"])
    grp["k1_bat_rate"]  = safe_rate(grp["k"],             grp["pa"])
    grp["runs1_scored"] = grp["runs_scored"]

    stat_cols = ["ob1_rate", "k1_bat_rate", "runs1_scored"]
    grp = add_rolling_features(grp, "batter_team", stat_cols, windows=ROLLING_WINDOWS_1ST)

    print(f"  Team batting 1st-inning rows: {len(grp):,}")
    return grp


# ── assemble game-level NRFI dataset ─────────────────────────────────────────

def build_nrfi_dataset(
    games:           pd.DataFrame,
    pitcher_1st:     pd.DataFrame,
    team_bat_1st:    pd.DataFrame,
    park_factors:    pd.DataFrame,
    pitcher_game_df: pd.DataFrame = None,  # full-game pitcher stats
    team_fg_df:      pd.DataFrame = None,  # full-game team offense stats
) -> pd.DataFrame:
    """
    One row per game. Merges:
      1. 1st-inning specific pitcher stats (noisy, inning-specific)
      2. Full-game pitcher stats (stable, high signal) — primary predictor
      3. 1st-inning team batting stats
      4. Full-game team offense stats (stable)
      5. Park factor
    All rolling features are shift(1) so there is no data leakage.
    """
    if games.empty:
        return pd.DataFrame()

    out = games.copy()
    pk  = "game_pk"

    # ── 1. Away starter 1st-inning stats ─────────────────────────────────────
    if not pitcher_1st.empty and "pitcher_team" in pitcher_1st.columns:
        away_cols = [pk, "pitcher", "pitcher_team"] + [
            c for c in pitcher_1st.columns
            if c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS_1ST)
        ]
        away_cols = [c for c in away_cols if c in pitcher_1st.columns]

        away_sp_slim = pitcher_1st[away_cols].copy().rename(columns={
            "pitcher": "away_sp_id",
            **{c: f"away_sp_{c}" for c in away_cols
               if c not in [pk, "pitcher", "pitcher_team"]}
        })

        out = out.merge(
            away_sp_slim.rename(columns={"pitcher_team": "_away_pt"}),
            left_on=[pk, "away_team"],
            right_on=[pk, "_away_pt"],
            how="left",
        ).drop(columns=["_away_pt"], errors="ignore")

    # ── 2. Home starter 1st-inning stats ─────────────────────────────────────
    if not pitcher_1st.empty and "pitcher_team" in pitcher_1st.columns:
        home_sp_slim = pitcher_1st[away_cols].copy().rename(columns={
            "pitcher": "home_sp_id",
            **{c: f"home_sp_{c}" for c in away_cols
               if c not in [pk, "pitcher", "pitcher_team"]}
        })

        out = out.merge(
            home_sp_slim.rename(columns={"pitcher_team": "_home_pt"}),
            left_on=[pk, "home_team"],
            right_on=[pk, "_home_pt"],
            how="left",
        ).drop(columns=["_home_pt"], errors="ignore")

    # ── 3. Away team 1st-inning batting stats ─────────────────────────────────
    if not team_bat_1st.empty:
        bat_cols = ["batter_team", pk] + [
            c for c in team_bat_1st.columns
            if c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS_1ST)
        ]
        bat_cols = [c for c in bat_cols if c in team_bat_1st.columns]

        away_bat = team_bat_1st[bat_cols].copy().rename(columns={
            "batter_team": "_bt",
            **{c: f"away_bat_{c}" for c in bat_cols if c not in ["batter_team", pk]}
        })
        out = out.merge(away_bat, left_on=[pk, "away_team"], right_on=[pk, "_bt"], how="left").drop(columns=["_bt"], errors="ignore")

        home_bat = team_bat_1st[bat_cols].copy().rename(columns={
            "batter_team": "_bt",
            **{c: f"home_bat_{c}" for c in bat_cols if c not in ["batter_team", pk]}
        })
        out = out.merge(home_bat, left_on=[pk, "home_team"], right_on=[pk, "_bt"], how="left").drop(columns=["_bt"], errors="ignore")

    # ── 4. Full-game pitcher stats (highest signal) ───────────────────────────
    if pitcher_game_df is not None and not pitcher_game_df.empty:
        fg_stat_cols = [c for c in pitcher_game_df.columns
                        if c not in ["pitcher", "game_date", "team"]
                        and (c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS_FULL)
                             or c == "days_rest")]

        fg_cols = ["pitcher", "game_date"] + fg_stat_cols
        fg_cols = [c for c in fg_cols if c in pitcher_game_df.columns]

        # Away SP — join by (pitcher_id, game_date)
        if "away_sp_id" in out.columns:
            fg_away = pitcher_game_df[fg_cols].rename(columns={
                "pitcher": "away_sp_id",
                **{c: f"away_sp_fg_{c}" for c in fg_stat_cols if c in pitcher_game_df.columns}
            })
            out = out.merge(fg_away, on=["away_sp_id", "game_date"], how="left")

        # Home SP — join by (pitcher_id, game_date)
        if "home_sp_id" in out.columns:
            fg_home = pitcher_game_df[fg_cols].rename(columns={
                "pitcher": "home_sp_id",
                **{c: f"home_sp_fg_{c}" for c in fg_stat_cols if c in pitcher_game_df.columns}
            })
            out = out.merge(fg_home, on=["home_sp_id", "game_date"], how="left")

        print(f"  Merged full-game pitcher stats: "
              f"{out['away_sp_fg_K_rate_last5'].notna().sum() if 'away_sp_fg_K_rate_last5' in out.columns else 0} away, "
              f"{out['home_sp_fg_K_rate_last5'].notna().sum() if 'home_sp_fg_K_rate_last5' in out.columns else 0} home non-null rows")

    # ── 5. Full-game team offense stats ───────────────────────────────────────
    if team_fg_df is not None and not team_fg_df.empty:
        fg_bat_stat_cols = [c for c in team_fg_df.columns
                            if c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS_FULL)]
        fg_bat_join_cols = ["batter_team", pk] + fg_bat_stat_cols
        fg_bat_join_cols = [c for c in fg_bat_join_cols if c in team_fg_df.columns]

        away_fg_bat = team_fg_df[fg_bat_join_cols].rename(columns={
            "batter_team": "_bt",
            **{c: f"away_bat_fg_{c}" for c in fg_bat_stat_cols if c in team_fg_df.columns}
        })
        out = out.merge(away_fg_bat, left_on=[pk, "away_team"], right_on=[pk, "_bt"], how="left").drop(columns=["_bt"], errors="ignore")

        home_fg_bat = team_fg_df[fg_bat_join_cols].rename(columns={
            "batter_team": "_bt",
            **{c: f"home_bat_fg_{c}" for c in fg_bat_stat_cols if c in team_fg_df.columns}
        })
        out = out.merge(home_fg_bat, left_on=[pk, "home_team"], right_on=[pk, "_bt"], how="left").drop(columns=["_bt"], errors="ignore")

    # ── 6. Park factor ────────────────────────────────────────────────────────
    if not park_factors.empty and "home_team" in out.columns:
        out = out.merge(park_factors, left_on="home_team", right_on="team", how="left")
        out["park_factor"] = out.get("park_factor", pd.Series(100.0, index=out.index)).fillna(100.0)
        out = out.drop(columns=["team"], errors="ignore")

    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build NRFI game-level dataset")
    parser.add_argument("--input",        required=True, help="Pitch-level CSV")
    parser.add_argument("--park-factors", default="data/park_factors.csv")
    parser.add_argument("--pitcher-game", default="data/pitcher_game_data.csv",
                        help="Full-game pitcher stats (from hitterspitchers_data.py)")
    parser.add_argument("--out",          default="data/nrfi_game_data.csv")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    df = load_and_prep(args.input)

    # Park factors
    pf_path = Path(args.park_factors)
    if pf_path.exists():
        park_factors = pd.read_csv(pf_path)
        park_factors.columns = park_factors.columns.str.strip()
        park_factors["park_factor"] = pd.to_numeric(park_factors["park_factor"], errors="coerce").fillna(100.0)
    else:
        park_factors = pd.DataFrame(columns=["team", "park_factor"])

    # Full-game pitcher stats (primary signal)
    print("\nLoading full-game pitcher data ...")
    pitcher_game_df = load_pitcher_game_data(args.pitcher_game)

    print("\nBuilding NRFI dataset ...")
    games        = build_game_outcomes(df)
    pitcher_1st  = build_pitcher_1st_stats(df)
    team_bat_1st = build_team_bat_1st_stats(df)
    team_fg_df   = build_team_full_game_offense(df)

    nrfi_df = build_nrfi_dataset(
        games, pitcher_1st, team_bat_1st, park_factors,
        pitcher_game_df=pitcher_game_df,
        team_fg_df=team_fg_df,
    )

    out_path = Path(args.out)
    nrfi_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(nrfi_df):,} rows, {len(nrfi_df.columns)} columns)")

    # Report fill rates for key feature groups
    fg_cols = [c for c in nrfi_df.columns if "fg_" in c and "last5" in c]
    if fg_cols:
        fill = nrfi_df[fg_cols].notna().mean()
        print(f"\nFull-game feature fill rates (sample):")
        print(fill.head(8).to_string())


if __name__ == "__main__":
    main()
