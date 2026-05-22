"""
pages/games_tab.py  — fixed v3
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from components.data_loader import (
    fetch_schedule, fmt_ev, fmt_odds, fmt_pct, fmt_time,
    get_hitter_projs, get_pitcher_projs,
    load_accuracy, load_predictions, load_projections,
    normalize_name, team_key, today_str,
)


def _safe(val, default="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

def _round(val, n=2, default="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return round(float(val), n)

def _stat_card(label, value, sub="", cls=""):
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value {cls}">{value}</div>
        <div class="stat-sub">{sub}</div>
    </div>"""


# ── Game card HTML ────────────────────────────────────────────────────────────

def _game_card_html(game: dict, pred: dict) -> str:
    is_value    = bool(pred.get("betRecommended", False))
    home_prob   = pred.get("homeWinProb", 0.5) or 0.5
    away_prob   = pred.get("awayWinProb", 0.5) or 0.5
    ev          = pred.get("ev")
    predicted   = pred.get("predictedWinner", "")
    winner_prob = pred.get("winnerProb", 0.5) or 0.5

    badge    = '<span class="badge-value">◆ Good Value</span>' if is_value else '<span class="badge-pass">Pass</span>'
    card_cls = "game-card value" if is_value else "game-card"
    game_time = fmt_time(pred.get("commenceTime") or game.get("commence_time", ""))

    home         = game.get("home", "")
    away         = game.get("away", "")
    home_starter = game.get("homeStarter", "TBD")
    away_starter = game.get("awayStarter", "TBD")
    home_ml      = fmt_odds(pred.get("home_ml"))
    away_ml      = fmt_odds(pred.get("away_ml"))

    if ev is not None and not (isinstance(ev, float) and np.isnan(ev)):
        ev_cls = "ev-pos" if float(ev) >= 0 else "ev-neg"
        ev_str = fmt_ev(float(ev))
    else:
        ev_cls = "ev-na"
        ev_str = "—"

    hp = int(round(home_prob * 100))
    ap = int(round(away_prob * 100))

    return f"""
    <div class="{card_cls}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div class="game-time-label">{game_time}</div>
        {badge}
      </div>

      <div class="matchup-row">
        <div class="team-block">
          <div class="team-name">{away}</div>
          <div class="starter-name">{away_starter}</div>
          <div class="ml-line">{away_ml}</div>
        </div>
        <div class="vs-divider">@</div>
        <div class="team-block right">
          <div class="team-name">{home}</div>
          <div class="starter-name">{home_starter}</div>
          <div class="ml-line">{home_ml}</div>
        </div>
      </div>

      <div class="prob-bar-wrap">
        <div style="height:3px;background:rgba(0,0,0,0.07);border-radius:3px;">
          <div style="height:3px;width:{hp}%;background:#1f9d55;border-radius:3px;"></div>
        </div>
        <div class="prob-labels">
          <span>{away} {ap}%</span>
          <span>{hp}% {home}</span>
        </div>
      </div>

      <div class="pick-row">
        <div>
          <div class="pick-label">Model Pick</div>
          <div class="pick-value">{predicted} <span style="color:#aaa;font-size:0.78rem;font-weight:400;">({int(round(winner_prob*100))}%)</span></div>
        </div>
        <div style="text-align:right;">
          <div class="pick-label">Edge (EV)</div>
          <div class="{ev_cls}">{ev_str}</div>
        </div>
      </div>
    </div>"""


# ── Pitcher box ───────────────────────────────────────────────────────────────

def _pitcher_box(col, team: str, starter_name: str, opponent: str,
                 pitcher_projs: pd.DataFrame, game_pk=None):
    """
    Find the correct pitcher projection for today's game.

    Priority:
      1. Match by game_pk (if proj_df has that column) — most precise
      2. Match by team AND starter name (normalized) — handles stale CSV rows
         for other dates that might have a different pitcher for the same team
      3. Fall back to team-only match (original behaviour) only as last resort
    """
    with col:
        p = pd.DataFrame()

        if not pitcher_projs.empty:
            # ── 1. game_pk match (best) ──────────────────────────────────────
            if game_pk is not None and "game_pk" in pitcher_projs.columns:
                p = pitcher_projs[
                    (pitcher_projs["game_pk"].astype(str) == str(game_pk)) &
                    (pitcher_projs["team"].str.upper() == team.upper())
                ]

            # ── 2. team + starter name match ─────────────────────────────────
            if p.empty and starter_name and starter_name != "TBD":
                norm_starter = normalize_name(starter_name)
                p = pitcher_projs[
                    (pitcher_projs["team"].str.upper() == team.upper()) &
                    (pitcher_projs["player_name"].apply(
                        lambda x: normalize_name(str(x)) == norm_starter
                    ))
                ]

            # ── 3. NO team-only fallback — that's what caused stale pitchers ──
            # If we can't match by game_pk or name, we show the API starter
            # name with a "no projection" message rather than the wrong pitcher.

        if not p.empty:
            row = p.iloc[0]
            ip  = _round(row.get("proj_ip"), 1)
            ks  = _round(row.get("proj_strikeouts"), 1)
            bb  = _round(row.get("proj_walks"), 1)
            ha  = _round(row.get("proj_hits_allowed"), 1)
            ra  = _round(row.get("proj_runs_allowed"), 1)
            # Always use the API starter_name as the display name —
            # the CSV player_name may be yesterday's pitcher if date
            # filtering didn't catch it. starter_name comes from the
            # MLB Stats API and is always today's probable.
            display_name = starter_name if starter_name and starter_name != "TBD" \
                           else row.get("player_name", "—")
            st.markdown(f"""
            <div class="pitcher-stat-block">
              <div class="pitcher-name-big">{display_name}</div>
              <div class="pitcher-sub">{team} vs {opponent}</div>
              <div class="pitcher-grid">
                <div class="p-stat-box"><div class="p-stat-label">IP</div><div class="p-stat-val">{ip}</div></div>
                <div class="p-stat-box"><div class="p-stat-label">K</div><div class="p-stat-val">{ks}</div></div>
                <div class="p-stat-box"><div class="p-stat-label">BB</div><div class="p-stat-val">{bb}</div></div>
                <div class="p-stat-box"><div class="p-stat-label">H</div><div class="p-stat-val">{ha}</div></div>
                <div class="p-stat-box"><div class="p-stat-label">ER</div><div class="p-stat-val">{ra}</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pitcher-stat-block">
              <div class="pitcher-name-big">{starter_name if starter_name != 'TBD' else '?'}</div>
              <div class="pitcher-sub">{team} — projections not yet available</div>
            </div>""", unsafe_allow_html=True)


# ── Lineup column ─────────────────────────────────────────────────────────────

def _mono(text, size="0.62rem", color="#888", bold=False):
    """Mono-spaced label for lineup cells."""
    fw = "600" if bold else "400"
    return (
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:{size};'
        f'color:{color};font-weight:{fw};line-height:1.4;">{text}</div>'
    )


# Column widths shared by header and every data row
_LINEUP_COLS = [0.35, 0.55, 3.2, 0.9, 0.9, 0.9, 0.9]


def _lineup_col(col, team: str, opponent: str,
                hitter_projs: pd.DataFrame, game_pk=None):
    """
    Render a team's projected lineup using st.columns() for the header and
    each row — Streamlit strips display:grid from inline HTML, so native
    columns are the only reliable way to get true column alignment.
    """
    with col:
        # Team heading
        st.markdown(
            f'<div style="font-family:\'Barlow Condensed\',sans-serif;'
            f'font-size:1.1rem;font-weight:700;color:#111;'
            f'margin-bottom:0.4rem;">{team}</div>',
            unsafe_allow_html=True,
        )

        # ── Header row via st.columns ─────────────────────────────────────
        hdr_cols = st.columns(_LINEUP_COLS)
        for c, lbl in zip(hdr_cols, ["#", "POS", "NAME", "H", "HR", "K", "BB"]):
            c.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:0.55rem;color:#bbb;text-transform:uppercase;'
                f'letter-spacing:0.06em;padding-bottom:0.25rem;'
                f'border-bottom:1px solid rgba(0,0,0,0.08);">{lbl}</div>',
                unsafe_allow_html=True,
            )

        if hitter_projs.empty:
            st.markdown(
                '<div style="color:#aaa;font-size:0.8rem;padding:0.5rem 0;">'
                'Run projections first.</div>',
                unsafe_allow_html=True,
            )
            return

        # ── Filter hitters ────────────────────────────────────────────────
        team_h = pd.DataFrame()

        # 1. game_pk match (most precise — prevents stale-date bleed)
        if game_pk is not None and "game_pk" in hitter_projs.columns:
            team_h = hitter_projs[
                (hitter_projs["game_pk"].astype(str) == str(game_pk)) &
                (hitter_projs["team"].str.upper() == team.upper())
            ].copy()

        # 2. team-only fallback (date already filtered upstream)
        if team_h.empty:
            team_h = hitter_projs[
                hitter_projs["team"].str.upper() == team.upper()
            ].copy()

        if team_h.empty:
            st.markdown(
                '<div style="color:#aaa;font-size:0.8rem;padding:0.5rem 0;">'
                'No lineup data for this team.</div>',
                unsafe_allow_html=True,
            )
            return

        # Sort by lineup spot
        if "lineup_spot" in team_h.columns:
            team_h["_spot"] = pd.to_numeric(team_h["lineup_spot"], errors="coerce")
            team_h = team_h.sort_values("_spot")

        # ── Data rows via st.columns ──────────────────────────────────────
        for _, hr in team_h.iterrows():
            spot = (
                str(int(hr["_spot"]))
                if "_spot" in team_h.columns and pd.notna(hr.get("_spot"))
                else "·"
            )
            pos  = str(hr.get("pos", ""))[:3]
            name = str(hr.get("player_name", "Unknown"))
            hits = _round(hr.get("proj_hits"), 2)
            hrr  = _round(hr.get("proj_hr"), 2)
            ks   = _round(hr.get("proj_strikeouts"), 2)
            bb   = _round(hr.get("proj_walks"), 2)

            row_cols = st.columns(_LINEUP_COLS)
            row_cols[0].markdown(_mono(spot, color="#bbb"), unsafe_allow_html=True)
            row_cols[1].markdown(_mono(pos,  color="#bbb", size="0.58rem"), unsafe_allow_html=True)
            row_cols[2].markdown(
                f'<div style="font-size:0.8rem;color:#111;font-weight:500;'
                f'line-height:1.4;">{name}</div>',
                unsafe_allow_html=True,
            )
            for c, v in zip(row_cols[3:], [hits, hrr, ks, bb]):
                c.markdown(_mono(v, color="#666"), unsafe_allow_html=True)

            st.markdown(
                '<div style="height:1px;background:rgba(0,0,0,0.04);'
                'margin:0 0 0.1rem 0;"></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="font-size:0.58rem;color:#ccc;margin-top:0.35rem;'
            'font-family:\'IBM Plex Mono\',monospace;">'
            'H · HR · K · BB — projected per game</div>',
            unsafe_allow_html=True,
        )


# ── Matchup detail ────────────────────────────────────────────────────────────

def _render_matchup_detail(game, pred, proj_df):
    home         = game.get("home", "")
    away         = game.get("away", "")
    home_starter = game.get("homeStarter", "TBD")
    away_starter = game.get("awayStarter", "TBD")
    game_pk      = game.get("game_pk")          # ← used for precise filtering

    st.markdown(f"""
    <div class="detail-header">{away} <span style="color:#ccc;">@</span> {home}</div>
    <div class="detail-sub">{away_starter} vs {home_starter} · {fmt_time(pred.get('commenceTime',''))}</div>
    """, unsafe_allow_html=True)

    # ── Pitching matchup ──────────────────────────────────────────────────
    pitcher_projs = get_pitcher_projs(proj_df)
    st.markdown('<div class="section-head">⚡ Pitching Matchup</div>', unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2)

    _pitcher_box(pcol1, away, away_starter, home, pitcher_projs, game_pk)
    _pitcher_box(pcol2, home, home_starter, away, pitcher_projs, game_pk)

    # ── Projected lineups ─────────────────────────────────────────────────
    hitter_projs = get_hitter_projs(proj_df)
    st.markdown(
        '<div class="section-head" style="margin-top:2rem;">🔥 Projected Lineups</div>',
        unsafe_allow_html=True,
    )
    lcol, rcol = st.columns(2)

    _lineup_col(lcol, away, home, hitter_projs, game_pk)
    _lineup_col(rcol, home, away, hitter_projs, game_pk)

    # ── Model line ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-head" style="margin-top:2rem;">📊 Model Line</div>',
        unsafe_allow_html=True,
    )
    ml1, ml2, ml3, ml4 = st.columns(4)
    with ml1:
        st.markdown(
            _stat_card("Model Pick", pred.get("predictedWinner", "—"),
                       f"{int(round(pred.get('winnerProb', 0.5) * 100))}% win prob"),
            unsafe_allow_html=True,
        )
    with ml2:
        ev_val = pred.get("ev")
        ev_str = fmt_ev(ev_val) if ev_val is not None else "—"
        ev_cls = "green" if ev_val and float(ev_val) >= 0 else "red" if ev_val else ""
        st.markdown(_stat_card("Edge (EV)", ev_str, "Expected value", ev_cls), unsafe_allow_html=True)
    with ml3:
        hw = fmt_odds(pred.get("home_ml"))
        aw = fmt_odds(pred.get("away_ml"))
        st.markdown(_stat_card("Lines", f"{away} {aw}", f"{home} {hw}"), unsafe_allow_html=True)
    with ml4:
        rec     = "✔ BET" if pred.get("betRecommended") else "Pass"
        rec_cls = "green" if pred.get("betRecommended") else ""
        st.markdown(
            _stat_card("Recommendation", rec, "Based on EV & probability", rec_cls),
            unsafe_allow_html=True,
        )


# ── Main render ───────────────────────────────────────────────────────────────

def _proj_is_stale(proj_path: Path, today: str) -> bool:
    """Returns True when CSV is missing, empty, unstamped, or from a previous day."""
    if not proj_path.exists():
        return True
    try:
        df = pd.read_csv(proj_path, nrows=5)
    except Exception:
        return True
    if df.empty:
        return True
    if "game_date" not in df.columns:
        return True   # pre-fix CSV — needs one re-run to get date stamp
    dates = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return not (dates == today).any()


def render(preds_path, proj_path, picks_file, odds_api_key, model_path, history_path):
    today = today_str()
    proj_path = Path(proj_path)

    # ── Stale projection warning ──────────────────────────────────────────
    if _proj_is_stale(proj_path, today):
        st.warning(
            "⚠️ Lineup & pitcher projections are from a previous day (or missing). "
            "Click **Run Projections** to fetch today's lineups before viewing matchup details."
        )
        if st.button("⚡ Run Projections Now", key="run_proj_stale_btn"):
            try:
                import sys
                sys.path.insert(0, str(proj_path.parent.parent))
                from hitterspitchers_today import run_projections
                with st.spinner("Fetching today's lineups and running projections…"):
                    run_projections()
                st.success("✅ Projections updated.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Projection error: {e}")

    # ── Run model button ──────────────────────────────────────────────────
    if st.button("🔄 Run Model", key="run_model_btn"):
        from daily_mlb_model_runner import run
        with st.spinner("Running today's game model..."):
            run(
                date=today,
                odds_api_key=odds_api_key,
                model_path=str(model_path),
                history_path=str(history_path),
                min_ev=0.001,
                save_today_csv=True,
                save_pick_log=True,
                picks_file=str(picks_file),
            )
        st.success("Updated game predictions.")
        st.cache_data.clear()
        st.rerun()

    # ── Load data ─────────────────────────────────────────────────────────
    # fetch_schedule hits the MLB Stats API (cached 5 min) — always today's
    # probable pitchers, never stale.
    schedule  = fetch_schedule(today)           # list[dict] from MLB API
    preds     = load_predictions(preds_path)    # list[dict] from CSV
    proj_df   = load_projections(proj_path)     # DataFrame (hitters + pitchers)

    # ── Date-filter proj_df to today only ────────────────────────────────
    # Drops any rows not stamped with today's date. After the one-line fix
    # to hitterspitchers_today.py adds game_date, this removes all stale rows.
    if "game_date" in proj_df.columns:
        proj_df["game_date"] = pd.to_datetime(proj_df["game_date"], errors="coerce")
        proj_df = proj_df[proj_df["game_date"].dt.strftime("%Y-%m-%d") == today]

    if not schedule:
        st.warning("No games found for today from the MLB API.")
        return

    # ── Build a pred lookup keyed by away@home ────────────────────────────
    pred_map: dict[str, dict] = {}
    for p in preds:
        k = team_key(p["home"], p["away"])
        pred_map[k] = p

    # ── Session state: which game is expanded ─────────────────────────────
    if "selected_game_pk" not in st.session_state:
        st.session_state["selected_game_pk"] = None

    # ── Render game cards ─────────────────────────────────────────────────
    for game in schedule:
        home = game.get("home", "")
        away = game.get("away", "")
        game_pk = game.get("game_pk")
        k = team_key(home, away)
        pred = pred_map.get(k, {
            "homeWinProb": 0.5, "awayWinProb": 0.5,
            "predictedWinner": home, "winnerProb": 0.5,
            "home_ml": None, "away_ml": None,
            "ev": None, "betRecommended": False,
            "commenceTime": game.get("commence_time"),
        })

        # Inject today's starters from the MLB API into the game dict
        # so the card always shows today's probable pitchers.
        game["homeStarter"] = game.get("homeStarter", "TBD")
        game["awayStarter"] = game.get("awayStarter", "TBD")

        st.markdown(_game_card_html(game, pred), unsafe_allow_html=True)

        # Expand / collapse toggle
        is_open = st.session_state["selected_game_pk"] == game_pk
        btn_label = "▲ Close" if is_open else "▼ Matchup Detail"
        if st.button(btn_label, key=f"toggle_{game_pk}"):
            st.session_state["selected_game_pk"] = None if is_open else game_pk
            st.rerun()

        if is_open:
            with st.container():
                st.markdown(
                    '<div style="background:#fff;border:1px solid rgba(0,0,0,0.08);'
                    'border-radius:12px;padding:1.5rem;margin-bottom:1rem;">',
                    unsafe_allow_html=True,
                )
                _render_matchup_detail(game, pred, proj_df)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)