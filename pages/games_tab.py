"""
pages/games_tab.py  — fixed v2
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components.data_loader import (
    fetch_schedule, fmt_ev, fmt_odds, fmt_pct, fmt_time,
    get_hitter_projs, get_pitcher_projs,
    load_accuracy, load_predictions, load_projections,
    team_key, today_str,
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

# ── Game card HTML (teams on same line, ET time, Good Value badge) ─────────────

def _game_card_html(game: dict, pred: dict) -> str:
    is_value = bool(pred.get("betRecommended", False))
    home_prob = pred.get("homeWinProb", 0.5) or 0.5
    away_prob = pred.get("awayWinProb", 0.5) or 0.5
    ev        = pred.get("ev")
    predicted = pred.get("predictedWinner", "")
    winner_prob = pred.get("winnerProb", 0.5) or 0.5

    badge     = '<span class="badge-value">◆ Good Value</span>' if is_value else '<span class="badge-pass">Pass</span>'
    card_cls  = "game-card value" if is_value else "game-card"
    game_time = fmt_time(pred.get("commenceTime") or game.get("commence_time", ""))

    home = game.get("home", "")
    away = game.get("away", "")
    home_starter = game.get("homeStarter", "TBD")
    away_starter = game.get("awayStarter", "TBD")
    home_ml = fmt_odds(pred.get("home_ml"))
    away_ml = fmt_odds(pred.get("away_ml"))

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
        <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:3px;">
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
          <div class="pick-value">{predicted} <span style="color:#6a7d94;font-size:0.78rem;font-weight:400;">({int(round(winner_prob*100))}%)</span></div>
        </div>
        <div style="text-align:right;">
          <div class="pick-label">Edge (EV)</div>
          <div class="{ev_cls}">{ev_str}</div>
        </div>
      </div>
    </div>"""

# ── Matchup detail ─────────────────────────────────────────────────────────────

def _render_matchup_detail(game, pred, proj_df):
    home = game.get("home", "")
    away = game.get("away", "")
    home_starter = game.get("homeStarter", "TBD")
    away_starter = game.get("awayStarter", "TBD")

    st.markdown(f"""
    <div class="detail-header">{away} <span style="color:#8a9db5;">@</span> {home}</div>
    <div class="detail-sub">{home_starter} vs {away_starter} · {fmt_time(pred.get('commenceTime',''))}</div>
    """, unsafe_allow_html=True)

    pitcher_projs = get_pitcher_projs(proj_df)
    st.markdown('<div class="section-head">⚡ Pitching Matchup</div>', unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2)

    def _pitcher_box(col, team, starter_name, opponent):
        with col:
            p = pd.DataFrame()
            if not pitcher_projs.empty:
                mask = pitcher_projs["team"].str.upper() == team.upper()
                p = pitcher_projs[mask]
                if p.empty and starter_name != "TBD":
                    from components.data_loader import normalize_name
                    norm = normalize_name(starter_name)
                    p = pitcher_projs[pitcher_projs["player_name"].apply(
                        lambda x: normalize_name(str(x)) == norm
                    )]
            if not p.empty:
                row = p.iloc[0]
                ip = _round(row.get("proj_ip"), 1)
                ks = _round(row.get("proj_strikeouts"), 1)
                bb = _round(row.get("proj_walks"), 1)
                ha = _round(row.get("proj_hits_allowed"), 1)
                ra = _round(row.get("proj_runs_allowed"), 1)
                st.markdown(f"""
                <div class="pitcher-stat-block">
                  <div class="pitcher-name-big">{row.get('player_name','—')}</div>
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

    _pitcher_box(pcol1, away, away_starter, home)
    _pitcher_box(pcol2, home, home_starter, away)

    # Lineups
    hitter_projs = get_hitter_projs(proj_df)
    st.markdown('<div class="section-head" style="margin-top:2rem;">🔥 Projected Lineups</div>', unsafe_allow_html=True)
    lcol, rcol = st.columns(2)

    def _lineup_col(col, team, opponent):
        with col:
            st.markdown(f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.1rem;font-weight:700;color:#f0f2f5;margin-bottom:0.5rem;">{team}</div>', unsafe_allow_html=True)
            st.markdown("""
                        <div style="
                        display:grid;
                        grid-template-columns:1.8rem 2.2rem 1fr 3rem 3rem 3rem 3rem;
                        gap:0;
                        padding:0.25rem 0;
                        border-bottom:1px solid rgba(0,0,0,0.12);
                        font-family:'IBM Plex Mono', monospace;
                        font-size:0.6rem;
                        color:#6a7d94;
                        align-items:center;
                    ">
                        <span style="text-align:left;">#</span>
                        <span style="text-align:left;">POS</span>
                        <span style="text-align:left;">PLAYER</span>
                        <span style="text-align:right;">H</span>
                        <span style="text-align:right;">HR</span>
                        <span style="text-align:right;">K</span>
                        <span style="text-align:right;">BB</span>
                    </div>
                    """, unsafe_allow_html=True)
            if not hitter_projs.empty:
                team_h = hitter_projs[hitter_projs["team"].str.upper() == team.upper()].copy()
                if not team_h.empty:
                    if "lineup_spot" in team_h.columns:
                        team_h["_spot"] = pd.to_numeric(team_h["lineup_spot"], errors="coerce")
                        team_h = team_h.sort_values("_spot")
                    for _, hr in team_h.iterrows():
                        spot = str(int(hr["_spot"])) if "_spot" in team_h.columns and pd.notna(hr.get("_spot")) else "·"
                        pos  = str(hr.get("pos", ""))[:3]
                        name = str(hr.get("player_name", "Unknown"))
                        hits = _round(hr.get("proj_hits"), 2)
                        hrr  = _round(hr.get("proj_hr"), 2)
                        ks   = _round(hr.get("proj_strikeouts"), 2)
                        bb   = _round(hr.get("proj_walks"), 2)
                        st.markdown(f"""
                        <div style="display:grid;grid-template-columns:1.8rem 2.2rem 1fr 3rem 3rem 3rem 3rem;gap:0;
                        padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);align-items:center;">
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a9db5;">{spot}</span>
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#8a9db5;">{pos}</span>
                          <span style="font-size:0.8rem;color:#f0f2f5;font-weight:500;">{name}</span>
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;color:#a0b0c4;text-align:right;">{hits}</span>
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;color:#a0b0c4;text-align:right;">{hrr}</span>
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;color:#a0b0c4;text-align:right;">{ks}</span>
                          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;color:#a0b0c4;text-align:right;">{bb}</span>
                        </div>""", unsafe_allow_html=True)
                    st.markdown('<div style="font-size:0.58rem;color:#8a9db5;margin-top:0.35rem;font-family:\'IBM Plex Mono\',monospace;">H · HR · K · BB — projected per game</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#6a7d94;font-size:0.8rem;padding:0.5rem 0;">No lineup data for this team.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#6a7d94;font-size:0.8rem;padding:0.5rem 0;">Run projections first.</div>', unsafe_allow_html=True)

    _lineup_col(lcol, away, home)
    _lineup_col(rcol, home, away)

    # Model line
    st.markdown('<div class="section-head" style="margin-top:2rem;">📊 Model Line</div>', unsafe_allow_html=True)
    ml1, ml2, ml3, ml4 = st.columns(4)
    with ml1:
        st.markdown(_stat_card("Model Pick", pred.get("predictedWinner","—"), f"{int(round(pred.get('winnerProb',0.5)*100))}% win prob"), unsafe_allow_html=True)
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
        rec     = "✓ BET" if pred.get("betRecommended") else "Pass"
        rec_cls = "green" if pred.get("betRecommended") else ""
        st.markdown(_stat_card("Recommendation", rec, "Based on EV & probability", rec_cls), unsafe_allow_html=True)

    if st.button("← Back to all games", key="back_btn"):
        st.session_state.pop("selected_game", None)
        st.rerun()

# ── Main render ────────────────────────────────────────────────────────────────

def render(preds_path, proj_path, picks_file, odds_api_key, model_path, history_path):

    with st.expander("🔄 Update today's predictions", expanded=False):
        st.markdown('<div style="color:#6a7d94;font-size:0.82rem;margin-bottom:1rem;">Fetches live odds, runs the betting model, and saves predictions to outputs/.</div>', unsafe_allow_html=True)
        if st.button("Run Model Now", key="run_model_btn"):
            try:
                from daily_mlb_model_runner import run as model_run
                with st.spinner("Running model…"):
                    model_run(
                        date=today_str(),
                        odds_api_key=odds_api_key,
                        model_path=str(model_path),
                        history_path=str(history_path),
                        min_ev=0.001,
                        save_today_csv=True,
                        save_pick_log=True,
                        picks_file=str(picks_file),
                    )
                st.success("✅ Model run complete.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Model error: {e}")

    games    = fetch_schedule(today_str())
    preds    = load_predictions(preds_path)
    proj_df  = load_projections(proj_path)
    acc      = load_accuracy(picks_file)
    pred_map = {team_key(p["home"], p["away"]): p for p in preds}
    bets     = [p for p in preds if p.get("betRecommended")]
    bet_probs = [p["winnerProb"] for p in bets if p.get("winnerProb")]
    avg_prob  = sum(bet_probs) / len(bet_probs) if bet_probs else None

    # Matchup detail view
    if "selected_game" in st.session_state:
        sel  = st.session_state["selected_game"]
        key  = team_key(sel["home"], sel["away"])
        pred = pred_map.get(key, {
            "home": sel["home"], "away": sel["away"],
            "homeWinProb": 0.5, "awayWinProb": 0.5,
            "predictedWinner": sel["home"], "winnerProb": 0.5,
            "home_ml": None, "away_ml": None, "ev": None,
            "betRecommended": False, "commenceTime": sel.get("commence_time"),
        })
        _render_matchup_detail(sel, pred, proj_df)
        return

    # Summary stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_stat_card("Games Today", str(len(games)), "On today's slate"), unsafe_allow_html=True)
    with c2:
        st.markdown(_stat_card("Good Value", str(len(bets)), "Positive EV picks", "green"), unsafe_allow_html=True)
    with c3:
        avg_s = f"{int(round(avg_prob*100))}%" if avg_prob else "—"
        st.markdown(_stat_card("Avg Win Prob", avg_s, "On value picks"), unsafe_allow_html=True)
    with c4:
        if acc["total_games"]:
            acc_s = f"{acc['overall_acc']*100:.1f}%"
            sub_s = f"{acc['correct_games']}/{acc['total_games']} graded"
        else:
            acc_s, sub_s = "—", "no graded picks yet"
        st.markdown(_stat_card("Season Accuracy", acc_s, sub_s, "green" if acc["total_games"] else ""), unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Filter
    if "show_value_only" not in st.session_state:
        st.session_state.show_value_only = False

    fl, fr = st.columns([2, 1])
    with fl:
        st.markdown(f'<div style="color:#6a7d94;font-size:0.82rem;">{len(games)} game{"s" if len(games)!=1 else ""} today</div>', unsafe_allow_html=True)
    with fr:
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("All Games", key="filter_all"):
                st.session_state.show_value_only = False
        with fb2:
            if st.button("Value Only", key="filter_value"):
                st.session_state.show_value_only = True

    filtered = games
    if st.session_state.show_value_only:
        filtered = [g for g in games if pred_map.get(team_key(g["home"], g["away"]), {}).get("betRecommended")]

    if not filtered:
        st.markdown('<div style="color:#6a7d94;padding:2rem 0;">No games match this filter.</div>', unsafe_allow_html=True)
        return

    # Game cards — button INSIDE the card via columns trick
    cols = st.columns(3)
    for i, game in enumerate(filtered):
        key  = team_key(game["home"], game["away"])
        pred = pred_map.get(key, {
            "home": game["home"], "away": game["away"],
            "homeWinProb": 0.5, "awayWinProb": 0.5,
            "predictedWinner": game["home"], "winnerProb": 0.5,
            "home_ml": None, "away_ml": None, "ev": None,
            "betRecommended": False, "commenceTime": game.get("commence_time"),
        })
        with cols[i % 3]:
            st.markdown(_game_card_html(game, pred), unsafe_allow_html=True)
            if st.button(f"View Matchup →", key=f"detail_{i}_{game['home']}_{game['away']}"):
                st.session_state["selected_game"] = game
                st.rerun()
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)