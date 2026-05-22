"""
MLB Edge — Main App Entry Point
================================
Run: streamlit run app.py
"""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Edge",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CONSTANTS ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "afa28350c34fba9f318ecd7ae4e21b63")
MODEL_PATH   = BASE_DIR / "betting_model.pkl"
HISTORY_PATH = BASE_DIR / "2025_model_data.csv"
PICKS_FILE   = BASE_DIR / "2026_picks_accuracy.csv"
PROJ_PATH    = OUTPUT_DIR / "hitterspitchers_today.csv"
PREDS_PATH   = OUTPUT_DIR / "today_predictions_with_ev.csv"

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Sportsbook Dark Theme ──
   Background:  #0f1923  (deep navy-charcoal — DK/FD standard)
   Surface:     #1a2535  (card backgrounds)
   Surface-2:   #243044  (raised elements, stat boxes)
   Border:      rgba(255,255,255,0.07)
   Green:       #00c853  (sharp action green)
   Green-dim:   #00a844
   Red:         #e53935  (loss red)
   Text-primary:#f0f2f5
   Text-muted:  #7a8a9e
   Text-dim:    #4a5a6e
*/

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0f1923 !important;
    color: #f0f2f5 !important;
}
.stApp { background-color: #0f1923 !important; }
.block-container { padding-top: 0.8rem !important; padding-bottom: 2rem !important; max-width: 1600px; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Topbar ── */
.topbar {
    background: #0f1923;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 0.65rem 0 0.55rem 0;
    margin-bottom: 0;
    display: flex;
    align-items: center;
}
.logo-mark {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #f0f2f5;
    letter-spacing: 0.01em;
    text-transform: uppercase;
}
.logo-mark span { color: #00c853; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4a5a6e;
    padding: 0.7rem 1.4rem;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00c853 !important;
    border-bottom: 2px solid #00c853 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.8rem; }

/* ── Stat cards ── */
.stat-card {
    background: #1a2535;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1.1rem 1.1rem 1rem 1.1rem;
}
.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    color: #4a5a6e;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.45rem;
}
.stat-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f2f5;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.stat-value.green { color: #00c853; }
.stat-value.red   { color: #e53935; }
.stat-sub { color: #4a5a6e; font-size: 0.72rem; }

/* ── Game card ── */
.game-card {
    background: #1a2535;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.6rem;
}
.game-card.value { border-color: rgba(0,200,83,0.4); }
.game-time-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a5a6e;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
/* Teams row — same baseline */
.matchup-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0.55rem 0 0.3rem 0;
}
.team-block { display: flex; flex-direction: column; }
.team-block.right { text-align: right; }
.team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f2f5;
    line-height: 1;
}
.starter-name { font-size: 0.72rem; color: #4a5a6e; margin-top: 0.15rem; }
.ml-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #7a8a9e;
    margin-top: 0.1rem;
}
.vs-divider {
    color: #243044;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0 0.5rem;
    flex-shrink: 0;
    align-self: center;
}
/* Prob bar */
.prob-bar-wrap { margin: 0.55rem 0 0.3rem 0; }
.prob-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #4a5a6e;
    margin-top: 0.18rem;
}
/* Pick row */
.pick-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 0.65rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 0.65rem;
}
.pick-label { color: #4a5a6e; font-size: 0.62rem; font-family:'IBM Plex Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; }
.pick-value { color: #f0f2f5; font-size: 0.9rem; font-weight: 600; margin-top: 0.1rem; }
/* Badges */
.badge-value {
    background: rgba(0,200,83,0.12);
    border: 1px solid rgba(0,200,83,0.35);
    color: #00c853;
    font-size: 0.62rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 600;
}
.badge-pass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #4a5a6e;
    font-size: 0.62rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    padding: 0.18rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
}
.ev-pos { color: #00c853; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; }
.ev-neg { color: #e53935; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; font-weight:600; }
.ev-na  { color: #4a5a6e; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; }

/* ── Matchup detail ── */
.detail-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f0f2f5;
    letter-spacing: 0.01em;
    line-height: 1;
}
.detail-sub { color: #4a5a6e; font-size: 0.8rem; margin-top: 0.3rem; margin-bottom: 1.8rem; }
.section-head {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: #00c853;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(0,200,83,0.2);
}

/* ── Pitcher stat block ── */
.pitcher-stat-block {
    background: #1a2535;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.pitcher-name-big {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f0f2f5;
    margin-bottom: 0.1rem;
}
.pitcher-sub { color: #4a5a6e; font-size: 0.72rem; margin-bottom: 0.7rem; }
.pitcher-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; }
.p-stat-box {
    background: #243044;
    border-radius: 6px;
    padding: 0.4rem 0.5rem;
    text-align: center;
}
.p-stat-label { font-size: 0.58rem; color: #4a5a6e; font-family:'IBM Plex Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; }
.p-stat-val { font-family:'Barlow Condensed',sans-serif; font-size:1.4rem; font-weight:700; color:#f0f2f5; }

/* ── Accuracy wrap ── */
.accuracy-wrap {
    background: #1a2535;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1.2rem;
    margin-top: 1.5rem;
}
.accuracy-title { font-family:'Barlow Condensed',sans-serif; font-size:1.3rem; font-weight:700; color:#f0f2f5; margin-bottom:0.1rem; }
.accuracy-sub { color:#4a5a6e; font-size:0.72rem; margin-bottom:0.8rem; }

/* ── Buttons ── */
.stButton > button {
    background: #00c853 !important;
    color: #000000 !important;
    border: none !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.3rem !important;
}
.stButton > button:hover { background: #00a844 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1a2535;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* ── Native Streamlit widget overrides for dark mode ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: #1a2535 !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #f0f2f5 !important;
}
.stRadio > div { color: #7a8a9e !important; }
[data-testid="stMarkdownContainer"] p { color: #7a8a9e; }
</style>
""", unsafe_allow_html=True)

# ─── TOPBAR (logo only) ───────────────────────────────────────
st.markdown('<div class="topbar"><div class="logo-mark">MLB <span>Edge</span></div></div>', unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────
tab_games, tab_pitchers, tab_hitters, tab_accuracy = st.tabs([
    "Games & Moneylines",
    "Pitchers Predictions",
    "Hitters Predictions",
    "Season Accuracy",
])

from pages.games_tab    import render as render_games
from pages.pitchers_tab import render as render_pitchers
from pages.hitters_tab  import render as render_hitters
from pages.accuracy_tab import render as render_accuracy

with tab_games:
    render_games(
        preds_path=PREDS_PATH,
        proj_path=PROJ_PATH,
        picks_file=PICKS_FILE,
        odds_api_key=ODDS_API_KEY,
        model_path=MODEL_PATH,
        history_path=HISTORY_PATH,
    )

with tab_pitchers:
    render_pitchers(proj_path=PROJ_PATH)

with tab_hitters:
    render_hitters(proj_path=PROJ_PATH)

with tab_accuracy:
    render_accuracy(picks_file=PICKS_FILE)