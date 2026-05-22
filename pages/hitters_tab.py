"""
pages/hitters_tab.py
====================
Hitter Projections tab — lineup cards per matchup, sortable table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components.data_loader import get_hitter_projs, load_projections


def _safe(val, n=2, default="—"):
    try:
        f = float(val)
        return "—" if np.isnan(f) else round(f, n)
    except Exception:
        return default


def render(proj_path: Path):
    proj_df = load_projections(proj_path)
    hitters = get_hitter_projs(proj_df)

    # ── Run projections button ─────────────────────────────────────────────
    with st.expander("🔄 Run hitter/pitcher projections", expanded=hitters.empty):
        st.markdown('<div style="color:#6a7d94;font-size:0.82rem;margin-bottom:1rem;">Scrapes today\'s lineups and runs projection models. Saves to outputs/hitterspitchers_today.csv.</div>', unsafe_allow_html=True)
        if st.button("Run Projections Now", key="run_hitter_proj_btn"):
            try:
                import sys
                sys.path.insert(0, str(proj_path.parent.parent))
                from hitterspitchers_today import run_projections
                with st.spinner("Fetching lineups and running projections…"):
                    run_projections()
                st.success("✅ Projections saved.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Projection error: {e}")

    if hitters.empty:
        st.markdown('<div style="color:#6a7d94;padding:2rem 0;font-size:0.9rem;">No hitter projections found. Run projections above or run <code>python hitterspitchers_today.py</code> from your terminal.</div>', unsafe_allow_html=True)
        return

    # ── View toggle ────────────────────────────────────────────────────────
    view_mode = st.radio("View", ["Matchup Cards", "Full Table"], horizontal=True, key="hitter_view")

    # ── Matchup cards view ─────────────────────────────────────────────────
    if view_mode == "Matchup Cards":
        matchups = []
        if "team" in hitters.columns and "opponent" in hitters.columns:
            pairs = hitters[["team", "opponent"]].drop_duplicates()
            seen = set()
            for _, pair in pairs.iterrows():
                key = tuple(sorted([pair["team"].upper(), pair["opponent"].upper()]))
                if key not in seen:
                    seen.add(key)
                    matchups.append((pair["team"], pair["opponent"]))

        for home_team, away_team in matchups:
            st.markdown(f'<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.2rem;font-weight:700;color:#f0f2f5;margin-top:1.5rem;margin-bottom:0.75rem;">{away_team} <span style="color:#c8d6e5;">@</span> {home_team}</div>', unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)

            for col, team in [(mc1, away_team), (mc2, home_team)]:
                team_h = hitters[hitters["team"].str.upper() == team.upper()].copy()
                if "lineup_spot" in team_h.columns:
                    team_h["_spot"] = pd.to_numeric(team_h["lineup_spot"], errors="coerce")
                    team_h = team_h.sort_values("_spot")

                with col:
                    st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.62rem;color:#6a7d94;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">{team}</div>', unsafe_allow_html=True)
                    # header
                    st.markdown("""
                    <div style="display:grid;grid-template-columns:1.8rem 2.2rem 1fr 2.8rem 2.8rem 2.8rem 2.8rem 2.8rem;gap:0;
                    font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#8a9db5;
                    padding:0.2rem 0.4rem;border-bottom:1px solid rgba(255,255,255,0.07);">
                    <span>#</span><span>POS</span><span>NAME</span>
                    <span style="text-align:right">PA</span>
                    <span style="text-align:right">H</span>
                    <span style="text-align:right">HR</span>
                    <span style="text-align:right">K</span>
                    <span style="text-align:right">BB</span>
                    </div>""", unsafe_allow_html=True)

                    if team_h.empty:
                        st.markdown('<div style="color:#6a7d94;font-size:0.78rem;padding:0.5rem 0.4rem;">No lineup data</div>', unsafe_allow_html=True)
                    else:
                        for _, hr in team_h.iterrows():
                            spot = str(int(hr["_spot"])) if "_spot" in team_h.columns and pd.notna(hr.get("_spot")) else "·"
                            pos = str(hr.get("pos", ""))[:3]
                            name = str(hr.get("player_name", "—"))
                            pa = _safe(hr.get("proj_pa"), 1)
                            hits = _safe(hr.get("proj_hits"), 2)
                            hrr = _safe(hr.get("proj_hr"), 2)
                            ks = _safe(hr.get("proj_strikeouts"), 2)
                            bb = _safe(hr.get("proj_walks"), 2)
                            conf = str(hr.get("confidence", "low")).lower()
                            conf_dot = {"high": "#1f9d55", "medium": "#f0c040", "low": "#c0392b"}.get(conf, "#555")

                            st.markdown(f"""
                            <div style="display:grid;grid-template-columns:1.8rem 2.2rem 1fr 2.8rem 2.8rem 2.8rem 2.8rem 2.8rem;gap:0;
                            padding:0.3rem 0.4rem;border-bottom:1px solid rgba(255,255,255,0.04);align-items:center;">
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#8a9db5;">{spot}</span>
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#6a7d94;">{pos}</span>
                              <span style="font-size:0.78rem;color:#f0f2f5;font-weight:500;">{name} <span style="color:{conf_dot};font-size:0.5rem;">●</span></span>
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#6a7d94;text-align:right;">{pa}</span>
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#a0b0c4;text-align:right;">{hits}</span>
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#a0b0c4;text-align:right;">{hrr}</span>
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#a0b0c4;text-align:right;">{ks}</span>
                              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#a0b0c4;text-align:right;">{bb}</span>
                            </div>""", unsafe_allow_html=True)

            st.markdown('<div style="height:1px;background:rgba(255,255,255,0.05);margin:1.5rem 0 0 0;"></div>', unsafe_allow_html=True)

    # ── Full table view ────────────────────────────────────────────────────
    else:
        sc1, sc2, sc3 = st.columns([1.2, 1.2, 2])
        with sc1:
            sort_by = st.selectbox(
                "Sort by",
                ["proj_hits", "proj_hr", "proj_pa", "proj_strikeouts", "proj_walks", "proj_runs", "proj_rbi"],
                format_func=lambda x: {
                    "proj_hits": "Proj H", "proj_hr": "Proj HR", "proj_pa": "Proj PA",
                    "proj_strikeouts": "Proj K", "proj_walks": "Proj BB",
                    "proj_runs": "Proj R", "proj_rbi": "Proj RBI",
                }.get(x, x),
                key="hitter_sort",
            )
        with sc2:
            conf_filter = st.selectbox("Confidence", ["All", "High", "Medium", "Low"], key="hitter_conf")
        with sc3:
            all_teams = sorted(hitters["team"].dropna().unique().tolist())
            team_filter = st.multiselect("Teams", all_teams, default=[], key="hitter_team")

        filt = hitters.copy()
        if team_filter:
            filt = filt[filt["team"].isin(team_filter)]
        if conf_filter != "All":
            filt = filt[filt["confidence"].str.lower() == conf_filter.lower()]
        if sort_by in filt.columns:
            filt = filt.sort_values(sort_by, ascending=False)

        st.markdown(f'<div style="color:#6a7d94;font-size:0.78rem;margin-bottom:1rem;">{len(filt)} hitters</div>', unsafe_allow_html=True)

        # header
        hcols = st.columns([2.5, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        hlbls = ["Player", "Team", "Opp", "Spot", "PA", "H", "HR", "K", "BB", "R", "Conf"]
        for col, lbl in zip(hcols, hlbls):
            col.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;color:#6a7d94;text-transform:uppercase;letter-spacing:0.08em;padding-bottom:0.3rem;border-bottom:1px solid rgba(255,255,255,0.07);">{lbl}</div>', unsafe_allow_html=True)

        for _, row in filt.iterrows():
            conf = str(row.get("confidence", "low")).lower()
            conf_color = {"high": "#1f9d55", "medium": "#f0c040", "low": "#c0392b"}.get(conf, "#555")
            spot = str(int(float(row["lineup_spot"]))) if pd.notna(row.get("lineup_spot")) else "—"

            vals_row = [
                str(row.get("player_name", "—")),
                str(row.get("team", "")).upper(),
                str(row.get("opponent", "")).upper(),
                spot,
                _safe(row.get("proj_pa"), 1),
                _safe(row.get("proj_hits"), 2),
                _safe(row.get("proj_hr"), 2),
                _safe(row.get("proj_strikeouts"), 2),
                _safe(row.get("proj_walks"), 2),
                _safe(row.get("proj_runs"), 2),
            ]

            rcols = st.columns([2.5, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
            for i, (col, val) in enumerate(zip(rcols[:10], vals_row)):
                is_name = i == 0
                s = "font-size:0.82rem;color:#f0f2f5;font-weight:500;" if is_name else "font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#a0b0c4;"
                col.markdown(f'<div style="{s}padding:0.35rem 0;">{val}</div>', unsafe_allow_html=True)
            rcols[10].markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;color:{conf_color};padding:0.35rem 0;">{conf.upper()[:3]}</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:1px;background:rgba(255,255,255,0.04);"></div>', unsafe_allow_html=True)