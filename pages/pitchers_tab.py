"""
pages/pitchers_tab.py — fixed v2 (confidence column removed)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from components.data_loader import get_pitcher_projs, load_projections


def _safe(val, n=2, default="—"):
    try:
        f = float(val)
        return "—" if np.isnan(f) else round(f, n)
    except Exception:
        return default


def render(proj_path: Path):
    proj_df  = load_projections(proj_path)
    pitchers = get_pitcher_projs(proj_df)

    with st.expander("🔄 Run pitcher projections", expanded=pitchers.empty):
        st.markdown('<div style="color:#6a7d94;font-size:0.82rem;margin-bottom:1rem;">Scrapes today\'s probable pitchers and runs projection models.</div>', unsafe_allow_html=True)
        if st.button("Run Projections Now", key="run_proj_btn"):
            try:
                import sys
                sys.path.insert(0, str(proj_path.parent.parent))
                from hitterspitchers_today import run_projections
                with st.spinner("Running projections…"):
                    run_projections()
                st.success("✅ Projections saved.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Projection error: {e}")

    if pitchers.empty:
        st.markdown('<div style="color:#6a7d94;padding:2rem 0;font-size:0.9rem;">No pitcher projections found. Run projections above or run <code>python hitterspitchers_today.py</code> from your terminal.</div>', unsafe_allow_html=True)
        return

    sort_col, filter_col = st.columns([1, 2])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            ["proj_strikeouts", "proj_ip", "proj_walks", "proj_hits_allowed", "proj_runs_allowed"],
            format_func=lambda x: {
                "proj_strikeouts": "Proj K",
                "proj_ip": "Proj IP",
                "proj_walks": "Proj BB",
                "proj_hits_allowed": "Proj H Allowed",
                "proj_runs_allowed": "Proj ER",
            }.get(x, x),
            key="pitcher_sort",
        )
    with filter_col:
        all_teams   = sorted(pitchers["team"].dropna().unique().tolist())
        team_filter = st.multiselect("Filter by team", options=all_teams, default=[], key="pitcher_team_filter")

    filtered = pitchers.copy()
    if team_filter:
        filtered = filtered[filtered["team"].isin(team_filter)]
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)

    st.markdown(f'<div style="color:#6a7d94;font-size:0.78rem;margin-bottom:1rem;">{len(filtered)} pitcher{"s" if len(filtered)!=1 else ""}</div>', unsafe_allow_html=True)

    # Header — NO confidence column
    cols_def = [2.5, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8]
    labels   = ["Pitcher", "Team", "Opp", "IP", "K", "BB", "H", "ER"]
    header_cols = st.columns(cols_def)
    for col, lbl in zip(header_cols, labels):
        col.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
            f'color:#6a7d94;text-transform:uppercase;letter-spacing:0.08em;'
            f'padding-bottom:0.3rem;border-bottom:1px solid rgba(255,255,255,0.06);">{lbl}</div>',
            unsafe_allow_html=True,
        )

    for _, row in filtered.iterrows():
        row_cols = st.columns(cols_def)
        vals = [
            str(row.get("player_name", "—")),
            str(row.get("team", "—")).upper(),
            str(row.get("opponent", "—")).upper(),
            _safe(row.get("proj_ip"), 1),
            _safe(row.get("proj_strikeouts"), 1),
            _safe(row.get("proj_walks"), 1),
            _safe(row.get("proj_hits_allowed"), 1),
            _safe(row.get("proj_runs_allowed"), 1),
        ]
        for j, (col, val) in enumerate(zip(row_cols, vals)):
            is_name = j == 0
            style = (
                "font-size:0.82rem;color:#f0f2f5;font-weight:500;"
                if is_name
                else "font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#a0b0c4;"
            )
            col.markdown(f'<div style="{style}padding:0.4rem 0;">{val}</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:rgba(255,255,255,0.05);margin:0;"></div>', unsafe_allow_html=True)