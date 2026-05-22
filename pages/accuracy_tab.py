"""
pages/accuracy_tab.py
=====================
Season Accuracy tab — cumulative accuracy chart + pick log.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_loader import load_accuracy


def render(picks_file: Path):
    acc = load_accuracy(picks_file)

    # ── Re-grade button ────────────────────────────────────────────────────
    with st.expander("🔄 Re-grade completed games", expanded=False):
        st.markdown('<div style="color:#6a7d94;font-size:0.82rem;margin-bottom:1rem;">Fetches final scores and updates correct/incorrect for all logged picks.</div>', unsafe_allow_html=True)
        if st.button("Grade Picks Now", key="grade_btn"):
            try:
                from daily_mlb_model_runner import grade_saved_picks
                with st.spinner("Grading picks…"):
                    grade_saved_picks(picks_file=str(picks_file), output_file=str(picks_file))
                st.success("✅ Picks re-graded.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error grading picks: {e}")

    if not acc["total_games"]:
        st.markdown("""
        <div style="color:#6a7d94;padding:2rem 0;">
          No graded picks found yet. Once games complete and picks are graded, the accuracy chart will appear here.
        </div>""", unsafe_allow_html=True)
        return

    # ── Summary cards ──────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns(4)
    cards = [
        ("Season Accuracy", f"{acc['overall_acc']*100:.1f}%", f"{acc['correct_games']} correct picks", "green"),
        ("Graded Picks", str(acc["total_games"]), "Total logged & graded", ""),
        ("Correct Picks", str(acc["correct_games"]), "Model was right", "green"),
        ("Wrong Picks", str(acc["total_games"] - acc["correct_games"]), "Model was wrong", "red"),
    ]
    for col, (lbl, val, sub, cls) in zip([sc1, sc2, sc3, sc4], cards):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{lbl}</div>
            <div class="stat-value {cls}">{val}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Accuracy chart ─────────────────────────────────────────────────────
    y_vals = [v * 100 for v in acc["cumulative_accuracy"]]
    y_min = max(0, min(y_vals) - 5)
    y_max = min(100, max(y_vals) + 5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=acc["labels"],
        y=y_vals,
        mode="lines",
        line=dict(color="#00c853", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,200,83,0.08)",
        hovertemplate="%{y:.1f}% cumulative accuracy<extra></extra>",
    ))

    # 50% reference line
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.12)", line_width=1)

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#1a2535",
        plot_bgcolor="#1a2535",
        font=dict(color="#f0f2f5", family="IBM Plex Mono"),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#4a5a6e", size=10),
            tickangle=-30,
        ),
        yaxis=dict(
            title=None, range=[y_min, y_max],
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#4a5a6e", size=10),
            ticksuffix="%",
        ),
    )

    st.markdown('<div class="accuracy-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="accuracy-title">Cumulative Pick Accuracy</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="accuracy-sub">{acc["overall_acc"]*100:.1f}% overall · {acc["correct_games"]}/{acc["total_games"]} graded picks · 2026 season</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recent picks table ─────────────────────────────────────────────────
    if picks_file.exists():
        try:
            df = pd.read_csv(picks_file, low_memory=False)
            df = df[df["actual_winner"].notna()].copy()
            if not df.empty:
                df = df.sort_values("game_date", ascending=False).head(30)
                df["correct"] = (
                    df["predicted_winner"].str.strip().str.upper() ==
                    df["actual_winner"].str.strip().str.upper()
                )

                st.markdown('<div style="margin-top:2rem;font-family:\'Barlow Condensed\',sans-serif;font-size:1rem;font-weight:700;color:#f0f2f5;margin-bottom:0.6rem;">Recent Picks (Last 30)</div>', unsafe_allow_html=True)

                # header
                hcols = st.columns([1, 1.2, 1.2, 1.2, 1.2, 0.8])
                for col, lbl in zip(hcols, ["Date", "Away", "Home", "Pick", "Result", "✓"]):
                    col.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;color:#6a7d94;text-transform:uppercase;letter-spacing:0.08em;padding-bottom:0.3rem;border-bottom:1px solid rgba(255,255,255,0.07);">{lbl}</div>', unsafe_allow_html=True)

                for _, row in df.iterrows():
                    correct = bool(row.get("correct", False))
                    check_color = "#1f9d55" if correct else "#c0392b"
                    check_sym = "✓" if correct else "✗"

                    rcols = st.columns([1, 1.2, 1.2, 1.2, 1.2, 0.8])
                    vals = [
                        str(row.get("game_date", ""))[:10],
                        str(row.get("away_team", "—")),
                        str(row.get("home_team", "—")),
                        str(row.get("predicted_winner", "—")),
                        str(row.get("actual_winner", "—")),
                    ]
                    for col, val in zip(rcols[:5], vals):
                        col.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#a0b0c4;padding:0.35rem 0;">{val}</div>', unsafe_allow_html=True)
                    rcols[5].markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:{check_color};padding:0.35rem 0;font-weight:600;">{check_sym}</div>', unsafe_allow_html=True)
                    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.04);"></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load pick log: {e}")