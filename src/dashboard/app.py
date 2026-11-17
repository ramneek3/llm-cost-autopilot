"""Streamlit dashboard for LLM Cost Autopilot.

Shows: net savings, routing distribution, quality scores,
escalation rate, Pareto chart, and cost over time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent.parent / "autopilot.db"


def get_conn():
    return sqlite3.connect(str(DB_PATH))


def load_requests() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM requests ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_quality() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT qs.*, r.model_name, r.tier, r.cost
           FROM quality_scores qs
           JOIN requests r ON qs.request_id = r.request_id""",
        conn,
    )
    conn.close()
    return df


def main():
    st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
    st.title("LLM Cost Autopilot Dashboard")

    df = load_requests()
    if df.empty:
        st.warning("No request data yet. Send some requests to the API first.")
        return

    quality_df = load_quality()

    # ── Headline metrics ──
    total_cost = df["cost"].sum()

    # Estimate baseline cost (all requests through most expensive model)
    # Using claude-sonnet-5-5 output rate as proxy
    baseline_rate = 0.000010  # $10/1M tokens
    total_tokens = (df["input_tokens"] + df["output_tokens"]).sum()
    baseline_cost = total_tokens * baseline_rate

    net_savings = baseline_cost - total_cost
    savings_pct = (net_savings / baseline_cost * 100) if baseline_cost > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", f"{len(df):,}")
    col2.metric("Total Cost", f"${total_cost:.4f}")
    col3.metric("Net Savings", f"${net_savings:.4f}", f"{savings_pct:.1f}%")
    col4.metric("Escalation Rate", f"{df['escalated'].mean() * 100:.1f}%")

    if not quality_df.empty:
        col_q1, col_q2 = st.columns(2)
        col_q1.metric("Avg Quality Score", f"{quality_df['rubric_score'].mean():.2f}/5")
        col_q2.metric("Quality Pass Rate", f"{quality_df['is_pass'].mean() * 100:.1f}%")

    st.divider()

    # ── Charts ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Routing Distribution")
        model_counts = df["model_name"].value_counts()
        fig_pie = px.pie(
            names=model_counts.index,
            values=model_counts.values,
            title="Requests by Model",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Cost by Model")
        model_costs = df.groupby("model_name")["cost"].sum().sort_values(ascending=True)
        fig_bar = px.bar(
            x=model_costs.values,
            y=model_costs.index,
            orientation="h",
            title="Total Cost by Model",
            labels={"x": "Cost ($)", "y": "Model"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Cost over time ──
    st.subheader("Cost Over Time")
    df_time = df.set_index("datetime").resample("1h")["cost"].sum().reset_index()
    if len(df_time) > 1:
        fig_time = px.area(df_time, x="datetime", y="cost", title="Hourly Cost")
        st.plotly_chart(fig_time, use_container_width=True)

    # ── Latency distribution ──
    st.subheader("Latency by Model")
    fig_lat = px.box(df, x="model_name", y="latency_ms", title="Latency Distribution (ms)")
    st.plotly_chart(fig_lat, use_container_width=True)

    # ── Quality over time ──
    if not quality_df.empty:
        st.subheader("Quality Scores")
        fig_q = px.histogram(
            quality_df, x="rubric_score", nbins=10,
            title="Quality Score Distribution",
            labels={"rubric_score": "Score (1-5)"},
        )
        st.plotly_chart(fig_q, use_container_width=True)

    # ── Tier breakdown ──
    st.subheader("Tier Analysis")
    tier_stats = df.groupby("tier").agg(
        count=("request_id", "count"),
        avg_cost=("cost", "mean"),
        avg_latency=("latency_ms", "mean"),
        total_cost=("cost", "sum"),
    ).round(6)
    st.dataframe(tier_stats)

    # ── Recent requests ──
    st.subheader("Recent Requests")
    st.dataframe(
        df[["datetime", "model_name", "tier", "input_tokens", "output_tokens",
            "cost", "latency_ms", "escalated", "routed_by"]]
        .head(50),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
