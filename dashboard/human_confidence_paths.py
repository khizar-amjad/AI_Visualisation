"""Paired confidence-path charts for the human-only analysis page.

Each chart is a slope plot: one thin line per implementation, connecting its
log-odds of a Yes judgment without a comment to its log-odds with a comment,
plus a bold line through the group medians.

The module draws from the DataFrame prepared by the page (columns listed in
``REQUIRED_COLUMNS``). It reads no files and runs no model inference.
``HUMAN_DIR`` is the single definition of the human-only data folder
(``dashboard/human-only/``); the Human-Only page imports it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HUMAN_DIR = Path(__file__).resolve().parent / "human-only"
CONDITIONS = ("misleading", "accurate")
VERSIONS = ("correct", "buggy")
CONDITION_COLORS = {"accurate": "#2ca02c", "misleading": "#ff7f0e"}
MEDIAN_COLOR = "#181818"
REQUIRED_COLUMNS = {"problem_id", "code_version", "comment_type", "logodds"}


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    keys = ["problem_id", "code_version", "comment_type"]
    if df.duplicated(keys).any():
        raise ValueError("duplicate problem, version and condition rows; cannot pair reliably")
    if not np.isfinite(df["logodds"]).all():
        raise ValueError("log-odds contain missing or infinite values")
    return df


def _paired_changes(df: pd.DataFrame, version: str, condition: str) -> pd.DataFrame:
    """One row per problem with log-odds without and with the comment."""
    subset = df[(df["code_version"] == version) & df["comment_type"].isin(["none", condition])]
    wide = subset.pivot(index="problem_id", columns="comment_type", values="logodds")
    wide = wide.dropna(subset=["none", condition])
    wide = wide.rename(columns={"none": "before", condition: "after"})[["before", "after"]]
    wide["change"] = wide["after"] - wide["before"]
    return wide.reset_index()


def _slope_chart(
    paired: pd.DataFrame, condition: str, version: str, y_range: list[float]
) -> go.Figure:
    color = CONDITION_COLORS[condition]
    label = condition.capitalize()
    n = len(paired)

    # One trace holds every path; NaN breaks the line between problems.
    x = np.tile([0, 1, np.nan], n)
    y = np.column_stack([paired["before"], paired["after"], np.full(n, np.nan)]).ravel()
    custom = np.repeat(
        np.column_stack([paired["problem_id"], paired["change"]]), 3, axis=0
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            customdata=custom,
            mode="lines+markers",
            line=dict(color=color, width=1),
            marker=dict(color=color, size=4),
            opacity=0.35,
            showlegend=False,
            hovertemplate=(
                "Problem %{customdata[0]:.0f}<br>"
                "Log-odds: %{y:.3f}<br>"
                "Change: %{customdata[1]:+.3f}<extra></extra>"
            ),
        )
    )
    if n:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[paired["before"].median(), paired["after"].median()],
                mode="lines+markers",
                line=dict(color=MEDIAN_COLOR, width=4),
                marker=dict(color=MEDIAN_COLOR, size=10),
                showlegend=False,
                hovertemplate="Group median log-odds: %{y:.3f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#aaaaaa")
    fig.update_layout(
        title=dict(text=f"{version.capitalize()} code (n = {n} problems)", font=dict(size=14)),
        height=380,
        margin=dict(l=50, r=15, t=50, b=40),
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["No comment", f"{label} comment"],
            range=[-0.15, 1.15],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(title="Log-odds of Yes", range=y_range),
        hovermode="closest",
    )
    return fig


def render_human_confidence_paths(df: pd.DataFrame, section: str = "Paired confidence paths") -> None:
    """Render the four paired charts (misleading and accurate, correct and buggy)."""
    st.header(section)
    st.caption(
        "Each thin line connects one implementation's log-odds of a Yes judgment "
        "without a comment to its value with a comment. The bold line connects the "
        "group medians. The horizontal axis lists conditions, not time. All four "
        "panels share one vertical axis."
    )

    try:
        df = _validate(df)
    except ValueError as exc:
        st.error(f"Confidence paths unavailable: {exc}.")
        return

    # The y-range is fixed on the full data so panels stay comparable when filtering.
    low, high = float(df["logodds"].min()), float(df["logodds"].max())
    pad = max(0.15, (high - low) * 0.06)
    y_range = [low - pad, high + pad]

    if "bug_type" in df.columns:
        types = sorted(df["bug_type"].dropna().astype(str).unique())
        chosen = st.multiselect(
            "Mutation types",
            types,
            default=[],
            key="human_paths_bug_types",
            help="Leave empty to include every mutation type.",
        )
        if chosen:
            df = df[df["bug_type"].astype(str).isin(chosen)]
    if df.empty:
        st.info("No observations match the selected mutation types.")
        return

    for condition in CONDITIONS:
        st.markdown(f"**{condition.capitalize()} comments versus no comment**")
        columns = st.columns(len(VERSIONS))
        for version, column in zip(VERSIONS, columns):
            paired = _paired_changes(df, version, condition)
            with column:
                if paired.empty:
                    st.info(f"No paired {version} observations for this selection.")
                    continue
                st.plotly_chart(
                    _slope_chart(paired, condition, version, y_range),
                    width="stretch",
                    key=f"human_paths_{condition}_{version}",
                )
                st.caption(
                    f"Median paired change: {paired['change'].median():+.3f} log-odds "
                    f"(n = {len(paired)})"
                )

    st.caption(
        "The median paired change is the median of per-problem differences. It need "
        "not equal the difference between the two bold-line endpoints, which are "
        "medians of the separate groups."
    )
