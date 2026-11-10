"""Four paired log-odds slope charts for the human-only Phase I results.

Drop this file into dashboard/human_confidence_paths.py and call
    from human_confidence_paths import render_human_confidence_paths
    render_human_confidence_paths()
from your dashboard/pages/Human-Only page (see INSTALL.txt).

Input: human-only/results_with_neutral.csv
No inference, no regenerated scores, and no additional ML packages required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_FILE = PROJECT_ROOT / "human-only" / "results_with_neutral.csv"
CONDITIONS = ("misleading", "accurate")
VERSIONS = ("correct", "buggy")
REQUIRED_COLS = {"problem_id", "code_version", "comment_type", "p_yes"}
EPS = 1e-7


@st.cache_data(show_spinner=False)
def _load_results(path: str, modified_ns: int) -> pd.DataFrame:
    """Cache read, invalidating if file modification time changes."""
    del modified_ns
    return pd.read_csv(path)


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results_with_neutral.csv: {', '.join(sorted(missing))}")

    df = df.copy()
    df["code_version"] = df["code_version"].astype(str).str.lower().str.strip()
    df["comment_type"] = df["comment_type"].astype(str).str.lower().str.strip()
    df = df.loc[
        df["code_version"].isin(VERSIONS)
        & df["comment_type"].isin(("none",) + CONDITIONS)
    ].copy()
    if df.empty:
        raise ValueError("No correct/buggy rows with none/accurate/misleading conditions were found.")

    keys = ["problem_id", "code_version", "comment_type"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate problem_id + code_version + comment_type rows; cannot pair reliably.")
    df["p_yes"] = pd.to_numeric(df["p_yes"], errors="coerce")
    if df["p_yes"].isna().any() or (~df["p_yes"].between(0, 1)).any():
        raise ValueError("p_yes contains missing or out-of-range values.")

    # This is mathematically equivalent to log(P(Yes) / P(No)) for scores
    # normalized over the binary Yes/No candidates.
    p = df["p_yes"].clip(EPS, 1 - EPS)
    df["log_odds"] = np.log(p / (1 - p))
    return df


def _get_pairs(df: pd.DataFrame, version: str, condition: str) -> pd.DataFrame:
    sub = df.loc[
        (df["code_version"] == version)
        & (df["comment_type"].isin(["none", condition]))
    ]
    paired = sub.pivot(index="problem_id", columns="comment_type", values="log_odds")
    paired = paired.dropna(subset=["none", condition]).copy()
    paired["change"] = paired[condition] - paired["none"]
    return paired.reset_index()


def _draw_panel(paired: pd.DataFrame, condition: str, version: str,
                y_range: list[float]) -> go.Figure:
    fig = go.Figure()
    color = "rgba(183, 113, 94, 0.22)" if condition == "misleading" else "rgba(66, 138, 155, 0.20)"

    # Every thin path is one paired original implementation.
    for row in paired.itertuples(index=False):
        baseline = float(row.none)
        after = float(getattr(row, condition))
        shift = float(row.change)
        problem = str(row.problem_id)
        hover = (
            f"Problem: {problem}<br>Version: {version}<br>Condition: {condition}<br>"
            f"No comment: {baseline:.4f}<br>{condition.title()}: {after:.4f}<br>"
            f"Change: {shift:+.4f}<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, after], mode="lines+markers",
            line=dict(color=color, width=1), marker=dict(color=color, size=3),
            showlegend=False, hovertemplate=hover, hoverlabel=dict(namelength=-1),
        ))

    if not paired.empty:
        med_none = float(paired["none"].median())
        med_condition = float(paired[condition].median())
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[med_none, med_condition], mode="lines+markers",
            name="Median", line=dict(color="#181818", width=3.6),
            marker=dict(color="#181818", size=9),
            hovertemplate="Median log-odds: %{y:.4f}<extra>Median</extra>",
        ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"{version.title()} implementations (n={len(paired)})", font=dict(size=14)),
        height=365,
        margin=dict(l=38, r=14, t=48, b=42),
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["None", condition.title()],
                   range=[-0.10, 1.10], showgrid=False, zeroline=False),
        yaxis=dict(title="Log-odds: log(P(Yes)/P(No))", range=y_range, zeroline=True,
                   zerolinecolor="#bbbbbb"),
        legend=dict(orientation="h", x=1, y=1.12, xanchor="right", yanchor="bottom"),
        hovermode="closest",
    )
    return fig


def render_human_confidence_paths() -> None:
    """Render 4 charts: misleading/accurate × correct/buggy."""
    st.subheader("Paired confidence paths — Human-only pair")
    st.caption(
        "Reproduces the four slope plots in the human-only notebook (pages 16–19). "
        "Each faint line connects the same implementation with and without a comment; "
        "the dark line connects the group medians. The x-axis shows experimental "
        "conditions, not elapsed time."
    )

    if not RESULTS_FILE.is_file():
        st.error(
            "Missing human-only results: `human-only/results_with_neutral.csv`. "
            "Add the file to your project and reload the page."
        )
        return

    try:
        df = _prepare_data(_load_results(str(RESULTS_FILE), RESULTS_FILE.stat().st_mtime_ns))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to prepare human-only confidence paths: {exc}")
        return

    categories = sorted(df["bug_type"].dropna().astype(str).unique()) if "bug_type" in df else []
    selected_types = st.multiselect(
        "Filter mutation types (optional)", categories, default=[],
        key="human_paths_bug_type_filter",
        help="Leave empty to show all 178 problems.",
    ) if categories else []
    if selected_types:
        df = df.loc[df["bug_type"].astype(str).isin(selected_types)].copy()

    # Common Y scale across the four panels, to make differences comparable.
    bounds = df["log_odds"].agg(["min", "max"])
    if df.empty or bounds.isna().any():
        st.warning("No observations match the selected filter.")
        return
    pad = max(0.18, (float(bounds["max"]) - float(bounds["min"])) * 0.08)
    common_y = [float(bounds["min"]) - pad, float(bounds["max"]) + pad]

    for condition in CONDITIONS:
        st.markdown(f"#### Effect of {condition.title()} Comments on Model Judgments")
        left, right = st.columns(2)
        for version, column in zip(VERSIONS, (left, right)):
            paired = _get_pairs(df, version, condition)
            with column:
                st.plotly_chart(
                    _draw_panel(paired, condition, version, common_y),
                    width='stretch',
                    key=f"human_paths_{condition}_{version}",
                )
                if paired.empty:
                    st.info("No paired observations for this selection.")
                else:
                    st.caption(
                        f"n={len(paired)} | Median paired change in log-odds: "
                        f"**{paired['change'].median():+.4f}**"
                    )

    st.caption(
        "The median *paired change* displayed beneath each panel is calculated from "
        "within-problem differences; it need not equal the difference between the "
        "two displayed median endpoints. Data: human-only `results_with_neutral.csv`."
    )
