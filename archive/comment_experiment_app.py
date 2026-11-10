
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"

st.set_page_config(
    page_title="Does AI Trust Code or Comments?",
    layout="wide"
)

st.title("Does AI Trust Code or Comments?")
st.caption(
    "Investigating Qwen2.5-Coder-1.5B-Instruct "
    "using the MBPP Python benchmark"
)

@st.cache_data
def load_data():
    results = pd.read_csv(DASHBOARD / "results.csv")
    annotations = pd.read_csv(DASHBOARD / "annotations.csv")
    return results, annotations

results, annotations = load_data()

# Sidebar filters
st.sidebar.header("Filters")

experiment = st.sidebar.selectbox(
    "Experiment",
    sorted(results["experiment"].dropna().unique())
)

filtered = results[
    results["experiment"] == experiment
].copy()

code_versions = st.sidebar.multiselect(
    "Code versions",
    sorted(filtered["code_version"].dropna().unique()),
    default=sorted(filtered["code_version"].dropna().unique())
)

filtered = filtered[
    filtered["code_version"].isin(code_versions)
]

st.subheader("Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Evaluations", len(filtered))
col2.metric("Unique problems", filtered["problem_id"].nunique())
col3.metric("Mean P(Yes)", f"{filtered['p_yes'].mean():.1%}")

st.subheader("Confidence Distribution")

fig = px.box(
    filtered,
    x="comment_type",
    y="p_yes",
    color="code_version",
    points="all",
    hover_data=["problem_id"],
    category_orders={
        "comment_type": ["none", "accurate", "misleading"]
    }
)

fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, width="stretch")


# Confidence shift analysis
st.subheader("Effect of Misleading Comments")

comment_data = results[
    results["experiment"] == "Comment experiment"
]

comparison = comment_data.pivot_table(
    index="problem_id",
    columns="comment_type",
    values="p_yes"
).reset_index()

comparison["shift"] = (
    comparison["misleading"] - comparison["none"]
)

import plotly.graph_objects as go

shift_fig = go.Figure()

for _, row in comparison.iterrows():
    shift_fig.add_trace(go.Scatter(
        x=["No comment", "Misleading"],
        y=[row["none"], row["misleading"]],
        mode="lines+markers",
        name=f"Problem {int(row['problem_id'])}",
        hovertemplate=(
            f"Problem {int(row['problem_id'])}<br>"
            "P(Yes): %{y:.3f}<extra></extra>"
        )
    ))

shift_fig.update_layout(
    yaxis_title="P(Yes)",
    yaxis_range=[0, 1],
    height=550,
    showlegend=True
)

st.plotly_chart(
    shift_fig,
    width="stretch"
)

st.caption(
    "Each line represents one problem. "
    "Downward slopes indicate reduced P(Yes) "
    "after adding a misleading comment."
)


st.subheader("Interactive Confidence-Shift Explorer")

st.caption(
    "Click a dot to inspect the corresponding Python function. "
    "Negative shifts mean misleading comments reduced P(Yes)."
)

shift_data = results[
    results["experiment"] == "Comment experiment"
].pivot_table(
    index="problem_id",
    columns="comment_type",
    values="p_yes"
).reset_index()

shift_data["shift"] = (
    shift_data["misleading"] - shift_data["none"]
)

shift_chart = px.scatter(
    shift_data,
    x="none",
    y="shift",
    custom_data=["problem_id"],
    hover_data=["problem_id", "misleading"],
    labels={
        "none": "P(Yes) without comment",
        "shift": "Change in P(Yes)"
    }
)

shift_chart.add_hline(
    y=0,
    line_dash="dash",
    line_color="gray"
)

shift_chart.update_traces(
    marker=dict(size=12)
)

shift_chart.update_layout(height=420)

event = st.plotly_chart(
    shift_chart,
    width="stretch",
    key="select_problem_chart",
    on_select="rerun",
    selection_mode="points"
)

clicked_problem_id = None

if event.selection.points:
    point = event.selection.points[0]
    clicked_problem_id = int(point["customdata"][0])

if clicked_problem_id is not None:
    st.success(f"Selected Problem {clicked_problem_id}")

st.subheader("Problem Inspector")

problem_ids = sorted(filtered["problem_id"].dropna().unique())

if problem_ids:
    selected_id = st.selectbox(
        "Select problem",
        problem_ids,
        index=(
            problem_ids.index(clicked_problem_id)
            if clicked_problem_id in problem_ids
            else 0
        )
    )

    problem_results = filtered[
        filtered["problem_id"] == selected_id
    ]

    annotation = annotations[
        annotations["problem_id"] == selected_id
    ]

    if not annotation.empty:
        row = annotation.iloc[0]

        st.write("**Task specification:**")
        st.write(row["specification"])

        st.write("**Original Python code:**")
        st.code(row["original_code"], language="python")

        st.write("**Accurate comment:**")
        st.write(row["accurate_comment"])

        st.write("**Misleading comment:**")
        st.write(row["misleading_comment"])

    fig2 = px.bar(
        problem_results,
        x="comment_type",
        y="p_yes",
        color="code_version",
        barmode="group",
        category_orders={
            "comment_type": ["none", "accurate", "misleading"]
        }
    )

    fig2.update_yaxes(range=[0, 1])
    st.plotly_chart(fig2, width="stretch")

else:
    st.warning("No problems match the filters.")
