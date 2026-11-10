
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Interactive Analysis",
    layout="wide"
)

BASE = Path(__file__).resolve().parents[1]

@st.cache_data
def load_data():
    results = pd.read_csv(BASE / "controlled_results.csv")
    shifts = pd.read_csv(BASE / "confidence_shifts.csv")
    inputs = pd.read_csv(BASE / "controlled_inputs.csv")
    return results, shifts, inputs

results, shifts, inputs = load_data()

st.title("Interactive Analysis")
st.caption(
    "178 MBPP problems | Qwen2.5-Coder-1.5B-Instruct | "
    "Correct versus benchmark-detected buggy code"
)

st.sidebar.header("Experiment Filters")

bug_types = sorted(shifts["bug_type"].unique())

selected_types = st.sidebar.multiselect(
    "Mutation types",
    bug_types,
    default=bug_types
)

versions = st.sidebar.multiselect(
    "Code versions",
    ["correct", "buggy"],
    default=["correct", "buggy"]
)

filtered_shifts = shifts[
    shifts["bug_type"].isin(selected_types)
    & shifts["code_version"].isin(versions)
].copy()

filtered_results = results[
    results["bug_type"].isin(selected_types)
    & results["code_version"].isin(versions)
].copy()

a, b, c = st.columns(3)

a.metric(
    "Unique problems",
    filtered_shifts["problem_id"].nunique()
)

b.metric(
    "Evaluations",
    len(filtered_results)
)

c.metric(
    "Mean misleading shift",
    f"{filtered_shifts['misleading_shift'].mean():+.2%}"
)

st.subheader("Confidence Shift Distribution")

fig = px.box(
    filtered_shifts,
    x="bug_type",
    y="misleading_shift",
    color="code_version",
    points="all",
    hover_data=["problem_id"],
    title="Effect of misleading comments by bug type"
)

fig.add_hline(y=0, line_dash="dash")
fig.update_layout(height=550)

st.plotly_chart(
    fig,
    width="stretch"
)

st.subheader("Confidence Shift Explorer")

fig2 = px.scatter(
    filtered_shifts,
    x="none",
    y="misleading_shift",
    color="code_version",
    custom_data=["problem_id"],
    hover_data={
        "problem_id": True,
        "bug_type": True,
        "none": ":.3f",
        "misleading": ":.3f",
        "misleading_shift": ":+.3f",
        "code_version": False
    },
    labels={
        "none": "P(Yes) without comment",
        "misleading_shift": "Change in P(Yes)",
        "code_version": "Code version"
    }
)

fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(
    height=500,
    legend=dict(
        title_text="",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)


st.plotly_chart(
    fig2,
    width="stretch"
)


st.subheader("Bug-Type Analysis")

st.caption(
    "Heatmaps show mean normalized P(Yes) and "
    "prediction accuracy across mutation types. "
    "Hover to inspect sample counts."
)

heatmap_results = filtered_results.copy()

if not heatmap_results.empty:

    heatmap_results["is_correct"] = (
        heatmap_results["prediction"].eq("Yes")
        == heatmap_results["code_version"].eq("correct")
    )

    summary = (
        heatmap_results.groupby(
            ["bug_type", "code_version", "comment_type"]
        )
        .agg(
            mean_p_yes=("p_yes", "mean"),
            accuracy=("is_correct", "mean"),
            evaluations=("problem_id", "count"),
            problems=("problem_id", "nunique")
        )
        .reset_index()
    )

    summary["condition"] = (
        summary["code_version"]
        + " / "
        + summary["comment_type"]
    )

    condition_order = [
        f"{version} / {comment}"
        for version in ["correct", "buggy"]
        for comment in ["none", "accurate", "misleading"]
        if version in versions
    ]

    import plotly.graph_objects as go

    def draw_heatmap(value_column, title, scale):
        data = summary.pivot(
            index="bug_type",
            columns="condition",
            values=value_column
        )

        data = data.reindex(
            columns=[
                c for c in condition_order
                if c in data.columns
            ]
        )

        counts = summary.pivot(
            index="bug_type",
            columns="condition",
            values="problems"
        ).reindex(
            index=data.index,
            columns=data.columns
        )

        fig = go.Figure(go.Heatmap(
            z=data.values,
            x=data.columns.tolist(),
            y=data.index.tolist(),
            customdata=counts.values,
            colorscale=scale,
            zmin=0,
            zmax=1,
            text=[
                [f"{v:.2f}" for v in row]
                for row in data.values
            ],
            texttemplate="%{text}",
            hovertemplate=(
                "Mutation: %{y}<br>"
                "Condition: %{x}<br>"
                "Value: %{z:.3f}<br>"
                "Problems: %{customdata}"
                "<extra></extra>"
            ),
            colorbar=dict(title="Score")
        ))

        fig.update_layout(
            title=title,
            height=max(480, len(data) * 32),
            xaxis_title="Code version / comment condition",
            yaxis_title="Mutation type"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    draw_heatmap(
        "mean_p_yes",
        "Mean P(Yes) by mutation type",
        "Blues"
    )

    draw_heatmap(
        "accuracy",
        "Prediction accuracy by mutation type",
        "Viridis"
    )

else:
    st.info("No evaluations match the current filters.")


st.subheader("Prediction Flip Analysis")

st.caption(
    "Problems where adding a misleading comment "
    "changes Qwen's Yes/No prediction."
)

flip_data = filtered_shifts.copy()

flip_data["prediction_before"] = (
    flip_data["none"] >= 0.5
).map({True: "Yes", False: "No"})

flip_data["prediction_after"] = (
    flip_data["misleading"] >= 0.5
).map({True: "Yes", False: "No"})

flipped = flip_data[
    flip_data["prediction_before"]
    != flip_data["prediction_after"]
].copy()

col1, col2 = st.columns(2)

col1.metric("Prediction flips", len(flipped))
col2.metric(
    "Flip rate",
    f"{len(flipped) / len(flip_data):.1%}"
    if len(flip_data) else "N/A"
)

if not flipped.empty:
    st.dataframe(
        flipped[
            [
                "problem_id",
                "code_version",
                "bug_type",
                "none",
                "misleading",
                "misleading_shift",
                "prediction_before",
                "prediction_after"
            ]
        ].sort_values(
            "misleading_shift",
            key=abs,
            ascending=False
        ).round(4),
        width="stretch",
        hide_index=True
    )
else:
    st.info("No prediction flips under current filters.")
