import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Interactive Analysis",
    layout="wide"
)

BASE = Path(__file__).resolve().parents[1]

VERSION_ORDER = ["correct", "buggy"]
VERSION_COLORS = {"correct": "#1f77b4", "buggy": "#d62728"}
COMMENT_ORDER = ["none", "accurate", "misleading"]
COMMENT_LABELS = {
    "none": "No comment",
    "accurate": "Accurate comment",
    "misleading": "Misleading comment"
}
NEUTRAL_LINK = "rgba(150, 150, 150, 0.5)"
ACCEPT_LINK = "rgba(214, 39, 40, 0.7)"
SANKEY_NODE_LABELS = [
    "No comment: Yes",
    "No comment: No",
    "Misleading: Yes",
    "Misleading: No"
]


@st.cache_data
def load_data():
    results = pd.read_csv(BASE / "controlled_results.csv")
    shifts = pd.read_csv(BASE / "confidence_shifts.csv")
    inputs = pd.read_csv(BASE / "controlled_inputs.csv")
    return results, shifts, inputs


@st.cache_data
def load_projection():
    path = BASE / "code_projection.csv"
    if not path.exists():
        return None
    proj = pd.read_csv(path)
    required = {"problem_id", "umap_x", "umap_y"}
    if not required.issubset(proj.columns):
        return None
    proj["problem_id"] = proj["problem_id"].astype(int)
    return proj


results, shifts, inputs = load_data()
projection = load_projection()

st.title("Interactive Analysis")
st.caption(
    "178 MBPP problems | Qwen2.5-Coder-1.5B-Instruct | "
    "Correct versus benchmark-detected buggy code"
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")

bug_types = sorted(shifts["bug_type"].unique())

selected_types = st.sidebar.multiselect(
    "Mutation types",
    bug_types,
    default=bug_types
)

versions = st.sidebar.multiselect(
    "Code versions",
    VERSION_ORDER,
    default=VERSION_ORDER
)

if not selected_types or not versions:
    st.warning("Select at least one mutation type and one code version.")
    st.stop()

filtered_shifts = shifts[
    shifts["bug_type"].isin(selected_types)
    & shifts["code_version"].isin(versions)
].copy()

filtered_results = results[
    results["bug_type"].isin(selected_types)
    & results["code_version"].isin(versions)
].copy()

if filtered_shifts.empty:
    st.info("No problems match the current filters.")
    st.stop()

panel_versions = [v for v in VERSION_ORDER if v in versions]
filtered_ids = sorted(int(x) for x in filtered_shifts["problem_id"].unique())

# Summary metrics are written into this slot once the selection is known.
summary_slot = st.container()

# ------------------------------------------------- view 1: projection
st.subheader("1. Projection Space Explorer")
st.caption(
    "Each point is one problem's correct code, embedded with CodeBERT "
    "(microsoft/codebert-base) and projected with UMAP. Use the box or lasso "
    "tool in the chart toolbar to select problems; views 2 and 3 follow the "
    "selection."
)

if st.button("Clear selection"):
    st.session_state["map_reset"] = st.session_state.get("map_reset", 0) + 1
map_key = f"projection_{st.session_state.get('map_reset', 0)}"

picked_ids = set()
if projection is None:
    st.warning(
        "Projection unavailable: dashboard/code_projection.csv is missing or lacks "
        "problem_id, umap_x, umap_y. Run scripts/build_code_projection.py (see README). "
        "No placeholder coordinates are drawn."
    )
else:
    color_by = st.radio(
        "Color points by",
        ["Misleading-comment confidence shift", "Mutation type"],
        horizontal=True
    )
    correct_shifts = shifts[
        (shifts["code_version"] == "correct") & shifts["bug_type"].isin(selected_types)
    ][["problem_id", "bug_type", "none", "misleading", "misleading_shift"]]
    proj_data = projection.merge(correct_shifts, on="problem_id", how="inner")

    hover = {
        "problem_id": True,
        "bug_type": True,
        "misleading_shift": ":+.3f",
        "umap_x": False,
        "umap_y": False
    }
    axis_labels = {"umap_x": "UMAP 1", "umap_y": "UMAP 2"}
    if color_by == "Mutation type":
        proj_fig = px.scatter(
            proj_data,
            x="umap_x",
            y="umap_y",
            color="bug_type",
            custom_data=["problem_id"],
            hover_data=hover,
            labels=axis_labels
        )
    else:
        limit = float(proj_data["misleading_shift"].abs().max()) or 1.0
        proj_fig = px.scatter(
            proj_data,
            x="umap_x",
            y="umap_y",
            color="misleading_shift",
            color_continuous_scale="RdBu",
            range_color=[-limit, limit],
            custom_data=["problem_id"],
            hover_data=hover,
            labels={**axis_labels, "misleading_shift": "Shift in P(Yes)"}
        )
    proj_fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    proj_fig.update_layout(height=520, margin=dict(t=30))
    proj_fig.update_xaxes(showticklabels=False)
    proj_fig.update_yaxes(showticklabels=False)

    proj_event = st.plotly_chart(
        proj_fig,
        width="stretch",
        key=map_key,
        on_select="rerun",
        selection_mode=["points", "box", "lasso"]
    )
    picked_ids = {
        int(point["customdata"][0])
        for point in proj_event.selection.points
        if point.get("customdata")
    }

st.markdown("**Confidence shift per problem (reference)**")
st.caption("P(Yes) without a comment versus the change with a misleading comment.")

shift_fig = px.scatter(
    filtered_shifts,
    x="none",
    y="misleading_shift",
    color="code_version",
    color_discrete_map=VERSION_COLORS,
    category_orders={"code_version": VERSION_ORDER},
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
        "misleading_shift": "Change in P(Yes) with misleading comment",
        "code_version": "Code version"
    }
)
shift_fig.add_hline(y=0, line_dash="dash", line_color="gray")
shift_fig.update_layout(
    height=460,
    margin=dict(t=60),
    legend=dict(
        title_text="",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11)
    )
)

st.plotly_chart(shift_fig, width="stretch", key="shift_map_reference")

selected_ids = sorted(picked_ids.intersection(filtered_ids))

# ------------------------------------------------- scope for all views
scope_ids = selected_ids if selected_ids else filtered_ids
scope_shifts = filtered_shifts[filtered_shifts["problem_id"].isin(scope_ids)].copy()
scope_results = filtered_results[filtered_results["problem_id"].isin(scope_ids)].copy()
scope_label = (
    f"{len(selected_ids)} selected problems" if selected_ids else "all filtered problems"
)

mean_shift = scope_shifts["misleading_shift"].mean()

with summary_slot:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Problems in filter", len(filtered_ids))
    m2.metric("Problems in scope", scope_shifts["problem_id"].nunique())
    m3.metric("Evaluations in scope", len(scope_results))
    m4.metric(
        "Mean misleading shift",
        f"{mean_shift:+.2%}" if pd.notna(mean_shift) else "N/A"
    )
    st.caption(
        f"Scope: {scope_label}. Views 2 and 3 and the supporting summaries "
        "update to this scope."
    )

# ------------------------------------------------- view 2: confidence paths
st.subheader("2. Confidence Path Explorer")
st.caption(
    "Each thin line connects one problem's P(Yes) across the three experimental "
    "conditions. The horizontal axis lists conditions, not time steps. Bold lines "
    "are means; hover over them for n."
)

path_data = scope_results[scope_results["comment_type"].isin(COMMENT_ORDER)]

if path_data.empty:
    st.info("No evaluations in scope.")
else:
    x_labels = [COMMENT_LABELS[c] for c in COMMENT_ORDER]
    path_titles = []
    for version in panel_versions:
        n_problems = path_data.loc[
            path_data["code_version"] == version, "problem_id"
        ].nunique()
        path_titles.append(f"{version.capitalize()} code (n = {n_problems} problems)")

    path_fig = make_subplots(
        rows=1,
        cols=len(panel_versions),
        shared_yaxes=True,
        subplot_titles=path_titles
    )

    for col, version in enumerate(panel_versions, start=1):
        version_data = path_data[path_data["code_version"] == version]
        color = VERSION_COLORS[version]

        for pid, group in version_data.groupby("problem_id"):
            values = group.set_index("comment_type")["p_yes"].reindex(COMMENT_ORDER)
            path_fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=values.values,
                    mode="lines+markers",
                    line=dict(color=color, width=1),
                    marker=dict(size=5, color=color),
                    opacity=0.3,
                    showlegend=False,
                    name=f"Problem {pid}",
                    hovertemplate=(
                        f"Problem {pid} ({version})<br>"
                        "%{x}<br>P(Yes): %{y:.3f}<extra></extra>"
                    )
                ),
                row=1,
                col=col
            )

        mean_values = (
            version_data.groupby("comment_type")["p_yes"].mean().reindex(COMMENT_ORDER)
        )
        counts = (
            version_data.groupby("comment_type")["p_yes"].count().reindex(COMMENT_ORDER)
        )
        path_fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=mean_values.values,
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=10, color=color),
                customdata=counts.values,
                name=f"{version} mean",
                showlegend=False,
                hovertemplate=(
                    "Mean P(Yes): %{y:.3f}<br>"
                    "n = %{customdata}<extra>" + version + " mean</extra>"
                )
            ),
            row=1,
            col=col
        )

    path_fig.update_yaxes(range=[0, 1], title_text="P(Yes)")
    path_fig.update_layout(height=480)
    st.plotly_chart(path_fig, width="stretch", key="path_chart")

# ------------------------------------------------- view 3: transitions
st.subheader("3. Prediction Transition Explorer")
st.caption(
    "Yes/No prediction without a comment (left) and with the misleading comment "
    "(right). For buggy code, a Yes prediction is an incorrect acceptance: the model "
    "accepts code that fails the benchmark tests."
)

decision = scope_results[scope_results["comment_type"].isin(["none", "misleading"])]
before = decision[decision["comment_type"] == "none"][
    ["problem_id", "code_version", "prediction"]
]
after = decision[decision["comment_type"] == "misleading"][
    ["problem_id", "code_version", "prediction"]
]
pairs = before.merge(
    after,
    on=["problem_id", "code_version"],
    suffixes=("_none", "_misleading")
)

transition_cols = st.columns(len(panel_versions)) if panel_versions else []

for col, version in zip(transition_cols, panel_versions):
    subset = pairs[pairs["code_version"] == version]
    n = len(subset)

    with col:
        st.markdown(
            f"**{version.capitalize()} code** "
            f"(n = {n} problems with both conditions)"
        )

        if n == 0:
            st.info("No problem has both conditions in scope.")
            continue

        counts = subset.groupby(["prediction_none", "prediction_misleading"]).size()
        left_totals = subset["prediction_none"].value_counts()
        right_totals = subset["prediction_misleading"].value_counts()

        node_totals = [
            left_totals.get("Yes", 0),
            left_totals.get("No", 0),
            right_totals.get("Yes", 0),
            right_totals.get("No", 0)
        ]
        node_labels = [
            f"{label} (n={total})"
            for label, total in zip(SANKEY_NODE_LABELS, node_totals)
        ]

        sources, targets, values, link_colors, link_text = [], [], [], [], []
        for (p_before, p_after), k in counts.items():
            accepted = version == "buggy" and p_after == "Yes"
            sources.append(0 if p_before == "Yes" else 1)
            targets.append(2 if p_after == "Yes" else 3)
            values.append(int(k))
            link_colors.append(ACCEPT_LINK if accepted else NEUTRAL_LINK)
            label = f"{p_before} to {p_after}: {k} problems"
            if accepted:
                label += " (incorrect acceptance)"
            link_text.append(label)

        sankey = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=node_labels,
                    pad=20,
                    thickness=18,
                    color=VERSION_COLORS[version]
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=link_text,
                    hovertemplate="%{customdata}<extra></extra>"
                )
            )
        )
        sankey.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(sankey, width="stretch", key=f"sankey_{version}")

        if version == "buggy":
            accepted_none = int((subset["prediction_none"] == "Yes").sum())
            accepted_misleading = int((subset["prediction_misleading"] == "Yes").sum())
            a1, a2 = st.columns(2)
            a1.metric("Incorrect acceptances, no comment", f"{accepted_none} / {n}")
            a2.metric(
                "Incorrect acceptances, misleading",
                f"{accepted_misleading} / {n}",
                delta=accepted_misleading - accepted_none,
                delta_color="inverse"
            )

st.markdown("**Prediction flips**")
st.caption(
    "Problems where adding a misleading comment changes Qwen's Yes/No prediction "
    "(threshold p_yes >= 0.5)."
)

flip_data = scope_shifts.copy()

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

# ------------------------------------------------- supporting summaries
with st.expander(f"Supporting summaries (scope: {scope_label})"):
    st.markdown("**Confidence Shift Distribution**")

    if scope_shifts.empty:
        st.info("No problems in scope.")
    else:
        fig = px.box(
            scope_shifts,
            x="bug_type",
            y="misleading_shift",
            color="code_version",
            color_discrete_map=VERSION_COLORS,
            category_orders={"code_version": VERSION_ORDER},
            points="all",
            hover_data=["problem_id"],
            title="Effect of misleading comments by bug type"
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=550)
        st.plotly_chart(fig, width="stretch", key="box_chart")

    st.markdown("**Bug-Type Analysis**")
    st.caption(
        "Heatmaps show mean normalized P(Yes) and "
        "prediction accuracy across mutation types. "
        "Hover to inspect sample counts."
    )

    heatmap_results = scope_results.copy()

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
            for version in VERSION_ORDER
            for comment in COMMENT_ORDER
            if version in versions
        ]

        def draw_heatmap(value_column, title, scale, key):
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

            heat = go.Figure(go.Heatmap(
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

            heat.update_layout(
                title=title,
                height=max(480, len(data) * 32),
                xaxis_title="Code version / comment condition",
                yaxis_title="Mutation type"
            )

            st.plotly_chart(heat, width="stretch", key=key)

        draw_heatmap(
            "mean_p_yes",
            "Mean P(Yes) by mutation type",
            "Blues",
            "heat_p_yes"
        )

        draw_heatmap(
            "accuracy",
            "Prediction accuracy by mutation type",
            "Viridis",
            "heat_accuracy"
        )

    else:
        st.info("No evaluations match the current filters.")
