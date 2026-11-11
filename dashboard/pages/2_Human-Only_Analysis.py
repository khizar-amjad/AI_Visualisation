from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

pio.templates.default = "plotly_white"

st.set_page_config(page_title="Human-Only Analysis", layout="wide")

BASE = Path(__file__).resolve().parents[1]  # dashboard/
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from human_confidence_paths import HUMAN_DIR, render_human_confidence_paths  # noqa: E402

HUMAN = HUMAN_DIR  # dashboard/human-only/

VERSION_ORDER = ["correct", "buggy"]
VERSION_COLORS = {"correct": "#1f77b4", "buggy": "#d62728"}
CONDITIONS = ["none", "neutral", "accurate", "misleading"]
COMPARED = ["neutral", "accurate", "misleading"]
COND_LABELS = {
    "none": "No comment",
    "neutral": "Neutral comment",
    "accurate": "Accurate comment",
    "misleading": "Misleading comment",
}
COND_COLORS = {
    "none": "#555555",
    "neutral": "#9467bd",
    "accurate": "#2ca02c",
    "misleading": "#ff7f0e",
}
AUROC_CONDITIONS = ["none", "accurate", "misleading"]
NOTEBOOK_AUROC = {"none": 0.728286, "accurate": 0.760084, "misleading": 0.621939}
AUROC_TOLERANCE = 5e-7
N_BOOT = 2000
BOOT_SEED = 42
KEY = ["problem_id", "code_version", "comment_type"]


# ----------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def load_human(folder: str):
    folder = Path(folder)
    results = pd.read_csv(folder / "results_with_neutral.csv")
    inputs = pd.read_csv(folder / "inputs_with_neutral.csv")
    neutral_cmp = pd.read_csv(folder / "neutral_condition_comparison.csv")
    wilcoxon_csv = pd.read_csv(folder / "logodds_wilcoxon_results.csv")
    return results, inputs, neutral_cmp, wilcoxon_csv


@st.cache_data(show_spinner=False)
def load_original(path: str):
    return pd.read_csv(path)


required_files = [
    "results_with_neutral.csv",
    "inputs_with_neutral.csv",
    "neutral_condition_comparison.csv",
    "logodds_wilcoxon_results.csv",
]
missing_files = [name for name in required_files if not (HUMAN / name).is_file()]
if missing_files:
    st.error(
        "Missing human-only CSV files: " + ", ".join(missing_files)
        + ". Expected in dashboard/human-only/."
    )
    st.stop()

try:
    results, inputs, neutral_cmp, wilcoxon_csv = load_human(str(HUMAN))
except (OSError, pd.errors.ParserError) as exc:
    st.error(f"Could not load the human-only CSV files: {exc}")
    st.stop()

original_file = BASE / "controlled_results.csv"
original = load_original(str(original_file)) if original_file.is_file() else None


def clean_code(code):
    return str(code).replace("\r\n", "\n").replace("\r", "\n")


def comment_text(code):
    parts = [line.split("#", 1)[1].strip() for line in clean_code(code).split("\n") if "#" in line]
    return " ".join(parts)


# -------------------------------------------------------------- validation
input_meta = ["specification", "code", "ground_truth"]
missing_columns = [c for c in KEY + input_meta if c not in inputs.columns]
if missing_columns:
    st.error(f"inputs_with_neutral.csv is missing columns: {', '.join(missing_columns)}")
    st.stop()
if "annotation_status" in inputs.columns:
    input_meta.append("annotation_status")
obs = results.merge(
    inputs[KEY + input_meta],
    on=KEY,
    how="left",
    validate="one_to_one",
)
obs["p_norm"] = obs["p_yes"] / (obs["p_yes"] + obs["p_no"])
obs["logodds"] = np.log(obs["p_yes"] / obs["p_no"])

per_implementation = obs.groupby(["problem_id", "code_version"])["comment_type"].nunique()
checks = [
    ("1,424 observations", len(obs) == 1424 and len(inputs) == 1424, f"{len(obs):,} result rows, {len(inputs):,} input rows"),
    ("178 unique problems", obs["problem_id"].nunique() == 178, f"{obs['problem_id'].nunique()} problems"),
    ("Four comment conditions", set(obs["comment_type"]) == set(CONDITIONS), ", ".join(sorted(obs["comment_type"].unique()))),
    ("No duplicate observations", not obs.duplicated(KEY).any(), f"{int(obs.duplicated(KEY).sum())} duplicates"),
    ("No missing values", not obs[["p_yes", "p_no", "ground_truth", "code"]].isna().any().any(), f"{int(obs.isna().sum().sum())} missing cells"),
    ("Every implementation has all four conditions", bool((per_implementation == 4).all()), f"{len(per_implementation)} implementations"),
    ("Ground truth matches code version", bool(((obs["code_version"] == "correct") == obs["ground_truth"]).all()), "correct = True, buggy = False"),
    ("P(Yes) + P(No) is about 1", bool(np.isclose(obs["p_yes"] + obs["p_no"], 1, atol=1e-6).all()), f"max deviation {float((obs['p_yes'] + obs['p_no'] - 1).abs().max()):.1e}"),
]
all_checks_pass = all(ok for _, ok, _ in checks)

# ----------------------------------------------------------------- shifts
baseline = obs[obs["comment_type"] == "none"][
    ["problem_id", "code_version", "bug_type", "p_norm", "logodds", "p_yes"]
].rename(columns={"p_norm": "base_p", "logodds": "base_lo", "p_yes": "base_p_yes"})
conditioned = obs[obs["comment_type"] != "none"][
    ["problem_id", "code_version", "comment_type", "p_norm", "logodds"]
].rename(columns={"p_norm": "cond_p", "logodds": "cond_lo"})
shift_df = conditioned.merge(baseline, on=["problem_id", "code_version"], validate="many_to_one")
shift_df["shift_pp"] = (shift_df["cond_p"] - shift_df["base_p"]) * 100
shift_df["shift_lo"] = shift_df["cond_lo"] - shift_df["base_lo"]

# ------------------------------------------------------------------ header
st.title("Human-Only Analysis")
st.caption(
    "178 MBPP problems | Qwen2.5-Coder-1.5B-Instruct | "
    "Four comment conditions: none, neutral, accurate, misleading"
)
st.info(
    "This page analyses the human-only data pair in `dashboard/human-only/`. "
    "It is a separate run from the AI-assisted Interactive Analysis page and adds a "
    "fourth, artificially generated neutral-comment condition. All numbers are "
    "computed from the CSV files when the page loads."
)

neutral_texts = obs.loc[obs["comment_type"] == "neutral", "code"].map(comment_text)
canonical_neutral = neutral_texts.mode().iloc[0]
malformed_neutral = int((neutral_texts != canonical_neutral).sum())

with st.expander("Limitations of the neutral comments and of this data pair", expanded=False):
    st.markdown(
        "- Neutral comments were generated artificially to describe no behaviour. "
        f"The most common text is \"{canonical_neutral}\".\n"
        f"- In **{malformed_neutral} of {len(neutral_texts)}** neutral-comment "
        "implementations the comment text differs from that text. Inspection shows "
        "truncated or repeated fragments such as `General note: no beh`. The neutral "
        "condition is therefore not a clean control, and its large confidence drop "
        "cannot be attributed to the wording alone.\n"
        "- Neutral comments lower confidence by more than misleading comments do. "
        "Comparing accurate or misleading comments with the no-comment baseline "
        "mixes the effect of having any comment with the effect of its content.\n"
        "- Each problem contributes one correct and one buggy implementation, so "
        "observations are paired and not independent. All uncertainty estimates "
        "resample or pair by problem ID.\n"
        "- The buggy implementations are test-detected mutations, not formally "
        "verified incorrect code. P(Yes) is normalised over Yes/No tokens and is "
        "not a calibrated probability.\n"
        "- Model probabilities take few distinct values (log-odds steps of 1/64), "
        "so ties are common."
    )

v1, v2, v3, v4 = st.columns(4)
v1.metric("Observations", f"{len(obs):,}")
v2.metric("Unique problems", obs["problem_id"].nunique())
v3.metric("Comment conditions", obs["comment_type"].nunique())
v4.metric("Validation checks passed", f"{sum(ok for _, ok, _ in checks)} / {len(checks)}")
if not all_checks_pass:
    st.error("At least one data validation check failed. See the table below.")
with st.expander("Data validation details"):
    st.dataframe(
        pd.DataFrame(
            {
                "Check": [name for name, _, _ in checks],
                "Result": ["Pass" if ok else "FAIL" for _, ok, _ in checks],
                "Detail": [detail for _, _, detail in checks],
            }
        ),
        hide_index=True,
        width="stretch",
    )

# ------------------------------------------------------------------ AUROC
st.header("1. AUROC by comment condition")
st.caption(
    "AUROC measures how well P(Yes) separates correct from buggy code "
    "(0.5 = chance). Intervals are 95% percentile bootstrap intervals from "
    f"{N_BOOT:,} resamples of problem IDs (seed {BOOT_SEED}). Each resampled problem "
    "keeps its correct and buggy implementation, which preserves the pairing."
)


@st.cache_data
def bootstrap_auroc(scores: pd.DataFrame, n_boot: int, seed: int):
    """Point AUROC and paired bootstrap over problem IDs, per condition."""
    rng = np.random.default_rng(seed)
    pair_matrices = {}
    for cond in AUROC_CONDITIONS:
        d = scores[scores["comment_type"] == cond]
        correct = d[d["code_version"] == "correct"].set_index("problem_id")["score"].sort_index()
        buggy = d[d["code_version"] == "buggy"].set_index("problem_id")["score"].sort_index()
        if not correct.index.equals(buggy.index):
            raise ValueError("Correct and buggy rows are not paired by problem ID")
        c, b = correct.to_numpy()[:, None], buggy.to_numpy()[None, :]
        pair_matrices[cond] = (c > b) + 0.5 * (c == b)
    n = next(iter(pair_matrices.values())).shape[0]
    weights = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
    point, boot = {}, {}
    for cond, pair in pair_matrices.items():
        point[cond] = float(pair.mean())
        boot[cond] = ((weights @ pair) * weights).sum(axis=1) / n**2
    rows = []
    for cond in AUROC_CONDITIONS:
        low, high = np.percentile(boot[cond], [2.5, 97.5])
        row = {"condition": cond, "auroc": point[cond], "ci_low": low, "ci_high": high, "n_problems": n}
        if cond != "none":
            diff = boot[cond] - boot["none"]
            d_low, d_high = np.percentile(diff, [2.5, 97.5])
            row.update(
                diff_vs_none=point[cond] - point["none"], diff_low=d_low, diff_high=d_high
            )
        rows.append(row)
    return pd.DataFrame(rows)


human_scores = obs[obs["comment_type"].isin(AUROC_CONDITIONS)][
    ["problem_id", "code_version", "comment_type", "p_yes"]
].rename(columns={"p_yes": "score"})
auroc_human = bootstrap_auroc(human_scores, N_BOOT, BOOT_SEED)
auroc_original = None
if original is not None:
    original_scores = original[original["comment_type"].isin(AUROC_CONDITIONS)][
        ["problem_id", "code_version", "comment_type", "p_yes"]
    ].rename(columns={"p_yes": "score"})
    auroc_original = bootstrap_auroc(original_scores, N_BOOT, BOOT_SEED)

source_choices = ["Human-only run (results_with_neutral.csv)"]
if auroc_original is not None:
    source_choices.append("AI-assisted controlled run (controlled_results.csv)")
source = st.radio("Data source for the chart", source_choices, horizontal=True)
auroc_shown = auroc_human if source.startswith("Human-only") else auroc_original

auroc_fig = go.Figure()
auroc_fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="chance")
for _, row in auroc_shown.iterrows():
    cond = row["condition"]
    auroc_fig.add_trace(
        go.Scatter(
            x=[COND_LABELS[cond]],
            y=[row["auroc"]],
            mode="markers",
            marker=dict(size=13, color=COND_COLORS[cond]),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["ci_high"] - row["auroc"]],
                arrayminus=[row["auroc"] - row["ci_low"]],
                thickness=2.5,
                width=10,
                color=COND_COLORS[cond],
            ),
            showlegend=False,
            hovertemplate=(
                f"{COND_LABELS[cond]}<br>AUROC %{{y:.4f}}<br>"
                f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<extra></extra>"
            ),
        )
    )
auroc_fig.update_yaxes(range=[0.4, 1.0], title_text="AUROC (correct vs buggy)")
auroc_fig.update_layout(height=400, margin=dict(t=20))

left, right = st.columns([3, 2])
left.plotly_chart(auroc_fig, width="stretch", key="auroc_chart")
table = auroc_shown.copy()
table["Condition"] = table["condition"].map(COND_LABELS)
table["AUROC"] = table["auroc"].round(6)
table["95% CI"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(table["ci_low"], table["ci_high"])]
table["Change vs no comment"] = [
    "" if pd.isna(d) else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    for d, lo, hi in zip(table["diff_vs_none"], table["diff_low"], table["diff_high"])
]
right.dataframe(
    table[["Condition", "AUROC", "95% CI", "Change vs no comment"]],
    hide_index=True,
    width="stretch",
)
right.caption(
    "The change column uses the same bootstrap resamples for both conditions, so it "
    "is a paired difference. An interval containing 0 means no clear change."
)

validation = pd.DataFrame({
    "Condition": [COND_LABELS[c] for c in AUROC_CONDITIONS],
    "Notebook value": [NOTEBOOK_AUROC[c] for c in AUROC_CONDITIONS],
    "Human-only run (recomputed)": auroc_human["auroc"].round(6).to_numpy(),
})
if auroc_original is not None:
    validation["AI-assisted run (recomputed)"] = auroc_original["auroc"].round(6).to_numpy()
for col in [c for c in validation.columns if c.endswith("(recomputed)")]:
    validation[f"{col} matches notebook"] = [
        "Yes" if abs(a - b) <= AUROC_TOLERANCE else "No"
        for a, b in zip(validation[col], validation["Notebook value"])
    ]
with st.expander("Validation against notebook values", expanded=False):
    st.dataframe(validation, hide_index=True, width="stretch")
    if original is not None:
        shared = results[results["comment_type"].isin(AUROC_CONDITIONS)].merge(
            original[KEY + ["p_yes"]], on=KEY, suffixes=("_human", "_original"),
            validate="one_to_one"
        )
        if not shared.empty:
            max_diff = float((shared["p_yes_human"] - shared["p_yes_original"]).abs().max())
            st.caption(
                f"Matched {len(shared):,} rows across both runs. "
                f"Maximum absolute P(Yes) difference: {max_diff:.6f}. "
                "Check the individual validation columns for agreement with the notebook."
            )
    else:
        st.caption("AI-assisted controlled results CSV not found; cross-run comparison skipped.")


render_human_confidence_paths(obs, section="2. Paired confidence paths")

# ------------------------------------------------- four-condition comparison
st.header("3. Four-condition comparison")
st.caption(
    "Normalised P(Yes) = P(Yes) / (P(Yes) + P(No)) for correct versus buggy "
    "implementations. Every panel shares the same vertical axis and colours. "
    "Points are single implementations; the dashed line in each box is the mean."
)

box_fig = px.box(
    obs,
    x="code_version",
    y="p_norm",
    color="code_version",
    facet_col="comment_type",
    points="all",
    category_orders={"comment_type": CONDITIONS, "code_version": VERSION_ORDER},
    color_discrete_map=VERSION_COLORS,
    hover_data={"problem_id": True, "bug_type": True, "p_norm": ":.3f", "code_version": False, "comment_type": False},
    labels={"p_norm": "Normalised P(Yes)", "code_version": "Code version"},
)
box_fig.update_traces(boxmean=True, marker=dict(size=4, opacity=0.5), jitter=0.4)
box_fig.update_yaxes(range=[0, 1])
box_fig.update_xaxes(title_text="")
box_fig.for_each_annotation(lambda a: a.update(text=COND_LABELS[a.text.split("=")[-1]]))
box_fig.update_layout(height=440, showlegend=False, margin=dict(t=40))
st.plotly_chart(box_fig, width="stretch", key="small_multiples")

summary = (
    obs.pivot_table(index="comment_type", columns="code_version", values="p_norm", aggfunc="mean")
    .reindex(CONDITIONS)[VERSION_ORDER]
)
summary["Gap (correct minus buggy)"] = summary["correct"] - summary["buggy"]
summary.index = [COND_LABELS[c] for c in summary.index]
summary = summary.rename(columns={"correct": "Mean P(Yes), correct", "buggy": "Mean P(Yes), buggy"})
st.dataframe(summary.round(4), width="stretch")

# --------------------------------------------------- confidence-shift explorer
st.header("4. Confidence-shift explorer")
st.caption(
    "Each point is one implementation under one comment condition. The horizontal "
    "axis is its P(Yes) without a comment and the vertical axis is the change when a "
    "comment is added (the same implementation, paired). Click a point to inspect "
    "the code, the task and the probabilities."
)

c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
sel_conditions = c1.multiselect(
    "Comment condition", COMPARED, default=COMPARED, format_func=COND_LABELS.get
)
sel_versions = c2.multiselect("Code version", VERSION_ORDER, default=VERSION_ORDER)
all_bug_types = sorted(shift_df["bug_type"].unique())
sel_bugs = c3.multiselect("Mutation type", all_bug_types, default=all_bug_types)
metric = c4.radio(
    "Vertical axis",
    ["Log-odds change", "Percentage-point change"],
    help="Log-odds change is log(P(Yes)/P(No)) with the comment minus without. "
    "Percentage-point change is the difference in P(Yes) times 100.",
)
y_col = "shift_lo" if metric == "Log-odds change" else "shift_pp"
y_label = (
    "Change in log-odds of Yes (comment minus no comment)"
    if y_col == "shift_lo"
    else "Change in P(Yes), percentage points"
)

explorer = shift_df[
    shift_df["comment_type"].isin(sel_conditions)
    & shift_df["code_version"].isin(sel_versions)
    & shift_df["bug_type"].isin(sel_bugs)
].copy()

if st.button("Clear selection", key="clear_shift_selection"):
    st.session_state["shift_reset"] = st.session_state.get("shift_reset", 0) + 1
shift_key = f"shift_explorer_{st.session_state.get('shift_reset', 0)}"

picked = None
if explorer.empty:
    st.info("No observations match the current filters.")
else:
    explorer["comment_label"] = explorer["comment_type"].map(COND_LABELS)
    shift_fig = px.scatter(
        explorer,
        x="base_p",
        y=y_col,
        color="code_version",
        symbol="comment_label",
        color_discrete_map=VERSION_COLORS,
        category_orders={"code_version": VERSION_ORDER},
        custom_data=["problem_id", "code_version", "comment_type"],
        hover_data={
            "problem_id": True,
            "bug_type": True,
            "base_p": ":.3f",
            "cond_p": ":.3f",
            "shift_pp": ":+.2f",
            "shift_lo": ":+.3f",
            "code_version": False,
            "comment_label": True,
        },
        labels={
            "base_p": "P(Yes) without comment",
            y_col: y_label,
            "code_version": "Code version",
            "comment_label": "Comment",
            "cond_p": "P(Yes) with comment",
            "shift_pp": "Change (pp)",
            "shift_lo": "Change (log-odds)",
        },
    )
    shift_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    shift_fig.update_traces(marker=dict(size=8, opacity=0.7))
    shift_fig.update_layout(height=520, margin=dict(t=30), legend=dict(title_text=""))
    event = st.plotly_chart(
        shift_fig,
        width="stretch",
        key=shift_key,
        on_select="rerun",
        selection_mode=["points"],
    )
    st.caption(f"{len(explorer):,} observations shown.")

    chosen = [
        (int(p["customdata"][0]), p["customdata"][1], p["customdata"][2])
        for p in event.selection.points
        if p.get("customdata")
    ]
    if len(chosen) > 1:
        picked = st.selectbox(
            "Several points selected. Inspect:",
            chosen,
            format_func=lambda t: f"Problem {t[0]} | {t[1]} code | {COND_LABELS[t[2]]}",
        )
    elif chosen:
        picked = chosen[0]

if picked is None:
    st.info("Select a point in the scatterplot to inspect its code and probabilities.")
else:
    pid, version, cond = picked
    impl = obs[(obs["problem_id"] == pid) & (obs["code_version"] == version)].set_index("comment_type")
    row = shift_df[
        (shift_df["problem_id"] == pid)
        & (shift_df["code_version"] == version)
        & (shift_df["comment_type"] == cond)
    ].iloc[0]
    st.subheader(f"Problem {pid} | {version} code | {COND_LABELS[cond]}")
    st.markdown(
        f"**Mutation type:** {row['bug_type']}  \n"
        f"**Ground truth:** {'correct implementation' if version == 'correct' else 'buggy implementation (fails the MBPP tests)'}  \n"
        f"**Task specification:** {impl.loc['none', 'specification']}"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(Yes) without comment", f"{row['base_p']:.3f}")
    m2.metric("P(Yes) with comment", f"{row['cond_p']:.3f}", delta=f"{row['shift_pp']:+.2f} pp")
    m3.metric("Log-odds without comment", f"{row['base_lo']:.3f}")
    m4.metric("Log-odds with comment", f"{row['cond_lo']:.3f}", delta=f"{row['shift_lo']:+.3f}")
    code_left, code_right = st.columns(2)
    code_left.markdown("**Without comment**")
    code_left.code(clean_code(impl.loc["none", "code"]), language="python")
    code_right.markdown(f"**{COND_LABELS[cond]}**")
    code_right.code(clean_code(impl.loc[cond, "code"]), language="python")
    st.markdown("**All four conditions for this implementation**")
    st.dataframe(
        pd.DataFrame(
            {
                "Condition": [COND_LABELS[c] for c in CONDITIONS],
                "Comment text": [comment_text(impl.loc[c, "code"]) or "(none)" for c in CONDITIONS],
                "P(Yes)": [round(float(impl.loc[c, "p_norm"]), 4) for c in CONDITIONS],
                "Log-odds": [round(float(impl.loc[c, "logodds"]), 4) for c in CONDITIONS],
            }
        ),
        hide_index=True,
        width="stretch",
    )

# ------------------------------------------------------------ bug-type heatmap
st.header("5. Shift by mutation type")
st.caption(
    "Mean change from the same implementation without a comment, per mutation "
    "category, comment condition and code version. Rows are labelled with the number "
    "of problems (n). Categories with few problems are marked and faded because their "
    "means are unreliable."
)

h1, h2, h3 = st.columns([2, 2, 2])
heat_metric = h1.radio(
    "Shift measure",
    ["Log-odds change", "Percentage-point change"],
    key="heat_metric",
)
small_n = h2.slider("Mark categories with fewer problems than", 2, 30, 10)
hide_small = h3.checkbox("Hide small categories", value=False)
heat_col = "shift_lo" if heat_metric == "Log-odds change" else "shift_pp"
heat_unit = "log-odds" if heat_col == "shift_lo" else "pp"

counts = shift_df.groupby(["bug_type", "code_version"])["problem_id"].nunique().unstack("code_version")
n_by_type = counts.min(axis=1)
order = n_by_type.sort_values(ascending=False, kind="stable").index.tolist()
if hide_small:
    order = [b for b in order if n_by_type[b] >= small_n]

if not order:
    st.info("No mutation categories meet the minimum size.")
else:
    labels = [
        f"{b} (n={int(n_by_type[b])}){' †' if n_by_type[b] < small_n else ''}" for b in order
    ]
    heat_fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=[f"{v.capitalize()} code" for v in VERSION_ORDER],
    )
    z_abs = 0.0
    for col_idx, version in enumerate(VERSION_ORDER, start=1):
        d = shift_df[shift_df["code_version"] == version]
        pivot = (
            d.pivot_table(index="bug_type", columns="comment_type", values=heat_col, aggfunc="mean")
            .reindex(index=order, columns=COMPARED)
        )
        z = pivot.to_numpy()
        z_abs = max(z_abs, float(np.nanmax(np.abs(z))))
        n_matrix = np.repeat(n_by_type.reindex(order).to_numpy()[:, None], len(COMPARED), axis=1)
        heat_fig.add_trace(
            go.Heatmap(
                z=z,
                x=[COND_LABELS[c] for c in COMPARED],
                y=labels,
                coloraxis="coloraxis",
                customdata=n_matrix,
                text=np.vectorize(lambda v: f"{v:+.2f}")(z),
                texttemplate="%{text}",
                hovertemplate=(
                    "%{y}<br>%{x}<br>Mean change: %{z:+.3f} " + heat_unit +
                    "<br>n = %{customdata} problems<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )
        for idx, b in enumerate(order):
            if n_by_type[b] < small_n:
                heat_fig.add_shape(
                    type="rect",
                    xref=f"x{col_idx}" if col_idx > 1 else "x",
                    yref=f"y{col_idx}" if col_idx > 1 else "y",
                    x0=-0.5,
                    x1=len(COMPARED) - 0.5,
                    y0=idx - 0.5,
                    y1=idx + 0.5,
                    fillcolor="rgba(255,255,255,0.65)",
                    line=dict(color="gray", width=1, dash="dot"),
                    layer="above",
                )
    heat_fig.update_yaxes(autorange="reversed")
    heat_fig.update_layout(
        height=max(380, 36 * len(order) + 140),
        margin=dict(t=50),
        coloraxis=dict(
            colorscale="RdBu",
            cmid=0,
            cmin=-z_abs,
            cmax=z_abs,
            colorbar=dict(title=f"Mean change<br>({heat_unit})"),
        ),
    )
    st.plotly_chart(heat_fig, width="stretch", key="bug_heatmap")
    st.caption(
        "Blue means higher confidence with the comment than without, red means lower. "
        "† marks categories with fewer problems than the threshold; treat these as "
        "anecdotal. Each cell is the mean of paired per-problem changes."
    )

# ------------------------------------------------------ statistical comparisons
st.header("6. Statistical comparisons")
st.caption(
    "Paired Wilcoxon signed-rank tests on the per-problem change in **log-odds** "
    "(comment minus no comment), computed with scipy (zero differences dropped). "
    "The percentage-point columns are descriptive only and are not what the test uses. "
    "p-values are not corrected for the six comparisons shown "
    "(Bonferroni threshold 0.05 / 6 = 0.0083)."
)

csv_reference = {}
for _, r in neutral_cmp.iterrows():
    csv_reference[(r["code_version"], r["condition"])] = (r["n"], r["median_shift"], r["p_value"])
for _, r in wilcoxon_csv.iterrows():
    csv_reference[(r["code_version"], r["comparison"].replace("none_to_", ""))] = (
        r["n"], r["median_shift"], r["p_value"],
    )

stat_rows = []
for version in VERSION_ORDER:
    for cond in COMPARED:
        d = shift_df[(shift_df["code_version"] == version) & (shift_df["comment_type"] == cond)]
        diff = d["shift_lo"]
        nonzero = int((diff != 0).sum())
        if np.allclose(diff.to_numpy(), 0):
            from types import SimpleNamespace
            test = SimpleNamespace(statistic=0.0, pvalue=1.0)
        else:
            test = wilcoxon(diff)
        ref = csv_reference.get((version, cond))
        matches = (
            ref is not None
            and int(ref[0]) == len(d)
            and bool(np.isclose(ref[1], diff.median(), atol=1e-9))
            and bool(np.isclose(ref[2], test.pvalue, rtol=1e-6))
        )
        stat_rows.append(
            {
                "code_version": version,
                "condition": cond,
                "n": len(d),
                "n_nonzero": nonzero,
                "median_lo": diff.median(),
                "mean_lo": diff.mean(),
                "statistic": test.statistic,
                "p_value": test.pvalue,
                "csv_p": np.nan if ref is None else ref[2],
                "matches_csv": matches,
                "mean_pp": d["shift_pp"].mean(),
                "median_pp": d["shift_pp"].median(),
            }
        )
stats_df = pd.DataFrame(stat_rows)

lo_table = pd.DataFrame(
    {
        "Code version": stats_df["code_version"].str.capitalize(),
        "Condition vs no comment": stats_df["condition"].map(COND_LABELS),
        "n problems": stats_df["n"],
        "n non-zero": stats_df["n_nonzero"],
        "Median change (log-odds)": stats_df["median_lo"].round(4),
        "Mean change (log-odds)": stats_df["mean_lo"].round(4),
        "Wilcoxon W": stats_df["statistic"].round(1),
        "p-value": stats_df["p_value"].map(lambda p: f"{p:.2e}"),
        "p-value in CSV": stats_df["csv_p"].map(lambda p: f"{p:.2e}"),
        "Reproduces CSV": stats_df["matches_csv"].map({True: "Yes", False: "No"}),
    }
)
pp_table = pd.DataFrame(
    {
        "Code version": stats_df["code_version"].str.capitalize(),
        "Condition vs no comment": stats_df["condition"].map(COND_LABELS),
        "Mean change (pp)": stats_df["mean_pp"].round(2),
        "Median change (pp)": stats_df["median_pp"].round(2),
    }
)

t1, t2 = st.tabs(["Log-odds changes (tested)", "Percentage-point changes (descriptive)"])
with t1:
    st.dataframe(lo_table, hide_index=True, width="stretch")
    st.caption(
        "Log-odds is log(P(Yes)/P(No)). A change of -0.64 means the odds of answering "
        "Yes were multiplied by about exp(-0.64) = 0.53. p-values are reproduced from "
        "the raw results and compared with logodds_wilcoxon_results.csv and "
        "neutral_condition_comparison.csv."
    )
with t2:
    st.dataframe(pp_table, hide_index=True, width="stretch")
    st.caption(
        "Percentage points are differences in P(Yes) multiplied by 100. Because P(Yes) "
        "is high (about 0.83 on average), the same log-odds change gives a smaller "
        "percentage-point change near the top of the scale."
    )

plot_df = stats_df.copy()
plot_df["Condition"] = plot_df["condition"].map(COND_LABELS)
bar_lo = px.bar(
    plot_df,
    x="Condition",
    y="median_lo",
    color="condition",
    facet_col="code_version",
    color_discrete_map=COND_COLORS,
    category_orders={"condition": COMPARED, "code_version": VERSION_ORDER},
    labels={"median_lo": "Median change in log-odds"},
    text=plot_df["p_value"].map(lambda p: f"p={p:.1e}"),
)
bar_lo.update_layout(height=360, showlegend=False, margin=dict(t=40))
bar_lo.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].capitalize() + " code"))
bar_lo.update_xaxes(title_text="")

bar_pp = px.bar(
    plot_df,
    x="Condition",
    y="mean_pp",
    color="condition",
    facet_col="code_version",
    color_discrete_map=COND_COLORS,
    category_orders={"condition": COMPARED, "code_version": VERSION_ORDER},
    labels={"mean_pp": "Mean change in P(Yes), percentage points"},
)
bar_pp.update_layout(height=360, showlegend=False, margin=dict(t=40))
bar_pp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].capitalize() + " code"))
bar_pp.update_xaxes(title_text="")

b1, b2 = st.columns(2)
b1.markdown("**Log-odds scale (the quantity tested)**")
b1.plotly_chart(bar_lo, width="stretch", key="bar_logodds")
b2.markdown("**Percentage-point scale (descriptive)**")
b2.plotly_chart(bar_pp, width="stretch", key="bar_pp")

if not stats_df["matches_csv"].all():
    st.warning("Some recomputed Wilcoxon results do not match the CSV files.")
