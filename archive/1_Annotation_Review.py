
import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(
    page_title="Annotation Review",
    layout="wide"
)

BASE = Path(__file__).resolve().parents[1] / "dashboard"
REVIEW_FILE = BASE / "approved_annotations.csv"

@st.cache_data
def load_sources():
    problems = pd.read_csv(BASE / "problems.csv")
    candidates = pd.read_csv(
        BASE / "annotation_candidates.csv"
    ).fillna("")
    return problems, candidates

problems, candidates = load_sources()

tests_df = pd.read_csv(BASE / "problem_tests.csv")


def load_approvals():
    if REVIEW_FILE.exists():
        return pd.read_csv(REVIEW_FILE).fillna("")
    return pd.DataFrame(columns=[
        "problem_id", "accurate_comment",
        "misleading_comment", "source", "reviewer_status"
    ])

approvals = load_approvals()

st.title("Annotation Review")
st.caption(
    "Verify actual behavior against code before "
    "approving comments for the experiment."
)

a, b, c = st.columns(3)
a.metric("Total problems", len(problems))
b.metric("Approved in this interface",
         approvals["problem_id"].nunique())
c.metric("Remaining",
         len(problems) - approvals["problem_id"].nunique())

approved_ids = set(approvals["problem_id"].astype(int))

show_pending = st.checkbox(
    "Show only problems not yet approved",
    value=True
)

ids = sorted(problems["problem_id"].astype(int))

if show_pending:
    ids = [i for i in ids if i not in approved_ids]

if not ids:
    st.success("No problems remaining in this view.")
    st.stop()

task_id = st.selectbox("Problem", ids)

p = problems.loc[
    problems["problem_id"] == task_id
].iloc[0]

st.subheader(f"MBPP Problem {task_id}")
st.write("**Task:**", p["specification"])
st.code(p["original_code"], language="python")

with st.expander("View MBPP test cases"):
    matching_tests = tests_df.loc[
        tests_df["problem_id"] == task_id
    ]

    if not matching_tests.empty:
        test_list = json.loads(
            matching_tests.iloc[0]["test_cases"]
        )

        st.code(
            "\\n".join(test_list),
            language="python"
        )
    else:
        st.info("No test cases found.")


options = candidates.loc[
    candidates["problem_id"] == task_id
].reset_index(drop=True)

if options.empty:
    st.warning("No candidate comments available.")
    st.stop()

source_options = sorted(options["source"].unique().tolist())

selected_sources = st.multiselect(
    "Candidate sources",
    options=source_options,
    default=source_options
)

options = options[
    options["source"].isin(selected_sources)
].reset_index(drop=True)

if options.empty:
    st.warning("No candidates match the selected sources.")
    st.stop()

candidate_index = st.selectbox(
    "Choose a comment candidate",
    range(len(options)),
    format_func=lambda i: (
        f"{options.iloc[i]['source']} — Candidate {i+1}"
    )
)

candidate = options.iloc[candidate_index]

existing = approvals.loc[
    approvals["problem_id"] == task_id
]

if not existing.empty:
    default_accurate = existing.iloc[0]["accurate_comment"]
    default_misleading = existing.iloc[0]["misleading_comment"]
else:
    default_accurate = candidate["accurate_comment"]
    default_misleading = candidate["misleading_comment"]

st.write("**Candidate source:**", candidate["source"])

with st.form(
    key=f"review_{task_id}_{candidate_index}"
):
    accurate = st.text_area(
        "Accurate comment",
        value=default_accurate,
        height=100
    )

    misleading = st.text_area(
        "Misleading comment",
        value=default_misleading,
        height=100
    )

    verified = st.checkbox(
        "I checked both claims against the actual code."
    )

    save = st.form_submit_button(
        "Approve and save",
        type="primary"
    )

if save:
    accurate = accurate.strip()
    misleading = misleading.strip()

    if not verified:
        st.error("Please verify the claims before approval.")
    elif not accurate or not misleading:
        st.error("Both comments must be provided.")
    elif accurate.casefold() == misleading.casefold():
        st.error("Comments must describe different behavior.")
    elif "\\n" in accurate or "\\n" in misleading:
        st.error("Each comment must be a single line.")
    else:
        new_row = {
            "problem_id": task_id,
            "accurate_comment": accurate,
            "misleading_comment": misleading,
            "source": candidate["source"],
            "reviewer_status": "approved"
        }

        approvals = approvals[
            approvals["problem_id"] != task_id
        ]

        approvals = pd.concat([
            approvals,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        temp_path = REVIEW_FILE.with_suffix(".tmp")
        approvals.to_csv(temp_path, index=False)
        temp_path.replace(REVIEW_FILE)

        st.success("Annotation approved and saved!")
        st.rerun()
