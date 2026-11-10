from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Does AI Trust Code or Comments?",
    layout="wide"
)

BASE = Path(__file__).resolve().parent


@st.cache_data
def load_data():
    results = pd.read_csv(BASE / "controlled_results.csv")
    shifts = pd.read_csv(BASE / "confidence_shifts.csv")
    return results, shifts


results, shifts = load_data()

st.title("Does AI Trust Code or Comments?")
st.caption(
    "Investigating Qwen2.5-Coder-1.5B-Instruct "
    "using the MBPP Python benchmark"
)

st.header("Research Question")
st.write(
    "Does misleading code documentation affect AI judgments "
    "about Python program correctness?"
)

st.header("Methodology")
st.markdown(
    "- **Baseline:** 257 MBPP Python problems.\n"
    "- **Controlled experiment:** 178 problems, each with a correct "
    "reference implementation and a mutated implementation detected "
    "by the supplied MBPP tests.\n"
    "- **Comment conditions:** no comment, accurate localized comment, "
    "misleading localized comment.\n"
    "- **Measure:** normalized P(Yes) for the model's correctness judgment."
)

st.header("Dataset Summary")

a, b, c, d = st.columns(4)
a.metric("Evaluations", f"{len(results):,}")
b.metric("Problems", results["problem_id"].nunique())
c.metric("Mutation types", results["bug_type"].nunique())
d.metric("Code versions", results["code_version"].nunique())

st.header("Key Findings")

mean_shift = shifts.groupby("code_version")["misleading_shift"].mean()

e, f = st.columns(2)
e.metric(
    "Mean misleading shift, correct code",
    f"{mean_shift['correct'] * 100:+.2f} pp"
)
f.metric(
    "Mean misleading shift, buggy code",
    f"{mean_shift['buggy'] * 100:+.2f} pp"
)

st.markdown(
    "- Qwen strongly favors Yes predictions.\n"
    "- Misleading comments reduce confidence on correct code.\n"
    "- Min/Max operations show particularly large decreases, "
    "and the effect persists under a second wording.\n"
    "- Two buggy implementations flip from No to Yes when misleading "
    "comments are introduced (Problems 62 and 89)."
)

st.info(
    "Open **Interactive Analysis** in the sidebar to explore "
    "confidence shifts, heatmaps, and prediction flips."
)
