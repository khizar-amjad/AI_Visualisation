# Does AI Trust Code or Comments?

Interactive visual analysis of Qwen2.5-Coder-1.5B-Instruct
on Python correctness judgments.

## Research Question

**Does misleading code documentation influence an AI model's ability to distinguish correct Python implementations from buggy ones?**

We investigate how Qwen2.5-Coder-1.5B-Instruct evaluates Python code under three comment conditions: no comment, accurate comment, and misleading comment.

For each evaluation, the model receives a task specification and Python implementation and produces a Yes/No correctness judgment. We analyze both the predicted answer and the normalized P(Yes) score.

Our controlled experiment contains **178 MBPP problems × 2 code versions × 3 comment conditions = 1,068 evaluations**.

The interactive dashboard explores confidence changes, mutation-type differences, prediction transitions, and code-similarity projections.

## AI-Assisted Research Documentation

- [Phase I AI Interaction Record](ai-interactions/phase1_interactions.md)
- [Midpoint AI Reflection](ai_reflection_midpoint.md)
- [Research Findings](docs/research_findings.md)

These documents describe the contributions of Generative AI, human decisions, validation procedures, failed approaches, and methodological limitations.

## Experiment

- Dataset: MBPP sanitized test split
- Baseline: 257 Python problems
- Controlled experiment: 178 problems
- Correct and benchmark-detected buggy implementations
- Three conditions: no comment, accurate, misleading
- Total controlled evaluations: 1,068

## Key Findings

- Correct code accepted: 534/534 evaluations
- Buggy code accepted: 528/534 evaluations
- Mean misleading-comment confidence shift:
  - Correct code: -2.37 percentage points
  - Buggy code: +0.25 percentage points
- Min/max mutations (`max_to_min`, `min_to_max`): all 20 of 20 correct-code
  problems showed lower confidence with misleading comments
  (135 of 178 correct-code problems overall)
- Two buggy prediction flips: Problems 62 and 89

## Run Dashboard

Create a virtual environment and install the dashboard dependencies (Windows PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

Start Streamlit:

    python -m streamlit run dashboard/app.py

The app has three pages: Overview (`app.py`), Interactive Analysis (AI-assisted
data) and Human-Only Analysis (the four-condition data pair in
`dashboard/human-only/`, including AUROC with bootstrap intervals, Wilcoxon tests and
a neutral-comment control). The Human-Only page needs `scipy`, which is in
`requirements.txt`.

### Code projection (optional regeneration)

The Projection Space Explorer reads the precomputed `dashboard/code_projection.csv`
(`problem_id`, `umap_x`, `umap_y`). The dashboard never computes embeddings and does
not need PyTorch or Transformers. To regenerate the file from the 178 correct-code
implementations (`microsoft/codebert-base`, attention-mask mean pooling, L2
normalisation, UMAP with `random_state=42`):

    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements-ml.txt
    python scripts/build_code_projection.py

The model is downloaded from Hugging Face on first use. UMAP parameters and library
versions are written to `dashboard/code_projection_params.json`.

## Repository Contents

- dashboard/: Streamlit application and runtime data
- scripts/: Preprocessing scripts (code projection)
- requirements.txt: Dashboard dependencies
- requirements-ml.txt: Optional dependencies for the preprocessing script
- archive/: Retired pages kept for reference
- docs/: Research findings and methodology
- ai-interactions/, ai_reflection_midpoint.md: AI-assistance documentation

## Limitations

Code correctness was evaluated using supplied MBPP tests.
The mutations are test-detected bugs, not formally verified
incorrect implementations.

Comments describe localized operations. Their semantic
validity has not been independently verified for every sample.

P(Yes) is normalized over Yes/No tokens and should not be
interpreted as calibrated correctness probability.

Repeated evaluations share underlying MBPP problems.

## AI Assistance

Generative AI supported code generation, debugging,
annotation design, visualization, and interpretation.
Its suggestions were checked and sometimes corrected.
