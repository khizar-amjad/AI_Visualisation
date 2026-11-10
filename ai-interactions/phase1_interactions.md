# Phase I — AI Interaction Record

**Project:** Does AI Trust Code or Comments?
**Model:** Qwen2.5-Coder-1.5B-Instruct
**Dataset:** MBPP (257 problems)
**AI tools:** ChatGPT (research assistance), GitHub Copilot (implementation)

## 1. Research Design and Dataset Expansion

**Interaction:** Asked ChatGPT to develop an experiment investigating whether accurate and misleading comments influence AI judgments about Python code correctness.

**AI contribution:** Proposed a controlled experiment with correct/buggy implementations and three comment conditions: none, accurate, and misleading.

**Human decision:** Requested expansion beyond the initial small sample to investigate systematic behavior.

**Outcome:** Established a 257-problem baseline and later a controlled experiment with 178 problems and 1,068 evaluations.

## 2. AI-Generated Annotations — Failure and Correction

**Interaction:** Used Qwen to generate accurate and misleading comments automatically.

**Problem:** Some generated comments were semantically incorrect, excessively verbose, or revealed their own falsehood.

**Example:** For MBPP Problem 59, AI incorrectly described a function calculating an octagonal number as checking whether a number was octagonal.

**Correction:** Introduced automated screening, human review, and later deterministic operation-specific comments.

**Lesson:** AI-generated descriptions cannot be treated as verified ground truth.

## 3. Automated Bug Generation

**Interaction:** Asked ChatGPT to expand mutation testing beyond simple `min/max` substitutions.

**AI contribution:** Helped implement AST-based mutations covering arithmetic, comparisons, equality, and Boolean operations.

**Verification:** Tested generated implementations against MBPP assertions and checked AST differences.

**Outcome:** 470 test-detected mutations across the combined datasets, covering 178 unique problems.

**Lesson:** AI accelerated dataset construction, but executable tests and structural validation were essential.

## 4. Visualization Design and Investigation

**Interaction:** Asked ChatGPT to interpret confidence distributions, mutation-type differences, and prediction changes.

**AI contribution:** Suggested heatmaps, confidence-shift plots, subgroup comparisons, and wording-robustness experiments.

**Key finding:** Misleading comments decreased P(Yes) for all 20 correct-code Min/Max examples, with an average decrease of 11.99 percentage points.

**Human decision:** Rejected reliance on crowded statistical plots and requested visualizations aligned with the lecture's investigative approach.

**Outcome:** Shifted toward linked projection, confidence-path, and prediction-transition visualizations.

## 5. Robustness and Critical Interpretation

**Interaction:** Tested whether changing misleading-comment wording affected the Min/Max finding.

**Outcome:** Both formulations decreased confidence in all 20 examples; the alternative produced an average decrease of 11.23 percentage points.

**Limitation:** Comment wording and mutation category may be confounded. The results do not prove that Qwen semantically understands the contradiction.

## 6. Dashboard Implementation and Debugging

**Interaction:** Used ChatGPT and Copilot to build and refine a Streamlit dashboard, prepare GitHub documentation, and implement CodeBERT/UMAP preprocessing.

**Challenges:** Python dependency conflicts, incomplete virtual-environment installations, confusing plots, and missing projection files.

**Corrections:** Separated dashboard and ML dependencies, simplified navigation, and required real embeddings instead of fabricated visualization coordinates.

**Lesson:** AI-assisted implementation increased productivity but introduced additional debugging and verification work.

## 7. Reflection

**Benefits:** Rapid prototyping, broader experimental coverage, multiple visualization alternatives, and faster exploration of research hypotheses.

**Limitations:** Incorrect semantic annotations, imperfect validation heuristics, overly complicated initial designs, and technical integration problems.

**Best practices:** Validate AI-generated data independently, preserve reproducible intermediate results, question AI interpretations, and use visualization to investigate behavior rather than merely display metrics.

**Overall:** GenAI was most valuable as an iterative research collaborator. Human decisions and independent validation were necessary to turn its suggestions into defensible findings.

---

*This is a curated summary of material interactions, not a verbatim transcript. Exact prompts and full responses should be retained separately if required.*
