
# AI Reflection — Phase I Midpoint

**Project:** Does AI Trust Code or Comments?
**Course:** AI and Visualization — WS 2026/27
**Track:** AI-assisted Vis4AI, Phase I

**AI-aided pair members:**

- **Student 1:** Hovhannes Davtyan — K12349259
- **Student 2:** [First name, last name] — [Student ID]

## 1. How We Used Generative AI

During Phase I, we used ChatGPT as a research and visualization collaborator and GitHub Copilot for implementation and debugging.

AI helped us design experiments, generate Python code, construct datasets, interpret results, and develop an interactive Streamlit dashboard.

Our initial approach used a small number of MBPP problems with automatically generated accurate and misleading comments. As the investigation developed, we expanded to 257 baseline problems and a controlled experiment containing 178 problems, each with correct and mutated implementations, evaluated under three comment conditions.

The controlled experiment produced **1,068 model evaluations**.

## 2. Benefits of AI Assistance

The main benefit was the speed at which we could explore different experimental approaches.

ChatGPT helped us develop an AST-based mutation pipeline, extending our initial three buggy examples to 178 problems with test-detected incorrect implementations.

AI also suggested different visualizations and helped us investigate unexpected patterns. For example, we discovered that misleading comments caused a particularly strong decrease in Qwen's confidence for correct code involving minimum and maximum operations.

We investigated this finding further using an alternative comment formulation. Both formulations reduced confidence in all 20 Min/Max examples.

Without AI assistance, building and exploring this many experimental variations would likely have required substantially more manual work.

## 3. Failures and Limitations

AI assistance also introduced significant problems.

**Incorrect annotations:** Automatically generated comments sometimes misrepresented the actual code. For example, one annotation confused calculating an octagonal number with checking whether a number was octagonal.

**Weak initial validation:** Some automatically generated comments were unnecessarily long, ambiguous, or revealed that they were incorrect.

**Visualization design:** Early AI suggestions focused heavily on standard charts such as box plots, scatter plots, and heatmaps. Although useful, these did not sufficiently support interactive investigation or communicate our strongest findings.

**Technical difficulties:** AI-generated implementation steps required corrections related to package dependencies, virtual environments, data paths, and integration of the embedding projection.

These issues showed that generating code or explanations quickly does not guarantee their correctness or usefulness.

## 4. Human Decisions and Corrections

We actively challenged and modified AI suggestions throughout the process.

We decided to expand beyond the small pilot dataset, use benchmark tests to validate mutated implementations, and replace unreliable generated descriptions with deterministic operation-specific comments.

We also rejected overly complicated and crowded dashboard designs. After revisiting the course slides, we directed the visualization toward linked exploratory views, including projection spaces, confidence changes, and prediction transitions.

The most important methodological decision was to separate verified experimental evidence from automatically generated or insufficiently validated material.

## 5. What We Learned About AI4Vis

Our experience suggests that Generative AI is most useful when treated as an **iterative collaborator rather than an unquestioned authority**.

AI helped us explore more alternatives and implement more ambitious experiments. However, verifying its outputs introduced additional work, particularly for semantic correctness and visualization design.

We also learned that visualizations should not merely display aggregate metrics. They should help investigate unexpected model behavior and test possible explanations.
Our strongest finding—the consistent Min/Max confidence decrease—emerged through repeated visual analysis, critical discussion, and follow-up experiments.

##6. Conclusion and Next Steps

Overall, AI assistance substantially broadened our investigation, but human judgment remained essential for research direction, validation, and interpretation.

For Phase II, we plan to investigate the internal model mechanisms associated with the behaviors identified in Phase I, potentially using representations, attention, or attribution techniques.

We will also compare our AI-assisted process and findings with those of the human-only pair to evaluate how GenAI affected the depth, efficiency, and reliability of the investigation.