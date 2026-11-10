# Research Findings

Date: 9 November 2026

## Research Question

Does misleading code documentation affect AI judgments
about Python program correctness?

## Experimental Design

257 MBPP baseline problems.

178 problems selected for the controlled experiment.
Each has a correct reference implementation and a
mutated implementation detected by supplied MBPP tests.

Three comment conditions:
- None
- Accurate localized comment
- Misleading localized comment

Total: 1,068 evaluations.

## Findings

1. Qwen strongly favors Yes predictions.
2. Misleading comments reduce confidence on correct code.
3. Min/Max operations show particularly large decreases.
4. The Min/Max effect persists under a second wording.
5. Two buggy implementations flip from No to Yes
   when misleading comments are introduced.

## Important Numbers

Mean misleading shift, correct: -0.0237
Mean misleading shift, buggy: +0.0025

Min/Max subgroup: 20 problems
Mean Min/Max shift: -0.1199
Alternative wording mean shift: -0.1123

Prediction flips: Problems 62 and 89

## Methodology Limitations

- Correctness assessed using benchmark tests
- Strong Yes-answer tendency
- Localized operation comments, not full documentation
- Some annotation semantics unverified
- Mutation type and comment wording may be confounded
- Prediction flips near 0.5 threshold
- One model and one benchmark

## Future Work

Investigate internal model representations and
attention/attribution methods in Phase II.
