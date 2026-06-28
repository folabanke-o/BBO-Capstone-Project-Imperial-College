# Week 4: Four New Bests in One Week

[Back to README](../README.md)  |  [Previous: Week 3](week3.md)

---

## Objective

Correct the specific issue that caused F5 to decline in Week 3, reassess the use of Thompson Sampling for F7 and F8 in light of mixed results, and continue exploiting confirmed good regions for the remaining functions.

## Method Changes From Week 3

| Function | Week 3 method | Week 4 method | Reason for change |
|---|---|---|---|
| F5 | UCB | Expected Improvement, with one input dimension constrained to a higher minimum value | Root cause of the Week 3 decline was traced to that dimension dropping too low |
| F7 | Thompson Sampling | Expected Improvement | Testing whether precise exploitation outperforms posterior sampling now that a strong region is confirmed |
| F8 | Thompson Sampling | Upper Confidence Bound, returned to the Week 1 best region | Three weeks without a repeat of the Week 1 result suggested returning to that specific region rather than exploring further |

## Results

| Function | Best before Week 4 | Week 4 result | Outcome |
|---|---|---|---|
| F1 | ~0.000000 | -1.6 x 10⁻¹⁶ | No meaningful gain |
| F2 | 0.611205 | 0.332479 | Best result since the initial dataset |
| F3 | -0.021428 | -0.006082 | New best |
| F4 | 0.533858 | 0.450709 | Second-best result on record |
| F5 | 3511.612 | 3905.150 | New best (+21.5% on the previous record) |
| F6 | -0.703638 | -0.680152 | New best |
| F7 | 1.919249 | 2.018863 | New best |
| F8 | 9.965293 | 9.960920 | Within 0.0044 of the all-time best |

Four functions (F3, F5, F6, F7) set new all-time bests in the same week, the strongest single round across the project so far. Three further functions (F2, F4, F8) recorded their second-best result ever, indicating the search was converging across the board even where the record itself was not broken.

## Why the Dimension Fix on F5 Worked

The Week 3 decline on F5 was caused by one specific input dimension falling outside the range that had produced strong results in the two prior weeks. Raising the lower bound on that dimension for Week 4 fully reversed the decline and produced a new all-time best, confirming the original diagnosis.

## Supporting Reflection

A separate written reflection for this week addressed how the function evaluations behaved like support vectors near regions of rapid change, why a Gaussian Process surrogate was preferred over a neural network at this data scale, how the project could be reframed as a classification task, and which model type best balanced interpretability against flexibility for this problem. The full reflection, including the supporting figures, is available as `week4_reflection.docx` in the project archive.

## Carried Into Week 5

- Tighten exploitation further for the four functions that just set new bests, since each has now demonstrated a reliable, productive region.
- Introduce a fundamentally different approach for F1, which has shown no meaningful signal across four consecutive queries.
- Make a final close attempt to beat the Week 1 record on F8, given how close Week 4's result came.

[Next: Week 5](week5.md)
