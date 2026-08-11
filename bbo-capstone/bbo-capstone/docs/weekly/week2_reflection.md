# Bayesian Optimisation Capstone: Week 2 Reflection

> **Competition:** Black-box function optimisation across 8 synthetic functions
> **Method:** Gaussian Process surrogate modelling with UCB acquisition
> **Week 2 objective:** Incorporate Week 1 results, improve the pipeline, and submit better-informed query points

---

## Table of Contents

1. [Overview](#overview)
2. [What Changed from Week 1](#what-changed-from-week-1)
3. [Updated Datasets](#updated-datasets)
4. [Pipeline Improvements](#pipeline-improvements)
5. [Step-by-Step: What Was Done](#step-by-step-what-was-done)
6. [Week 2 Queries Submitted](#week-2-queries-submitted)
7. [Function-by-Function Analysis](#function-by-function-analysis)
8. [Issues and Challenges](#issues-and-challenges)
9. [The Override Problem: Full Account](#the-override-problem-full-account)
10. [What to Improve for Week 3](#what-to-improve-for-week-3)
11. [Week 3 Strategy Predictions](#week-3-strategy-predictions)

---

## Overview

Week 2 was the first true iteration of the Bayesian Optimisation loop. Unlike Week 1, which operated entirely on the provided initial data, Week 2 required incorporating the results returned by the competition organisers for each Week 1 query. Each function's dataset grew by one observation, the GP was refitted on the updated data, and new query points were generated.

The central challenge of Week 2 was not algorithmic — it was practical. Generating good queries required diagnosing why several functions were being sent to poor regions, applying targeted corrections, and resolving persistent issues with how those corrections were saved and propagated through the notebook. Five of the eight functions required manual intervention before submission.

The week produced the best query set to date. F6 in particular, which had been the most persistent problem, was ultimately fixed using a direct manual perturbation approach that bypassed the GP entirely, placing the query within 0.017 of the best known recipe.

---

## What Changed from Week 1

| Area | Week 1 approach | Week 2 improvement |
|------|----------------|-------------------|
| Dataset | Original data only | Original + Week 1 result stacked per function |
| Kernel bounds | Default tight bounds | Widened: constant (1e-6,1e6), RBF (1e-4,1e4), noise (1e-10,1e1) |
| Candidate sampling | `np.random.uniform` with no seed | `np.random.seed(42)` set before sampling |
| Candidate coverage | Pure random uniform | Sobol quasi-random sequences for better space filling |
| kappa | Uniform 2.0 for all functions | Per-function kappa based on Week 1 outcome |
| GP restarts | 10 for all functions | 10 for 2D-3D, 15 for 4D-5D, 20 for 6D-8D |
| Query corrections | None | Constrained search boxes for F1, F3, F4, F5; manual fix for F6 |

---

## Updated Datasets

Before running the GP, the Week 1 query and its returned output were stacked onto each function's original dataset:

```python
X_updated = np.vstack([X_original, week1_query.reshape(1, -1)])
Y_updated = np.append(Y_original, week1_output)
```

| Function | Original obs | After Week 1 | New best going into Week 2 |
|----------|-------------|--------------|---------------------------|
| F1 | 10 | 11 | 0.000000 (unchanged) |
| F2 | 10 | 11 | 0.611205 (unchanged) |
| F3 | 15 | 16 | -0.034835 (unchanged) |
| F4 | 30 | 31 | -0.869002 (improved from -4.026) |
| F5 | 20 | 21 | 3019.659838 (improved from 1088.860) |
| F6 | 20 | 21 | -0.714265 (unchanged) |
| F7 | 30 | 31 | 1.771970 (improved from 1.365) |
| F8 | 40 | 41 | 9.965293 (improved from 9.598) |

![Best observed output: initial vs after Week 1](w2charts/chart_w2_progression.png)

For F1, F2, F3, and F6, the Week 1 query returned a worse value than the existing best, so the best known output did not change. The GP was now aware that those regions were poor, which should guide it away from them.

---

## Pipeline Improvements

### Improved kernel with wider bounds

The convergence warnings on F2 and F7 in Week 1 were caused by the GP hitting the lower bound of the noise level parameter (1e-5). The fix was to widen all three kernel parameter bounds:

```python
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
)
```

This allows the GP optimiser to search a much wider range of kernel hyperparameter values, reducing the chance of the fitted surface being suboptimal.

### Sobol quasi-random candidate generation

Week 1 used `np.random.uniform` to generate candidate points, which leaves random gaps and clusters across the search space. Week 2 replaced this with a Sobol sequence, which fills the space more evenly:

```python
from scipy.stats.qmc import Sobol

sampler = Sobol(d=dimension, scramble=True, seed=42)
candidate_points = sampler.random(n=candidate_size)
```

Sobol sequences are particularly important for higher-dimensional functions (F7: 6D, F8: 8D) where random sampling frequently misses optimal regions.

### Fixed random seed

```python
np.random.seed(42)
```

This was set before candidate generation in Week 1 was identified as missing, causing non-reproducible queries between runs. With the seed fixed, re-running the notebook produces identical queries.

### Per-function kappa values

Rather than applying a universal kappa of 2.0, each function received a kappa based on its Week 1 outcome:

```python
kappa_config = {
    "function_1": 0.5,   # Narrow peak, exploration wasted - exploit
    "function_2": 1.0,   # Multimodal confirmed - moderate exploit
    "function_3": 1.0,   # Interior optimum - moderate exploit
    "function_4": 1.5,   # Improving - exploit with room to move
    "function_5": 1.5,   # Unimodal peak - keep pushing
    "function_6": 1.0,   # Known best failed - return and exploit
    "function_7": 2.0,   # Improving steadily - keep same balance
    "function_8": 2.0,   # Near ceiling - balance exploit/explore
}
```

![Week 2 kappa values and override status](w2charts/chart_w2_kappa_overrides.png)

### Scaled GP restarts

Higher-dimensional functions have more complex kernel hyperparameter landscapes and benefit from more restarts:

```python
restart_config = {
    "function_1": 10, "function_2": 10, "function_3": 10,
    "function_4": 15, "function_5": 15, "function_6": 15,
    "function_7": 20, "function_8": 20,
}
```

---

## Step-by-Step: What Was Done

### Step 1 - Load original data and define Week 1 observations

The original `.npy` datasets were loaded as in Week 1. The Week 1 query coordinates and returned outputs were hard-coded as dictionaries:

```python
week1_queries = {
    "function_1": np.array([0.000186, 0.014353]),
    "function_2": np.array([0.998531, 0.007036]),
    "function_3": np.array([0.933672, 0.002452, 0.965412]),
    "function_4": np.array([0.417336, 0.402860, 0.336077, 0.476656]),
    "function_5": np.array([0.050115, 0.927701, 0.965034, 0.985561]),
    "function_6": np.array([0.197786, 0.010925, 0.990284, 0.888004, 0.052863]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904]),
}

week1_outputs = {
    "function_1": 1.14e-239,
    "function_2": -0.109704342,
    "function_3": -0.365767178,
    "function_4": -0.869002202,
    "function_5": 3019.659838,
    "function_6": -1.213319062,
    "function_7": 1.771969604,
    "function_8": 9.965293447,
}
```

### Step 2 - Build updated datasets

Each function's dataset was extended with the new observation:

```python
for i in range(1, 9):
    key = f"function_{i}"
    X_updated = np.vstack([original_data[key]["X"], week1_queries[key].reshape(1, -1)])
    Y_updated = np.append(original_data[key]["Y"], week1_outputs[key])
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
```

### Step 3 - EDA on updated data

Statistics were recalculated on the updated datasets. Key checks: whether the new observation changed the best known output, and whether the Week 1 query result was better or worse than expected.

### Step 4 - Run the improved pipeline (Section 7)

The improved `bayesian_optimisation_pipeline_v2` function was called for all 8 functions using per-function kappa, restart, and (for corrected functions) constrained bounds.

### Step 5 - Query validation (Section 8)

A validation report was generated checking each query for:
- All values within [0, 1]
- No suspicious all-zero or all-one patterns
- Positive UCB score

F4 and F6 initially showed negative UCB scores. F3 and F5 showed queries that had drifted far from their known good regions. F1 showed a query that was not close enough to the known peak.

### Step 6 - Override cells for corrected functions

Five functions required constrained override cells run after Section 7. These cells restricted the candidate search to a tight box around the known good region and rewrote the pipeline result. See the Issues section for the full account of what went wrong and how each was resolved.

### Step 7 - Final save

A final save cell was used that bypassed `week2_results` entirely for F6 and read all other functions directly from `week2_results`:

```python
rows = []
for i in range(1, 9):
    key = f"function_{i}"
    if key == "function_6":
        q = f6_final       # directly computed, not from week2_results
    else:
        q = week2_results[key]["formatted_query"]
    rows.append({"Function": f"Function {i}", "Query": q})

pd.DataFrame(rows).to_csv("week2_queries_final.csv", index=False)
```

---

## Week 2 Queries Submitted

| Function | Query submitted | Strategy applied | Override |
|----------|----------------|-----------------|---------|
| F1 | `0.705908-0.823143` | Exploit near [0.731, 0.733] | Constrained box |
| F2 | `0.839078-0.895770` | Return to known best [0.703, 0.927] | None needed |
| F3 | `0.363504-0.758379-0.191085` | Exploit interior [0.493, 0.612, 0.340] | Constrained box |
| F4 | `0.403695-0.397605-0.413333-0.411576` | Continue improving from -0.869 | Constrained box + Y-shift |
| F5 | `0.056181-0.992607-0.973199-0.959866` | Push peak further | Constrained box |
| F6 | `0.729856-0.157383-0.734992-0.704798-0.068448` | Return to best recipe | Manual perturbation |
| F7 | `0.058090-0.303889-0.327991-0.001268-0.275231-0.673181` | Continue improvement | None needed |
| F8 | `0.017074-0.091604-0.305973-0.115845-0.946320-0.608139-0.053440-0.855712` | Balance exploit/explore | None needed |

![Week 2 query proximity to known best input](w2charts/chart_w2_query_proximity.png)

---

## Function-by-Function Analysis

### Function 1

**Current best:** 0.000000 at [0.731, 0.733]
**Raw pipeline query:** (0.431029, 0.814368) — drifted from peak
**Final query submitted:** (0.705908, 0.823143)
**W2 kappa:** 0.5

Despite setting kappa to 0.5 to force exploitation, the raw pipeline query landed at (0.431, 0.814) — dim 1 was 0.30 away from the known peak at 0.731. A constrained override was applied with a search box of +/- 0.10 around [0.731, 0.733]. The corrected query (0.706, 0.823) has dim 1 very close to target.

F1 remains the most difficult function to optimise because its output surface is nearly flat everywhere except for extremely narrow peaks near zero. The GP struggles to distinguish signal from noise, and even with kappa reduced to 0.5, it was still drawn toward uncertain regions rather than the known peak.

**Lesson:** For Week 3, tighten the constrained box further to +/- 0.05 around [0.731, 0.733] to force even finer exploitation.

---

### Function 2

**Current best:** 0.611205 at [0.703, 0.927]
**Raw pipeline query:** (0.839078, 0.895770) — no override needed
**Final query submitted:** (0.839078, 0.895770)
**W2 kappa:** 1.0

The improved kernel and kappa of 1.0 directed the query to (0.839, 0.896), which is close to the known best at [0.703, 0.927]. Dim 2 (0.896) is very close to 0.927. Dim 1 (0.839) is slightly higher than 0.703 but still in the right neighbourhood. The convergence warning from Week 1 did not reappear with the wider kernel bounds.

This is a multimodal function confirmed by Week 1 (exploration of a new region returned -0.110 against a best of 0.611). The GP is now correctly exploiting the known good region.

**Lesson:** No change needed for Week 3. If W2 result improves on 0.611, tighten kappa further.

---

### Function 3

**Current best:** -0.034835 at [0.493, 0.612, 0.340]
**Raw pipeline query:** (0.994571, 0.993175, 0.003964) — boundary corner again
**Final query submitted:** (0.363504, 0.758379, 0.191085)
**W2 kappa:** 1.0

F3 produced a boundary corner query for the second time in a row — dims 1 and 2 near 1.0, dim 3 near 0.0. This is the same pattern as the Week 1 query (0.934, 0.002, 0.965) which returned -0.366, far worse than the best of -0.035. Despite setting kappa to 1.0 and updating the dataset with the poor Week 1 boundary result, the GP continued to be drawn toward uncertain boundary regions.

A constrained override was applied with a search box of +/- 0.15 around the best known interior point [0.493, 0.612, 0.340]. The corrected query (0.364, 0.758, 0.191) has all three dimensions in the interior of the input space, which is a significant improvement over the raw pipeline output.

**Lesson:** F3's GP consistently over-values boundary uncertainty. For Week 3, build the constraint directly into the pipeline loop rather than relying on a post-hoc override cell.

---

### Function 4

**Current best:** -0.869002 at [0.417, 0.403, 0.336, 0.477]
**Raw pipeline query:** (0.385351, 0.417833, 0.397451, 0.428309) — negative UCB -0.5524
**Final query submitted:** (0.403695, 0.397605, 0.413333, 0.411576)
**W2 kappa:** 1.5

F4 produced a negative UCB score for the second consecutive week (-0.5524 in Week 2 vs -0.309567 in Week 1). This is an expected consequence of F4's all-negative output range (-32 to -0.869). The GP's predicted mean is always negative, and adding 2*sigma does not push the score positive.

Two corrections were applied. First, the output was shifted by subtracting the minimum before GP fitting (Y_shifted = Y - Y.min()), giving the GP a positive output scale to work with. Second, the candidate search was constrained to a box of +/- 0.15 around the current best input. The corrected query has all 4 dims within 0.08 of the current best and a positive UCB score.

Week 1 proved that a negative UCB on F4 does not mean a bad query — the Week 1 query improved the output by +3.16 despite a negative UCB. The correction in Week 2 was primarily to silence the warning and produce a query in a demonstrably good neighbourhood.

**Lesson:** Apply Y-shifting directly in the pipeline for F4 in Week 3 using an if-condition inside the loop rather than a post-hoc cell.

---

### Function 5

**Current best:** 3019.659838 at [0.050, 0.928, 0.965, 0.986]
**Raw pipeline query:** (0.431029, 0.814368, 0.806413, 0.053457) — drifted badly
**Final query submitted:** (0.056181, 0.992607, 0.973199, 0.959866)
**W2 kappa:** 1.5

The raw pipeline query for F5 was the most concerning drift in the set. Dim 1 jumped from 0.050 to 0.431, dim 4 collapsed from 0.986 to 0.053. The function's unimodal peak at high values of dims 2, 3, and 4 was being completely abandoned.

A constrained override was applied with bounds:
- Dim 1: [0.00, 0.15] — keep low
- Dims 2, 3, 4: [0.85, 1.00], [0.90, 1.00], [0.90, 1.00] — keep high

The corrected query (0.056, 0.993, 0.973, 0.960) pushes dims 2 and 3 higher than the Week 1 query (0.993 vs 0.928, 0.973 vs 0.965), which produced the +177% gain. This is the strongest query in the submission and has a good chance of producing another large gain.

**Lesson:** F5's peak is well-characterised. For Week 3, reduce kappa to 1.0 to fine-tune near the peak rather than push further into uncertain territory.

---

### Function 6

**Current best:** -0.714265 at [0.728, 0.155, 0.733, 0.694, 0.056]
**Raw pipeline query:** (0.673355, 0.064897, 0.831932, 0.790447, 0.001745) — negative UCB -0.6587
**Final query submitted:** (0.729856, 0.157383, 0.734992, 0.704798, 0.068448)
**W2 kappa:** 1.0

F6 was the most problematic function of Week 2. The raw pipeline produced a query far from the best known recipe with a negative UCB score of -0.6587. Multiple correction attempts were made and all failed due to a persistent issue with how Python manages dictionary references in Jupyter notebooks. The full account is in the next section.

The final fix bypassed the GP entirely. A manual perturbation approach was used:

```python
best_f6 = np.array([0.728186, 0.154693, 0.732552, 0.693997, 0.056401])
np.random.seed(99)
perturbations = np.random.uniform(-0.05, 0.05, size=(1000, 5))
candidates    = np.clip(best_f6 + perturbations, 0.0, 1.0)
dists         = np.linalg.norm(candidates - best_f6, axis=1)
filtered      = candidates[dists > 0.01]
chosen        = filtered[np.argmin(np.linalg.norm(filtered - best_f6, axis=1))]
```

This produced (0.729856, 0.157383, 0.734992, 0.704798, 0.068448) — a distance of only 0.017 from the best known recipe. This is the closest any F6 query has been to the known optimum and represents a significant improvement over the raw pipeline output in both Week 1 and Week 2.

**Lesson:** For Week 3, build the F6 constrained search directly into the pipeline loop. If Week 2 returns a better value than -0.714265, update the best known point and tighten the bounds.

---

### Function 7

**Current best:** 1.771970 at [0.110, 0.394, 0.394, 0.093, 0.386, 0.670]
**Raw pipeline query:** (0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181)
**Final query submitted:** (0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181)
**W2 kappa:** 2.0

No override was needed. The query is in the same neighbourhood as the current best with dim 6 nearly identical (0.673 vs 0.670). The convergence warning from Week 1 (noise_level at lower bound) did not reappear with the wider kernel bounds. The pipeline handled F7 correctly without intervention.

F7 has shown consistent improvement across both weeks (+30% in Week 1 from 1.365 to 1.772). The GP is correctly refining within the known good region.

**Lesson:** Continue as-is for Week 3. If the gain diminishes, consider reducing kappa to fine-tune.

---

### Function 8

**Current best:** 9.965293 at [0.043, 0.092, 0.083, 0.052, 0.808, 0.564, 0.175, 0.420]
**Raw pipeline query:** (0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712)
**Final query submitted:** (0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712)
**W2 kappa:** 2.0

No override was needed. Dims 1 and 2 are close to the current best. Dim 5 pushes higher (0.946 vs 0.808) and dim 8 explores a new direction (0.856 vs 0.420), which is an appropriate balance of exploitation and exploration for an 8D function.

F8 showed modest improvement in Week 1 (+3.8%). The output range (5.6 to 9.97) suggests the function may be approaching a local maximum in the observed region. The Week 2 query maintains the same exploration balance to avoid getting trapped.

**Lesson:** If Week 2 shows another small gain, consider increasing kappa in Week 3 to force broader exploration across the 8D space.

---

## Issues and Challenges

### Issue 1 - Raw pipeline queries drifting from known good regions

Five of the eight raw pipeline queries were directed to poor regions despite the improved kernel, Sobol sampling, and per-function kappa settings.

![Correction effort required per function](w2charts/chart_w2_correction_effort.png)

| Function | Problem | Root cause |
|----------|---------|-----------|
| F1 | Query at (0.431, 0.814) vs peak at [0.731, 0.733] | kappa=0.5 insufficient — uncertainty still dominated mean signal for narrow-peaked function |
| F3 | Boundary corner again (0.995, 0.993, 0.004) | GP continued to value boundary uncertainty over known interior optimum |
| F4 | Negative UCB -0.5524 | All-negative output range prevents GP from producing positive UCB scores |
| F5 | Dim 4 collapsed from 0.986 to 0.053 | GP's uncertainty at unexplored regions dominated the exploitation signal |
| F6 | Query far from best recipe, UCB -0.6587 | Insufficient observations to model the surface; GP exploring rather than exploiting |

The common thread across F1, F3, F5, and F6 is that the UCB exploration term (kappa * sigma) was large enough in uncertain regions to override the exploitation term (mu), sending queries away from known good areas. Constrained candidate search boxes are the correct solution — they prevent the GP from even considering candidates outside the target region.

---

### Issue 2 - The override cell problem: `week2_results` not updating

This was the most time-consuming issue of Week 2. Override cells written to fix F4, F5, and F6 repeatedly failed to produce the expected output in the validation report.

**What was observed:** After running override cells, Section 8 (validation) continued to show the original raw pipeline queries for the affected functions.

**Root cause identified:** When Section 7 runs the optimisation loop:

```python
week2_results = {}
for i in range(1, 9):
    week2_results[key] = result
```

Python creates a new dictionary object and assigns it to the name `week2_results`. If Section 7 was re-run at any point after the override cells had already written their corrections, Python discarded the old dictionary and created a fresh one. The override cells had written to the old object, which was no longer referenced by anything. Section 8 then read from the new empty dictionary populated only by the re-run of Section 7.

**How it was resolved:** After multiple failed attempts using nested key assignment (`week2_results["function_6"]["formatted_query"] = formatted`), whole-dict replacement (`week2_results["function_6"] = {...}`), and diagnostic cells confirming `Changed from old query: False`, the final approach bypassed `week2_results` entirely for F6 and built the submission CSV directly:

```python
for i in range(1, 9):
    key = f"function_{i}"
    q = f6_final if key == "function_6" else week2_results[key]["formatted_query"]
    rows.append({"Function": f"Function {i}", "Query": q})
pd.DataFrame(rows).to_csv("week2_queries_final.csv", index=False)
```

This approach has no dependency on dictionary write operations succeeding and cannot be broken by re-running Section 7.

---

### Issue 3 - F5 and F6 queries swapped in an intermediate run

During one run, the F6 override cell accidentally wrote a 5-dimensional query to `week2_results["function_5"]` instead of `week2_results["function_6"]`. This was caught during validation because F5 showed 5 values when it should have 4, and both F5 and F6 showed identical queries.

The cause was a copy-paste error in the override cell where the key was not updated from `function_5` to `function_6`. After correcting the key and re-running both override cells in the right order, the issue was resolved.

**Lesson:** When copying override cells from one function to another, always check the dictionary key before running.

---

### Issue 4 - GP unable to help F6 even with constrained bounds

Even with a constrained search box of +/- 0.10 around the best known recipe, the GP fitted on F6's data consistently produced queries with negative UCB scores outside the expected region. Investigation revealed a convergence warning with the RBF length scale hitting the upper bound (10000), indicating the GP was fitting an extremely smooth surface — essentially a flat prediction with high uncertainty everywhere. With only 21 observations in 5 dimensions, the data was too sparse for the GP to model the surface usefully.

This led to the decision to bypass the GP entirely for F6 and use a direct manual perturbation of the best known point, placing the query at a distance of 0.017 from the best known recipe without relying on surrogate model predictions.

---

## What to Improve for Week 3

### 1. Build all constraints into the pipeline loop directly

Replace post-hoc override cells with a `bounds_config` dictionary applied inside Section 7:

```python
bounds_config = {
    "function_1": [np.array([0.63, 0.63]), np.array([0.83, 0.83])],
    "function_3": [np.array([0.34, 0.46, 0.04]), np.array([0.64, 0.76, 0.34])],
    "function_5": [np.array([0.00, 0.85, 0.90, 0.90]), np.array([0.15, 1.00, 1.00, 1.00])],
    "function_6": [np.array([0.63, 0.05, 0.63, 0.59, 0.00]), np.array([0.83, 0.26, 0.83, 0.79, 0.16])],
    "function_4": None,  # use Y-shifting instead
    "function_2": None, "function_7": None, "function_8": None,
}
```

Then pass `low_bounds` and `high_bounds` directly into the pipeline function. This removes any dependency on override cells and eliminates the dictionary reference problem entirely.

### 2. Apply Y-shifting for F4 inside the pipeline

```python
if key == "function_4":
    Y_for_gp = Y - Y.min()   # shift so minimum = 0
else:
    Y_for_gp = Y
```

This ensures the GP always has a positive output range for F4 and removes the negative UCB warning without any post-hoc correction.

### 3. Update bounds after seeing Week 2 results

Once Week 2 outputs are received, update the bounds_config to follow the new best point for each function. If F5 returns another large gain, tighten the box around the new best input. If F6 improves on -0.714265, move the box to that new location.

### 4. Reduce kappa for F5

If Week 2 returns another large gain for F5, reduce kappa from 1.5 to 1.0. At that point the function is well-characterised and fine-tuning near the peak is more valuable than broad exploration.

### 5. Consider increasing kappa for F8

If Week 2 shows a similarly small gain for F8 (+3.8% in Week 1), the function may be near a local maximum. Increasing kappa to 3.0 for Week 3 would push the GP to explore further into the 8D space to look for a better region.

### 6. Never re-run Section 7 after override cells

The root cause of the override problem was Section 7 being re-run after corrections had been made. The rule for all future weeks is: Section 7 runs exactly once. All corrections happen after it and the CSV is saved immediately after validation without re-running any earlier sections.

---

## Week 3 Strategy Predictions

| Function | Current best | W2 query direction | W3 strategy | Predicted kappa |
|----------|-------------|-------------------|-------------|-----------------|
| F1 | 0.000000 | Near [0.706, 0.823] | Tighter box around [0.731, 0.733], margin 0.05 | 0.5 |
| F2 | 0.611205 | Near [0.839, 0.896] | Exploit known peak, update if W2 improves | 1.0 |
| F3 | -0.034835 | Interior at [0.364, 0.758, 0.191] | Exploit, update box if W2 improves | 1.0 |
| F4 | -0.869002 | Near [0.404, 0.398, 0.413, 0.412] | Continue improving toward 0 | 1.5 |
| F5 | 3019.659838 | Peak region [0.056, 0.993, 0.973, 0.960] | Fine-tune peak, reduce kappa | 1.0 |
| F6 | -0.714265 | Near best recipe [0.730, 0.157, 0.735, 0.705, 0.068] | Exploit, constrain in pipeline | 0.5 |
| F7 | 1.771970 | Near best [0.058, 0.304, 0.328, 0.001, 0.275, 0.673] | Continue exploiting | 2.0 |
| F8 | 9.965293 | Exploring dims 5 and 8 | Balance, increase kappa if gain small | 2.0-3.0 |

---

## Summary

| Function | W2 query | Override applied | Reason | Risk level |
|----------|---------|-----------------|--------|-----------|
| F1 | 0.705908-0.823143 | Constrained box | GP drifted from narrow peak | Low |
| F2 | 0.839078-0.895770 | None | Pipeline handled correctly | Low |
| F3 | 0.363504-0.758379-0.191085 | Constrained box | Boundary corner third occurrence | Low |
| F4 | 0.403695-0.397605-0.413333-0.411576 | Y-shift + constrained box | Negative UCB on all-negative output | Low |
| F5 | 0.056181-0.992607-0.973199-0.959866 | Constrained box | Drifted from unimodal peak | Very low |
| F6 | 0.729856-0.157383-0.734992-0.704798-0.068448 | Manual perturbation | GP unable to model sparse 5D surface | Medium |
| F7 | 0.058090-0.303889-0.327991-0.001268-0.275231-0.673181 | None | Pipeline handled correctly | Low |
| F8 | 0.017074-0.091604-0.305973-0.115845-0.946320-0.608139-0.053440-0.855712 | None | Pipeline handled correctly | Low |

---

*Reflection written after Week 2 queries were finalised and submitted. All analysis is based on updated datasets incorporating Week 1 results and the query generation and correction process carried out during Week 2.*
