# Bayesian Optimisation Capstone: Week 1 Reflection

> **Competition:** Black-box function optimisation across 8 synthetic functions
> **Method:** Gaussian Process surrogate modelling with UCB acquisition
> **Week 1 objective:** Submit one query point per function based on initial data

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Summary](#dataset-summary)
3. [Step-by-Step: What Was Done](#step-by-step-what-was-done)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Week 1 Queries Submitted](#week-1-queries-submitted)
6. [Week 2 Outputs Received](#week-2-outputs-received)
7. [Function-by-Function Analysis](#function-by-function-analysis)
8. [Issues Encountered](#issues-encountered)
9. [Challenges](#challenges)
10. [What to Improve for Week 2](#what-to-improve-for-week-2)
11. [Week 2 Strategy Predictions](#week-2-strategy-predictions)

---

## Overview

This project involves optimising eight synthetic black-box functions over multiple weeks. Each function takes a multi-dimensional input array and returns a scalar output. The goal is to maximise each function's output using as few queries as possible, which is a classic **sample-efficient optimisation** problem.

The approach used is **Bayesian Optimisation (BO)**, which works by:

1. Fitting a **Gaussian Process (GP)** surrogate model to observed data
2. Using an **acquisition function** to decide where to query next
3. Querying the function at that point and updating the model

In Week 1, each function was queried once, producing one new output per function. The results of those queries form the updated dataset going into Week 2.

---

## Dataset Summary

The initial datasets provided varied in size and dimensionality across the eight functions:

| Function | Dimensions | Initial observations | Description |
|----------|-----------|---------------------|-------------|
| F1 | 2D | 10 | Contamination source detection |
| F2 | 2D | 10 | Noisy ML log-likelihood surface |
| F3 | 3D | 15 | Drug compound side effects |
| F4 | 4D | 30 | Warehouse product placement |
| F5 | 4D | 20 | Chemical process yield |
| F6 | 5D | 20 | Cake recipe scoring |
| F7 | 6D | 30 | ML hyperparameter tuning |
| F8 | 8D | 40 | 8-dimensional black-box |

All inputs were normalised to the range **[0, 1]** per dimension. All tasks are framed as **maximisation** problems, including functions where the real-world objective is minimisation (e.g. F3: side effects), which are transformed by negation so that higher output always means better.

The chart below shows the spread of initial output values across all eight functions, illustrating the highly varied scales and distributions the GP had to work with:

![Initial output distributions per function](charts/chart4_output_distributions.png)

---

## Step-by-Step: What Was Done

### Step 1 - Data loading

All eight datasets were loaded from `.npy` files using `numpy`:

```python
import numpy as np

data = {}
for i in range(1, 9):
    X = np.load(f"function_{i}/initial_inputs.npy")
    Y = np.load(f"function_{i}/initial_outputs.npy")
    data[f"function_{i}"] = {"X": X, "Y": Y}
```

Each function's input matrix `X` and output vector `Y` were stored in a dictionary for easy access throughout the pipeline.

---

### Step 2 - Exploratory Data Analysis (EDA)

For each function, the following statistics were computed and examined:

- Input matrix shape
- Output vector shape
- First 5 input samples (display only, did **not** restrict the model)
- Minimum, maximum, mean, and standard deviation of outputs
- Best current observation (highest output value and its corresponding input)

**Example output for Function 5:**

```
Function 5 Output Statistics
----------------------------------------
Minimum output       : 0.112940
Maximum output       : 1088.859618
Mean output          : 151.271876
Standard deviation   : 245.575981

Best Current Observation
----------------------------------------
Best input point : [0.22418902 0.84648049 0.87948418 0.87851568]
Best output value: 1088.859618
```

A scatter plot was also generated for each function to visualise the spatial distribution of sampled points, colour-coded by output value. For higher-dimensional functions (3D+), the first two dimensions were used as the plot axes.

> **Note on "first 5 entries" misunderstanding:** During EDA, only the first 5 rows were *displayed* using `print(X[:5])`. This was purely for inspection. The GP model was trained on the full dataset (`gp_model.fit(X, Y)`), so the optimisation used all available observations throughout.

---

### Step 3 - Gaussian Process model setup

A GP was fitted separately for each function using scikit-learn's `GaussianProcessRegressor`. The kernel was composed of three components:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

kernel = (
    ConstantKernel(1.0)
    * RBF(length_scale=1.0)
    + WhiteKernel(noise_level=1e-5)
)

gp_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)
```

**Kernel component roles:**

| Component | Role |
|-----------|------|
| `ConstantKernel` | Scales the overall amplitude of the function |
| `RBF` (Radial Basis Function) | Models smooth, continuous variation between nearby points |
| `WhiteKernel` | Accounts for observation noise in the outputs |

**Input scaling:** Before fitting, all inputs were standardised using `StandardScaler` (zero mean, unit variance). This is important because the RBF kernel uses Euclidean distance. Without scaling, dimensions with larger raw values would appear more important to the GP than they actually are.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gp_model.fit(X_scaled, Y)
```

---

### Step 4 - Candidate point generation

Rather than using a gradient-based search, candidate points were sampled uniformly at random from the [0, 1]^d hypercube, where d is the number of dimensions. The candidate size was scaled with dimensionality:

```python
candidate_size = 5000 * dimension

candidate_points = np.random.uniform(
    low=0,
    high=1,
    size=(candidate_size, dimension)
)
```

This random sampling approach covers the search space broadly without requiring derivative information from the black-box function, which is appropriate since the function internals are unknown.

> **Reproducibility note:** `np.random.seed()` was not set before generating candidate points. The GP model used `random_state=42`, but this only fixed the GP's internal optimisation. The candidate sampling remained stochastic, which is why re-running the notebook produced slightly different query coordinates even though UCB scores were nearly identical. This is addressed in Week 2 improvements.

---

### Step 5 - UCB acquisition function

The **Upper Confidence Bound (UCB)** acquisition function was used to score each candidate point:

```
UCB(x) = mu(x) + kappa * sigma(x)
```

Where:
- `mu(x)` is the GP's predicted mean at point x (exploitation term: how good we think this region is)
- `sigma(x)` is the GP's predicted standard deviation at point x (exploration term: how uncertain we are)
- `kappa = 2.0` is the exploration-exploitation trade-off parameter

```python
mean_prediction, std_prediction = gp_model.predict(
    candidate_points_scaled,
    return_std=True
)

ucb_scores = mean_prediction + kappa * std_prediction
```

A **kappa of 2.0** balances exploration and exploitation. A higher kappa pushes the algorithm toward unexplored regions; a lower kappa focuses it near already-observed good values.

The chart below shows how the UCB score for each function was split between the predicted mean (exploitation) and the uncertainty contribution (exploration):

![UCB score decomposition](charts/chart2_ucb_decomposition.png)

---

### Step 6 - Query selection and formatting

The candidate point with the highest UCB score was selected as the query:

```python
best_index = np.argmax(ucb_scores)
best_query = scaler.inverse_transform([candidate_points_scaled[best_index]])[0]

formatted_query = "-".join([f"{x:.6f}" for x in best_query])
```

The query is inverse-transformed back to the original [0, 1] scale before submission. The formatting joins coordinates with hyphens, as required by the competition submission format.

---

### Step 7 - Results export

All queries were saved to a CSV file:

```python
import pandas as pd

query_df = pd.DataFrame({
    "Function": [f"Function {i}" for i in range(1, 9)],
    "Query": [results[f"function_{i}"]["formatted_query"] for i in range(1, 9)]
})

query_df.to_csv("week1_queries.csv", index=False)
```

---

## Pipeline Architecture

```
Initial .npy data
        |
        v
   Load all 8 datasets
        |
        v
   EDA per function
   (statistics, best point, scatter plot)
        |
        v
   StandardScaler (fit on X, transform X)
        |
        v
   GP fit on (X_scaled, Y)
   Kernel: ConstantKernel x RBF + WhiteKernel
   n_restarts_optimizer = 10
        |
        v
   Generate 5000xd random candidates in [0,1]^d
   Transform candidates with fitted scaler
        |
        v
   GP predict (mean, std) on all candidates
        |
        v
   UCB score = mean + 2.0 x std
        |
        v
   Select argmax(UCB)
   Inverse-transform to original scale
        |
        v
   Format and save to week1_queries.csv
```

---

## Week 1 Queries Submitted

| Function | Query submitted | Predicted mean | Uncertainty (sigma) | UCB score |
|----------|----------------|----------------|---------------------|-----------|
| F1 | `0.000186-0.014353` | -0.000328 | 0.003303 | 0.006278 |
| F2 | `0.998531-0.007036` | 0.362024 | 0.251206 | 0.864436 |
| F3 | `0.933672-0.002452-0.965412` | -0.005087 | 0.140970 | 0.276854 |
| F4 | `0.417336-0.402860-0.336077-0.476656` | -1.343167 | 0.516800 | -0.309567 |
| F5 | `0.050115-0.927701-0.965034-0.985561` | 130.289249 | 257.323641 | 644.936531 |
| F6 | `0.197786-0.010925-0.990284-0.888004-0.052863` | -0.669588 | 0.413021 | 0.156454 |
| F7 | `0.110110-0.393658-0.394356-0.092883-0.385807-0.669789` | 1.278565 | 0.185536 | 1.649638 |
| F8 | `0.042700-0.092462-0.083390-0.051299-0.808162-0.563756-0.175217-0.419904` | 9.998760 | 0.293155 | 10.585069 |

---

## Week 2 Outputs Received

These are the function values returned by the competition organisers for the Week 1 query points:

| Function | Week 1 query result | Previous best | Change | Outcome |
|----------|---------------------|---------------|--------|---------|
| F1 | 1.14e-239 | 0.000000 | approx 0 | No gain |
| F2 | -0.109704 | 0.611205 | -0.721 | Declined |
| F3 | -0.365767 | -0.034835 | -0.331 | Declined |
| F4 | -0.869002 | -4.025542 | +3.157 | **Improved** |
| F5 | 3019.659838 | 1088.859618 | +1930.800 | **Improved (+177%)** |
| F6 | -1.213319 | -0.714265 | -0.499 | Declined |
| F7 | 1.771970 | 1.364968 | +0.407 | **Improved (+30%)** |
| F8 | 9.965293 | 9.598482 | +0.367 | **Improved (+3.8%)** |

**5 of 8 functions improved. 2 declined. 1 produced no signal.**

The charts below show the raw output comparison and the percentage change across all functions:

![Week 1 results: initial best vs query output](charts/chart1_results_comparison.png)

![Percentage change in best observed output after Week 1 query](charts/chart3_pct_change.png)

---

## Function-by-Function Analysis

### Function 1

**Initial best:** 0.000000 at [0.731, 0.733]
**Query sent to:** (0.000186, 0.014353) near the (0, 0) corner
**Result received:** 1.14e-239

The GP directed the query to a low-density corner because of high uncertainty there (sigma = 0.003303), not because it predicted good values (mu = -0.000328). The UCB score was barely positive at 0.006278. The returned value is for all practical purposes zero, confirming this region is flat and uninformative.

F1 appears to have extremely sharp, narrow Gaussian-like peaks. The entire observed output range is near zero, with the best value also recorded as 0.000000. The function is likely unimodal with a very tight peak somewhere near [0.73, 0.73] based on initial data.

**Lesson:** For functions with near-zero outputs everywhere, the GP struggles to distinguish signal from noise. Exploration of uncertain regions is wasteful when the peak is already known. Week 2 should query close to [0.73, 0.73] with small perturbations.

---

### Function 2

**Initial best:** 0.611205 at [0.703, 0.927]
**Query sent to:** (0.998531, 0.007036) opposite corner
**Result received:** -0.109704

The GP, affected by convergence warnings during fitting, sent the query to a new region with moderate uncertainty. This turned out to be a poor area. The initial best at [0.70, 0.93] is a genuine local optimum.

F2 is confirmed as **noisy and multimodal**. The UCB exploration of a new region failed, suggesting the function has sharp peaks rather than a broad plateau. The convergence warning (constant_value and noise_level hitting lower bounds) means the GP surface was not optimally fitted.

**Lesson:** Fix kernel bounds for F2 in Week 2. Exploit the known peak at [0.70, 0.93] more directly rather than exploring new regions.

---

### Function 3

**Initial best:** -0.034835 at [0.493, 0.612, 0.340]
**Query sent to:** (0.933672, 0.002452, 0.965412) boundary corner
**Result received:** -0.365767

The GP sent the query to a boundary region with high uncertainty. The returned value is much worse than the current best. F3 outputs are all negative (side effects), and the best observed value is at moderate input values, not at the boundary.

**Lesson:** Boundary exploration is risky for F3. The optimum appears to be in the interior of the input space around [0.49, 0.61, 0.34]. Week 2 should exploit this neighbourhood.

---

### Function 4

**Initial best:** -4.025542 at [0.578, 0.429, 0.426, 0.249]
**Query sent to:** (0.417336, 0.402860, 0.336077, 0.476656)
**Result received:** -0.869002

Despite the Week 1 UCB score being **negative (-0.309567)**, the only function where this occurred, the actual returned value was a significant improvement over the previous best. This is a case where the GP's uncertainty estimate led to a useful query despite the pessimistic mean prediction.

F4's all-negative outputs (range -32 to -4) made it difficult for the GP to identify promising regions, but the query landed in a genuinely better area.

**Lesson:** Negative UCB scores do not always mean bad queries. The GP was uncertain, not confident of failure. For Week 2, query close to the Week 1 result input to continue improving.

---

### Function 5

**Initial best:** 1088.859618 at [0.224, 0.846, 0.879, 0.879]
**Query sent to:** (0.050115, 0.927701, 0.965034, 0.985561)
**Result received:** 3019.659838

The strongest result of Week 1. The query was directed to a region with both high predicted mean (130.29) and very high uncertainty (sigma = 257.32), producing a UCB score of 644.94. The returned value nearly tripled the previous best.

F5 is clearly **unimodal** with a strong, broad peak in the region of high dimensions 2 to 4. The function responds well to increasing values in dimensions 2, 3, and 4.

**Lesson:** Continue exploiting this region in Week 2. The peak may not yet be reached. Pushing dimension values closer to 1.0 in dimensions 2 to 4 while adjusting dimension 1 may yield further improvement.

---

### Function 6

**Initial best:** -0.714265 at [0.728, 0.155, 0.733, 0.694, 0.056]
**Query sent to:** (0.197786, 0.010925, 0.990284, 0.888004, 0.052863)
**Result received:** -1.213319

The query moved away from the best-known recipe combination and found a worse score. The GP predicted this region would be better (mu = -0.669588), which turned out to be incorrect. The function surface is more complex than the GP modelled.

**Lesson:** Return to the neighbourhood of the known best [0.728, 0.155, 0.733, 0.694, 0.056] in Week 2. Fine-tune around those coordinates rather than exploring new regions.

---

### Function 7

**Initial best:** 1.364968 at [0.058, 0.492, 0.247, 0.218, 0.420, 0.731]
**Query sent to:** (0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789)
**Result received:** 1.771970

A solid improvement of +0.407 (+30%). The query was directed to a nearby region of the best-known point, and the GP's prediction proved correct. F7 has a clear, exploitable region.

There was a convergence warning (noise_level at lower bound), but this did not appear to significantly harm the query quality.

**Lesson:** Continue exploiting this region. Fix the convergence warning bounds. The best is now 1.771970 and Week 2 should query close to its input coordinates.

---

### Function 8

**Initial best:** 9.598482 at [0.056, 0.066, 0.023, 0.039, 0.404, 0.801, 0.488, 0.893]
**Query sent to:** (0.042700, 0.092462, 0.083390, 0.051299, 0.808162, 0.563756, 0.175217, 0.419904)
**Result received:** 9.965293

A modest improvement of +0.367 (+3.8%). The output range for F8 is 5.6 to 9.6 in initial data, so 9.965 is approaching the observed ceiling. The GP correctly identified a region slightly better than the current best.

The high dimensionality (8D) makes F8 the hardest function to optimise with limited queries. Each query covers a tiny fraction of the search space.

**Lesson:** The function may be near a local maximum. Week 2 should combine fine-grained exploitation near the current best with some structured uncertainty reduction in underexplored dimensions.

---

## Issues Encountered

### Issue 1 - Non-reproducible queries

**What happened:** Re-running the notebook with identical code produced different query coordinates.

**Root cause:** `np.random.seed()` was not set before generating the candidate point pool:
```python
candidate_points = np.random.uniform(low=0, high=1, size=(candidate_size, dimension))
```
The GP's `random_state=42` only fixed the GP's internal kernel optimisation, not the candidate sampling. Different random candidates produced a different point winning the UCB competition, even though UCB scores were nearly identical.

**Impact:** The submitted queries and re-run queries had similar UCB scores but different coordinates. Both are valid; the difference is cosmetic, not algorithmic.

**Fix for Week 2:**
```python
np.random.seed(42)
candidate_points = np.random.uniform(low=0, high=1, size=(candidate_size, dimension))
```

---

### Issue 2 - Convergence warnings on Functions 2 and 7

**What happened:** scikit-learn raised `ConvergenceWarning` during GP fitting for F2 and F7:

```
ConvergenceWarning: The optimal value found for dimension 0 of parameter
k1__k1__constant_value is close to the specified lower bound 1e-05.
Decreasing the bound and calling fit again may find a better value.

ConvergenceWarning: The optimal value found for dimension 0 of parameter
k2__noise_level is close to the specified lower bound 1e-05.
```

**Root cause:** The kernel parameters hit their search boundary during log-marginal-likelihood optimisation. The GP wanted to set the noise level lower than `1e-5` (suggesting the function has very little noise), but the bound prevented it.

**Impact:** The fitted GP surface for F2 and F7 was slightly suboptimal. The surrogate model's uncertainty estimates were less accurate, leading to queries in suboptimal regions.

**Fix for Week 2:**
```python
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
)
```

---

### Issue 3 - F4 negative UCB score

**What happened:** Function 4 produced a UCB score of -0.309567, the only negative UCB score across all functions.

**Root cause:** F4's outputs are all negative (range: -32 to -4). The GP's predicted mean at the selected candidate was -1.343167, and even adding 2*sigma did not push the score positive. The GP had no candidate it was truly optimistic about.

**Impact:** Counterintuitively, the query still returned a much better value (-0.869 vs previous best of -4.03). This highlights that a negative UCB score for an all-negative function is not necessarily a bad query. The GP found the least-bad region available.

**Fix for Week 2:** Normalise outputs by subtracting the minimum before fitting, or apply a log transform where appropriate, to help the GP better distinguish between "bad" and "very bad" regions.

---

### Issue 4 - F1 query sent to a dead zone

**What happened:** The F1 query went to the (0, 0) corner and returned a value of essentially zero (1.14e-239).

**Root cause:** F1 has extremely narrow peaks and most of the input space returns near-zero values. The GP's uncertainty was highest at unexplored corners, so UCB (which rewards uncertainty) sent the query there. The predicted mean was negative (-0.000328), but the uncertainty term dominated.

**Impact:** Wasted query. No useful information was gained and the best value was not improved.

**Fix for Week 2:** Reduce kappa for F1 specifically to bias toward exploitation (querying near the known best at [0.73, 0.73]) rather than exploration.

---

## Challenges

### Challenge 1 - Limited data per function

With only 10 to 40 initial observations across 2 to 8 dimensions, the GP had very limited information to build an accurate surrogate surface. In high-dimensional spaces (F8: 8D with 40 points), the data is extremely sparse relative to the volume of the search space.

### Challenge 2 - Heterogeneous function scales

The eight functions operate on completely different output scales, from near-zero (F1) to thousands (F5) and negative ranges (F3, F4, F6). A single pipeline with fixed hyperparameters (same kappa, same kernel bounds) cannot be equally optimal for all functions. The GP for F5 was driven largely by uncertainty (sigma = 257 vs mu = 130), while F8 was more mean-driven.

### Challenge 3 - Unknown function shape

Without knowing whether each function is unimodal or multimodal, it is hard to decide the right exploration-exploitation balance. F2 appeared multimodal (exploring a new region found a worse value), while F5 appears unimodal (pushing further in the same direction tripled the output).

### Challenge 4 - One query per week per function

With only one query per function per week, every decision carries high weight. There is no opportunity to test a region cheaply before committing. Each query is both the experiment and the result.

### Challenge 5 - Random candidate sampling coverage

Using random uniform sampling over the input space means there is no guarantee that the true optimal region is well-represented in the candidate pool. For high-dimensional functions, the density of candidates near any given point decreases rapidly with dimension.

---

## What to Improve for Week 2

### 1. Fix reproducibility

```python
np.random.seed(42)  # Set before candidate generation
```

### 2. Widen kernel bounds for all functions

Prevent convergence warnings and allow the GP more flexibility:

```python
kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
)
```

### 3. Update datasets with Week 1 results

Before running Week 2 optimisation, stack the new observations into the dataset:

```python
X_updated = np.vstack([X_original, week1_query])
Y_updated = np.append(Y_original, week1_output)
```

### 4. Use function-specific kappa values

Rather than a universal kappa = 2.0, tune the exploration-exploitation balance per function:

| Function | Situation | Recommended kappa |
|----------|-----------|-------------------|
| F1 | Known peak, exploration wasted | 0.5 (exploit) |
| F2 | Confirmed multimodal, peak known | 1.0 (moderate exploit) |
| F3 | Interior optimum, boundary failed | 1.0 (exploit) |
| F4 | Improving, keep exploiting | 1.5 |
| F5 | Strong unimodal peak, keep pushing | 1.5 |
| F6 | Known best, exploration failed | 1.0 (exploit) |
| F7 | Improving steadily | 2.0 (keep same) |
| F8 | Near ceiling, balance needed | 2.0 |

The chart below shows the recommended kappa values for Week 2 relative to the Week 1 universal value of 2.0:

![Week 2 recommended kappa values per function](charts/chart5_kappa_recommendations.png)

### 5. Use quasi-random (Sobol) sampling for candidates

Replace uniform random candidates with a Sobol sequence for better space coverage:

```python
from scipy.stats.qmc import Sobol

sampler = Sobol(d=dimension, scramble=True, seed=42)
candidate_points = sampler.random(n=candidate_size)
```

Sobol sequences fill the space more evenly than pure random sampling, reducing the chance of missing the optimal region.

### 6. Increase restarts for complex functions

For higher-dimensional functions (F7, F8), increase `n_restarts_optimizer` to reduce the chance of the GP converging to a poor local optimum:

```python
gp_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,  # Up from 10
    random_state=42
)
```

---

## Week 2 Strategy Predictions

Based on the Week 1 results, the following strategies are predicted for each function:

### F1 - Exploit the known peak

The query should target the neighbourhood of [0.731, 0.733] with small perturbations. The function has narrow peaks, making wide exploration wasteful. Reduce kappa to 0.5.

**Predicted query region:** near [0.73, 0.73]

---

### F2 - Return to best known region

The exploration of (0.998, 0.007) failed. The best value at (0.703, 0.927) = 0.611 remains the best. Query near this point with moderate uncertainty. Fix convergence warning bounds.

**Predicted query region:** near [0.70, 0.93]

---

### F3 - Exploit interior region

The boundary exploration failed. Query near the best-known interior point [0.493, 0.612, 0.340].

**Predicted query region:** near [0.49, 0.61, 0.34]

---

### F4 - Continue improving

The Week 1 query improved by +3.16 despite a negative UCB score. The new best is at the Week 1 query input. Query close to that point to continue climbing.

**Predicted query region:** near [0.417, 0.403, 0.336, 0.477]

---

### F5 - Keep pushing the peak

The strongest gain (+177%). F5 is unimodal with the peak in the high-value region of dimensions 2 to 4. Continue pushing toward [x1 low, x2 near 1.0, x3 near 1.0, x4 near 1.0].

**Predicted query region:** near [0.05, 0.93, 0.97, 0.99] or further toward [0.0, 1.0, 1.0, 1.0]

---

### F6 - Exploit known best recipe

The query moved away from the best and found worse values. Return to [0.728, 0.155, 0.733, 0.694, 0.056] and refine around it.

**Predicted query region:** near [0.73, 0.15, 0.73, 0.69, 0.06]

---

### F7 - Continue steady improvement

The +30% gain confirms the region near [0.06 to 0.11, 0.39 to 0.49, 0.25 to 0.39, 0.09 to 0.22, 0.39 to 0.42, 0.67 to 0.73] is genuinely good. Fix convergence bounds and keep exploiting.

**Predicted query region:** near [0.110, 0.394, 0.394, 0.093, 0.386, 0.670]

---

### F8 - Balance exploitation with targeted exploration

The gain was modest (+3.8%), suggesting the function may be near a local maximum. Refine near the new best but allow some uncertainty exploration in the less-constrained dimensions (5 to 8).

**Predicted query region:** near [0.043, 0.092, 0.083, 0.051, 0.808, 0.564, 0.175, 0.420]

---

## Summary Table

| Function | W1 outcome | Root cause of result | W2 strategy |
|----------|-----------|----------------------|-------------|
| F1 | No gain | Narrow peak, query in dead zone | Exploit [0.73, 0.73] |
| F2 | Declined | Multimodal, exploration hit bad region | Exploit [0.70, 0.93] |
| F3 | Declined | Boundary exploration failed | Exploit [0.49, 0.61, 0.34] |
| F4 | Improved | GP uncertainty led to better region | Continue exploiting |
| F5 | +177% | Unimodal, query near true peak | Keep pushing peak direction |
| F6 | Declined | GP mispredicted, moved away from best | Return to best known |
| F7 | +30% | Query near best, GP prediction correct | Continue exploiting |
| F8 | +3.8% | Near local max, modest gain | Refine + targeted exploration |

---

*Reflection written after Week 1 query results were received. All analysis is based on the initial datasets provided by the competition organisers and the single query-response cycle completed in Week 1.*
