# Bayesian Optimisation Capstone: Week 2 Strategy and Code

> **Building on:** Week 1 queries and the outputs received from the competition organisers
> **Week 2 objective:** Use updated datasets (original + Week 1 result) to generate improved query points

---

## Table of Contents

1. [What Changes in Week 2](#what-changes-in-week-2)
2. [Updated Datasets](#updated-datasets)
3. [Key Improvements Over Week 1](#key-improvements-over-week-1)
4. [Step-by-Step Code](#step-by-step-code)
   - [Step 1 - Load original data](#step-1---load-original-data)
   - [Step 2 - Define Week 1 queries and results](#step-2---define-week-1-queries-and-results)
   - [Step 3 - Build updated datasets](#step-3---build-updated-datasets)
   - [Step 4 - EDA on updated data](#step-4---eda-on-updated-data)
   - [Step 5 - Improved GP pipeline](#step-5---improved-gp-pipeline)
   - [Step 6 - Function-specific kappa values](#step-6---function-specific-kappa-values)
   - [Step 7 - Run optimisation for all functions](#step-7---run-optimisation-for-all-functions)
   - [Step 8 - Inspect and validate queries](#step-8---inspect-and-validate-queries)
   - [Step 9 - Save Week 2 queries](#step-9---save-week-2-queries)
5. [Per-Function Strategy](#per-function-strategy)
6. [Full Notebook Code](#full-notebook-code)

---

## What Changes in Week 2

Week 2 is not a fresh start. It is the next iteration of the same Bayesian Optimisation loop. The key difference is that you now have **one additional observation per function** (the Week 1 query result), which you must incorporate before running the GP.

The loop is:

```
Week 1 data
    +
Week 1 query result
    =
Updated dataset --> GP refit --> UCB --> Week 2 query
```

Every improvement made in Week 2 builds directly on what was learned in Week 1. Functions where Week 1 improved (F4, F5, F7, F8) should be exploited further. Functions where Week 1 declined (F2, F3, F6) or produced no signal (F1) need a redirected strategy.

---

## Updated Datasets

After stacking the Week 1 query and its returned output, each function's dataset grows by one row:

![Dataset size before and after Week 1 update](w2charts/chart_w2_dataset_growth.png)

| Function | Original obs | After Week 1 | New best known |
|----------|-------------|--------------|----------------|
| F1 | 10 | 11 | 0.000000 (unchanged) |
| F2 | 10 | 11 | 0.611205 (unchanged) |
| F3 | 15 | 16 | -0.034835 (unchanged) |
| F4 | 30 | 31 | **-0.869002** (improved) |
| F5 | 20 | 21 | **3019.659838** (improved) |
| F6 | 20 | 21 | -0.714265 (unchanged) |
| F7 | 30 | 31 | **1.771970** (improved) |
| F8 | 40 | 41 | **9.965293** (improved) |

---

## Key Improvements Over Week 1

The following changes are applied in Week 2 to address the issues identified in the Week 1 reflection:

| Issue | Week 1 problem | Week 2 fix |
|-------|---------------|------------|
| Reproducibility | No random seed on candidates | `np.random.seed(42)` before sampling |
| Convergence warnings | Kernel bounds too tight | Widened bounds on all three kernel components |
| Uniform kappa | kappa = 2.0 for all functions | Per-function kappa based on Week 1 outcome |
| Candidate coverage | Pure random uniform sampling | Sobol quasi-random sequences |
| Dataset stale | Only original data used | Original + Week 1 result stacked per function |
| GP restarts | 10 restarts for all | 20 restarts for high-dimensional functions |

---

## Step-by-Step Code

### Step 1 - Load original data

Load the original `.npy` files exactly as in Week 1. These are the baseline observations:

```python
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load original datasets
original_data = {}

for i in range(1, 9):
    X = np.load(f"function_{i}/initial_inputs.npy")
    Y = np.load(f"function_{i}/initial_outputs.npy")
    original_data[f"function_{i}"] = {"X": X, "Y": Y}
    print(f"Function {i} | X: {X.shape} | Y: {Y.shape}")
```

---

### Step 2 - Define Week 1 queries and results

Hard-code the queries that were submitted and the outputs that were returned. These are the new observations to add:

```python
# Week 1 queries (the points that were submitted)
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

# Week 1 outputs (the values returned by the competition organisers)
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

---

### Step 3 - Build updated datasets

Stack the Week 1 observation onto each function's original data:

```python
updated_data = {}

for i in range(1, 9):
    key = f"function_{i}"

    X_original = original_data[key]["X"]
    Y_original = original_data[key]["Y"]

    # New observation from Week 1
    X_new = week1_queries[key].reshape(1, -1)
    Y_new = np.array([week1_outputs[key]])

    # Stack into updated dataset
    X_updated = np.vstack([X_original, X_new])
    Y_updated = np.append(Y_original, Y_new)

    updated_data[key] = {"X": X_updated, "Y": Y_updated}

    print(f"Function {i} | Updated X: {X_updated.shape} | Updated Y: {Y_updated.shape}")
```

**Expected output:**
```
Function 1 | Updated X: (11, 2)  | Updated Y: (11,)
Function 2 | Updated X: (11, 2)  | Updated Y: (11,)
Function 3 | Updated X: (16, 3)  | Updated Y: (16,)
Function 4 | Updated X: (31, 4)  | Updated Y: (31,)
Function 5 | Updated X: (21, 4)  | Updated Y: (21,)
Function 6 | Updated X: (21, 5)  | Updated Y: (21,)
Function 7 | Updated X: (31, 6)  | Updated Y: (31,)
Function 8 | Updated X: (41, 8)  | Updated Y: (41,)
```

---

### Step 4 - EDA on updated data

Before running the GP, inspect the updated datasets to understand what has changed since Week 1:

```python
def run_eda(data, function_name):

    X = data["X"]
    Y = data["Y"]

    print(f"\n{'='*50}")
    print(f"{function_name.upper()} - Updated Dataset")
    print(f"{'='*50}")
    print(f"Shape         : X={X.shape}, Y={Y.shape}")
    print(f"Output range  : [{Y.min():.6f}, {Y.max():.6f}]")
    print(f"Output mean   : {Y.mean():.6f}")
    print(f"Output std    : {Y.std():.6f}")

    best_idx = np.argmax(Y)
    print(f"Best output   : {Y[best_idx]:.6f}")
    print(f"Best input    : {X[best_idx]}")
    print(f"Last added    : Y={Y[-1]:.6f} at X={X[-1]}")

for i in range(1, 9):
    run_eda(updated_data[f"function_{i}"], f"function_{i}")
```

You should also visualise the updated sample distribution for 2D functions:

```python
def plot_updated_samples(data, function_name, week1_query, week1_output):

    X = data["X"]
    Y = data["Y"]

    if X.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Original points
    sc = ax.scatter(X[:-1, 0], X[:-1, 1], c=Y[:-1],
                    cmap='viridis', s=100, edgecolor='black',
                    linewidth=0.5, label='Original + Week 1 data', zorder=2)

    # Week 1 query point (highlighted)
    ax.scatter(week1_query[0], week1_query[1],
               c='red', s=200, marker='*', zorder=3,
               label=f'Week 1 query (Y={week1_output:.4f})')

    plt.colorbar(sc, ax=ax, label='Output value')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(f'{function_name} - Updated Sample Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot 2D functions
for i in [1, 2]:
    key = f"function_{i}"
    plot_updated_samples(
        updated_data[key],
        key,
        week1_queries[key],
        week1_outputs[key]
    )
```

---

### Step 5 - Improved GP pipeline

The core pipeline is updated with three improvements over Week 1:
- Wider kernel bounds to prevent convergence warnings
- Sobol sequences for better candidate coverage
- Fixed random seed for reproducibility

```python
def bayesian_optimisation_pipeline_v2(
    X,
    Y,
    dimension,
    kappa=2.0,
    candidate_size=None,
    n_restarts=10,
    random_seed=42
):
    """
    Improved Bayesian Optimisation pipeline for Week 2.

    Changes from Week 1:
    - Wider kernel parameter bounds (prevents ConvergenceWarning)
    - Sobol sequence for candidate generation (better space coverage)
    - Fixed random seed on candidate sampling (reproducibility)
    - Per-function kappa (passed as argument)
    - Increased restarts for high-dimensional functions
    """

    if candidate_size is None:
        candidate_size = 5000 * dimension

    # SCALE INPUTS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # IMPROVED KERNEL with wider bounds
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )

    # TRAIN GP MODEL
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        random_state=42
    )
    gp_model.fit(X_scaled, Y)

    # SOBOL CANDIDATE GENERATION (reproducible, better coverage)
    sampler = Sobol(d=dimension, scramble=True, seed=random_seed)
    candidate_points = sampler.random(n=candidate_size)

    candidate_points_scaled = scaler.transform(candidate_points)

    # GP PREDICTIONS
    mean_prediction, std_prediction = gp_model.predict(
        candidate_points_scaled,
        return_std=True
    )

    # UCB ACQUISITION
    ucb_scores = mean_prediction + kappa * std_prediction

    # SELECT BEST QUERY
    best_index = np.argmax(ucb_scores)
    best_query_scaled = candidate_points_scaled[best_index]
    best_query = scaler.inverse_transform([best_query_scaled])[0]

    # Clip to [0, 1] for safety
    best_query = np.clip(best_query, 0.0, 1.0)

    formatted_query = "-".join([f"{x:.6f}" for x in best_query])

    return {
        "query": best_query,
        "formatted_query": formatted_query,
        "predicted_mean": mean_prediction[best_index],
        "uncertainty": std_prediction[best_index],
        "ucb_score": ucb_scores[best_index],
        "kernel_fitted": str(gp_model.kernel_),
    }
```

> **Why Sobol over random uniform?**
> Random uniform sampling leaves gaps and clusters by chance. Sobol sequences are designed to fill the space evenly, so every region of the [0,1]^d hypercube gets proportional coverage. This matters most for F7 (6D) and F8 (8D) where random sampling is very likely to miss the optimal region entirely.

> **Why clip to [0, 1]?**
> Inverse-transforming the scaled candidate back to original space can occasionally produce values slightly outside [0, 1] due to floating point arithmetic. Clipping prevents invalid queries from being submitted.

---

### Step 6 - Function-specific kappa values

Based on Week 1 outcomes, each function gets its own kappa value:

```python
# Per-function kappa based on Week 1 analysis
kappa_config = {
    "function_1": 0.5,   # Query went to dead zone - exploit known peak at [0.73, 0.73]
    "function_2": 1.0,   # Exploration failed - exploit known best at [0.70, 0.93]
    "function_3": 1.0,   # Boundary failed - exploit interior at [0.49, 0.61, 0.34]
    "function_4": 1.5,   # Improved - continue exploiting with moderate exploration
    "function_5": 1.5,   # Strong gain - keep pushing the unimodal peak
    "function_6": 1.0,   # Exploration failed - return to best known recipe
    "function_7": 2.0,   # Improved steadily - keep same balance
    "function_8": 2.0,   # Near ceiling - balance exploitation with exploration
}

# Restart config: more restarts for higher-dimensional functions
restart_config = {
    "function_1": 10,
    "function_2": 10,
    "function_3": 10,
    "function_4": 15,
    "function_5": 15,
    "function_6": 15,
    "function_7": 20,
    "function_8": 20,
}
```

**Rationale for each kappa choice:**

![kappa values Week 1 vs Week 2](w2charts/chart_w2_kappa.png)

| Function | kappa | Reasoning |
|----------|-------|-----------|
| F1 | 0.5 | Peaked function, exploration wasted a query. Force exploitation near [0.73, 0.73]. |
| F2 | 1.0 | Confirmed multimodal. Exploring a new region failed. Stay near the known best. |
| F3 | 1.0 | Interior optimum, boundary failed. Moderate exploitation toward [0.49, 0.61, 0.34]. |
| F4 | 1.5 | Improved. The GP found a better area. Keep exploiting with a little room to explore. |
| F5 | 1.5 | Unimodal, strong gain. Continue pushing toward peak while allowing small moves. |
| F6 | 1.0 | Exploration moved away from best and found worse values. Return and exploit. |
| F7 | 2.0 | Steady +30% gain. The same kappa worked well. No reason to change. |
| F8 | 2.0 | Modest gain near ceiling. Keep balance in case a better region exists. |

---

### Step 7 - Run optimisation for all functions

The chart below summarises the Week 2 strategy assigned to each function based on Week 1 outcomes:

![Week 2 strategy per function](w2charts/chart_w2_strategy.png)

```python
week2_results = {}

for i in range(1, 9):
    key = f"function_{i}"

    print(f"\n{'='*60}")
    print(f"FUNCTION {i}")
    print(f"{'='*60}")

    X = updated_data[key]["X"]
    Y = updated_data[key]["Y"]
    dimension = X.shape[1]
    kappa = kappa_config[key]
    restarts = restart_config[key]

    print(f"Dataset size  : {X.shape[0]} observations")
    print(f"Dimensions    : {dimension}D")
    print(f"kappa         : {kappa}")
    print(f"Current best  : {Y.max():.6f}")

    result = bayesian_optimisation_pipeline_v2(
        X=X,
        Y=Y,
        dimension=dimension,
        kappa=kappa,
        n_restarts=restarts,
        random_seed=42
    )

    week2_results[key] = result

    print(f"\nFormatted query  : {result['formatted_query']}")
    print(f"Predicted mean   : {result['predicted_mean']:.6f}")
    print(f"Uncertainty      : {result['uncertainty']:.6f}")
    print(f"UCB score        : {result['ucb_score']:.6f}")
    print(f"Fitted kernel    : {result['kernel_fitted']}")
```

---

### Step 8 - Inspect and validate queries

Before saving, run a validation check on every query to catch obvious problems:

```python
def validate_query(query, function_name, Y_current):
    """
    Run sanity checks on a generated query point.
    Returns a list of warnings if anything looks wrong.
    """
    warnings_list = []

    # Check 1: all values in [0, 1]
    if np.any(query < 0) or np.any(query > 1):
        warnings_list.append("OUT OF RANGE: query contains values outside [0, 1]")

    # Check 2: not all zeros or all ones
    if np.all(query < 0.01):
        warnings_list.append("SUSPICIOUS: all query values near 0 - may be a dead zone")
    if np.all(query > 0.99):
        warnings_list.append("SUSPICIOUS: all query values near 1 - check if valid")

    # Check 3: UCB score positive
    ucb = week2_results[function_name]["ucb_score"]
    if ucb < 0:
        warnings_list.append(f"NEGATIVE UCB: score = {ucb:.4f} - GP pessimistic about this region")

    return warnings_list


print("\nQUERY VALIDATION REPORT")
print("="*60)

all_clear = True
for i in range(1, 9):
    key = f"function_{i}"
    query = week2_results[key]["query"]
    Y = updated_data[key]["Y"]
    issues = validate_query(query, key, Y)

    status = "OK" if not issues else "WARNINGS"
    print(f"\nFunction {i} [{status}]")
    print(f"  Query : {week2_results[key]['formatted_query']}")

    if issues:
        all_clear = False
        for w in issues:
            print(f"  !! {w}")

if all_clear:
    print("\nAll queries passed validation.")
else:
    print("\nReview warnings before submitting.")
```

---

### Step 9 - Save Week 2 queries

```python
function_names = []
formatted_queries = []
predicted_means = []
uncertainties = []
ucb_scores = []
current_bests = []

for i in range(1, 9):
    key = f"function_{i}"
    result = week2_results[key]
    Y = updated_data[key]["Y"]

    function_names.append(f"Function {i}")
    formatted_queries.append(result["formatted_query"])
    predicted_means.append(round(result["predicted_mean"], 6))
    uncertainties.append(round(result["uncertainty"], 6))
    ucb_scores.append(round(result["ucb_score"], 6))
    current_bests.append(round(float(Y.max()), 6))

query_df = pd.DataFrame({
    "Function"       : function_names,
    "Query"          : formatted_queries,
    "Predicted_mean" : predicted_means,
    "Uncertainty"    : uncertainties,
    "UCB_score"      : ucb_scores,
    "Current_best"   : current_bests,
})

print(query_df.to_string(index=False))

query_df[["Function", "Query"]].to_csv("week2_queries.csv", index=False)
print("\nweek2_queries.csv saved.")

query_df.to_csv("week2_queries_full.csv", index=False)
print("week2_queries_full.csv saved (includes GP diagnostics).")
```

---

## Per-Function Strategy

The chart below shows the best known output value for each function going into Week 2, which sets the baseline each query needs to beat:

![Current best values entering Week 2](w2charts/chart_w2_current_best.png)

### Function 1

**Situation:** Week 1 query returned essentially zero. The known best remains 0.000000 at [0.731, 0.733]. The function has extremely narrow peaks and most of the space is flat.

**Strategy:** Exploit the known peak. With kappa = 0.5, the GP will focus on refining near the best-known point rather than exploring unknown corners.

**What to expect:** A query close to [0.73, 0.73]. The GP should now recognise the peak region has higher mean than the surrounding flat space.

**Watch for:** If the query is still sent far from [0.73, 0.73], consider manually constraining the search to a small neighbourhood around the best point.

```python
# Optional manual constraint for F1 if GP still explores
# Restrict candidate points to within 0.1 of the known best
best_f1 = np.array([0.731, 0.733])
candidates_f1 = np.random.uniform(
    low=np.clip(best_f1 - 0.1, 0, 1),
    high=np.clip(best_f1 + 0.1, 0, 1),
    size=(10000, 2)
)
```

---

### Function 2

**Situation:** Exploration of (0.998, 0.007) returned -0.109704. The best remains 0.611205 at [0.703, 0.927]. F2 is confirmed multimodal and noisy. The convergence warning in Week 1 may have caused the GP to misestimate the surface.

**Strategy:** Exploit the known best region with kappa = 1.0. The wider kernel bounds applied in Week 2 should produce a better-fitted GP, directing the query closer to [0.70, 0.93].

**What to expect:** A query near (0.70, 0.93) or in the high-output region of the top-right area of the input space.

**Watch for:** If the convergence warning reappears even with wider bounds, the function may genuinely have very low noise and the GP is struggling with the near-deterministic surface.

---

### Function 3

**Situation:** Boundary query returned -0.365767 against a best of -0.034835 at [0.493, 0.612, 0.340]. The function's best region is clearly in the interior, not at the extremes.

**Strategy:** Exploit the interior with kappa = 1.0. The GP should now have a clearer picture of the surface with the boundary confirmed as poor.

**What to expect:** A query somewhere near [0.49, 0.61, 0.34] or a small perturbation of it.

**Watch for:** The function outputs are all negative. If the GP generates a query with a positive predicted mean, that is a sign the surrogate is not well-fitted and should be examined.

---

### Function 4

**Situation:** The best improved from -4.025542 to -0.869002 after Week 1. This is the function where the negative UCB score (-0.309567) still led to a useful query. With kappa = 1.5, the GP can continue exploiting while leaving room to find even better nearby regions.

**Strategy:** Continue exploiting the neighbourhood of the Week 1 query input [0.417, 0.403, 0.336, 0.477]. The GP now has a much better picture of the landscape.

**What to expect:** A query close to or slightly beyond the Week 1 query point, pushing the output further toward zero.

**Watch for:** Normalising outputs (shifting by the minimum) before fitting may help the GP distinguish better between regions in Week 2, since all outputs are still negative.

```python
# Optional output normalisation for F4 (shift so minimum = 0)
Y_f4 = updated_data["function_4"]["Y"]
Y_f4_shifted = Y_f4 - Y_f4.min()
# Use Y_f4_shifted in the GP, then interpret results in original scale
```

---

### Function 5

**Situation:** The strongest Week 1 result. Output went from 1088.86 to 3019.66, a gain of +177%. F5 is clearly unimodal with a strong peak at high values of dimensions 2, 3, and 4.

**Strategy:** Keep pushing toward the peak. With kappa = 1.5, the GP should continue exploiting the high-output corner while maintaining some flexibility.

**What to expect:** A query with dimensions 2, 3, and 4 close to or slightly above the Week 1 query values (0.927, 0.965, 0.986). The best direction for dimension 1 may be around 0.05 to 0.10.

**Watch for:** The output scale for F5 is in the thousands. The GP's predicted mean and uncertainty will both be large numbers. This is expected and not a problem. If the UCB score drops dramatically compared to Week 1 (644.94), it may mean the GP has become more confident and is narrowing in on the peak.

---

### Function 6

**Situation:** The Week 1 query moved away from the best-known recipe combination and found -1.213319 against a best of -0.714265 at [0.728, 0.155, 0.733, 0.694, 0.056]. The GP mispredicted this region.

**Strategy:** Return to the neighbourhood of the best-known input with kappa = 1.0. The new observation (the worse value from Week 1) gives the GP additional evidence that the boundary regions of F6 are poor, which should guide the surrogate surface toward the known good region.

**What to expect:** A query close to [0.73, 0.15, 0.73, 0.69, 0.06].

**Watch for:** F6 outputs are all negative (range -2.57 to -0.71). The goal is to get as close to 0 as possible. The best output of -0.714265 is already near the top of the observed range. It is possible the true optimum is very close to the known best and only fine-tuning is needed.

---

### Function 7

**Situation:** Consistent improvement of +30% from 1.364968 to 1.771970. F7 has a clear exploitable region. The convergence warning (noise_level at lower bound) was minor and did not hurt the query quality.

**Strategy:** Keep the same kappa = 2.0 since it worked well in Week 1. The wider kernel bounds will resolve the convergence warning and should produce an even better-fitted GP.

**What to expect:** A query in a similar neighbourhood to the Week 1 query [0.110, 0.394, 0.394, 0.093, 0.386, 0.670], potentially with small adjustments as the GP refines its understanding of the surface.

**Watch for:** If Week 2 produces another strong improvement (over 1.77), F7 may still have significant room to grow. If the gain is smaller, the function is approaching a local maximum and further exploitation will yield diminishing returns.

---

### Function 8

**Situation:** Modest improvement of +3.8% from 9.598482 to 9.965293. F8 is the highest-dimensional function (8D) and the output appears to be approaching a ceiling in the observed range (5.6 to 9.6). The gain is real but small.

**Strategy:** Maintain kappa = 2.0 to balance exploitation near the new best with exploration of the uncertain higher-dimensional space. With 8 dimensions and 41 observations, there is still a lot of unexplored space.

**What to expect:** A query close to the current best input [0.043, 0.092, 0.083, 0.051, 0.808, 0.564, 0.175, 0.420] with possible adjustments in the less-constrained dimensions (5 to 8).

**Watch for:** F8 may have a better global maximum in a completely different region. If Week 2 produces minimal gain again, it may be worth increasing kappa in Week 3 to force broader exploration.

---

## Full Notebook Code

Below is the complete Week 2 notebook code, ready to copy and run:

```python
# ============================================================
# WEEK 2 BAYESIAN OPTIMISATION - FULL NOTEBOOK
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats.qmc import Sobol


# ============================================================
# SECTION 1: LOAD ORIGINAL DATA
# ============================================================

original_data = {}

for i in range(1, 9):
    X = np.load(f"function_{i}/initial_inputs.npy")
    Y = np.load(f"function_{i}/initial_outputs.npy")
    original_data[f"function_{i}"] = {"X": X, "Y": Y}
    print(f"Function {i} | X: {X.shape} | Y: {Y.shape}")


# ============================================================
# SECTION 2: WEEK 1 QUERIES AND RETURNED OUTPUTS
# ============================================================

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


# ============================================================
# SECTION 3: BUILD UPDATED DATASETS
# ============================================================

updated_data = {}

for i in range(1, 9):
    key = f"function_{i}"

    X_original = original_data[key]["X"]
    Y_original = original_data[key]["Y"]

    X_new = week1_queries[key].reshape(1, -1)
    Y_new = np.array([week1_outputs[key]])

    X_updated = np.vstack([X_original, X_new])
    Y_updated = np.append(Y_original, Y_new)

    updated_data[key] = {"X": X_updated, "Y": Y_updated}

    print(f"Function {i} | Updated X: {X_updated.shape} | Updated Y: {Y_updated.shape}")


# ============================================================
# SECTION 4: EDA ON UPDATED DATA
# ============================================================

print("\n" + "="*60)
print("UPDATED DATASET SUMMARY")
print("="*60)

for i in range(1, 9):
    key = f"function_{i}"
    X = updated_data[key]["X"]
    Y = updated_data[key]["Y"]
    best_idx = np.argmax(Y)

    print(f"\nFunction {i}")
    print(f"  Observations : {X.shape[0]}")
    print(f"  Output range : [{Y.min():.6f}, {Y.max():.6f}]")
    print(f"  Best output  : {Y[best_idx]:.6f} at {X[best_idx]}")
    print(f"  Week 1 added : Y={Y[-1]:.6f}")

    w1_change = week1_outputs[key] - original_data[key]["Y"].max()
    direction = "IMPROVED" if w1_change > 0 else "DECLINED" if w1_change < -1e-10 else "NO CHANGE"
    print(f"  W1 outcome   : {direction} ({w1_change:+.6f})")


# ============================================================
# SECTION 5: IMPROVED GP PIPELINE
# ============================================================

def bayesian_optimisation_pipeline_v2(
    X, Y, dimension,
    kappa=2.0,
    candidate_size=None,
    n_restarts=10,
    random_seed=42
):
    if candidate_size is None:
        candidate_size = 5000 * dimension

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )

    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        random_state=42
    )
    gp_model.fit(X_scaled, Y)

    sampler = Sobol(d=dimension, scramble=True, seed=random_seed)
    candidate_points = sampler.random(n=candidate_size)
    candidate_points_scaled = scaler.transform(candidate_points)

    mean_prediction, std_prediction = gp_model.predict(
        candidate_points_scaled,
        return_std=True
    )

    ucb_scores = mean_prediction + kappa * std_prediction

    best_index = np.argmax(ucb_scores)
    best_query_scaled = candidate_points_scaled[best_index]
    best_query = scaler.inverse_transform([best_query_scaled])[0]
    best_query = np.clip(best_query, 0.0, 1.0)

    formatted_query = "-".join([f"{x:.6f}" for x in best_query])

    return {
        "query"          : best_query,
        "formatted_query": formatted_query,
        "predicted_mean" : mean_prediction[best_index],
        "uncertainty"    : std_prediction[best_index],
        "ucb_score"      : ucb_scores[best_index],
        "kernel_fitted"  : str(gp_model.kernel_),
    }


# ============================================================
# SECTION 6: PER-FUNCTION CONFIGURATION
# ============================================================

kappa_config = {
    "function_1": 0.5,
    "function_2": 1.0,
    "function_3": 1.0,
    "function_4": 1.5,
    "function_5": 1.5,
    "function_6": 1.0,
    "function_7": 2.0,
    "function_8": 2.0,
}

restart_config = {
    "function_1": 10,
    "function_2": 10,
    "function_3": 10,
    "function_4": 15,
    "function_5": 15,
    "function_6": 15,
    "function_7": 20,
    "function_8": 20,
}


# ============================================================
# SECTION 7: RUN OPTIMISATION FOR ALL FUNCTIONS
# ============================================================

week2_results = {}

for i in range(1, 9):
    key = f"function_{i}"

    print(f"\n{'='*60}")
    print(f"FUNCTION {i}")
    print(f"{'='*60}")

    X = updated_data[key]["X"]
    Y = updated_data[key]["Y"]
    dimension = X.shape[1]
    kappa = kappa_config[key]
    restarts = restart_config[key]

    print(f"Observations  : {X.shape[0]}")
    print(f"Dimensions    : {dimension}D")
    print(f"kappa         : {kappa}")
    print(f"Current best  : {Y.max():.6f}")

    result = bayesian_optimisation_pipeline_v2(
        X=X,
        Y=Y,
        dimension=dimension,
        kappa=kappa,
        n_restarts=restarts,
        random_seed=42
    )

    week2_results[key] = result

    print(f"\nFormatted query  : {result['formatted_query']}")
    print(f"Predicted mean   : {result['predicted_mean']:.6f}")
    print(f"Uncertainty      : {result['uncertainty']:.6f}")
    print(f"UCB score        : {result['ucb_score']:.6f}")


# ============================================================
# SECTION 8: QUERY VALIDATION
# ============================================================

print("\n" + "="*60)
print("QUERY VALIDATION REPORT")
print("="*60)

all_clear = True

for i in range(1, 9):
    key = f"function_{i}"
    query = week2_results[key]["query"]
    ucb = week2_results[key]["ucb_score"]
    issues = []

    if np.any(query < 0) or np.any(query > 1):
        issues.append("OUT OF RANGE: values outside [0, 1]")
    if np.all(query < 0.01):
        issues.append("SUSPICIOUS: all values near 0")
    if np.all(query > 0.99):
        issues.append("SUSPICIOUS: all values near 1")
    if ucb < 0:
        issues.append(f"NEGATIVE UCB: {ucb:.4f}")

    status = "OK" if not issues else "WARNING"
    print(f"\nFunction {i} [{status}]")
    print(f"  Query : {week2_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues:
            print(f"  !! {w}")

print()
print("All queries valid." if all_clear else "Review warnings above before submitting.")


# ============================================================
# SECTION 9: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key = f"function_{i}"
    result = week2_results[key]
    Y = updated_data[key]["Y"]

    rows.append({
        "Function"       : f"Function {i}",
        "Query"          : result["formatted_query"],
        "Predicted_mean" : round(result["predicted_mean"], 6),
        "Uncertainty"    : round(result["uncertainty"], 6),
        "UCB_score"      : round(result["ucb_score"], 6),
        "Current_best"   : round(float(Y.max()), 6),
    })

query_df = pd.DataFrame(rows)
print(query_df.to_string(index=False))

query_df[["Function", "Query"]].to_csv("week2_queries.csv", index=False)
print("\nweek2_queries.csv saved.")

query_df.to_csv("week2_queries_full.csv", index=False)
print("week2_queries_full.csv saved.")
```

---

*Week 2 strategy document. All query coordinates, kappa values, and improvement decisions are derived from Week 1 results and the updated datasets.*
