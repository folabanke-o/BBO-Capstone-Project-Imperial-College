# Bayesian Optimisation Capstone: Week 3 Strategy and Code

> **Building on:** Week 1 and Week 2 results now incorporated
> **Week 3 objective:** Generate better-informed queries using two full iterations of data

---

## Table of Contents

1. [Week 2 Results Analysis](#week-2-results-analysis)
2. [Updated Datasets](#updated-datasets)
3. [Key Changes from Week 2](#key-changes-from-week-2)
4. [Per-Function Strategy](#per-function-strategy)
5. [Full Notebook Code](#full-notebook-code)

---

## Week 2 Results Analysis

| Function | W2 query | W2 output | Best before W2 | New best | Outcome |
|----------|----------|-----------|----------------|----------|---------|
| F1 | [0.706, 0.823] | -4.68e-32 | 0.000000 | 0.000000 | No gain |
| F2 | [0.839, 0.896] | 0.14504 | 0.611205 | 0.611205 | Declined |
| F3 | [0.364, 0.758, 0.191] | -0.11945 | -0.034835 | -0.034835 | Declined |
| F4 | [0.404, 0.398, 0.413, 0.412] | **0.53386** | -0.869002 | **0.533858** | Improved |
| F5 | [0.056, 0.993, 0.973, 0.960] | **3511.612** | 3019.660 | **3511.612** | Improved |
| F6 | [0.730, 0.157, 0.735, 0.705, 0.068] | -0.80060 | -0.714265 | -0.714265 | Declined |
| F7 | [0.058, 0.304, 0.328, 0.001, 0.275, 0.673] | 1.28885 | 1.771970 | 1.771970 | Declined |
| F8 | [0.017, 0.092, 0.306, 0.116, 0.946, 0.608, 0.053, 0.856] | 9.81687 | 9.965293 | 9.965293 | Declined |

**Key observations:**

F4 is the standout result. It broke through zero for the first time, going from -0.869 to +0.534. This is a significant signal about where the optimum lies.

F5 continued its upward trend (+16.3%) confirming the peak region at high dims 2-4. Three weeks of consistent gains in the same direction is a strong signal.

F2, F3, F6, F7, and F8 all declined in Week 2. F7 declining after a strong Week 1 gain (+30%) suggests the W2 query moved slightly away from the true optimum. F8 declining marginally suggests it may be near a local maximum.

F1 has now returned near-zero across three queries. The function's output surface is essentially flat outside extremely narrow peaks.

---

## Updated Datasets

After stacking Week 2 observations, each dataset now contains:

| Function | Original | After W1 | After W2 | New best |
|----------|---------|---------|---------|---------|
| F1 | 10 | 11 | 12 | 0.000000 |
| F2 | 10 | 11 | 12 | 0.611205 |
| F3 | 15 | 16 | 17 | -0.034835 |
| F4 | 30 | 31 | 32 | **0.533858** |
| F5 | 20 | 21 | 22 | **3511.612** |
| F6 | 20 | 21 | 22 | -0.714265 |
| F7 | 30 | 31 | 32 | 1.771970 |
| F8 | 40 | 41 | 42 | 9.965293 |

---

## Key Changes from Week 2

| Area | Week 2 | Week 3 change |
|------|--------|--------------|
| kappa F4 | 1.5 | 1.0 - F4 now positive, tighten around new best |
| kappa F5 | 1.5 | 1.0 - peak well characterised, fine-tune |
| kappa F7 | 2.0 | 1.5 - declined in W2, pull back toward known best |
| kappa F8 | 2.0 | 2.5 - modest/negative gains, force broader exploration |
| kappa F1 | 0.5 | 0.3 - three wasted queries, maximum exploitation |
| kappa F2 | 1.0 | 0.5 - declined twice, tight exploitation of 0.611 peak |
| kappa F3 | 1.0 | 0.5 - declined twice, tight exploitation of -0.035 best |
| kappa F6 | 1.0 | 0.5 - declined twice, tight exploitation of -0.714 best |
| F6 fix | Manual perturbation | Built into pipeline loop directly |
| Constraints | Post-hoc override cells | Built into pipeline v3, no overrides needed |

---

## Per-Function Strategy

### Function 1

Three queries have all returned near-zero. The function has extremely narrow peaks and most of the space is flat. The best known output remains 0.000000 at [0.731, 0.733] from the initial data.

The two W2 query coordinates (0.706, 0.823) are in the right neighbourhood but clearly not close enough. Dim 2 needs to come down from 0.823 toward 0.733. Use kappa 0.3 and a very tight constraint box of margin 0.05 around [0.731, 0.733].

**kappa:** 0.3
**Bounds:** [0.681, 0.683] to [0.781, 0.783]

---

### Function 2

Declined in both W1 and W2. Best remains 0.611205 at [0.703, 0.927]. W2 query returned 0.145 from (0.839, 0.896) which is not an improvement. The GP keeps drifting slightly away from the confirmed best. Use kappa 0.5 with a tight box around [0.703, 0.927].

**kappa:** 0.5
**Bounds:** [0.603, 0.827] to [0.803, 1.000]

---

### Function 3

Declined in both W1 and W2. Best remains -0.034835 at [0.493, 0.612, 0.340]. W2 query returned -0.119, which is still worse than the initial best. The interior region needs tighter targeting. Use kappa 0.5 with a box of margin 0.10 around [0.493, 0.612, 0.340].

**kappa:** 0.5
**Bounds:** [0.393, 0.512, 0.240] to [0.593, 0.712, 0.440]

---

### Function 4

The most exciting W2 result. Output went from -0.869 to +0.534, crossing zero for the first time. The W2 query input [0.404, 0.398, 0.413, 0.412] is now the best known input. Tighten the box around this new best with kappa 1.0 to continue climbing.

**kappa:** 1.0
**Bounds:** [0.304, 0.298, 0.313, 0.312] to [0.504, 0.498, 0.513, 0.512]

---

### Function 5

Consistent gains across all three weeks: 1088 to 3019 to 3511. F5 is clearly unimodal and the peak is at high dims 2-4 with dim 1 low. The W2 query (0.056, 0.993, 0.973, 0.960) produced the highest output yet. Keep the constraint on the high-value corner and reduce kappa to 1.0 for fine-tuning.

**kappa:** 1.0
**Bounds:** [0.00, 0.90, 0.92, 0.90] to [0.12, 1.00, 1.00, 1.00]

---

### Function 6

Declined in both W1 and W2. Best remains -0.714265 at [0.728, 0.155, 0.733, 0.694, 0.056]. The W2 query was the closest yet to the best known recipe (distance 0.017) but still returned a worse value (-0.801). This suggests the optimum is very tightly localised. Use kappa 0.5 with a margin of just 0.07 around the best known input.

**kappa:** 0.5
**Bounds:** margin 0.07 around [0.728, 0.155, 0.733, 0.694, 0.056]

---

### Function 7

Declined in W2 (1.289 vs best 1.772). The W2 query (0.058, 0.304, 0.328, 0.001, 0.275, 0.673) moved away from the W1 best at [0.110, 0.394, 0.394, 0.093, 0.386, 0.670]. For W3, return to the W1 best location since that produced the highest output. Use kappa 1.5 with a constraint box around the W1 best input.

**kappa:** 1.5
**Bounds:** margin 0.12 around [0.110, 0.394, 0.394, 0.093, 0.386, 0.670]

---

### Function 8

Declined slightly in W2 (9.817 vs best 9.965). With 42 observations in 8D the function is likely near a local maximum in the current region. Increase kappa to 2.5 and widen the search to allow the GP to find a potentially better region elsewhere in the 8D space.

**kappa:** 2.5
**Bounds:** full [0,1]^8 space

---

## Full Notebook Code

```python
# ============================================================
# WEEK 3 BAYESIAN OPTIMISATION - FULL NOTEBOOK
# ============================================================

import numpy as np
import pandas as pd
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
# SECTION 2: ALL QUERY HISTORY (W1 and W2)
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

week2_queries = {
    "function_1": np.array([0.705908, 0.823143]),
    "function_2": np.array([0.839078, 0.895770]),
    "function_3": np.array([0.363504, 0.758379, 0.191085]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.729856, 0.157383, 0.734992, 0.704798, 0.068448]),
    "function_7": np.array([0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712]),
}

week2_outputs = {
    "function_1": -4.676913887169069e-32,
    "function_2": 0.14503569246975664,
    "function_3": -0.11944712762491103,
    "function_4": 0.5338577755032223,
    "function_5": 3511.611905490813,
    "function_6": -0.8006000173001564,
    "function_7": 1.2888474165310304,
    "function_8": 9.8168730656046,
}


# ============================================================
# SECTION 3: BUILD UPDATED DATASETS (original + W1 + W2)
# ============================================================

updated_data = {}

for i in range(1, 9):
    key = f"function_{i}"

    X_orig = original_data[key]["X"]
    Y_orig = original_data[key]["Y"]

    X_updated = np.vstack([
        X_orig,
        week1_queries[key].reshape(1, -1),
        week2_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(Y_orig, [week1_outputs[key], week2_outputs[key]])

    updated_data[key] = {"X": X_updated, "Y": Y_updated}

    best_idx = np.argmax(Y_updated)
    print(f"Function {i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f}")


# ============================================================
# SECTION 4: EDA ON UPDATED DATA
# ============================================================

print("\n" + "="*60)
print("UPDATED DATASET SUMMARY - ENTERING WEEK 3")
print("="*60)

for i in range(1, 9):
    key = f"function_{i}"
    X = updated_data[key]["X"]
    Y = updated_data[key]["Y"]
    best_idx = np.argmax(Y)

    w2_change = week2_outputs[key] - max(
        original_data[key]["Y"].max(),
        week1_outputs[key]
    )
    direction = "IMPROVED" if w2_change > 1e-10 else (
        "NO CHANGE" if abs(w2_change) < 1e-10 else "DECLINED"
    )

    print(f"\nFunction {i}")
    print(f"  Observations : {X.shape[0]}")
    print(f"  Best output  : {Y[best_idx]:.6f} at {np.round(X[best_idx], 4)}")
    print(f"  W2 output    : {week2_outputs[key]:.6f} | {direction}")
    print(f"  Output range : [{Y.min():.4f}, {Y.max():.4f}]")


# ============================================================
# SECTION 5: PIPELINE V3 WITH BOUNDS BUILT IN
# ============================================================

def bayesian_optimisation_pipeline_v3(
    X, Y, dimension,
    kappa=2.0,
    candidate_size=None,
    n_restarts=10,
    random_seed=42,
    low_bounds=None,
    high_bounds=None,
    y_shift=False,
):
    """
    Week 3 pipeline. Improvements over Week 2:
    - Bounds passed directly (no post-hoc override cells needed)
    - Optional Y-shift for all-negative output functions
    - Sobol candidate generation
    - Wider kernel bounds
    - Fixed random seed
    """

    if candidate_size is None:
        candidate_size = 5000 * dimension

    if low_bounds is None:
        low_bounds = np.zeros(dimension)
    if high_bounds is None:
        high_bounds = np.ones(dimension)

    Y_fit = Y - Y.min() if y_shift else Y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e4))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        random_state=42
    )
    gp.fit(X_scaled, Y_fit)

    sampler = Sobol(d=dimension, scramble=True, seed=random_seed)
    candidates = sampler.random(n=candidate_size)

    # Rescale Sobol [0,1] output to the actual bounds
    candidates = low_bounds + candidates * (high_bounds - low_bounds)
    candidates = np.clip(candidates, 0.0, 1.0)

    candidates_scaled = scaler.transform(candidates)
    mean_pred, std_pred = gp.predict(candidates_scaled, return_std=True)
    ucb = mean_pred + kappa * std_pred

    best_idx   = np.argmax(ucb)
    best_query = np.clip(candidates[best_idx], 0.0, 1.0)
    formatted  = "-".join([f"{x:.6f}" for x in best_query])

    return {
        "query"          : best_query,
        "formatted_query": formatted,
        "predicted_mean" : float(mean_pred[best_idx]),
        "uncertainty"    : float(std_pred[best_idx]),
        "ucb_score"      : float(ucb[best_idx]),
        "kernel_fitted"  : str(gp.kernel_),
    }


# ============================================================
# SECTION 6: PER-FUNCTION CONFIGURATION
# ============================================================

# Best known inputs going into Week 3
best_known = {
    "function_1": np.array([0.731024, 0.732999]),
    "function_2": np.array([0.702637, 0.926564]),
    "function_3": np.array([0.492581, 0.611593, 0.340176]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),  # W2 query
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),  # W2 query
    "function_6": np.array([0.728186, 0.154693, 0.732552, 0.693997, 0.056401]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),  # W1 query
    "function_8": None,  # full space exploration
}

# Margins for constrained search boxes
margins = {
    "function_1": 0.05,   # very tight - narrow peak
    "function_2": 0.10,   # tight - confirmed multimodal
    "function_3": 0.10,   # tight - interior optimum
    "function_4": 0.10,   # moderate - new positive best
    "function_5": None,   # custom bounds below
    "function_6": 0.07,   # tight - best recipe
    "function_7": 0.12,   # moderate - return to W1 best
    "function_8": None,   # full space
}

# Build bounds from best known + margin
def make_bounds(fn_key):
    best = best_known[fn_key]
    margin = margins[fn_key]
    if best is None or margin is None:
        return None, None
    low  = np.clip(best - margin, 0.0, 1.0)
    high = np.clip(best + margin, 0.0, 1.0)
    return low, high

# F5 gets custom asymmetric bounds
f5_low  = np.array([0.00, 0.90, 0.92, 0.90])
f5_high = np.array([0.12, 1.00, 1.00, 1.00])

kappa_config = {
    "function_1": 0.3,   # maximum exploitation - 3 wasted queries
    "function_2": 0.5,   # tight exploit - declined twice
    "function_3": 0.5,   # tight exploit - declined twice
    "function_4": 1.0,   # exploit new positive best
    "function_5": 1.0,   # fine-tune peak, consistent gains
    "function_6": 0.5,   # tight exploit - declined twice
    "function_7": 1.5,   # return to W1 best, moderate exploit
    "function_8": 2.5,   # force broader exploration
}

restart_config = {
    "function_1": 10, "function_2": 10, "function_3": 10,
    "function_4": 15, "function_5": 15, "function_6": 15,
    "function_7": 20, "function_8": 20,
}

# Y-shift: apply to functions with all-negative outputs
# F4 now positive so no longer needed; F3 and F6 still negative
y_shift_config = {
    "function_1": False, "function_2": False, "function_3": True,
    "function_4": False, "function_5": False, "function_6": True,
    "function_7": False, "function_8": False,
}

print("\nWEEK 3 CONFIGURATION")
print("="*60)
print(f"{'Function':<12} {'kappa':<8} {'restarts':<10} {'Y-shift':<10} {'constrained'}")
print("-"*60)
for i in range(1, 9):
    key = f"function_{i}"
    low, high = make_bounds(key) if key != "function_5" else (f5_low, f5_high)
    constrained = "Yes" if low is not None else "Full space"
    print(f"F{i:<11} {kappa_config[key]:<8} {restart_config[key]:<10} "
          f"{str(y_shift_config[key]):<10} {constrained}")


# ============================================================
# SECTION 7: RUN OPTIMISATION FOR ALL FUNCTIONS
# ============================================================

week3_results = {}

for i in range(1, 9):
    key = f"function_{i}"

    print(f"\n{'='*55}")
    print(f"FUNCTION {i}")
    print(f"{'='*55}")

    X = updated_data[key]["X"]
    Y = updated_data[key]["Y"]
    dimension = X.shape[1]

    if key == "function_5":
        low_b, high_b = f5_low, f5_high
    else:
        low_b, high_b = make_bounds(key)

    result = bayesian_optimisation_pipeline_v3(
        X=X, Y=Y, dimension=dimension,
        kappa=kappa_config[key],
        n_restarts=restart_config[key],
        random_seed=42,
        low_bounds=low_b,
        high_bounds=high_b,
        y_shift=y_shift_config[key],
    )

    week3_results[key] = result

    constrained = "YES" if low_b is not None else "Full space"
    print(f"Observations   : {X.shape[0]}")
    print(f"Current best   : {Y.max():.6f}")
    print(f"kappa          : {kappa_config[key]}")
    print(f"Constrained    : {constrained}")
    print(f"Query          : {result['formatted_query']}")
    print(f"Predicted mean : {result['predicted_mean']:.6f}")
    print(f"Uncertainty    : {result['uncertainty']:.6f}")
    print(f"UCB score      : {result['ucb_score']:.6f}")


# ============================================================
# SECTION 8: QUERY VALIDATION
# ============================================================

print("\n" + "="*55)
print("QUERY VALIDATION REPORT")
print("="*55)

all_clear = True

for i in range(1, 9):
    key = f"function_{i}"
    query = week3_results[key]["query"]
    ucb   = week3_results[key]["ucb_score"]
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
    print(f"  Query : {week3_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues:
            print(f"  !! {w}")

if all_clear:
    print("\nAll queries passed validation.")
else:
    print("\nReview warnings before submitting.")


# ============================================================
# SECTION 9: PROXIMITY CHECK
# ============================================================

print("\n" + "="*55)
print("PROXIMITY TO BEST KNOWN INPUT")
print("="*55)

best_inputs = {
    "function_1": np.array([0.731024, 0.732999]),
    "function_2": np.array([0.702637, 0.926564]),
    "function_3": np.array([0.492581, 0.611593, 0.340176]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.728186, 0.154693, 0.732552, 0.693997, 0.056401]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712]),
}

for i in range(1, 9):
    key = f"function_{i}"
    q   = week3_results[key]["query"]
    b   = best_inputs[key]
    d   = np.linalg.norm(q - b)
    flag = "ok" if d < 0.20 else "far"
    print(f"Function {i} | distance from best input: {d:.4f} [{flag}]")


# ============================================================
# SECTION 10: SAVE QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key = f"function_{i}"
    Y   = updated_data[key]["Y"]
    r   = week3_results[key]

    rows.append({
        "Function"       : f"Function {i}",
        "Query"          : r["formatted_query"],
        "Predicted_mean" : round(r["predicted_mean"], 6),
        "Uncertainty"    : round(r["uncertainty"], 6),
        "UCB_score"      : round(r["ucb_score"], 6),
        "Current_best"   : round(float(Y.max()), 6),
        "kappa"          : kappa_config[key],
    })

df = pd.DataFrame(rows)
print("\nFINAL WEEK 3 QUERIES")
print(df[["Function", "Query", "UCB_score", "Current_best"]].to_string(index=False))

df[["Function", "Query"]].to_csv("week3_queries.csv", index=False)
print("\nweek3_queries.csv saved.")

df.to_csv("week3_queries_full.csv", index=False)
print("week3_queries_full.csv saved.")
```

---

## Important notes for Week 3

**No override cells needed.** The constraints and Y-shifts are built directly into the pipeline loop via `bounds_config` and `y_shift_config`. Section 7 runs once and that is final. Do not re-run Section 7 after validation.

**F7 returns to Week 1 best.** The W2 query moved too far from the W1 best location that produced 1.772. The W3 bounds are centred on the W1 query input [0.110, 0.394, 0.394, 0.093, 0.386, 0.670] rather than the W2 query input.

**F8 kappa increased to 2.5.** Two consecutive weeks of modest or negative gains suggest a local maximum. Higher kappa forces the GP to explore the broader 8D space.

**F4 is now unconstrained on Y.** Now that F4 outputs are positive (0.534) the Y-shift is no longer needed. The GP can distinguish good from bad regions directly.

**Update kappa after results come back.** If F5 gains again, reduce to 0.5. If F8 improves with kappa 2.5, keep it. If F1 still returns near-zero, consider a completely different region.

---

*Week 3 strategy document. All decisions derived from Week 1 and Week 2 empirical results.*
