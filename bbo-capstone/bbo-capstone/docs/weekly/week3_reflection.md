# Bayesian Optimisation Capstone: Week 3 Reflection

> **Week 3 objective:** Incorporate Week 2 results, update bounds, revise kappa, and generate improved queries
> **Key result:** 2 of 8 functions improved in Week 2. 5 declined or held. Pipeline constraints now built into the loop.

---

## Table of Contents

1. [Week 2 Results Analysis](#week-2-results-analysis)
2. [What Changed for Week 3](#what-changed-for-week-3)
3. [Updated Datasets](#updated-datasets)
4. [Pipeline: Key Changes](#pipeline-key-changes)
5. [Week 3 Queries](#week-3-queries)
6. [Function-by-Function Analysis](#function-by-function-analysis)
7. [Issues and Challenges](#issues-and-challenges)
8. [Strategy Reflection](#strategy-reflection)
9. [Week 4 Predictions](#week-4-predictions)

---

## Week 2 Results Analysis

| Function | W2 query sent | W2 output | Previous best | Change | Outcome |
|----------|--------------|-----------|---------------|--------|---------|
| F1 | 0.705908-0.823143 | -4.68e-32 | 0.000000 | ~0 | No gain |
| F2 | 0.839078-0.895770 | 0.145036 | 0.611205 | -0.466 | Declined |
| F3 | 0.363504-0.758379-0.191085 | -0.119447 | -0.034835 | -0.085 | Declined |
| F4 | 0.403695-0.397605-0.413333-0.411576 | **0.533858** | -0.869002 | +1.403 | **Improved** |
| F5 | 0.056181-0.992607-0.973199-0.959866 | **3511.612** | 3019.660 | +491.95 | **Improved** |
| F6 | 0.729856-0.157383-0.734992-0.704798-0.068448 | -0.800600 | -0.714265 | -0.086 | Declined |
| F7 | 0.058090-0.303889-0.327991-0.001268-0.275231-0.673181 | 1.288847 | 1.771970 | -0.483 | Declined |
| F8 | 0.017074-0.091604-0.305973-0.115845-0.946320-0.608139-0.053440-0.855712 | 9.816873 | 9.965293 | -0.148 | Declined |

**Only F4 and F5 improved. Six functions declined or held.**

This is a significant step back from Week 1 where 5 of 8 improved. The most likely causes are that several constrained search boxes were too loose (F2, F3) or the GP surface was still not well-characterised enough to exploit correctly (F7, F8). F4 and F5 continue to improve steadily, confirming their regions are well-identified.

---

## What Changed for Week 3

| Area | Week 2 approach | Week 3 change |
|------|----------------|--------------|
| Constraint method | Post-hoc override cells | Constraints built into pipeline loop directly |
| F4 | Y-shift applied as override | Y-shift inside pipeline, tighter box around new best |
| F5 | Constrained [0.00-0.15, 0.85-1.0, 0.90-1.0, 0.90-1.0] | Tightened to [0.00-0.10, 0.95-1.0, 0.95-1.0, 0.93-1.0] |
| F6 | Manual perturbation bypass | GP re-attempted with tighter box, kappa reduced to 0.5 |
| F7 | kappa 2.0, no constraint | kappa reduced to 1.5, constrained near W1 best |
| F8 | kappa 2.0, full space | kappa kept 2.0, full space (W2 decline suggests exploring) |
| F2 | Constrained near [0.60-0.80, 0.82-1.0] | Tightened to [0.60-0.80, 0.88-1.0] |
| F3 | Constrained near best | Bounds updated to include W2 query region |

---

## Updated Datasets

Each function's dataset now contains original observations plus Week 1 and Week 2 results.

| Function | Obs after W2 | Best known output | Best known input |
|----------|-------------|-------------------|-----------------|
| F1 | 12 | 0.000000 | [0.731, 0.733] |
| F2 | 12 | 0.611205 | [0.703, 0.927] |
| F3 | 16 | -0.034835 | [0.493, 0.612, 0.340] |
| F4 | 31 | **0.533858** | [0.404, 0.398, 0.413, 0.412] |
| F5 | 22 | **3511.612** | [0.056, 0.993, 0.973, 0.960] |
| F6 | 21 | -0.714265 | [0.728, 0.155, 0.733, 0.694, 0.056] |
| F7 | 31 | 1.771970 | [0.110, 0.394, 0.394, 0.093, 0.386, 0.670] |
| F8 | 41 | 9.965293 | [0.043, 0.092, 0.083, 0.051, 0.808, 0.564, 0.175, 0.420] |

The stacking code for Week 3:

```python
w2_queries = {
    "function_1": np.array([0.705908, 0.823143]),
    "function_2": np.array([0.839078, 0.895770]),
    "function_3": np.array([0.363504, 0.758379, 0.191085]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.729856, 0.157383, 0.734992, 0.704798, 0.068448]),
    "function_7": np.array([0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712]),
}

w2_outputs = {
    "function_1": -4.676913887169069e-32,
    "function_2": 0.14503569246975664,
    "function_3": -0.11944712762491103,
    "function_4": 0.5338577755032223,
    "function_5": 3511.611905490813,
    "function_6": -0.8006000173001564,
    "function_7": 1.2888474165310304,
    "function_8": 9.8168730656046,
}

for i in range(1, 9):
    key = f"function_{i}"
    X_updated = np.vstack([updated_data[key]["X"], w2_queries[key].reshape(1, -1)])
    Y_updated = np.append(updated_data[key]["Y"], w2_outputs[key])
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
```

---

## Pipeline: Key Changes

### Constraints built into the loop

The Week 2 override cell approach was replaced with a `bounds_config` dictionary applied directly inside the pipeline loop. This eliminates the dictionary reference problem entirely.

```python
bounds_config = {
    "function_1": (np.array([0.68, 0.68]),  np.array([0.78, 0.78])),
    "function_2": (np.array([0.60, 0.88]),  np.array([0.80, 1.00])),
    "function_3": (np.array([0.34, 0.56, 0.19]), np.array([0.64, 0.76, 0.49])),
    "function_4": (np.array([0.35, 0.35, 0.36, 0.36]), np.array([0.50, 0.50, 0.50, 0.50])),
    "function_5": (np.array([0.00, 0.95, 0.95, 0.93]), np.array([0.10, 1.00, 1.00, 1.00])),
    "function_6": (np.array([0.678, 0.105, 0.682, 0.644, 0.006]), np.array([0.778, 0.205, 0.782, 0.744, 0.106])),
    "function_7": None,
    "function_8": None,
}

kappa_config = {
    "function_1": 0.5,
    "function_2": 1.0,
    "function_3": 1.0,
    "function_4": 1.5,
    "function_5": 1.0,
    "function_6": 0.5,
    "function_7": 2.0,
    "function_8": 2.0,
}
```

### F5 kappa reduced from 1.5 to 1.0

F5 gained another +16.3% in Week 2 (3019 to 3512). The function is clearly unimodal and the peak is well-characterised. Reducing kappa to 1.0 shifts the GP toward fine-tuning rather than exploring within the constrained box.

### F6 kappa reduced from 1.0 to 0.5

F6 declined again in Week 2 despite the manual perturbation placing the query close to the best known recipe (distance 0.017). The GP needs to exploit even more tightly. Kappa 0.5 forces the GP to stay very close to the best known point.

### F7 kappa reduced from 2.0 to 2.0 but re-examined

F7 declined in Week 2 (1.772 to 1.289). The Week 2 query drifted from the Week 1 best, and the GP did not recover. For Week 3, the full space search is kept but the best known input from Week 1 (1.771970) is the target.

---

## Week 3 Queries

| Function | Query | mu | sigma | UCB | Risk |
|----------|-------|----|-------|-----|------|
| F1 | `0.779674-0.779956` | ~0 | 0.000011 | 0.000005 | Low |
| F2 | `0.600784-0.999903` | 0.379701 | 0.218456 | 0.598156 | Low |
| F3 | `0.433023-0.757795-0.489963` | -0.030314 | 0.047199 | 0.016885 | Low |
| F4 | `0.382185-0.372807-0.445018-0.391213` | 9.272 | 10.414 | 24.893 | Low |
| F5 | `0.050664-0.950310-0.983145-0.994092` | 3849.82 | 519.71 | 4369.53 | Very low |
| F6 | `0.764584-0.198888-0.781272-0.739919-0.006594` | -0.038 | 0.753 | 0.339 | Medium |
| F7 | `0.164343-0.288939-0.470804-0.074161-0.191549-0.651121` | 1.727 | 0.758 | 3.244 | Medium |
| F8 | `0.982918-0.951370-0.056865-0.937888-0.904726-0.121533-0.134019-0.035698` | 15.211 | 5.189 | 25.589 | Medium |

---

## Function-by-Function Analysis

### Function 1

**W2 result:** -4.68e-32 (effectively zero, no gain)
**New best:** 0.000000 unchanged at [0.731, 0.733]
**W3 query:** 0.779674-0.779956
**W3 kappa:** 0.5

The constrained box narrowed around the best known peak produced a query at (0.780, 0.780), very close to [0.731, 0.733]. The query has moved slightly further from the best in dim 1 (0.780 vs 0.706), but both dims are still within the constrained box. F1 consistently returns near-zero values everywhere except at the narrow peak. The GP is doing its best with very little signal to work with.

**Lesson:** Consider tightening the box to +/- 0.03 in Week 4. At this point even small moves may matter.

---

### Function 2

**W2 result:** 0.145036 (declined from best of 0.611205)
**New best:** 0.611205 unchanged at [0.703, 0.927]
**W3 query:** 0.600784-0.999903
**W3 kappa:** 1.0

The Week 2 query at (0.839, 0.896) returned 0.145, confirming that region is not as good as the known best at (0.703, 0.927). The Week 3 query moves dim 1 closer to 0.703 (0.601 vs 0.839) and pushes dim 2 higher (0.9999 vs 0.896). This is a sensible refinement toward the confirmed best region.

**Lesson:** F2 is multimodal and the GP is beginning to home in on the right area after two poor exploration results.

---

### Function 3

**W2 result:** -0.119447 (declined from best of -0.034835)
**New best:** -0.034835 unchanged at [0.493, 0.612, 0.340]
**W3 query:** 0.433023-0.757795-0.489963
**W3 kappa:** 1.0

The Week 2 query at (0.364, 0.758, 0.191) returned -0.119, worse than the best but better than the boundary results. The Week 3 query (0.433, 0.758, 0.490) stays near dim 2 (0.758 vs 0.758 identical) but adjusts dims 1 and 3. Dim 3 moves from 0.191 to 0.490, getting closer to the best known value of 0.340. This is the most interior query F3 has received.

**Lesson:** The GP is learning that the interior region around [0.49, 0.61, 0.34] is the right area. Week 3 should return a result closer to the best.

---

### Function 4

**W2 result:** 0.533858 (improved from -0.869002)
**New best:** 0.533858 at [0.404, 0.398, 0.413, 0.412]
**W3 query:** 0.382185-0.372807-0.445018-0.391213
**W3 kappa:** 1.5

F4 has now crossed zero for the first time, which is significant since the function represents the difference from an expensive baseline. A positive value means the model is now outperforming the baseline. The Week 3 query stays in the same neighbourhood. The very large UCB score (24.893) reflects high uncertainty in the region after the surprising positive result.

**Lesson:** F4 is the most improved function of the competition. Keep exploiting this neighbourhood.

---

### Function 5

**W2 result:** 3511.612 (improved from 3019.660, +16.3%)
**New best:** 3511.612 at [0.056, 0.993, 0.973, 0.960]
**W3 query:** 0.050664-0.950310-0.983145-0.994092
**W3 kappa:** 1.0 (reduced from 1.5)

F5 continues its consistent improvement. Three consecutive gains: 1089 to 3020 to 3512. The W3 query keeps dim 1 low (0.051) and pushes dims 3 and 4 higher (0.983 vs 0.973, 0.994 vs 0.960). Dim 2 drops slightly (0.950 vs 0.993) which may reflect the GP finding a slightly different peak angle. The predicted mean alone (3849) is already above the current best.

**Lesson:** F5 is the standout function. The GP has the peak well-characterised and kappa 1.0 is the right setting for fine-tuning.

---

### Function 6

**W2 result:** -0.800600 (declined from -0.714265)
**New best:** -0.714265 unchanged at [0.728, 0.155, 0.733, 0.694, 0.056]
**W3 query:** 0.764584-0.198888-0.781272-0.739919-0.006594
**W3 kappa:** 0.5

F6 has now declined in all three query attempts. The Week 2 manual perturbation placing the query 0.017 from the best still returned a worse value, which suggests the best known point at [0.728, 0.155, 0.733, 0.694, 0.056] may be very close to a local maximum and small perturbations consistently land on the downslope. The Week 3 query (0.765, 0.199, 0.781, 0.740, 0.007) is within the constrained box but is the furthest from the best known point of any F6 query. Kappa 0.5 was intended to prevent this drift but the GP with the tighter bounds still explored toward the box boundary. This is flagged as a concern.

**Lesson:** For Week 4, consider a direct query at exactly the best known point [0.728186, 0.154693, 0.732552, 0.693997, 0.056401] with no perturbation.

---

### Function 7

**W2 result:** 1.288847 (declined from 1.771970)
**New best:** 1.771970 unchanged at [0.110, 0.394, 0.394, 0.093, 0.386, 0.670]
**W3 query:** 0.164343-0.288939-0.470804-0.074161-0.191549-0.651121
**W3 kappa:** 2.0

F7's Week 2 decline (1.289 vs best of 1.772) shows the GP moved away from the Week 1 best region. The Week 3 query has dim 6 close to the best (0.651 vs 0.670), which is positive. Dims 1 through 5 are in a new neighbourhood. Given the consistent good performance of F7 in Week 1, the GP exploring slightly different regions is reasonable. The function may have a broader optimum region that Week 3 can reveal.

**Lesson:** If Week 3 returns below 1.772, constrain the search to a box around [0.110, 0.394, 0.394, 0.093, 0.386, 0.670] in Week 4.

---

### Function 8

**W2 result:** 9.816873 (declined from 9.965293)
**New best:** 9.965293 unchanged at [0.043, 0.092, 0.083, 0.051, 0.808, 0.564, 0.175, 0.420]
**W3 query:** 0.982918-0.951370-0.056865-0.937888-0.904726-0.121533-0.134019-0.035698
**W3 kappa:** 2.0

The Week 3 query is a significant departure from previous F8 queries. Dims 1, 2, 4, and 5 are all very high (0.983, 0.951, 0.938, 0.905), which is a very different region from the current best neighbourhood (all low dims 1-4). This is the GP exploring under kappa 2.0 in a high-uncertainty region. The predicted mean of 15.21 is well above any observed output (current max 9.965), which suggests either the GP has identified a genuinely better region or its surface is poorly calibrated. Given the large sigma (5.189) this is speculative but worth submitting since the alternative is staying near a region that declined in Week 2.

**Lesson:** High risk, high potential. If it returns near 15, the GP has found a much better region. If it returns near current best, constrain to the low-dim-1-4 neighbourhood in Week 4.

---

## Issues and Challenges

### Issue 1 - Six functions declined in Week 2

The wider-than-expected decline in Week 2 suggests the constrained search boxes from the override approach were not consistently applied. Three functions (F7, F8, F2) had no constraints applied in Week 2, and all three declined. This reinforces the decision to build constraints into the pipeline loop for Week 3.

### Issue 2 - F6 declining despite being close to best known point

The Week 2 F6 query was placed 0.017 from the best known recipe yet still returned -0.800 against a best of -0.714. This pattern across three weeks (W1: -1.213, W2: -0.800, best: -0.714) suggests either: (a) the best known point is at or very near a local maximum and any perturbation makes things worse, or (b) the function is noisy and results near the best are inconsistent. Either way, Week 4 should query exactly the best known point without perturbation.

### Issue 3 - F7 regression after strong Week 1

F7 gained +30% in Week 1 then declined in Week 2. The Week 2 query moved dim 4 to near-zero (0.001) which likely moved the query away from the region that produced the Week 1 gain. For Week 3 the search reverts to the full space under kappa 2.0, hoping the GP uses both weeks of data to find the good region again.

### Issue 4 - F8 Week 3 query is a high-uncertainty gamble

The GP has produced a Week 3 F8 query in a completely new region with a very high predicted mean (15.21) and high uncertainty (5.19). This is the UCB's exploration term dominating on sparse 8D data. It may find a genuinely better region or it may be a GP artefact. With kappa 2.0 this kind of exploratory jump is expected and accepted.

---

## Strategy Reflection

The main strategic question after Week 2 results is whether the constrained exploitation approach is working. The answer is mixed. F4 and F5 improved consistently across all weeks, confirming that well-characterised functions respond well to constrained exploitation. Functions where the surface is poorly understood (F7, F8) or where the GP consistently produces poor fits (F6) did not benefit from the same approach.

The clearest lesson from Week 2 is that constraints only help when the GP has enough data to model the region inside the box accurately. For F6, even with 21 observations and a tight box, the GP could not distinguish between points near the best known recipe. For F7 and F8 with 6-8 dimensions, the data is too sparse for constrained exploitation to work reliably without first understanding the landscape better.

For Week 3, the strategy accepts more risk on F7 and F8 by allowing broader exploration, while maintaining tight constraints on F1, F3, F4, F5, and F6 where the landscape is more familiar.

---

## Week 4 Predictions

| Function | Current best | W3 query direction | Predicted W4 strategy |
|----------|-------------|-------------------|----------------------|
| F1 | 0.000000 | Near [0.780, 0.780] | Tighten to +/-0.03 around [0.731, 0.733] |
| F2 | 0.611205 | Near best [0.601, 1.0] | If W3 improves, exploit new best; else return to [0.703, 0.927] |
| F3 | -0.034835 | Interior [0.433, 0.758, 0.490] | Exploit if W3 returns near -0.035; tighten further |
| F4 | 0.533858 | Near [0.382, 0.373, 0.445, 0.391] | Continue exploiting positive region |
| F5 | 3511.612 | Peak [0.051, 0.950, 0.983, 0.994] | Fine-tune, kappa 1.0 |
| F6 | -0.714265 | Box boundary [0.765, 0.199, 0.781, 0.740, 0.007] | Query exactly [0.728, 0.155, 0.733, 0.694, 0.056] |
| F7 | 1.771970 | New region [0.164, 0.289, 0.471, 0.074, 0.192, 0.651] | If declined, constrain around W1 best |
| F8 | 9.965293 | High-uncertainty [0.983, 0.951, 0.057, 0.938, 0.905...] | Accept result; if large gain keep exploring |

---

*Reflection written after Week 2 results received and Week 3 queries generated. All pipeline improvements were applied to updated datasets containing original observations plus Week 1 and Week 2 results.*
