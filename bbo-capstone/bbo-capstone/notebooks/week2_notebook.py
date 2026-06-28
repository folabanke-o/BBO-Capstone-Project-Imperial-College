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
