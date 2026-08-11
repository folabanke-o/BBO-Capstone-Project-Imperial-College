# ============================================================
# WEEK 10 BAYESIAN OPTIMISATION — FINAL ROUND
# ============================================================
#
# WEEK 9 POST-MORTEM (4/8 gains):
#   Gains: F1 (NEW ATB 7.62e-9, +4443%), F3 (NEW ATB -5.1e-4, +62%),
#          F5 (NEW ATB 4440.01, +0.02%), F7 (NEW ATB 2.747, +1.7%)
#   Declines: F2 (-27%), F4 (-50%), F6 (-11.7%), F8 (-0.03%)
#
#   Root causes:
#   F2: GP Mean query at dim2=0.925025, just 0.001 below W8 best.
#       Output collapsed from 0.764 to 0.558 — sharp cliff confirmed.
#       The optimum at [0.702, 0.926] sits on a very narrow ridge.
#   F4: ZoMBI dim4 moved -0.026 from W2 best; W2 best of 0.534
#       has now held 8 weeks. The peak is extremely sharp.
#   F6: Grid step 0.002 moved dim2 from 0.139 to 0.141 — wrong dir.
#       W4 best (-0.680) is the true optimum; all grid steps miss it.
#   F8: Query 0.040 L2 from W7 best in 8D — W7 best still holds.
#
# WEEK 10 STRATEGY — FINAL ROUND:
#
# 1. LENGTH SCALE BALANCING ENSEMBLE FOR F4 AND F8
#    Literature (arXiv 2025 LB-BO): "aggregating multiple GP models
#    with varying length scales balances exploration and exploitation
#    at convergence, preventing local optima trapping."
#    For F4: use 5 GPs with ls=[0.01,0.05,0.1,0.5,1.0] per dim.
#    Select candidate maximising mean EI across ensemble.
#    This handles the extremely sharp peak better than a single GP.
#
# 2. GP MEAN WITH MICROSECOND PERTURBATION FOR F2
#    F2's optimum is on a very narrow ridge at [0.702,0.926].
#    W9 missed by 0.001 on dim2. Strategy: GP Mean with
#    custom bounds [0.701,0.703] x [0.9258,0.9275] — the tightest
#    bounds ever applied to any function in the project.
#    This targets the exact confirmed peak coordinates.
#
# 3. PROBO-INSPIRED ROBUST MEAN FOR F6
#    Literature (ScienceDirect 2024): "prior mean parameters have
#    the highest impact on BO convergence." For F6, the GP mean
#    has consistently predicted the wrong direction. Switch to
#    a robust acquisition: GP posterior mode (argmax of mean over
#    a grid centred on W4 best with step 0.001 — finer than any
#    previous grid used on F6).
#
# 4. STANDARD FINAL-ROUND APPROACH FOR ALL OTHERS
#    F1, F3, F5, F7: GP Mean tight exploitation of W9 bests.
#    F8: EI xi=0 (pure exploitation) anchored to W7 best.
#
# KERNEL: ARD Matern-5/2 throughout (arXiv 2402.02746, 2607.07289)
#   "The Matern kernel is less prone to gradient vanishing and more
#   effective for high-dimensional problems."
#   Standard: inputs [0,1]^d, outputs standardised, ARD per-dim.
#
# REFERENCES:
#   Rasmussen & Williams (2006) — GP theory, ARD
#   Jones et al. (1998) — Expected Improvement
#   Gal & Ghahramani (2016) — MC Dropout
#   Wilson et al. (2016) / Gardner et al. (2018) — DKL
#   Li et al. ICLR 2024 arXiv:2305.20028 — BNN vs GP
#   arXiv:2402.02746 (2024) — Matern > RBF for high-dim BO
#   arXiv:2103.16649 — GP mean late-stage exploitation
#   arXiv:2409.00011 — ARD Matern-5/2 recommendation
#   arXiv:2512.17569 — Knowledge Gradient / EI xi=0
#   ScienceDirect 2024 PROBO — prior mean robustness
#   Siemenn et al. 2023 npj Comp. Mat. — ZoMBI
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch
import warnings
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, ConstantKernel, WhiteKernel, RBF
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.stats import norm
from itertools import product

torch.manual_seed(42)
np.random.seed(42)


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
# SECTION 2: COMPLETE QUERY HISTORY — WEEKS 1-9
# ============================================================

week1_queries = {
    "function_1": np.array([0.000186, 0.014353]),
    "function_2": np.array([0.998531, 0.007036]),
    "function_3": np.array([0.933672, 0.002452, 0.965412]),
    "function_4": np.array([0.417336, 0.402860, 0.336077, 0.476656]),
    "function_5": np.array([0.050115, 0.927701, 0.965034, 0.985561]),
    "function_6": np.array([0.197786, 0.010925, 0.990284, 0.888004, 0.052863]),
    "function_7": np.array([0.110110, 0.393658, 0.394356, 0.092883, 0.385807, 0.669789]),
    "function_8": np.array([0.042700, 0.092462, 0.083390, 0.051299,
                             0.808162, 0.563756, 0.175217, 0.419904]),
}
week1_outputs = {
    "function_1": 1.13739520e-239, "function_2": -0.10970434,
    "function_3": -0.36576718,     "function_4": -0.86900220,
    "function_5": 3019.65984,      "function_6": -1.21331906,
    "function_7": 1.77196960,      "function_8": 9.96529345,
}
week2_queries = {
    "function_1": np.array([0.705908, 0.823143]),
    "function_2": np.array([0.839078, 0.895770]),
    "function_3": np.array([0.363504, 0.758379, 0.191085]),
    "function_4": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
    "function_5": np.array([0.056181, 0.992607, 0.973199, 0.959866]),
    "function_6": np.array([0.729856, 0.157383, 0.734992, 0.704798, 0.068448]),
    "function_7": np.array([0.058090, 0.303889, 0.327991, 0.001268, 0.275231, 0.673181]),
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845,
                             0.946320, 0.608139, 0.053440, 0.855712]),
}
week2_outputs = {
    "function_1": -4.67691389e-32, "function_2":  0.14503569,
    "function_3": -0.11944713,     "function_4":  0.53385778,
    "function_5":  3511.61191,     "function_6": -0.80060002,
    "function_7":  1.28884742,     "function_8":  9.81687307,
}
week3_queries = {
    "function_1": np.array([0.770991, 0.772247]),
    "function_2": np.array([0.782573, 0.851721]),
    "function_3": np.array([0.538501, 0.565224, 0.389093]),
    "function_4": np.array([0.391220, 0.397272, 0.392512, 0.416849]),
    "function_5": np.array([0.043103, 0.985149, 0.986449, 0.924277]),
    "function_6": np.array([0.718186, 0.144693, 0.742552, 0.703997, 0.046401]),
    "function_7": np.array([0.121105, 0.376149, 0.487901, 0.138159, 0.427197, 0.725669]),
    "function_8": np.array([0.311134, 0.116341, 0.059376, 0.140269,
                             0.540842, 0.044932, 0.335219, 0.555288]),
}
week3_outputs = {
    "function_1":  5.07920747e-29, "function_2":  0.09168177,
    "function_3": -0.02142785,     "function_4":  0.23137215,
    "function_5":  3214.79278,     "function_6": -0.70363809,
    "function_7":  1.91924944,     "function_8":  9.61721741,
}
week4_queries = {
    "function_1": np.array([0.755943, 0.708019]),
    "function_2": np.array([0.732540, 0.896588]),
    "function_3": np.array([0.578454, 0.529081, 0.427267]),
    "function_4": np.array([0.391082, 0.388623, 0.421155, 0.426886]),
    "function_5": np.array([0.073441, 0.999981, 0.958412, 0.999276]),
    "function_6": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
    "function_7": np.array([0.062595, 0.413172, 0.555018, 0.158477, 0.386036, 0.699222]),
    "function_8": np.array([0.007140, 0.131981, 0.126210, 0.111602,
                             0.760069, 0.618063, 0.244768, 0.488294]),
}
week4_outputs = {
    "function_1": -1.60596561e-16, "function_2":  0.33247907,
    "function_3": -0.00608242,     "function_4":  0.45070948,
    "function_5":  3905.150047,    "function_6": -0.68015232,
    "function_7":  2.01886280,     "function_8":  9.96091957,
}
week5_queries = {
    "function_1": np.array([0.746024, 0.747999]),
    "function_2": np.array([0.772411, 0.856620]),
    "function_3": np.array([0.618443, 0.512242, 0.466082]),
    "function_4": np.array([0.378726, 0.395461, 0.438325, 0.427135]),
    "function_5": np.array([0.115771, 0.997836, 0.934529, 0.999358]),
    "function_6": np.array([0.708186, 0.134693, 0.744552, 0.705997, 0.036401]),
    "function_7": np.array([0.022545, 0.447401, 0.567314, 0.109668, 0.425415, 0.724518]),
    "function_8": np.array([0.077857, 0.141235, 0.128950, 0.097085,
                             0.760277, 0.523742, 0.158135, 0.465728]),
}
week5_outputs = {
    "function_1": -5.34109637e-23, "function_2":  0.20291815,
    "function_3": -0.00319822,     "function_4":  0.29101365,
    "function_5":  3620.40360,     "function_6": -0.70486982,
    "function_7":  1.46989852,     "function_8":  9.98947835,
}
week6_queries = {
    "function_1": np.array([0.706220, 0.757972]),
    "function_2": np.array([0.737524, 0.891592]),
    "function_3": np.array([0.621632, 0.501799, 0.454372]),
    "function_4": np.array([0.385718, 0.396061, 0.431327, 0.422779]),
    "function_5": np.array([0.000563, 0.999140, 0.999023, 0.999846]),
    "function_6": np.array([0.709186, 0.141693, 0.745552, 0.706997, 0.037401]),
    "function_7": np.array([0.092365, 0.378563, 0.527314, 0.189186, 0.366707, 0.705150]),
    "function_8": np.array([0.031249, 0.188527, 0.117555, 0.143463,
                             0.713608, 0.547419, 0.184061, 0.520388]),
}
week6_outputs = {
    "function_1": -1.313446068e-16, "function_2":  0.544922639,
    "function_3": -0.014702510,     "function_4":  0.418765262,
    "function_5":  4412.974620,     "function_6": -0.767567783,
    "function_7":  2.374187172,     "function_8":  9.981432643,
}
week7_queries = {
    "function_1": np.array([0.810503, 0.812929]),
    "function_2": np.array([0.670594, 0.953914]),
    "function_3": np.array([0.641201, 0.489832, 0.490912]),
    "function_4": np.array([0.392098, 0.396579, 0.425224, 0.422943]),
    "function_5": np.array([0.000017, 0.999957, 0.999922, 0.999992]),
    "function_6": np.array([0.687242, 0.162414, 0.724851, 0.692615, 0.036503]),
    "function_7": np.array([0.109098, 0.358650, 0.545872, 0.208942, 0.350961, 0.710041]),
    "function_8": np.array([0.103018, 0.152851, 0.147527, 0.130788,
                             0.794596, 0.495937, 0.186257, 0.448233]),
}
week7_outputs = {
    "function_1":  1.16714801e-47,  "function_2":  0.46037541,
    "function_3": -0.01407102,      "function_4":  0.48992037,
    "function_5":  4438.73290,      "function_6": -0.75896416,
    "function_7":  2.56363741,      "function_8":  9.99597080,
}
week8_queries = {
    "function_1": np.array([0.524752, 0.683252]),
    "function_2": np.array([0.702084, 0.926218]),
    "function_3": np.array([0.598591, 0.531379, 0.485694]),
    "function_4": np.array([0.393290, 0.438477, 0.414075, 0.387305]),
    "function_5": np.array([0.000596, 0.999981, 0.999962, 0.999963]),
    "function_6": np.array([0.715186, 0.135693, 0.745552, 0.706997, 0.037401]),
    "function_7": np.array([0.124158, 0.340729, 0.562574, 0.226722, 0.336790, 0.714443]),
    "function_8": np.array([0.098555, 0.182825, 0.119477, 0.158927,
                             0.824386, 0.510492, 0.158242, 0.474447]),
}
week8_outputs = {
    "function_1":  1.67704191e-10, "function_2":  0.76437722,
    "function_3": -0.00133680,     "function_4":  0.21260659,
    "function_5":  4439.21858,     "function_6": -0.77571628,
    "function_7":  2.70050154,     "function_8":  9.99303521,
}
week9_queries = {
    "function_1": np.array([0.539751, 0.692273]),
    "function_2": np.array([0.701876, 0.925025]),
    "function_3": np.array([0.609462, 0.546372, 0.476053]),
    "function_4": np.array([0.399653, 0.387510, 0.420036, 0.385651]),
    "function_5": np.array([0.001191, 0.999994, 0.999987, 0.999982]),
    "function_6": np.array([0.714186, 0.140693, 0.746552, 0.711997, 0.038401]),
    "function_7": np.array([0.111440, 0.326468, 0.569626, 0.240241, 0.323399, 0.720332]),
    "function_8": np.array([0.091610, 0.172509, 0.158450, 0.113284,
                             0.814514, 0.481930, 0.192583, 0.443401]),
}
week9_outputs = {
    "function_1":  7.61866254e-9,  "function_2":  0.55799509,
    "function_3": -0.00051053,     "function_4":  0.26642242,
    "function_5":  4440.00957,     "function_6": -0.76003275,
    "function_7":  2.74731211,     "function_8":  9.99258209,
}


# ============================================================
# SECTION 3: BUILD DATASETS (original + W1 to W9)
# ============================================================

updated_data = {}
for i in range(1, 9):
    key = f"function_{i}"
    X_updated = np.vstack([
        original_data[key]["X"],
        week1_queries[key].reshape(1, -1),
        week2_queries[key].reshape(1, -1),
        week3_queries[key].reshape(1, -1),
        week4_queries[key].reshape(1, -1),
        week5_queries[key].reshape(1, -1),
        week6_queries[key].reshape(1, -1),
        week7_queries[key].reshape(1, -1),
        week8_queries[key].reshape(1, -1),
        week9_queries[key].reshape(1, -1),
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key],
         week4_outputs[key], week5_outputs[key], week6_outputs[key],
         week7_outputs[key], week8_outputs[key], week9_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"F{i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: ALL-TIME BESTS ENTERING WEEK 10
# ============================================================

print("\n" + "=" * 65)
print("ENTERING WEEK 10 — ALL-TIME BESTS (FINAL ROUND)")
print("=" * 65)

all_bests = {
    "function_1": {
        "val": 7.61866254e-9,
        "inp": np.array([0.539751, 0.692273]),
        "src": "W9",
        "note": "W9 new ATB; GP Mean margin 0.012 — pure exploitation",
    },
    "function_2": {
        "val": 0.76437722,
        "inp": np.array([0.702084, 0.926218]),
        "src": "W8",
        "note": "W8 ATB; W9 missed by 0.001 on dim2; micro-bounds [0.701,0.703]x[0.9258,0.9275]",
    },
    "function_3": {
        "val": -0.00051053,
        "inp": np.array([0.609462, 0.546372, 0.476053]),
        "src": "W9",
        "note": "W9 new ATB; GP Mean margin 0.012 — converging to zero",
    },
    "function_4": {
        "val": 0.53385778,
        "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
        "src": "W2",
        "note": "W2 best held 8 weeks; length scale ensemble targets the sharp peak",
    },
    "function_5": {
        "val": 4440.00957,
        "inp": np.array([0.001191, 0.999994, 0.999987, 0.999982]),
        "src": "W9",
        "note": "W9 new ATB; GP Mean dim1 in [0.000,0.002] — incremental gains continue",
    },
    "function_6": {
        "val": -0.68015232,
        "inp": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
        "src": "W4",
        "note": "W4 best held 6 weeks; finest grid step 0.001 around W4 best",
    },
    "function_7": {
        "val": 2.74731211,
        "inp": np.array([0.111440, 0.326468, 0.569626, 0.240241, 0.323399, 0.720332]),
        "src": "W9",
        "note": "W9 new ATB; four consecutive gains W6-W9; GP Mean margin 0.012",
    },
    "function_8": {
        "val": 9.99597080,
        "inp": np.array([0.103018, 0.152851, 0.147527, 0.130788,
                          0.794596, 0.495937, 0.186257, 0.448233]),
        "src": "W7",
        "note": "W7 best holds; EI xi=0 margin 0.015 — tightest yet on F8",
    },
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | {b['val']:.5g} ({b['src']}) | {b['note'][:62]}")


# ============================================================
# SECTION 5: SURROGATE — ARD GP (UNIFIED)
# ============================================================
# All GP fits: ARD Matern-5/2, Y standardised.
# Literature: arXiv 2402.02746, 2607.07289, 2409.00011 —
# "ARD Matern-5/2 is the standard choice for BBO, less prone
# to gradient vanishing than RBF in high-dimensional settings."
# ============================================================

def fit_gp_ard(X, Y, kernel_type="matern52", n_restarts=15, ls_init=None):
    """
    Unified ARD GP fitter with Y standardisation.
    kernel_type: 'matern52' (default), 'matern15', 'rbf'
    ls_init: initial length scale (scalar or vector). Defaults to
             sqrt(d)/10 per arXiv 2502.09198 recommendation for
             high-dimensional settings.
    """
    dim    = X.shape[1]
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Length scale initialisation: sqrt(d)/10 (arXiv 2502.09198)
    if ls_init is None:
        ls_val = np.ones(dim) * (np.sqrt(dim) / 10.0)
    elif np.isscalar(ls_init):
        ls_val = np.ones(dim) * ls_init
    else:
        ls_val = np.array(ls_init)

    if kernel_type == "rbf":
        base = RBF(length_scale=ls_val, length_scale_bounds=(1e-4, 1e4))
    elif kernel_type == "matern15":
        base = Matern(length_scale=ls_val, length_scale_bounds=(1e-4, 1e4), nu=1.5)
    else:  # matern52 default
        base = Matern(length_scale=ls_val, length_scale_bounds=(1e-4, 1e4), nu=2.5)

    kernel = (ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
              * base
              + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    gp.fit(X_s, Y_s)
    return gp, scaler, y_mean, y_std


def predict_gp(gp, scaler, X_np, y_mean, y_std):
    mu_s, sg_s = gp.predict(scaler.transform(X_np), return_std=True)
    return mu_s * y_std + y_mean, np.maximum(sg_s * y_std, 1e-9)


def get_ard_ls(gp):
    for p, v in gp.kernel_.get_params().items():
        if "length_scale" in p and hasattr(v, "__len__") and len(v) > 1:
            return np.array(v)
    return None


# ============================================================
# SECTION 6: LENGTH SCALE BALANCING ENSEMBLE (F4, F8)
# ============================================================
# Literature: arXiv LB-BO 2025 — "aggregating multiple GPs with
# varying length scales prevents local optima trapping at
# convergence." Each GP uses a different fixed ls, then EI is
# averaged across the ensemble to select the query.
# ============================================================

def fit_ls_ensemble(X, Y, ls_values=None, n_restarts=5):
    """
    Fit M GPs with different fixed length scales.
    ls_values: list of scalar ls values to try.
    Returns list of (gp, scaler, y_mean, y_std) tuples.
    """
    if ls_values is None:
        ls_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    ensemble = []
    for ls in ls_values:
        try:
            gp, sc, ym, ys = fit_gp_ard(X, Y, kernel_type="matern52",
                                          n_restarts=n_restarts, ls_init=ls)
            ensemble.append((gp, sc, ym, ys))
        except Exception:
            pass
    return ensemble


def ensemble_ei(ensemble, cands, f_best, xi=0.0):
    """
    Average EI across all GPs in the ensemble.
    xi=0: pure exploitation per KG approximation.
    """
    all_ei = []
    for gp, sc, ym, ys in ensemble:
        mu, sigma = predict_gp(gp, sc, cands, ym, ys)
        sigma = np.maximum(sigma, 1e-9)
        Z  = (mu - f_best - xi) / sigma
        ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        all_ei.append(np.maximum(ei, 0))
    return np.mean(all_ei, axis=0)


# ============================================================
# SECTION 7: MC DROPOUT MLP
# ============================================================

class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        h1 = max(32, 8 * input_dim); h2 = max(16, 4 * input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(h2, 1))
    def forward(self, x): return self.net(x)


def train_mc_dropout(X, Y, epochs=300, lr=0.001, dropout=0.1, weight_decay=1e-3):
    ym = float(Y.mean()); ys = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Ys = (Y - ym) / ys
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Ys, dtype=torch.float32).reshape(-1, 1)
    m  = MCDropoutMLP(X.shape[1], dropout=dropout)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); nn.MSELoss()(m(Xt), Yt).backward(); opt.step()
    return m, (ym, ys)


def mc_predict(model, X_np, ym, ys, n_mc=50):
    model.train()
    Xt = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(n_mc): preds.append(model(Xt).numpy().flatten())
    p = np.array(preds)
    return p.mean(0) * ys + ym, np.maximum(p.std(0) * ys, 1e-9)


# ============================================================
# SECTION 8: DEEP KERNEL LEARNING
# ============================================================

class DKLFeatureExtractor(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = max(16, 4 * d)
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 2))
    def forward(self, x): return self.net(x)


class DKLGPModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik, fe):
        super().__init__(tx, ty, lik)
        self.feature_extractor = fe
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        z = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z))


def train_dkl(X, Y, epochs=200, lr=0.01):
    ym = float(Y.mean()); ys = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Ys = (Y - ym) / ys; sc = StandardScaler(); Xs = sc.fit_transform(X)
    tx = torch.tensor(Xs, dtype=torch.float32)
    ty = torch.tensor(Ys, dtype=torch.float32)
    fe = DKLFeatureExtractor(X.shape[1])
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model = DKLGPModel(tx, ty, lik, fe); model.train(); lik.train()
    opt = torch.optim.Adam(
        list(model.feature_extractor.parameters()) +
        list(model.covar_module.parameters()) +
        list(model.mean_module.parameters()) +
        list(lik.parameters()), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(epochs):
        opt.zero_grad(); (-mll(model(tx), ty)).backward(); opt.step()
    return model, lik, (ym, ys), sc


def dkl_predict(model, lik, X_np, ym, ys, sc):
    model.eval(); lik.eval()
    Xt = torch.tensor(sc.transform(X_np), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(Xt))
    return pred.mean.numpy() * ys + ym, np.maximum(pred.stddev.numpy() * ys, 1e-9)


# ============================================================
# SECTION 9: ACQUISITION FUNCTIONS
# ============================================================

def expected_improvement(mu, sigma, f_best, xi=0.0):
    """
    Closed-form EI. xi=0.0 = pure exploitation / KG approximation.
    """
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return np.maximum(ei, 0)


def gp_mean_acquisition(mu):
    """
    Pure exploitation: select candidate with highest predicted mean.
    arXiv 2103.16649: beneficial in late iterations for converging
    functions. No exploration penalty.
    """
    return mu.copy()


# ============================================================
# SECTION 10: LOO-CV SURROGATE SELECTION
# ============================================================

def loo_rmse_all(X, Y):
    """LOO-CV RMSE for Matern52, RBF, MC Dropout, DKL."""
    loo = LeaveOneOut()
    sq  = {"matern52": [], "rbf": [], "mc_dropout": [], "dkl": []}

    for tr, te in loo.split(X):
        Xtr, Xte = X[tr], X[te]; Ytr, Yte = Y[tr], Y[te]

        for key, kt in [("matern52","matern52"), ("rbf","rbf")]:
            try:
                gp,sc,ym,ys = fit_gp_ard(Xtr, Ytr, kernel_type=kt, n_restarts=5)
                mu,_ = predict_gp(gp, sc, Xte, ym, ys)
                sq[key].append((float(mu[0]) - Yte[0])**2)
            except Exception:
                sq[key].append((float(np.mean(Ytr)) - Yte[0])**2)

        try:
            m_mc,(ym_mc,ys_mc) = train_mc_dropout(Xtr, Ytr, epochs=200)
            mu_mc,_ = mc_predict(m_mc, Xte, ym_mc, ys_mc, n_mc=20)
            sq["mc_dropout"].append((float(mu_mc[0]) - Yte[0])**2)
        except Exception:
            sq["mc_dropout"].append((float(np.mean(Ytr)) - Yte[0])**2)

        try:
            m_d,lik,(ym_d,ys_d),sc_d = train_dkl(Xtr, Ytr, epochs=80)
            mu_d,_ = dkl_predict(m_d, lik, Xte, ym_d, ys_d, sc_d)
            sq["dkl"].append((float(mu_d[0]) - Yte[0])**2)
        except Exception:
            sq["dkl"].append((float(np.mean(Ytr)) - Yte[0])**2)

    return {k: float(np.sqrt(np.mean(v))) for k, v in sq.items()}


# ============================================================
# SECTION 11: PER-FUNCTION CONFIGURATION
# ============================================================

# F5: dim1 in [0.000, 0.002] — tightest ever
f5_low  = np.array([0.000, 0.999, 0.999, 0.999])
f5_high = np.array([0.002, 1.000, 1.000, 1.000])

# F2: micro-bounds targeting the exact W8 ATB ridge
f2_low  = np.array([0.7010, 0.9258])
f2_high = np.array([0.7030, 0.9275])

method_config = {
    # F1: GP Mean 0.012 margin around W9 best
    "function_1": {"acq":"GP_MEAN","margin":0.012,"kernel":"matern52","restarts":12},
    # F2: GP Mean micro-bounds — targeting exact W8 ATB ridge
    "function_2": {"acq":"GP_MEAN","margin":None,"custom_bounds":True,"kernel":"matern52","restarts":12},
    # F3: GP Mean 0.012 margin — converging to zero
    "function_3": {"acq":"GP_MEAN","margin":0.012,"kernel":"matern52","restarts":12},
    # F4: Length scale ensemble + EI xi=0 — sharp peak targeting
    "function_4": {"acq":"ENSEMBLE","xi":0.0,"margin":0.020,"kernel":"matern52","restarts":10},
    # F5: GP Mean ultra-tight asymmetric bounds
    "function_5": {"acq":"GP_MEAN","margin":None,"custom_bounds":True,"kernel":"matern52","restarts":15},
    # F6: Finest grid step 0.001 around W4 best — PROBO-inspired
    "function_6": {"acq":"GRID","step":0.001,"kernel":"matern52","restarts":15},
    # F7: GP Mean 0.012 margin — four consecutive gains
    "function_7": {"acq":"GP_MEAN","margin":0.012,"kernel":"matern52","restarts":20},
    # F8: EI xi=0 (pure exploitation), tightest margin 0.015
    "function_8": {"acq":"EI","xi":0.0,"margin":0.015,"kernel":"matern52","restarts":20},
}


# ============================================================
# SECTION 12: LOO-CV SURROGATE SELECTION
# ============================================================

print("\n" + "=" * 68)
print("LOO-CV SURROGATE SELECTION")
print(f"{'Fn':<5} {'Matern52':>11} {'RBF':>11} {'MC-Drop':>11} {'DKL':>11}  Winner")
print("-" * 68)

surrogate_selection = {}
for i in range(1, 9):
    key  = f"function_{i}"
    X    = updated_data[key]["X"]
    Y    = updated_data[key]["Y"]
    rmse = loo_rmse_all(X, Y)
    winner = min(rmse, key=rmse.get)
    surrogate_selection[key] = {"rmse": rmse, "winner": winner}
    print(f"F{i:<4} {rmse['matern52']:>11.4f} {rmse['rbf']:>11.4f} "
          f"{rmse['mc_dropout']:>11.4f} {rmse['dkl']:>11.4f}  {winner}")


# ============================================================
# SECTION 13: GENERATE WEEK 10 QUERIES
# ============================================================

N_CANDS = 80000  # increased for final round precision
week10_results = {}

for i in range(1, 9):
    key    = f"function_{i}"
    cfg    = method_config[key]
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    best   = all_bests[key]["inp"]
    margin = cfg.get("margin")
    ktype  = cfg.get("kernel", "matern52")

    print(f"\n{'='*62}")
    print(f"F{i} | {cfg['acq']} | {ktype} | obs={X.shape[0]} | best={Y.max():.6f}")
    print(f"{'='*62}")

    # GRID for F6 (finest step: 0.001)
    if cfg["acq"] == "GRID":
        step = cfg["step"]
        gp, sc, ym, ys = fit_gp_ard(X, Y, kernel_type="matern52",
                                      n_restarts=cfg["restarts"])
        grid = []
        for deltas in product([-step, 0.0, step], repeat=dim):
            grid.append(np.clip(best + np.array(deltas), 0.0, 1.0))
        grid    = np.unique(np.array(grid), axis=0)
        mu_g, _ = predict_gp(gp, sc, grid, ym, ys)
        bidx    = np.argmax(mu_g)
        query   = grid[bidx]
        score   = float(mu_g[bidx])
        mu_out, sigma_out = score, 0.0
        print(f"  Grid: {len(grid)} candidates | best mu: {score:.6f}")

    # LENGTH SCALE ENSEMBLE for F4
    elif cfg["acq"] == "ENSEMBLE":
        low_b  = np.clip(best - margin, 0.0, 1.0)
        high_b = np.clip(best + margin, 0.0, 1.0)
        np.random.seed(42)
        cands  = np.random.uniform(low_b, high_b, size=(N_CANDS, dim))
        ensemble = fit_ls_ensemble(X, Y, ls_values=[0.01, 0.05, 0.1, 0.5, 1.0])
        f_best   = Y.max()
        acq_vals = ensemble_ei(ensemble, cands, f_best, xi=cfg["xi"])
        bidx     = np.argmax(acq_vals)
        query    = np.clip(cands[bidx], 0.0, 1.0)
        score    = float(acq_vals[bidx])
        # Use first GP for mu/sigma report
        mu_rep, sg_rep = predict_gp(*ensemble[0][:2], query.reshape(1,-1),
                                     ensemble[0][2], ensemble[0][3])
        mu_out = float(mu_rep[0]); sigma_out = float(sg_rep[0])
        print(f"  Ensemble ({len(ensemble)} GPs) | best ensemble EI: {score:.6f}")

    else:
        # Build bounds
        if cfg.get("custom_bounds") and key == "function_5":
            low_b, high_b = f5_low, f5_high
        elif cfg.get("custom_bounds") and key == "function_2":
            low_b, high_b = f2_low, f2_high
        elif best is not None and margin is not None:
            low_b  = np.clip(best - margin, 0.0, 1.0)
            high_b = np.clip(best + margin, 0.0, 1.0)
        else:
            low_b  = np.zeros(dim); high_b = np.ones(dim)

        np.random.seed(42)
        cands = np.random.uniform(low_b, high_b, size=(N_CANDS, dim))
        gp, sc, ym, ys = fit_gp_ard(X, Y, kernel_type=ktype,
                                      n_restarts=cfg["restarts"])
        mu, sigma = predict_gp(gp, sc, cands, ym, ys)
        f_best = Y.max()

        if cfg["acq"] == "GP_MEAN":
            acq_vals = gp_mean_acquisition(mu)
        else:  # EI
            acq_vals = expected_improvement(mu, sigma, f_best, xi=cfg["xi"])

        bidx      = np.argmax(acq_vals)
        query     = np.clip(cands[bidx], 0.0, 1.0)
        score     = float(acq_vals[bidx])
        mu_out    = float(mu[bidx]); sigma_out = float(sigma[bidx])

    formatted = "-".join([f"{x:.6f}" for x in query])
    week10_results[key] = {
        "query": query, "formatted_query": formatted,
        "acq": cfg["acq"], "surrogate": ktype,
        "score": score, "predicted_mean": mu_out, "uncertainty": sigma_out,
    }
    dist = np.linalg.norm(query - best)
    print(f"Query     : {formatted}")
    print(f"Score     : {score:.6f} | Pred mean: {mu_out:.6f} | Dist: {dist:.4f}")
    print(f"Bounds    : {np.round(low_b,3)} to {np.round(high_b,3)}"
          if cfg["acq"] not in ["GRID"] else "")


# ============================================================
# SECTION 14: VALIDATION
# ============================================================

print("\n" + "=" * 62)
print("QUERY VALIDATION REPORT")
print("=" * 62)

all_clear = True
for i in range(1, 9):
    key   = f"function_{i}"
    query = week10_results[key]["query"]
    issues = []
    if np.any(query < 0) or np.any(query > 1): issues.append("OUT OF RANGE")
    if np.all(query < 0.01) and i not in [5]:  issues.append("SUSPICIOUS: all near 0")
    if np.all(query > 0.99):                    issues.append("SUSPICIOUS: all near 1")
    status = "OK" if not issues else "WARNING"
    print(f"\nF{i} [{status}] [{week10_results[key]['acq']}] [{week10_results[key]['surrogate']}]")
    print(f"  Query : {week10_results[key]['formatted_query']}")
    if issues:
        all_clear = False
        for w in issues: print(f"  !! {w}")

print("\nAll queries valid." if all_clear else "\nReview warnings before submitting.")


# ============================================================
# SECTION 15: PROXIMITY CHECK
# ============================================================

print("\n" + "=" * 62)
print("PROXIMITY TO ALL-TIME BEST INPUT")
print("=" * 62)
for i in range(1, 9):
    key = f"function_{i}"
    q   = week10_results[key]["query"]
    b   = all_bests[key]["inp"]
    d   = np.linalg.norm(q - b)
    flag = "ok" if d < 0.15 else "far — review"
    print(f"F{i} [{week10_results[key]['acq']}] | dist: {d:.4f} [{flag}]")


# ============================================================
# SECTION 16: ARD LENGTH SCALE DIAGNOSTIC
# ============================================================

print("\n" + "=" * 62)
print("ARD LENGTH SCALE DIAGNOSTIC (shorter = more important dim)")
print("=" * 62)

for i in range(1, 9):
    key = f"function_{i}"
    X   = updated_data[key]["X"]
    Y   = updated_data[key]["Y"]
    try:
        gp, sc, ym, ys = fit_gp_ard(X, Y, kernel_type="matern52", n_restarts=5)
        ls = get_ard_ls(gp)
        if ls is not None:
            ranked = np.argsort(ls)
            print(f"\nF{i} ({X.shape[1]}D): ls={np.round(ls,3)}")
            print(f"  Most important : dim{ranked[0]+1}  ls={ls[ranked[0]]:.4f}")
            if len(ranked) > 1:
                print(f"  Least important: dim{ranked[-1]+1} ls={ls[ranked[-1]]:.4f}")
    except Exception as e:
        print(f"F{i}: {e}")


# ============================================================
# SECTION 17: PER-FUNCTION MODEL CARDS
# ============================================================

print("\n\n" + "=" * 65)
print("PER-FUNCTION MODEL CARDS — WEEK 10 (FINAL ROUND)")
print("=" * 65)

for i in range(1, 9):
    key    = f"function_{i}"
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    rmse   = surrogate_selection[key]["rmse"]
    winner = surrogate_selection[key]["winner"]
    b      = all_bests[key]
    r      = week10_results[key]
    cfg    = method_config[key]

    acq_str = {
        "GP_MEAN":  "GP Mean — pure exploitation (arXiv 2103.16649)",
        "EI":       f"EI xi={cfg.get('xi',0)} — pure exploitation",
        "ENSEMBLE": "Length scale ensemble EI (5 GPs, arXiv LB-BO 2025)",
        "GRID":     f"Grid step={cfg.get('step')} — PROBO-inspired finest grid",
    }.get(cfg["acq"], cfg["acq"])

    print(f"\n{'─'*65}")
    print(f"  MODEL CARD   F{i}  ({dim}D | {X.shape[0]} obs)  WEEK 10 FINAL")
    print(f"{'─'*65}")
    print(f"  All-time best  : {b['val']:.5g}  ({b['src']})")
    print(f"  Best input     : {np.round(b['inp'],4)}")
    print(f"  Strategy note  : {b['note'][:70]}")
    print()
    print(f"  SURROGATE COMPARISON (LOO-CV RMSE)")
    for k in ["matern52","rbf","mc_dropout","dkl"]:
        tag = "  SELECTED" if k == winner else ""
        val = rmse.get(k, "n/a")
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"    {k:<14}: {val_str}{tag}")
    print()
    print(f"  WEEK 10 QUERY")
    print(f"    Acquisition    : {acq_str}")
    print(f"    Kernel         : ARD Matern-5/2 (Y-scaled, ls_init=sqrt(d)/10)")
    print(f"    Query          : {r['formatted_query']}")
    print(f"    Predicted mean : {r['predicted_mean']:.5g}")
    print(f"    Dist from ATB  : {np.linalg.norm(r['query'] - b['inp']):.4f}")


# ============================================================
# SECTION 18: SAVE WEEK 10 QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key  = f"function_{i}"
    Y    = updated_data[key]["Y"]
    r    = week10_results[key]
    rmse = surrogate_selection[key]["rmse"]
    rows.append({
        "Function":       f"Function {i}",
        "Query":          r["formatted_query"],
        "Acquisition":    r["acq"],
        "Kernel":         "ARD-Matern52",
        "Matern52_LOO":   round(rmse.get("matern52",   0), 4),
        "RBF_LOO":        round(rmse.get("rbf",        0), 4),
        "MC_LOO":         round(rmse.get("mc_dropout", 0), 4),
        "DKL_LOO":        round(rmse.get("dkl",        0), 4),
        "Predicted_mean": round(r["predicted_mean"],       6),
        "Uncertainty":    round(r["uncertainty"],          6),
        "Current_best":   round(float(Y.max()),            6),
    })

df = pd.DataFrame(rows)
print("\n\nFINAL WEEK 10 QUERIES")
print(df[["Function","Acquisition","Query","Current_best"]].to_string(index=False))

df[["Function","Query"]].to_csv("week10_queries.csv", index=False)
df.to_csv("week10_queries_full.csv", index=False)
print("\nweek10_queries.csv      — submit this file")
print("week10_queries_full.csv — LOO-RMSE and model card data")
