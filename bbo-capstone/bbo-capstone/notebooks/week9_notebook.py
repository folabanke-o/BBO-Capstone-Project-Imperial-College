# ============================================================
# WEEK 9 BAYESIAN OPTIMISATION — GP MEAN + ARD MATERN-5/2
# ============================================================
#
# WEEK 8 POST-MORTEM (5/8 gains — best round of the project):
#   Gains: F1 (ZoMBI found new region, 1.68e-10),
#          F2 (NEW BEST 0.764, +25%), F3 (NEW BEST -0.00134, +58%),
#          F5 (+0.01%), F7 (NEW BEST 2.701, +5.3%)
#   Declines: F4 (pinned ZoMBI missed; W2 best holds),
#             F6 (grid step moved wrong direction),
#             F8 (margin 0.030 too wide in 8D)
#
# WEEK 9 STRATEGY:
#
# 1. GP MEAN ACQUISITION (late-stage pure exploitation)
#    Reference: arXiv 2103.16649 — "Using an occasional GP mean
#    acquisition is slightly beneficial in late iterations for
#    unimodal functions where additional exploitation helps."
#    Applied to F1, F2, F3, F5, F7 — all just set new ATBs.
#    GP mean selects the candidate with the highest surrogate
#    prediction with NO exploration penalty.
#
# 2. ARD MATERN-5/2 AS DEFAULT KERNEL
#    Reference: arXiv 2409.00011 — "After testing different
#    kernels, the ARD Matern-5/2 kernel was chosen."
#    More flexible than Matern-2.5 for sharp near-optimum features.
#
# 3. PINNED ZoMBI FOR F4 (refined from W8)
#    W8 ZoMBI missed F4 peak because W2 best was not in top-5.
#    Fix: force the W2 best input into memory regardless of rank.
#    EI with xi=0 (pure exploitation / KG approximation).
#    Reference: arXiv 2512.17569 — Knowledge Gradient.
#
# 4. TIGHTER GRID FOR F6, TIGHTER EI FOR F8
#    F6: grid step 0.002 (tighter than W8 0.003)
#    F8: EI xi=0.001, margin 0.020 (tighter than W8 0.030)
#
# REFERENCES:
#   Rasmussen & Williams (2006) — ARD GP theory
#   Jones et al. (1998) — Expected Improvement
#   Gal & Ghahramani (2016) — MC Dropout
#   Wilson et al. (2016) / Gardner et al. (2018) — DKL
#   Li et al. ICLR 2024 arXiv:2305.20028 — BNN vs GP
#   arXiv:2103.16649 — GP mean late-stage exploitation
#   arXiv:2409.00011 — ARD Matern-5/2 recommendation
#   arXiv:2512.17569 — Knowledge Gradient / EI xi=0
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
    RBF, ConstantKernel, WhiteKernel, Matern
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
# SECTION 2: COMPLETE QUERY HISTORY — WEEKS 1-8
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
    "function_8": np.array([0.017074, 0.091604, 0.305973, 0.115845, 0.946320, 0.608139, 0.053440, 0.855712]),
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
    "function_8": np.array([0.311134, 0.116341, 0.059376, 0.140269, 0.540842, 0.044932, 0.335219, 0.555288]),
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
    "function_8": np.array([0.007140, 0.131981, 0.126210, 0.111602, 0.760069, 0.618063, 0.244768, 0.488294]),
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
    "function_8": np.array([0.077857, 0.141235, 0.128950, 0.097085, 0.760277, 0.523742, 0.158135, 0.465728]),
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
    "function_8": np.array([0.031249, 0.188527, 0.117555, 0.143463, 0.713608, 0.547419, 0.184061, 0.520388]),
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
    "function_8": np.array([0.103018, 0.152851, 0.147527, 0.130788, 0.794596, 0.495937, 0.186257, 0.448233]),
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
    "function_8": np.array([0.098555, 0.182825, 0.119477, 0.158927, 0.824386, 0.510492, 0.158242, 0.474447]),
}
week8_outputs = {
    "function_1":  1.67704191e-10, "function_2":  0.76437722,
    "function_3": -0.00133680,     "function_4":  0.21260659,
    "function_5":  4439.21858,     "function_6": -0.77571628,
    "function_7":  2.70050154,     "function_8":  9.99303521,
}


# ============================================================
# SECTION 3: BUILD DATASETS (original + W1 to W8)
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
    ])
    Y_updated = np.append(
        original_data[key]["Y"],
        [week1_outputs[key], week2_outputs[key], week3_outputs[key],
         week4_outputs[key], week5_outputs[key], week6_outputs[key],
         week7_outputs[key], week8_outputs[key]]
    )
    updated_data[key] = {"X": X_updated, "Y": Y_updated}
    best_idx = np.argmax(Y_updated)
    print(f"F{i} | obs: {X_updated.shape[0]} | best: {Y_updated[best_idx]:.6f} "
          f"at {np.round(X_updated[best_idx], 4)}")


# ============================================================
# SECTION 4: ALL-TIME BESTS ENTERING WEEK 9
# ============================================================

print("\n" + "=" * 65)
print("ENTERING WEEK 9 — ALL-TIME BESTS")
print("=" * 65)

all_bests = {
    "function_1": {
        "val": 1.67704191e-10, "src": "W8",
        "inp": np.array([0.524752, 0.683252]),
        "note": "W8 ZoMBI found new region; exploit with GP mean margin 0.015",
    },
    "function_2": {
        "val": 0.76437722, "src": "W8",
        "inp": np.array([0.702084, 0.926218]),
        "note": "W8 +25%; GP mean ultra-tight custom bounds",
    },
    "function_3": {
        "val": -0.00133680, "src": "W8",
        "inp": np.array([0.598591, 0.531379, 0.485694]),
        "note": "W8 +58%; GP mean margin 0.015",
    },
    "function_4": {
        "val": 0.53385778, "src": "W2",
        "inp": np.array([0.403695, 0.397605, 0.413333, 0.411576]),
        "note": "W2 best held 7 weeks; pinned ZoMBI forces W2 into memory",
    },
    "function_5": {
        "val": 4439.21858, "src": "W8",
        "inp": np.array([0.000596, 0.999981, 0.999962, 0.999963]),
        "note": "W8 new best; dim1 near-zero confirmed; GP mean ultra-tight",
    },
    "function_6": {
        "val": -0.68015232, "src": "W4",
        "inp": np.array([0.712186, 0.138693, 0.748552, 0.709997, 0.040401]),
        "note": "W4 best holds; tighter grid step 0.002",
    },
    "function_7": {
        "val": 2.70050154, "src": "W8",
        "inp": np.array([0.124158, 0.340729, 0.562574, 0.226722, 0.336790, 0.714443]),
        "note": "Three consecutive gains W6-W8; GP mean margin 0.015",
    },
    "function_8": {
        "val": 9.99597080, "src": "W7",
        "inp": np.array([0.103018, 0.152851, 0.147527, 0.130788,
                          0.794596, 0.495937, 0.186257, 0.448233]),
        "note": "W7 still best; EI xi=0.001 margin 0.020",
    },
}

for i in range(1, 9):
    key = f"function_{i}"
    b = all_bests[key]
    print(f"F{i} | {b['val']:.6g} ({b['src']}) | {b['note'][:60]}")


# ============================================================
# SECTION 5: SURROGATE — ARD GP (UNIFIED FITTER)
# ============================================================
# All variants use:
#   ARD: per-dimension length_scale vector (colleague suggestion)
#   Y standardisation: zero mean, unit std (colleague suggestion)
#   Matern-5/2 as default (arXiv 2409.00011 recommendation)
# ============================================================

def fit_gp_ard(X, Y, kernel_type="matern52", n_restarts=15):
    """
    Unified ARD GP fitter with Y scaling.
    kernel_type options:
      "matern52" — ARD Matern-5/2 (default, arXiv 2409.00011)
      "matern15" — ARD Matern-1.5 (sharp peaks, F4)
      "rbf"      — ARD RBF (infinite smoothness)
    Returns gp, scaler, y_mean, y_std
    """
    dim    = X.shape[1]
    y_mean = float(Y.mean())
    y_std  = float(Y.std()) if Y.std() > 1e-12 else 1.0
    Y_s    = (Y - y_mean) / y_std
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    if kernel_type == "rbf":
        base = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-4, 1e4))
    elif kernel_type == "matern15":
        base = Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-4, 1e4), nu=1.5)
    else:  # matern52 (default)
        base = Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-4, 1e4), nu=2.5)

    kernel = (ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))
              * base
              + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    gp.fit(X_s, Y_s)
    return gp, scaler, y_mean, y_std


def predict_gp(gp, scaler, X_np, y_mean, y_std):
    """Predict in original Y scale."""
    mu_s, sg_s = gp.predict(scaler.transform(X_np), return_std=True)
    return mu_s * y_std + y_mean, np.maximum(sg_s * y_std, 1e-9)


def get_ard_lengthscales(gp):
    """Extract per-dim length scales after fitting."""
    for p, v in gp.kernel_.get_params().items():
        if "length_scale" in p and hasattr(v, "__len__") and len(v) > 1:
            return np.array(v)
    return None


# ============================================================
# SECTION 6: MC DROPOUT MLP
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
# SECTION 7: DEEP KERNEL LEARNING
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
# SECTION 8: ACQUISITION FUNCTIONS
# ============================================================

def expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Closed-form EI (Jones et al. 1998).
    xi=0.0: pure exploitation / KG approximation.
    Compatible with any surrogate returning (mu, sigma).
    """
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def gp_mean_acquisition(mu):
    """
    GP mean acquisition: pure exploitation.
    arXiv 2103.16649: beneficial in late iterations for unimodal
    functions. Selects the candidate with the highest predicted
    mean value — no exploration penalty whatsoever.
    """
    return mu.copy()


def upper_confidence_bound(mu, sigma, kappa=0.8):
    return mu + kappa * sigma


# ============================================================
# SECTION 9: PINNED ZoMBI FOR F4
# ============================================================
# Siemenn et al. 2023 npj Comp. Mat.: ZoMBI zooms bounds using
# top-m best observations and prunes low-performing history.
# Pinned variant: force a specific input (W2 best for F4)
# into the memory set regardless of its rank.
# This prevents the confirmed sharp peak from being pruned out.
# ============================================================

def pinned_zombi(X, Y, pin_inp, m=5, kernel_type="matern15"):
    """
    Pinned ZoMBI: top-(m-1) by Y + 1 pinned best input.
    pin_inp: input to force into memory (e.g. W2 best for F4).
    Returns gp, scaler, y_mean, y_std, low_b, high_b, f_best_mem.
    """
    m = min(m, len(Y))
    dists   = np.linalg.norm(X - np.array(pin_inp), axis=1)
    pin_idx = int(np.argmin(dists))
    mask    = np.ones(len(Y), dtype=bool); mask[pin_idx] = False
    top_idx = np.argsort(Y[mask])[-(m-1):]
    orig_idx = np.where(mask)[0][top_idx]
    mem_idx  = np.unique(np.append(orig_idx, pin_idx))
    X_mem = X[mem_idx]; Y_mem = Y[mem_idx]
    dim   = X.shape[1]
    low_b = np.zeros(dim); high_b = np.ones(dim)
    for d in range(dim):
        lo  = X_mem[:, d].min(); hi = X_mem[:, d].max()
        buf = max((hi - lo) * 0.20, 0.03)
        low_b[d]  = max(0.0, lo - buf)
        high_b[d] = min(1.0, hi + buf)
    gp, sc, ym, ys = fit_gp_ard(X_mem, Y_mem, kernel_type=kernel_type, n_restarts=10)
    return gp, sc, ym, ys, low_b, high_b, float(Y_mem.max())


# ============================================================
# SECTION 10: LOO-CV SURROGATE SELECTION
# ============================================================

def loo_rmse_all(X, Y):
    """LOO-CV RMSE for ARD-Matern52, ARD-RBF, MC Dropout, DKL."""
    loo = LeaveOneOut()
    sq  = {"matern52": [], "rbf": [], "mc_dropout": [], "dkl": []}

    for tr, te in loo.split(X):
        Xtr, Xte = X[tr], X[te]; Ytr, Yte = Y[tr], Y[te]

        for key, kt in [("matern52", "matern52"), ("rbf", "rbf")]:
            try:
                gp, sc, ym, ys = fit_gp_ard(Xtr, Ytr, kernel_type=kt, n_restarts=5)
                mu, _ = predict_gp(gp, sc, Xte, ym, ys)
                sq[key].append((float(mu[0]) - Yte[0]) ** 2)
            except Exception:
                sq[key].append((float(np.mean(Ytr)) - Yte[0]) ** 2)

        try:
            m_mc, (ym_mc, ys_mc) = train_mc_dropout(Xtr, Ytr, epochs=200)
            mu_mc, _ = mc_predict(m_mc, Xte, ym_mc, ys_mc, n_mc=20)
            sq["mc_dropout"].append((float(mu_mc[0]) - Yte[0]) ** 2)
        except Exception:
            sq["mc_dropout"].append((float(np.mean(Ytr)) - Yte[0]) ** 2)

        try:
            m_d, lik, (ym_d, ys_d), sc_d = train_dkl(Xtr, Ytr, epochs=80)
            mu_d, _ = dkl_predict(m_d, lik, Xte, ym_d, ys_d, sc_d)
            sq["dkl"].append((float(mu_d[0]) - Yte[0]) ** 2)
        except Exception:
            sq["dkl"].append((float(np.mean(Ytr)) - Yte[0]) ** 2)

    return {k: float(np.sqrt(np.mean(v))) for k, v in sq.items()}


# ============================================================
# SECTION 11: PER-FUNCTION CONFIGURATION
# ============================================================

# F5: dim1 tighter than W8
f5_low  = np.array([0.000, 0.999, 0.999, 0.999])
f5_high = np.array([0.002, 1.000, 1.000, 1.000])

# F2: joint bounds anchored to W8 best
f2_low  = np.array([0.695, 0.918])
f2_high = np.array([0.712, 0.935])

method_config = {
    "function_1": {"acq": "GP_MEAN", "margin": 0.015, "kernel": "matern52", "restarts": 12},
    "function_2": {"acq": "GP_MEAN", "margin": None, "custom_bounds": True,
                   "kernel": "matern52", "restarts": 12},
    "function_3": {"acq": "GP_MEAN", "margin": 0.015, "kernel": "matern52", "restarts": 12},
    "function_4": {"acq": "EI", "xi": 0.0, "zombi": "pinned",
                   "kernel": "matern15", "restarts": 15},
    "function_5": {"acq": "GP_MEAN", "margin": None, "custom_bounds": True,
                   "kernel": "matern52", "restarts": 15},
    "function_6": {"acq": "GRID", "step": 0.002, "kernel": "matern52", "restarts": 15},
    "function_7": {"acq": "GP_MEAN", "margin": 0.015, "kernel": "matern52", "restarts": 20},
    "function_8": {"acq": "EI", "xi": 0.001, "margin": 0.020,
                   "kernel": "matern52", "restarts": 20},
}


# ============================================================
# SECTION 12: LOO-CV SURROGATE SELECTION
# ============================================================

print("\n" + "=" * 68)
print("LOO-CV SURROGATE SELECTION (Matern52 / RBF / MC-Drop / DKL)")
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
# SECTION 13: GENERATE WEEK 9 QUERIES
# ============================================================

N_CANDS = 60000
week9_results = {}

for i in range(1, 9):
    key    = f"function_{i}"
    cfg    = method_config[key]
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    best   = all_bests[key]["inp"]
    margin = cfg.get("margin")
    ktype  = cfg.get("kernel", "matern52")

    print(f"\n{'=' * 62}")
    print(f"F{i} | {cfg['acq']} | {ktype} | obs={X.shape[0]} | best={Y.max():.6f}")
    print(f"{'=' * 62}")

    # GRID for F6
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

    # Pinned ZoMBI for F4
    elif cfg.get("zombi") == "pinned":
        w2_best = np.array([0.403695, 0.397605, 0.413333, 0.411576])
        gp, sc, ym, ys, lb, hb, f_best_mem = pinned_zombi(
            X, Y, pin_inp=w2_best, m=5, kernel_type="matern15")
        print(f"  Pinned ZoMBI: {np.round(lb,3)} to {np.round(hb,3)}")
        np.random.seed(42)
        cands    = np.random.uniform(lb, hb, size=(N_CANDS, dim))
        mu_s, sg = gp.predict(sc.transform(cands), return_std=True)
        mu       = mu_s * ys + ym
        sigma    = np.maximum(sg * ys, 1e-9)
        acq_vals = expected_improvement(mu, sigma, f_best_mem, xi=0.0)
        bidx     = np.argmax(acq_vals)
        query    = np.clip(cands[bidx], 0.0, 1.0)
        score    = float(acq_vals[bidx])
        mu_out   = float(mu[bidx]); sigma_out = float(sigma[bidx])

    else:
        # Build candidate bounds
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
        elif cfg["acq"] == "EI":
            acq_vals = expected_improvement(mu, sigma, f_best, xi=cfg["xi"])
        else:
            acq_vals = upper_confidence_bound(mu, sigma, kappa=cfg.get("kappa", 0.8))

        bidx      = np.argmax(acq_vals)
        query     = np.clip(cands[bidx], 0.0, 1.0)
        score     = float(acq_vals[bidx])
        mu_out    = float(mu[bidx]); sigma_out = float(sigma[bidx])

    formatted = "-".join([f"{x:.6f}" for x in query])
    week9_results[key] = {
        "query": query, "formatted_query": formatted,
        "acq": cfg["acq"], "surrogate": ktype,
        "score": score, "predicted_mean": mu_out, "uncertainty": sigma_out,
    }
    dist = np.linalg.norm(query - best)
    print(f"Query     : {formatted}")
    print(f"Score     : {score:.6f} | Pred mean: {mu_out:.6f} | Dist: {dist:.4f}")


# ============================================================
# SECTION 14: VALIDATION
# ============================================================

print("\n" + "=" * 62)
print("QUERY VALIDATION REPORT")
print("=" * 62)

all_clear = True
for i in range(1, 9):
    key   = f"function_{i}"
    query = week9_results[key]["query"]
    issues = []
    if np.any(query < 0) or np.any(query > 1): issues.append("OUT OF RANGE")
    if np.all(query < 0.01) and i not in [5]:  issues.append("SUSPICIOUS: all near 0")
    if np.all(query > 0.99):                    issues.append("SUSPICIOUS: all near 1")
    status = "OK" if not issues else "WARNING"
    print(f"\nF{i} [{status}] [{week9_results[key]['acq']}] [{week9_results[key]['surrogate']}]")
    print(f"  Query : {week9_results[key]['formatted_query']}")
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
    q   = week9_results[key]["query"]
    b   = all_bests[key]["inp"]
    d   = np.linalg.norm(q - b)
    flag = "ok" if d < 0.20 else "far — review"
    print(f"F{i} [{week9_results[key]['acq']}] | dist: {d:.4f} [{flag}]")


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
        ls = get_ard_lengthscales(gp)
        if ls is not None:
            ranked = np.argsort(ls)
            print(f"\nF{i} ({X.shape[1]}D): ls={np.round(ls, 3)}")
            print(f"  Most important : dim{ranked[0]+1}  ls={ls[ranked[0]]:.4f}")
            print(f"  Least important: dim{ranked[-1]+1} ls={ls[ranked[-1]]:.4f}")
    except Exception as e:
        print(f"F{i}: {e}")


# ============================================================
# SECTION 17: PER-FUNCTION MODEL CARDS
# ============================================================

print("\n\n" + "=" * 65)
print("PER-FUNCTION MODEL CARDS — WEEK 9")
print("=" * 65)

for i in range(1, 9):
    key    = f"function_{i}"
    X      = updated_data[key]["X"]
    Y      = updated_data[key]["Y"]
    dim    = X.shape[1]
    rmse   = surrogate_selection[key]["rmse"]
    winner = surrogate_selection[key]["winner"]
    b      = all_bests[key]
    r      = week9_results[key]
    cfg    = method_config[key]

    acq_str = ("GP Mean — pure exploitation (arXiv 2103.16649)"
               if cfg["acq"] == "GP_MEAN"
               else f"EI xi={cfg.get('xi',0)}{' [Pinned ZoMBI]' if cfg.get('zombi') else ''}"
               if cfg["acq"] == "EI"
               else f"Grid step={cfg.get('step')}")

    print(f"\n{'─'*65}")
    print(f"  MODEL CARD   F{i}  ({dim}D | {X.shape[0]} obs)")
    print(f"{'─'*65}")
    print(f"  All-time best  : {b['val']:.6g}  ({b['src']})")
    print(f"  Strategy note  : {b['note'][:70]}")
    print()
    print(f"  LOO-CV RMSE")
    for k in ["matern52", "rbf", "mc_dropout", "dkl"]:
        tag = "  SELECTED" if k == winner else ""
        val = rmse.get(k, "n/a")
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"    {k:<14}: {val_str}{tag}")
    print()
    print(f"  WEEK 9 QUERY")
    print(f"    Acquisition    : {acq_str}")
    print(f"    Kernel         : {cfg.get('kernel','matern52')} (ARD + Y-scaled)")
    print(f"    Query          : {r['formatted_query']}")
    print(f"    Predicted mean : {r['predicted_mean']:.6g}")
    print(f"    Dist from ATB  : {np.linalg.norm(r['query'] - b['inp']):.4f}")


# ============================================================
# SECTION 18: SAVE WEEK 9 QUERIES
# ============================================================

rows = []
for i in range(1, 9):
    key  = f"function_{i}"
    Y    = updated_data[key]["Y"]
    r    = week9_results[key]
    rmse = surrogate_selection[key]["rmse"]
    rows.append({
        "Function":        f"Function {i}",
        "Query":           r["formatted_query"],
        "Acquisition":     r["acq"],
        "Kernel":          r["surrogate"],
        "Matern52_LOO":    round(rmse.get("matern52",   0), 4),
        "RBF_LOO":         round(rmse.get("rbf",        0), 4),
        "MC_LOO":          round(rmse.get("mc_dropout", 0), 4),
        "DKL_LOO":         round(rmse.get("dkl",        0), 4),
        "Predicted_mean":  round(r["predicted_mean"],       6),
        "Uncertainty":     round(r["uncertainty"],          6),
        "Current_best":    round(float(Y.max()),            6),
    })

df = pd.DataFrame(rows)
print("\n\nFINAL WEEK 9 QUERIES")
print(df[["Function","Kernel","Acquisition","Query","Current_best"]].to_string(index=False))

df[["Function","Query"]].to_csv("week9_queries.csv", index=False)
df.to_csv("week9_queries_full.csv", index=False)
print("\nweek9_queries.csv      — submit this file")
print("week9_queries_full.csv — LOO-RMSE and model card data")
