# BBO Capstone Project — Data Sheet

> Framework: Gebru et al. (2021) *Datasheets for Datasets*

---

## 1. Motivation

**Why was this dataset created?**
To support sequential black-box optimisation of eight synthetic objective functions over ten weekly iterations as part of a university capstone project. The dataset documents the complete query-response record of a student-led optimisation campaign so that decisions can be audited, reproduced, and extended.

**Who created it and for what task?**
Created by a single researcher across ten weeks. The task is Bayesian Black-Box Optimisation (BBO): given a fixed budget of one irreversible query per function per week, find the input that maximises each function's output.

---

## 2. Composition

**What does the dataset contain?**

| Component | Format | Description |
|---|---|---|
| `initial_inputs.npy` | NumPy binary | Original initial dataset inputs, float64 |
| `initial_outputs.npy` | NumPy binary | Corresponding initial outputs, float64 |
| `week{n}_queries.csv` | CSV | Submitted input + returned output per week |
| `week{n}_notebook.py` | Python | Notebook with full pipeline and model cards |

**Function dimensions and initial observation counts:**

| Function | Dim | Initial obs | Total obs (after W10) |
|---|---|---|---|
| F1 | 2D | 8 | ~18 |
| F2 | 2D | 8 | ~18 |
| F3 | 3D | 12 | ~22 |
| F4 | 4D | ~15 | ~25 |
| F5 | 4D | ~15 | ~25 |
| F6 | 5D | ~20 | ~30 |
| F7 | 6D | ~25 | ~35 |
| F8 | 8D | ~25 | ~35 |

**Gaps:** Dataset is sparse relative to input dimensionality. Early queries (W1-W3) used broad exploration and are far from the confirmed optimal regions, creating a spatial bias.

---

## 3. Collection Process

**How were queries generated?**
Via a Bayesian optimisation pipeline selecting one of four surrogates per function (GP ARD-RBF, GP ARD-Matern-5/2, MC Dropout MLP, DKL) using LOO-CV RMSE, then applying EI / UCB / GP Mean / Grid / ZoMBI acquisition.

**Timeframe:** Ten weekly cycles. One irreversible evaluation per function per week.

**Exclusions:** None. All portal outputs including near-zero (F1) and negative values are retained. ZoMBI pruning affects which points are used for GP fitting but does not delete records.

---

## 4. Preprocessing and Intended Uses

**Preprocessing applied:**
- `StandardScaler` on X for all GP and neural surrogate inputs
- Y standardisation (zero mean, unit std) on all surrogates from W8 onward
- ARD length scales initialised to `sqrt(d)/10` per arXiv:2502.09198
- No imputation or augmentation

**Intended uses:**
- Reproducible BBO research (fixed `random_state=42`)
- Surrogate model benchmarking on small-n problems
- Acquisition function convergence analysis
- Teaching: real-world BBO with irreversible evaluations

**Inappropriate uses:**
- General ML benchmarking without disclosing capstone budget constraints
- Functions requiring positive outputs without preprocessing (F1 near-zero history)
- Claiming state-of-the-art performance without context

---

## 5. Distribution and Maintenance

**Availability:** Public GitHub repository (this repo). Initial `.npy` files are not redistributed; obtain via the capstone portal.

**Licence:** Weekly query records and notebooks: MIT. Initial data: capstone provider copyright.

**Maintenance:** Maintained by the student researcher. Issues via the GitHub issue tracker. Dataset is complete after Week 10.

---

*Data sheet follows Gebru et al. (2021). Last updated: Week 10.*
