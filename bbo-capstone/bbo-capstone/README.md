# Bayesian Black-Box Optimisation Capstone

This repository documents a five-week capstone project applying Bayesian Optimisation to eight synthetic black-box functions. Each function accepts a multi-dimensional input and returns a single scalar output, with no access to the underlying formula or gradients. One query per function is permitted per week, and the goal throughout has been to maximise every function's output using as few queries as possible.

The project has run as a structured weekly cycle: submit a query, receive the result the following week, analyse what happened, and refine the strategy before the next submission. This README is the entry point. Detailed strategy, results, and reflections for each week are kept in their own files under `docs/`, with supporting charts under `images/` and runnable code under `notebooks/`.

---

## Contents

| Section | Description |
|---|---|
| [Project overview](#project-overview) | Purpose, goals, and real-world relevance |
| [Repository structure](#repository-structure) | How this repository is organised |
| [Weekly summary](#weekly-summary) | Quick-reference table of progress across all five weeks |
| [Detailed weekly reports](#detailed-weekly-reports) | Links to each week's full strategy and reflection |
| [Repository and documentation review](docs/repo_review.md) | Reflection on structure, libraries, and documentation gaps |
| [Technical approach](#technical-approach) | Surrogate model, acquisition methods, and tools used |
| [How to run the notebooks](#how-to-run-the-notebooks) | Setup instructions |

---

## Project Overview

Bayesian Optimisation is a method for finding the maximum (or minimum) of an expensive, unknown function using as few evaluations as possible. It works by fitting a probabilistic surrogate model to the data observed so far, then using that surrogate to decide intelligently where to query next, rather than searching at random.

This problem structure mirrors common real-world challenges: tuning machine learning hyperparameters, optimising chemical or material formulations, and calibrating expensive simulations. In each case, a single evaluation is costly, and the input space is too large to explore exhaustively.

The eight functions in this project vary in dimensionality from two to eight inputs, and their underlying behaviour (smooth, narrow-peaked, multimodal, or noisy) was discovered empirically across the five weeks rather than given in advance.

---

## Repository Structure

```
bbo-capstone/
├── README.md                  This file
├── docs/                      Weekly strategy documents and reflections
│   ├── week1.md
│   ├── week2.md
│   ├── week3.md
│   ├── week4.md
│   ├── week5.md
│   └── archive/                Original Word documents (full reflections, model cards)
├── images/                    Charts referenced from the weekly documents
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   ├── week5/
├── notebooks/                  Runnable Python scripts for each week's pipeline
│   ├── week2_notebook.py
│   ├── week3_notebook.py
│   ├── week4_notebook.py
│   ├── week5_notebook.py
│   └── week5_nn_comparison_notebook.py
└── results/                    Query and output logs (add your own CSV exports here)
```

---

## Weekly Summary

| Function | Dimensions | Best after Week 1 | Best after Week 5 | Overall change |
|---|---|---|---|---|
| F1 | 2 | ~0.000000 | ~0.000000 | No meaningful gain |
| F2 | 2 | -0.109705 | 0.332479 | Recovered toward initial best of 0.611205 |
| F3 | 3 | -0.365767 | -0.006082 | Improved steadily across three consecutive weeks |
| F4 | 4 | -0.869002 | 0.533858 | Crossed from negative to positive in Week 2 |
| F5 | 4 | 3019.660 | 3905.150 | Strongest performer; gained in four of five weeks |
| F6 | 5 | -1.213319 | -0.680152 | Improved for three consecutive weeks from Week 3 |
| F7 | 6 | 1.771970 | 2.018863 | Two consecutive new bests in Weeks 3 and 4 |
| F8 | 8 | 9.965293 | 9.960920 | Came within 0.004 of the all-time best in Week 4 |

A full breakdown of each week's queries, outputs, and reasoning is available in the linked documents below.

---

## Detailed Weekly Reports

- **[Week 1](docs/week1.md)** — Initial uniform strategy, first results, and the first signs of which functions needed a different approach.
- **[Week 2](docs/week2.md)** — Introduction of per-function tuning, constrained search boxes, and the first reflection on exploration versus exploitation.
- **[Week 3](docs/week3.md)** — Adoption of multiple acquisition methods (Expected Improvement, Upper Confidence Bound, Thompson Sampling, grid search) matched to each function's observed behaviour.
- **[Week 4](docs/week4.md)** — Four functions reached new all-time bests in a single week. Includes the supporting reflection on support vectors, surrogate gradients, and model choice.
- **[Week 5](docs/week5.md)** — Introduction of a neural network surrogate for comparison, model cards for each function, and reflections connecting the project to wider machine learning concepts.
- **[Repository and documentation review](docs/repo_review.md)** — Reflection on repository structure, library choices, and documentation gaps, written against the structure shown above.

---

## Technical Approach

**Surrogate model.** A Gaussian Process with a `ConstantKernel x RBF + WhiteKernel` kernel, fitted using scikit-learn, has been the primary surrogate throughout. It was chosen because it provides calibrated uncertainty estimates from very small datasets (12 to 47 observations per function), which the acquisition methods below depend on directly.

**Acquisition methods used across the five weeks:**

| Method | Used for | Reasoning |
|---|---|---|
| Upper Confidence Bound (UCB) | Functions with well-characterised surfaces | Balances exploration and exploitation through a single tunable parameter |
| Expected Improvement (EI) | Functions with a confirmed good region | Only recommends a query if genuine improvement is expected |
| Thompson Sampling | Functions needing posterior-based exploration | Less prone to overcommitting to a single point than high-kappa UCB |
| Deterministic grid search | Functions where the surrogate proved unreliable | Exhaustively samples a small neighbourhood without relying on uncertainty estimates |

**Neural network comparison (Week 5).** A small multilayer perceptron was trained per function purely for comparison against the Gaussian Process, not to replace it. Every function in this project has substantially more network parameters than observations, which is the main reason the Gaussian Process remains the production surrogate. Full reasoning and per-function model cards are in [Week 5](docs/week5.md).

**Libraries.** scikit-learn (Gaussian Process, scaling), scipy (Sobol sequences, statistical functions for Expected Improvement), numpy and pandas (data handling), and PyTorch (the Week 5 neural network comparison only).

---

## How to Run the Notebooks

1. Install the required packages:
   ```
   pip install numpy pandas scikit-learn scipy torch
   ```
2. Place the original function datasets in a folder named `function_1` through `function_8`, each containing `initial_inputs.npy` and `initial_outputs.npy`.
3. Run any week's script directly:
   ```
   python notebooks/week5_notebook.py
   ```
4. Each script prints the generated queries to the console and saves them to a CSV file in the working directory.

---

*This repository is a personal capstone project and is updated weekly as new results are received.*
