# BBO Capstone Project

**Bayesian Black-Box Optimisation across 8 synthetic functions over 10 weekly rounds**

This repository contains the complete record of a university capstone project in which a Bayesian optimisation pipeline was developed iteratively over ten weeks to maximise eight black-box functions with dimensionality ranging from 2D to 8D. Each week, one irreversible query was submitted per function via the capstone portal.

---

## Repository Structure

```
bbo-capstone/
├── notebooks/              Python notebooks for each weekly round
│   ├── week2_notebook.py
│   ├── week3_notebook.py
│   ├── ...
│   └── week10_notebook.py
│
├── docs/
│   ├── weekly/             Strategy notes and post-mortems for each week
│   │   ├── week1_reflection.md
│   │   ├── week2_strategy.md
│   │   ├── ...
│   │   └── week10_strategy.md
│   │
│   ├── reflections/        Discussion board reflections and written submissions
│   │   ├── hp_reflection_final2.docx      Hyperparameter tuning reflection
│   │   ├── prompt_reflection.docx         Prompting and decoding strategies
│   │   ├── justification_reflection.docx  Technical justification
│   │   ├── industry_cnn_reflection.docx   CNN applicability reflection
│   │   ├── cnn_reflection.docx            CNN industry challenge
│   │   ├── w9_reflection.docx             Week 9 scaling laws reflection
│   │   └── w10_reflection.docx            Week 10 final reflection
│   │
│   └── cards/              Data sheet and model card
│       ├── DATASHEET.md
│       ├── MODEL_CARD.md
│       └── bbo_datasheet_modelcard.docx   Combined Word document
│
├── results/
│   └── performance_self_assessment_v2.xlsx   Full 5-column performance tracker
│
├── images/                 Charts and visualisations
│
└── README.md
```

---

## Documentation

| Document | Description | Link |
|---|---|---|
| Data Sheet | Dataset motivation, composition, collection, preprocessing, distribution | [DATASHEET.md](docs/cards/DATASHEET.md) |
| Model Card | Surrogate architectures, acquisition schedule, performance, limitations | [MODEL_CARD.md](docs/cards/MODEL_CARD.md) |

---

## All-Time Bests After Week 9

| Function | Dim | Initial Best | All-Time Best | Set in Week | Gain |
|---|---|---|---|---|---|
| F1 | 2D | 7.71e-16 | 7.62e-9 | W9 | +4,443% |
| F2 | 2D | 0.611 | 0.764 | W8 | +25.1% |
| F3 | 3D | -0.035 | -5.1e-4 | W9 | +98.5% |
| F4 | 4D | -4.026 | 0.534 | W2 | +113.3% |
| F5 | 4D | 1,089 | 4,440 | W9 | +307.7% |
| F6 | 5D | -0.714 | -0.680 | W4 | +4.8% |
| F7 | 6D | 1.365 | 2.747 | W9 | +101.2% |
| F8 | 8D | 9.598 | 9.996 | W7 | +4.1% |

---

## Gains Per Week

| Week | Gains | Key Development |
|---|---|---|
| W1 | 4/8 | Uniform GP-UCB initial sweep |
| W2 | 2/8 | First per-function bounds introduced |
| W3 | 3/8 | Thompson sampling added for F8 |
| W4 | 4/8 | Anchor-to-all-time-best rule validated |
| W5 | 2/8 | LOO-CV surrogate selection introduced |
| W6 | 2/8 | DKL discovered F5 dim1-near-zero region |
| W7 | 3/8 | ARD kernels and ZoMBI added (colleague suggestion) |
| W8 | 5/8 | Best round: ZoMBI breakthrough on F1 and F2 |
| W9 | 4/8 | GP Mean exploitation; F3, F5, F7 new ATBs |
| W10 | TBD | Length scale ensemble and micro-bounds (final round) |

---

## Pipeline Overview

### Surrogate Models

| Model | Library | When used |
|---|---|---|
| GP with ARD-RBF | scikit-learn | W1-W7 default |
| GP with ARD-Matern-5/2 | scikit-learn | W8-W10 default |
| MC Dropout MLP | PyTorch | Selected by LOO-CV from W5 |
| Deep Kernel Learning | GPyTorch | Selected by LOO-CV from W5; found F5 peak |
| Length scale ensemble | scikit-learn x5 | F4 in W10 |
| ZoMBI pruned GP | scikit-learn | F1 W7-W9, F4 W8-W10 |

### Acquisition Functions

| Acquisition | Parameter | Phase |
|---|---|---|
| UCB | kappa=2.0 | W1-W2 exploration |
| Expected Improvement | xi=0.001-0.010 | W3-W9 exploitation |
| GP Mean | pure exploitation | W8-W10 converging functions |
| EI xi=0 | pure exploitation | W8-W10 sharp peak targeting |
| Grid search | step=0.001-0.003 | F6 throughout |

### Key Design Principles

1. **Anchor-to-ATB rule (W4 onwards):** all search bounds anchored to the confirmed all-time best input. Every week this rule was violated, the output declined.
2. **LOO-CV surrogate selection (W5 onwards):** surrogate chosen per function by lowest leave-one-out RMSE across four options.
3. **ARD kernels (W7 onwards):** per-dimension length scales allow the GP to learn which input dimensions drive the output.
4. **ZoMBI memory pruning:** prunes low-performing historical observations before GP fitting, preventing early bad queries from biasing the surrogate mean.

---

## Reflections and Written Submissions

| Document | Topic |
|---|---|
| [Hyperparameter Reflection](docs/reflections/hp_reflection_final2.docx) | Hyperparameter tuning strategy across ten rounds |
| [Prompting Reflection](docs/reflections/prompt_reflection.docx) | Prompting and decoding strategies |
| [Technical Justification](docs/reflections/justification_reflection.docx) | Literature review and library choices |
| [CNN Reflection](docs/reflections/cnn_reflection.docx) | CNN applicability to industry challenge |
| [Industry CNN Discussion](docs/reflections/industry_cnn_reflection.docx) | Manufacturing defect detection case study |
| [Week 9 Reflection](docs/reflections/w9_reflection.docx) | Scaling laws and emergent behaviour |
| [Week 10 Reflection](docs/reflections/w10_reflection.docx) | Strategy, transparency, assumptions and limitations |

---

## Performance Tracker

The file [results/performance_self_assessment_v2.xlsx](results/performance_self_assessment_v2.xlsx) contains a five-column tracker for all eight functions across all ten rounds:

1. Weekly output value
2. Change vs previous week
3. Running all-time best
4. L2 distance from ATB input (key drift diagnostic)
5. Surrogate and method used

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
torch>=2.0
gpytorch>=1.11
```

Install with:
```bash
pip install numpy pandas scikit-learn scipy torch gpytorch
```

---

## References

- Rasmussen and Williams (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- Jones, Schonlau and Welch (1998). Efficient Global Optimisation. *Journal of Global Optimization.*
- Gal and Ghahramani (2016). Dropout as a Bayesian Approximation. *ICML.*
- Wilson et al. (2016). Deep Kernel Learning. *AISTATS.*
- Gardner et al. (2018). GPyTorch. *NeurIPS.*
- Siemenn et al. (2023). Fast Bayesian Optimisation with ZoMBI. *npj Computational Materials.*
- Li et al. (2024). arXiv:2305.20028.
- Gebru et al. (2021). Datasheets for Datasets. *Communications of the ACM.*
- Mitchell et al. (2019). Model Cards for Model Reporting. *FAccT.*

---

## Licence

Notebooks and documentation: MIT Licence.
Initial dataset (.npy files): intellectual property of the capstone course provider. Obtain via the official portal.
