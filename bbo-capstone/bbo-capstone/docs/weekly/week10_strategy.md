# Week 10 Strategy (Final Round)

## Context
Week 9 produced 4/8 gains. F2 missed its confirmed ridge by 0.001 on dim2. F4 has held its Week 2 best for eight consecutive rounds. This is the final round.

## New this week
- Length scale balancing ensemble for F4 (arXiv LB-BO 2025): five GPs with ls in {0.01, 0.05, 0.1, 0.5, 1.0}, EI averaged across ensemble
- Micro-bounds for F2: [0.7010, 0.7030] x [0.9258, 0.9275] — tightest bounds applied to any function
- ARD length scale initialisation sqrt(d)/10 per arXiv:2502.09198
- Grid step 0.001 for F6 (finest step in the project)

## Per-function decisions
| Function | Acquisition | Key change |
|---|---|---|
| F1 | GP Mean margin 0.012 | Tighter around W9 best |
| F2 | GP Mean micro-bounds | 0.002 window targeting confirmed ridge |
| F3 | GP Mean margin 0.012 | Converging toward zero |
| F4 | 5-GP ensemble + EI xi=0 | Sharp peak; 8 rounds unbeaten |
| F5 | GP Mean dim1 in [0, 0.002] | Maximum exploitation |
| F6 | Grid step 0.001 | Finest grid in project |
| F7 | GP Mean margin 0.012 | Four consecutive new ATBs W6-W9 |
| F8 | EI xi=0, margin 0.015 | Tightest margin on F8 |
