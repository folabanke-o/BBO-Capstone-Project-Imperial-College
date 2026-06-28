# Repository Structure, Libraries, and Documentation Review

[Back to README](../README.md)

---

## Repository Structure

**Current state.** The repository is organised into five top-level folders: `docs/` for weekly strategy and reflection files, `images/` for supporting charts grouped by week, `notebooks/` for runnable pipeline scripts, `results/` for query and output logs, and `docs/archive/` for the original Word documents the shorter summaries are drawn from. This replaced an earlier flat structure where roughly fifteen documents from five weeks of strategy notes, notebooks, and reflections sat side by side with no separation by purpose.

Each week now has its own file (`week1.md` through `week5.md`), linked from a contents table in `README.md`, with charts referenced from the matching `images/weekN/` folder. Notebooks are named consistently by week, and the Week 5 neural network script sits alongside the main pipeline notebook rather than mixed in with strategy documents.

![Current flat structure compared with the structure now adopted](../images/week5/repo_structure_comparison.png)

**Changes still planned.** The Gaussian Process fitter and the four acquisition functions (Expected Improvement, Upper Confidence Bound, Thompson Sampling, and the grid search used for F6) are currently duplicated across each week's notebook. These will move into a shared `src/` module so later weeks import rather than copy code. A single `results/query_log.csv`, appending every submitted query and returned output, will replace the need to read across five files to reconstruct the project's history. The `docs/archive/` folder is kept as a permanent record rather than merged into the main narrative, since the longer reflections are more detailed than the weekly summaries need to be.

## Coding Libraries and Packages

Three libraries form the core of the approach used throughout, with a fourth added in Week 5 for comparison only.

| Library | Used for | Why appropriate | Trade-off considered |
|---|---|---|---|
| scikit-learn | Gaussian Process regression, input scaling | Exact inference and calibrated uncertainty from very small datasets (12 to 47 observations per function) | Slower than a custom GP implementation at scale, but the project never approaches that scale |
| scipy | Sobol sequences for candidate generation, statistical functions for Expected Improvement | Quasi-random candidates give more even space coverage than uniform random sampling | Requires scipy 1.7 or above, a minor version dependency worth flagging in setup instructions |
| numpy / pandas | Array maths, query history, CSV export | Standard, well documented, no realistic alternative needed at this scale | None significant |
| PyTorch | Week 5 multilayer perceptron surrogate, comparison only | Flexible for a small custom architecture, straightforward MC Dropout implementation | Every function has 7 to 19 times more network parameters than observations, so results are documented in model cards rather than used to generate queries |

![Approximate role of each core library across the full pipeline](../images/week5/repo_library_usage.png)

TensorFlow was not adopted for the Week 5 comparison. Both frameworks would have served a small four-layer network equally well; PyTorch was chosen for its more direct syntax when defining and training a custom architecture quickly, matching the exploratory, frequently revised nature of that part of the project. This choice is documented in the README's Technical Approach section, alongside the architecture diagram and parameter-to-observation comparison saved in `images/week5/`.

## Documentation

**Current state.** The `README.md` states the project's purpose, the input and output format (hyphen-separated decimal vectors in the range zero to one, with a single float returned per query), and a weekly summary table giving the best result for all eight functions after Week 1 and after Week 5. A Technical Approach section lists the four acquisition methods in use and explains why the Gaussian Process remains the primary surrogate. This is a substantial improvement on the previous draft, which had not been updated since Week 2 and described only the original uniform UCB strategy.

**Updates still needed.** The README's summary table currently compares only Week 1 and Week 5; a fuller version showing every week's output per function would make the top-level overview more useful without requiring a reader to open all five linked files. The model cards referenced from `week5.md` were built using representative data rather than the real `.npy` files, since these were not available when produced, and this caveat needs to stay visible until the cards are regenerated against the actual datasets. Finally, `results/` is currently a placeholder; populating it with the real weekly query and output CSVs is the next concrete step toward the reproducibility goal described above.

---

[Back to README](../README.md)
