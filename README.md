# Simulation code — Kumar & Salwekar, *Cascade Seeding, Delay Timescales, and Coupling Artifacts in Tiered Pharmaceutical Supply Networks*

Every number, table and figure in the manuscript regenerates from this
repository (release v2.2; concept DOI 10.5281/zenodo.21780232).

## Layout

| file | role |
|---|---|
| `model.py` | network builder (seed 42, tiers 20/35/50/41, ER p=0.15), the **v1 indicator-coupled** simulator, metrics, discounted-reachability score chi(j) |
| `decisions.py` | `simulate_v` = flexible simulator (aggregator `reqmin`/`ces`, damping `binary`/`smooth`, sourcing `indicator`/`logistic`); experiments D1–D6 of **Section 4** (artifact anatomy) -> `results_decisions.json` |
| `revision_suite.py` | corrected-model suite B1–B13 for **Sections 6–7** (targeting, delay, lambda, rho, tau/alpha/delta, game, R_c Jacobian, topology/size replication, eta/budget, concentration regression, cisplatin back-test, smooth-sourcing robustness) -> `results_v2.json` |
| `qa_robustness.py` | v2.2 supplementary checks Q1–Q5 (Katz on the 40 topology draws, budget sweep with Katz, the tau x alpha robustness grid = **Table VI**, back-test statistics, peak inventory) -> `results_qa.json` |
| `figures_v2.py` | Figures 1–6 from the three JSON files (`results_qa.json` adds the Katz line to Fig. 4) |
| `robustness.py`, `new_experiments.py` | embedding/curvature machinery and PA topology builder used by the suite |
| `experiments.py`, `figures.py`, `results_e124.json` | **v1 pipeline (obsolete)** — kept only so the preliminary version's numbers remain reproducible |

Corrected baseline: CES aggregator rho=-10, smooth (shortfall-depth) damping,
L=3, lambda=2, tau=0.3, alpha=1.2, delta=0.2, T=50 (T=200 for durations),
100 common-random-number runs (seeds 0–99), topology seeds 10000–10019.

## Reproduce

```
pip install -r requirements.txt      # numpy, scipy, matplotlib
python3 decisions.py                 # Section 4  (~5 min)  -> results_decisions.json
python3 revision_suite.py            # Sections 6–7 (~35 min) -> results_v2.json
python3 qa_robustness.py             # v2.2 checks (~15 min) -> results_qa.json
python3 figures_v2.py                # fig1_transition ... fig6_quasi (.pdf)
```

`python3 revision_suite.py --only-b13` re-runs only the smooth-sourcing
robustness block and merges it into an existing `results_v2.json`.

Python 3.13, NumPy 2.x, SciPy 1.x; ~40 min total on an Apple M-series laptop.

## Changelog

* v2.2 — IEEE Access revision. New `qa_robustness.py` / `results_qa.json`: Katz
  centrality added to the 40-draw topology replication and to the budget sweep
  (Fig. 4, right); the tau x alpha robustness grid of Table VI (lambda sweep,
  B=4 allocation and T=200 delay comparison in every cell); single-facility
  back-test statistics (propagating runs, duration IQR); peak-inventory check.
  `figures_v2.py`: Fig. 3 marks the binding-switch interval
  lambda* = ln(x_h/x_f) in [1.2, 2.3]; Fig. 4 reads the Katz line from
  `results_qa.json` when present. No change to any v2.1 result.
* v2.1 — D4 transition sweep now at the baseline delay L=3 with the grid
  extended to eps=0.90; new B13 smooth-sourcing robustness (logistic
  surrogate for the sourcing indicator); figure label fixes; metadata
  updated to the current title.
* v2.0 — corrected model (CES + smooth damping), artifact anatomy, full
  referee-response suite.
* v1.x — preliminary indicator-coupled version.
