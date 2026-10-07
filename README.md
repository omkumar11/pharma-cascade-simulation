# Simulation code — Kumar & Salwekar, *Threshold Gating, Delay Equivariance, and Coupling Artifacts in Cascading Failures of Tiered Production Networks*

Every number, table and figure in the manuscript regenerates from this
repository (release v2.4; concept DOI 10.5281/zenodo.21780232).

## Layout

| file | role |
|---|---|
| `model.py` | network builder (seed 42, tiers 20/35/50/41, ER p=0.15), the **v1 indicator-coupled** simulator, metrics, discounted-reachability score chi(j) |
| `decisions.py` | `simulate_v` = flexible simulator (aggregator `reqmin`/`ces`, damping `binary`/`smooth`, sourcing `indicator`/`logistic`); experiments D1–D6 of **Section 4** (artifact anatomy) -> `results_decisions.json` |
| `revision_suite.py` | corrected-model suite B1–B13 for **Sections 6–7** (targeting, delay, lambda, rho, tau/alpha/delta, game, R_c Jacobian, topology/size replication, eta/budget, concentration regression, cisplatin back-test, smooth-sourcing robustness) -> `results_v2.json` |
| `qa_robustness.py` | v2.2 supplementary checks Q1–Q5 (Katz on the 40 topology draws, budget sweep with Katz, the tau x alpha robustness grid = **Table VI**, back-test statistics, peak inventory) -> `results_qa.json` |
| `game_exact.py` | v2.3 check Q6: the **Section 7** game with the tier-1 private problem in closed form (dominant strategy), equilibrium and planner by 100-run iterated best response, unimodality of downstream private payoffs -> `results_game.json` |
| `theory_checks.py` | v2.4: numerical verification of every theorem of the v3 manuscript (viability invariance, confinement margin, quasi-steady state, finite-time recovery, delay equivariance under matched lags, rung thresholds, node-level coupling facts) -> `results_theory.json` |
| `figures_v3.py` | v2.4: Figure 1 of the v3 manuscript (`fig7_theory.pdf`) from `results_theory.json` |
| `theory_checks_v4.py` | v2.4: comparison principle, channel isolation, lag-potential equivariance, convergence rate -> `results_theory_v4.json` |
| `theory_checks_v5.py` | v2.4: damping-lead mechanism, persistent ever-failed benchmark, rate genericity recomputation -> `results_theory_v5.json` |
| `theory_checks_v6.py` | v2.4: interval enclosure (Theorem 5) and lag sandwich (Theorem 8) -> `results_theory_v6.json` |
| `theory_checks_v7.py` | v2.4: onset-window breakdown of the lag sandwich over failing non-source nodes (Table 2 of the final manuscript) -> `results_theory_v7.json` |
| `spillover_check.py` | v2.4: spillover of one source's reserve below the private optimum -> `results_spillover_small.json` |
| `figures_v2.py` | Figures 1–6 from the four JSON files (`results_qa.json` adds the Katz line to Fig. 4; `results_game.json` gives the three-panel Fig. 6) |
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
python3 game_exact.py                # v2.3 game   (~10 min) -> results_game.json
python3 figures_v2.py                # fig1_transition ... fig6_quasi (.pdf)
python3 theory_checks.py             # v2.4 theory checks (~6 min) -> results_theory.json
python3 figures_v3.py                # fig7_theory (.pdf/.png)
python3 theory_checks_v4.py          # v2.4 checks for manuscript v4 (~12 min) -> results_theory_v4.json
python3 spillover_check.py           # source-reserve spillover 0 -> 0.008 (~4 min) -> results_spillover_small.json
python3 theory_checks_v5.py          # v2.4 checks for manuscript v5 (~6 min) -> results_theory_v5.json
python3 theory_checks_v6.py          # v2.4 checks for manuscript v6 (~10 min on 10 cores) -> results_theory_v6.json
python3 theory_checks_v7.py          # v2.4 onset-window breakdown (<1 min) -> results_theory_v7.json
```

`python3 revision_suite.py --only-b13` re-runs only the smooth-sourcing
robustness block and merges it into an existing `results_v2.json`.

Python 3.13, NumPy 2.x, SciPy 1.x; ~65 min total on an Apple M-series laptop.

## Changelog

* v2.4 — MDPI Mathematics v3 manuscript (retitled). New `theory_checks.py`
  (`simulate_m`: the corrected simulator with a `lag_mode` switch,
  "baseline" byte-identical to `decisions.simulate_v`, "matched" = every
  inter-firm channel read at the production lag; `x_inf`: forward-substitution
  quasi-steady state) and `results_theory.json`; new `figures_v3.py` /
  `fig7_theory`. No change to any earlier script or result.
  Manuscript v4-v6 additions: `theory_checks_v4.py` (`simulate_g`: the
  corrected simulator with explicit source paths and edge-specific lags;
  reproduces `simulate_m` exactly for uniform lags) and
  `results_theory_v4.json`; `spillover_check.py` and
  `results_spillover_small.json`; `theory_checks_v5.py` (`simulate_s`:
  `simulate_m` with an inventory record and a switch that disables
  inventory accumulation) and `results_theory_v5.json`;
  `theory_checks_v6.py` (interval enclosure and lag sandwich, uses
  `multiprocessing.Pool`) and `results_theory_v6.json`; `theory_checks_v7.py`
  and `results_theory_v7.json` (final manuscript).

* v2.3 — IEEE Access submission. New `game_exact.py` / `results_game.json`:
  the Section 7 game recomputed with the tier-1 private problem solved in
  closed form (a source firm's failures depend only on its own shock and
  reserve, so its reserve choice is a dominant strategy) and with 100
  common-random-number runs for every simulated evaluation (the v2.x
  `b6_externality` used 10 inner and 30 evaluation runs, and its tier-1
  equilibrium reserve of 0.02 was a Monte Carlo artifact: the exact private
  optimum is 0.008). `figures_v2.py`: Fig. 6 becomes three panels (tier-1
  closed form vs Monte Carlo; planner's objective; downstream private
  payoffs) and embeds figure text as TrueType (Type 42) rather than Type 3
  fonts, as IEEE requires (Figs. 1-6 regenerated, content unchanged). No
  change to any other result.
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
