# Simulation package — pharma cascade paper (regenerated)
- model.py        canonical model (design decisions M1-M5 in header docstring)
- experiments.py  regenerates every Section 6 number -> results_e124.json (~5 min)
- figures.py      regenerates Figures 1-5 (run after experiments.py)
- robustness.py   embedding/curvature machinery (used by experiments, reusable)
- results_e124.json  the results behind the revised manuscript's tables
- fig1..fig5      regenerated figures (pdf for LaTeX, png for preview)

VERIFY BEFORE SUBMISSION: run `python3 experiments.py && python3 figures.py`
and diff results_e124.json against the manuscript numbers. Both authors must
be able to explain and defend M1-M5.
Publish this folder to GitHub + Zenodo and cite the DOI in Data Availability.
