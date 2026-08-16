# Publishing release v2.1 (~10 minutes, do BEFORE submission)

The manuscript's Data Availability statement cites release **v2.1** under
the concept DOI 10.5281/zenodo.21780232. Until v2.1 exists on Zenodo the
statement is not yet true.

1. VERIFY (non-negotiable): `python3 decisions.py && python3 revision_suite.py
   && python3 figures_v2.py`, then diff results_decisions.json /
   results_v2.json against the manuscript tables. You are signing your
   names to these numbers.
2. `git push origin main`.
3. On GitHub create a release tagged `v2.1` (title: "v2.1 — smooth-sourcing
   robustness, D4 at L=3, metadata"). Zenodo (already linked) archives it
   automatically and mints a new version DOI; the concept DOI resolves to it.
4. Check the Zenodo record title now reads "Simulation code: Cascade
   Seeding, Delay Timescales, and Coupling Artifacts in Tiered Pharmaceutical
   Supply Networks" (it comes from CITATION.cff / .zenodo.json). If the old
   "Geometry, Thresholds, and Control..." title appears, edit the record
   metadata on Zenodo.
5. Optionally paste the v2.1 version DOI into the Data Availability
   statement next to the concept DOI, recompile, re-upload to Overleaf.
