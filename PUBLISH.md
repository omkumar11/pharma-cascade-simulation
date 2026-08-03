# Publishing this archive (~10 minutes, do BEFORE submission)

1. VERIFY FIRST (non-negotiable): `pip install -r requirements.txt` then
   `python3 experiments.py && python3 figures.py`. Diff results_e124.json
   against the manuscript tables. You are signing your names to these numbers.
2. Create a GitHub repo (e.g. pharma-cascade-simulation), push this folder:
     git init && git add -A && git commit -m "Simulation code v1.0"
     git remote add origin <your-repo-url> && git push -u origin main
3. Mint the DOI: log into zenodo.org with GitHub -> enable the repo under
   GitHub integration -> back in GitHub, create a release tagged v1.0 ->
   Zenodo automatically archives it and issues a DOI (the .zenodo.json in
   this folder pre-fills the metadata).
4. Open revised-manuscript.tex, search for "REPLACE-WITH" (2 hits in the
   Data Availability Statement), paste the repo URL and the DOI, recompile.
   The manuscript compiles but MUST NOT be submitted while either
   placeholder remains.
