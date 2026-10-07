"""
figures_v3.py -- Figure 7 of the v3 manuscript (theory versus simulation),
from results_theory.json (theory_checks.py) and results_decisions.json.
Run after theory_checks.py.  Style matches figures_v2.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
TH = json.load(open("results_theory.json"))
DEC = json.load(open("results_decisions.json"))
C1, C2, C3, C4 = "#B44E2C", "#2C6FB4", "#5E8C61", "#8C5E8C"


def fig7_theory():
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1))

    # (a) quasi-steady state: prediction = persistent simulation; stochastic recovery below
    ax = axes[0]
    t3 = TH["T3"]
    eps = sorted(float(e) for e in t3["pred_limit_failing_pct"])
    pred = [t3["pred_limit_failing_pct"][str(e) if str(e) in t3["pred_limit_failing_pct"] else repr(e)] for e in eps]
    sim = [t3["sim_ever_failed_pct_persistent"][str(e) if str(e) in t3["sim_ever_failed_pct_persistent"] else repr(e)] for e in eps]
    ax.plot(eps, pred, "-", color=C2, lw=1.6, label=r"forward substitution $x^\infty$")
    ax.plot(eps, sim, "o", color=C2, ms=4.5, mfc="none", label="simulation, persistent outage")
    d4 = DEC["d4"]
    ax.plot(d4["epss"], d4["spread"]["ces-10/smooth"], "--s", color=C1, ms=3.5, lw=1.1,
            label=r"simulation, recovery $p_{\rm rec}=0.06$")
    rung = TH["T6"]["rung_s1"]
    for m in ("1", "2", "4", "7"):
        ax.axvline(rung[m], color="gray", lw=0.6, ls=":")
        ax.annotate(f"m={m}", (rung[m], 28), fontsize=6.5, color="gray",
                    rotation=90, ha="right", va="bottom")
    ax.set_xlabel(r"shock depth $\epsilon$")
    ax.set_ylabel("failing nodes (% of network)")
    ax.set_title("(a) Rung thresholds and the limit state")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    # (b) cumulative failures versus L
    ax = axes[1]
    t5 = TH["T5"]
    Ls = list(range(7))
    for key, lab, c, st in (("matched", "matched lags: exact invariance", C2, "-o"),
                            ("baseline_delta0_lam0", r"baseline, $\delta=\lambda=0$", C3, "-^"),
                            ("baseline", "baseline (information lead)", C1, "--s")):
        y = [t5[key]["total_mean"][str(L)] for L in Ls]
        ax.plot(Ls, y, st, color=c, ms=4, lw=1.3, label=lab)
    ax.set_xlabel(r"production delay $L$")
    ax.set_ylabel(r"total failures, $T=400$ (node-periods)")
    ax.set_title("(b) Cumulative failure mass")
    ax.legend(fontsize=6.5, frameon=False, loc="center right")
    ax.set_ylim(850, 1300)

    # (c) recovery time versus L with the Theorem 5 bound
    ax = axes[2]
    for key, lab, c, st in (("matched", "matched lags", C2, "-o"),
                            ("baseline", "baseline", C1, "--s")):
        y = [t5[key]["recovery_mean"][str(L)] for L in Ls]
        ax.plot(Ls, y, st, color=c, ms=4, lw=1.3, label=lab)
    y0 = t5["matched"]["recovery_mean"]["0"]
    ax.plot(Ls, [y0 + 3 * L for L in Ls], ":", color="k", lw=0.9, label=r"slope $K-1=3$ (upper envelope)")
    ax.set_xlabel(r"production delay $L$")
    ax.set_ylabel("recovery time (periods)")
    ax.set_title("(c) Duration grows, at most linearly")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig7_theory.pdf"); fig.savefig("fig7_theory.png", dpi=600); plt.close(fig)


if __name__ == "__main__":
    fig7_theory()
