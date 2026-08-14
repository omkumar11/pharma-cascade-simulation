"""
figures_v2.py — regenerate all manuscript figures from results_decisions.json
and results_v2.json (corrected model). Run after decisions.py and
revision_suite.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
DEC = json.load(open("results_decisions.json"))
V2 = json.load(open("results_v2.json"))
C1, C2, C3, C4 = "#B44E2C", "#2C6FB4", "#5E8C61", "#8C5E8C"


def fig1_transition():
    d = DEC["d4"]; epss = d["epss"]
    labels = {"reqmin/binary": ("requirements form + indicator damping", C1, "-o"),
              "reqmin/smooth": ("requirements form + smooth damping", C1, "--s"),
              "ces-10/binary": ("CES + indicator damping", C2, "-o"),
              "ces-10/smooth": ("CES + smooth damping (baseline)", C2, "--s")}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for key, (lab, c, style) in labels.items():
        ax.plot(epss, d["spread"][key], style, color=c, label=lab, ms=4,
                lw=1.4, alpha=0.9 if "smooth" in key else 0.55)
    ax.axvline(0.70, color="gray", lw=0.8, ls=":")
    ax.annotate(r"seeding threshold $\epsilon=1-\tau$", (0.701, 60),
                fontsize=8, color="gray")
    ax.set_xlabel(r"shock magnitude $\epsilon$")
    ax.set_ylabel("cascade spread (% of nodes)")
    ax.set_title("Discontinuous coupling manufactures the all-or-nothing jump")
    ax.legend(fontsize=7.5, frameon=False)
    fig.savefig("fig1_transition.pdf"); plt.close(fig)


def fig2_delay():
    b2 = V2["b2"]
    Ls = sorted(int(k) for k in b2["T50"])
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, key, ttl in ((axes[0], "T50", "T = 50 (censored 19-25%)"),
                         (axes[1], "T200", "T = 200 (uncensored)")):
        mu = [b2[key][str(L)]["total"][0] for L in Ls]
        lo = [b2[key][str(L)]["total"][2] for L in Ls]
        hi = [b2[key][str(L)]["total"][3] for L in Ls]
        ax.plot(Ls, mu, "-o", color=C2, ms=4)
        ax.fill_between(Ls, lo, hi, color=C2, alpha=0.2, lw=0)
        ax.set_xlabel(r"production delay $L$")
        ax.set_ylabel("total node-period failures")
        ax.set_title(ttl)
    mu = [b2["T200"][str(L)]["recov"][0] for L in Ls]
    lo = [b2["T200"][str(L)]["recov"][2] for L in Ls]
    hi = [b2["T200"][str(L)]["recov"][3] for L in Ls]
    axes[2].plot(Ls, mu, "-o", color=C1, ms=4)
    axes[2].fill_between(Ls, lo, hi, color=C1, alpha=0.2, lw=0)
    co = np.polyfit(Ls, mu, 1)
    axes[2].plot(Ls, np.polyval(co, Ls), "--", color="k", lw=0.9,
                 label=f"linear fit: {co[0]:.1f} periods / unit $L$")
    axes[2].set_xlabel(r"$L$"); axes[2].set_ylabel("recovery time (periods)")
    axes[2].set_title("Delay stretches the transient")
    axes[2].legend(fontsize=7.5, frameon=False)
    fig.savefig("fig2_delay.pdf"); plt.close(fig)


def fig3_lambda():
    b3 = V2["b3"]
    lams = [k for k in b3 if k.replace('.', '').replace('-', '').isdigit()]
    lams = sorted(float(k) for k in lams)
    mu = [b3[str(g if g % 1 else int(g)) if str(g) in b3 else str(g)]  # robust key
          for g in []]
    def key(g):
        for cand in (str(g), str(int(g)) if g == int(g) else None, f"{g:.1f}"):
            if cand and cand in b3: return cand
        raise KeyError(g)
    mu = [b3[key(g)]["total"][0] for g in lams]
    lo = [b3[key(g)]["total"][2] for g in lams]
    hi = [b3[key(g)]["total"][3] for g in lams]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax.plot(lams, mu, "-o", color=C2, ms=4)
    ax.fill_between(lams, lo, hi, color=C2, alpha=0.2, lw=0)
    ax.set_xlabel(r"sourcing sensitivity $\lambda$")
    ax.set_ylabel("total node-period failures")
    ax.set_title("Corrected model: monotone benefit (95% CI)")
    d3 = DEC["d3"]
    ax2.plot(d3["lams"], d3["failures"], "-o", color=C1, ms=4)
    ax2.axvline(2.0, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel("total failures")
    ax2.set_title("Indicator-coupled variant: artifactual\ninterior optimum "
                  "(fine grid)")
    fig.savefig("fig3_lambda.pdf"); plt.close(fig)


def fig4_targeting():
    b1 = V2["b1"]
    order = ["baseline", "random", "uniform", "pagerank", "hub", "outdeg",
             "katz", "tier1_uniform", "C_targeted", "greedy"]
    names = {"baseline": "no control", "random": "random", "uniform": "uniform",
             "pagerank": "PageRank", "hub": "HITS hub", "outdeg": "out-degree",
             "katz": "Katz", "tier1_uniform": "tier-1 uniform",
             "C_targeted": r"$\chi(j)$ (ours)", "greedy": "greedy (coarse)"}
    mu = [b1[k]["total"][0] for k in order]
    err = [(b1[k]["total"][0] - b1[k]["total"][2],
            b1[k]["total"][3] - b1[k]["total"][0]) if len(b1[k]["total"]) > 2
           else (0, 0) for k in order]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    ypos = np.arange(len(order))
    cols = [C1 if k in ("C_targeted",) else (C3 if k == "greedy" else C2)
            for k in order]
    ax.barh(ypos, mu, xerr=np.array(err).T, color=cols, alpha=0.85, height=0.62)
    ax.set_yticks(ypos); ax.set_yticklabels([names[k] for k in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("total node-period failures (95% CI)")
    ax.set_title(f"Allocation rules at matched budget")
    b9 = V2["b9"]["budget"]
    Bs = sorted(float(b) for b in b9)
    for nm, c in (("C", C1), ("tier1", C4), ("uniform", C2)):
        ax2.plot(Bs, [b9[str(int(b)) if str(int(b)) in b9 else str(b)][nm]
                      for b in Bs], "-o", ms=3.5, color=c,
                 label={"C": "C(j)", "tier1": "tier-1 uniform",
                        "uniform": "uniform"}[nm])
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("budget B"); ax2.set_ylabel("failure reduction (%)")
    ax2.set_title("Budget sweep"); ax2.legend(fontsize=7.5, frameon=False)
    fig.savefig("fig4_targeting.pdf"); plt.close(fig)


def fig5_dynamics():
    from model import build_network
    from decisions import simulate_v
    from revision_suite import BASE, centralities, alloc_from_score
    A, sup, tier, _ = build_network()
    cent = centralities(A)
    u = alloc_from_score(cent["C"])
    fb = np.array([simulate_v(sup, tier, run_seed=s, **BASE)["F"].sum(1)
                   for s in range(30)])
    ft = np.array([simulate_v(sup, tier, run_seed=s, u=u, **BASE)["F"].sum(1)
                   for s in range(30)])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0))
    t = np.arange(fb.shape[1])
    for arr, c, lab in ((fb, C2, "baseline"), (ft, C1, "C(j)-targeted")):
        mu = arr.mean(0); se = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        ax.plot(t, mu, color=c, label=lab, lw=1.4)
        ax.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.2, lw=0)
    ax.set_xlabel("period"); ax.set_ylabel("simultaneous failures")
    ax.set_title("Cascade dynamics (mean, 95% CI)"); ax.legend(fontsize=8, frameon=False)
    rec_b = [int(np.nonzero(r)[0].max()) if r.any() else 0 for r in fb > 0]
    rec_t = [int(np.nonzero(r)[0].max()) if r.any() else 0 for r in ft > 0]
    ax2.boxplot([rec_b, rec_t], tick_labels=["baseline", "targeted"], widths=0.5)
    ax2.set_ylabel("recovery time (periods)")
    ax2.set_title("Recovery time (own axis)")
    fig.savefig("fig5_dynamics.pdf"); plt.close(fig)


def fig6_quasi():
    b6 = V2["b6"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(b6["quasi_grid"], b6["quasi_payoff"], "-o", color=C2, ms=3.5)
    ax.set_xlabel(r"own investment $u_j$ (tier-1 firm)")
    ax.set_ylabel("private payoff (failure cost + invest cost)")
    ax.set_title("Numerical check of payoff unimodality")
    fig.savefig("fig6_quasi.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig1_transition(); print("fig1 ok")
    fig2_delay(); print("fig2 ok")
    fig3_lambda(); print("fig3 ok")
    fig4_targeting(); print("fig4 ok")
    fig5_dynamics(); print("fig5 ok")
    fig6_quasi(); print("fig6 ok")
