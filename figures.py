"""figures.py — regenerates Figures 1-5 from results_e124.json + stashed arrays.
Run AFTER experiments.py:  python3 figures.py
"""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import build_network
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
R = json.load(open("results_e124.json"))
A, sup, tier, ranks = build_network()

S = np.load("series_e5.npy"); t = np.arange(S.shape[2]); E5 = R["E5_target"]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for arr, lab, col in ((S[0], "Baseline", "#c0392b"), (S[1], "Targeted control", "#2471a3")):
    mu, sd = arr.mean(0), arr.std(0)
    ax[0].plot(t, mu, color=col, label=lab, lw=1.6, ls="--" if "T" in lab[0] else "-")
    ax[0].fill_between(t, mu-sd, mu+sd, color=col, alpha=0.18)
ax[0].set_xlabel("Time period $t$"); ax[0].set_ylabel("Simultaneous failures")
ax[0].set_title("(a) Cascade dynamics"); ax[0].legend(frameon=False)
labels = ["Total\nfailures", "Peak\nfailures", "Recovery\ntime (×5)"]
b = [E5["baseline"]["total"][0], E5["baseline"]["peak"][0], 5*E5["baseline"]["recovery"][0]]
c = [E5["targeted"]["total"][0], E5["targeted"]["peak"][0], 5*E5["targeted"]["recovery"][0]]
be = [E5["baseline"]["total"][1], E5["baseline"]["peak"][1], 5*E5["baseline"]["recovery"][1]]
ce = [E5["targeted"]["total"][1], E5["targeted"]["peak"][1], 5*E5["targeted"]["recovery"][1]]
xp = np.arange(3)
ax[1].bar(xp-0.18, b, 0.34, yerr=be, color="#c0392b", label="Baseline", capsize=3)
ax[1].bar(xp+0.18, c, 0.34, yerr=ce, color="#2471a3", label="Controlled", capsize=3)
ax[1].set_xticks(xp, labels); ax[1].set_title("(b) Aggregate metrics")
ax[1].legend(frameon=False); ax[1].set_ylabel("Count (recovery ×5)")
fig.tight_layout(); fig.savefig("fig1_cascade.pdf"); fig.savefig("fig1_cascade.png")

E2 = R["E2_delay"]; Ls = sorted(int(k) for k in E2)
fig, ax = plt.subplots(1, 3, figsize=(9.5, 3))
for i, (key, name) in enumerate((("total","Total failures"),("peak","Peak failures"),
                                 ("recovery","Recovery time (periods)"))):
    mu = [E2[str(L)][key][0] for L in Ls]; sd = [E2[str(L)][key][1] for L in Ls]
    ax[i].errorbar(Ls, mu, yerr=sd, fmt="o-", color="#c0392b", capsize=3, ms=4)
    ax[i].annotate(f"{(mu[-1]/mu[0]-1)*100:+.0f}%", (Ls[-1], mu[-1]),
                   textcoords="offset points", xytext=(-24, 8), color="#c0392b")
    ax[i].set_xlabel("Delay $L$"); ax[i].set_title(name); ax[i].set_xticks(Ls)
fig.suptitle("Effect of production delay $L$", y=1.02)
fig.tight_layout(); fig.savefig("fig2_delay.pdf"); fig.savefig("fig2_delay.png")

E3 = R["E3_lambda"]; lams = E3["lams"]
F = [f for f,_ in E3["F"]]; Fe = [s for _,s in E3["F"]]
Csp = np.array([c for c,_ in E3["C"]["spectral"]]); Csp = Csp/Csp[0]
Ctr = np.array([c for c,_ in E3["C"]["tier_rank"]]); Ctr = Ctr/Ctr[0]
fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
ax1.errorbar(lams, F, yerr=Fe, fmt="o-", color="#c0392b", capsize=3,
             label="Aggregate fragility $\\mathcal{F}$")
ax1.set_xlabel("Sourcing sensitivity $\\lambda$")
ax1.set_ylabel("Aggregate fragility $\\mathcal{F}$", color="#c0392b")
ax1.axvspan(2, 5, color="#8e44ad", alpha=0.06)
ax1.annotate("herding zone:\n$\\mathcal{F}$ rises again", (3.1, F[0]*0.55),
             color="#8e44ad", fontsize=8)
ax2 = ax1.twinx()
ax2.plot(lams, Ctr, "s--", color="#8e44ad", label="max$_t\\,\\mathcal{C}$ (tier–rank)", ms=4)
ax2.plot(lams, Csp, "^:", color="#5b2c6f", label="max$_t\\,\\mathcal{C}$ (spectral)", ms=4)
ax2.set_ylabel("Normalised curvature (rel. $\\lambda=0$)", color="#8e44ad")
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, frameon=False, fontsize=8, loc="upper right")
ax1.set_title("Substitution–fragility with sourcing herding beyond $\\lambda=2$")
fig.tight_layout(); fig.savefig("fig3_lambda.pdf"); fig.savefig("fig3_lambda.png")

theta = np.load("theta.npy")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
sc = ax[0].scatter(tier/3.0, ranks, c=theta/theta.max(), cmap="YlOrRd", s=26,
                   edgecolor="k", lw=0.3)
plt.colorbar(sc, ax=ax[0], label="$\\Theta(j)$ (normalised)")
ax[0].set_xticks([0,1/3,2/3,1.0], ["Tier 1\n(API)","Tier 2\n(Synth.)","Tier 3\n(Form.)","Tier 4\n(Dist.)"])
ax[0].set_ylabel("Within-tier rank"); ax[0].set_title("(a) Targeting score $\\Theta(j)$")
for arr, lab, col, ls in ((S[0], f"Baseline ({E5['baseline']['spread'][0]:.1f}% spread)", "#c0392b", "-"),
                          (S[1], f"Targeted ({E5['targeted']['spread'][0]:.1f}% spread)", "#2471a3", "--"),
                          (S[2], f"Uniform ({E5['uniform']['spread'][0]:.1f}% spread)", "#7f8c8d", ":")):
    mu, sd = arr.mean(0), arr.std(0)
    ax[1].plot(t, mu, color=col, ls=ls, lw=1.5, label=lab)
    ax[1].fill_between(t, mu-sd, mu+sd, color=col, alpha=0.12)
ax[1].set_xlabel("Time period $t$"); ax[1].set_ylabel("Simultaneous failures")
ax[1].set_title("(b) Cascade suppression (matched budget $B=4$)")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig4_target.pdf"); fig.savefig("fig4_target.png")

Fb = np.load("F_baseline_runs.npy")[0]
z = np.column_stack([tier/3.0, ranks]); sigma = 0.15
def field_and_curv(fvec, G=90, pad=0.45):
    gx = np.linspace(-pad, 1+pad, G); cell=(gx[1]-gx[0])**2
    GX, GY = np.meshgrid(gx, gx); c=1/(2*np.pi*sigma**2)
    idx = np.nonzero(fvec)[0]
    if not len(idx): return GX, GY, np.zeros_like(GX), 0.0
    ux = GX[...,None]-z[idx,0]; uy = GY[...,None]-z[idx,1]
    K = c*np.exp(-(ux**2+uy**2)/(2*sigma**2))
    H11=(K*(ux*ux/sigma**4-1/sigma**2)).sum(-1)
    H22=(K*(uy*uy/sigma**4-1/sigma**2)).sum(-1)
    H12=(K*(ux*uy/sigma**4)).sum(-1)
    return GX, GY, K.sum(-1), float(((H11**2+2*H12**2+H22**2).sum())*cell)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), sharey=True)
vmax = None
for i, tt in enumerate([0, 5, 15, 35]):
    GX, GY, phi, Cval = field_and_curv(Fb[tt])
    if vmax is None: vmax = max(phi.max(), 1e-9)
    im = axes[i].pcolormesh(GX, GY, phi, cmap="YlOrBr", vmin=0, vmax=vmax, shading="auto")
    axes[i].scatter(z[:,0], z[:,1], c=np.where(tier==0, "#1a5276", "#d68910"), s=5, alpha=0.6)
    axes[i].set_title(f"$t={tt}$ ($n_{{fail}}={int(Fb[tt].sum())}$)\n$\\mathcal{{C}}={Cval:.2e}$",
                      fontsize=8.5)
    axes[i].set_xlim(-0.2, 1.2); axes[i].set_ylim(-0.2, 1.2); axes[i].set_xlabel("Tier axis")
axes[0].set_ylabel("Within-tier axis")
fig.colorbar(im, ax=axes, label="$\\phi(z,t)$", shrink=0.85)
fig.suptitle("Fragility field $\\phi(z,t)$, baseline run", y=1.04)
fig.savefig("fig5_field.pdf", bbox_inches="tight"); fig.savefig("fig5_field.png", bbox_inches="tight")
print("figures 1-5 regenerated")
