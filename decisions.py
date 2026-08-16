"""
decisions.py — the four load-bearing referee experiments, run BEFORE any
rewrite. Outcomes determine what the revised paper can claim.

D1  Single-node aggregator diagnostic (referee I.2 table) + binding-channel
    decomposition.
D2  Disjoint- vs shared-supplier two-tier networks, lambda sweep, ORIGINAL
    aggregator: if the interior optimum persists with disjoint suppliers,
    herding is falsified.
D3  Fine lambda grid (0.25 spacing) on the full seed-42 network, original
    model, 100 CRN runs, plus realized order-concentration HHI(lambda).
D4  Smooth- vs binary-damping epsilon sweep through the threshold: does the
    all-or-nothing jump survive smooth coupling? Run for BOTH the original
    requirements-form aggregator and the corrected CES(-10) aggregator.
D5  Concentration-predicts-growth regression: dNfail(t+1) on C(t)
    controlling for Nfail(t), run fixed effects, 30 baseline runs.
D6  Corrected-model (CES rho=-10, smooth damping, L=3) baseline feel.

Usage: python3 decisions.py
"""
import json
import numpy as np
from model import build_network, metrics, TIER_SIZES

TAU, XBAR, ALPHA, H0, DELTA, PREC = 0.3, 1.0, 1.2, 0.1, 0.2, 0.06
EPSR = (0.6, 0.9)


# ----------------------------------------------------------------------
# Flexible simulator: aggregator in {reqmin, ces}; damping in {binary, smooth}
# Random protocol identical to model.simulate (CRN-compatible).
# ----------------------------------------------------------------------
def simulate_v(suppliers, tier, *, run_seed=0, agg="reqmin", rho=-10.0,
               damping="binary", lam=2.0, L=2, delta=DELTA, alpha=ALPHA,
               tau=TAU, h0=H0, eps=None, shock_frac=0.25, T=50, u=None,
               collect_w=False, sourcing="indicator", kappa=0.02):
    """sourcing: "indicator" (baseline; weights respond to f_i = 1{x_i < tau})
    or "logistic" (smooth surrogate; weights respond to
    s_kappa(x_i) = 1/(1+exp((x_i - tau)/kappa)), which -> f_i as kappa -> 0).
    Default behavior is byte-identical to the archived release."""
    rng = np.random.default_rng(run_seed)
    n = len(tier)
    t1 = np.nonzero(tier == 0)[0]
    k = max(1, int(round(shock_frac * len(t1))))
    hit = rng.choice(t1, size=k, replace=False)
    m = np.array([max(len(s), 1) for s in suppliers])
    uj = np.zeros(n) if u is None else np.asarray(u, float)
    eps_j = np.zeros(n)
    if eps is not None:
        eps_j[hit] = eps
    else:
        eps_j[hit] = rng.uniform(*EPSR, size=len(hit))
    hist = np.ones((L + 1, n))
    x = np.ones(n)
    x[hit] = np.minimum(XBAR, XBAR * (1 - eps_j[hit]) + uj[hit])
    h = np.full(n, h0) + uj
    w = {j: np.full(len(suppliers[j]), 1.0 / len(suppliers[j]))
         for j in range(n) if len(suppliers[j])}
    shocked = np.zeros(n, bool); shocked[hit] = True
    F = np.zeros((T + 1, n), bool); F[0] = x < tau
    X = np.zeros((T + 1, n)); X[0] = x
    Wsum = np.zeros(n)   # realized demand weight received by each supplier
    Wcnt = 0

    def shortfall(xv):
        return np.maximum(0.0, (tau - xv) / tau)

    for t in range(T):
        xlag = hist[-1]; f = F[t].astype(float)
        rec = shocked & (rng.random(n) < PREC); shocked &= ~rec
        xn = np.empty(n); hn = h.copy()
        for j in range(n):
            S = suppliers[j]
            if len(S) == 0:
                if shocked[j]:
                    xn[j] = min(XBAR, XBAR * (1 - eps_j[j]) + uj[j])
                else:
                    xn[j] = XBAR
                continue
            wj = w[j]
            if agg == "reqmin":
                I = float(np.min(xlag[S] / np.clip(wj * m[j], 1e-9, None)))
                I = min(I, float(xlag[S].max()))
            elif agg == "ces":
                if abs(rho) < 1e-9:
                    I = float(np.exp(wj @ np.log(np.maximum(xlag[S], 1e-12))))
                else:
                    I = float((wj @ np.maximum(xlag[S], 1e-12) ** rho) ** (1.0 / rho))
            xt = min(XBAR, alpha * (I + h[j]))
            if damping == "binary":
                damp = max(0.0, 1.0 - delta * f[S].sum())
            else:
                damp = max(0.0, 1.0 - delta * shortfall(X[t][S]).sum())
            xn[j] = xt * damp
            hn[j] = max(0.0, h[j] + I - xn[j])
        for j in range(n):
            S = suppliers[j]
            if len(S):
                if sourcing == "logistic":
                    sig = 1.0 / (1.0 + np.exp(np.clip((X[t][S] - tau) / kappa,
                                                      -500, 500)))
                    e = np.exp(-lam * sig)
                else:
                    e = np.exp(-lam * f[S])
                w[j] = e / e.sum()
                if collect_w:
                    Wsum[S] += w[j]
        if collect_w:
            Wcnt += 1
        h = hn
        hist = np.vstack([xn, hist[:-1]])
        F[t + 1] = xn < tau; X[t + 1] = xn
    out = {"F": F, "X": X}
    if collect_w:
        share = Wsum / max(Wcnt, 1)
        s = share.sum()
        out["hhi"] = float(((share / s) ** 2).sum()) if s > 0 else 0.0
    return out


# ----------------------------------------------------------------------
def d1_single_node():
    print("=" * 72)
    print("D1: single-node aggregator diagnostic (m=5, one supplier at 0.29)")
    print("=" * 72)
    xs = np.array([0.29, 1.0, 1.0, 1.0, 1.0]); f = np.array([1., 0, 0, 0, 0])
    m = 5
    rows = []
    for lam in (0., 1., 2., 3., 5.):
        e = np.exp(-lam * f); w = e / e.sum()
        req = xs / (w * m)
        I = min(req.min(), xs.max())
        binding = "failed" if req.argmin() == 0 else "healthy"
        rows.append((lam, w[0] * m, w[1] * m, binding, I))
        print(f"  lam={lam:3.0f}: wf*m={w[0]*m:.3f} wh*m={w[1]*m:.3f} "
              f"binding={binding:>7} I={I:.3f}")
    Is = [r[4] for r in rows]
    imax = int(np.argmax(Is))
    print(f"  -> interior max of I at lam={rows[imax][0]:.0f} "
          f"(referee predicted lam=2: {'CONFIRMED' if rows[imax][0]==2 else 'NOT confirmed'})")
    print("  pathology check: supplier at 0.31 (healthy, uniform w): I=0.310;")
    e = np.exp(-2.0 * f); w = e / e.sum()
    req = np.array([0.29, 1, 1, 1, 1]) / (w * m)
    print(f"  supplier at 0.29 (failed, lam=2): I={min(req.min(),1.0):.3f} "
          "-> failure REWARDS the customer" )
    return {"table": rows}


def build_two_tier(seed, shared):
    """20 buyers. shared=False: 100 suppliers, 5 private each (herding
    impossible). shared=True: 20 suppliers, each buyer draws 5 of them
    (heavily shared)."""
    rng = np.random.default_rng(seed)
    if shared:
        ns, nb = 20, 20
    else:
        ns, nb = 100, 20
    n = ns + nb
    tier = np.array([0] * ns + [1] * nb)
    suppliers = [np.array([], dtype=int) for _ in range(n)]
    for b in range(nb):
        if shared:
            sel = rng.choice(ns, size=5, replace=False)
        else:
            sel = np.arange(5 * b, 5 * b + 5)
        suppliers[ns + b] = sel
    return suppliers, tier


def d2_disjoint(n_runs=100):
    print("=" * 72)
    print("D2: disjoint vs shared suppliers, ORIGINAL aggregator, lambda sweep")
    print("    (herding impossible in disjoint case by construction)")
    print("=" * 72)
    lams = [0., 1., 1.5, 2., 2.5, 3., 5.]
    out = {}
    for label, shared in (("disjoint", False), ("shared", True)):
        sup, tier = build_two_tier(7, shared)
        tot = {lam: [] for lam in lams}
        for s in range(n_runs):
            for lam in lams:
                r = simulate_v(sup, tier, run_seed=s, agg="reqmin",
                               damping="binary", lam=lam)
                tot[lam].append(int(r["F"].sum()))
        out[label] = {lam: (float(np.mean(v)), float(np.std(v, ddof=1)))
                      for lam, v in tot.items()}
        means = [out[label][lam][0] for lam in lams]
        argmin = lams[int(np.argmin(means))]
        print(f"  [{label:>8}] " + " ".join(f"l{lam:g}:{mu:7.1f}" for lam, mu
                                            in zip(lams, means)))
        rev = means[-1] > min(means) and argmin not in (lams[0], lams[-1])
        print(f"  [{label:>8}] interior minimum at lambda={argmin} | "
              f"reversal beyond minimum: {rev}")
    return out


def d3_fine_grid(n_runs=100):
    print("=" * 72)
    print("D3: fine lambda grid, full network, original model, CRN + HHI")
    print("=" * 72)
    from model import build_network
    A, sup, tier0, _ = build_network()
    tier = tier0
    lams = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    tot = {lam: [] for lam in lams}; hhi = {lam: [] for lam in lams}
    for s in range(n_runs):
        for lam in lams:
            r = simulate_v(sup, tier, run_seed=s, agg="reqmin",
                           damping="binary", lam=lam, collect_w=True)
            tot[lam].append(int(r["F"].sum())); hhi[lam].append(r["hhi"])
    means = [float(np.mean(tot[lam])) for lam in lams]
    hh = [float(np.mean(hhi[lam])) for lam in lams]
    for lam, mu, h in zip(lams, means, hh):
        print(f"  lam={lam:5.2f}: failures {mu:7.1f}  HHI {h:.5f}")
    print(f"  -> minimum at lambda={lams[int(np.argmin(means))]}")
    return {"lams": lams, "failures": means, "hhi": hh}


def d4_smooth_damping(n_runs=30, L=3):
    print("=" * 72)
    print("D4: binary vs smooth damping, epsilon sweep through tau threshold "
          f"(L={L}, the corrected-model baseline delay)")
    print("=" * 72)
    from model import build_network
    A, sup, tier, _ = build_network()
    epss = [0.65, 0.70, 0.705, 0.71, 0.72, 0.75, 0.80, 0.85, 0.90]
    out = {}
    for agg, rho, aggname in (("reqmin", None, "reqmin"), ("ces", -10.0, "ces-10")):
        for damping in ("binary", "smooth"):
            key = f"{aggname}/{damping}"
            sp = []
            for e in epss:
                r = [100.0 * simulate_v(sup, tier, run_seed=s, agg=agg,
                                        rho=(rho or -10.0), damping=damping,
                                        eps=e, L=L)["F"].any(0).sum() / len(tier)
                     for s in range(n_runs)]
                sp.append(float(np.mean(r)))
            out[key] = sp
            print(f"  [{key:>15}] " + " ".join(f"e{e:g}:{v:5.1f}" for e, v
                                               in zip(epss, sp)))
    print("  (jump at 0.705 under binary vs continuous rise under smooth?)")
    return {"epss": epss, "spread": out, "L": L}


def d5_regression(n_runs=30):
    print("=" * 72)
    print("D5: does concentration C(t) predict cascade growth given Nfail(t)?")
    print("=" * 72)
    from model import build_network, simulate
    from robustness import curvature_series, embeddings
    A, sup, tier, ranks = build_network()
    Z = embeddings(A, tier, ranks)["tier_rank"]
    rows = []
    for s in range(n_runs):
        F = simulate(A, sup, tier, run_seed=s)
        cs = curvature_series(F, Z, sigma=0.15, grid=50)
        nf = F.sum(1).astype(float)
        for t in range(len(nf) - 1):
            rows.append((s, cs[t][0], nf[t], nf[t + 1] - nf[t]))
    rows = np.array(rows)
    runs, C, N, dN = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    keep = N > 0
    runs, C, N, dN = runs[keep], C[keep], N[keep], dN[keep]
    Cn = (C - C.mean()) / C.std(); Nn = (N - N.mean()) / N.std()
    dummies = np.array([(runs == s).astype(float) for s in np.unique(runs)]).T
    Xd = np.column_stack([Cn, Nn, dummies])
    beta, *_ = np.linalg.lstsq(Xd, dN, rcond=None)
    resid = dN - Xd @ beta
    dof = len(dN) - Xd.shape[1]
    XtXinv = np.linalg.pinv(Xd.T @ Xd)
    # cluster-robust (by run) covariance
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for s in np.unique(runs):
        sel = runs == s
        g = Xd[sel].T @ resid[sel]
        meat += np.outer(g, g)
    V = XtXinv @ meat @ XtXinv
    se = np.sqrt(np.diag(V))
    tC, tN = beta[0] / se[0], beta[1] / se[1]
    print(f"  obs={len(dN)}, runs={len(np.unique(runs))}")
    print(f"  beta_C (std.)  = {beta[0]:+.4f}  (cluster t = {tC:+.2f})")
    print(f"  beta_N (std.)  = {beta[1]:+.4f}  (cluster t = {tN:+.2f})")
    verdict = "PREDICTIVE" if (beta[0] > 0 and abs(tC) > 2) else \
              ("NEGATIVE-predictive" if (beta[0] < 0 and abs(tC) > 2) else "NOT predictive")
    print(f"  -> concentration is {verdict} of next-period cascade growth")
    return {"beta_C": float(beta[0]), "t_C": float(tC),
            "beta_N": float(beta[1]), "t_N": float(tN), "n": int(len(dN))}


def d6_corrected_baseline(n_runs=30):
    print("=" * 72)
    print("D6: corrected model baseline (CES rho=-10, smooth damping, L=3)")
    print("=" * 72)
    from model import build_network
    A, sup, tier, _ = build_network()
    for L in (2, 3):
        ms = []
        for s in range(n_runs):
            F = simulate_v(sup, tier, run_seed=s, agg="ces", rho=-10.0,
                           damping="smooth", lam=2.0, L=L)["F"]
            ms.append(metrics(F))
        tt = [m["total"] for m in ms]; sp = [m["spread"] for m in ms]
        print(f"  L={L}: failures {np.mean(tt):7.1f} +- {np.std(tt,ddof=1):6.1f} "
              f"| spread {np.mean(sp):5.1f}%")
    return {}


if __name__ == "__main__":
    res = {}
    res["d1"] = d1_single_node()
    res["d2"] = d2_disjoint()
    res["d3"] = d3_fine_grid()
    res["d4"] = d4_smooth_damping()
    res["d5"] = d5_regression()
    res["d6"] = d6_corrected_baseline()
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    json.dump(clean(res), open("results_decisions.json", "w"), indent=1)
    print("\nwrote results_decisions.json")
