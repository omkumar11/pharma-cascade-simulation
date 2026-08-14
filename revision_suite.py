"""
revision_suite.py — full corrected-model experiment suite for the major
revision. Corrected baseline (M2): CES aggregator rho=-10 with adaptive
softmax weights, SMOOTH damping (damage proportional to shortfall depth),
L=3, lambda=2, tau=0.3, 100 runs with common random numbers.

Outputs: results_v2.json. Run: python3 revision_suite.py
"""
import json
import numpy as np
from scipy import stats as st
from model import build_network, TIER_SIZES
from decisions import simulate_v

N = sum(TIER_SIZES)
BUDGET = 4.0
NRUNS = 100
BASE = dict(agg="ces", rho=-10.0, damping="smooth", lam=2.0, L=3)


def total(F): return int(F.sum())
def spread(F): return 100.0 * F.any(0).sum() / F.shape[1]
def peak(F): return int(F.sum(1).max())
def recov(F):
    nz = np.nonzero(F.any(1))[0]
    return int(nz.max()) if len(nz) else 0


def summarize(vals):
    a = np.asarray(vals, float)
    return [float(a.mean()), float(a.std(ddof=1)),
            float(a.mean() - 1.96 * a.std(ddof=1) / np.sqrt(len(a))),
            float(a.mean() + 1.96 * a.std(ddof=1) / np.sqrt(len(a)))]


def paired(a, b, n_boot=10000, seed=7):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = d[d != 0]
    w = st.wilcoxon(nz).pvalue if len(nz) else 1.0
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
                wilcoxon=float(w), frac_neg=float((d < 0).mean()))


def run_metrics(sup, tier, seeds, **kw):
    cfg = {**BASE, **kw}
    out = {"total": [], "spread": [], "peak": [], "recov": [], "cens": 0}
    for s in seeds:
        F = simulate_v(sup, tier, run_seed=s, **cfg)["F"]
        out["total"].append(total(F)); out["spread"].append(spread(F))
        out["peak"].append(peak(F)); out["recov"].append(recov(F))
        out["cens"] += int(F[-1].any())
    return out


# ----------------------------------------------------------------------
# Centralities on the DAG
# ----------------------------------------------------------------------
def centralities(A):
    n = A.shape[0]
    Af = A.astype(float)
    # discounted reachability C(j), eta=0.5 (ours)
    from model import centrality
    C = centrality(A)
    out_deg = Af.sum(1)
    # Katz downstream influence: k = sum_l beta^l A^l 1  (A nilpotent -> finite)
    beta = 0.5
    k = np.zeros(n); v = np.ones(n); Al = Af.copy(); bl = beta
    for _ in range(6):
        k += bl * (Al @ v); Al = Al @ Af; bl *= beta
        if not Al.any(): break
    # PageRank on reversed graph (importance as a supplier)
    d = 0.85; pr = np.full(n, 1.0 / n)
    Arev = Af.T
    colsum = Arev.sum(0); colsum[colsum == 0] = 1.0
    M = Arev / colsum
    for _ in range(100):
        pr = (1 - d) / n + d * (M @ pr)
    # HITS hub score (hubs point to many authorities downstream)
    hub = np.ones(n)
    for _ in range(50):
        auth = Af.T @ hub; auth /= max(auth.max(), 1e-12)
        hub = Af @ auth; hub /= max(hub.max(), 1e-12)
    return dict(C=C, outdeg=out_deg, katz=k, pagerank=pr, hub=hub)


def alloc_from_score(score, B=BUDGET):
    s = np.maximum(score, 0.0)
    return B * s / s.sum() if s.sum() > 0 else np.full(len(score), B / len(score))


# ----------------------------------------------------------------------
def b1_targeting(A, sup, tier, seeds):
    print("=" * 72); print("B1: targeting benchmarks, corrected model, "
                           f"{len(seeds)} CRN runs, B={BUDGET}")
    cent = centralities(A)
    n = len(tier)
    t1 = tier == 0
    rng = np.random.default_rng(123)
    allocs = {
        "baseline": None,
        "uniform": np.full(n, BUDGET / n),
        "C_targeted": alloc_from_score(cent["C"]),
        "tier1_uniform": np.where(t1, BUDGET / t1.sum(), 0.0),
        "outdeg": alloc_from_score(cent["outdeg"]),
        "katz": alloc_from_score(cent["katz"]),
        "pagerank": alloc_from_score(cent["pagerank"]),
        "hub": alloc_from_score(cent["hub"]),
        "random": None,  # handled per-run
    }
    res = {}
    raw = {}
    for name, u in allocs.items():
        tot = []
        oth = {"spread": [], "peak": [], "recov": []}
        for s in seeds:
            uu = u
            if name == "random":
                rr = np.random.default_rng(1000 + s)
                uu = np.zeros(n); pick = rr.choice(n, 10, replace=False)
                uu[pick] = BUDGET / 10
            F = simulate_v(sup, tier, run_seed=s, u=uu, **BASE)["F"]
            tot.append(total(F)); oth["spread"].append(spread(F))
            oth["peak"].append(peak(F)); oth["recov"].append(recov(F))
        raw[name] = tot
        res[name] = {"total": summarize(tot), "spread": summarize(oth["spread"]),
                     "peak": summarize(oth["peak"]), "recov": summarize(oth["recov"])}
        print(f"  {name:>14}: total {res[name]['total'][0]:7.1f} "
              f"[{res[name]['total'][2]:.0f},{res[name]['total'][3]:.0f}] "
              f"spread {res[name]['spread'][0]:5.1f}%")
    # greedy (simulation-based upper bound), 8 chunks, tier-1 + top tier-2 cands
    order = np.argsort(-cent["C"])
    cands = [int(j) for j in order[:30]]
    u = np.zeros(n); chunk = BUDGET / 8
    gseeds = list(range(20))
    for _ in range(8):
        best, bestv = None, np.inf
        for j in cands:
            u2 = u.copy(); u2[j] += chunk
            v = np.mean([total(simulate_v(sup, tier, run_seed=s, u=u2, **BASE)["F"])
                         for s in gseeds])
            if v < bestv: bestv, best = v, j
        u[best] += chunk
    tot = [total(simulate_v(sup, tier, run_seed=s, u=u, **BASE)["F"]) for s in seeds]
    raw["greedy"] = tot
    res["greedy"] = {"total": summarize(tot)}
    print(f"  {'greedy':>14}: total {res['greedy']['total'][0]:7.1f}")
    # paired stats vs baseline and vs tier1_uniform
    res["paired"] = {
        "C_vs_baseline": paired(raw["C_targeted"], raw["baseline"]),
        "C_vs_uniform": paired(raw["C_targeted"], raw["uniform"]),
        "C_vs_tier1": paired(raw["C_targeted"], raw["tier1_uniform"]),
        "uniform_vs_baseline": paired(raw["uniform"], raw["baseline"]),
    }
    for k, v in res["paired"].items():
        print(f"  paired {k}: {v['mean']:+7.1f} [{v['lo']:.1f},{v['hi']:.1f}] "
              f"wilcoxon p={v['wilcoxon']:.2g}")
    res["greedy_alloc_nodes"] = [int(i) for i in np.nonzero(u)[0]]
    return res, raw


def b2_delay(sup, tier, seeds):
    print("=" * 72); print("B2: delay sweep, corrected model")
    res = {"T50": {}, "T200": {}}
    for L in (1, 2, 3, 4, 5, 6):
        r = run_metrics(sup, tier, seeds, L=L)
        res["T50"][L] = {k: summarize(v) for k, v in r.items() if k != "cens"}
        res["T50"][L]["censored_frac"] = r["cens"] / len(seeds)
        print(f"  L={L} T=50 : total {res['T50'][L]['total'][0]:7.1f} "
              f"censored {res['T50'][L]['censored_frac']:.2f}")
    for L in (1, 2, 3, 4, 5, 6):
        r = run_metrics(sup, tier, range(30), L=L, T=200)
        res["T200"][L] = {k: summarize(v) for k, v in r.items() if k != "cens"}
        res["T200"][L]["censored_frac"] = r["cens"] / 30
        print(f"  L={L} T=200: total {res['T200'][L]['total'][0]:7.1f} "
              f"recov {res['T200'][L]['recov'][0]:5.1f} "
              f"censored {res['T200'][L]['censored_frac']:.2f}")
    return res


def b3_lambda(sup, tier, seeds):
    print("=" * 72); print("B3: lambda sweep, corrected model, CRN + HHI")
    lams = [0., 0.5, 1., 1.5, 2., 2.5, 3., 4., 5.]
    res = {}
    raws = {}
    for lam in lams:
        tot, hhi = [], []
        for s in seeds:
            r = simulate_v(sup, tier, run_seed=s, u=None, collect_w=True,
                           **{**BASE, "lam": lam})
            tot.append(total(r["F"])); hhi.append(r["hhi"])
        raws[lam] = tot
        res[lam] = {"total": summarize(tot), "hhi": float(np.mean(hhi))}
        print(f"  lam={lam:3g}: total {res[lam]['total'][0]:7.1f}  "
              f"HHI {res[lam]['hhi']:.5f}")
    means = [res[lam]["total"][0] for lam in lams]
    mono = all(means[i+1] <= means[i] + 1e-9 for i in range(len(means)-1))
    res["monotone_decreasing"] = bool(mono)
    res["paired_l5_vs_l2"] = paired(raws[5.], raws[2.])
    print(f"  monotone decreasing: {mono}; l5 vs l2 paired: "
          f"{res['paired_l5_vs_l2']['mean']:+.1f} "
          f"[{res['paired_l5_vs_l2']['lo']:.1f},{res['paired_l5_vs_l2']['hi']:.1f}]")
    return res


def b4_rho(sup, tier, seeds):
    print("=" * 72); print("B4: substitutability sweep (CES rho), corrected model")
    res = {}
    for rho in (-10., -5., -2., -1., -0.5, 0.0, 0.5, 1.0):
        r = run_metrics(sup, tier, seeds, rho=rho)
        res[rho] = {"total": summarize(r["total"]), "spread": summarize(r["spread"])}
        print(f"  rho={rho:5g}: total {res[rho]['total'][0]:8.1f} "
              f"spread {res[rho]['spread'][0]:5.1f}%")
    return res


def b5_tau_alpha(sup, tier, seeds):
    print("=" * 72); print("B5: tau, alpha, and delta sweeps, corrected model")
    res = {"tau": {}, "alpha": {}, "delta": {}}
    for dl in (0.0, 0.1, 0.2, 0.3, 0.4):
        r = run_metrics(sup, tier, seeds, delta=dl)
        res["delta"][dl] = {"total": summarize(r["total"])}
        print(f"  delta={dl:4g}: total {res['delta'][dl]['total'][0]:8.1f}")
    for tau in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        r = run_metrics(sup, tier, seeds, tau=tau)
        res["tau"][tau] = {"total": summarize(r["total"]),
                           "spread": summarize(r["spread"])}
        print(f"  tau={tau:4g}: total {res['tau'][tau]['total'][0]:8.1f} "
              f"spread {res['tau'][tau]['spread'][0]:5.1f}%")
    for a in (1.05, 1.1, 1.2, 1.35, 1.5):
        r = run_metrics(sup, tier, seeds, alpha=a)
        res["alpha"][a] = {"total": summarize(r["total"]),
                           "spread": summarize(r["spread"])}
        print(f"  alpha={a:4g}: total {res['alpha'][a]['total'][0]:8.1f} "
              f"spread {res['alpha'][a]['spread'][0]:5.1f}%")
    return res


def b6_externality(A, sup, tier):
    print("=" * 72); print("B6: externality, quasiconcavity, spillover sign")
    n = len(tier)
    BETA_C, PSI = 6e-4, 1.0
    GRID = np.round(np.arange(0.0, 0.32, 0.02), 2)
    tsizes = [int((tier == k).sum()) for k in range(4)]

    def eval_alloc(ut, runs=10):
        u = np.array([ut[tier[j]] for j in range(n)])
        per = np.zeros(4); tot_ = 0.0
        for s in range(runs):
            F = simulate_v(sup, tier, run_seed=s, u=u, **BASE)["F"]
            tot_ += F.sum()
            for k in range(4):
                per[k] += F[:, tier == k].sum() / tsizes[k]
        return per / runs, tot_ / runs

    def cost(g): return g * g / (2 * BETA_C)

    def solve(objective):
        ut = np.zeros(4)
        for _ in range(4):
            ch = False
            for k in range(4):
                best = (1e18, ut[k])
                for g in GRID:
                    t2 = ut.copy(); t2[k] = g
                    per, tot_ = eval_alloc(t2)
                    o = objective(k, per, tot_, t2)
                    if o < best[0] - 1e-9: best = (o, g)
                if best[1] != ut[k]: ch = True
                ut[k] = best[1]
            if not ch: break
        return ut

    u_br = solve(lambda k, per, tot_, t2: PSI * per[k] + cost(t2[k]))
    u_sp = solve(lambda k, per, tot_, t2: PSI * tot_ +
                 sum(tsizes[j] * cost(t2[j]) for j in range(4)))
    _, F_br = eval_alloc(u_br, runs=30)
    _, F_sp = eval_alloc(u_sp, runs=30)
    _, F_0 = eval_alloc(np.zeros(4), runs=30)
    print(f"  BR u={list(u_br)} -> F={F_br:.1f} | planner u={list(u_sp)} -> "
          f"F={F_sp:.1f} | none F={F_0:.1f} | wedge "
          f"{100*(F_br/F_sp-1):.1f}%")

    # quasiconcavity: firm-level payoff of a tier-1 firm vs own u, others at BR
    uprof = np.array([u_br[tier[j]] for j in range(n)])
    j0 = int(np.nonzero(tier == 0)[0][0])
    grid = np.round(np.arange(0.0, 0.31, 0.02), 2)
    payoff = []
    for g in grid:
        u2 = uprof.copy(); u2[j0] = g
        own = np.mean([simulate_v(sup, tier, run_seed=s, u=u2, **BASE)["F"][:, j0].sum()
                       for s in range(30)])
        payoff.append(float(-PSI * own - cost(g)))
    d = np.diff(payoff)
    sign_changes = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-9])) != 0))
    print(f"  quasiconcavity: payoff curve sign changes = {sign_changes} "
          f"(0 or 1 = unimodal)")

    # spillover sign: harden one tier-1 node, effect on OTHERS' failures
    deltas = []
    for s in range(100):
        F0 = simulate_v(sup, tier, run_seed=s, u=None, **BASE)["F"]
        u2 = np.zeros(n); u2[j0] = 0.06
        F1 = simulate_v(sup, tier, run_seed=s, u=u2, **BASE)["F"]
        others = np.arange(n) != j0
        deltas.append(int(F1[:, others].sum()) - int(F0[:, others].sum()))
    pr = paired(deltas, np.zeros(len(deltas)))
    print(f"  spillover on others: mean {np.mean(deltas):+.2f} "
          f"[{pr['lo']:.2f},{pr['hi']:.2f}] "
          f"(negative = hardening helps others) frac<0 runs among nonzero: "
          f"{np.mean(np.array(deltas)[np.array(deltas)!=0]<0) if any(d!=0 for d in deltas) else float('nan'):.2f}")
    return dict(u_br=[float(x) for x in u_br], u_sp=[float(x) for x in u_sp],
                F_br=float(F_br), F_sp=float(F_sp), F_0=float(F_0),
                quasi_grid=[float(g) for g in grid], quasi_payoff=payoff,
                quasi_sign_changes=sign_changes,
                spillover_mean=float(np.mean(deltas)),
                spillover_ci=[pr["lo"], pr["hi"]])


def b7_rc(A, sup, tier):
    print("=" * 72); print("B7: cascade number R_c = rho(J) numerics")
    n = len(tier)
    is_src = np.array([len(s) == 0 for s in sup])
    ns = int((~is_src).sum())
    idx = np.nonzero(~is_src)[0]
    pos = {j: i for i, j in enumerate(idx)}
    TAU, XBAR = 0.3, 1.0

    def one_step(xlags, h, L, delta, alpha, rho=-10.0, xsrc=0.25):
        # xlags: (L+1, ns) newest-first among non-source nodes
        xfull_lag = np.full(n, xsrc); xfull_lag[idx] = xlags[-1]
        xfull_now = np.full(n, xsrc); xfull_now[idx] = xlags[0]
        xn = np.empty(ns); hn = h.copy()
        for ii, j in enumerate(idx):
            S = sup[j]
            w = np.full(len(S), 1.0 / len(S))
            I = float((w @ np.maximum(xfull_lag[S], 1e-12) ** rho) ** (1.0 / rho))
            xt = min(XBAR, alpha * (I + h[ii]))
            sf = np.maximum(0.0, (TAU - xfull_now[S]) / TAU)
            damp = max(0.0, 1.0 - delta * sf.sum())
            xn[ii] = xt * damp
            hn[ii] = max(0.0, h[ii] + I - xn[ii])
        new_lags = np.vstack([xn, xlags[:-1]])
        return new_lags, hn

    def jacobian(L, delta, alpha, x0=0.25, h0=0.05):
        dim = ns * (L + 1) + ns
        s0 = np.concatenate([np.full(ns * (L + 1), x0), np.full(ns, h0)])
        def f(svec):
            xl = svec[:ns * (L + 1)].reshape(L + 1, ns)
            h = svec[ns * (L + 1):]
            nl, nh = one_step(xl, h, L, delta, alpha)
            return np.concatenate([nl.ravel(), nh])
        f0 = f(s0)
        J = np.zeros((dim, dim)); ep = 1e-6
        for k in range(dim):
            sp_ = s0.copy(); sp_[k] += ep
            J[:, k] = (f(sp_) - f0) / ep
        return J

    res = {"L": {}, "delta": {}, "alpha": {}}
    for L in (1, 2, 3, 4, 5, 6):
        rc = float(np.abs(np.linalg.eigvals(jacobian(L, 0.2, 1.2))).max())
        res["L"][L] = rc
        print(f"  R_c(L={L}) = {rc:.4f}")
    for dl in (0.0, 0.1, 0.2, 0.3, 0.4):
        rc = float(np.abs(np.linalg.eigvals(jacobian(3, dl, 1.2))).max())
        res["delta"][dl] = rc
        print(f"  R_c(delta={dl}) = {rc:.4f}")
    for al in (1.05, 1.1, 1.2, 1.35, 1.5):
        rc = float(np.abs(np.linalg.eigvals(jacobian(3, 0.2, al))).max())
        res["alpha"][al] = rc
        print(f"  R_c(alpha={al}) = {rc:.4f}")
    # theorem check: delta=0, inventory frozen -> mu^L = eig(alpha * D_I)
    L = 3
    Jt = jacobian(L, 0.0, 1.2, h0=0.0)
    # zero out inventory rows/cols to isolate input channel
    Jt[ns * (L + 1):, :] = 0.0; Jt[:, ns * (L + 1):] = 0.0
    mu = np.abs(np.linalg.eigvals(Jt)).max()
    # direct A: A_ij = alpha * dI_j/dx_i at symmetric point = alpha * w_ij*(x/I)^(rho-1) = alpha*w_ij
    Amat = np.zeros((ns, ns))
    for ii, j in enumerate(idx):
        S = sup[j]
        for i in S:
            if i in pos:
                Amat[ii, pos[i]] = 1.2 / len(S)
    rhoA = np.abs(np.linalg.eigvals(Amat)).max()
    pred = rhoA ** (1.0 / L)
    print(f"  theorem check L={L}: rho(J_input) = {mu:.4f} vs rho(A)^(1/L) = "
          f"{pred:.4f}  (rho(A)={rhoA:.4f})")
    res["theorem_check"] = {"rho_J_input": float(mu), "rho_A_pow": float(pred),
                            "rho_A": float(rhoA)}
    return res


def b8_topology(seeds20=range(20), runs_per=20):
    print("=" * 72); print("B8: topology replication (ER + PA), corrected model")
    from new_experiments import build_network_pa
    from model import centrality
    res = {}
    for topo in ("er", "pa"):
        red_c, red_u, red_t1, wins = [], [], [], 0
        for ts in seeds20:
            if topo == "er":
                A, sup, tier, _ = build_network(seed=10000 + ts)
            else:
                A, sup, tier = build_network_pa(10000 + ts)
            n = len(tier)
            C = centrality(A)
            t1 = tier == 0
            allocs = {"baseline": None,
                      "uniform": np.full(n, BUDGET / n),
                      "C_targeted": alloc_from_score(C),
                      "tier1_uniform": np.where(t1, BUDGET / t1.sum(), 0.0)}
            sums = {k: 0.0 for k in allocs}
            for r in range(runs_per):
                for name, u in allocs.items():
                    F = simulate_v(sup, tier, run_seed=100 + r, u=u, **BASE)["F"]
                    sums[name] += F.sum()
            base = sums["baseline"] / runs_per
            if base <= 0: continue
            rc = 100 * (1 - sums["C_targeted"] / runs_per / base)
            ru = 100 * (1 - sums["uniform"] / runs_per / base)
            rt = 100 * (1 - sums["tier1_uniform"] / runs_per / base)
            red_c.append(rc); red_u.append(ru); red_t1.append(rt)
            wins += rc > ru
        res[topo] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
                     for k, v in (("C", red_c), ("uniform", red_u),
                                  ("tier1", red_t1))}
        res[topo]["wins_C_over_uniform"] = int(wins)
        print(f"  [{topo}] C: {res[topo]['C'][0]:.1f}% "
              f"({res[topo]['C'][1]:.1f}-{res[topo]['C'][2]:.1f}) | uniform "
              f"{res[topo]['uniform'][0]:.1f}% | tier1 {res[topo]['tier1'][0]:.1f}% "
              f"| C>u in {wins}/{len(red_c)}")
    return res


def b9_eta_budget(A, sup, tier, seeds):
    print("=" * 72); print("B9: eta rank-robustness and budget sweep")
    from collections import deque
    n = A.shape[0]
    def C_eta(eta):
        C = np.zeros(n)
        for j in range(n):
            dist = np.full(n, np.inf); dist[j] = 0
            q = deque([j])
            while q:
                uu = q.popleft()
                for v in np.nonzero(A[uu])[0]:
                    if dist[v] == np.inf:
                        dist[v] = dist[uu] + 1; q.append(v)
            reach = np.isfinite(dist) & (dist > 0)
            C[j] = float(np.sum(eta ** dist[reach]))
        return C
    C5 = C_eta(0.5)
    etas = [0.1, 0.3, 0.5, 0.7, 0.9]
    rho_rank = {}
    for e in etas:
        rho_rank[e] = float(st.spearmanr(C5, C_eta(e))[0])
    print("  Spearman rank corr of C(eta) vs C(0.5): " +
          " ".join(f"{e}:{rho_rank[e]:.3f}" for e in etas))
    Bs = [1., 2., 4., 8., 16.]
    bres = {}
    t1 = tier == 0
    for B in Bs:
        arr = {}
        for name, u in (("C", alloc_from_score(C5, B)),
                        ("uniform", np.full(n, B / n)),
                        ("tier1", np.where(t1, B / t1.sum(), 0.0)),
                        ("baseline", None)):
            tot = [total(simulate_v(sup, tier, run_seed=s, u=u, **BASE)["F"])
                   for s in seeds]
            arr[name] = float(np.mean(tot))
        bres[B] = {k: (100 * (1 - v / arr["baseline"]) if k != "baseline" else v)
                   for k, v in arr.items()}
        print(f"  B={B:4g}: C {bres[B]['C']:5.1f}% | uniform "
              f"{bres[B]['uniform']:5.1f}% | tier1 {bres[B]['tier1']:5.1f}%")
    return {"eta_rank_corr": rho_rank, "budget": bres}


def b10_size(runs_per=20):
    print("=" * 72); print("B10: size replication n=300/500, corrected model")
    from model import centrality
    res = {}
    for n_, sizes in ((300, (41, 72, 103, 84)), (500, (68, 120, 171, 141))):
        A, sup, tier, _ = build_network(seed=42, tier_sizes=sizes)
        B = BUDGET * n_ / N
        C = centrality(A); t1 = tier == 0
        allocs = {"baseline": None, "uniform": np.full(n_, B / n_),
                  "C": alloc_from_score(C, B),
                  "tier1": np.where(t1, B / t1.sum(), 0.0)}
        sums = {k: 0.0 for k in allocs}; sp = []
        for s in range(runs_per):
            for name, u in allocs.items():
                F = simulate_v(sup, tier, run_seed=s, u=u, **BASE)["F"]
                sums[name] += F.sum()
                if name == "baseline": sp.append(spread(F))
        base = sums["baseline"] / runs_per
        res[n_] = {"spread": float(np.mean(sp)),
                   "C": float(100 * (1 - sums["C"] / runs_per / base)),
                   "uniform": float(100 * (1 - sums["uniform"] / runs_per / base)),
                   "tier1": float(100 * (1 - sums["tier1"] / runs_per / base))}
        print(f"  n={n_}: spread {res[n_]['spread']:.1f}% | C {res[n_]['C']:.1f}% "
              f"| uniform {res[n_]['uniform']:.1f}% | tier1 {res[n_]['tier1']:.1f}%")
    return res


def b11_regression(sup, tier, A, ranks):
    print("=" * 72); print("B11: concentration regression, corrected model")
    from robustness import curvature_series, embeddings
    Z = embeddings(A, tier, ranks)["tier_rank"]
    rows = []
    for s in range(30):
        F = simulate_v(sup, tier, run_seed=s, **BASE)["F"]
        cs = curvature_series(F, Z, sigma=0.15, grid=50)
        nf = F.sum(1).astype(float)
        for t in range(len(nf) - 1):
            rows.append((s, cs[t][0], nf[t], nf[t + 1] - nf[t]))
    rows = np.array(rows)
    runs, C, Nn, dN = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    keep = Nn > 0
    runs, C, Nn, dN = runs[keep], C[keep], Nn[keep], dN[keep]
    Cn = (C - C.mean()) / C.std(); Nz = (Nn - Nn.mean()) / Nn.std()
    dummies = np.array([(runs == s).astype(float) for s in np.unique(runs)]).T
    Xd = np.column_stack([Cn, Nz, dummies])
    beta, *_ = np.linalg.lstsq(Xd, dN, rcond=None)
    resid = dN - Xd @ beta
    XtXinv = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for s in np.unique(runs):
        sel = runs == s
        g = Xd[sel].T @ resid[sel]
        meat += np.outer(g, g)
    V = XtXinv @ meat @ XtXinv
    tC = beta[0] / np.sqrt(V[0, 0])
    print(f"  beta_C = {beta[0]:+.3f} (cluster t = {tC:+.2f}), n={len(dN)}")
    return {"beta_C": float(beta[0]), "t_C": float(tC), "n": int(len(dN))}


def b12_cisplatin(sup, tier):
    print("=" * 72); print("B12: single-facility back-test (cisplatin-like)")
    # one tier-1 node, near-total loss (eps=0.95), long outage (no recovery
    # for first 12 periods enforced by seeding choice is not possible; report
    # distribution over recovery draws)
    onset, spr, dur = [], [], []
    for s in range(100):
        F = simulate_v(sup, tier, run_seed=s, shock_frac=0.05, eps=0.95,
                       **{**BASE, "T": 100})["F"]
        ds = np.nonzero(F[1:].any(1))[0]
        nf = F.sum(1)
        downstream = np.nonzero((F[:, tier > 0]).any(1))[0]
        onset.append(int(downstream.min()) if len(downstream) else -1)
        spr.append(spread(F)); dur.append(recov(F))
    ok = [o for o in onset if o >= 0]
    print(f"  downstream onset period: median {np.median(ok):.0f} "
          f"(L={BASE['L']}); spread {np.mean(spr):.1f}%; duration "
          f"median {np.median(dur):.0f} periods")
    return {"onset_median": float(np.median(ok)), "spread_mean": float(np.mean(spr)),
            "duration_median": float(np.median(dur))}


if __name__ == "__main__":
    A, sup, tier, ranks = build_network()
    seeds = list(range(NRUNS))
    out = {}
    out["b1"], raw1 = b1_targeting(A, sup, tier, seeds)
    out["b2"] = b2_delay(sup, tier, seeds)
    out["b3"] = b3_lambda(sup, tier, seeds)
    out["b4"] = b4_rho(sup, tier, seeds)
    out["b5"] = b5_tau_alpha(sup, tier, seeds)
    out["b6"] = b6_externality(A, sup, tier)
    out["b7"] = b7_rc(A, sup, tier)
    out["b8"] = b8_topology()
    out["b9"] = b9_eta_budget(A, sup, tier, seeds)
    out["b10"] = b10_size()
    out["b11"] = b11_regression(sup, tier, A, ranks)
    out["b12"] = b12_cisplatin(sup, tier)
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    json.dump(clean(out), open("results_v2.json", "w"), indent=1)
    np.save("raw_b1_totals.npy", np.array([raw1[k] for k in
        ("baseline", "uniform", "C_targeted", "tier1_uniform")]))
    print("\nwrote results_v2.json")
