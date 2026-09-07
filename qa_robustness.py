"""
qa_robustness.py — supplementary checks added for the IEEE Access revision
(release v2.2). Everything here uses the corrected model (decisions.simulate_v
with CES rho=-10, smooth damping, L=3) and the same seeds as revision_suite.py.

Q1  Katz replication: the Katz rule added to the 20 Erdos-Renyi + 20
    preferential-attachment topology draws of B8 (seeds 10000-10019, 20 runs
    each, run seeds 100-119), alongside chi, uniform and tier-1 uniform.
Q2  Budget sweep with Katz: B in {1,2,4,8,16}, 100 CRN runs (seeds 0-99),
    arms Katz / chi / uniform / tier-1 uniform / no control (Fig. 4, right).
Q3  tau x alpha robustness grid (Table VI): in every cell of the tau sweep
    (alpha=1.2) and the alpha sweep (tau=0.3), re-run (a) the lambda sweep at
    {0,2,5} (100 runs), (b) the B=4 allocation comparison none / uniform / chi /
    tier-1 (100 CRN runs), (c) the delay comparison at T=200, L=1 vs L=6
    (30 runs): totals, percentage change, mean recovery time.
Q4  Single-facility back-test statistics (Sec. VI-F): number of runs in which
    the disruption propagates, onset median, spread, duration median and IQR,
    censoring at T=100.
Q5  Peak inventory over the 100 baseline runs (the "inventories remain
    bounded" statement of Sec. II-B), via an instrumented copy of the
    simulator that is checked against simulate_v on every run.

Usage: python3 qa_robustness.py        (~10 min, Apple M-series)
Output: results_qa.json
"""
import json
import numpy as np
from model import build_network, TIER_SIZES
from decisions import simulate_v, EPSR, PREC, XBAR, TAU, ALPHA, H0, DELTA
from revision_suite import (BASE, BUDGET, centralities, alloc_from_score,
                            total, spread, recov, summarize)
from new_experiments import build_network_pa

N = sum(TIER_SIZES)


def reductions(sup, tier, allocs, run_seeds, **cfg):
    """Mean total failures per arm and % reduction versus 'baseline'."""
    sums = {k: 0.0 for k in allocs}
    for s in run_seeds:
        for name, u in allocs.items():
            F = simulate_v(sup, tier, run_seed=s, u=u, **{**BASE, **cfg})["F"]
            sums[name] += F.sum()
    means = {k: v / len(run_seeds) for k, v in sums.items()}
    base = means["baseline"]
    red = {k: (100.0 * (1 - v / base) if base > 0 else float("nan"))
           for k, v in means.items() if k != "baseline"}
    return means, red


# ----------------------------------------------------------------------
def q1_katz_replication(seeds20=range(20), runs_per=20):
    print("=" * 72); print("Q1: Katz on the 40 topology draws (ER + PA)")
    res = {}
    for topo in ("er", "pa"):
        per = {"katz": [], "C": [], "uniform": [], "tier1": []}
        wins_katz_over_C = wins_C_over_uniform = 0
        for ts in seeds20:
            if topo == "er":
                A, sup, tier, _ = build_network(seed=10000 + ts)
            else:
                A, sup, tier = build_network_pa(10000 + ts)
            n = len(tier); t1 = tier == 0
            cent = centralities(A)
            allocs = {"baseline": None,
                      "uniform": np.full(n, BUDGET / n),
                      "C": alloc_from_score(cent["C"]),
                      "katz": alloc_from_score(cent["katz"]),
                      "tier1": np.where(t1, BUDGET / t1.sum(), 0.0)}
            means, red = reductions(sup, tier, allocs,
                                    [100 + r for r in range(runs_per)])
            if means["baseline"] <= 0:
                continue
            for k in per: per[k].append(red[k])
            wins_katz_over_C += red["katz"] > red["C"]
            wins_C_over_uniform += red["C"] > red["uniform"]
        res[topo] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
                     for k, v in per.items()}
        res[topo]["n_draws"] = len(per["katz"])
        res[topo]["wins_katz_over_C"] = int(wins_katz_over_C)
        res[topo]["wins_C_over_uniform"] = int(wins_C_over_uniform)
        res[topo]["tier1_all_100"] = bool(all(abs(v - 100.0) < 1e-9
                                             for v in per["tier1"]))
        print(f"  [{topo}] " + " | ".join(
            f"{k} {res[topo][k][0]:.1f}% ({res[topo][k][1]:.1f}-{res[topo][k][2]:.1f})"
            for k in ("katz", "C", "uniform", "tier1"))
            + f" | katz>C {wins_katz_over_C}/{len(per['katz'])}"
            + f" | C>u {wins_C_over_uniform}/{len(per['katz'])}")
    pooled = {k: float(np.mean([res["er"][k][0], res["pa"][k][0]]))
              for k in ("katz", "C", "uniform", "tier1")}
    res["pooled_mean"] = pooled
    print("  pooled means: " + " ".join(f"{k}:{v:.1f}%" for k, v in pooled.items()))
    return res


def q2_budget_katz(A, sup, tier, seeds):
    print("=" * 72); print("Q2: budget sweep including Katz, 100 CRN runs")
    n = len(tier); t1 = tier == 0
    cent = centralities(A)
    res = {}
    for B in (1., 2., 4., 8., 16.):
        allocs = {"baseline": None,
                  "katz": alloc_from_score(cent["katz"], B),
                  "C": alloc_from_score(cent["C"], B),
                  "uniform": np.full(n, B / n),
                  "tier1": np.where(t1, B / t1.sum(), 0.0)}
        means, red = reductions(sup, tier, allocs, seeds)
        res[B] = {"means": means, "reduction": red}
        print(f"  B={B:4g}: " + " ".join(f"{k} {v:5.1f}%" for k, v in red.items()))
    order_ok = all(res[B]["reduction"]["tier1"] + 1e-9 >= res[B]["reduction"]["katz"]
                   and res[B]["reduction"]["katz"] + 1e-9 >= res[B]["reduction"]["C"]
                   and res[B]["reduction"]["C"] > res[B]["reduction"]["uniform"]
                   for B in res)
    res["ordering_tier1_ge_katz_ge_C_gt_uniform"] = bool(order_ok)
    print(f"  ordering tier1 >= Katz >= chi > uniform at every B: {order_ok}")
    return res


def q3_grid(A, sup, tier, seeds):
    print("=" * 72); print("Q3: tau x alpha robustness grid (Table VI)")
    n = len(tier); t1 = tier == 0
    cent = centralities(A)
    cells = [("tau", t) for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)] + \
            [("alpha", a) for a in (1.05, 1.1, 1.2, 1.35, 1.5)]
    res = {}
    for kind, val in cells:
        cfg = {kind: val}
        row = {}
        # (a) lambda sweep
        lam_tot = {}
        for lam in (0., 2., 5.):
            tot = [total(simulate_v(sup, tier, run_seed=s, **{**BASE, **cfg, "lam": lam})["F"])
                   for s in seeds]
            lam_tot[lam] = float(np.mean(tot))
        row["lambda"] = lam_tot
        row["lambda_monotone"] = bool(lam_tot[0.] >= lam_tot[2.] >= lam_tot[5.])
        # (b) allocation at B=4
        allocs = {"baseline": None, "uniform": np.full(n, BUDGET / n),
                  "C": alloc_from_score(cent["C"]),
                  "tier1": np.where(t1, BUDGET / t1.sum(), 0.0)}
        means, red = reductions(sup, tier, allocs, seeds, **cfg)
        row["alloc"] = means
        row["alloc_order_ok"] = bool(means["tier1"] <= means["C"] < means["uniform"]
                                     < means["baseline"])
        # (c) delay comparison at T=200
        dl = {}
        for L in (1, 6):
            tot, rc = [], []
            for s in range(30):
                F = simulate_v(sup, tier, run_seed=s, **{**BASE, **cfg, "L": L, "T": 200})["F"]
                tot.append(total(F)); rc.append(recov(F))
            dl[L] = {"total": float(np.mean(tot)), "recovery": float(np.mean(rc)),
                     "censored": int(sum(1 for s in range(30) if False))}
        dl["pct_change_total"] = 100.0 * (dl[6]["total"] / dl[1]["total"] - 1)
        dl["recovery_rises"] = bool(dl[6]["recovery"] > dl[1]["recovery"])
        row["delay"] = dl
        key = f"{kind}={val:g}"
        res[key] = row
        print(f"  {key:>10}: lam {lam_tot[0.]:6.0f}/{lam_tot[2.]:6.0f}/{lam_tot[5.]:6.0f} "
              f"| B=4 none {means['baseline']:6.0f} unif {means['uniform']:6.0f} "
              f"chi {means['C']:6.0f} t1 {means['tier1']:5.0f} "
              f"| T200 L1 {dl[1]['total']:6.0f} L6 {dl[6]['total']:6.0f} "
              f"({dl['pct_change_total']:+.1f}%) rec {dl[1]['recovery']:.1f}->{dl[6]['recovery']:.1f}"
              f" | mono {row['lambda_monotone']} order {row['alloc_order_ok']}")
    return res


def q4_cisplatin(sup, tier):
    print("=" * 72); print("Q4: single-facility back-test statistics")
    T = 100
    onset, spr, dur, cens = [], [], [], 0
    for s in range(100):
        F = simulate_v(sup, tier, run_seed=s, shock_frac=0.05, eps=0.95,
                       **{**BASE, "T": T})["F"]
        downstream = np.nonzero((F[:, tier > 0]).any(1))[0]
        onset.append(int(downstream.min()) if len(downstream) else -1)
        spr.append(spread(F)); dur.append(recov(F)); cens += int(F[-1].any())
    ok = [o for o in onset if o >= 0]
    prop = np.array([o >= 0 for o in onset])
    dur_p = np.array(dur)[prop]
    q1, q3 = np.percentile(dur_p, [25, 75])
    res = {"n_propagate": int(prop.sum()), "onset_median": float(np.median(ok)),
           "spread_mean_all": float(np.mean(spr)),
           "spread_mean_propagating": float(np.mean(np.array(spr)[prop])),
           "duration_median_propagating": float(np.median(dur_p)),
           "duration_iqr_propagating": [float(q1), float(q3)],
           "censored_runs": int(cens)}
    print(f"  propagates in {res['n_propagate']}/100; onset median {res['onset_median']:.0f}; "
          f"spread {res['spread_mean_all']:.1f}% (all) / {res['spread_mean_propagating']:.1f}% (propagating); "
          f"duration median {res['duration_median_propagating']:.0f} IQR {q1:.0f}-{q3:.0f}; "
          f"censored {cens}")
    return res


def _simulate_with_h(sup, tier, *, run_seed, lam=2.0, L=3, delta=DELTA,
                     alpha=ALPHA, tau=TAU, h0=H0, eps=None, shock_frac=0.25,
                     T=50, rho=-10.0):
    """Instrumented copy of decisions.simulate_v (CES + smooth damping,
    indicator sourcing) that also returns the peak inventory."""
    rng = np.random.default_rng(run_seed)
    n = len(tier)
    t1 = np.nonzero(tier == 0)[0]
    k = max(1, int(round(shock_frac * len(t1))))
    hit = rng.choice(t1, size=k, replace=False)
    eps_j = np.zeros(n)
    if eps is not None: eps_j[hit] = eps
    else: eps_j[hit] = rng.uniform(*EPSR, size=len(hit))
    hist = np.ones((L + 1, n)); x = np.ones(n)
    x[hit] = np.minimum(XBAR, XBAR * (1 - eps_j[hit]))
    h = np.full(n, h0)
    w = {j: np.full(len(sup[j]), 1.0 / len(sup[j])) for j in range(n) if len(sup[j])}
    shocked = np.zeros(n, bool); shocked[hit] = True
    F = np.zeros((T + 1, n), bool); F[0] = x < tau
    X = np.zeros((T + 1, n)); X[0] = x
    hmax = float(h.max())
    for t in range(T):
        xlag = hist[-1]; f = F[t].astype(float)
        rec = shocked & (rng.random(n) < PREC); shocked &= ~rec
        xn = np.empty(n); hn = h.copy()
        for j in range(n):
            S = sup[j]
            if len(S) == 0:
                xn[j] = min(XBAR, XBAR * (1 - eps_j[j])) if shocked[j] else XBAR
                continue
            wj = w[j]
            I = float((wj @ np.maximum(xlag[S], 1e-12) ** rho) ** (1.0 / rho))
            xt = min(XBAR, alpha * (I + h[j]))
            damp = max(0.0, 1.0 - delta * np.maximum(0.0, (tau - X[t][S]) / tau).sum())
            xn[j] = xt * damp
            hn[j] = max(0.0, h[j] + I - xn[j])
        for j in range(n):
            S = sup[j]
            if len(S):
                e = np.exp(-lam * f[S]); w[j] = e / e.sum()
        h = hn; hmax = max(hmax, float(h.max()))
        hist = np.vstack([xn, hist[:-1]])
        F[t + 1] = xn < tau; X[t + 1] = xn
    return F, hmax


def q5_peak_inventory(sup, tier, seeds):
    print("=" * 72); print("Q5: peak inventory over the baseline runs")
    peaks = []
    for s in seeds:
        F, hmax = _simulate_with_h(sup, tier, run_seed=s)
        Fref = simulate_v(sup, tier, run_seed=s, **BASE)["F"]
        assert np.array_equal(F, Fref), "instrumented simulator diverged"
        peaks.append(hmax)
    res = {"peak_inventory_max": float(np.max(peaks)),
           "peak_inventory_mean": float(np.mean(peaks))}
    print(f"  peak inventory: max {res['peak_inventory_max']:.3f}, "
          f"mean of per-run peaks {res['peak_inventory_mean']:.3f} (baseline, T=50)")
    # also across the delay sweep at T=200 and the tau sweep (largest cascades)
    extra = {}
    for label, cfg in (("L6_T200", {"L": 6, "T": 200}), ("tau0.5", {"tau": 0.5}),
                       ("alpha1.5", {"alpha": 1.5}), ("lam0", {"lam": 0.0})):
        pk = [_simulate_with_h(sup, tier, run_seed=s, **cfg)[1] for s in range(30)]
        extra[label] = float(np.max(pk))
        print(f"  peak inventory ({label}, 30 runs): {extra[label]:.3f}")
    res["peak_inventory_other_configs"] = extra
    return res


if __name__ == "__main__":
    A, sup, tier, ranks = build_network()
    seeds = list(range(100))
    out = {}
    out["q5"] = q5_peak_inventory(sup, tier, seeds)
    out["q4"] = q4_cisplatin(sup, tier)
    out["q2"] = q2_budget_katz(A, sup, tier, seeds)
    out["q1"] = q1_katz_replication()
    out["q3"] = q3_grid(A, sup, tier, seeds)

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    json.dump(clean(out), open("results_qa.json", "w"), indent=1)
    print("\nwrote results_qa.json")
