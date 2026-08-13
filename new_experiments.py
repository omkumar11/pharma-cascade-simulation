"""
new_experiments.py — revision experiments, ported onto the archived model.

This is the RECONCILED port (option (a) of the integrity note in the
Desktop draft new_experiments.py): all experiments below run on the same
model.simulate that generated results_e124.json, so every number that
enters the revised manuscript comes from one codebase. The lambda=2 /
Leontief / baseline arms at run seeds 0..29 reproduce the published
Table 1 rows exactly (verified: total 285.17+-148.96, spread 54.63%).

Common random numbers hold natively: model.simulate draws the shocked
set, shock magnitudes, and all recovery uniforms from run_seed alone,
and no experimental arm (lambda, rho, u, hcap) consumes randomness, so
identical run_seed => identical shock realization across arms.

Experiments:
  E0 cap_check       Inventory-cap diagnostics for Section 2 (binding
                     frequency at hbar=5; metric invariance at hbar=10).
  E1 lambda_paired   Paired lambda sweep, 100 runs, CRN. Fills
                     Table tab:lambda_paired.
  E2 topology_repl   20 ER draws + 20 preferential-attachment draws,
                     20 runs each, baseline/uniform/targeted at B=4.
                     Fills the Sec 5.9 topology paragraph.
  E3 ces_check       CES diagnostics: crossings by tier, intermediate
                     rho sweep, harsher-shock separation. Fills the
                     instrumented-replication sentence in Sec 6.5.

Usage: python3 new_experiments.py [e0|e1|e2|e3|all]
Outputs: printed summaries + CSVs in ./results/
"""
import os
import sys
import csv
import json
import numpy as np
from scipy import stats

from model import build_network, simulate, metrics, centrality, TIER_SIZES

N = sum(TIER_SIZES)
BUDGET = 4.0
HCAP_PAPER = 5.0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def write_csv(name, rows, header):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  -> wrote {path}")


# ----------------------------------------------------------------------
# Preferential-attachment inter-tier topology (heavy-tailed supplier
# out-degree), matching the ER family's expected in-degree at p=0.15.
# ----------------------------------------------------------------------
def build_network_pa(seed, p=0.15):
    rng = np.random.default_rng(seed)
    n = N
    tier = np.zeros(n, dtype=int)
    idx = 0
    bounds = []
    for k, s in enumerate(TIER_SIZES):
        tier[idx:idx + s] = k
        bounds.append((idx, idx + s))
        idx += s
    A = np.zeros((n, n), dtype=bool)
    for k in range(len(TIER_SIZES) - 1):
        u0, u1 = bounds[k]
        d0, d1 = bounds[k + 1]
        up = np.arange(u0, u1)
        outdeg = np.zeros(len(up))
        for j in range(d0, d1):
            m_j = max(1, rng.binomial(len(up), p))
            m_j = min(m_j, len(up))
            w = outdeg + 1.0
            w = w / w.sum()
            sel = rng.choice(len(up), size=m_j, replace=False, p=w)
            A[up[sel], j] = True
            outdeg[sel] += 1
    suppliers = [np.nonzero(A[:, j])[0] for j in range(n)]
    return A, suppliers, tier


# ----------------------------------------------------------------------
# Statistics helper
# ----------------------------------------------------------------------
def paired_stats(a, b, n_boot=10000, seed=7):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    w_p = stats.wilcoxon(d[d != 0]).pvalue if np.any(d != 0) else 1.0
    t_p = stats.ttest_rel(a, b).pvalue
    return dict(mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                ci_lo=ci_lo, ci_hi=ci_hi, wilcoxon_p=w_p, paired_t_p=t_p,
                frac_positive=float(np.mean(d > 0)), n=len(d))


# ----------------------------------------------------------------------
# E0: inventory-cap diagnostics (Section 2 storage-cap paragraph)
# ----------------------------------------------------------------------
def run_e0(n_runs=30):
    print("=" * 70)
    print("E0: inventory cap — binding frequency and metric invariance")
    print("=" * 70)
    A, sup, tier, _ = build_network()
    rows = []
    metrics_by_cap = {}
    maxh_uncapped = []
    for cap in (np.inf, HCAP_PAPER, 2 * HCAP_PAPER):
        ms, binds = [], 0
        for s in range(n_runs):
            hs = {}
            F = simulate(A, sup, tier, run_seed=s, hcap=cap, h_stats=hs)
            ms.append(metrics(F))
            if cap == np.inf:
                maxh_uncapped.append(hs["max_h"])
            binds += bool(hs["cap_bound"])
        agg = {k: (float(np.mean([m[k] for m in ms])),
                   float(np.std([m[k] for m in ms]))) for k in
               ("total", "peak", "spread", "recovery")}
        metrics_by_cap[cap] = agg
        print(f"hcap={cap}: total {agg['total'][0]:.2f}+-{agg['total'][1]:.2f} "
              f"spread {agg['spread'][0]:.2f} | cap binding in {binds}/{n_runs} runs")
        rows.append([cap, binds, n_runs] +
                    [f"{agg[k][0]:.4f}" for k in ("total", "peak", "spread", "recovery")])
    mh = np.array(maxh_uncapped)
    print(f"uncapped max inventory across runs: mean {mh.mean():.3f}, "
          f"max {mh.max():.3f}, runs with max_h>=5: {(mh >= 5).sum()}/{n_runs}, "
          f">=10: {(mh >= 10).sum()}/{n_runs}")
    same_5 = all(abs(metrics_by_cap[HCAP_PAPER][k][0] - metrics_by_cap[np.inf][k][0]) < 1e-9
                 for k in ("total", "peak", "spread", "recovery"))
    same_10 = all(abs(metrics_by_cap[2 * HCAP_PAPER][k][0] - metrics_by_cap[HCAP_PAPER][k][0]) < 1e-9
                  for k in ("total", "peak", "spread", "recovery"))
    print(f"metrics identical capped@5 vs uncapped: {same_5}")
    print(f"metrics identical capped@10 vs capped@5: {same_10}")
    write_csv("e0_cap_check.csv", rows,
              ["hcap", "runs_binding", "n_runs", "total", "peak", "spread", "recovery"])


# ----------------------------------------------------------------------
# E1: paired lambda sweep (Table tab:lambda_paired)
# ----------------------------------------------------------------------
def run_e1(n_runs=100):
    print("=" * 70)
    print(f"E1: paired lambda sweep, CRN, {n_runs} runs, lambda in {{0,1,2,3,5}}")
    print("=" * 70)
    A, sup, tier, _ = build_network()
    lambdas = [0.0, 1.0, 2.0, 3.0, 5.0]
    totals = {lam: [] for lam in lambdas}
    for s in range(n_runs):
        for lam in lambdas:
            F = simulate(A, sup, tier, run_seed=s, lam=lam)
            totals[lam].append(int(F.sum()))
    print("\nMarginal means (total node-period failures):")
    for lam in lambdas:
        arr = np.array(totals[lam])
        print(f"  lambda={lam:>3}: {arr.mean():8.1f} +- {arr.std(ddof=1):7.1f}")
    comparisons = [("l3_vs_l2", 3.0, 2.0), ("l5_vs_l2", 5.0, 2.0),
                   ("l2_vs_l1", 2.0, 1.0)]
    rows = []
    print("\nPaired comparisons (positive diff = first arm worse):")
    for name, la, lb in comparisons:
        st = paired_stats(totals[la], totals[lb])
        print(f"{name:>9} mean {st['mean_diff']:8.1f} "
              f"CI [{st['ci_lo']:8.1f}, {st['ci_hi']:8.1f}] "
              f"wilcoxon {st['wilcoxon_p']:.3g} t {st['paired_t_p']:.3g} "
              f"frac>0 {st['frac_positive']:.2f}")
        rows.append([name, st["n"], f"{st['mean_diff']:.2f}", f"{st['sd_diff']:.2f}",
                     f"{st['ci_lo']:.2f}", f"{st['ci_hi']:.2f}",
                     f"{st['wilcoxon_p']:.4g}", f"{st['paired_t_p']:.4g}",
                     f"{st['frac_positive']:.3f}"])
    write_csv("e1_lambda_paired.csv", rows,
              ["comparison", "n", "mean_diff", "sd_diff", "ci_lo", "ci_hi",
               "wilcoxon_p", "paired_t_p", "frac_positive"])
    write_csv("e1_lambda_raw.csv",
              [[s] + [totals[lam][s] for lam in lambdas] for s in range(n_runs)],
              ["run"] + [f"lambda_{lam:g}" for lam in lambdas])


# ----------------------------------------------------------------------
# E2: topology replication (Sec 5.9)
# ----------------------------------------------------------------------
def run_e2(n_topologies=20, runs_per=20):
    print("=" * 70)
    print(f"E2: topology replication — {n_topologies} draws x {runs_per} runs, "
          "ER + preferential attachment")
    print("=" * 70)
    rows = []
    summary = {}
    for topo in ("er", "pa"):
        red_t, red_u, wins = [], [], 0
        for ts in range(n_topologies):
            seed = 10_000 + ts
            if topo == "er":
                A, sup, tier, _ = build_network(seed=seed)
            else:
                A, sup, tier = build_network_pa(seed)
            C = centrality(A)
            theta = C / C.sum() if C.sum() > 0 else np.full(N, 1.0 / N)
            allocs = {"baseline": None,
                      "uniform": np.full(N, BUDGET / N),
                      "targeted": BUDGET * theta}
            sums = {r: 0.0 for r in allocs}
            for r in range(runs_per):
                for regime, u in allocs.items():
                    F = simulate(A, sup, tier, run_seed=100 + r, u=u)
                    sums[regime] += F.sum()
            base = sums["baseline"] / runs_per
            rt = 100 * (1 - (sums["targeted"] / runs_per) / base)
            ru = 100 * (1 - (sums["uniform"] / runs_per) / base)
            red_t.append(rt)
            red_u.append(ru)
            wins += rt > ru
            rows.append([topo, seed, f"{base:.1f}", f"{rt:.1f}", f"{ru:.1f}"])
        rt, ru = np.array(red_t), np.array(red_u)
        summary[topo] = dict(t_mean=rt.mean(), t_min=rt.min(), t_max=rt.max(),
                             u_mean=ru.mean(), u_min=ru.min(), u_max=ru.max(),
                             wins=int(wins), n=len(red_t))
        print(f"[{topo}] targeted: mean {rt.mean():.1f}% "
              f"(range {rt.min():.1f}-{rt.max():.1f}) | "
              f"uniform: mean {ru.mean():.1f}% "
              f"(range {ru.min():.1f}-{ru.max():.1f}) | "
              f"targeted wins {wins}/{len(red_t)}")
    write_csv("e2_topology.csv", rows,
              ["topology", "topo_seed", "baseline_failures",
               "targeted_reduction_pct", "uniform_reduction_pct"])
    with open(os.path.join(RESULTS_DIR, "e2_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


# ----------------------------------------------------------------------
# E3: CES diagnostics (Sec 6.5)
# ----------------------------------------------------------------------
def run_e3(n_runs=30):
    print("=" * 70)
    print("E3: CES diagnostics")
    print("=" * 70)
    A, sup, tier, _ = build_network()

    # check 0: CES(rho -> -inf, uniform w) ~= min(x)
    rng = np.random.default_rng(1)
    rho_lim = -200.0
    for _ in range(1000):
        k = rng.integers(2, 6)
        xs = rng.uniform(0.05, 1.0, size=k)
        w = np.full(k, 1.0 / k)
        ces = float((w @ np.maximum(xs, 1e-12) ** rho_lim) ** (1.0 / rho_lim))
        tol = xs.min() * (k ** (1.0 / abs(rho_lim)) - 1.0) + 1e-9
        assert xs.min() - 1e-9 <= ces <= xs.min() + tol
    print("check 0 PASSED: CES(rho=-200, uniform w) ~= min(x) on 1000 draws")

    # check 1: crossings by tier at baseline shocks (seeds 0..29 = Table E4)
    print("\ncheck 1: mean distinct threshold-crossing nodes by tier")
    arms = [("leontief", None), ("ces", -10.0), ("ces", -2.0), ("ces", -0.5),
            ("ces", 0.01), ("ces", 0.5), ("ces", 1.0)]
    rows = []
    for aggname, rho in arms:
        tf, sp = [], []
        crossings = np.zeros(4)
        for s in range(n_runs):
            F = simulate(A, sup, tier, run_seed=s, agg=aggname, rho=rho)
            tf.append(int(F.sum()))
            sp.append(100.0 * F.any(0).sum() / N)
            ever = F.any(0)
            for k in range(4):
                crossings[k] += ever[tier == k].sum()
        crossings /= n_runs
        label = "leontief" if rho is None else f"ces rho={rho:g}"
        print(f"{label:>16}: failures {np.mean(tf):8.1f}+-{np.std(tf, ddof=1):7.1f} "
              f"spread {np.mean(sp):5.1f}% | tiers " +
              " ".join(f"{c:5.2f}" for c in crossings))
        rows.append([label, f"{np.mean(tf):.1f}", f"{np.std(tf, ddof=1):.1f}",
                     f"{np.mean(sp):.2f}"] + [f"{c:.2f}" for c in crossings])
    write_csv("e3_ces_check.csv", rows,
              ["aggregator", "mean_failures", "sd_failures", "mean_spread_pct",
               "tier1_crossings", "tier2_crossings", "tier3_crossings",
               "tier4_crossings"])

    # check 2: rho >= 0 under harsher shocks (75% of tier 1, eps = 0.9)
    print("\ncheck 2: rho>=0 arms under shock_frac=0.75, eps=0.9")
    for rho in (0.01, 0.5, 1.0):
        tf = []
        for s in range(n_runs):
            F = simulate(A, sup, tier, run_seed=500 + s, agg="ces", rho=rho,
                         shock_frac=0.75, eps=0.9)
            tf.append(int(F.sum()))
        print(f"  ces rho={rho:g}: failures {np.mean(tf):8.1f} "
              f"+- {np.std(tf, ddof=1):7.1f}")


# ----------------------------------------------------------------------
# E4: network-size replication (Sec 5.9, first paragraph)
# ----------------------------------------------------------------------
def run_e4(runs_per=20):
    print("=" * 70)
    print("E4: size replication — n=300 and n=500, proportional tiers, "
          "budget scaled with n")
    print("=" * 70)
    rows = []
    for n, sizes in ((300, (41, 72, 103, 84)), (500, (68, 120, 171, 141))):
        A, sup, tier, _ = build_network(seed=42, tier_sizes=sizes)
        B = BUDGET * n / N
        C = centrality(A)
        theta = C / C.sum()
        allocs = {"baseline": None, "uniform": np.full(n, B / n),
                  "targeted": B * theta}
        tot = {k: [] for k in allocs}
        spread = []
        for s in range(runs_per):
            for regime, u in allocs.items():
                F = simulate(A, sup, tier, u=u, run_seed=s)
                tot[regime].append(F.sum())
                if regime == "baseline":
                    spread.append(100.0 * F.any(0).sum() / n)
        b = np.mean(tot["baseline"])
        rt = 100 * (1 - np.mean(tot["targeted"]) / b)
        ru = 100 * (1 - np.mean(tot["uniform"]) / b)
        print(f"n={n}: baseline spread {np.mean(spread):.1f}% | "
              f"targeted reduction {rt:.1f}% | uniform reduction {ru:.1f}%")
        rows.append([n, f"{np.mean(spread):.1f}", f"{rt:.1f}", f"{ru:.1f}"])
    write_csv("e4_size.csv", rows,
              ["n", "baseline_spread_pct", "targeted_reduction_pct",
               "uniform_reduction_pct"])


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cmds = sys.argv[1:] or ["all"]
    if "all" in cmds:
        cmds = ["e0", "e1", "e2", "e3", "e4"]
    for c in cmds:
        {"e0": run_e0, "e1": run_e1, "e2": run_e2, "e3": run_e3,
         "e4": run_e4}[c]()
        print()
