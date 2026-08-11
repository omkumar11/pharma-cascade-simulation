"""
new_experiments.py
==================
New experiment suite for the revision of

  "Geometry, Thresholds, and Control of Cascading Failures in Tiered
   Pharmaceutical Supply Networks" (Kumar & Salwekar)

Three experiments, addressing the referee's empirical points:

  E1  lambda_paired      Paired statistics for the sourcing-sensitivity
                         sweep (lambda in {0,1,2,3,5}), 100 runs, common
                         random numbers across lambda values. Produces
                         every number needed for Table `tab:lambda_paired`
                         (mean paired difference, bootstrap 95% CI,
                         Wilcoxon signed-rank p, paired t-test p, fraction
                         of runs with positive difference).

  E2  topology_repl      Replicates baseline / uniform / targeted
                         intervention across 20 independent ER topology
                         draws AND a heavy-tailed (preferential-attachment)
                         inter-tier topology family. Produces the numbers
                         for the topology paragraph in Sec. 5.9.

  E3  ces_check          Diagnoses the identical rho >= 0 rows in Table
                         `tab:ces_leontief`: instruments threshold
                         crossings by tier, sweeps intermediate negative
                         rho, re-runs rho >= 0 under harsher shocks, and
                         asserts CES(rho -> -inf) == Leontief(min form)
                         on matched inputs.

IMPORTANT — INTEGRITY / RECONCILIATION NOTE FOR OM:
  This file is a fresh, self-contained implementation written from the
  model specification in Sections 2 and 5 of the manuscript. It is NOT
  the archived repository code. Before any number from this file goes
  into the paper, you must EITHER (a) port these experiment functions
  into the archived repo (pharma-cascade-simulation) so that published
  numbers come from the same codebase as the rest of Section 6, OR
  (b) verify that this implementation reproduces the paper's baseline
  numbers (Table `tab:results` row 1: total failures ~285+-149, spread
  ~54.6%, at seed 42) and document any discrepancy. Run
  `python new_experiments.py verify` first — it prints the baseline
  metrics for direct comparison against Table `tab:results`.

  TODO(Om): eta and omega_k for the centrality C(j) are set to 0.5 and
  1.0 below. Confirm these match the archived repo's values; the paper
  never states them (add them to Sec. 5 while you're at it).

Usage:
  python new_experiments.py verify     # baseline sanity check vs paper
  python new_experiments.py e1         # lambda paired stats
  python new_experiments.py e2         # topology replication
  python new_experiments.py e3         # CES diagnostics
  python new_experiments.py all

Outputs: CSVs + printed summaries in ./results/
Requires: numpy, scipy (pandas optional, not used).
"""

import os
import sys
import csv
import numpy as np
from scipy import stats

# ----------------------------------------------------------------------
# Model parameters (paper Section 5, Table of parameters)
# ----------------------------------------------------------------------
TIER_SIZES = [20, 35, 50, 41]          # API, synthesis, formulation, distribution
N = sum(TIER_SIZES)                    # 146
P_ER = 0.15                            # inter-tier ER probability
XBAR = 1.0
TAU = 0.3
ALPHA = 1.2
H0 = 0.1
HCAP = 5.0                             # storage cap \bar h (revision); track binding
DELTA = 0.2
LAMBDA_BASE = 2.0
L_BASE = 2
SHOCK_SHARE = 0.25                     # 25% of tier-1 disrupted at t=0
EPS_LO, EPS_HI = 0.6, 0.9
P_REC = 0.06
T_HORIZON = 50
BUDGET = 4.0
ETA = 0.5                              # TODO(Om): confirm vs archived repo
OMEGA = 1.0                            # TODO(Om): confirm vs archived repo
BASE_TOPOLOGY_SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------------------------------------------------
# Network construction
# ----------------------------------------------------------------------
def build_network(seed=BASE_TOPOLOGY_SEED, topology="er"):
    """
    Returns:
      suppliers: list of np.array of supplier indices for each node
                 (empty for tier-1)
      tier:      np.array of tier index (1..4) per node
    topology: "er"  — Erdos-Renyi inter-tier edges at p=0.15
              "pa"  — preferential attachment: each downstream node draws
                      the same expected number of suppliers, but chooses
                      them with probability proportional to (current
                      out-degree + 1), producing heavy-tailed supplier
                      out-degree within each upstream tier.
    Every non-source node is guaranteed >= 1 supplier.
    """
    rng = np.random.default_rng(seed)
    tier = np.concatenate([np.full(s, k + 1) for k, s in enumerate(TIER_SIZES)])
    idx_by_tier = [np.where(tier == k + 1)[0] for k in range(len(TIER_SIZES))]
    suppliers = [np.array([], dtype=int) for _ in range(N)]

    for k in range(1, len(TIER_SIZES)):          # tiers 2..4
        up, down = idx_by_tier[k - 1], idx_by_tier[k]
        if topology == "er":
            for j in down:
                mask = rng.random(len(up)) < P_ER
                sel = up[mask]
                if sel.size == 0:
                    sel = np.array([rng.choice(up)])
                suppliers[j] = sel
        elif topology == "pa":
            outdeg = np.zeros(len(up))
            mean_m = max(1, int(round(P_ER * len(up))))  # match expected in-degree
            for j in down:
                m_j = max(1, rng.binomial(len(up), P_ER))
                m_j = min(m_j, len(up))
                w = (outdeg + 1.0)
                w = w / w.sum()
                sel = rng.choice(up, size=m_j, replace=False, p=w)
                suppliers[j] = np.sort(sel)
                for s in sel:
                    outdeg[np.where(up == s)[0][0]] += 1
            _ = mean_m
        else:
            raise ValueError(topology)
    return suppliers, tier


def downstream_centrality(suppliers, tier, eta=ETA, omega=OMEGA):
    """C(j) = sum_k eta^{d(j,k)} * omega over nodes k reachable from j (BFS)."""
    children = [[] for _ in range(N)]
    for j in range(N):
        for i in suppliers[j]:
            children[i].append(j)
    C = np.zeros(N)
    for j in range(N):
        dist = {j: 0}
        frontier = [j]
        while frontier:
            nxt = []
            for a in frontier:
                for b in children[a]:
                    if b not in dist:
                        dist[b] = dist[a] + 1
                        nxt.append(b)
            frontier = nxt
        C[j] = sum(eta ** d * omega for k, d in dist.items() if k != j)
    return C


# ----------------------------------------------------------------------
# Aggregators
# ----------------------------------------------------------------------
def agg_leontief_req(x_lag, w, m):
    """Requirements-form Leontief: min( min_i x_i/(w_i m), max_i x_i )."""
    return min(np.min(x_lag / (w * m)), np.max(x_lag))


def agg_ces(x_lag, w, rho):
    """CES with weights w (sum 1). rho=0 handled as Cobb-Douglas limit.
    For rho<0, any zero input forces I=0 (Leontief-side behavior)."""
    if rho == 0.0:
        if np.any(x_lag <= 0):
            return 0.0
        return float(np.exp(np.sum(w * np.log(x_lag))))
    if rho < 0 and np.any(x_lag <= 1e-12):
        return 0.0
    return float(np.sum(w * np.maximum(x_lag, 1e-12) ** rho) ** (1.0 / rho))


# ----------------------------------------------------------------------
# Shock process (drawn separately so common random numbers work)
# ----------------------------------------------------------------------
def draw_shock_process(run_seed, tier):
    """All run-level randomness, independent of model configuration, so the
    same draws can be replayed across lambda values / regimes / aggregators
    (common random numbers)."""
    rng = np.random.default_rng(run_seed)
    tier1 = np.where(tier == 1)[0]
    n_dis = max(1, int(round(SHOCK_SHARE * len(tier1))))
    disrupted = rng.choice(tier1, size=n_dis, replace=False)
    eps = {int(j): float(rng.uniform(EPS_LO, EPS_HI)) for j in disrupted}
    # recovery: pre-draw uniforms for T periods per disrupted node
    rec_u = {int(j): rng.random(T_HORIZON + 1) for j in disrupted}
    return disrupted, eps, rec_u


# ----------------------------------------------------------------------
# Core simulation
# ----------------------------------------------------------------------
def simulate(suppliers, tier, shock, *, lam=LAMBDA_BASE, L=L_BASE,
             delta=DELTA, aggregator="leontief", rho=None, u=None,
             eps_override=None, T=T_HORIZON):
    """
    One trajectory. Returns dict of metrics + diagnostics.
      shock: (disrupted, eps, rec_u) from draw_shock_process
      u: length-N array of pre-positioned reserves (or None)
      eps_override: scalar epsilon for controlled threshold experiments
    Implements Section 2 dynamics exactly:
      I_j from x(t-L) and w(t); xtilde = min(xbar, alpha*(I+h));
      x(t+1) = xtilde * max(0, 1 - delta * sum_i f_i(t));
      h(t+1) = min(hcap, max(0, h + I - x(t+1)));
      w(t+1) = softmax(-lam * f_i(t));
      f(t+1) = 1{x(t+1) < tau}; tier-1 overridden by disruption state.
    """
    disrupted, eps, rec_u = shock
    if eps_override is not None:
        eps = {j: eps_override for j in eps}
    if u is None:
        u = np.zeros(N)

    m = np.array([max(len(s), 1) for s in suppliers])
    is_source = np.array([len(s) == 0 for s in suppliers])

    # state
    x_hist = [np.full(N, XBAR) for _ in range(L + 1)]  # x(t-L)..x(t); healthy history
    h = np.full(N, H0)
    h[~is_source] += u[~is_source]                     # reserves -> initial inventory
    h = np.minimum(h, HCAP)
    w = [np.full(len(s), 1.0 / len(s)) if len(s) else np.array([]) for s in suppliers]

    active = {int(j): True for j in disrupted}          # disruption state at t

    def source_output(j, t_active):
        if t_active.get(j, False):
            return min(XBAR, XBAR * (1.0 - eps[j]) + u[j])
        return XBAR

    # t = 0 state
    x = np.full(N, XBAR)
    for j in disrupted:
        x[j] = source_output(int(j), active)
    x_hist[-1] = x.copy()
    f = (x < TAU).astype(float)

    ever_failed = f.astype(bool).copy()
    total_failures = f.sum()
    peak = f.sum()
    last_fail_t = 0 if f.any() else -1
    max_inventory = h.max()
    crossings_by_tier = {k: set() for k in range(1, 5)}
    for j in np.where(f > 0)[0]:
        crossings_by_tier[int(tier[j])].add(int(j))

    for t in range(T_HORIZON):
        x_lagL = x_hist[0]      # x(t-L)
        x_new = np.empty(N)

        # recovery draws for period t -> t+1
        for j in list(active.keys()):
            if active[j] and rec_u[j][t] < P_REC:
                active[j] = False

        for j in range(N):
            if is_source[j]:
                x_new[j] = source_output(j, active)
                continue
            sl = suppliers[j]
            xs = x_lagL[sl]
            if aggregator == "leontief":
                I = agg_leontief_req(xs, w[j], m[j])
            else:
                I = agg_ces(xs, w[j], rho)
            xt = min(XBAR, ALPHA * (I + h[j]))
            damp = max(0.0, 1.0 - delta * f[sl].sum())
            x_new[j] = xt * damp
            h[j] = min(HCAP, max(0.0, h[j] + I - x_new[j]))

        # sourcing update from f(t)
        for j in range(N):
            sl = suppliers[j]
            if len(sl):
                e = np.exp(-lam * f[sl])
                w[j] = e / e.sum()

        f_new = (x_new < TAU).astype(float)
        x_hist = x_hist[1:] + [x_new.copy()]
        x = x_new
        f = f_new

        total_failures += f.sum()
        peak = max(peak, f.sum())
        ever_failed |= f.astype(bool)
        if f.any():
            last_fail_t = t + 1
        max_inventory = max(max_inventory, h.max())
        for j in np.where(f > 0)[0]:
            crossings_by_tier[int(tier[j])].add(int(j))

    return {
        "total_failures": float(total_failures),
        "peak": float(peak),
        "spread_pct": 100.0 * ever_failed.sum() / N,
        "recovery_time": float(last_fail_t if last_fail_t >= 0 else 0),
        "max_inventory": float(max_inventory),
        "inventory_cap_bound": bool(max_inventory >= HCAP - 1e-9),
        "crossings_by_tier": {k: len(v) for k, v in crossings_by_tier.items()},
    }


# ----------------------------------------------------------------------
# Interventions
# ----------------------------------------------------------------------
def allocation(suppliers, tier, regime):
    if regime == "baseline":
        return np.zeros(N)
    if regime == "uniform":
        return np.full(N, BUDGET / N)
    if regime == "targeted":
        C = downstream_centrality(suppliers, tier)
        theta = C  # kappa(j) node-invariant under homogeneous kernel (paper Def 3.11)
        if theta.sum() <= 0:
            return np.full(N, BUDGET / N)
        return BUDGET * theta / theta.sum()
    raise ValueError(regime)


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def paired_stats(a, b, n_boot=10000, seed=0):
    """Statistics for paired differences d = a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    try:
        w_p = stats.wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        w_p = float("nan")
    t_p = stats.ttest_rel(a, b).pvalue
    return {
        "mean_diff": d.mean(),
        "sd_diff": d.std(ddof=1),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": w_p,
        "paired_t_p": t_p,
        "frac_positive": float(np.mean(d > 0)),
        "n": len(d),
    }


def write_csv(path, rows, header):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  -> wrote {path}")


# ----------------------------------------------------------------------
# VERIFY: baseline sanity check against Table tab:results
# ----------------------------------------------------------------------
def run_verify(n_runs=30):
    print("=" * 70)
    print("VERIFY: baseline vs Table tab:results (expect ~285+-149 failures,")
    print("        spread ~54.6%, peak ~41, recovery ~34, at ER seed 42)")
    print("=" * 70)
    suppliers, tier = build_network(BASE_TOPOLOGY_SEED, "er")
    out = {r: [] for r in ("baseline", "uniform", "targeted")}
    cap_hit = False
    for r in range(n_runs):
        shock = draw_shock_process(1000 + r, tier)
        for regime in out:
            u = allocation(suppliers, tier, regime)
            res = simulate(suppliers, tier, shock, u=u)
            out[regime].append(res)
            cap_hit |= res["inventory_cap_bound"]
    for regime, results in out.items():
        tf = np.array([r["total_failures"] for r in results])
        sp = np.array([r["spread_pct"] for r in results])
        pk = np.array([r["peak"] for r in results])
        rt = np.array([r["recovery_time"] for r in results])
        print(f"{regime:>9}: failures {tf.mean():7.1f}+-{tf.std(ddof=1):6.1f} | "
              f"peak {pk.mean():5.1f} | spread {sp.mean():5.1f}% | "
              f"recovery {rt.mean():5.1f}")
    print(f"inventory cap (hbar={HCAP}) ever binding: {cap_hit}")
    print("If these do NOT match the paper's Table 1 within sampling error,")
    print("reconcile this implementation against the archived repo before")
    print("using ANY number from this file in the manuscript.")


# ----------------------------------------------------------------------
# E1: lambda paired statistics
# ----------------------------------------------------------------------
def run_e1(n_runs=100):
    print("=" * 70)
    print("E1: paired lambda sweep, common random numbers, "
          f"{n_runs} runs, lambda in {{0,1,2,3,5}}")
    print("=" * 70)
    suppliers, tier = build_network(BASE_TOPOLOGY_SEED, "er")
    lambdas = [0.0, 1.0, 2.0, 3.0, 5.0]
    totals = {lam: [] for lam in lambdas}
    cap_hit = False
    for r in range(n_runs):
        shock = draw_shock_process(2000 + r, tier)   # SAME shock across lambdas
        for lam in lambdas:
            res = simulate(suppliers, tier, shock, lam=lam)
            totals[lam].append(res["total_failures"])
            cap_hit |= res["inventory_cap_bound"]

    print("\nMarginal means (total node-period failures):")
    for lam in lambdas:
        arr = np.array(totals[lam])
        print(f"  lambda={lam:>3}: {arr.mean():8.1f} +- {arr.std(ddof=1):7.1f}")

    comparisons = [("l3_vs_l2", 3.0, 2.0), ("l5_vs_l2", 5.0, 2.0),
                   ("l2_vs_l1", 2.0, 1.0)]
    rows = []
    print("\nPaired comparisons (positive diff = first arm worse):")
    print(f"{'cmp':>9} {'mean_diff':>10} {'95% CI':>22} {'wilcoxon_p':>11} "
          f"{'paired_t_p':>11} {'frac>0':>7}")
    for name, la, lb in comparisons:
        s = paired_stats(totals[la], totals[lb], seed=7)
        print(f"{name:>9} {s['mean_diff']:10.1f} "
              f"[{s['ci_lo']:8.1f},{s['ci_hi']:8.1f}] "
              f"{s['wilcoxon_p']:11.4g} {s['paired_t_p']:11.4g} "
              f"{s['frac_positive']:7.2f}")
        rows.append([name, s["n"], f"{s['mean_diff']:.2f}", f"{s['sd_diff']:.2f}",
                     f"{s['ci_lo']:.2f}", f"{s['ci_hi']:.2f}",
                     f"{s['wilcoxon_p']:.4g}", f"{s['paired_t_p']:.4g}",
                     f"{s['frac_positive']:.3f}"])
    write_csv(os.path.join(RESULTS_DIR, "e1_lambda_paired.csv"), rows,
              ["comparison", "n", "mean_diff", "sd_diff", "ci_lo", "ci_hi",
               "wilcoxon_p", "paired_t_p", "frac_positive"])
    write_csv(os.path.join(RESULTS_DIR, "e1_lambda_raw.csv"),
              [[r] + [totals[lam][r] for lam in lambdas] for r in range(n_runs)],
              ["run"] + [f"lambda_{lam}" for lam in lambdas])
    print(f"\ninventory cap ever binding: {cap_hit}")
    print("\nPaper actions:")
    print(" * Fill Table tab:lambda_paired from e1_lambda_paired.csv.")
    print(" * If l3_vs_l2 / l5_vs_l2 are significant (p<0.05, CI excludes 0):")
    print("   keep the 'interior optimum / reversal' language.")
    print(" * If not: switch abstract, Sec 6.4, Sec 7 to the weaker")
    print("   'marginal value exhausted at moderate sensitivity' framing,")
    print("   per the TOFILL instructions in the revised .tex.")


# ----------------------------------------------------------------------
# E2: topology replication
# ----------------------------------------------------------------------
def run_e2(n_topologies=20, runs_per=20):
    print("=" * 70)
    print(f"E2: topology replication — {n_topologies} draws x "
          f"{runs_per} runs, ER and preferential-attachment families")
    print("=" * 70)
    rows = []
    for topo in ("er", "pa"):
        red_targeted, red_uniform, wins = [], [], 0
        for ts in range(n_topologies):
            suppliers, tier = build_network(10_000 + ts, topo)
            sums = {r: 0.0 for r in ("baseline", "uniform", "targeted")}
            for r in range(runs_per):
                shock = draw_shock_process(3000 + 97 * ts + r, tier)
                for regime in sums:
                    u = allocation(suppliers, tier, regime)
                    res = simulate(suppliers, tier, shock, u=u)
                    sums[regime] += res["total_failures"]
            base = sums["baseline"] / runs_per
            if base <= 0:
                continue
            rt = 100 * (1 - (sums["targeted"] / runs_per) / base)
            ru = 100 * (1 - (sums["uniform"] / runs_per) / base)
            red_targeted.append(rt)
            red_uniform.append(ru)
            wins += rt > ru
            rows.append([topo, ts, f"{base:.1f}", f"{rt:.1f}", f"{ru:.1f}"])
        rt, ru = np.array(red_targeted), np.array(red_uniform)
        print(f"\n[{topo.upper()}] targeted reduction: mean {rt.mean():.1f}% "
              f"(min {rt.min():.1f}, max {rt.max():.1f})")
        print(f"[{topo.upper()}] uniform  reduction: mean {ru.mean():.1f}% "
              f"(min {ru.min():.1f}, max {ru.max():.1f})")
        print(f"[{topo.upper()}] targeted beats uniform in "
              f"{wins}/{len(red_targeted)} topology draws")
    write_csv(os.path.join(RESULTS_DIR, "e2_topology.csv"), rows,
              ["topology", "topo_seed", "baseline_failures",
               "targeted_reduction_pct", "uniform_reduction_pct"])
    print("\nPaper action: fill the TOFILL block in Sec 5.9 with these")
    print("distributional numbers (mean/min/max reductions, win fraction,")
    print("both families).")


# ----------------------------------------------------------------------
# E3: CES diagnostics
# ----------------------------------------------------------------------
def run_e3(n_runs=30):
    print("=" * 70)
    print("E3: CES diagnostics — is the identical rho>=0 row a bug?")
    print("=" * 70)
    suppliers, tier = build_network(BASE_TOPOLOGY_SEED, "er")

    # (0) unit consistency check: CES rho->-inf vs Leontief min-form at
    # uniform weights on random inputs
    rng = np.random.default_rng(1)
    for _ in range(1000):
        k = rng.integers(2, 6)
        xs = rng.uniform(0.05, 1.0, size=k)
        wgt = np.full(k, 1.0 / k)
        ces = agg_ces(xs, wgt, -60.0)
        assert abs(ces - xs.min()) < 1e-2, (ces, xs.min())
    print("check 0 PASSED: CES(rho=-60, uniform w) ~= min(x) on 1000 draws")

    # (1) instrument rho >= 0 arms at baseline shocks
    print("\ncheck 1: threshold crossings by tier at baseline shocks")
    header = f"{'aggregator':>16} {'failures':>10} {'spread%':>8} " + \
             " ".join(f"tier{k}" for k in range(1, 5))
    print(header)
    rows = []
    arms = [("leontief", None), ("ces", -10.0), ("ces", -2.0), ("ces", -0.5),
            ("ces", 0.0), ("ces", 0.5), ("ces", 1.0)]
    for aggname, rho in arms:
        tf, sp = [], []
        crossings = np.zeros(4)
        for r in range(n_runs):
            shock = draw_shock_process(1000 + r, tier)   # same seeds as verify
            res = simulate(suppliers, tier, shock, aggregator=aggname, rho=rho)
            tf.append(res["total_failures"])
            sp.append(res["spread_pct"])
            for k in range(1, 5):
                crossings[k - 1] += res["crossings_by_tier"][k]
        label = "leontief" if rho is None else f"ces rho={rho}"
        tfm, spm = np.mean(tf), np.mean(sp)
        cr = crossings / n_runs
        print(f"{label:>16} {tfm:10.1f} {spm:8.1f} " +
              " ".join(f"{c:5.1f}" for c in cr))
        rows.append([label, f"{tfm:.1f}", f"{np.std(tf, ddof=1):.1f}",
                     f"{spm:.2f}"] + [f"{c:.2f}" for c in cr])
    print("Interpretation: if for rho>=0 the tier2-4 crossing counts are ~0,")
    print("the identical rows are structural (only seeded tier-1 nodes fail),")
    print("not a bug — state that in Sec 6.5. If tier2-4 crossings are")
    print("nonzero yet totals are byte-identical across rho, suspect a bug.")

    # (2) harsher-shock arm for rho >= 0
    print("\ncheck 2: rho>=0 under harsher shocks (share=0.75, eps=0.9)")
    global SHOCK_SHARE
    saved = SHOCK_SHARE
    SHOCK_SHARE = 0.75
    for aggname, rho in [("ces", 0.0), ("ces", 0.5), ("ces", 1.0)]:
        tf = []
        for r in range(n_runs):
            shock = draw_shock_process(5000 + r, tier)
            res = simulate(suppliers, tier, shock, aggregator=aggname, rho=rho,
                           eps_override=0.9)
            tf.append(res["total_failures"])
        print(f"  ces rho={rho}: failures {np.mean(tf):8.1f} "
              f"+- {np.std(tf, ddof=1):7.1f}")
    SHOCK_SHARE = saved
    print("If these three arms now DIFFER, the aggregators are healthy and")
    print("the baseline identity was ceiling/floor saturation; if they are")
    print("still byte-identical under harsh shocks, dig into the repo's CES")
    print("implementation (likely a shared code path ignoring rho for")
    print("rho>=0).")

    write_csv(os.path.join(RESULTS_DIR, "e3_ces_check.csv"), rows,
              ["aggregator", "mean_failures", "sd_failures", "mean_spread_pct",
               "tier1_crossings", "tier2_crossings", "tier3_crossings",
               "tier4_crossings"])
    print("\nPaper action: resolve the TOFILL block in Sec 6.5 accordingly.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    cmds = sys.argv[1:] or ["all"]
    if "all" in cmds:
        cmds = ["verify", "e1", "e2", "e3"]
    for c in cmds:
        {"verify": run_verify, "e1": run_e1, "e2": run_e2, "e3": run_e3}[c]()
        print()
