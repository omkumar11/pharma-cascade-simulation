"""
theory_checks_v4.py -- numerical verification of the results added in the v4
manuscript (MDPI Mathematics), complementing theory_checks.py:

  T9   comparison principle for the quantity-only model (delta = lambda = 0):
       pathwise order preservation in shock depth, outage duration, reserves
       and initial inventories; monotone-in-time relaxation under persistent
       disruption; ever-failed set = eventual failing set; persistent
       disruption is the worst case.  Violation counts when delta > 0 or
       lambda > 0 (the comparison principle is then false).
  T10  channel isolation: (a) excess of the spread with recovery over the
       persistent limit at eps = 0.80, and (b) deviation of the baseline
       (information-lead) totals from the matched-lag invariant, each with the
       damping and sourcing channels switched on one at a time.
  T11  lag-potential equivariance: heterogeneous edge lags L_ij on non-graded
       DAGs.  If L_ij = theta_j - theta_i for a node potential theta (theta = 0
       on sources), the matched-lag trajectory is the zero-lag trajectory with
       node j shifted by theta_j, exactly.  Uniform lags on the same non-graded
       graphs (no potential) break the identity.  Minimal three-node
       counterexample.  Generalized recovery bound.
  T12  convergence rate to the quasi-steady cascade state: fitted asymptotic
       rate versus r* = max over inventory-buffered nodes of (1 - alpha d_j).

Run: python3 theory_checks_v4.py   (~12 min)  -> results_theory_v4.json
     python3 theory_checks_v4.py --only-extra   re-runs only the addendum
"""
import json
import numpy as np
from model import build_network
from theory_checks import simulate_m, x_inf, g, ces, TAU, XBAR, ALPHA, H0, DELTA, RHO, LAM

PREC = 0.06


def simulate_g(suppliers, src_path, *, lags=0, lag_mode="matched", lam=LAM,
               delta=DELTA, alpha=ALPHA, tau=TAU, rho=RHO, h0=H0, T=200):
    """Corrected model with explicit source paths and edge-specific lags.

    src_path: dict {source index: array of length T+1} of source outputs
              x_i(0..T); the dispatch record is y_i(t) = x_i(t) for t >= 1 and
              xbar for t <= 0.
    lags:     int (uniform lag) or dict {(i, j): L_ij >= 0}.
    lag_mode: "matched"  -> inputs, damping and flags of supplier i read by j
                            all carry the lag L_ij;
              "baseline" -> inputs carry L_ij, damping reads x_i(t), weights
                            read f_i(t) (information lead).
    Returns X (T+1, n) and F (T+1, n)."""
    n = len(suppliers)
    hv = np.full(n, float(h0)) if np.isscalar(h0) else np.asarray(h0, float).copy()
    X = np.ones((T + 1, n)) * XBAR
    for i, p in src_path.items():
        X[:, i] = p
    lag = (lambda i, j: lags) if isinstance(lags, (int, np.integer)) else (lambda i, j: lags[(i, j)])
    Lij = {j: np.array([lag(i, j) for i in suppliers[j]], int) for j in range(n) if len(suppliers[j])}
    Sj = {j: np.asarray(suppliers[j], int) for j in Lij}
    w = {j: np.full(len(Sj[j]), 1.0 / len(Sj[j])) for j in Sj}

    def rec(idx, times):          # dispatch record y_i(t)
        out = np.full(len(idx), XBAR)
        m = times >= 1
        out[m] = X[times[m], idx[m]]
        return out

    for t in range(T):
        wn = {}
        for j in Sj:
            S = Sj[j]
            ylag = rec(S, t - Lij[j])
            I = ces(w[j], ylag, rho)
            dread = X[t, S] if lag_mode == "baseline" else ylag
            damp = max(0.0, 1.0 - delta * g(dread, tau).sum())
            xn = min(XBAR, alpha * (I + hv[j])) * damp
            hv[j] = max(0.0, hv[j] + I - xn)
            X[t + 1, j] = xn
            fw = (X[t, S] < tau) if lag_mode == "baseline" else (ylag < tau)
            e = np.exp(-lam * fw.astype(float)); wn[j] = e / e.sum()
        w = wn
    return X, X < tau


def source_paths(tier, rng, T, *, frac=0.25, eps_rng=(0.6, 0.9), p_rec=PREC, eps=None):
    """Random disruption: returns (hit, depths, recovery times)."""
    t1 = np.nonzero(tier == 0)[0]
    k = max(1, int(round(frac * len(t1))))
    hit = rng.choice(t1, size=k, replace=False)
    dep = np.full(k, eps) if eps is not None else rng.uniform(*eps_rng, size=k)
    rt = rng.geometric(p_rec, size=k) if p_rec > 0 else np.full(k, T + 10)
    return hit, dep, rt


def paths_from(hit, dep, rt, T, u=0.0):
    out = {}
    for i, e, r in zip(hit, dep, rt):
        p = np.ones(T + 1) * XBAR
        p[: min(int(r), T + 1)] = min(XBAR, XBAR * (1 - e) + u)
        out[int(i)] = p
    return out


def random_dag(rng, sizes=(6, 8, 8, 6), p=0.25, skip=True):
    """Tiered DAG; with skip=True edges may join any earlier tier (non-graded)."""
    tier = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    n = len(tier); sup = [[] for _ in range(n)]
    for j in range(n):
        if tier[j] == 0: continue
        cand = [i for i in range(n) if (tier[i] < tier[j] if skip else tier[i] == tier[j] - 1)]
        S = [i for i in cand if rng.random() < p]
        if not S: S = [int(rng.choice(cand))]
        sup[j] = S
    return sup, tier


def main():
    A, sup, tier, _ = build_network(42)
    n = len(tier); K = tier.max() + 1
    R = {}

    # ---- T0 regression: simulate_g reproduces simulate_m ---------------
    worst = 0.0
    for s in range(5):
        for mode in ("baseline", "matched"):
            for L in (0, 3):
                ref = simulate_m(sup, tier, run_seed=s, lag_mode=mode, L=L, T=120)
                sp = {int(i): ref["X"][:, i] for i in np.nonzero(tier == 0)[0]}
                Xg, _ = simulate_g(sup, sp, lags=L, lag_mode=mode, T=120)
                worst = max(worst, float(np.abs(Xg - ref["X"]).max()))
    R["T0"] = dict(max_abs_diff_vs_simulate_m=worst)
    print("T0", R["T0"])

    # ---- T9 comparison principle ---------------------------------------
    T = 150
    def order_test(delta, lam, mode, npairs=60, seed0=0):
        viol = 0; worst = 0.0
        for s in range(npairs):
            rng = np.random.default_rng(1000 + seed0 + s)
            hit, dep, rt = source_paths(tier, rng, T)
            L = int(rng.integers(0, 5))
            h_lo = rng.uniform(0.0, 0.3, size=n); h_hi = h_lo + rng.uniform(0.0, 0.2, size=n)
            # "worse" scenario: deeper shocks, longer outages, lower inventories, no reserve
            dep2 = np.minimum(0.99, dep + rng.uniform(0.0, 0.1, size=len(dep)))
            rt2 = rt + rng.integers(0, 15, size=len(rt))
            u = float(rng.uniform(0.0, 0.05))
            Xa, _ = simulate_g(sup, paths_from(hit, dep, rt, T, u=u), lags=L, lag_mode=mode,
                               delta=delta, lam=lam, h0=h_hi, T=T)
            Xb, _ = simulate_g(sup, paths_from(hit, dep2, rt2, T), lags=L, lag_mode=mode,
                               delta=delta, lam=lam, h0=h_lo, T=T)
            gap = float((Xb - Xa).max())      # > 0 means the worse scenario is somewhere better
            worst = max(worst, gap); viol += int(gap > 1e-12)
        return dict(pairs=npairs, pairs_violating_order=viol, max_violation=worst)
    R["T9_order"] = {
        "delta0_lam0": order_test(0.0, 0.0, "baseline"),
        "delta0.2_lam0_baseline": order_test(DELTA, 0.0, "baseline"),
        "delta0.2_lam0_matched": order_test(DELTA, 0.0, "matched"),
        "delta0_lam2_baseline": order_test(0.0, LAM, "baseline"),
        "delta0.2_lam2_baseline": order_test(DELTA, LAM, "baseline"),
    }
    print("T9 order", R["T9_order"])

    # persistent disruption, quantity-only model: monotone in time, ever-failed = eventual set,
    # recovery runs dominated by the persistent run
    mono_viol = 0; set_mismatch = 0; dom_viol = 0; cases = 0; max_err = 0.0
    for eps in (0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90):
        for s in range(20):
            rng = np.random.default_rng(2000 + s)
            hit, dep, rt = source_paths(tier, rng, 400, eps=eps)
            Xp, Fp = simulate_g(sup, paths_from(hit, dep, np.full(len(hit), 10**6), 400), lags=3,
                                delta=0.0, lam=0.0, T=400)
            mono_viol += int((np.diff(Xp[1:], axis=0) > 1e-12).any())
            ej = np.zeros(n); ej[hit] = dep
            xi, _ = x_inf(sup, tier, hit, ej, lam=0.0, delta=0.0)
            max_err = max(max_err, float(np.abs(Xp[-1] - xi).max()))
            set_mismatch += int((Fp.any(0) != (xi < TAU)).any())
            Xr, Fr = simulate_g(sup, paths_from(hit, dep, rt, 400), lags=3, delta=0.0, lam=0.0, T=400)
            dom_viol += int((Xr < Xp - 1e-12).any() or (Fr.any(0) & ~(xi < TAU)).any())
            cases += 1
    R["T9_persistent"] = dict(cases=cases, time_monotonicity_violations=mono_viol,
                              everfailed_vs_limit_set_mismatches=set_mismatch,
                              recovery_not_dominated=dom_viol, max_abs_err_vs_xinf=max_err)
    print("T9 persistent", R["T9_persistent"])

    # ---- T10 channel isolation -----------------------------------------
    def spread_pair(delta, lam, mode, eps=0.80, runs=30):
        rec_s, per_s = [], []
        for s in range(runs):
            F = simulate_m(sup, tier, run_seed=s, eps=eps, lag_mode=mode, delta=delta, lam=lam, T=50)["F"]
            rec_s.append(100.0 * F.any(0).mean())
            out = simulate_m(sup, tier, run_seed=s, eps=eps, lag_mode=mode, delta=delta, lam=lam,
                             recover_p=0.0, T=400)
            xi, _ = x_inf(sup, tier, out["hit"], out["eps_j"], lam=lam, delta=delta)
            per_s.append(100.0 * (xi < TAU).mean())
        return dict(spread_with_recovery=float(np.mean(rec_s)), persistent_limit=float(np.mean(per_s)),
                    runs_exceeding_limit=int(sum(a > b + 1e-9 for a, b in zip(rec_s, per_s))))
    R["T10_spread"] = {f"delta{d}_lam{l}_{mode}": spread_pair(d, l, mode)
                       for d, l in ((0.0, 0.0), (DELTA, 0.0), (0.0, LAM), (DELTA, LAM))
                       for mode in ("baseline", "matched")}
    print("T10 spread", R["T10_spread"])

    def lead_dev(delta, lam, runs=30):
        Ls = [0, 1, 3, 6]; out = {}
        for L in Ls:
            b = np.mean([simulate_m(sup, tier, run_seed=s, L=L, lag_mode="baseline", delta=delta, lam=lam, T=400)["F"].sum() for s in range(runs)])
            m = np.mean([simulate_m(sup, tier, run_seed=s, L=L, lag_mode="matched", delta=delta, lam=lam, T=400)["F"].sum() for s in range(runs)])
            out[L] = dict(baseline=float(b), matched=float(m), dev_pct=float(100 * (b - m) / m))
        return out
    R["T10_lead"] = {f"delta{d}_lam{l}": lead_dev(d, l) for d, l in ((DELTA, 0.0), (0.0, LAM), (DELTA, LAM))}
    print("T10 lead", R["T10_lead"])

    # ---- T11 lag-potential equivariance --------------------------------
    T = 220; exact = 0; checked = 0; tot_equal = 0; unif_break = 0; unif_checked = 0
    rec_viol = 0; rec_checked = 0; nongraded = 0
    for s in range(60):
        rng = np.random.default_rng(3000 + s)
        gsup, gtier = random_dag(rng)
        gn = len(gtier)
        # depth and gradedness
        kap = np.zeros(gn, int)
        for j in range(gn):
            if gsup[j]: kap[j] = 1 + max(kap[i] for i in gsup[j])
        graded = all(kap[j] == kap[i] + 1 for j in range(gn) for i in gsup[j])
        nongraded += int(not graded)
        theta = np.zeros(gn, int)
        for j in range(gn):
            if gsup[j]: theta[j] = max(theta[i] for i in gsup[j]) + int(rng.integers(0, 4))
        lags = {(i, j): int(theta[j] - theta[i]) for j in range(gn) for i in gsup[j]}
        par = dict(delta=float(rng.uniform(0, 0.4)), lam=float(rng.uniform(0, 4)),
                   alpha=float(rng.uniform(1.0, 1.5)), rho=float(rng.choice([-10, -5, -2, -1, 0.5, 1.0])),
                   h0=float(rng.uniform(0, 0.3)))
        hit, dep, rt = source_paths(gtier, rng, T, frac=0.5, eps_rng=(0.6, 0.97), p_rec=0.08)
        sp = paths_from(hit, dep, rt, T)
        X0, F0 = simulate_g(gsup, sp, lags=0, lag_mode="matched", T=T, **par)
        XL, FL = simulate_g(gsup, sp, lags=lags, lag_mode="matched", T=T, **par)
        ok = True
        for j in range(gn):
            th = theta[j]
            if not (np.array_equal(XL[th:, j], X0[: T + 1 - th, j]) and np.all(XL[:th, j] == XBAR)):
                ok = False
        exact += int(ok); checked += 1
        if F0[-1].sum() == 0 and FL[-1].sum() == 0:
            tot_equal += int(F0.sum() == FL.sum())
        # uniform lag on the same graph (a potential exists iff the graph is graded)
        if not graded:
            XU, FU = simulate_g(gsup, sp, lags=2, lag_mode="matched", T=T, **par)
            same = all(np.array_equal(XU[2 * kap[j]:, j], X0[: T + 1 - 2 * kap[j], j]) for j in range(gn))
            unif_break += int(not same); unif_checked += 1
        # generalized recovery bound (both conventions)
        tR = int(max(rt)) if len(rt) else 0
        sig = np.zeros(gn, int)
        for j in range(gn):
            if gsup[j]: sig[j] = max(sig[i] + lags[(i, j)] + 1 for i in gsup[j])
        for mode in ("matched", "baseline"):
            Xm = XL if mode == "matched" else simulate_g(gsup, sp, lags=lags, lag_mode="baseline", T=T, **par)[0]
            for j in range(gn):
                t0 = tR + sig[j]
                if t0 <= T:
                    rec_viol += int((Xm[t0:, j] != XBAR).any())
            rec_checked += 1
    R["T11"] = dict(instances=checked, nongraded_instances=nongraded, exact_shift=exact,
                    totals_equal_when_uncensored=tot_equal,
                    uniform_lag_nongraded_checked=unif_checked, uniform_lag_nongraded_broken=unif_break,
                    recovery_bound_checked=rec_checked, recovery_bound_violations=rec_viol)
    print("T11", R["T11"])

    # minimal counterexample: source 0 -> 1 -> 2 and 0 -> 2, uniform lag L, matched lags
    tsup = [[], [0], [0, 1]]
    p = np.ones(61); p[:12] = 0.1
    tot = {}
    for L in (0, 1, 2, 3):
        _, F = simulate_g(tsup, {0: p}, lags=L, lag_mode="matched", T=60)
        tot[L] = [int(F[:, j].sum()) for j in range(3)]
    R["T11_triangle"] = dict(source_output=0.1, outage_periods=12, failures_by_node_vs_L=tot)
    print("T11 triangle", R["T11_triangle"])

    # ---- T12 convergence rate ------------------------------------------
    rows = []; finite = 0; nB_total = 0
    for s in range(40):
        rng = np.random.default_rng(4000 + s)
        hit, dep, rt = source_paths(tier, rng, 400, eps_rng=(0.75, 0.97))
        mode = "baseline" if s % 2 == 0 else "matched"
        X, _ = simulate_g(sup, paths_from(hit, dep, np.full(len(hit), 10**6), 400), lags=3,
                          lag_mode=mode, T=400)
        ej = np.zeros(n); ej[hit] = dep
        xi, margin = x_inf(sup, tier, hit, ej)
        # r* over inventory-buffered nodes: alpha d < 1 and Psi < d xbar
        rstar = 0.0; gen = np.inf
        for j in np.argsort(tier, kind="stable"):
            S = sup[j]
            if len(S) == 0: continue
            xs = xi[S]; e = np.exp(-LAM * (xs < TAU)); wj = e / e.sum()
            Psi = ces(wj, xs); d = max(0.0, 1.0 - DELTA * g(xs).sum())
            gen = min(gen, abs(ALPHA * d - 1.0), abs(Psi - d * XBAR) if ALPHA * d < 1 else abs(ALPHA * Psi - XBAR))
            if ALPHA * d < 1 and Psi < d * XBAR and d > 0:
                rstar = max(rstar, 1.0 - ALPHA * d)
        err = np.abs(X - xi).max(1)
        idx = np.nonzero((err < 1e-3) & (err > 1e-11))[0]
        fit = None
        if rstar > 0 and len(idx) >= 8:
            tt = idx[len(idx) // 2:]
            fit = float(np.exp(np.polyfit(tt, np.log(err[tt]), 1)[0]))
        t_exact = int(np.nonzero(err > 1e-13)[0].max()) + 1 if (err > 1e-13).any() else 0
        if rstar == 0.0: finite += int(err[-1] < 1e-13)
        else: nB_total += 1
        rows.append(dict(seed=s, mode=mode, r_star=rstar, fitted_rate=fit, t_to_1e13=t_exact,
                         genericity_margin=float(min(gen, margin)), final_err=float(err[-1])))
    fits = [(r["fitted_rate"], r["r_star"]) for r in rows if r["fitted_rate"] is not None]
    R["T12"] = dict(rows=rows, instances=len(rows), instances_with_buffered_nodes=nB_total,
                    finite_time_instances_ok=finite,
                    max_fitted_minus_rstar=float(max(a - b for a, b in fits)) if fits else None,
                    max_abs_fitted_minus_rstar=float(max(abs(a - b) for a, b in fits)) if fits else None,
                    n_fits=len(fits))
    print("T12", {k: v for k, v in R["T12"].items() if k != "rows"})

    with open("results_theory_v4.json", "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else o)


def extra():
    """Addendum: (a) balanced lags on the triangle of the counterexample restore
    equivariance; (b) shallow persistent shocks with no inventory-buffered node
    reach the limit state in finitely many periods (Theorem 3, rate part).
    Merges keys into results_theory_v4.json."""
    A, sup, tier, _ = build_network(42)
    n = len(tier)
    R = json.load(open("results_theory_v4.json"))
    tsup = [[], [0], [0, 1]]
    p = np.ones(61); p[:12] = 0.1
    tot = {}
    for L in (0, 1, 2, 3):
        _, F = simulate_g(tsup, {0: p}, lags={(0, 1): L, (1, 2): L, (0, 2): 2 * L}, lag_mode="matched", T=60)
        tot[L] = [int(F[:, j].sum()) for j in range(3)]
    R["T11_triangle"]["balanced_failures_by_node_vs_L"] = tot
    rows = []
    for s in range(40):
        rng = np.random.default_rng(5000 + s)
        hit, dep, rt = source_paths(tier, rng, 400, eps_rng=(0.71, 0.80))
        mode = "baseline" if s % 2 == 0 else "matched"
        X, _ = simulate_g(sup, paths_from(hit, dep, np.full(len(hit), 10**6), 400), lags=3, lag_mode=mode, T=400)
        ej = np.zeros(n); ej[hit] = dep
        xi, margin = x_inf(sup, tier, hit, ej)
        buffered = 0
        for j in range(n):
            S = sup[j]
            if len(S) == 0: continue
            xs = xi[S]; e = np.exp(-LAM * (xs < TAU)); wj = e / e.sum()
            Psi = ces(wj, xs); d = max(0.0, 1.0 - DELTA * g(xs).sum())
            buffered += int(ALPHA * d < 1 and Psi < d * XBAR)
        err = np.abs(X - xi).max(1)
        t_exact = int(np.nonzero(err > 1e-14)[0].max()) + 1 if (err > 1e-14).any() else 0
        rows.append(dict(seed=s, buffered_nodes=buffered, t_exact=t_exact, final_err=float(err[-1])))
    nb = [r for r in rows if r["buffered_nodes"] == 0]
    R["T12_finite"] = dict(instances=len(rows), instances_without_buffered_nodes=len(nb),
                           reached_exactly=int(sum(r["final_err"] < 1e-14 for r in nb)),
                           max_periods_to_limit=int(max(r["t_exact"] for r in nb)) if nb else None)
    json.dump(R, open("results_theory_v4.json", "w"), indent=1)
    print("T11 triangle (balanced)", tot); print("T12 finite", R["T12_finite"])


if __name__ == "__main__":
    import sys
    if "--only-extra" not in sys.argv:
        main()
    extra()
