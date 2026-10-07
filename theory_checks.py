"""
theory_checks.py -- numerical verification of the analytical results of the
v3 manuscript (MDPI Mathematics):

  T1  viability invariance (seeding gate): sub-threshold shocks never fail a
      non-source node, for any recovery times
  T2  confinement margin: explicit lower bound eps_bar on the shock depth at
      which failures can leave the seed set (per seed set), checked on a fine
      eps grid with stochastic recovery
  T3  quasi-steady cascade state under persistent disruption: simulated
      outputs at large t versus the forward-substitution prediction x_inf
  T4  finite-time recovery: last failure <= t_R + (K-1)(L+1) - 1
  T5  delay equivariance: matched-lag model, totals/spread exactly invariant
      in L, recovery = max_k (R_k(0) + (k-1) L); baseline model with
      delta = lambda = 0 likewise; baseline (information-lead) deviation
  T6  rung spectrum: per-in-degree tier-2 thresholds eps_2^inf(m, s=1)
  T7  node-level coupling facts (req-form lambda*, CES jump bound)

Run: python3 theory_checks.py   (~7 min)  -> results_theory.json
     python3 theory_checks.py --only-t8   re-runs only the T8 addendum
"""
import json
import numpy as np
from model import build_network, TIER_SIZES

TAU, XBAR, ALPHA, H0, DELTA, PREC, RHO, LAM = 0.3, 1.0, 1.2, 0.1, 0.2, 0.06, -10.0, 2.0
EPSR = (0.6, 0.9)


def g(xv, tau=TAU):
    return np.maximum(0.0, (tau - xv) / tau)


def ces(w, xs, rho=RHO):
    return float((w @ np.maximum(xs, 1e-12) ** rho) ** (1.0 / rho))


def simulate_m(suppliers, tier, *, run_seed=0, lag_mode="baseline", L=3,
               lam=LAM, delta=DELTA, alpha=ALPHA, tau=TAU, rho=RHO, h0=H0,
               eps=None, recover_p=PREC, T=200, u=None, shock_frac=0.25):
    """Corrected model (CES + smooth damping + indicator softmax sourcing).
    lag_mode="baseline": byte-for-byte the protocol of decisions.simulate_v
      (damping reads x_i(t), weights w(t) = softmax(-lam f(t-1)), inputs
      x_i(t-L));
    lag_mode="matched": every inter-node read carries the same lag: damping
      and inputs read the dispatched record y_i(t-L), weights
      w(t) = softmax(-lam 1{y(t-1-L) < tau}).
    Returns F (failures), X (outputs), and t_R (first period from which all
    sources produce xbar)."""
    rng = np.random.default_rng(run_seed)
    n = len(tier)
    t1 = np.nonzero(tier == 0)[0]
    k = max(1, int(round(shock_frac * len(t1))))
    hit = rng.choice(t1, size=k, replace=False)
    uj = np.zeros(n) if u is None else np.asarray(u, float)
    eps_j = np.zeros(n)
    if eps is not None:
        eps_j[hit] = eps
    else:
        eps_j[hit] = rng.uniform(*EPSR, size=len(hit))
    hist = np.ones((L + 1, n))          # dispatched record y(t), ..., y(t-L)
    x = np.ones(n)
    x[hit] = np.minimum(XBAR, XBAR * (1 - eps_j[hit]) + uj[hit])
    h = np.full(n, h0) + uj
    w = {j: np.full(len(suppliers[j]), 1.0 / len(suppliers[j]))
         for j in range(n) if len(suppliers[j])}
    shocked = np.zeros(n, bool); shocked[hit] = True
    F = np.zeros((T + 1, n), bool); F[0] = x < tau
    X = np.zeros((T + 1, n)); X[0] = x
    tR = 0 if not shocked.any() else None
    for t in range(T):
        xlag = hist[-1]
        f = F[t].astype(float)
        rec = shocked & (rng.random(n) < recover_p); shocked &= ~rec
        xn = np.empty(n); hn = h.copy()
        dread = X[t] if lag_mode == "baseline" else xlag
        for j in range(n):
            S = suppliers[j]
            if len(S) == 0:
                xn[j] = min(XBAR, XBAR * (1 - eps_j[j]) + uj[j]) if shocked[j] else XBAR
                continue
            I = ces(w[j], xlag[S], rho)
            xt = min(XBAR, alpha * (I + h[j]))
            damp = max(0.0, 1.0 - delta * g(dread[S], tau).sum())
            xn[j] = xt * damp
            hn[j] = max(0.0, h[j] + I - xn[j])
        fw = f if lag_mode == "baseline" else (xlag < tau).astype(float)
        for j in range(n):
            S = suppliers[j]
            if len(S):
                e = np.exp(-lam * fw[S]); w[j] = e / e.sum()
        h = hn
        hist = np.vstack([xn, hist[:-1]])
        F[t + 1] = xn < tau; X[t + 1] = xn
        if tR is None and not shocked.any():
            tR = t + 1
    return dict(F=F, X=X, tR=tR, hit=hit, eps_j=eps_j)


def x_inf(suppliers, tier, hit, eps_j, *, lam=LAM, delta=DELTA, alpha=ALPHA,
          tau=TAU, rho=RHO, u=None):
    """Forward substitution for the quasi-steady cascade state (Theorem 3):
    x_j = min{ d_j xbar, max{1, alpha d_j} Psi_j }."""
    n = len(tier)
    uj = np.zeros(n) if u is None else np.asarray(u, float)
    xi = np.ones(n)
    xi[hit] = np.minimum(XBAR, XBAR * (1 - eps_j[hit]) + uj[hit])
    order = np.argsort(tier, kind="stable")
    margin = np.inf
    for j in order:
        S = suppliers[j]
        if len(S) == 0:
            continue
        xs = xi[S]
        e = np.exp(-lam * (xs < tau)); wj = e / e.sum()
        Psi = ces(wj, xs, rho)
        d = max(0.0, 1.0 - delta * g(xs, tau).sum())
        xi[j] = min(d * XBAR, max(1.0, alpha * d) * Psi)
        margin = min(margin, abs(xi[j] - tau), abs(alpha * d - 1.0))
    return xi, margin


def total(F): return int(F.sum())
def spread(F): return 100.0 * F.any(0).sum() / F.shape[1]
def recov(F):
    nz = np.nonzero(F.any(1))[0]
    return int(nz.max()) if len(nz) else 0


def tier_last(F, tier):
    out = []
    for k in range(tier.max() + 1):
        nz = np.nonzero(F[:, tier == k].any(1))[0]
        out.append(int(nz.max()) if len(nz) else None)
    return out


def bisect(fun, lo, hi, it=80):
    """largest eps in [lo, hi] with fun(eps) >= 0, fun nonincreasing."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fun(mid) >= 0: lo = mid
        else: hi = mid
    return lo


def eps_bar(suppliers, tier, hit, alpha=ALPHA, delta=DELTA, tau=TAU):
    """Theorem 2 confinement margin for a seed set at a common depth eps."""
    hs = set(hit.tolist()); best = 1.0
    for j in range(len(tier)):
        s = sum(1 for i in suppliers[j] if i in hs)
        if s == 0: continue
        fun = lambda e: min(XBAR, alpha * (1 - e)) * max(0.0, 1 - delta * s * g(1 - e, tau)) - tau
        best = min(best, bisect(fun, 1 - tau, 1.0))
    return best


def main():
    A, sup, tier, _ = build_network(42)
    K = tier.max() + 1
    R = {}

    # ---- T1 viability invariance ------------------------------------
    viol = 0; runs = 0
    for eps in (0.5, 0.6, 0.65, 0.7):
        for s in range(30):
            for mode in ("baseline", "matched"):
                F = simulate_m(sup, tier, run_seed=s, eps=eps, lag_mode=mode, T=120)["F"]
                viol += int(F[:, tier > 0].any()); runs += 1
    R["T1"] = dict(runs=runs, runs_with_downstream_failure=viol)
    print("T1", R["T1"])

    # ---- T2 confinement margin --------------------------------------
    rows = []; viol = 0; grid = np.round(np.arange(0.70, 0.80001, 0.0025), 4)
    for s in range(30):
        hit = simulate_m(sup, tier, run_seed=s, eps=0.75, T=1)["hit"]
        eb = eps_bar(sup, tier, hit)
        first = None
        for eps in grid:
            F = simulate_m(sup, tier, run_seed=s, eps=float(eps), T=120)["F"]
            if F[:, tier > 0].any():
                first = float(eps); break
        rows.append(dict(seed=s, eps_bar=eb, first_propagating_eps=first))
        if first is not None and first <= eb: viol += 1
    ebs = [r["eps_bar"] for r in rows]; fps = [r["first_propagating_eps"] for r in rows if r["first_propagating_eps"]]
    R["T2"] = dict(rows=rows, violations=viol, eps_bar_min=min(ebs), eps_bar_max=max(ebs),
                   first_prop_min=min(fps) if fps else None, first_prop_max=max(fps) if fps else None,
                   eps_bar_universal_s5=bisect(lambda e: min(1, ALPHA*(1-e))*max(0, 1-DELTA*5*g(1-e)) - TAU, 0.7, 1.0),
                   eps_bar_universal_s1=bisect(lambda e: min(1, ALPHA*(1-e))*max(0, 1-DELTA*1*g(1-e)) - TAU, 0.7, 1.0))
    print("T2", {k: v for k, v in R["T2"].items() if k != "rows"})

    # ---- T3 quasi-steady state under persistent disruption ----------
    errs = []; fail_mismatch = 0; cases = 0; margins = []
    pred_spread = {}; sim_spread = {}
    for eps in (0.705, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90):
        ps, ss = [], []
        for s in range(20):
            for mode in ("baseline", "matched"):
                out = simulate_m(sup, tier, run_seed=s, eps=eps, recover_p=0.0,
                                 lag_mode=mode, T=400)
                xp, mg = x_inf(sup, tier, out["hit"], out["eps_j"])
                errs.append(float(np.abs(out["X"][-1] - xp).max()))
                fail_mismatch += int(((out["X"][-1] < TAU) != (xp < TAU)).any())
                cases += 1; margins.append(mg)
                if mode == "baseline":
                    ps.append(100.0 * (xp < TAU).mean()); ss.append(spread(out["F"]))
        pred_spread[str(eps)] = float(np.mean(ps)); sim_spread[str(eps)] = float(np.mean(ss))
    R["T3"] = dict(cases=cases, max_abs_err=max(errs), median_abs_err=float(np.median(errs)),
                   failure_set_mismatches=fail_mismatch, min_margin=float(min(margins)),
                   pred_limit_failing_pct=pred_spread, sim_ever_failed_pct_persistent=sim_spread)
    print("T3", R["T3"])

    # ---- T4 finite-time recovery ------------------------------------
    viol = 0; slack = []; n4 = 0
    for L in (1, 3, 6):
        for s in range(100):
            out = simulate_m(sup, tier, run_seed=s, L=L, T=400)
            last = recov(out["F"]); bound = out["tR"] + (K - 1) * (L + 1) - 1
            viol += int(last > bound); slack.append(bound - last); n4 += 1
    R["T4"] = dict(runs=n4, violations=viol, min_slack=int(min(slack)), mean_slack=float(np.mean(slack)))
    print("T4", R["T4"])

    # ---- T5 delay equivariance --------------------------------------
    Ls = [0, 1, 2, 3, 4, 5, 6]; T5 = {}
    for label, kw in (("matched", dict(lag_mode="matched")),
                      ("baseline_delta0_lam0", dict(lag_mode="baseline", delta=0.0, lam=0.0)),
                      ("baseline", dict(lag_mode="baseline"))):
        tot = {L: [] for L in Ls}; spr = {L: [] for L in Ls}; rec = {L: [] for L in Ls}
        exact_shift = 0; checked = 0; rec_formula_ok = 0
        for s in range(30):
            ref = None
            for L in Ls:
                F = simulate_m(sup, tier, run_seed=s, L=L, T=400, **kw)["F"]
                tot[L].append(total(F)); spr[L].append(spread(F)); rec[L].append(recov(F))
                if L == 0:
                    ref = F; refk = tier_last(F, tier)
                else:
                    ok = True
                    for kk in range(K):
                        cols = tier == kk; sh = kk * L
                        A0 = ref[: F.shape[0] - sh, cols]; A1 = F[sh:, cols]
                        if not np.array_equal(A0, A1) or F[:sh, cols].any(): ok = False
                    exact_shift += int(ok); checked += 1
                    pred = max(v + kk * L for kk, v in enumerate(refk) if v is not None)
                    rec_formula_ok += int(pred == recov(F))
        T5[label] = dict(total_mean={L: float(np.mean(tot[L])) for L in Ls},
                         total_identical_across_L=int(all(len(set(col)) == 1 for col in zip(*[tot[L] for L in Ls]))),
                         spread_mean={L: float(np.mean(spr[L])) for L in Ls},
                         recovery_mean={L: float(np.mean(rec[L])) for L in Ls},
                         exact_shift_runs=exact_shift, checked=checked,
                         recovery_formula_ok=rec_formula_ok)
        print("T5", label, T5[label]["total_mean"], T5[label]["recovery_mean"],
              "shift", exact_shift, "/", checked, "recform", rec_formula_ok)
    R["T5"] = T5

    # ---- T6 rung spectrum (tier-2 node, s = 1 failed supplier) ------
    def rung(m, s=1, lam=LAM):
        def fun(e):
            xs = np.array([1 - e] * s + [1.0] * (m - s))
            ee = np.exp(-lam * (xs < TAU)); wj = ee / ee.sum()
            Psi = ces(wj, xs); d = max(0.0, 1 - DELTA * g(xs).sum())
            return min(d, max(1.0, ALPHA * d) * Psi) - TAU
        return bisect(fun, 1 - TAU + 1e-9, 1.0)
    indeg2 = [len(sup[j]) for j in range(len(tier)) if tier[j] == 1]
    R["T6"] = dict(rung_s1={m: rung(m) for m in range(1, 13)},
                   rung_s2={m: rung(m, 2) for m in range(2, 13)},
                   tier2_indegree_counts={m: indeg2.count(m) for m in sorted(set(indeg2))})
    print("T6", R["T6"])

    # ---- T7 node-level coupling facts -------------------------------
    m = 5; xf, xh = 0.29, 1.0
    lam_star = float(np.log(xh / xf)); Imax = ((m - 1) * xh + xf) / m
    def ces_node(x_f, flagged, lam=LAM):
        xs = np.array([x_f] + [1.0] * (m - 1)); fl = np.array([1.0 if flagged else 0.0] + [0.0] * (m - 1))
        e = np.exp(-lam * fl); return ces(e / e.sum(), xs)
    def req_node(x_f, flagged, lam=LAM):
        xs = np.array([x_f] + [1.0] * (m - 1)); fl = np.array([1.0 if flagged else 0.0] + [0.0] * (m - 1))
        e = np.exp(-lam * fl); w_ = e / e.sum(); return float(min((xs / (w_ * m)).min(), xs.max()))
    R["T7"] = dict(lambda_star=lam_star, Ireq_max=Imax,
                   ces_jump=ces_node(0.29, True) / ces_node(0.31, False),
                   ces_jump_bound=float(((m - 1 + np.exp(-LAM)) * np.exp(LAM) / m) ** (1 / abs(RHO))),
                   req_jump=req_node(0.29, True) / req_node(0.31, False),
                   ces_values=(ces_node(0.31, False), ces_node(0.29, True)),
                   req_values=(req_node(0.31, False), req_node(0.29, True)))
    print("T7", R["T7"])

    with open("results_theory.json", "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())




def extra_checks():
    """T8 (v2.4 addendum): (a) seed-confinement budget: uniform source reserve
    u = eps_max - eps_bar(5) confines failures to the seed set; (b) spread at
    eps = 0.80 with stochastic recovery under both lag conventions versus the
    persistent limit.  Merges key "T8" into results_theory.json."""
    A, sup, tier, _ = build_network(42)
    n = len(tier)
    eb5 = bisect(lambda e: min(1, ALPHA*(1-e))*max(0, 1-DELTA*5*g(1-e)) - TAU, 0.7, 1.0)
    u_src = 0.9 - eb5
    u = np.where(tier == 0, u_src, 0.0)
    bad = 0; seedfail = 0
    for s in range(100):
        F = simulate_m(sup, tier, run_seed=s, u=u, T=200)["F"]
        bad += int(F[:, tier > 0].any()); seedfail += int(F.any())
    sp = {}
    for mode in ("baseline", "matched"):
        sp[mode] = float(np.mean([spread(simulate_m(sup, tier, run_seed=s, eps=0.80, lag_mode=mode, T=50)["F"])
                                  for s in range(30)]))
    R = json.load(open("results_theory.json"))
    R["T8"] = dict(u_per_source=u_src, budget=20 * u_src, runs=100,
                   runs_with_downstream_failure=bad, runs_with_any_failure=seedfail,
                   spread_eps080_recovery=sp)
    json.dump(R, open("results_theory.json", "w"), indent=1)
    print("T8", R["T8"])


if __name__ == "__main__":
    import sys
    if "--only-t8" not in sys.argv:
        main()
    extra_checks()
