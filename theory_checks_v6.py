"""
theory_checks_v6.py -- numerical verification of the interval-enclosure
theorem added in the v6 manuscript (MDPI Mathematics), complementing
theory_checks.py / _v4.py / _v5.py:

  T15  interval enclosure (bracketing) for the FULL model (smooth damping,
       softmax indicator sourcing, either lag convention, edge lags):
       (a) validity: for random disruptions, parameters, non-graded DAGs and
           lag structures, every trajectory whose source paths lie between the
           bounding paths lies between the two bracketing recursions, in
           outputs AND inventories, pathwise;
       (b) worst case over recovery paths on the calibrated network at
           eps = 0.80: the set of nodes the lower bracket certifies as never
           failing, versus the nodes never observed failing over many
           recovery draws, and versus the persistent run;
       (c) exactness in the quantity-only model: the brackets coincide with
           the persistent and the instant-recovery trajectories;
       (d) enclosure over a BOX of lag structures (every L_ij in [a_ij, b_ij]):
           validity on random non-graded DAGs and Example 1, and the certified
           lag-robust safe set versus the observed one.

Run: python3 theory_checks_v6.py   -> results_theory_v6.json
"""
import json
import itertools
import sys
import numpy as np
from multiprocessing import Pool
from model import build_network
from theory_checks import simulate_m, x_inf, g, ces, TAU, XBAR, ALPHA, H0, DELTA, RHO, LAM
from theory_checks_v4 import simulate_g, source_paths, paths_from, random_dag

PREC = 0.06
TOL = 1e-12


def _lagfun(lags):
    if isinstance(lags, (int, np.integer)):
        return lambda i, j: (int(lags), int(lags))
    return lambda i, j: (lags[(i, j)] if isinstance(lags[(i, j)], tuple) else (lags[(i, j)], lags[(i, j)]))


def simulate_gh(suppliers, src_path, *, lags=0, lag_mode="matched", lam=LAM,
                delta=DELTA, alpha=ALPHA, tau=TAU, rho=RHO, h0=H0, T=200):
    """simulate_g with the inventory path returned as well (same arithmetic)."""
    n = len(suppliers)
    hv = np.full(n, float(h0)) if np.isscalar(h0) else np.asarray(h0, float).copy()
    X = np.ones((T + 1, n)) * XBAR
    H = np.zeros((T + 1, n)); H[0] = hv
    for i, p in src_path.items():
        X[:, i] = p
    lf = _lagfun(lags)
    Lij = {j: np.array([lf(i, j)[0] for i in suppliers[j]], int) for j in range(n) if len(suppliers[j])}
    Sj = {j: np.asarray(suppliers[j], int) for j in Lij}
    w = {j: np.full(len(Sj[j]), 1.0 / len(Sj[j])) for j in Sj}

    def rec(idx, times):
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
        H[t + 1] = hv
    return X, H


def enclose(suppliers, src_lo, src_hi, *, lags=0, lag_mode="matched", lam=LAM,
            delta=DELTA, alpha=ALPHA, tau=TAU, rho=RHO, h0=H0, T=200):
    """Interval enclosure (Theorem 5).  src_lo / src_hi: dicts {source: path}
    with src_lo <= src_hi pointwise.  lags: int, or dict {(i,j): L} or
    {(i,j): (a,b)} (a box of admissible lags; every L_ij in [a_ij, b_ij]).
    Returns XL, XH, HL, HH with, for every trajectory whose source paths lie
    between src_lo and src_hi and whose lags lie in the box,
    XL <= X <= XH and HL <= H <= HH."""
    n = len(suppliers)
    h_init = np.full(n, float(h0)) if np.isscalar(h0) else np.asarray(h0, float).copy()
    XL = np.ones((T + 1, n)) * XBAR; XH = XL.copy()
    for i, p in src_lo.items(): XL[:, i] = p
    for i, p in src_hi.items(): XH[:, i] = p
    hL = h_init.copy(); hH = h_init.copy()
    HL = np.zeros((T + 1, n)); HH = np.zeros((T + 1, n)); HL[0] = hL; HH[0] = hH
    lf = _lagfun(lags)
    Sj = {j: np.asarray(suppliers[j], int) for j in range(n) if len(suppliers[j])}
    box = {j: np.array([lf(i, j) for i in suppliers[j]], int) for j in Sj}      # (m, 2)
    # candidate weight vectors for every flag pattern, per in-degree
    wcache = {}

    def weights(m):
        if m not in wcache:
            pats = np.array(list(itertools.product((0, 1), repeat=m)), float)   # (2^m, m)
            e = np.exp(-lam * pats); wcache[m] = (pats, e / e.sum(1, keepdims=True))
        return wcache[m]

    def recL(idx, a, b, t):      # min over window of the lower record
        out = np.empty(len(idx))
        for k, i in enumerate(idx):
            s0, s1 = t - b[k], t - a[k]
            vals = [XBAR if s <= 0 else XL[s, i] for s in range(s0, s1 + 1)]
            out[k] = min(vals)
        return out

    def recH(idx, a, b, t):
        out = np.empty(len(idx))
        for k, i in enumerate(idx):
            s0, s1 = t - b[k], t - a[k]
            vals = [XBAR if s <= 0 else XH[s, i] for s in range(s0, s1 + 1)]
            out[k] = max(vals)
        return out

    for t in range(T):
        for j in Sj:
            S = Sj[j]; a = box[j][:, 0]; b = box[j][:, 1]; m = len(S)
            yL = recL(S, a, b, t); yH = recH(S, a, b, t)
            # feasible flag patterns for the weights used at t (flags read at t-1)
            if t == 0:
                fixed1 = np.zeros(m, bool); fixed0 = np.ones(m, bool)      # uniform weights
            else:
                if lag_mode == "baseline":
                    sL = XL[t - 1, S]; sH = XH[t - 1, S]
                else:
                    sL = recL(S, a, b, t - 1); sH = recH(S, a, b, t - 1)
                fixed1 = sH < tau; fixed0 = sL >= tau
            pats, W = weights(m)
            ok = np.all((pats[:, fixed1] == 1), axis=1) & np.all((pats[:, fixed0] == 0), axis=1)
            Wf = W[ok]
            zL = np.maximum(yL, 1e-12) ** rho; zH = np.maximum(yH, 1e-12) ** rho
            IL = float(((Wf @ zL) ** (1.0 / rho)).min()); IH = float(((Wf @ zH) ** (1.0 / rho)).max())
            dLread = XL[t, S] if lag_mode == "baseline" else yL
            dHread = XH[t, S] if lag_mode == "baseline" else yH
            dL = max(0.0, 1.0 - delta * g(dLread, tau).sum()); dH = max(0.0, 1.0 - delta * g(dHread, tau).sum())
            cL = min(XBAR, alpha * (IL + hL[j])); cH = min(XBAR, alpha * (IH + hH[j]))
            XL[t + 1, j] = dL * cL; XH[t + 1, j] = dH * cH
            hL[j] = max(0.0, hL[j] + IL - dH * cL)
            hH[j] = max(0.0, hH[j] + IH - dL * cH)
        HL[t + 1] = hL; HH[t + 1] = hH
    return XL, XH, HL, HH


def _check(XL, XH, HL, HH, X, H):
    return float(max((XL - X).max(), (X - XH).max(), (HL - H).max(), (H - HH).max()))


# ---------------------------------------------------------------- T15 (a)
def t15a_instance(s):
    rng = np.random.default_rng(6000 + s)
    sup, tier = random_dag(rng)
    n = len(tier); T = 160
    srcs = np.nonzero(tier == 0)[0]
    k = max(1, len(srcs) // 2)
    hit = rng.choice(srcs, size=k, replace=False)
    elo = rng.uniform(0.55, 0.85, size=k); ehi = np.minimum(0.99, elo + rng.uniform(0.0, 0.12, size=k))
    mode = "baseline" if s % 2 == 0 else "matched"
    pars = dict(delta=float(rng.uniform(0, 0.4)), lam=float(rng.uniform(0, 4)), alpha=float(rng.uniform(1, 1.5)),
                rho=float(rng.choice([-10, -5, -2, -1, 0.5, 1])), h0=rng.uniform(0, 0.3, size=n), tau=TAU)
    lags = {(int(i), int(j)): int(rng.integers(0, 4)) for j in range(n) for i in sup[j]}
    # bounding source paths: lower = deepest depth, never recovers; upper = shallowest depth, recovers at 1
    lo = paths_from(hit, ehi, np.full(k, T + 10), T)
    hi = paths_from(hit, elo, np.ones(k), T)
    XL, XH, HL, HH = enclose(sup, lo, hi, lags=lags, lag_mode=mode, T=T, **pars)
    worst = -np.inf; ntraj = 6
    for r in range(ntraj):
        dep = rng.uniform(elo, ehi); rt = rng.geometric(0.08, size=k)
        if r == 0: rt = np.full(k, T + 10)
        if r == 1: rt = np.ones(k)
        X, H = simulate_gh(sup, paths_from(hit, dep, rt, T), lags=lags, lag_mode=mode, T=T, **pars)
        worst = max(worst, _check(XL, XH, HL, HH, X, H))
    width = float((XH - XL).max())
    return dict(seed=s, mode=mode, worst_violation=worst, max_width=width, trajectories=ntraj)


# ---------------------------------------------------------------- T15 (b)
def t15b_cell(args):
    s, d, l, mode, eps, ndraw, T = args
    A, sup, tier, _ = build_network(42)
    n = len(tier)
    base = simulate_m(sup, tier, run_seed=s, eps=eps, lag_mode=mode, delta=d, lam=l, recover_p=0.0, T=T)
    hit = base["hit"]; k = len(hit)
    lo = paths_from(hit, np.full(k, eps), np.full(k, T + 10), T)
    hi = paths_from(hit, np.full(k, eps), np.ones(k), T)
    XL, XH, HL, HH = enclose(sup, lo, hi, lags=3, lag_mode=mode, lam=l, delta=d, T=T)
    # persistent trajectory (= lower source path) and instant-recovery trajectory
    Xp, Hp = simulate_gh(sup, lo, lags=3, lag_mode=mode, lam=l, delta=d, T=T)
    Xi, Hi = simulate_gh(sup, hi, lags=3, lag_mode=mode, lam=l, delta=d, T=T)
    worst = max(_check(XL, XH, HL, HH, Xp, Hp), _check(XL, XH, HL, HH, Xi, Hi))
    ever = (Xp < TAU).any(0) | (Xi < TAU).any(0)
    rng = np.random.default_rng(90000 + s)
    for r in range(ndraw):
        rt = rng.geometric(PREC, size=k)
        X, H = simulate_gh(sup, paths_from(hit, np.full(k, eps), rt, T), lags=3, lag_mode=mode, lam=l, delta=d, T=T)
        worst = max(worst, _check(XL, XH, HL, HH, X, H))
        ever |= (X < TAU).any(0)
    certified_fail = (XH < TAU).any(0)           # fail under every admissible path
    may_fail = (XL < TAU).any(0)                 # complement = certified safe
    xi, _ = x_inf(sup, tier, hit, base["eps_j"], lam=l, delta=d)
    exact_lo = float(np.abs(XL - Xp).max()); exact_hi = float(np.abs(XH - Xi).max())
    return dict(seed=s, delta=d, lam=l, mode=mode, worst_violation=worst,
                certified_safe_pct=100.0 * (1 - may_fail.mean()),
                observed_never_failed_pct=100.0 * (1 - ever.mean()),
                certified_fail_pct=100.0 * certified_fail.mean(),
                persistent_ever_pct=100.0 * (Xp < TAU).any(0).mean(),
                persistent_limit_pct=100.0 * (xi < TAU).mean(),
                observed_union_minus_persistent=int((ever & ~(Xp < TAU).any(0)).sum()),
                slack_nodes=int((may_fail & ~ever).sum()),
                bracket_equals_persistent=exact_lo, bracket_equals_instant=exact_hi,
                lower_limit_nodes_below_tau_pct=100.0 * (XL[-1] < TAU).mean())


# ---------------------------------------------------------------- T15 (d)
def t15d_instance(s):
    rng = np.random.default_rng(7000 + s)
    sup, tier = random_dag(rng)
    n = len(tier); T = 160
    srcs = np.nonzero(tier == 0)[0]
    k = max(1, len(srcs) // 2)
    hit = rng.choice(srcs, size=k, replace=False)
    dep = rng.uniform(0.6, 0.97, size=k); rt = rng.geometric(0.08, size=k)
    mode = "matched"
    pars = dict(delta=float(rng.uniform(0, 0.4)), lam=float(rng.uniform(0, 4)), alpha=float(rng.uniform(1, 1.5)),
                rho=float(rng.choice([-10, -5, -2, -1])), h0=rng.uniform(0, 0.3, size=n), tau=TAU)
    Lmax = 3
    box = {(int(i), int(j)): (0, Lmax) for j in range(n) for i in sup[j]}
    path = paths_from(hit, dep, rt, T)
    XL, XH, HL, HH = enclose(sup, path, path, lags=box, lag_mode=mode, T=T, **pars)
    worst = -np.inf; ever = np.zeros(n, bool); nl = 8
    for r in range(nl):
        lags = {e: int(rng.integers(0, Lmax + 1)) for e in box}
        if r == 0: lags = {e: 0 for e in box}
        if r == 1: lags = {e: Lmax for e in box}
        X, H = simulate_gh(sup, path, lags=lags, lag_mode=mode, T=T, **pars)
        worst = max(worst, _check(XL, XH, HL, HH, X, H)); ever |= (X < TAU).any(0)
    may_fail = (XL < TAU).any(0)
    return dict(seed=s, worst_violation=worst, lag_structures=nl,
                certified_safe_pct=100.0 * (1 - may_fail.mean()),
                observed_never_failed_pct=100.0 * (1 - ever.mean()),
                certified_fail_pct=100.0 * (XH < TAU).any(0).mean(),
                observed_always_failed_pct=None)


def t15d_triangle():
    sup = [[], [0], [0, 1]]; T = 60
    path = {0: np.array([0.1 if t < 12 else XBAR for t in range(T + 1)])}
    out = {}
    for Lmax in (1, 2, 3):
        box = {(0, 1): (0, Lmax), (1, 2): (0, Lmax), (0, 2): (0, Lmax)}
        XL, XH, HL, HH = enclose(sup, path, path, lags=box, lag_mode="matched", T=T)
        lo_fail = [int((XL[:, j] < TAU).sum()) for j in range(3)]
        hi_fail = [int((XH[:, j] < TAU).sum()) for j in range(3)]
        obs = {}
        worst = -np.inf
        for L01 in range(Lmax + 1):
            for L12 in range(Lmax + 1):
                for L02 in range(Lmax + 1):
                    lags = {(0, 1): L01, (1, 2): L12, (0, 2): L02}
                    X, H = simulate_gh(sup, path, lags=lags, lag_mode="matched", T=T)
                    worst = max(worst, _check(XL, XH, HL, HH, X, H))
                    obs[f"{L01}{L12}{L02}"] = [int((X[:, j] < TAU).sum()) for j in range(3)]
        node2 = [v[2] for v in obs.values()]
        out[f"Lmax={Lmax}"] = dict(worst_violation=worst, lower_bracket_failure_periods=lo_fail,
                                   upper_bracket_failure_periods=hi_fail,
                                   observed_node2_failure_periods_range=[min(node2), max(node2)],
                                   lag_structures=len(obs))
    return out


# ---------------------------------------------------------------- T15 (e)
def t15e_cell(args):
    s, eps, ndraw, T = args
    A, sup, tier, _ = build_network(42)
    base = simulate_m(sup, tier, run_seed=s, eps=eps, lag_mode="baseline", recover_p=0.0, T=T)
    hit = base["hit"]; k = len(hit)
    lo = paths_from(hit, np.full(k, eps), np.full(k, T + 10), T)
    hi = paths_from(hit, np.full(k, eps), np.ones(k), T)
    XL, XH, HL, HH = enclose(sup, lo, hi, lags=3, lag_mode="baseline", T=T)
    Xp, _ = simulate_gh(sup, lo, lags=3, lag_mode="baseline", T=T)
    ever = (Xp < TAU).any(0)
    rng = np.random.default_rng(91000 + s)
    for r in range(ndraw):
        rt = rng.geometric(PREC, size=k)
        X, _ = simulate_gh(sup, paths_from(hit, np.full(k, eps), rt, T), lags=3, lag_mode="baseline", T=T)
        ever |= (X < TAU).any(0)
    return dict(seed=s, eps=eps, certified_safe_pct=100.0 * (1 - (XL < TAU).any(0).mean()),
                observed_never_failed_pct=100.0 * (1 - ever.mean()),
                persistent_never_failed_pct=100.0 * (1 - (Xp < TAU).any(0).mean()))


# ---------------------------------------------------------------- T16 lag sandwich
def path_lags(sup, tier, lags):
    """theta_minus / theta_plus: minimal and maximal total lag over source-to-node paths."""
    n = len(tier); tm = np.zeros(n, int); tp = np.zeros(n, int)
    for j in np.argsort(tier, kind="stable"):
        if len(sup[j]) == 0: continue
        tm[j] = min(tm[i] + lags[(int(i), int(j))] for i in sup[j])
        tp[j] = max(tp[i] + lags[(int(i), int(j))] for i in sup[j])
    return tm, tp


def first_fail(X, tau=TAU):
    out = np.full(X.shape[1], -1)
    for j in range(X.shape[1]):
        nz = np.nonzero(X[:, j] < tau)[0]
        if len(nz): out[j] = int(nz[0])
    return out


def t16_instance(s):
    rng = np.random.default_rng(8000 + s)
    sup, tier = random_dag(rng)
    n = len(tier); T = 200
    srcs = np.nonzero(tier == 0)[0]
    k = max(1, len(srcs) // 2)
    hit = rng.choice(srcs, size=k, replace=False)
    dep = rng.uniform(0.6, 0.97, size=k)
    path = paths_from(hit, dep, np.full(k, T + 10), T)         # persistent
    lags = {(int(i), int(j)): int(rng.integers(0, 4)) for j in range(n) for i in sup[j]}
    tm, tp = path_lags(sup, tier, lags)
    pars = dict(alpha=float(rng.uniform(1, 1.5)), rho=float(rng.choice([-10, -5, -2, -1, 0.5, 1])),
                h0=rng.uniform(0, 0.3, size=n), tau=TAU)
    out = {}
    for name, (d, l) in (("quantity_only", (0.0, 0.0)), ("full", (float(rng.uniform(0.1, 0.4)), float(rng.uniform(0.5, 4))))):
        mode = "baseline" if s % 2 == 0 else "matched"
        X0, _ = simulate_gh(sup, path, lags=0, lag_mode=mode, delta=d, lam=l, T=T, **pars)
        XL, _ = simulate_gh(sup, path, lags=lags, lag_mode=mode, delta=d, lam=l, T=T, **pars)
        # sandwich x0(t - theta_minus) <= xL(t) <= x0(t - theta_plus), with x0 = xbar at negative times
        lo = np.ones_like(XL) * XBAR; hi = np.ones_like(XL) * XBAR
        for j in range(n):
            for t in range(T + 1):
                if t - tm[j] >= 0: lo[t, j] = X0[t - tm[j], j]      # x0(t - theta_minus) <= xL(t)
                if t - tp[j] >= 0: hi[t, j] = X0[t - tp[j], j]      # xL(t) <= x0(t - theta_plus)
        viol = float(max((lo - XL).max(), (XL - hi).max()))
        f0 = first_fail(X0); fL = first_fail(XL)
        both = (f0 >= 0) & (fL >= 0)
        shift = fL[both] - f0[both]
        inside = bool(np.all((shift >= tm[both]) & (shift <= tp[both]))) and bool(np.all((f0 >= 0) == (fL >= 0)))
        out[name] = dict(max_violation=viol, sandwich_holds=viol <= TOL, first_failure_shift_in_window=inside,
                         failing_nodes=int(both.sum()),
                         nodes_attaining_upper=int(np.sum(shift == tp[both])),
                         nodes_attaining_lower=int(np.sum(shift == tm[both])),
                         nodes_imbalanced=int(np.sum(tp[both] > tm[both])),
                         mean_imbalance=float(np.mean(tp - tm)))
    return dict(seed=s, **out)


def main_extra(quick=False):
    R = json.load(open("results_theory_v6.json")) if __import__("os").path.exists("results_theory_v6.json") else {}
    with Pool() as pool:
        rows = pool.map(t16_instance, range(10 if quick else 60))
        R["T16_lag_sandwich"] = dict(
            instances=len(rows),
            quantity_only=dict(instances_violating=int(sum(not r["quantity_only"]["sandwich_holds"] for r in rows)),
                               max_violation=max(r["quantity_only"]["max_violation"] for r in rows),
                               first_failure_window_ok=int(sum(r["quantity_only"]["first_failure_shift_in_window"] for r in rows)),
                               failing_nodes=int(sum(r["quantity_only"]["failing_nodes"] for r in rows)),
                               imbalanced_failing_nodes=int(sum(r["quantity_only"]["nodes_imbalanced"] for r in rows)),
                               attaining_upper=int(sum(r["quantity_only"]["nodes_attaining_upper"] for r in rows)),
                               attaining_lower=int(sum(r["quantity_only"]["nodes_attaining_lower"] for r in rows))),
            full_model=dict(instances_violating=int(sum(not r["full"]["sandwich_holds"] for r in rows)),
                            max_violation=max(r["full"]["max_violation"] for r in rows),
                            first_failure_window_ok=int(sum(r["full"]["first_failure_shift_in_window"] for r in rows))),
            rows=rows)
        print("T16", {k: v for k, v in R["T16_lag_sandwich"].items() if k != "rows"}, flush=True)
        seeds = range(5 if quick else 30); ndraw = 5 if quick else 20
        jobs = [(s, e, ndraw, 200) for e in (0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85) for s in seeds]
        rows = pool.map(t15e_cell, jobs)
        sweep = {}
        for e in (0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85):
            rr = [r for r in rows if r["eps"] == e]
            sweep[str(e)] = dict(certified_safe_pct=float(np.mean([r["certified_safe_pct"] for r in rr])),
                                 observed_never_failed_pct=float(np.mean([r["observed_never_failed_pct"] for r in rr])),
                                 persistent_never_failed_pct=float(np.mean([r["persistent_never_failed_pct"] for r in rr])))
        R["T15e_eps_sweep"] = dict(seeds=len(seeds), draws=ndraw, sweep=sweep)
        print("T15e", sweep, flush=True)
    json.dump(R, open("results_theory_v6.json", "w"), indent=1)


def main(quick=False):
    R = {}
    with Pool() as pool:
        # (a) validity on random instances
        rows = pool.map(t15a_instance, range(10 if quick else 60))
        R["T15a_validity"] = dict(instances=len(rows), worst_violation=max(r["worst_violation"] for r in rows),
                                  instances_violating=int(sum(r["worst_violation"] > TOL for r in rows)),
                                  trajectories_per_instance=rows[0]["trajectories"], rows=rows)
        print("T15a", {k: v for k, v in R["T15a_validity"].items() if k != "rows"}, flush=True)
        # (b) worst case over recovery paths, calibrated network
        seeds = range(5 if quick else 30); ndraw = 10 if quick else 50; T = 200
        jobs = [(s, d, l, mode, 0.80, ndraw, T) for mode in ("baseline", "matched")
                for (d, l) in ((0.0, 0.0), (DELTA, 0.0), (0.0, LAM), (DELTA, LAM)) for s in seeds]
        rows = pool.map(t15b_cell, jobs)
        cells = {}
        for mode in ("baseline", "matched"):
            for (d, l) in ((0.0, 0.0), (DELTA, 0.0), (0.0, LAM), (DELTA, LAM)):
                rr = [r for r in rows if r["mode"] == mode and r["delta"] == d and r["lam"] == l]
                cells[f"delta{d}_lam{l}_{mode}"] = dict(
                    seeds=len(rr), recovery_draws_per_seed=ndraw, T=T,
                    worst_violation=max(r["worst_violation"] for r in rr),
                    certified_safe_pct=float(np.mean([r["certified_safe_pct"] for r in rr])),
                    observed_never_failed_pct=float(np.mean([r["observed_never_failed_pct"] for r in rr])),
                    certified_fail_pct=float(np.mean([r["certified_fail_pct"] for r in rr])),
                    persistent_ever_pct=float(np.mean([r["persistent_ever_pct"] for r in rr])),
                    persistent_limit_pct=float(np.mean([r["persistent_limit_pct"] for r in rr])),
                    mean_slack_nodes=float(np.mean([r["slack_nodes"] for r in rr])),
                    max_slack_nodes=int(max(r["slack_nodes"] for r in rr)),
                    seeds_with_zero_slack=int(sum(r["slack_nodes"] == 0 for r in rr)),
                    runs_observed_beyond_persistent=int(sum(r["observed_union_minus_persistent"] > 0 for r in rr)),
                    bracket_equals_persistent_max=max(r["bracket_equals_persistent"] for r in rr),
                    bracket_equals_instant_max=max(r["bracket_equals_instant"] for r in rr),
                    lower_bracket_limit_below_tau_pct=float(np.mean([r["lower_limit_nodes_below_tau_pct"] for r in rr])))
                print(mode, d, l, cells[f"delta{d}_lam{l}_{mode}"], flush=True)
        R["T15b_recovery_worst_case"] = dict(cells=cells, rows=rows)
        # (d) lag box
        rows = pool.map(t15d_instance, range(10 if quick else 60))
        R["T15d_lagbox"] = dict(instances=len(rows), worst_violation=max(r["worst_violation"] for r in rows),
                                instances_violating=int(sum(r["worst_violation"] > TOL for r in rows)),
                                certified_safe_pct=float(np.mean([r["certified_safe_pct"] for r in rows])),
                                observed_never_failed_pct=float(np.mean([r["observed_never_failed_pct"] for r in rows])),
                                certified_fail_pct=float(np.mean([r["certified_fail_pct"] for r in rows])),
                                rows=rows)
        print("T15d", {k: v for k, v in R["T15d_lagbox"].items() if k != "rows"}, flush=True)
    R["T15d_triangle"] = t15d_triangle()
    print("T15d triangle", R["T15d_triangle"], flush=True)
    json.dump(R, open("results_theory_v6.json", "w"), indent=1)


if __name__ == "__main__":
    q = "--quick" in sys.argv
    if "--only-extra" not in sys.argv:
        main(quick=q)
    main_extra(quick=q)
