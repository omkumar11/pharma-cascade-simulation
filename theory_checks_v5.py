"""
theory_checks_v5.py -- v5 addendum (MDPI Mathematics manuscript v5):

  T13  mechanism of the protective damping lead (Table "channels"):
       damping-only model (delta = 0.2, lambda = 0), 30 seed sets, T = 400.
       (a) mean inventory, at the period before the physical shortfall first
           reaches depth one (t = L+1), of nodes with a disrupted supplier,
           baseline versus matched lags;
       (b) causal test: the same comparison with inventory accumulation
           disabled ("no storage": h(t+1) <= h(t)).  If the damping lead
           protects by building inventory ahead of the shortfall, its
           protective deviation from the matched-lag invariant must vanish
           when storage is disabled.  (Equivariance still holds under matched
           lags without storage: the node map remains a function of lagged
           reads only.)

  T14  (a) benchmark for "recovery fails nodes a persistent disruption never
           fails": per channel cell and lag convention at eps = 0.80, the
           spread of the persistent run's EVER-failed set (not only its limit
           set) and the number of the 30 runs in which the recovery run fails
           a node that the persistent run never fails;
       (b) rate check of results_theory_v4.json T12 recomputed with the
           genericity condition restricted to nodes with d_inf > 0 that have a
           disrupted ancestor, without the 1e-12 floor of ces(); range of
           fitted rate minus r*;
       (c) recovery-time slopes in L from results_theory.json (T5).

Run: python3 theory_checks_v5.py   (~10 min) -> results_theory_v5.json
     python3 theory_checks_v5.py --only-t14
"""
import json
import numpy as np
from model import build_network
from theory_checks import g, ces, TAU, XBAR, ALPHA, H0, DELTA, RHO, PREC, LAM

EPSR = (0.6, 0.9)


def simulate_s(suppliers, tier, *, run_seed=0, lag_mode="baseline", L=3,
               lam=0.0, delta=DELTA, storage=True, T=400):
    """simulate_m (theory_checks.py) with an inventory record and a switch
    that disables inventory accumulation.  storage=True reproduces simulate_m
    exactly (checked in main)."""
    rng = np.random.default_rng(run_seed)
    n = len(tier)
    t1 = np.nonzero(tier == 0)[0]
    k = max(1, int(round(0.25 * len(t1))))
    hit = rng.choice(t1, size=k, replace=False)
    eps_j = np.zeros(n); eps_j[hit] = rng.uniform(*EPSR, size=len(hit))
    hist = np.ones((L + 1, n))
    x = np.ones(n); x[hit] = XBAR * (1 - eps_j[hit])
    h = np.full(n, H0)
    w = {j: np.full(len(suppliers[j]), 1.0 / len(suppliers[j]))
         for j in range(n) if len(suppliers[j])}
    shocked = np.zeros(n, bool); shocked[hit] = True
    F = np.zeros((T + 1, n), bool); F[0] = x < TAU
    X = np.zeros((T + 1, n)); X[0] = x
    Hs = np.zeros((T + 1, n)); Hs[0] = h
    for t in range(T):
        xlag = hist[-1]; f = F[t].astype(float)
        rec = shocked & (rng.random(n) < PREC); shocked &= ~rec
        xn = np.empty(n); hn = h.copy()
        dread = X[t] if lag_mode == "baseline" else xlag
        for j in range(n):
            S = suppliers[j]
            if len(S) == 0:
                xn[j] = XBAR * (1 - eps_j[j]) if shocked[j] else XBAR
                continue
            I = ces(w[j], xlag[S], RHO)
            xt = min(XBAR, ALPHA * (I + h[j]))
            damp = max(0.0, 1.0 - delta * g(dread[S]).sum())
            xn[j] = xt * damp
            hnew = max(0.0, h[j] + I - xn[j])
            hn[j] = hnew if storage else min(h[j], hnew)
        fw = f if lag_mode == "baseline" else (xlag < TAU).astype(float)
        for j in range(n):
            S = suppliers[j]
            if len(S):
                e = np.exp(-lam * fw[S]); w[j] = e / e.sum()
        h = hn
        hist = np.vstack([xn, hist[:-1]])
        F[t + 1] = xn < TAU; X[t + 1] = xn; Hs[t + 1] = h
    return dict(F=F, X=X, H=Hs, hit=hit)


def main():
    from theory_checks import simulate_m
    A, sup, tier, _ = build_network(42)
    n = len(tier)
    # exactness of simulate_s against simulate_m (storage on)
    same = True
    for s in range(3):
        for mode in ("baseline", "matched"):
            a = simulate_s(sup, tier, run_seed=s, lag_mode=mode, T=120)
            b = simulate_m(sup, tier, run_seed=s, lag_mode=mode, lam=0.0, T=120)
            same &= bool(np.array_equal(a["F"], b["F"]) and np.array_equal(a["X"], b["X"]))
    R = {"T0_reproduces_simulate_m": same}
    seeds = range(30)
    out = {}
    for storage in (True, False):
        for L in (1, 3, 6):
            tot = {}; hb = {}
            for mode in ("baseline", "matched"):
                tots = []; hs = []
                for s in seeds:
                    o = simulate_s(sup, tier, run_seed=s, lag_mode=mode, L=L, storage=storage)
                    tots.append(int(o["F"].sum()))
                    hs_ = set(o["hit"].tolist())
                    expo = [j for j in range(n) if any(i in hs_ for i in sup[j])]
                    hs.append(float(o["H"][L + 1, expo].mean()))
                tot[mode] = float(np.mean(tots)); hb[mode] = float(np.mean(hs))
            key = f"storage={storage},L={L}"
            out[key] = dict(total_baseline=tot["baseline"], total_matched=tot["matched"],
                            deviation_pct=100.0 * (tot["baseline"] / tot["matched"] - 1),
                            inventory_at_L1_baseline=hb["baseline"],
                            inventory_at_L1_matched=hb["matched"])
            print(key, out[key])
    R["T13"] = out
    json.dump(R, open("results_theory_v5.json", "w"), indent=1)


def t14():
    from theory_checks import simulate_m, x_inf
    from theory_checks_v4 import simulate_g, source_paths, paths_from
    A, sup, tier, _ = build_network(42)
    n = len(tier)
    R = json.load(open("results_theory_v5.json")) if __import__("os").path.exists("results_theory_v5.json") else {}
    cells = {}
    for d, l in ((0.0, 0.0), (DELTA, 0.0), (0.0, 2.0), (DELTA, 2.0)):
        for mode in ("baseline", "matched"):
            rec_s, ever_s, lim_s, n_new, n_more = [], [], [], 0, 0
            for s in range(30):
                Fr = simulate_m(sup, tier, run_seed=s, eps=0.80, lag_mode=mode, delta=d, lam=l, T=50)["F"]
                o = simulate_m(sup, tier, run_seed=s, eps=0.80, lag_mode=mode, delta=d, lam=l,
                               recover_p=0.0, T=400)
                xi, _ = x_inf(sup, tier, o["hit"], o["eps_j"], lam=l, delta=d)
                er, ep = Fr.any(0), o["F"].any(0)
                rec_s.append(100 * er.mean()); ever_s.append(100 * ep.mean()); lim_s.append(100 * (xi < TAU).mean())
                n_new += int((er & ~ep).any()); n_more += int(er.sum() > ep.sum())
            cells[f"delta{d}_lam{l}_{mode}"] = dict(spread_recovery=float(np.mean(rec_s)),
                persistent_ever_failed=float(np.mean(ever_s)), persistent_limit=float(np.mean(lim_s)),
                runs_recovery_fails_node_persistent_never_fails=n_new,
                runs_recovery_spread_above_persistent_ever=n_more)
            print(mode, d, l, cells[f"delta{d}_lam{l}_{mode}"])
    # (b) genericity / rate recomputation
    anc = [set() for _ in range(n)]
    for j in np.argsort(tier, kind="stable"):
        for i in sup[j]:
            anc[j] |= anc[i] | {i}
    rows = []
    for s in range(40):
        rng = np.random.default_rng(4000 + s)
        hit, dep, rt = source_paths(tier, rng, 400, eps_rng=(0.75, 0.97))
        mode = "baseline" if s % 2 == 0 else "matched"
        X, _ = simulate_g(sup, paths_from(hit, dep, np.full(len(hit), 10**6), 400), lags=3, lag_mode=mode, T=400)
        ej = np.zeros(n); ej[hit] = dep
        xi, _ = x_inf(sup, tier, hit, ej)
        hs = set(hit.tolist()); gen = np.inf; nondeg = np.inf; rstar = 0.0
        for j in np.argsort(tier, kind="stable"):
            S = sup[j]
            if len(S) == 0: continue
            xs = xi[S]; e = np.exp(-LAM * (xs < TAU)); wj = e / e.sum()
            Psi = float((wj @ xs ** RHO) ** (1 / RHO)) if (xs > 0).all() else 0.0
            dd = max(0.0, 1.0 - DELTA * g(xs).sum())
            nondeg = min(nondeg, abs(xi[j] - TAU))
            if dd > 0 and (anc[j] & hs):
                gen = min(gen, abs(Psi - dd * XBAR) if ALPHA * dd < 1 else abs(ALPHA * Psi - XBAR))
            if ALPHA * dd < 1 and Psi < dd * XBAR and dd > 0:
                rstar = max(rstar, 1.0 - ALPHA * dd)
        err = np.abs(X - xi).max(1)
        idx = np.nonzero((err < 1e-3) & (err > 1e-11))[0]
        fit = None
        if rstar > 0 and len(idx) >= 8:
            tt = idx[len(idx) // 2:]
            fit = float(np.exp(np.polyfit(tt, np.log(err[tt]), 1)[0]))
        rows.append(dict(seed=s, r_star=rstar, fitted=fit, genericity_margin=float(gen), nondegeneracy_margin=float(nondeg)))
    fits = [(r["fitted"] - r["r_star"]) for r in rows if r["fitted"] is not None]
    gens = [r["genericity_margin"] for r in rows]
    # (c) recovery slopes from T5
    T5 = json.load(open("results_theory.json"))["T5"]
    def slope(key, Ls):
        y = [T5[key]["recovery_mean"][str(L)] for L in Ls]
        return float(np.polyfit(Ls, y, 1)[0])
    R["T14"] = dict(channels=cells, rate_rows=rows, rate_fit_minus_rstar_min=float(min(fits)),
                    rate_fit_minus_rstar_max=float(max(fits)), n_fits=len(fits),
                    genericity_margin_min=float(min(gens)), instances_generic=int(sum(g_ > 1e-9 for g_ in gens)),
                    recovery_slope=dict(matched_L0_6=slope("matched", range(7)), matched_L1_6=slope("matched", range(1, 7)),
                                        baseline_L0_6=slope("baseline", range(7)), baseline_L1_6=slope("baseline", range(1, 7)),
                                        baseline_d0l0_L0_6=slope("baseline_delta0_lam0", range(7))))
    print({k: v for k, v in R["T14"].items() if k not in ("channels", "rate_rows")})
    json.dump(R, open("results_theory_v5.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    if "--only-t14" not in sys.argv:
        main()
    t14()
