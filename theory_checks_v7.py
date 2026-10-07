"""
theory_checks_v7.py -- v7 addendum (final MDPI Mathematics manuscript):

  T17  breakdown of the onset-window check of the lag sandwich (Theorem 8)
       over FAILING NON-SOURCE nodes only (disrupted sources trivially have
       zero imbalance and attain both ends of the window): failing non-source
       nodes, how many have positive lag imbalance, and where their first
       failure falls in the window [theta^-(j), theta^+(j)] -- at the upper
       end, at the lower end, or strictly inside.  Same 60 instances
       (seeds 8000-8059) and same quantity-only trajectories as T16 in
       theory_checks_v6.py.

Run: python3 theory_checks_v7.py   (~2 min) -> results_theory_v7.json
"""
import json
import numpy as np
from multiprocessing import Pool
from theory_checks import TAU
from theory_checks_v4 import paths_from, random_dag
from theory_checks_v6 import simulate_gh, path_lags, first_fail


def t17_instance(s):
    # identical random stream to theory_checks_v6.t16_instance
    rng = np.random.default_rng(8000 + s)
    sup, tier = random_dag(rng)
    n = len(tier); T = 200
    srcs = np.nonzero(tier == 0)[0]
    k = max(1, len(srcs) // 2)
    hit = rng.choice(srcs, size=k, replace=False)
    dep = rng.uniform(0.6, 0.97, size=k)
    path = paths_from(hit, dep, np.full(k, T + 10), T)
    lags = {(int(i), int(j)): int(rng.integers(0, 4)) for j in range(n) for i in sup[j]}
    tm, tp = path_lags(sup, tier, lags)
    pars = dict(alpha=float(rng.uniform(1, 1.5)), rho=float(rng.choice([-10, -5, -2, -1, 0.5, 1])),
                h0=rng.uniform(0, 0.3, size=n), tau=TAU)
    mode = "baseline" if s % 2 == 0 else "matched"
    X0, _ = simulate_gh(sup, path, lags=0, lag_mode=mode, delta=0.0, lam=0.0, T=T, **pars)
    XL, _ = simulate_gh(sup, path, lags=lags, lag_mode=mode, delta=0.0, lam=0.0, T=T, **pars)
    f0 = first_fail(X0); fL = first_fail(XL)
    ns = np.array([len(sup[j]) > 0 for j in range(n)])
    both = (f0 >= 0) & (fL >= 0) & ns
    shift = fL[both] - f0[both]; lo = np.asarray(tm)[both]; hi = np.asarray(tp)[both]
    imb = hi > lo
    return dict(seed=s, failing_nonsource=int(both.sum()),
                in_window=int(np.sum((shift >= lo) & (shift <= hi))),
                at_upper=int(np.sum(shift == hi)), at_lower=int(np.sum(shift == lo)),
                imbalanced=int(imb.sum()),
                imb_at_upper=int(np.sum(imb & (shift == hi))), imb_at_lower=int(np.sum(imb & (shift == lo))),
                imb_inside=int(np.sum(imb & (shift > lo) & (shift < hi))))


if __name__ == "__main__":
    with Pool() as pool:
        rows = pool.map(t17_instance, range(60))
    tot = {k: int(sum(r[k] for r in rows)) for k in rows[0] if k != "seed"}
    print("T17", tot)
    json.dump({"T17_onset_window_nonsource": dict(totals=tot, rows=rows)},
              open("results_theory_v7.json", "w"), indent=1)
