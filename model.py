"""
model.py — Canonical simulation for Kumar & Salwekar,
"Geometry, Thresholds, and Control of Cascading Failures in Tiered
Pharmaceutical Supply Networks" (revised).

This module REPLACES the lost original implementation. All Section 6
results are regenerated from this code. Design decisions, all of which
must be reflected in the revised paper text:

  M1. Input aggregation (baseline "Leontief"): requirements form
          I_j = min_i x_i / (w_ij * m_j),  m_j = |N-(j)|,
      capped at max_i x_i. Under uniform weights this equals min_i x_i;
      adaptive downweighting of a failing supplier relaxes its
      requirement. This replaces the manuscript's I = min_i w_ij x_i,
      which is inconsistent with the CES limit (the weighted power mean
      tends to min_i x_i as rho -> -inf) and produces full-network
      collapse under any shock that disables one supplier.
  M2. CES variant: I_j = (sum_i w_ij x_i^rho)^(1/rho), rho in (-inf,1]\{0}.
  M3. Shock protocol: at t=0, a uniformly random 25% of tier-1 nodes
      are disrupted; each disrupted node's capacity reduction eps_j is
      drawn U(0.6, 0.9) per run, matching the 60-90% disruption profile
      of Kim et al. (2022) cited in Section 1.4. (A scalar eps override
      is retained for controlled threshold experiments.)
      Each shocked node recovers independently with probability
      p_rec = 0.06 per period (geometric duration, mean ~17 periods,
      i.e. a multi-lead-time-cycle disruption). Run-to-run randomness
      = shocked-set draw + recovery draws. Network topology fixed at
      seed 42 per the paper.
  M4. Damping factor clamped at 0 (the paper's Feasibility Assumption
      delta <= 1/max in-degree is violated by delta >= 0.2 at p=0.15).
  M5. Intervention channel (pre-positioned reserves): control u_j
      (i) adds to the initial inventory buffer of non-source nodes,
      h_j(0) = h0 + u_j, and (ii) for source (tier-1) nodes, offsets
      shocked capacity: while shocked, x_j = min(xbar, xbar(1-eps)+u_j).
      Interpretation: a reserve stock released during disruption.
"""
import numpy as np

TIER_SIZES = (20, 35, 50, 41)
DEFAULTS = dict(T=50, L=2, delta=0.2, lam=2.0, alpha=1.2, tau=0.3,
                xbar=1.0, h0=0.1, eps=None, eps_range=(0.6, 0.9),
                recover_p=0.06, shock_frac=0.25)


def build_network(seed=42, tier_sizes=TIER_SIZES, p=0.15):
    rng = np.random.default_rng(seed)
    n = sum(tier_sizes)
    tier = np.zeros(n, dtype=int)
    idx = 0; bounds = []
    for k, s in enumerate(tier_sizes):
        tier[idx:idx + s] = k
        bounds.append((idx, idx + s)); idx += s
    A = np.zeros((n, n), dtype=bool)
    for k in range(len(tier_sizes) - 1):
        u0, u1 = bounds[k]; d0, d1 = bounds[k + 1]
        block = rng.random((u1 - u0, d1 - d0)) < p
        for jc in range(d1 - d0):
            if not block[:, jc].any():
                block[rng.integers(0, u1 - u0), jc] = True
        A[u0:u1, d0:d1] = block
    suppliers = [np.nonzero(A[:, j])[0] for j in range(n)]
    ranks = np.zeros(n)
    for k, (a, b) in enumerate(bounds):
        m = b - a
        ranks[a:b] = np.arange(m) / max(m - 1, 1)
    return A, suppliers, tier, ranks


def simulate(A, suppliers, tier, *, run_seed=0, agg="leontief", rho=None,
             u=None, collect_x=False, **kw):
    p = {**DEFAULTS, **kw}
    T, L = p["T"], p["L"]
    rng = np.random.default_rng(run_seed)
    n = len(tier)
    t1 = np.nonzero(tier == 0)[0]
    k = max(1, int(round(p["shock_frac"] * len(t1))))
    hit = rng.choice(t1, size=k, replace=False)
    m = np.array([max(len(s), 1) for s in suppliers])

    uj_res = np.zeros(n) if u is None else np.asarray(u, float)
    eps_j = np.zeros(n)
    if p["eps"] is not None:
        eps_j[hit] = p["eps"]
    else:
        eps_j[hit] = rng.uniform(*p["eps_range"], size=len(hit))
    hist = np.ones((L + 1, n))
    x = np.ones(n)
    x[hit] = np.minimum(p["xbar"], p["xbar"] * (1 - eps_j[hit]) + uj_res[hit])
    h = np.full(n, p["h0"]) + uj_res
    w = {j: np.full(len(suppliers[j]), 1.0 / len(suppliers[j]))
         for j in range(n) if len(suppliers[j])}
    shocked = np.zeros(n, bool); shocked[hit] = True
    F = np.zeros((T + 1, n), bool); F[0] = x < p["tau"]
    X = np.zeros((T + 1, n)); X[0] = x

    for t in range(T):
        xlag = hist[-1]; f = F[t].astype(float)
        rec = shocked & (rng.random(n) < p["recover_p"]); shocked &= ~rec
        xn = np.empty(n); hn = h.copy()
        for j in range(n):
            S = suppliers[j]
            if len(S) == 0:
                if shocked[j]:
                    xn[j] = min(p["xbar"], p["xbar"] * (1 - eps_j[j]) + uj_res[j])
                else:
                    xn[j] = p["xbar"]
                continue
            wj = w[j]
            if agg == "leontief":
                I = float(np.min(xlag[S] / np.clip(wj * m[j], 1e-9, None)))
                I = min(I, float(xlag[S].max()))
            else:
                I = float((wj @ np.maximum(xlag[S], 1e-12) ** rho) ** (1.0 / rho))
            xt = min(p["xbar"], p["alpha"] * (I + h[j]))
            damp = max(0.0, 1.0 - p["delta"] * f[S].sum())
            xn[j] = xt * damp
            hn[j] = max(0.0, h[j] + I - xn[j])
        for j in range(n):
            S = suppliers[j]
            if len(S):
                e = np.exp(-p["lam"] * f[S]); w[j] = e / e.sum()
        h = hn
        hist = np.vstack([xn, hist[:-1]])
        F[t + 1] = xn < p["tau"]; X[t + 1] = xn
    return (F, X) if collect_x else F


def metrics(F):
    anyf = np.nonzero(F.any(1))[0]
    return dict(total=int(F.sum()), peak=int(F.sum(1).max()),
                spread=100.0 * F.any(0).sum() / F.shape[1],
                recovery=int(anyf.max()) if len(anyf) else 0)


def centrality(A, eta=0.5):
    """Downstream influence C(j) = sum_k eta^d(j,k), omega=1 (Def 3.13)."""
    from collections import deque
    n = A.shape[0]
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
        C[j] = float(np.sum(0.5 ** dist[reach]))
    return C
