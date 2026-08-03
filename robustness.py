"""
Embedding & curvature robustness module — Section 6.8 experiment.
Designed to run on the ORIGINAL simulation's output, not the reimplementation.

Required inputs:
  F        : (T+1, n) boolean failure matrix from one simulation run
  A        : (n, n) boolean adjacency (A[i,j]=1 means i supplies j)
  tier     : (n,) int tier assignment (0-indexed upstream->downstream)
  ranks    : (n,) float within-tier rank normalized to [0,1]

Usage in Om's repo:
  from robustness import run_experiment
  run_experiment(simulate_fn, A, tier, ranks, lams=[0,1,2,3,5], n_runs=30)
where simulate_fn(lam, seed) -> F matrix from the audited original code.
"""
import numpy as np
from scipy.stats import spearmanr


def embeddings(A, tier, ranks):
    K = tier.max()
    z_tr = np.column_stack([tier / max(K, 1), ranks])
    As = (A | A.T).astype(float)
    Lap = np.diag(As.sum(1)) - As
    _, vecs = np.linalg.eigh(Lap)
    z_sp = vecs[:, 1:3]
    z_sp = (z_sp - z_sp.min(0)) / np.ptp(z_sp, 0).clip(1e-12)
    return {"tier_rank": z_tr, "spectral": z_sp}


def curvature_series(F, z, sigma=0.15, grid=70, pad=3.0):
    """C(t) = integral ||Hess phi||_F^2 over a grid covering the field
    (pad in units of sigma beyond [0,1]^2). Returns list of (C, n_fail)."""
    lo, hi = -pad * sigma, 1 + pad * sigma
    gx = np.linspace(lo, hi, grid)
    cell = (gx[1] - gx[0]) ** 2
    GX, GY = np.meshgrid(gx, gx)
    c = 1.0 / (2 * np.pi * sigma ** 2)
    out = []
    for t in range(F.shape[0]):
        idx = np.nonzero(F[t])[0]
        if len(idx) == 0:
            out.append((0.0, 0)); continue
        ux = GX[..., None] - z[idx, 0]
        uy = GY[..., None] - z[idx, 1]
        Kv = c * np.exp(-(ux ** 2 + uy ** 2) / (2 * sigma ** 2))
        H11 = (Kv * (ux * ux / sigma ** 4 - 1 / sigma ** 2)).sum(-1)
        H22 = (Kv * (uy * uy / sigma ** 4 - 1 / sigma ** 2)).sum(-1)
        H12 = (Kv * (ux * uy / sigma ** 4)).sum(-1)
        C = float(((H11 ** 2 + 2 * H12 ** 2 + H22 ** 2).sum()) * cell)
        out.append((C, len(idx)))
    return out


def run_experiment(simulate_fn, A, tier, ranks, lams=(0, 1, 2, 3, 5),
                   n_runs=30, sigma=0.15):
    Z = embeddings(np.asarray(A, bool), np.asarray(tier), np.asarray(ranks))
    rows = {emb: {"maxC": [], "maxCn": []} for emb in Z}
    Ftot = []
    for lam in lams:
        Fs = [np.asarray(simulate_fn(lam, s), bool) for s in range(n_runs)]
        Ftot.append((np.mean([F.sum() for F in Fs]),
                     np.std([F.sum() for F in Fs])))
        for emb, z in Z.items():
            mC, mCn = [], []
            for F in Fs:
                cs = curvature_series(F, z, sigma=sigma)
                mC.append(max(c for c, _ in cs))
                mCn.append(max((c / nf ** 2 if nf else 0.0) for c, nf in cs))
            rows[emb]["maxC"].append((np.mean(mC), np.std(mC)))
            rows[emb]["maxCn"].append((np.mean(mCn), np.std(mCn)))
    # ---- report (paste-ready for Section 6.8) ----
    print("lambda | total failures F      | " +
          " | ".join(f"max C(t) [{e}]" for e in Z))
    for i, lam in enumerate(lams):
        f = Ftot[i]
        cells = " | ".join(f"{rows[e]['maxC'][i][0]:.3g} ± {rows[e]['maxC'][i][1]:.2g}"
                           for e in Z)
        print(f"{lam:6} | {f[0]:8.0f} ± {f[1]:5.0f} | {cells}")
    for e in Z:
        mu = [m for m, _ in rows[e]["maxC"]]
        rho = spearmanr(lams, mu)[0] if len(set(mu)) > 1 else float("nan")
        mono_up = all(mu[i + 1] >= mu[i] for i in range(len(mu) - 1))
        mono_dn = all(mu[i + 1] <= mu[i] for i in range(len(mu) - 1))
        print(f"[{e}] Spearman(lambda, maxC) = {rho:+.2f}; "
              f"monotone increasing: {mono_up}; monotone decreasing: {mono_dn}")
    fmu = [m for m, _ in Ftot]
    print(f"F decreasing in lambda: "
          f"{all(fmu[i+1] <= fmu[i] for i in range(len(fmu)-1))}")
    return rows, Ftot
