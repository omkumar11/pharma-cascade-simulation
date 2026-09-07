"""
game_exact.py -- v2.3 supplementary check Q6: the resilience-investment game
of Section VII with the tier-1 (source) private problem solved in closed form.

A source firm's failures depend only on its own shock draw and its own
reserve (source output is exogenous), so its expected failure cost is

    Phi_1(u) = psi * P_d * S_T * max(0, eps_max - (1 - tau) - u) / (eps_max - eps_min),
    P_d = k / n_1,   S_T = sum_{t=0}^{T} (1 - p_rec)^t,

independent of every other firm's reserve: the private optimum
u_priv = min{ beta_c * psi * P_d * S_T / (eps_max - eps_min), eps_max - (1 - tau) }
is a dominant strategy. Everything downstream is computed by simulation with
100 common-random-number runs (seeds 0-99), replacing the 10-run / 30-run
Monte Carlo of the v2.x suite (b6_externality in revision_suite.py).

Q6a  closed form vs Monte Carlo for the tier-1 private cost
Q6b  equilibrium: tier 1 at u_priv, tiers 2-4 by tier-symmetric iterated
     best response on the 0.02 grid (100 runs per evaluation)
Q6c  planner: tier-symmetric iterated best response on the 0.02 grid, then a
     0.005-step refinement of the tier-1 reserve
Q6d  fragility (expected total failures) of no control / equilibrium / planner
Q6e  private payoff of a representative firm in each tier versus own reserve,
     others at the equilibrium profile (unimodality check)

Writes results_game.json.  ~10 min on an Apple M-series laptop.
"""
import json
import time
import numpy as np
from model import build_network
from decisions import simulate_v, PREC, EPSR, TAU

BASE = dict(agg="ces", rho=-10.0, damping="smooth", lam=2.0, L=3)
BETA_C, PSI, T = 6e-4, 1.0, 50
SHOCK_FRAC = 0.25
GRID = np.round(np.arange(0.0, 0.32, 0.02), 2)
FINE = np.round(np.arange(0.0, 0.1201, 0.005), 3)
SEEDS = range(100)

A, sup, tier, ranks = build_network()
n = len(tier)
tsizes = [int((tier == k).sum()) for k in range(4)]
k_hit = max(1, int(round(SHOCK_FRAC * tsizes[0])))
P_d = k_hit / tsizes[0]
S_T = float(sum((1 - PREC) ** t for t in range(T + 1)))
margin = EPSR[1] - (1 - TAU)
width = EPSR[1] - EPSR[0]


def cost(g):
    return g * g / (2 * BETA_C)


def phi1(u):
    """closed-form expected failure cost of a tier-1 firm holding reserve u"""
    return PSI * P_d * S_T * max(0.0, margin - u) / width


def private1(u):
    return phi1(u) + cost(u)


u_priv = min(BETA_C * PSI * P_d * S_T / width, margin)

CACHE = {}


def evaluate(u_by_tier, seeds=SEEDS):
    """per-tier mean own failures (node-periods per firm) and total failures,
    averaged over seeds, at a tier-symmetric profile"""
    key = tuple(float(x) for x in u_by_tier)
    if key in CACHE:
        return CACHE[key]
    u = np.array([u_by_tier[tier[j]] for j in range(n)])
    per = np.zeros(4); tot = 0.0
    for s in seeds:
        F = simulate_v(sup, tier, run_seed=s, u=u, T=T, **BASE)["F"]
        tot += F.sum()
        for k in range(4):
            per[k] += F[:, tier == k].sum() / tsizes[k]
    out = (per / len(seeds), tot / len(seeds))
    CACHE[key] = out
    return out


def iterate_br(objective, tiers, ut0, label):
    ut = np.array(ut0, float)
    for sweep in range(6):
        changed = False
        for k in tiers:
            best = (np.inf, ut[k])
            for g in GRID:
                t2 = ut.copy(); t2[k] = g
                per, tot = evaluate(t2)
                o = objective(k, per, tot, t2)
                if o < best[0] - 1e-9:
                    best = (o, g)
            if best[1] != ut[k]:
                changed = True
            ut[k] = best[1]
        print(f"  {label}: sweep {sweep + 1}: u = {list(ut)}")
        if not changed:
            break
    return ut


t0 = time.time()
res = {"constants": dict(beta_c=BETA_C, psi=PSI, T=T, p_rec=PREC,
                         eps_range=list(EPSR), tau=TAU, P_d=P_d, S_T=S_T,
                         margin=margin, width=width, seeds=[0, 99])}

# ---- Q6a: closed form vs Monte Carlo --------------------------------------
print("=" * 72); print("Q6a: tier-1 private cost, closed form vs Monte Carlo")
print(f"  P_d = {P_d:.3f}, S_T = {S_T:.3f}, margin = {margin:.2f}, "
      f"width = {width:.2f}")
print(f"  u_priv (closed form) = {u_priv:.5f}")
mc = {}
for g in (0.0, 0.02, 0.04, u_priv):
    per, _ = evaluate([g, 0, 0, 0])
    mc[float(g)] = float(per[0])
    print(f"  u = {g:.3f}: Phi_1 exact {phi1(g):.3f}  MC (20 firms x 100 runs) "
          f"{per[0]:.3f}  private cost exact {private1(g):.3f}")
res["q6a"] = dict(u_priv=float(u_priv),
                  exact={str(g): [phi1(g), private1(g)] for g in
                         (0.0, 0.02, 0.04, float(u_priv))},
                  mc=mc,
                  grid_exact_argmin=float(GRID[np.argmin([private1(g) for g in GRID])]))

# ---- Q6b: equilibrium ------------------------------------------------------
print("=" * 72); print("Q6b: equilibrium (tier 1 at u_priv; tiers 2-4 by BR)")
u_eq = iterate_br(lambda k, per, tot, t2: PSI * per[k] + cost(t2[k]),
                  tiers=(1, 2, 3), ut0=[u_priv, 0, 0, 0], label="eq")
per_eq, F_eq = evaluate(u_eq)
res["q6b"] = dict(u_eq=[float(x) for x in u_eq], F_eq=float(F_eq),
                  per_eq=[float(x) for x in per_eq])
print(f"  u_eq = {list(u_eq)} -> F_eq = {F_eq:.1f}")

# ---- Q6c: planner ----------------------------------------------------------
print("=" * 72); print("Q6c: planner")
def social(k, per, tot, t2):
    return PSI * tot + sum(tsizes[j] * cost(t2[j]) for j in range(4))
u_sp = iterate_br(social, tiers=(0, 1, 2, 3), ut0=[0, 0, 0, 0], label="planner")
curve = []
for g in FINE:
    t2 = u_sp.copy(); t2[0] = g
    per, tot = evaluate(t2)
    curve.append((float(g), float(tot), float(social(0, per, tot, t2))))
g_fine = min(curve, key=lambda r: r[2])[0]
u_sp_fine = u_sp.copy(); u_sp_fine[0] = g_fine
per_sp, F_sp = evaluate(u_sp_fine)
res["q6c"] = dict(u_sp_grid=[float(x) for x in u_sp],
                  u_sp=[float(x) for x in u_sp_fine], F_sp=float(F_sp),
                  planner_curve=curve)
print(f"  planner grid u = {list(u_sp)}; refined tier-1 reserve = {g_fine}"
      f" -> F_sp = {F_sp:.1f}")

# ---- Q6d: fragility comparison --------------------------------------------
print("=" * 72); print("Q6d: fragility")
_, F_0 = evaluate([0, 0, 0, 0])
soc_eq = PSI * F_eq + sum(tsizes[j] * cost(u_eq[j]) for j in range(4))
soc_sp = PSI * F_sp + sum(tsizes[j] * cost(u_sp_fine[j]) for j in range(4))
print(f"  none {F_0:.1f} | equilibrium {F_eq:.1f} | planner {F_sp:.1f} | "
      f"ratio eq/planner {F_eq / F_sp:.2f} | social cost eq {soc_eq:.1f} "
      f"vs planner {soc_sp:.1f}")
res["q6d"] = dict(F_0=float(F_0), F_eq=float(F_eq), F_sp=float(F_sp),
                  ratio=float(F_eq / F_sp), social_eq=float(soc_eq),
                  social_sp=float(soc_sp))

# ---- Q6e: unimodality of private payoffs ----------------------------------
print("=" * 72); print("Q6e: private payoff of a representative firm per tier")
uprof = np.array([u_eq[tier[j]] for j in range(n)])
q6e = {}
for k in range(4):
    j0 = int(np.nonzero(tier == k)[0][0])
    pay = []
    for g in GRID:
        u2 = uprof.copy(); u2[j0] = g
        own = np.mean([simulate_v(sup, tier, run_seed=s, u=u2, T=T,
                                  **BASE)["F"][:, j0].sum() for s in SEEDS])
        pay.append(float(-PSI * own - cost(g)))
    d = np.diff(pay)
    sc = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-9])) != 0))
    q6e[f"tier{k + 1}"] = dict(node=j0, grid=[float(g) for g in GRID],
                               payoff=pay, sign_changes=sc,
                               argmax=float(GRID[int(np.argmax(pay))]))
    print(f"  tier {k + 1} (node {j0}): argmax {GRID[int(np.argmax(pay))]}, "
          f"sign changes {sc}")
q6e["tier1_exact"] = dict(grid=[float(g) for g in GRID],
                          payoff=[-private1(g) for g in GRID])
res["q6e"] = q6e
res["runtime_min"] = (time.time() - t0) / 60
json.dump(res, open("results_game.json", "w"), indent=1)
print(f"done in {res['runtime_min']:.1f} min -> results_game.json")
