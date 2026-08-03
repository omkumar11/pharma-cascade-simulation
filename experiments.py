"""
experiments.py — all Section 6 experiments. Run: python3 experiments.py
Regenerates results_e124.json (all tables) and stashes arrays for figures.
Then run figures.py. ~5 minutes total.
"""
import numpy as np, json
from scipy.stats import spearmanr
from model import build_network, simulate, metrics, centrality

A, sup, tier, ranks = build_network()
n = len(tier); out = {}

# ---------- Baseline ----------
Fs = [simulate(A, sup, tier, run_seed=s) for s in range(30)]
ms = [metrics(F) for F in Fs]
out["baseline"] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                   for k in ("total","peak","spread","recovery")}
np.save("F_baseline_runs.npy", np.array(Fs, dtype=bool))

# ---------- E1: threshold sweeps ----------
E1 = {"delta": {}, "alpha": {}}
for d in (0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50):
    r=[metrics(simulate(A,sup,tier,delta=d,run_seed=s))["spread"] for s in range(30)]
    E1["delta"][d]=(float(np.mean(r)),float(np.std(r)))
for a in (1.05,1.1,1.2,1.35,1.5):
    r=[metrics(simulate(A,sup,tier,alpha=a,run_seed=s))["spread"] for s in range(30)]
    E1["alpha"][a]=(float(np.mean(r)),float(np.std(r)))
out["E1_threshold"]=E1
E1b={"eps":{}, "h0":{}}
for e in (0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.8):
    r=[metrics(simulate(A,sup,tier,eps=e,run_seed=s))["spread"] for s in range(30)]
    E1b["eps"][e]=(float(np.mean(r)),float(np.std(r)))
for h in (0.05,0.1,0.2,0.3,0.4,0.5,0.6):
    r=[metrics(simulate(A,sup,tier,h0=h,run_seed=s))["spread"] for s in range(30)]
    E1b["h0"][h]=(float(np.mean(r)),float(np.std(r)))
out["E1b_threshold"]=E1b
fine={}; seed_sz={}
for e in (0.70,0.705,0.71,0.72,0.75):
    r=[metrics(simulate(A,sup,tier,eps=e,run_seed=s))["spread"] for s in range(30)]
    fine[e]=(float(np.mean(r)),float(np.std(r)))
for sf in (0.05,0.10,0.25):
    r=[metrics(simulate(A,sup,tier,shock_frac=sf,run_seed=s))["spread"] for s in range(30)]
    seed_sz[sf]=(float(np.mean(r)),float(np.std(r)))
out["E1c_fine"]={"eps_fine":fine,"seed_size":seed_sz}

# ---------- E2: delay ----------
E2={}
for L in (1,2,3,4):
    ms=[metrics(simulate(A,sup,tier,L=L,run_seed=s)) for s in range(30)]
    E2[L]={k:(float(np.mean([m[k] for m in ms])),float(np.std([m[k] for m in ms])))
           for k in ("total","peak","recovery")}
out["E2_delay"]=E2

# ---------- E3: lambda sweep + curvature under two embeddings ----------
def embeds(A,tier,ranks):
    z_tr=np.column_stack([tier/3.0,ranks])
    As=(A|A.T).astype(float); Lap=np.diag(As.sum(1))-As
    _,vecs=np.linalg.eigh(Lap); z=vecs[:,1:3]
    return {"tier_rank":z_tr,"spectral":(z-z.min(0))/np.ptp(z,0)}
def curv_max(F,z,sigma=0.15,G=60,pad=0.45):
    gx=np.linspace(-pad,1+pad,G); cell=(gx[1]-gx[0])**2
    GX,GY=np.meshgrid(gx,gx); c=1/(2*np.pi*sigma**2); best=0.0
    for t in range(F.shape[0]):
        idx=np.nonzero(F[t])[0]
        if not len(idx): continue
        ux=GX[...,None]-z[idx,0]; uy=GY[...,None]-z[idx,1]
        K=c*np.exp(-(ux**2+uy**2)/(2*sigma**2))
        H11=(K*(ux*ux/sigma**4-1/sigma**2)).sum(-1)
        H22=(K*(uy*uy/sigma**4-1/sigma**2)).sum(-1)
        H12=(K*(ux*uy/sigma**4)).sum(-1)
        best=max(best,float(((H11**2+2*H12**2+H22**2).sum())*cell))
    return best
Z=embeds(A,tier,ranks); LAMS=[0,1,2,3,5]
E3={"lams":LAMS,"F":[],"C":{e:[] for e in Z},"spearman":{}}
for lam in LAMS:
    Fs=[simulate(A,sup,tier,lam=lam,run_seed=s) for s in range(30)]
    tot=[int(F.sum()) for F in Fs]
    E3["F"].append((float(np.mean(tot)),float(np.std(tot))))
    for e,z in Z.items():
        cs=[curv_max(F,z) for F in Fs]
        E3["C"][e].append((float(np.mean(cs)),float(np.std(cs))))
for e in Z:
    mu=[c for c,_ in E3["C"][e]]; E3["spearman"][e]=float(spearmanr(LAMS,mu)[0])
out["E3_lambda"]=E3

# ---------- E4: aggregator comparison ----------
E4={}
for agg,rho,name in [("leontief",None,"Leontief (req-min)"),("ces",-10.0,"CES rho=-10"),
                     ("ces",0.01,"CES rho~0"),("ces",0.5,"CES rho=0.5"),("ces",1.0,"Linear rho=1")]:
    ms=[metrics(simulate(A,sup,tier,agg=agg,rho=rho,run_seed=s)) for s in range(30)]
    E4[name]={k:(float(np.mean([m[k] for m in ms])),float(np.std([m[k] for m in ms])))
              for k in ("total","peak","spread","recovery")}
out["E4_ces"]=E4

# ---------- E5: targeted vs uniform (B=4) ----------
C=centrality(A); theta=C/C.sum(); B=4.0
E5={}; series={}
for name,u in (("baseline",None),("uniform",np.full(n,B/n)),("targeted",B*theta)):
    Fs=[simulate(A,sup,tier,u=u,run_seed=s) for s in range(30)]
    ms=[metrics(F) for F in Fs]
    E5[name]={k:(float(np.mean([m[k] for m in ms])),float(np.std([m[k] for m in ms])))
              for k in ("total","peak","spread","recovery")}
    series[name]=np.array([F.sum(1) for F in Fs])
out["E5_target"]=E5
np.save("series_e5.npy",np.array([series["baseline"],series["targeted"],series["uniform"]]))
np.save("theta.npy",theta)

# ---------- E6: externality (tier-symmetric BR vs planner) ----------
BETA_C=0.0006; PSI=1.0; GRID=np.round(np.arange(0.0,0.32,0.02),2); NR=10
tsizes=[int((tier==k).sum()) for k in range(4)]
def eval_alloc(ut,runs=NR):
    u=np.array([ut[tier[j]] for j in range(n)]); per=np.zeros(4); tot=0.0
    for s in range(runs):
        F=simulate(A,sup,tier,u=u,run_seed=s); tot+=F.sum()
        for k in range(4): per[k]+=F[:,tier==k].sum()/tsizes[k]
    return per/runs,tot/runs
def cost(g): return g*g/(2*BETA_C)
ut=np.zeros(4)
for it in range(4):
    ch=False
    for k in range(4):
        best=(1e18,ut[k])
        for g in GRID:
            t2=ut.copy(); t2[k]=g; pt,_=eval_alloc(t2)
            o=PSI*pt[k]+cost(g)
            if o<best[0]-1e-9: best=(o,g)
        if best[1]!=ut[k]: ch=True
        ut[k]=best[1]
    if not ch: break
u_mpe=ut.copy(); _,F_mpe=eval_alloc(u_mpe,runs=30)
ut=np.zeros(4)
for it in range(4):
    ch=False
    for k in range(4):
        best=(1e18,ut[k])
        for g in GRID:
            t2=ut.copy(); t2[k]=g; _,tot=eval_alloc(t2)
            o=PSI*tot+sum(tsizes[j]*cost(t2[j]) for j in range(4))
            if o<best[0]-1e-9: best=(o,g)
        if best[1]!=ut[k]: ch=True
        ut[k]=best[1]
    if not ch: break
u_sp=ut.copy(); _,F_sp=eval_alloc(u_sp,runs=30)
_,F_none=eval_alloc(np.zeros(4),runs=30)
out["E6_externality"]=dict(beta_c=BETA_C,psi=PSI,u_mpe=list(map(float,u_mpe)),
    u_planner=list(map(float,u_sp)),F_mpe=float(F_mpe),F_planner=float(F_sp),F_none=float(F_none))

json.dump(out,open("results_e124.json","w"),indent=1)
print("experiments complete -> results_e124.json")
