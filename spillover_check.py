import numpy as np, json
from model import build_network
from theory_checks import simulate_m
A,sup,tier,_=build_network(42); n=len(tier)
out={}
for j in range(0,20):
    d=[]
    for s in range(100):
        base=simulate_m(sup,tier,run_seed=s,T=50)
        if j not in base["hit"]: d.append(0.0); continue
        u=np.zeros(n); u[j]=0.008
        F=simulate_m(sup,tier,run_seed=s,T=50,u=u)["F"]
        m=np.ones(n,bool); m[j]=False
        d.append(float(F[:,m].sum()-base["F"][:,m].sum()))
    d=np.array(d); out[j]=dict(mean=float(d.mean()),n_neg=int((d<0).sum()),n_pos=int((d>0).sum()))
tot=dict(mean_over_sources=float(np.mean([v["mean"] for v in out.values()])),runs_others_fail_less=int(sum(v["n_neg"] for v in out.values())),runs_others_fail_more=int(sum(v["n_pos"] for v in out.values())),sources_with_positive_mean=int(sum(v["mean"]>0 for v in out.values())))
print(tot); json.dump(dict(per_source=out,summary=tot),open("results_spillover_small.json","w"),indent=1)
