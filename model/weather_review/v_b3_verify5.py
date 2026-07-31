"""ADVERSARIAL VERIFICATION of B3 — clean length standardization, mechanism
panel with analytic cluster-robust SEs, and the race-DP translation check.
"""
from __future__ import annotations
import sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/home/user/pickleball")
SCR=Path("/tmp/claude-0/-home-user-pickleball/a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
sys.path.insert(0,str(ROOT/"web")); sys.path.insert(0,str(ROOT/"model/weather_review"))
sys.argv=["x",str(SCR)]
import rally_favorites_allmatches as P1
from sitelib.race import game_win_prob

cells,_=P1.build_cells()

def irls(X,y,n,iters=60):
    beta=np.zeros(X.shape[1])
    for _ in range(iters):
        z=np.clip(X@beta,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
        H=(X*W[:,None]).T@X
        try: step=np.linalg.solve(H,X.T@(y-n*p))
        except np.linalg.LinAlgError: return None
        beta=beta+step
        if np.max(np.abs(step))<1e-12: break
    return beta
def swse(X,y,n,beta,cl_):
    z=np.clip(X@beta,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
    B=np.linalg.inv((X*W[:,None]).T@X); s=X*(y-n*p)[:,None]
    cl=defaultdict(list)
    for i,c in enumerate(cl_): cl[c].append(i)
    M=sum(np.outer(s[ix].sum(0),s[ix].sum(0)) for ix in cl.values()); G=len(cl)
    return np.sqrt(np.diag(B@(M*G/(G-1))@B)), G

def bfit(sub):
    X=np.array([[1.0,c["adv"]] for c in sub]); y=np.array([c["wins"] for c in sub],float)
    n=np.array([c["n"] for c in sub],float)
    b=irls(X,y,n)
    if b is None: return None
    se,G=swse(X,y,n,b,[c["ev"] for c in sub])
    return b[1], se[1], int(n.sum())

edges=[0,20,28,34,40,46,52,60,10**9]
def lb(n):
    for i in range(len(edges)-1):
        if edges[i]<=n<edges[i+1]: return i
    return len(edges)-2

print("=== LENGTH standardization, apples-to-apples (stratum b's on both sides) ===")
for setting in ("outdoor","indoor"):
    A=[c for c in cells if c["setting"]==setting]
    top=[c for c in A if c["wind"]>=16]; calm=[c for c in A if c["wind"]<12]
    wt_t=defaultdict(int); wt_c=defaultdict(int)
    for c in top: wt_t[lb(c["n"])]+=c["n"]
    for c in calm: wt_c[lb(c["n"])]+=c["n"]
    Tt=sum(wt_t.values()); Tc=sum(wt_c.values())
    rows=[]
    for k in sorted(set(wt_t)|set(wt_c)):
        st=[c for c in top if lb(c["n"])==k]; sc=[c for c in calm if lb(c["n"])==k]
        ft=bfit(st) if len(st)>=6 and sum(c["n"] for c in st)>200 else None
        fc=bfit(sc) if len(sc)>=6 else None
        rows.append((k, wt_t[k]/Tt, wt_c[k]/Tc, ft, fc))
    # marginal (pooled) fits
    pt=bfit(top); pc=bfit(calm)
    print(f"--- {setting}:  pooled top {pt[0]:+.4f}  pooled calm {pc[0]:+.4f}  raw diff {pt[0]-pc[0]:+.4f}")
    print(f"   {'len bin':>10} {'w_top':>6} {'w_calm':>6} {'b_top':>18} {'b_calm':>18} {'diff':>8}")
    num=den=0.0; varsum=0.0
    for k,wt,wc,ft,fc in rows:
        hi = edges[k+1] if k+1 < len(edges)-1 else 9999
        lab=f"{edges[k]}-{'+' if hi==9999 else hi}"
        st = f"{ft[0]:+.3f}+-{ft[1]:.3f}" if ft else "     --      "
        sc = f"{fc[0]:+.3f}+-{fc[1]:.3f}" if fc else "     --      "
        d  = f"{ft[0]-fc[0]:+.3f}" if (ft and fc) else "   --"
        print(f"   {lab:>10} {wt:>6.1%} {wc:>6.1%} {st:>18} {sc:>18} {d:>8}")
        if ft and fc:
            num += wt*(ft[0]-fc[0]); den += wt
            varsum += (wt**2)*(ft[1]**2+fc[1]**2)
    std = num/den
    print(f"   ==> LENGTH-STANDARDIZED top-minus-calm (weights = top bin's mix) "
          f"= {std:+.4f} +- {math.sqrt(varsum)/den:.4f}   [raw {pt[0]-pc[0]:+.4f}]")
    # pure composition term: same stratum b's (calm), two different mixes
    comp = sum(wt*fc[0] for k,wt,wc,ft,fc in rows if fc) / sum(wt for k,wt,wc,ft,fc in rows if fc) \
         - sum(wc*fc[0] for k,wt,wc,ft,fc in rows if fc) / sum(wc for k,wt,wc,ft,fc in rows if fc)
    print(f"   pure composition term (calm stratum b's, top mix minus calm mix) = {comp:+.4f}")

print("\n=== MECHANISM panel, analytic cluster-robust SEs (tester used bootstrap) ===")
def one(sub, ykey, nkey, label):
    sub=[c for c in sub if c[nkey]>=4]
    X=np.array([[1.0,c["w"]] for c in sub]); y=np.array([c[ykey] for c in sub],float)
    n=np.array([c[nkey] for c in sub],float)
    b=irls(X,y,n); se,G=swse(X,y,n,b,[c["ev"] for c in sub])
    p0=1/(1+math.exp(-b[0])); sc=p0*(1-p0)*100
    print(f"  {label:<44} logit={b[1]:+.4f} se={se[1]:.4f} "
          f"[{b[1]-1.96*se[1]:+.4f},{b[1]+1.96*se[1]:+.4f}]  "
          f"= {b[1]*sc:+.2f} pp/+10mph  (base {p0:.1%}, n={int(n.sum())})")
for setting in ("outdoor","indoor"):
    A=[c for c in cells if c["setting"]==setting]
    for c in A: c["_all"]=c["n"]
    one(A,"wins","n",f"[{setting}] serve-win rate (k) ~ wind")
    one([dict(c) for c in A],"n2","n",f"[{setting}] P(2nd-server rally) ~ wind")
    one([dict(c) for c in A],"w2","n2",f"[{setting}] P(win | 2nd server) ~ wind")
    one([dict(c) for c in A],"w1","n1",f"[{setting}] P(win | 1st server) ~ wind")

print("\n=== race-DP translation check (game to 11) ===")
def sig(x): return 1/(1+math.exp(-x))
def eta_for(p):
    lo,hi=0.0,3.0
    for _ in range(200):
        m=(lo+hi)/2
        if game_win_prob(m,11)<p: lo=m
        else: hi=m
    return (lo+hi)/2
b=0.5016
for lab,d in (("point -0.0191",-0.0191),("CI floor -0.0481",-0.0481),("CI ceil +0.0134",0.0134),
              ("T1 floor -0.0568",-0.0568),("MDE -0.044",-0.044)):
    for wmph in (20.0,):
        mult=(b+d*wmph/10.0)/b
        s=f"  {lab:<20} 20mph mult={mult:.3f}: "
        for p0 in (0.65,0.75,0.90):
            e=eta_for(p0); p1=game_win_prob(e*mult,11)
            s+=f"{p0:.0%}->{p1:.1%} ({100*(p1-p0):+.1f}pp)  "
        print(s)
