"""ADVERSARIAL VERIFICATION of B3 part 3 — T1/T4 re-derived, seed sweep,
length standardization done properly, and does wind actually lengthen matches?
"""
from __future__ import annotations
import sys, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/user/pickleball")
SCR = Path("/tmp/claude-0/-home-user-pickleball/a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
sys.path.insert(0, str(ROOT/"web")); sys.path.insert(0, str(ROOT/"model/weather_review"))
sys.argv=["x",str(SCR)]
import rally_favorites_allmatches as P1

cells,_ = P1.build_cells()
BINS=[(0,4),(4,8),(8,12),(12,16),(16,40)]
def binof(w):
    for i,(lo,hi) in enumerate(BINS):
        if lo<=w<hi: return i
    return None
for c in cells: c["bin"]=binof(c["wind"])

def fit2(rows, b0=None, iters=30):
    a,b = (b0 if b0 else (0.0,0.0))
    for _ in range(iters):
        g0=g1=h00=h01=h11=0.0
        for wins,n,x in rows:
            z=max(-30.0,min(30.0,a+b*x)); p=1.0/(1.0+math.exp(-z))
            r=wins-n*p; w=n*p*(1-p)
            g0+=r; g1+=r*x; h00+=w; h01+=w*x; h11+=w*x*x
        det=h00*h11-h01*h01
        if abs(det)<1e-12: return None
        da=(h11*g0-h01*g1)/det; db=(h00*g1-h01*g0)/det
        a+=da; b+=db
        if abs(da)<1e-11 and abs(db)<1e-11: break
    return a,b

def wls(xs,ys,ws):
    sw=sum(ws); mx=sum(w*x for w,x in zip(ws,xs))/sw; my=sum(w*y for w,y in zip(ws,ys))/sw
    num=sum(w*(x-mx)*(y-my) for w,x,y in zip(ws,xs,ys)); den=sum(w*(x-mx)**2 for w,x in zip(ws,xs))
    return num/den if den else float("nan")

def pct(v,q):
    v=sorted(v); return v[int(q*(len(v)-1))]

# ---- build per-event row store ------------------------------------------
byev=defaultdict(lambda: defaultdict(list))
rows=defaultdict(list); xbar={}
for setting in ("outdoor","indoor"):
    for i in range(5):
        sub=[c for c in cells if c["setting"]==setting and c["bin"]==i]
        key=(setting,i)
        rows[key]=[(c["wins"],c["n"],c["adv"]) for c in sub]
        nn=sum(c["n"] for c in sub); xbar[key]=sum(c["n"]*c["wind"] for c in sub)/nn
        for c in sub: byev[c["ev"]][key].append((c["wins"],c["n"],c["adv"]))
    rows[(setting,"rest")]=sum((rows[(setting,i)] for i in range(3)),[])
    for ev in list(byev):
        r=sum((byev[ev].get((setting,i),[]) for i in range(3)),[])
        if r: byev[ev][(setting,"rest")]=r
hat={k:fit2(v) for k,v in rows.items()}
events=sorted(byev)
W={ (s,i): sum(n for _,n,_ in rows[(s,i)]) for s in ("outdoor","indoor") for i in range(5)}

def run_boot(seed, R=1000, stratified=False):
    rng=random.Random(seed)
    ev_out=[e for e in events if (("outdoor",0) in byev[e] or ("outdoor","rest") in byev[e]
            or any(("outdoor",i) in byev[e] for i in range(5)))]
    ev_in =[e for e in events if any(("indoor",i) in byev[e] for i in range(5))]
    tr={"outdoor":[],"indoor":[]}; tr14={"outdoor":[],"indoor":[]}
    top={"outdoor":[],"indoor":[]}; did=[]; mono={"outdoor":0,"indoor":0}; nok=0
    for _ in range(R):
        if stratified:
            pick_o=[rng.choice(ev_out) for _ in ev_out]; pick_i=[rng.choice(ev_in) for _ in ev_in]
        else:
            pk=[rng.choice(events) for _ in events]; pick_o=pick_i=pk
        pool=defaultdict(list)
        for ev in pick_o:
            for key,r in byev[ev].items():
                if key[0]=="outdoor": pool[key].extend(r)
        for ev in pick_i:
            for key,r in byev[ev].items():
                if key[0]=="indoor": pool[key].extend(r)
        bs={}; ok=True
        for s in ("outdoor","indoor"):
            for i in list(range(5))+["rest"]:
                r=pool.get((s,i))
                if not r or len(r)<6: ok=False; break
                f=fit2(r,b0=hat[(s,i)],iters=6)
                if f is None: ok=False; break
                bs[(s,i)]=f[1]
            if not ok: break
        if not ok: continue
        nok+=1
        for s in ("outdoor","indoor"):
            ys=[bs[(s,i)] for i in range(5)]; xs=[xbar[(s,i)] for i in range(5)]; ws=[W[(s,i)] for i in range(5)]
            tr[s].append(wls(xs,ys,ws)*10.0)
            tr14[s].append(wls(xs[:4],ys[:4],ws[:4])*10.0)
            if all(ys[j]>ys[j+1] for j in range(4)): mono[s]+=1
            top[s].append(bs[(s,4)]-bs[(s,"rest")])
        did.append(top["outdoor"][-1]-top["indoor"][-1])
    return tr,tr14,top,did,mono,nok

print("=== T1 / T1b / T3 / T4 re-derived, seed sweep, joint vs stratified bootstrap ===")
for s in ("outdoor","indoor"):
    ys=[hat[(s,i)][1] for i in range(5)]; xs=[xbar[(s,i)] for i in range(5)]; ws=[W[(s,i)] for i in range(5)]
    print(f"  point: {s:>7} T1(5 bins)={wls(xs,ys,ws)*10:+.4f}  T1b(bins1-4)={wls(xs[:4],ys[:4],ws[:4])*10:+.4f}  "
          f"T3={hat[(s,4)][1]-hat[(s,'rest')][1]:+.4f}")
print(f"  point: DiD={(hat[('outdoor',4)][1]-hat[('outdoor','rest')][1])-(hat[('indoor',4)][1]-hat[('indoor','rest')][1]):+.4f}")
for seed in (20260731, 1, 777, 20260801):
    for strat in (False,True):
        tr,tr14,top,did,mono,nok=run_boot(seed,R=600,stratified=strat)
        lab = "strat" if strat else "joint"
        print(f"  seed={seed:<9} {lab}  nok={nok:3d}  "
              f"T1out=[{pct(tr['outdoor'],.025):+.4f},{pct(tr['outdoor'],.975):+.4f}] "
              f"T1bout=[{pct(tr14['outdoor'],.025):+.4f},{pct(tr14['outdoor'],.975):+.4f}] "
              f"T3out=[{pct(top['outdoor'],.025):+.4f},{pct(top['outdoor'],.975):+.4f}] "
              f"DiD=[{pct(did,.025):+.4f},{pct(did,.975):+.4f}] monoOut={mono['outdoor']/max(nok,1):.1%}")

# --------------- length standardization (proper, not stratum-conditioning) ---
print("\n=== LENGTH: does the calm reference reproduce the top-bin drop once "
      "standardized to the top bin's match-side length distribution? ===")
edges=[0,20,28,34,40,46,52,60,10**9]
def lenbin(n):
    for i in range(len(edges)-1):
        if edges[i]<=n<edges[i+1]: return i
    return len(edges)-2
for setting in ("outdoor","indoor"):
    A=[c for c in cells if c["setting"]==setting]
    top=[c for c in A if c["wind"]>=16]; calm=[c for c in A if c["wind"]<12]
    wt=defaultdict(int)
    for c in top: wt[lenbin(c["n"])]+=c["n"]
    tot=sum(wt.values())
    # stratum-specific b in the CALM arm, then re-weight by the TOP bin's mix
    num=0.0; cov=0
    parts=[]
    for k in sorted(wt):
        sub=[c for c in calm if lenbin(c["n"])==k]
        f=fit2([(c["wins"],c["n"],c["adv"]) for c in sub]) if len(sub)>=6 else None
        if f is None: continue
        num += (wt[k]/tot)*f[1]; cov += wt[k]
        parts.append((k,wt[k]/tot,f[1]))
    std_calm = num*tot/cov
    raw_calm = fit2([(c["wins"],c["n"],c["adv"]) for c in calm])[1]
    raw_top  = fit2([(c["wins"],c["n"],c["adv"]) for c in top])[1]
    # and the same standardization applied to the TOP arm (should be ~raw_top)
    print(f"  [{setting}] raw top {raw_top:+.4f}  raw calm {raw_calm:+.4f}  raw diff {raw_top-raw_calm:+.4f}")
    print(f"           calm standardized to TOP length mix = {std_calm:+.4f}  "
          f"=> length-adjusted diff {raw_top-std_calm:+.4f}")
    print("           calm stratum b by length bin: " + " ".join(f"{edges[k]}-{edges[k+1] if k+1<len(edges)-1 else '+'}:{b:+.3f}(w{w:.0%})" for k,w,b in parts))

print("\n=== does WIND lengthen matches?  (the post-treatment worry) ===")
for setting in ("outdoor","indoor"):
    A=[c for c in cells if c["setting"]==setting]
    xs=np.array([c["wind"]/10.0 for c in A]); ys=np.array([float(c["n"]) for c in A])
    ev=[c["ev"] for c in A]
    X=np.column_stack([np.ones(len(xs)),xs])
    beta=np.linalg.lstsq(X,ys,rcond=None)[0]
    r=ys-X@beta
    cl=defaultdict(list)
    for i,e in enumerate(ev): cl[e].append(i)
    B=np.linalg.inv(X.T@X); M=np.zeros((2,2))
    for ix in cl.values():
        u=(X[ix]*r[ix,None]).sum(0); M+=np.outer(u,u)
    G=len(cl); V=B@(M*G/(G-1))@B; se=math.sqrt(V[1,1])
    print(f"  [{setting}] match-side serve rallies per +10 mph = {beta[1]:+.3f} se={se:.3f} "
          f"[{beta[1]-1.96*se:+.3f},{beta[1]+1.96*se:+.3f}]  (mean length {ys.mean():.1f})")
