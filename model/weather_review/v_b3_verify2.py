"""ADVERSARIAL VERIFICATION of B3 part 2 — the binned dose-response chase.

Independent routes:
  * per-bin b re-fit with numpy IRLS + analytic cluster-robust SE
  * T1 trend with its OWN bootstrap CI for the bins-1-4 variant (the tester
    reported that estimate with the CONTINUOUS interaction's CI)
  * era (year) composition control — a channel the tester never tested, and
    one that is NOT post-treatment the way match length is
  * length-neutral cut: refit b using only rallies at leader score 0-5
    (every game supplies these; total match length barely enters)
  * DiD with 5 seeds and with a STRATIFIED (per-arm) cluster bootstrap
"""
from __future__ import annotations
import sys, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/user/pickleball")
SCR = Path("/tmp/claude-0/-home-user-pickleball/a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
sys.path.insert(0, str(ROOT / "web")); sys.path.insert(0, str(ROOT / "model/weather_review"))
sys.argv = ["x", str(SCR)]
import rally_favorites_allmatches as P1

cells, _ = P1.build_cells()
BINS = [(0,4),(4,8),(8,12),(12,16),(16,40)]
def binof(w):
    for i,(lo,hi) in enumerate(BINS):
        if lo <= w < hi: return i
    return None
for c in cells: c["bin"] = binof(c["wind"])

def irls(X, y, n, iters=60, b0=None):
    beta = np.zeros(X.shape[1]) if b0 is None else np.array(b0,float)
    for _ in range(iters):
        z = np.clip(X@beta,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
        H=(X*W[:,None]).T@X
        try: step=np.linalg.solve(H, X.T@(y-n*p))
        except np.linalg.LinAlgError: return None
        beta=beta+step
        if np.max(np.abs(step))<1e-11: break
    return beta

def sw(X,y,n,beta,clust):
    z=np.clip(X@beta,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
    B=np.linalg.inv((X*W[:,None]).T@X); s=X*(y-n*p)[:,None]
    cl=defaultdict(list)
    for i,c in enumerate(clust): cl[c].append(i)
    M=sum(np.outer(s[ix].sum(0),s[ix].sum(0)) for ix in cl.values())
    G=len(cl); return B@(M*G/(G-1))@B, G

def fitb(sub, key="wins", nkey="n"):
    if len(sub)<6: return None
    X=np.array([[1.0,c["adv"]] for c in sub]); y=np.array([c[key] for c in sub],float)
    n=np.array([c[nkey] for c in sub],float)
    if n.sum()<200: return None
    return irls(X,y,n)

def fitb_se(sub, key="wins", nkey="n"):
    X=np.array([[1.0,c["adv"]] for c in sub]); y=np.array([c[key] for c in sub],float)
    n=np.array([c[nkey] for c in sub],float); b=irls(X,y,n)
    V,G=sw(X,y,n,b,[c["ev"] for c in sub]); return b[1], math.sqrt(V[1,1]), G, int(n.sum())

print("=== A. per-bin b re-fit (numpy IRLS) + analytic cluster-robust SE ===")
hat={}
for setting in ("outdoor","indoor"):
    for i,(lo,hi) in enumerate(BINS):
        sub=[c for c in cells if c["setting"]==setting and c["bin"]==i]
        b,se,G,N=fitb_se(sub); hat[(setting,i)]=b
        mw=sum(c["n"]*c["wind"] for c in sub)/N
        print(f"  {setting:>7} {lo:>2}-{hi:<2}  meanW={mw:4.1f}  b={b:+.4f} se={se:.4f} "
              f"[{b-1.96*se:+.4f},{b+1.96*se:+.4f}]  N={N} ev={G}")
    sub=[c for c in cells if c["setting"]==setting and c["bin"] is not None and c["bin"]<=2]
    b,se,G,N=fitb_se(sub); hat[(setting,"rest")]=b
    print(f"  {setting:>7} 0-12 pool b={b:+.4f} se={se:.4f} N={N}")

# ---------------- B. ERA composition -----------------------------------
print("\n=== B. is b era-dependent?  (v2 = CURRENT form applied retroactively) ===")
for setting in ("outdoor","indoor"):
    for yr in ("2024","2025","2026"):
        sub=[c for c in cells if c["setting"]==setting and c["date"][:4]==yr]
        if len(sub)<50: continue
        b,se,G,N=fitb_se(sub)
        print(f"  {setting:>7} {yr}: b={b:+.4f} se={se:.4f}  N={N} ev={G}")

print("\n=== C. top-vs-calm contrast WITHIN each year (era-controlled) ===")
for setting in ("outdoor","indoor"):
    for yr in ("2024","2025","2026"):
        t=[c for c in cells if c["setting"]==setting and c["date"][:4]==yr and c["bin"]==4]
        k=[c for c in cells if c["setting"]==setting and c["date"][:4]==yr and c["bin"] is not None and c["bin"]<=2]
        ft,fk=fitb(t),fitb(k)
        if ft is None or fk is None:
            print(f"  {setting:>7} {yr}: too thin"); continue
        print(f"  {setting:>7} {yr}: top b={ft[1]:+.4f} (N={sum(c['n'] for c in t)})  "
              f"calm b={fk[1]:+.4f} (N={sum(c['n'] for c in k)})  diff={ft[1]-fk[1]:+.4f}")

print("\n=== C2. one model: adv x top with YEAR x adv fixed effects (outdoor) ===")
for setting in ("outdoor","indoor"):
    sub=[c for c in cells if c["setting"]==setting and c["bin"] is not None and (c["bin"]==4 or c["bin"]<=2)]
    yrs=sorted({c["date"][:4] for c in sub})[1:]   # drop base year
    X=[];y=[];n=[]
    for c in sub:
        top = 1.0 if c["bin"]==4 else 0.0
        row=[1.0, c["adv"], top, c["adv"]*top]
        for yy in yrs:
            d = 1.0 if c["date"][:4]==yy else 0.0
            row += [d, d*c["adv"]]
        X.append(row); y.append(c["wins"]); n.append(c["n"])
    X=np.array(X); y=np.array(y,float); n=np.array(n,float)
    b=irls(X,y,n); V,G=sw(X,y,n,b,[c["ev"] for c in sub]); se=np.sqrt(np.diag(V))
    print(f"  [{setting}] adv x top (year x adv controlled) = {b[3]:+.4f} se={se[3]:.4f} "
          f"[{b[3]-1.96*se[3]:+.4f},{b[3]+1.96*se[3]:+.4f}]   (uncontrolled = "
          f"{hat[(setting,4)]-hat[(setting,'rest')]:+.4f})")

# ---------------- D. length-neutral cut: early-score rallies only ------
print("\n=== D. per-bin b using ONLY leader-score 0-5 rallies (length-neutral) ===")
for c in cells:
    c["en"]=sum(c["buck"][i][0] for i in range(6)); c["ew"]=sum(c["buck"][i][1] for i in range(6))
for setting in ("outdoor","indoor"):
    for i,(lo,hi) in enumerate(BINS):
        sub=[c for c in cells if c["setting"]==setting and c["bin"]==i and c["en"]>=4]
        b,se,G,N=fitb_se(sub,key="ew",nkey="en")
        print(f"  {setting:>7} {lo:>2}-{hi:<2}: b_early={b:+.4f} se={se:.4f} N={N}")
    sub=[c for c in cells if c["setting"]==setting and c["bin"] is not None and c["bin"]<=2 and c["en"]>=4]
    b,se,G,N=fitb_se(sub,key="ew",nkey="en")
    print(f"  {setting:>7} 0-12 pool: b_early={b:+.4f} se={se:.4f} N={N}")
