import sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path("/home/user/pickleball"); SCR=Path("/tmp/claude-0/-home-user-pickleball/a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
sys.path.insert(0,str(ROOT/"web")); sys.path.insert(0,str(ROOT/"model/weather_review")); sys.argv=["x",str(SCR)]
import rally_favorites_allmatches as P1
cells,_=P1.build_cells()
def irls(X,y,n):
    b=np.zeros(X.shape[1])
    for _ in range(60):
        z=np.clip(X@b,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
        s=np.linalg.solve((X*W[:,None]).T@X, X.T@(y-n*p)); b=b+s
        if np.max(np.abs(s))<1e-12: break
    return b
def se_(X,y,n,b,cl_):
    z=np.clip(X@b,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
    B=np.linalg.inv((X*W[:,None]).T@X); s=X*(y-n*p)[:,None]
    cl=defaultdict(list)
    for i,c in enumerate(cl_): cl[c].append(i)
    M=sum(np.outer(s[i].sum(0),s[i].sum(0)) for i in cl.values()); G=len(cl)
    return np.sqrt(np.diag(B@(M*G/(G-1))@B)),G
def run(sub,lab):
    if len(sub)<50: print(f"  {lab}: too thin"); return
    X=np.array([[1.,c["adv"],c["w"],c["adv"]*c["w"]] for c in sub])
    y=np.array([c["wins"] for c in sub],float); n=np.array([c["n"] for c in sub],float)
    b=irls(X,y,n); se,G=se_(X,y,n,b,[c["ev"] for c in sub])
    print(f"  {lab:<44} n={int(n.sum()):>7} ev={G:>3} b={b[1]:+.4f} d={b[3]:+.4f} se={se[3]:.4f} [{b[3]-1.96*se[3]:+.4f},{b[3]+1.96*se[3]:+.4f}]")
print("=== label-confidence split of the OUTDOOR headline ===")
O=[c for c in cells if c["setting"]=="outdoor"]
run(O,"all outdoor")
run([c for c in O if c["conf"] in ("high","medium")],"web-verified labels (high+medium)")
run([c for c in O if c["conf"]=="high"],"high confidence only")
run([c for c in O if c["conf"]=="heuristic"],"UNAUDITED heuristic-outdoor only")
run([c for c in O if c["conf"]=="low"],"low confidence only")
print("\n=== outdoor headline dropping the top bin / restricting range ===")
run([c for c in O if c["wind"]<16],"outdoor, wind < 16 mph")
run([c for c in O if c["wind"]<12],"outdoor, wind < 12 mph")
print("\n=== does the interaction survive event fixed effects? (within-event wind variation) ===")
evs=sorted({c["ev"] for c in O}); idx={e:i for i,e in enumerate(evs)}
X=[];y=[];n=[]
for c in O:
    row=[0.0]*len(evs); row[idx[c["ev"]]]=1.0
    X.append(row+[c["adv"],c["w"],c["adv"]*c["w"]]); y.append(c["wins"]); n.append(c["n"])
X=np.array(X);y=np.array(y,float);n=np.array(n,float)
b=irls(X,y,n); se,G=se_(X,y,n,b,[c["ev"] for c in O])
print(f"  event FE:  b={b[-3]:+.4f}  c={b[-2]:+.4f}  d={b[-1]:+.4f} se={se[-1]:.4f} [{b[-1]-1.96*se[-1]:+.4f},{b[-1]+1.96*se[-1]:+.4f}]")
