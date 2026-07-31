"""ADVERSARIAL VERIFICATION of B3 — decompose the published indoor d = -0.060.

The tester claims the published indoor -0.060 is "a decider-selection
artifact", demonstrated by getting -0.1013 in their went-the-distance
subsample.  But the published number differs from the tester's on TWO axes:
  (a) sample: published = deciding GAMES + all MLP games (game-level rows of
      data/decider_serve_splits.csv);  tester = ALL rallies of every match
      that went the distance (or was MLP).  Those are different samples.
  (b) labels: published = heuristic event_geo settings; tester = corrected
      data/venue_overrides.csv.
This runs the published spec on the published sample and flips ONE axis at a
time, so the -0.060 -> -0.013 move can be attributed.
"""
from __future__ import annotations
import sys, math, random, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/home/user/pickleball")
sys.path.insert(0,str(ROOT/"web"))
from sitelib.race import team_eta

def rd(p):
    with open(p) as f: return list(csv.DictReader(f))

v2={r["player_id"].lower():float(r["value_now_mean"]) for r in rd(ROOT/"data/v2_players.csv")}
geo={r["event_id"]:r["setting"] for r in rd(ROOT/"data/event_geo.csv")}
ovr={r["event_id"]:r["setting"] for r in rd(ROOT/"data/venue_overrides.csv")}
hourly={}
for r in rd(ROOT/"data/event_weather_hourly.csv"):
    try: hourly[(r["event_id"],r["local_time"][:13])]=float(r["windspeed_10m"])
    except (TypeError,ValueError): pass
hour={}
for r in rd(ROOT/"data/match_times.csv"):
    ts=r["start_local"] or r["planned_start_local"]
    if ts: hour[r["match_id"]]=ts[:13]
match={}
for g in rd(ROOT/"data/games.csv"):
    if g["is_dreambreaker"]=="True" or g["is_forfeit"]=="True": continue
    m=g["match_id"]
    if m in match: match[m]["ngames"]+=1; continue
    match[m]={"ev":g["event_id"],"tour":g["tour"],"ngames":1,
              "t1":(g["t1_p1"].lower(),g["t1_p2"].lower()),
              "t2":(g["t2_p1"].lower(),g["t2_p2"].lower())}

def build(label_src):
    out=defaultdict(list)
    for r in rd(ROOT/"data/decider_serve_splits.csv"):
        meta=match.get(r["match_id"])
        if not meta: continue
        vals=[v2.get(p) for p in meta["t1"]+meta["t2"]]
        if not all(v is not None for v in vals): continue
        eta=team_eta(*vals)
        h=hour.get(r["match_id"])
        wind=hourly.get((meta["ev"],h)) if h else None
        if wind is None: continue
        s = label_src.get(meta["ev"]) if label_src is ovr else geo.get(meta["ev"])
        if label_src is ovr: s = ovr.get(meta["ev"], geo.get(meta["ev"]))
        for side,sgn in (("a",1.0),("b",-1.0)):
            n=int(r[f"r{side}_pre"])+int(r[f"r{side}_post"])
            wins=int(r[f"w{side}_pre"])+int(r[f"w{side}_post"])
            if n<4: continue
            out[s].append({"ev":meta["ev"],"n":n,"wins":wins,"adv":sgn*eta,"w":wind/10.0})
    return out

def irls(X,y,n,iters=60):
    beta=np.zeros(X.shape[1])
    for _ in range(iters):
        z=np.clip(X@beta,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
        step=np.linalg.solve((X*W[:,None]).T@X, X.T@(y-n*p)); beta=beta+step
        if np.max(np.abs(step))<1e-12: break
    return beta
def swse(X,y,n,beta,cl_):
    z=np.clip(X@beta,-30,30); p=1/(1+np.exp(-z)); W=n*p*(1-p)
    B=np.linalg.inv((X*W[:,None]).T@X); s=X*(y-n*p)[:,None]
    cl=defaultdict(list)
    for i,c in enumerate(cl_): cl[c].append(i)
    M=sum(np.outer(s[ix].sum(0),s[ix].sum(0)) for ix in cl.values()); G=len(cl)
    return np.sqrt(np.diag(B@(M*G/(G-1))@B)), G

def run(rows,label):
    X=np.array([[1.0,r["adv"],r["w"],r["adv"]*r["w"]] for r in rows])
    y=np.array([r["wins"] for r in rows],float); n=np.array([r["n"] for r in rows],float)
    b=irls(X,y,n); se,G=swse(X,y,n,b,[r["ev"] for r in rows])
    print(f"  {label:<46} n_rallies={int(n.sum()):>7} ev={G:>3}  b={b[1]:+.4f}  "
          f"d={b[3]:+.4f} se={se[3]:.4f} [{b[3]-1.96*se[3]:+.4f},{b[3]+1.96*se[3]:+.4f}]")
    return b[3]

print("=== published decider-GAME sample (data/decider_serve_splits.csv) ===")
for src,lab in ((geo,"HEURISTIC labels (as published)"),(ovr,"CORRECTED labels")):
    d=build(src)
    print(f"-- {lab}")
    for s in ("outdoor","indoor"):
        run(d[s], f"{s}")
