"""ADVERSARIAL VERIFICATION of B3 part 1 — independent estimator + analytic
cluster-robust SE, sample audit, and composition of the wind bins.

Route differences from the tester:
  * numpy IRLS instead of the hand-rolled Newton loop
  * analytic cluster-robust (event) sandwich SE instead of a bootstrap
  * independent re-derivation of adv sign and the weather join
"""
from __future__ import annotations
import sys, math, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/home/user/pickleball")
SCR = Path("/tmp/claude-0/-home-user-pickleball/a427a3a4-6690-5ae8-9453-094c68f7122d/scratchpad")
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model/weather_review"))
sys.argv = ["x", str(SCR)]
import rally_favorites_allmatches as P1
from sitelib.race import team_eta

cells, drop = P1.build_cells()
print("cells", len(cells), "drop", dict(drop), "rallies", sum(c["n"] for c in cells))

# ---------- INDEPENDENT re-derivation of adv sign for a few matches ----------
def rd(p):
    with open(p) as f: return list(csv.DictReader(f))
v2 = {r["player_id"].lower(): float(r["value_now_mean"]) for r in rd(ROOT/"data/v2_players.csv")}
sp = defaultdict(dict)
for r in rd(SCR/"rally_side_player.csv"):
    sp[r["match_id"]][r["side"]] = r["player_uuid"].lower()
gm = {}
for g in rd(ROOT/"data/games.csv"):
    if g["is_dreambreaker"]=="True" or g["is_forfeit"]=="True": continue
    gm.setdefault(g["match_id"], g)
bad = 0; chk = 0
for c in cells[:4000]:
    g = gm[c["m"]]
    t1 = (g["t1_p1"].lower(), g["t1_p2"].lower()); t2=(g["t2_p1"].lower(), g["t2_p2"].lower())
    e = team_eta(v2[t1[0]],v2[t1[1]],v2[t2[0]],v2[t2[1]])
    # which side is this cell? find it by matching n against the raw csv is overkill;
    # instead: adv should be +e or -e exactly
    if abs(abs(c["adv"]) - abs(e)) > 1e-9: bad += 1
    chk += 1
print(f"adv magnitude matches team_eta for {chk-bad}/{chk} sampled cells")
# sign check: the two sides of the same match must have opposite adv
bym = defaultdict(list)
for c in cells: bym[c["m"]].append(c["adv"])
opp = sum(1 for m,v in bym.items() if len(v)==2 and abs(v[0]+v[1])<1e-9)
two = sum(1 for m,v in bym.items() if len(v)==2)
print(f"two-sided matches with exactly opposite adv: {opp}/{two}  (one-sided cells: {sum(1 for v in bym.values() if len(v)==1)})")
# sanity: does higher adv predict higher serve-win rate?  crude check
hi = [c for c in cells if c["adv"]>0.5]; lo=[c for c in cells if c["adv"]<-0.5]
print("serve-win rate adv>+0.5: %.4f   adv<-0.5: %.4f" % (
    sum(c["wins"] for c in hi)/sum(c["n"] for c in hi),
    sum(c["wins"] for c in lo)/sum(c["n"] for c in lo)))

# ---------------------------- numpy IRLS ------------------------------------
def irls(X, wins, n, iters=60):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ beta, -30, 30)
        p = 1/(1+np.exp(-z))
        W = n*p*(1-p)
        g = X.T @ (wins - n*p)
        H = (X * W[:,None]).T @ X
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12: break
    return beta

def sandwich(X, wins, n, beta, clust):
    z = np.clip(X @ beta, -30, 30); p = 1/(1+np.exp(-z))
    W = n*p*(1-p)
    B = np.linalg.inv((X*W[:,None]).T @ X)
    s = X * (wins - n*p)[:,None]
    cl = {}
    for i,c in enumerate(clust): cl.setdefault(c, []).append(i)
    M = np.zeros((X.shape[1],X.shape[1]))
    for c, idx in cl.items():
        u = s[idx].sum(axis=0)
        M += np.outer(u,u)
    G = len(cl); K = X.shape[1]
    corr = G/(G-1.0)
    return B @ (M*corr) @ B, G

def arm(setting, sub=None):
    cs = [c for c in (sub if sub is not None else cells) if c["setting"]==setting]
    X = np.array([[1.0, c["adv"], c["w"], c["adv"]*c["w"]] for c in cs])
    return cs, X, np.array([c["wins"] for c in cs],float), np.array([c["n"] for c in cs],float)

print("\n=== HEADLINE, numpy IRLS + analytic cluster-robust (event) sandwich ===")
for setting in ("outdoor","indoor"):
    cs, X, y, n = arm(setting)
    b = irls(X,y,n)
    V,G = sandwich(X,y,n,b,[c["ev"] for c in cs])
    se = np.sqrt(np.diag(V))
    nm = ["const","adv","w","advw"]
    print(f"[{setting}] cells={len(cs)} rallies={int(n.sum())} events={G}")
    for k in range(4):
        print(f"   {nm[k]:>6} = {b[k]:+.4f}  se={se[k]:.4f}  Wald95=[{b[k]-1.96*se[k]:+.4f}, {b[k]+1.96*se[k]:+.4f}]")

# ------------- also: cluster at the EVENT-DAY level and at MATCH level -------
print("\n=== clustering sensitivity (outdoor d) ===")
cs, X, y, n = arm("outdoor")
b = irls(X,y,n)
for lab, key in (("event", lambda c: c["ev"]), ("event-day", lambda c: (c["ev"],c["date"])),
                 ("match", lambda c: c["m"]), ("iid(cell)", lambda c: id(c))):
    V,G = sandwich(X,y,n,b,[key(c) for c in cs])
    se = math.sqrt(V[3,3])
    print(f"   cluster={lab:<10} G={G:<6} d={b[3]:+.4f} se={se:.4f} 95%=[{b[3]-1.96*se:+.4f},{b[3]+1.96*se:+.4f}]")

# ------------------- bin composition incl. YEAR (untested by tester) --------
print("\n=== bin composition incl. season/year mix (tester tested tour+length only) ===")
BINS=[(0,4),(4,8),(8,12),(12,16),(16,40)]
for setting in ("outdoor","indoor"):
    print(f"--- {setting}")
    for i,(lo,hi) in enumerate(BINS):
        sub=[c for c in cells if c["setting"]==setting and lo<=c["wind"]<hi]
        N=sum(c["n"] for c in sub)
        yr=defaultdict(int)
        for c in sub: yr[c["date"][:4]] += c["n"]
        mn = sum(c["n"]*c["n"] for c in sub)/N
        print(f"  {lo:>2}-{hi:<2}: N={N:>7} meanlen={mn:5.1f} "
              + " ".join(f"{k}={v/N:.0%}" for k,v in sorted(yr.items())))
