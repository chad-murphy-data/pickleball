"""Score model/registered_predictions.md (frozen 2026-07-12).

Registered method: fixed-effects regression (model/fixed_effects_dyads.py)
on games dated > 2026-07-12 only, using FULL-SEASON player FEs.

Implementation: fit player FEs + tour intercepts on the full 2026 season,
then estimate each registered dyad's coefficient by FWL restricted to the
H2 subset. Cluster-robust (CR1) SEs by match over H2 clusters.
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy import sparse

DATA = Path("data")
SPLIT = "2026-07-12"

rows = list(csv.DictReader((DATA / "model_data.csv").open()))
dyads = list(csv.DictReader((DATA / "model_dyads.csv").open()))
n = len(rows)
a = np.array([[int(r[k]) for k in ("a1","a2","a3","a4")] for r in rows])
d1 = np.array([int(r["dyad1"]) for r in rows])
d2 = np.array([int(r["dyad2"]) for r in rows])
match = np.array([int(r["match_idx"]) for r in rows])
tour = np.array([int(r["tour_idx"]) for r in rows])
y = np.array([float(r["margin"]) for r in rows])
date = np.array([r["date"] for r in rows])
H2 = date > SPLIT
n_players = a.max() + 1

print(f"full season: {n} games | H2 (> {SPLIT}): {H2.sum()} games")

ii, jj, vv = [], [], []
for slot, sign in ((0,1),(1,1),(2,-1),(3,-1)):
    ii.extend(range(n)); jj.extend(a[:,slot]); vv.extend([sign]*n)
ii.extend(range(n)); jj.extend(n_players + tour); vv.extend([1.0]*n)
X = sparse.csr_matrix((vv,(ii,jj)), shape=(n, n_players+2))
XtX_inv = np.linalg.pinv((X.T @ X).toarray(), rcond=1e-10)
beta0 = XtX_inv @ (X.T @ y)
resid0 = y - X @ beta0

name_by_uuid = {}
for d in dyads:
    name_by_uuid[(d["p1_name"], d["p2_name"])] = d

def score(n1, n2):
    di = None
    for i, d in enumerate(dyads):
        if {d["p1_name"], d["p2_name"]} == {n1, n2}:
            di = i; meta = d; break
    if di is None:
        return None
    z = (d1 == di).astype(float) - (d2 == di).astype(float)
    gamma = XtX_inv @ (X.T @ z)
    z_t = z - X @ gamma
    res = {}
    for label, mask in (("full", np.ones(n, bool)), ("H2", H2)):
        zt, r0, mt = z_t[mask], resid0[mask], match[mask]
        denom = float(zt @ zt)
        g_played = int(((d1==di)|(d2==di))[mask].sum())
        if denom < 1e-8 or g_played == 0:
            res[label] = dict(games=g_played, beta=None, se=None)
            continue
        beta = float(zt @ r0) / denom
        e = r0 - beta*zt
        uniq, inv = np.unique(mt, return_inverse=True)
        sc = np.zeros(len(uniq)); np.add.at(sc, inv, zt*e)
        gcl = len(uniq)
        meat = float((sc**2).sum()) * gcl/(gcl-1)
        se = np.sqrt(meat)/denom
        res[label] = dict(games=g_played, beta=beta, se=se)
    res["context"] = meta["context"]
    return res

PAIRS = [("Anna Leigh Waters","Anna Bright"),
         ("Christian Alshon","Hayden Patriquin"),
         ("Anna Bright","Hayden Patriquin")]
out = {}
for p in PAIRS:
    r = score(*p)
    out[" + ".join(p)] = r
    if r is None:
        print(f"\n{p[0]} + {p[1]}: DYAD NOT FOUND"); continue
    print(f"\n{p[0]} + {p[1]}  ({r['context']})")
    for lab in ("full","H2"):
        d = r[lab]
        if d["beta"] is None:
            print(f"  {lab:5s}: {d['games']} games together — not estimable")
        else:
            print(f"  {lab:5s}: {d['games']} games together   "
                  f"beta {d['beta']:+.3f} ± {d['se']:.3f}  (t={d['beta']/d['se']:+.2f})")
json.dump(out, open(DATA / "registered_scoring.json", "w"), indent=1, default=str)
print("\nwrote data/registered_scoring.json")
