"""INDEPENDENT re-derivation of C1 texture numbers (adversarial verification).

Nothing imported from model/weather_review/c1_*.py — everything rebuilt here.
Estimator: within-transformation (absorb event + hour-of-day) + OLS, with
BOTH (a) analytic CR1 cluster-robust SE over events and (b) a nonparametric
pairs cluster bootstrap that treats a duplicated cluster as a SEPARATE unit
(so duplicated events get their own FE) -- the textbook version, different
from the tester's weighted/multinomial implementation.
"""
import csv, math, datetime as dt, sys
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo
import numpy as np

ROOT = "/home/user/pickleball/"
TZ_ALIAS = {"US/Arizona": "America/Phoenix"}
END = ["g1_end_utc","g2_end_utc","g3_end_utc","g4_end_utc","g5_end_utc"]

def rd(p):
    return list(csv.DictReader(open(ROOT+p)))

def f(x):
    try: return float(x)
    except: return None

# ---------- labels (corrected) ----------
geo = {r["event_id"]: r for r in rd("data/event_geo.csv")}
lab = {e: (g["setting"] if g["setting"] in ("indoor","outdoor") else None) for e,g in geo.items()}
for r in rd("data/venue_overrides.csv"):
    lab[r["event_id"]] = r["setting"] if r["setting"] in ("indoor","outdoor") else None

# ---------- weather ----------
wx = {}
for r in rd("data/event_weather_hourly.csv"):
    wx[(r["event_id"], r["local_time"][:13])] = (f(r["windspeed_10m"]), f(r["windgusts_10m"]),
                                                 f(r["temperature_2m"]), f(r["precipitation"]))

_tzc = {}
def tzof(e):
    n = geo.get(e,{}).get("timezone","")
    if n not in _tzc:
        try: _tzc[n] = ZoneInfo(TZ_ALIAS.get(n,n))
        except Exception: _tzc[n] = None
    return _tzc[n]

def putc(s):
    if not s: return None
    try: return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError: return None

def pnaive(s):
    if not s: return None
    try: return dt.datetime.fromisoformat(s.replace("Z",""))
    except ValueError: return None

# ---------- offset calibration (independent: use MODE of hour-rounded resid, not median) ----------
mt = rd("data/match_times.csv")
resid = defaultdict(list)
for r in mt:
    tz = tzof(r["event_id"])
    if tz is None: continue
    ends = [putc(r[c]) for c in END]; ends=[e for e in ends if e]
    comp = pnaive(r["completed_local"])
    if not ends or comp is None: continue
    want = ends[-1].astimezone(tz).replace(tzinfo=None)
    resid[r["event_id"]].append(round((comp-want).total_seconds()/3600.0))
OFF = {}
for e,v in resid.items():
    if len(v) >= 5:
        OFF[e] = dt.timedelta(hours=Counter(v).most_common(1)[0][0])

# ---------- games ----------
gm = defaultdict(list)
for g in rd("data/games.csv"):
    if g["is_dreambreaker"]=="True" or g["is_forfeit"]=="True": continue
    gm[g["match_id"]].append(g)
for m in gm: gm[m].sort(key=lambda r:int(r["game_number"]))

MT = {r["match_id"]: r for r in mt}

# ---------- build pace rows ----------
rows=[]
for mid, gs in gm.items():
    t = MT.get(mid)
    if not t: continue
    ev = gs[0]["event_id"]; tz = tzof(ev)
    if tz is None: continue
    ends = {}
    for i,c in enumerate(END,1):
        v = putc(t[c])
        if v: ends[i]=v
    if not ends: continue
    sl = pnaive(t["start_local"])
    start = (sl-OFF[ev]).replace(tzinfo=tz).astimezone(dt.timezone.utc) if (sl and ev in OFF) else None
    for g in gs:
        gn=int(g["game_number"])
        if gn not in ends: continue
        if gn-1 in ends: dur=(ends[gn]-ends[gn-1]).total_seconds(); first=0
        elif gn==1 and start is not None: dur=(ends[gn]-start).total_seconds(); first=1
        else: continue
        pts=int(g["t1_score"])+int(g["t2_score"])
        if pts<=0: continue
        hk = ends[gn].astimezone(tz).strftime("%Y-%m-%dT%H")
        w = wx.get((ev,hk))
        if not w or w[0] is None or w[2] is None: continue
        rows.append(dict(gid=g["game_id"],mid=mid,ev=ev,date=g["date"],tour=g["tour"],
                         fmt=g["scoring_format"],gn=gn,first=first,dur=dur,pts=pts,
                         wind=w[0],gust=w[1],temp=w[2],precip=w[3] or 0.0,
                         hour=int(hk[11:13]),arm=lab.get(ev),spp=dur/pts,
                         day=hk[:10]))
print("independent pace rows:", len(rows), "events", len({r['ev'] for r in rows}))
print(" arm:", Counter(r["arm"] or "dropped" for r in rows))

def win(rs, dmin=120., dmax=2700., smin=8., smax=200.):
    return [r for r in rs if dmin<=r["dur"]<=dmax and smin<=r["spp"]<=smax]

OUT = win([r for r in rows if r["arm"]=="outdoor"])
IND = win([r for r in rows if r["arm"]=="indoor"])
print("outdoor kept", len(OUT), "events", len({r['ev'] for r in OUT}))
print("indoor  kept", len(IND), "events", len({r['ev'] for r in IND}))
print("outdoor mean s/pt %.2f sd %.2f mean pts %.2f mean dur %.0f" %
      (np.mean([r['spp'] for r in OUT]), np.std([r['spp'] for r in OUT]),
       np.mean([r['pts'] for r in OUT]), np.mean([r['dur'] for r in OUT])))

# ---------------- estimator ----------------
def demean(vals, groups_list):
    """alternating projection onto multiple FE sets"""
    out = [np.array(v,float) for v in vals]
    gs=[]
    for gl in groups_list:
        lv={v:i for i,v in enumerate(sorted(set(gl)))}
        gs.append(np.fromiter((lv[x] for x in gl),int,len(gl)))
    for _ in range(60):
        for g in gs:
            cnt=np.bincount(g)
            for m in out:
                m -= (np.bincount(g,weights=m)/cnt)[g]
    return out

def fit(rows, xcols, fes, ycol, cluster="ev", nboot=1000, seed=7, want=0):
    y = np.array([ycol(r) for r in rows],float)
    Xr = [np.array([c(r) for r in rows],float) for _,c in xcols]
    gl = [[fe(r) for r in rows] for fe in fes]
    mats = demean(Xr+[y], gl)
    X = np.column_stack(mats[:-1]); yy = mats[-1]
    keepc=[i for i in range(X.shape[1]) if X[:,i].std()>1e-10 or i==want]
    X=X[:,keepc]; want=keepc.index(want)
    XtX = X.T@X
    b = np.linalg.lstsq(XtX, X.T@yy, rcond=None)[0]
    e = yy - X@b
    # CR1 cluster robust
    cl = [r[cluster] for r in rows]
    keys=sorted(set(cl)); ix={k:i for i,k in enumerate(keys)}
    cid=np.fromiter((ix[c] for c in cl),int,len(cl))
    G=len(keys); p=X.shape[1]
    meat=np.zeros((p,p))
    for c in range(G):
        m=cid==c
        s=(X[m]*e[m,None]).sum(0)
        meat+=np.outer(s,s)
    Ainv=np.linalg.pinv(XtX)
    n=len(rows)
    scale = (G/(G-1))*((n-1)/(n-p))
    V=Ainv@meat@Ainv*scale
    se=math.sqrt(V[want,want])
    # pairs cluster bootstrap: duplicated cluster = separate unit (own FE)
    rng=np.random.default_rng(seed)
    byc=defaultdict(list)
    for r,c in zip(rows,cl): byc[c].append(r)
    bs=[]
    for _ in range(nboot):
        rep=[]
        for j,k in enumerate(rng.choice(keys,size=G)):
            for r in byc[k]:
                r2=dict(r); r2["_rep"]=j; rep.append(r2)
        yb=np.array([ycol(r) for r in rep],float)
        Xb=[np.array([c(r) for r in rep],float) for _,c in xcols]
        glb=[[str(fe(r))+"|"+str(r["_rep"]) for r in rep] for fe in fes]
        mb=demean(Xb+[yb],glb)
        Xm=np.column_stack(mb[:-1])[:,keepc]; ym=mb[-1]
        try: bb=np.linalg.solve(Xm.T@Xm+1e-8*np.eye(Xm.shape[1]), Xm.T@ym)
        except np.linalg.LinAlgError: continue
        bs.append(bb[want])
    bs=np.sort(np.array(bs))
    lo,hi = bs[int(.025*len(bs))], bs[int(.975*len(bs))]
    return b[want], se, b[want]-1.96*se, b[want]+1.96*se, lo, hi, len(rows)

XC = [("wind10", lambda r: r["wind"]/10.),
      ("temp10", lambda r: r["temp"]/10.),
      ("invp",   lambda r: 1.0/r["pts"]),
      ("first",  lambda r: float(r["first"])),
      ("fmt15",  lambda r: 1.0 if r["fmt"]=="sideout_15" else 0.0),
      ("gn3",    lambda r: 1.0 if r["gn"]>=3 else 0.0)]
FE_EV = [lambda r: r["ev"], lambda r: r["hour"]]
FE_DAY= [lambda r: r["ev"]+"|"+r["day"], lambda r: r["hour"]]

def show(tag, res):
    b,se,alo,ahi,blo,bhi,n = res
    print("  %-38s n=%6d  b=%+.3f  CR1se %.3f -> [%+.3f,%+.3f] | pairsboot [%+.3f,%+.3f]"
          % (tag,n,b,se,alo,ahi,blo,bhi))

print("\n=== PACE, event+hour FE ===")
show("outdoor (all games)", fit(OUT, XC, FE_EV, lambda r: r["spp"]))
show("indoor  (all games)", fit(IND, XC, FE_EV, lambda r: r["spp"]))
XC2=[c for c in XC if c[0]!="first"]
show("outdoor GAMES 2+ only (pure UTC diff)", fit([r for r in OUT if not r["first"]], XC2, FE_EV, lambda r: r["spp"]))
show("outdoor GAME 1 only (start_local)", fit([r for r in OUT if r["first"]], XC2, FE_EV, lambda r: r["spp"]))
print("\n=== PACE, event-DAY + hour FE ===")
show("outdoor", fit(OUT, XC, FE_DAY, lambda r: r["spp"]))
show("outdoor GAMES 2+ only", fit([r for r in OUT if not r["first"]], XC2, FE_DAY, lambda r: r["spp"]))
