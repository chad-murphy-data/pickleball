"""Independent re-derivation of C1 measures 2 (side-out texture) and 3
(score shape), plus the checks the tester did not run:
  * game-1-only score shape (drops the game-3 collider)
  * true-game-hour-only subsample (attenuation probe)
  * filter-attrition vs wind (does the sanity window drop the tail?)
  * seed stability
"""
import csv, math, datetime as dt
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo
import numpy as np
import sys
sys.path.insert(0, "/home/user/pickleball/web")
from sitelib.race import team_eta

ROOT="/home/user/pickleball/"
TZ_ALIAS={"US/Arizona":"America/Phoenix"}
END=["g1_end_utc","g2_end_utc","g3_end_utc","g4_end_utc","g5_end_utc"]
rd=lambda p: list(csv.DictReader(open(ROOT+p)))
def f(x):
    try: return float(x)
    except: return None

geo={r["event_id"]:r for r in rd("data/event_geo.csv")}
lab={e:(g["setting"] if g["setting"] in ("indoor","outdoor") else None) for e,g in geo.items()}
for r in rd("data/venue_overrides.csv"):
    lab[r["event_id"]]=r["setting"] if r["setting"] in ("indoor","outdoor") else None
wx={}
for r in rd("data/event_weather_hourly.csv"):
    wx[(r["event_id"],r["local_time"][:13])]=(f(r["windspeed_10m"]),f(r["windgusts_10m"]),
                                              f(r["temperature_2m"]),f(r["precipitation"]))
_tzc={}
def tzof(e):
    n=geo.get(e,{}).get("timezone","")
    if n not in _tzc:
        try:_tzc[n]=ZoneInfo(TZ_ALIAS.get(n,n))
        except Exception:_tzc[n]=None
    return _tzc[n]
def putc(s):
    if not s: return None
    try: return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError: return None
def pnaive(s):
    if not s: return None
    try: return dt.datetime.fromisoformat(s.replace("Z",""))
    except ValueError: return None

mt=rd("data/match_times.csv")
resid=defaultdict(list)
for r in mt:
    tz=tzof(r["event_id"])
    if tz is None: continue
    ends=[putc(r[c]) for c in END]; ends=[e for e in ends if e]
    comp=pnaive(r["completed_local"])
    if not ends or comp is None: continue
    resid[r["event_id"]].append(round((comp-ends[-1].astimezone(tz).replace(tzinfo=None)).total_seconds()/3600.))
OFF={e:dt.timedelta(hours=Counter(v).most_common(1)[0][0]) for e,v in resid.items() if len(v)>=5}

# match hour key (true where possible, planned fallback) + a flag
mhour={}
for r in mt:
    ev=r["event_id"]; tz=tzof(ev)
    if tz is None: continue
    ends=[putc(r[c]) for c in END]; ends=[e for e in ends if e]
    if ends:
        mhour[r["match_id"]]=(ev, ends[0].astimezone(tz).strftime("%Y-%m-%dT%H"), 1)
    else:
        sl=pnaive(r["start_local"] or r["planned_start_local"])
        if sl is None or ev not in OFF: continue
        mhour[r["match_id"]]=(ev,(sl-OFF[ev]).strftime("%Y-%m-%dT%H"),0)
ghour={}
for r in mt:
    tz=tzof(r["event_id"])
    if tz is None: continue
    for i,c in enumerate(END,1):
        v=putc(r[c])
        if v: ghour[(r["match_id"],i)]=v.astimezone(tz).strftime("%Y-%m-%dT%H")

# ---------- estimator: within-transform + CR1 ----------
def demean(vals,gls):
    out=[np.array(v,float) for v in vals]; gs=[]
    for gl in gls:
        lv={v:i for i,v in enumerate(sorted(set(gl)))}
        gs.append(np.fromiter((lv[x] for x in gl),int,len(gl)))
    for _ in range(60):
        for g in gs:
            cnt=np.bincount(g)
            for m in out: m-=(np.bincount(g,weights=m)/cnt)[g]
    return out

def fit(rows,xcols,fes,ycol,cluster="ev",want=0):
    y=np.array([ycol(r) for r in rows],float)
    Xr=[np.array([c(r) for r in rows],float) for _,c in xcols]
    mats=demean(Xr+[y],[[fe(r) for r in rows] for fe in fes])
    X=np.column_stack(mats[:-1]); yy=mats[-1]
    keep=[i for i in range(X.shape[1]) if X[:,i].std()>1e-10 or i==want]
    X=X[:,keep]; want=keep.index(want)
    XtX=X.T@X; b=np.linalg.lstsq(XtX,X.T@yy,rcond=None)[0]; e=yy-X@b
    cl=[r[cluster] for r in rows]; keys=sorted(set(cl)); ix={k:i for i,k in enumerate(keys)}
    cid=np.fromiter((ix[c] for c in cl),int,len(cl)); G=len(keys); p=X.shape[1]
    meat=np.zeros((p,p))
    for c in range(G):
        m=cid==c; s=(X[m]*e[m,None]).sum(0); meat+=np.outer(s,s)
    Ai=np.linalg.pinv(XtX); n=len(rows)
    V=Ai@meat@Ai*(G/(G-1))*((n-1)/(n-p))
    se=math.sqrt(V[want,want])
    return b[want],se,len(rows),G

def rep(tag,res,scale=1.0,unit="",dec=3):
    b,se,n,G=res
    print("  %-46s n=%6d G=%3d  %+.*f [%+.*f, %+.*f] %s (MDE80 %.*f)"
          %(tag,n,G,dec,b*scale,dec,(b-1.96*se)*scale,dec,(b+1.96*se)*scale,unit,dec,2.8016*se*scale))

FE=[lambda r: r["ev"], lambda r: r["hour"]]


print("\n=== MEASURE 3 (score shape), independent ===")
v2={r["player_id"]:float(r["value_now_mean"]) for r in rd("data/v2_players.csv")}
gm=defaultdict(list)
for g in rd("data/games.csv"):
    if g["is_dreambreaker"]=="True" or g["is_forfeit"]=="True": continue
    gm[g["match_id"]].append(g)
for m in gm: gm[m].sort(key=lambda r:int(r["game_number"]))
srows=[]
for mid,gs in gm.items():
    h=mhour.get(mid)
    if not h: continue
    ev,mhk,tru=h
    for g in gs:
        if g["scoring_format"]!="sideout_11": continue
        gn=int(g["game_number"])
        hk=ghour.get((mid,gn),mhk)
        w=wx.get((ev,hk))
        if not w or w[0] is None or w[2] is None: continue
        vals=[v2.get(g[k]) for k in ("t1_p1","t1_p2","t2_p1","t2_p2")]
        if not all(x is not None for x in vals): continue
        eta=abs(team_eta(*vals))
        a,b=int(g["t1_score"]),int(g["t2_score"]); hi,lo=max(a,b),min(a,b)
        srows.append(dict(ev=ev,hour=int(hk[11:13]),day=hk[:10],wind=w[0],gust=w[1],temp=w[2],
                          eta=eta,margin=hi-lo,blow=1.0 if lo<=4 else 0.0,
                          deuce=1.0 if hi>11 else 0.0,gn=gn,tru=tru and (mid,gn) in ghour,
                          arm=lab.get(ev),nb=len(gs)))
O=[r for r in srows if r["arm"]=="outdoor"]
I=[r for r in srows if r["arm"]=="indoor"]
g1=[r for r in O if r["gn"]==1]
i1=[r for r in I if r["gn"]==1]
print("\n=== STRESS TEST: outdoor GAME-1 deuce rate vs wind ===")
print(" base deuce rate g1 outdoor %.4f (n=%d), indoor %.4f (n=%d)"
      % (np.mean([r["deuce"] for r in g1]),len(g1),np.mean([r["deuce"] for r in i1]),len(i1)))
C=[("w",lambda r:r["wind"]/10.),("t",lambda r:r["temp"]/10.),
   ("g",lambda r:r["eta"]),("g2",lambda r:r["eta"]**2)]
FED=[lambda r: r["ev"]+"|"+r["day"], lambda r: r["hour"]]

# other close-game outcomes in the same sample
for nm,fn in (("deuce (hi>11)",lambda r:r["deuce"]),
              ("margin<=2",lambda r:1.0 if r["margin"]<=2 else 0.0),
              ("margin<=3",lambda r:1.0 if r["margin"]<=3 else 0.0),
              ("blowout(lo<=4)",lambda r:r["blow"]),
              ("margin",lambda r:float(r["margin"]))):
    sc=100. if nm!="margin" else 1.
    rep("  g1 outdoor "+nm,fit(g1,C,FE,fn),sc,"",3 if nm=="margin" else 2)

print("\n -- pairs cluster bootstrap over events (1000, seed 99), g1 outdoor deuce")
keys=sorted({r["ev"] for r in g1}); byc=defaultdict(list)
for r in g1: byc[r["ev"]].append(r)
rng=np.random.default_rng(99); bs=[]
for _ in range(1000):
    repl=[]
    for j,k in enumerate(rng.choice(keys,size=len(keys))):
        for r in byc[k]:
            r2=dict(r); r2["ev"]=r["ev"]+"#%d"%j; repl.append(r2)
    try: bs.append(fit(repl,C,FE,lambda r:r["deuce"])[0])
    except Exception: pass
bs=np.sort(np.array(bs))
print("    boot mean %+.4f  95%% [%+.2f, %+.2f] pp  (share<0: %.3f)"
      % (bs.mean()*100, bs[25]*100, bs[974]*100, (bs<0).mean()))

print("\n -- leave-one-event-out jackknife")
base=fit(g1,C,FE,lambda r:r["deuce"])[0]*100
jk=[]
for e in keys:
    s=[r for r in g1 if r["ev"]!=e]
    jk.append((fit(s,C,FE,lambda r:r["deuce"])[0]*100,e,sum(1 for r in g1 if r["ev"]==e)))
jk.sort()
print("    full %+.2f pp ; LOO range %+.2f .. %+.2f over %d events" % (base,jk[0][0],jk[-1][0],len(keys)))
for b,e,n in sorted(jk,key=lambda t:-abs(t[0]-base))[:4]:
    print("      drop %s (n=%4d) -> %+.2f" % (e[:8],n,b))

print("\n -- wind bins (g1 outdoor, event+hour FE, ref 0-4 mph), deuce pp vs ref")
edges=[0,4,8,12,99]
def blab(r):
    for i in range(len(edges)-1):
        if edges[i]<=r["wind"]<edges[i+1]: return "%02d-%s"%(edges[i],edges[i+1] if edges[i+1]<99 else "+")
    return "?"
labs=sorted({blab(r) for r in g1})[1:]
CB=[("t",lambda r:r["temp"]/10.),("g",lambda r:r["eta"]),("g2",lambda r:r["eta"]**2)]
for i,L in enumerate(labs):
    CB.append((L,lambda r,L=L:1.0 if blab(r)==L else 0.0))
cnt=Counter(blab(r) for r in g1)
print("    ref 00-4 n=%d"%cnt["00-4"])
for i,L in enumerate(labs):
    b,se,n,G=fit(g1,CB,FE,lambda r:r["deuce"],want=3+i)
    print("      %-6s n=%5d  %+.2f [%+.2f, %+.2f] pp"%(L,cnt[L],100*b,100*(b-1.96*se),100*(b+1.96*se)))

print("\n -- label-arm sensitivity (g1 deuce)")
hi={}
for r in rd("data/venue_overrides.csv"):
    hi[r["event_id"]]= r["setting"] if (r["confidence"]=="high" and r["setting"] in ("indoor","outdoor")) else None
pub={e:(g["setting"] if g["setting"] in ("indoor","outdoor") else None) for e,g in geo.items()}
for nm,lb in (("corrected_all",lab),("corrected_HIGH conf",hi),("published heuristic",pub)):
    s=[r for r in srows if r["gn"]==1 and lb.get(r["ev"])=="outdoor"]
    if len(s)>400: rep("    outdoor g1 deuce ["+nm+"]",fit(s,C,FE,lambda r:r["deuce"]),100.,"pp/10mph",2)
    s=[r for r in srows if r["gn"]==1 and lb.get(r["ev"])=="indoor"]
    if len(s)>400: rep("    indoor  g1 deuce ["+nm+"]",fit(s,C,FE,lambda r:r["deuce"]),100.,"pp/10mph",2)

print("\n -- gust version, g1 outdoor")
CG=[("w",lambda r:r["gust"]/10.),("t",lambda r:r["temp"]/10.),("g",lambda r:r["eta"]),("g2",lambda r:r["eta"]**2)]
rep("    g1 outdoor deuce vs gust",fit([r for r in g1 if r["gust"] is not None],CG,FE,lambda r:r["deuce"]),100.,"pp/10mph gust",2)

print("\n -- tour split, g1 outdoor deuce")
tour={}
for g in rd("data/games.csv"): tour[g["game_id"]]=g["tour"]
print("    (skipped: game_id not carried)")

print("\n -- pooled outdoor+indoor interaction (g1 only)")
both=[r for r in srows if r["gn"]==1 and r["arm"] in ("outdoor","indoor")]
CI=[("wxout",lambda r:(r["wind"]/10.)*(1.0 if r["arm"]=="outdoor" else 0.0)),
    ("w",lambda r:r["wind"]/10.),("t",lambda r:r["temp"]/10.),
    ("g",lambda r:r["eta"]),("g2",lambda r:r["eta"]**2)]
rep("    OUT-IN interaction, g1 deuce",fit(both,CI,FE,lambda r:r["deuce"],want=0),100.,"pp/10mph",2)
