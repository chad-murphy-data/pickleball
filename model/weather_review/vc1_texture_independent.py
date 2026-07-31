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

# =========== MEASURE 2 ===========
print("\n=== MEASURE 2 (side-out texture), independent ===")
rows=[]
for r in rd("data/match_rally_summary.csv"):
    h=mhour.get(r["match_id"])
    if not h: continue
    ev,hk,tru=h
    w=wx.get((ev,hk))
    if not w or w[0] is None or w[2] is None: continue
    try: npts,nso,nral,k=int(r["n_points"]),int(r["n_sideouts"]),int(r["n_rallies"]),float(r["k_match"])
    except (ValueError,TypeError): continue
    if r["discipline"]!="doubles" or npts<15 or nral<20: continue
    rows.append(dict(ev=ev,hour=int(hk[11:13]),wind=w[0],temp=w[2],tru=tru,
                     so=nso/npts,sr=npts/nral,k=k,arm=lab.get(ev)))
XC=[("w",lambda r:r["wind"]/10.),("t",lambda r:r["temp"]/10.)]
for arm in ("outdoor","indoor"):
    sub=[r for r in rows if r["arm"]==arm]
    print(" [%s] n=%d" % (arm,len(sub)))
    for v,dec in (("so",4),("k",4),("sr",4)):
        rep("  "+v,fit(sub,XC,FE,lambda r,v=v:r[v]),1.0,"per +10mph",dec)
sub=[r for r in rows if r["arm"]=="outdoor" and r["tru"]]
print(" [outdoor, TRUE match hour only]")
for v,dec in (("so",4),("k",4),("sr",4)):
    rep("  "+v,fit(sub,XC,FE,lambda r,v=v:r[v]),1.0,"per +10mph",dec)

# =========== MEASURE 3 ===========
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
SC=[("w",lambda r:r["wind"]/10.),("t",lambda r:r["temp"]/10.),
    ("g",lambda r:r["eta"]),("g2",lambda r:r["eta"]**2)]
def scoreset(sub,tag):
    print(" [%s] n=%d events=%d" % (tag,len(sub),len({r['ev'] for r in sub})))
    rep("  blowout",fit(sub,SC,FE,lambda r:r["blow"]),100.,"pp/10mph",2)
    rep("  deuce",  fit(sub,SC,FE,lambda r:r["deuce"]),100.,"pp/10mph",2)
    rep("  margin", fit(sub,SC,FE,lambda r:r["margin"]),1.,"pts/10mph",3)
    y=np.array([r["margin"] for r in sub],float)
    Xr=[np.array([c(r) for r in sub],float) for _,c in SC[1:]]
    m=demean(Xr+[y],[[fe(r) for r in sub] for fe in FE])
    X=np.column_stack(m[:-1]); yy=m[-1]
    bb=np.linalg.lstsq(X,yy,rcond=None)[0]; r2=(yy-X@bb)**2
    res=fit(sub,SC,FE,lambda r,d=dict(zip(range(len(sub)),r2)):0.0) # placeholder
    # redo properly: regress squared resid on wind
    for i,r in enumerate(sub): r["_r2"]=float(r2[i])
    b,se,n,G=fit(sub,SC,FE,lambda r:r["_r2"])
    mm=r2.mean()
    print("  %-46s n=%6d G=%3d  %+.1f%% [%+.1f%%, %+.1f%%] margin-var (mean %.2f)"
          %("  margin-var",n,G,100*b/mm,100*(b-1.96*se)/mm,100*(b+1.96*se)/mm,mm))
for arm in ("outdoor","indoor"):
    scoreset([r for r in srows if r["arm"]==arm],arm)
scoreset([r for r in srows if r["arm"]=="outdoor" and r["gn"]==1],"outdoor GAME 1 ONLY (no decider collider)")
scoreset([r for r in srows if r["arm"]=="outdoor" and r["tru"]],"outdoor TRUE game hour only")

# gust for deuce, outdoor
sub=[r for r in srows if r["arm"]=="outdoor" and r["gust"] is not None]
GC=[("w",lambda r:r["gust"]/10.),("t",lambda r:r["temp"]/10.),
    ("g",lambda r:r["eta"]),("g2",lambda r:r["eta"]**2)]
rep(" deuce vs GUST outdoor",fit(sub,GC,FE,lambda r:r["deuce"]),100.,"pp/10mph gust",2)

# =========== filter attrition vs wind ===========
print("\n=== filter attrition vs wind (outdoor pace rows) ===")
