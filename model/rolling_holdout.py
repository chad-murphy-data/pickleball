"""Rolling-cutoff holdout: score games with fits frozen at Jun1/Aug1/Sep1 (2026-10-10).
Fits: SRM2_SUFFIX=_cut0801 SRM2_DATE_BEFORE=2026-08-01 SRM2_SAVE_DRAWS=1 python model/fit_v2.py (likewise _cut0901).
Run: python model/rolling_holdout.py out.pkl"""
import csv, json, sys, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'model'))
from v2_holdout import race_win_table
D=str(ROOT/'data')+'/'; M=str(ROOT/'model')+'/'
FITS={'Jun1':'_train','Aug1':'_cut0801','Sep1':'_cut0901'}
CUT={'Jun1':'2026-06-01','Aug1':'2026-08-01','Sep1':'2026-09-01'}
names={r['player_id']:r['full_name'] for r in csv.DictReader(open(D+'players.csv'))}
grid=np.linspace(0.01,0.99,981); tab={11:race_win_table(11,grid),15:race_win_table(15,grid)}
rng=np.random.default_rng(3)
def load(sfx):
    pl={r['player_id']:r for r in csv.DictReader(open(f'{D}v2_players{sfx}.csv'))}
    ch={frozenset((r['p1_name'],r['p2_name'])):float(r['chem_logit_mean']) for r in csv.DictReader(open(f'{D}v2_dyads{sfx}.csv'))}
    g=json.load(open(f'{M}v2_fit_summary{sfx}.json'))['scalars']['gamma']['mean']
    dr=None
    try:
        z=np.load(f'{M}v2_draws{sfx}.npz',allow_pickle=True)
        dr=dict(v0=z['v0'],wl=z['walk_last'],dyn=z['dyn_id'],isd=z['is_dyn'],gam=z['gamma'],col={str(p):i for i,p in enumerate(z['player_ids'])})
    except Exception: pass
    return pl,ch,g,dr
F={k:load(v) for k,v in FITS.items()}
games=[g for g in csv.DictReader(open(D+'games.csv')) if g['is_forfeit']=='False' and g['scoring_format'] in('sideout_11','sideout_15') and g['date']>='2026-06-01']
def price(k,us,T,c,mode):
    pl,ch,gam,dr=F[k]
    cc=ch.get(frozenset((names.get(us[0],''),names.get(us[1],''))),0)-ch.get(frozenset((names.get(us[2],''),names.get(us[3],''))),0)
    if mode=='draws':
        vs=[]
        for u in us:
            i=dr['col'][u];v=dr['v0'][:,i].copy()
            if dr['isd'][i]: v+=dr['wl'][:,dr['dyn'][i]]
            vs.append(v);gm=dr['gam']
    else:
        m=np.array([float(pl[u]['value_now_mean']) for u in us]);s=np.array([float(pl[u]['value_now_sd']) for u in us])
        vs=list(m[:,None]+s[:,None]*rng.standard_normal((4,400)));gm=gam
    e=(vs[0]+vs[1]+gm*abs(vs[0]-vs[1]))-(vs[2]+vs[3]+gm*abs(vs[2]-vs[3]))+cc
    return float(np.interp(1/(1+np.exp(-e)),grid,tab[T]).mean())
rows=[]
for g in games:
    us=[g['t1_p1'],g['t1_p2'],g['t2_p1'],g['t2_p2']]
    if any(u not in F[k][0] or int(F[k][0][u]['games'])<10 for k in F for u in us): continue
    T=11 if g['scoring_format']=='sideout_11' else 15
    r=dict(date=g['date'],won=int(g['margin'])>0,match=g['match_id'])
    for k in F:
        r[k]=price(k,us,T,None,'approx')
        if F[k][3] is not None: r[k+'_d']=price(k,us,T,None,'draws')
    rows.append(r)
import pickle;pickle.dump(rows,open(sys.argv[1] if len(sys.argv)>1 else 'rows_rolling.pkl','wb'))
def slope(p,w):
    p=np.clip(p,1e-4,1-1e-4);l=np.log(p/(1-p));X=np.c_[np.ones(len(l)),l];th=np.array([0.,1.])
    for _ in range(50):
        q=1/(1+np.exp(-X@th));H=X.T@(X*(q*(1-q))[:,None])+1e-9*np.eye(2);th+=np.linalg.solve(H,X.T@(w-q))
    q=1/(1+np.exp(-X@th));return th,np.mean((q-w)**2)
def rep(lbl,x,keys):
    w=np.array([i['won'] for i in x]);print(f'-- {lbl} n={len(x)}')
    for k in keys:
        p=np.array([i[k] for i in x]);pc=np.clip(p,1e-6,1-1e-6)
        th,bcal=slope(p,w)
        print(f'  {k:8s} acc={np.mean((p>.5)==w):.3f} brier={np.mean((p-w)**2):.4f} ll={-np.mean(np.where(w,np.log(pc),np.log(1-pc))):.4f} | recal a={th[0]:+.3f} slope={th[1]:.3f} brier_after_recal={bcal:.4f}')
print('common sample, all fits')
for lbl,lo,hi in[('Aug1-Aug31','2026-08-01','2026-09-01'),('Sep1+','2026-09-01','9999'),('Aug1+ (both)','2026-08-01','9999')]:
    x=[i for i in rows if lo<=i['date']<hi]
    ks=['Jun1','Aug1']+(['Sep1'] if lo>='2026-09-01' else [])
    rep(lbl,x,ks+[k+'_d' for k in ks if k+'_d' in x[0] and k!='Jun1'])
