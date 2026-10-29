"""Offline experiment: robust incoming/outgoing image motion near a paddle.

Thresholds are exploratory. Bounces can still mimic contacts. No tuning loop.
Ground-truth times are used only by the evaluator, not detect().
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from auto_contact_baseline import distance_stream, evaluate
from ball_hitter_baseline import propose

PARAMETERS=dict(half_window_s=.15,center_gap_s=.025,min_points=5,
                max_fit_error_px=12.,max_join_gap_px=25.,min_velocity_change_px_s=150.,
                min_fit_improvement=.35,max_paddle_distance_body_heights=.4,separation_s=.20)


def fit(t,xy):
    # Median pairwise slope resists isolated jumps without choosing a random seed.
    slopes=[]
    for i in range(len(t)):
        for j in range(i+1,len(t)):
            if t[j]-t[i]>.025:slopes.append((xy[j]-xy[i])/(t[j]-t[i]))
    if not slopes:return None
    velocity=np.median(slopes,axis=0)
    origin=np.median(xy-t[:,None]*velocity,axis=0)
    residual=np.linalg.norm(xy-(origin+t[:,None]*velocity),axis=1)
    return origin,velocity,float(np.median(residual))


def detect(ball,distances):
    t=ball[:,0];xy=ball[:,1:];candidates=[]
    for i,tc in enumerate(t):
        if not np.isfinite(distances[i]) or distances[i]>.4:continue
        before=np.flatnonzero((t>=tc-.15)&(t<=tc-.025))
        after=np.flatnonzero((t>=tc+.025)&(t<=tc+.15))
        if min(len(before),len(after))<5:continue
        if max(np.diff(t[before]).max(),np.diff(t[after]).max())>.04:continue
        left=fit(t[before]-tc,xy[before]);right=fit(t[after]-tc,xy[after])
        if left is None or right is None:continue
        p0,v0,e0=left;p1,v1,e1=right
        gap=float(np.linalg.norm(p0-p1));change=float(np.linalg.norm(v1-v0))
        if max(e0,e1)>12 or gap>25 or change<150:continue
        both=np.r_[before,after];whole=fit(t[both]-tc,xy[both])
        if whole is None:continue
        split=(e0+e1)/2
        improvement=(whole[2]-split)/max(whole[2],1.)
        if improvement<.35:continue
        quality=(whole[2]-split)/(1+split+gap/2)
        candidates.append(dict(t_s=float(tc),quality=float(quality),
            fit_error_px=[e0,e1],join_gap_px=gap,velocity_change_px_s=change,
            fit_improvement=float(improvement),paddle_distance=float(distances[i])))
    picked=[]
    for c in sorted(candidates,key=lambda x:(-x['quality'],x['t_s'])):
        if all(abs(c['t_s']-p['t_s'])>=.20 for p in picked):picked.append(c)
    return sorted(picked,key=lambda x:x['t_s'])


def run(a):
    truth=json.loads(a.hitter_report.read_text());baseline=json.loads(a.baseline_report.read_text())
    with a.state.open(newline='') as f:state=list(csv.DictReader(f))
    results=[]
    with zipfile.ZipFile(a.predictions_zip) as archive:
        for w in truth['rallies']:
            rally=w['rally'];pose=a.pose_dir/f'r{rally:04d}.npz'
            member=f'data/vision/ball_hitter_predictions/r{rally}/predictions.csv'
            raw=archive.read(member)
            expected=set(w['provenance'].values())
            if hashlib.sha256(raw).hexdigest() not in expected or hashlib.sha256(pose.read_bytes()).hexdigest() not in expected:
                raise ValueError('Prediction/pose hash differs from scored run')
            old=next(x for x in baseline['rallies'] if x['rally']==rally)
            if old['provenance']!=w['provenance']:raise ValueError('Reports describe different inputs')
            with np.load(pose,allow_pickle=False) as z:poses={k:z[k] for k in z.files}
            ball=np.array([[float(x[k]) for k in ('t_s','predicted_x','predicted_y')]
                for x in csv.DictReader(io.StringIO(raw.decode()))])
            if not np.isfinite(ball).all() or np.any(np.diff(ball[:,0])<=0):raise ValueError('Invalid ball stream')
            names={int(float(x['t_s'])):x['player'] for x in state if int(x['rally_cum'])==rally and x['kind']=='track_assign'}
            # Confirm the identities against the earlier report's saved track names.
            for tr in w.get('tracks',[]):
                if tr['name']!='Unassigned' and names.get(tr['track'])!=tr['name']:raise ValueError('Identity mismatch')
            d=distance_stream(poses,names,ball)
            events=detect(ball,d)
            for e in events:e['hitter']=propose(poses,names,ball,e['t_s'])
            labels=[dict(shot=e['shot'],t_s=e['t_s'],evaluation_hitter=e['evaluation_hitter']) for e in w['events']]
            score=evaluate(events,labels)
            results.append(dict(rally=rally,score=score,events=events,baseline_score=old['score']))
            print('Rally',rally,{k:score[k] for k in ('matched','missed','extra','correct_hitter')},flush=True)
    summary={key:sum(r['score'][key] for r in results) for key in ('labeled_contacts','predicted_contacts','matched','missed','extra','correct_hitter')}
    summary['wrong_hitter']=sum(len(r['score']['wrong_hitter']) for r in results)
    summary['unknown_hitter']=sum(len(r['score']['unknown_hitter']) for r in results)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(dict(parameters=PARAMETERS,summary=summary,rallies=results,
        scope='Exploratory same-data comparison; no held-out claim; current hitter method unchanged',
        caveats=['Image motion is not 3D velocity','Bounces and tracking jumps can mimic motion changes',
                 'Rally windows and identities supplied; not whole-match automation'],
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.hitter_report,a.baseline_report,a.state,a.predictions_zip)}),indent=2))
    print('TOTAL',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir',type=Path,required=True)
    p.add_argument('--predictions-zip',type=Path,required=True)
    p.add_argument('--hitter-report',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--state',type=Path,default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
