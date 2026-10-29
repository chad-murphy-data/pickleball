"""Small, regularized contact classifier with leave-one-rally-out evaluation.

Fits only on four rallies per fold. No threshold sweep. Upstream ball training
is unchanged, so this is NOT a wholly held-out end-to-end system evaluation.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from auto_contact_baseline import distance_stream, evaluate, match_times
from ball_hitter_baseline import propose
from motion_contact_experiment import fit

FEATURES=['paddle_distance_now','paddle_distance_min','paddle_distance_median',
          'speed_before','speed_after','velocity_change','direction_cosine',
          'fit_error_before','fit_error_after','join_gap',
          'arm_reach_max','torso_relative_wrist_speed_max']


def features(z,names,ball):
    distance=distance_stream(z,names,ball);t=ball[:,0]
    pose_times=np.unique(z['t']);rows={float(q):np.flatnonzero(z['t']==q) for q in pose_times}
    sample_indices=np.arange(0,len(ball),3)  # fixed 20Hz at 60fps input
    output=[]
    for i in sample_indices:
        tc=t[i];v=np.full(len(FEATURES),np.nan)
        local=distance[abs(t-tc)<=.10];good=local[np.isfinite(local)]
        v[0]=distance[i]
        if len(good):v[1:3]=[good.min(),np.median(good)]
        before=np.flatnonzero((t>=tc-.15)&(t<=tc-.025))
        after=np.flatnonzero((t>=tc+.025)&(t<=tc+.15))
        if min(len(before),len(after))>=5 and max(np.diff(t[before]).max(),np.diff(t[after]).max())<.04:
            left=fit(t[before]-tc,ball[before,1:]);right=fit(t[after]-tc,ball[after,1:])
            if left is not None and right is not None:
                p0,u0,e0=left;p1,u1,e1=right;s0=np.linalg.norm(u0);s1=np.linalg.norm(u1)
                v[3:10]=[s0,s1,np.linalg.norm(u1-u0),np.dot(u0,u1)/max(s0*s1,1e-6),e0,e1,np.linalg.norm(p0-p1)]
        j=np.searchsorted(pose_times,tc);choices=[q for q in (j-1,j) if 0<=q<len(pose_times)]
        pt=pose_times[min(choices,key=lambda q:abs(pose_times[q]-tc))]
        reaches=[];speeds=[]
        if abs(pt-tc)<=.51/60:
            for k in rows[float(pt)]:
                tid=int(z['track'][k])
                if tid not in names:continue
                kp,cf,b=z['kpt'][k],z['kpc'][k],z['box'][k];h=max(float(b[3]-b[1]),20)
                past=np.flatnonzero((z['track']==tid)&(z['t']>=pt-.09)&(z['t']<=pt-.04))
                prev=past[-1] if len(past) else None
                for wrist,shoulder in ((9,5),(10,6)):
                    if min(cf[wrist],cf[shoulder])<=.3:continue
                    relative=(kp[wrist]-kp[shoulder])/h;reaches.append(float(np.linalg.norm(relative)))
                    if prev is not None and min(z['kpc'][prev,wrist],z['kpc'][prev,shoulder])>.3:
                        oldh=max(float(z['box'][prev,3]-z['box'][prev,1]),20)
                        old=(z['kpt'][prev,wrist]-z['kpt'][prev,shoulder])/oldh
                        speeds.append(float(np.linalg.norm(relative-old)/(pt-z['t'][prev])))
        if reaches:v[10]=max(reaches)
        if speeds:v[11]=max(speeds)
        # Compress heavy tails. Retain signed direction cosine separately.
        for q in range(len(v)):
            if q!=6 and np.isfinite(v[q]):v[q]=np.log1p(max(0,v[q]))
        output.append(v)
    return t[sample_indices],np.asarray(output)


def targets(t,labels):
    nearest=np.min(abs(t[:,None]-np.array([x['t_s'] for x in labels])[None,:]),axis=1)
    # Ambiguous flanks excluded from training, never removed from evaluation.
    return (nearest<=.075).astype(float),(nearest<=.075)|(nearest>=.25)


def transform(x,median,scale):
    missing=~np.isfinite(x);filled=np.where(missing,median,x)
    return np.c_[np.clip((filled-median)/scale,-8,8),missing.astype(float),np.ones(len(x))]


def train(x,y):
    if len(np.unique(y))!=2:raise ValueError('Training needs positives and negatives')
    median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
    filled=np.where(np.isfinite(x),x,median);scale=np.maximum(np.std(filled,axis=0),.05)
    a=transform(x,median,scale);beta=np.zeros(a.shape[1])
    weights=np.where(y==1,.5/(y==1).sum(),.5/(y==0).sum())
    for _ in range(1200):
        p=1/(1+np.exp(-np.clip(a@beta,-30,30)))
        penalty=.02*beta;penalty[-1]=0
        beta-=.1*(a.T@(weights*(p-y))+penalty)
    return median,scale,beta


def predict(model,x):
    m,s,b=model
    return 1/(1+np.exp(-np.clip(transform(x,m,s)@b,-30,30)))


def peaks(t,score,threshold=.5):
    candidates=[i for i in range(len(t)) if score[i]>=threshold and
                score[i]>=score[max(0,i-1):min(len(t),i+2)].max()]
    selected=[]
    for i in sorted(candidates,key=lambda k:(-score[k],t[k])):
        if all(abs(t[i]-t[j])>=.2 for j in selected):selected.append(i)
    return [dict(t_s=float(t[i]),classifier_score=float(score[i])) for i in sorted(selected)]


def select_threshold(training):
    """Inner leave-one-rally-out scores; outer test data never enter here.

    Fixed grid and F0.5 objective declared before outer results. Count event
    matches, not frame accuracy. Pool inner rallies; ties prefer stricter values.
    """
    curves={float(round(x,2)):dict(matched=0,predicted=0,labeled=0)
            for x in np.arange(.5,1.,.05)}
    folds=[]
    for validation in training:
        fitting=[d for d in training if d['rally']!=validation['rally']]
        model=train(np.concatenate([d['x'][d['keep']] for d in fitting]),
                    np.concatenate([d['y'][d['keep']] for d in fitting]))
        scores=predict(model,validation['x'])
        folds.append(dict(validation_rally=validation['rally'],training_rallies=[d['rally'] for d in fitting]))
        for threshold,counts in curves.items():
            events=peaks(validation['t'],scores,threshold)
            counts['matched']+=len(match_times(events,validation['labels']))
            counts['predicted']+=len(events)
            counts['labeled']+=len(validation['labels'])
    rows=[]
    for threshold,c in curves.items():
        # F_beta = (1+beta^2) TP / (predicted + beta^2 * labeled).
        denom=c['predicted']+.25*c['labeled']
        rows.append(dict(threshold=threshold,**c,f05=1.25*c['matched']/denom if denom else 0.))
    best=max(rows,key=lambda x:(x['f05'],x['threshold']))
    return best['threshold'],dict(objective='Pooled inner-fold event F0.5; ties prefer higher threshold',folds=folds,curve=rows)


def run(a):
    truth=json.loads(a.hitter_report.read_text());baseline=json.loads(a.baseline_report.read_text())
    with a.state.open(newline='') as f:state=list(csv.DictReader(f))
    datasets=[]
    with zipfile.ZipFile(a.predictions_zip) as archive:
        for w in truth['rallies']:
            r=w['rally'];pose=a.pose_dir/f'r{r:04d}.npz';raw=archive.read(f'data/vision/ball_hitter_predictions/r{r}/predictions.csv')
            if any(h not in w['provenance'].values() for h in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(pose.read_bytes()).hexdigest())):
                raise ValueError('Pose/prediction hash mismatch')
            old=next(x for x in baseline['rallies'] if x['rally']==r)
            if old['provenance']!=w['provenance']:raise ValueError('Comparison input mismatch')
            with np.load(pose,allow_pickle=False) as z:z={k:z[k] for k in z.files}
            if z['hw'].tolist()!=[720,1280] or abs(float(z['fps'][0])-60)>.01:raise ValueError('Unsupported pose geometry')
            ball=np.array([[float(x[k]) for k in ('t_s','predicted_x','predicted_y')] for x in csv.DictReader(io.StringIO(raw.decode()))])
            if not np.isfinite(ball).all() or np.any(np.diff(ball[:,0])<=0):raise ValueError('Invalid ball stream')
            names={int(float(x['t_s'])):x['player'] for x in state if int(x['rally_cum'])==r and x['kind']=='track_assign'}
            labels=[dict(shot=e['shot'],t_s=e['t_s'],evaluation_hitter=e['evaluation_hitter']) for e in w['events']]
            t,x=features(z,names,ball);y,keep=targets(t,labels)
            datasets.append(dict(rally=r,z=z,names=names,ball=ball,labels=labels,t=t,x=x,y=y,keep=keep))
            print('Features ready:',r,flush=True)
    results=[]
    for test in datasets:
        training=[d for d in datasets if d['rally']!=test['rally']]
        threshold,selection=(select_threshold(training) if getattr(a,'nested_threshold',False) else (.5,None))
        print('Selected threshold for rally',test['rally'],threshold,flush=True)
        model=train(np.concatenate([d['x'][d['keep']] for d in training]),np.concatenate([d['y'][d['keep']] for d in training]))
        scores=predict(model,test['x']);events=peaks(test['t'],scores,threshold)
        for e in events:e['hitter']=propose(test['z'],test['names'],test['ball'],e['t_s'])
        evaluation=evaluate(events,test['labels'])
        results.append(dict(rally=test['rally'],training_rallies=[d['rally'] for d in training],threshold=threshold,threshold_selection=selection,score=evaluation,events=events,
            model=dict(median=model[0].tolist(),scale=model[1].tolist(),weights=model[2].tolist())))
        print('Held-out rally',test['rally'],{k:evaluation[k] for k in ('matched','missed','extra','correct_hitter')},flush=True)
    summary={k:sum(r['score'][k] for r in results) for k in ('labeled_contacts','predicted_contacts','matched','missed','extra','correct_hitter')}
    summary.update(wrong_hitter=sum(len(r['score']['wrong_hitter']) for r in results),unknown_hitter=sum(len(r['score']['unknown_hitter']) for r in results))
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(dict(summary=summary,rallies=results,features=FEATURES,
        parameters=dict(sample_fps=20,positive_window_s=.075,negative_min_distance_s=.25,score_threshold='nested F0.5' if getattr(a,'nested_threshold',False) else .5,min_separation_s=.2,ridge=.02,steps=1200),
        scope='Leave-one-rally-out contact classifier; exploratory same-match development, upstream detector not held out',
        caveats=['Balanced classifier score is not a calibrated probability','Unlabeled times treated as negatives outside exclusion buffer',
                 'Human identities and contact-derived clip windows retained','Rally 10 includes upstream ball-training data',
                 'These rallies informed feature choice, so this is not an untouched final test'],
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.state,a.predictions_zip,a.hitter_report,a.baseline_report)}),indent=2))
    print('TOTAL',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir',type=Path,required=True)
    p.add_argument('--predictions-zip',type=Path,required=True)
    p.add_argument('--hitter-report',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--state',type=Path,default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nested-threshold',action='store_true',help='Choose stricter thresholds by inner rally validation, never outer test labels')
    run(p.parse_args())


if __name__=='__main__':main()
