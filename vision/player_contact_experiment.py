"""Player-specific contact features with nested rally validation.

Each row describes one player at one time. No player name or track ID is a
numeric feature. Reuses the prior feature definitions and logistic optimizer.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from learned_contact_experiment import features, train, predict, peaks, FEATURES
from auto_contact_baseline import evaluate, match_times
from ball_hitter_baseline import propose


def player_features(z,names,ball):
    players=sorted(set(names.values()));matrices=[];grid=None
    if len(players)!=4:raise ValueError('Need four human identities')
    for player in players:
        selected={tid:name for tid,name in names.items() if name==player}
        t,x=features(z,selected,ball)
        if grid is not None and not np.array_equal(t,grid):raise ValueError('Feature grids differ')
        grid=t;matrices.append(x)
    return grid,np.stack(matrices,axis=1),players


def player_targets(t,players,labels):
    delta=abs(t[:,None]-np.array([e['t_s'] for e in labels])[None,:])
    nearest=delta.min(axis=1)
    y=np.zeros((len(t),len(players)))
    for k,player in enumerate(players):
        matches=[j for j,e in enumerate(labels) if e['evaluation_hitter']==player]
        if matches:y[:,k]=(delta[:,matches].min(axis=1)<=.075)
    keep=np.repeat(((nearest<=.075)|(nearest>=.25))[:,None],len(players),axis=1)
    return y,keep


def fit_fold(ds):
    return train(np.concatenate([d['x'][d['keep']] for d in ds]),
                 np.concatenate([d['y'][d['keep']] for d in ds]))


def scores_for(model,d):
    n,k,f=d['x'].shape
    scores=predict(model,d['x'].reshape(-1,f)).reshape(n,k)
    return scores.max(axis=1),scores.argmax(axis=1)


def threshold_for(training):
    curves={round(float(x),2):[0,0,0] for x in np.arange(.5,1.,.05)};folds=[]
    for validation in training:
        fitting=[d for d in training if d['rally']!=validation['rally']]
        model=fit_fold(fitting);score,_=scores_for(model,validation)
        folds.append(dict(validation=validation['rally'],training=[d['rally'] for d in fitting]))
        for threshold,c in curves.items():
            events=peaks(validation['t'],score,threshold)
            c[0]+=len(match_times(events,validation['labels']));c[1]+=len(events);c[2]+=len(validation['labels'])
    rows=[dict(threshold=q,matched=c[0],predicted=c[1],labeled=c[2],
               f05=1.25*c[0]/max(c[1]+.25*c[2],1e-9)) for q,c in curves.items()]
    best=max(rows,key=lambda r:(r['f05'],r['threshold']))
    return best['threshold'],dict(folds=folds,curve=rows)


def run(a):
    truth=json.loads(a.hitter_report.read_text());baseline=json.loads(a.baseline_report.read_text())
    with a.state.open(newline='') as f:state=list(csv.DictReader(f))
    datasets=[]
    with zipfile.ZipFile(a.predictions_zip) as archive:
        for w in truth['rallies']:
            r=w['rally'];pose=a.pose_dir/f'r{r:04d}.npz'
            raw=archive.read(f'data/vision/ball_hitter_predictions/r{r}/predictions.csv')
            for sha in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(pose.read_bytes()).hexdigest()):
                if sha not in w['provenance'].values():raise ValueError('Input hash mismatch')
            old=next(x for x in baseline['rallies'] if x['rally']==r)
            if old['provenance']!=w['provenance']:raise ValueError('Reports differ in inputs')
            with np.load(pose,allow_pickle=False) as z:z={k:z[k] for k in z.files}
            if z['hw'].tolist()!=[720,1280] or abs(float(z['fps'][0])-60)>.01:raise ValueError('Unsupported pose geometry')
            ball=np.array([[float(row[k]) for k in ('t_s','predicted_x','predicted_y')]
                           for row in csv.DictReader(io.StringIO(raw.decode()))])
            if not np.isfinite(ball).all() or np.any(np.diff(ball[:,0])<=0):raise ValueError('Invalid ball stream')
            names={int(float(row['t_s'])):row['player'] for row in state if int(row['rally_cum'])==r and row['kind']=='track_assign'}
            labels=[dict(shot=e['shot'],t_s=e['t_s'],evaluation_hitter=e['evaluation_hitter']) for e in w['events']]
            t,x,players=player_features(z,names,ball);y,keep=player_targets(t,players,labels)
            datasets.append(dict(rally=r,z=z,names=names,ball=ball,t=t,x=x,y=y,keep=keep,players=players,labels=labels))
            print('Player features ready:',r,flush=True)
    results=[]
    for test in datasets:
        training=[d for d in datasets if d['rally']!=test['rally']]
        threshold,selection=threshold_for(training);model=fit_fold(training)
        scores,chosen=scores_for(model,test)
        evaluations={};event_sets={}
        for label,q in [('fixed',.5),('nested',threshold)]:
            events=peaks(test['t'],scores,q)
            for e in events:
                idx=int(np.argmin(abs(test['t']-e['t_s'])))
                e['classifier_player']=test['players'][chosen[idx]]
                # Preserve the proven hitter method; do not silently replace it.
                e['hitter']=propose(test['z'],test['names'],test['ball'],e['t_s'])
            evaluations[label]=evaluate(events,test['labels']);event_sets[label]=events
        results.append(dict(rally=test['rally'],training_rallies=[d['rally'] for d in training],
            threshold=threshold,selection=selection,scores=evaluations,events=event_sets,
            model=dict(median=model[0].tolist(),scale=model[1].tolist(),weights=model[2].tolist())))
        print('Held-out rally',test['rally'],'threshold',threshold,{mode:{k:s[k] for k in ('matched','missed','extra')} for mode,s in evaluations.items()},flush=True)
    summary={mode:{k:sum(r['scores'][mode][k] for r in results) for k in
                   ('labeled_contacts','predicted_contacts','matched','missed','extra','correct_hitter')} for mode in ('fixed','nested')}
    for mode in summary:
        summary[mode]['wrong_hitter']=sum(len(r['scores'][mode]['wrong_hitter']) for r in results)
        summary[mode]['unknown_hitter']=sum(len(r['scores'][mode]['unknown_hitter']) for r in results)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(dict(summary=summary,rallies=results,features=FEATURES,
        scope='Player-specific classifier; nested rally validation; exploratory same-match development',
        parameters=dict(threshold_grid=[round(float(x),2) for x in np.arange(.5,1.,.05)],objective='pooled event F0.5',
                        sample_fps=20,positive_window_s=.075,negative_buffer_s=.25,separation_s=.2,ridge=.02,steps=1200),
        limitations=['Human identities supplied; upstream ball detector not held out',
                     'These rallies informed feature choice; not an untouched final test',
                     'All non-hitting players are negatives at labeled contacts; labels may contain errors',
                     'One feature row per player, but each player still aggregates both arms'],
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.state,a.hitter_report,a.baseline_report,a.predictions_zip)}),indent=2))
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
