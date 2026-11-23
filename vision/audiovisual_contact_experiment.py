"""Fuse fixed-delay audio transients with visual contact features.

Outer leave-one-rally-out evaluation with nested threshold selection. Audio
delays are declared in advance and are not fitted using the test rally.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from audio_contact_probe_v2 import flux
from auto_contact_baseline import evaluate
from ball_hitter_baseline import propose
from learned_contact_experiment import (FEATURES, features, targets, train,
                                        predict, peaks, select_threshold)

AUDIO_DELAYS=np.array([.080,.105,.130,.155])
AUDIO_FEATURES=[f'audio_peak_delay_{int(x*1000)}ms' for x in AUDIO_DELAYS]


def audio_signal(audio_path):
    t,f,meta=flux(audio_path)
    energy=f.mean(axis=1)  # channel energy average; no waveform cancellation
    scaled=np.log1p(energy/np.maximum(np.median(energy,axis=0),1e-8)).mean(axis=1)
    return t,rankdata(scaled)/len(scaled),meta


def audio_features(grid,audio_t,audio_score):
    out=np.full((len(grid),len(AUDIO_DELAYS)),np.nan)
    for i,tc in enumerate(grid):
        for j,delay in enumerate(AUDIO_DELAYS):
            m=abs(audio_t-(tc+delay))<=.025
            if m.any():out[i,j]=audio_score[m].max()
    return out


def prepare(a):
    truth=json.loads(a.hitter_report.read_text());baseline=json.loads(a.baseline_report.read_text())
    with a.state.open(newline='') as f:state=list(csv.DictReader(f))
    at,ascore,meta=audio_signal(a.audio);datasets=[]
    with zipfile.ZipFile(a.predictions_zip) as archive:
        for w in truth['rallies']:
            r=w['rally'];pose=a.pose_dir/f'r{r:04d}.npz'
            raw=archive.read(f'data/vision/ball_hitter_predictions/r{r}/predictions.csv')
            if any(h not in w['provenance'].values() for h in
                   (hashlib.sha256(raw).hexdigest(),hashlib.sha256(pose.read_bytes()).hexdigest())):
                raise ValueError('Pose/prediction differs from prior scored input')
            old=next(x for x in baseline['rallies'] if x['rally']==r)
            if old['provenance']!=w['provenance']:raise ValueError('Visual reports differ in inputs')
            with np.load(pose,allow_pickle=False) as z:z={k:z[k] for k in z.files}
            ball=np.array([[float(row[k]) for k in ('t_s','predicted_x','predicted_y')]
                           for row in csv.DictReader(io.StringIO(raw.decode()))])
            if not np.isfinite(ball).all() or np.any(np.diff(ball[:,0])<=0):raise ValueError('Invalid ball stream')
            names={int(float(row['t_s'])):row['player'] for row in state
                   if int(row['rally_cum'])==r and row['kind']=='track_assign'}
            labels=[dict(shot=e['shot'],t_s=e['t_s'],evaluation_hitter=e['evaluation_hitter']) for e in w['events']]
            t,x=features(z,names,ball)
            x=np.c_[x,audio_features(t,at,ascore)]
            y,keep=targets(t,labels)
            datasets.append(dict(rally=r,z=z,names=names,ball=ball,labels=labels,t=t,x=x,y=y,keep=keep))
            print('Audiovisual features ready:',r,flush=True)
    return datasets,meta


def run(a):
    datasets,meta=prepare(a);results=[]
    for test in datasets:
        training=[d for d in datasets if d['rally']!=test['rally']]
        threshold,selection=select_threshold(training)
        model=train(np.concatenate([d['x'][d['keep']] for d in training]),
                    np.concatenate([d['y'][d['keep']] for d in training]))
        score=predict(model,test['x']);events=peaks(test['t'],score,threshold)
        for e in events:e['hitter']=propose(test['z'],test['names'],test['ball'],e['t_s'])
        result=evaluate(events,test['labels'])
        results.append(dict(rally=test['rally'],training_rallies=[d['rally'] for d in training],
            threshold=threshold,threshold_selection=selection,score=result,events=events,
            model=dict(median=model[0].tolist(),scale=model[1].tolist(),weights=model[2].tolist())))
        print('Held-out rally',test['rally'],'threshold',threshold,
              {k:result[k] for k in ('matched','missed','extra','correct_hitter')},flush=True)
    summary={k:sum(r['score'][k] for r in results) for k in
             ('labeled_contacts','predicted_contacts','matched','missed','extra','correct_hitter')}
    summary.update(wrong_hitter=sum(len(r['score']['wrong_hitter']) for r in results),
                   unknown_hitter=sum(len(r['score']['unknown_hitter']) for r in results))
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(dict(summary=summary,rallies=results,
        features=FEATURES+AUDIO_FEATURES,audio=meta,
        parameters=dict(audio_delays_s=AUDIO_DELAYS.tolist(),audio_peak_half_window_s=.025,
                        threshold_grid=[round(float(x),2) for x in np.arange(.5,1.,.05)],
                        objective='pooled inner-fold event F0.5',sample_fps=20,
                        positive_window_s=.075,negative_min_distance_s=.25,
                        separation_s=.2,ridge=.02,steps=1200),
        scope='Audiovisual contact classifier; nested rally validation; exploratory same-match development',
        caveats=['Fixed audio delays informed by the prior same-rally diagnostic range',
                 'Human identities and contact-derived windows supplied',
                 'Upstream ball detector not held out; rally 10 includes its training frames',
                 'These five rallies informed feature choices; not an untouched final evaluation',
                 'Audio percentile normalization uses unlabeled full audio'],
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (a.audio,a.state,a.hitter_report,a.baseline_report,a.predictions_zip)}),indent=2))
    print('TOTAL',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audio','pose-dir','predictions-zip','hitter-report','baseline-report','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--state',type=Path,default=Path('data/vision/state_labels_chicago0725.csv'))
    run(p.parse_args())


if __name__=='__main__':main()
