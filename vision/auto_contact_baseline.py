"""Detect ball/paddle approaches without contact labels, then score separately.

Development baseline on existing rally clips; not a validated contact detector.
Uses human track identities and existing video windows, not whole-match discovery.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ball_hitter_baseline import read_ball, propose
from make_player_hitter_audit import find_pose
from player_hitter_review import build


def distance_stream(z, names, ball):
    times = np.unique(z['t'])
    rows = {float(t): np.flatnonzero(z['t'] == t) for t in times}
    out = np.full(len(ball), np.nan)
    for n, (t, bx, by) in enumerate(ball):
        j = np.searchsorted(times, t)
        choices = [i for i in (j-1, j) if 0 <= i < len(times)]
        pt = times[min(choices, key=lambda i: abs(times[i]-t))]
        if abs(pt-t) > .51/60:
            continue
        distances = []
        for i in rows[float(pt)]:
            if int(z['track'][i]) not in names:
                continue
            k, c, b = z['kpt'][i], z['kpc'][i], z['box'][i]
            for wrist, elbow in ((9,7),(10,8)):
                if min(c[wrist], c[elbow]) <= .3:
                    continue
                delta = .5*(k[wrist]-k[elbow])
                u = np.clip(np.dot([bx-k[wrist,0],by-k[wrist,1]],delta)/max(float(np.dot(delta,delta)),1e-9),0,1)
                d = np.linalg.norm(np.array([bx,by])-(k[wrist]+u*delta))/max(float(b[3]-b[1]),20)
                distances.append(float(d))
        if distances:
            out[n] = min(distances)
    return out


def pick_contacts(t, d, radius=.4, rise=.1, separation=.20):
    """Local proximity minima with approach/departure evidence; no labels.

    A small median filter suppresses isolated position jumps. Gaps invalidate
    the neighborhood. Constants are provisional; report errors before tuning.
    """
    smooth = np.full(len(d), np.nan)
    for i in range(1,len(d)-1):
        if np.isfinite(d[i-1:i+2]).all() and t[i+1]-t[i-1] < .05:
            smooth[i] = np.median(d[i-1:i+2])
    candidates = []
    for i in range(len(t)):
        if not np.isfinite(smooth[i]) or smooth[i] > radius:
            continue
        local = (t >= t[i]-.15) & (t <= t[i]+.15)
        before = (t >= t[i]-.15) & (t <= t[i]-.05)
        after = (t >= t[i]+.05) & (t <= t[i]+.15)
        if before.sum()<3 or after.sum()<3 or not np.isfinite(smooth[local]).all():
            continue
        if np.diff(t[local]).max() > .04:
            continue
        if smooth[i] > smooth[local].min()+1e-9:
            continue
        prominence = min(float(np.median(smooth[before])),float(np.median(smooth[after])))-smooth[i]
        if prominence >= rise:
            candidates.append(dict(t_s=float(t[i]),distance=float(smooth[i]),rise=float(prominence)))
    kept = []
    for c in sorted(candidates,key=lambda x:(-x['rise'],x['distance'],x['t_s'])):
        if all(abs(c['t_s']-k['t_s'])>=separation for k in kept):
            kept.append(c)
    return sorted(kept,key=lambda x:x['t_s'])


def match_times(predictions, labels, tolerance=.15):
    """Ordered one-to-one matching: maximize matches, then minimize time error.

    Hitter identities never influence matching. Inputs must be time-sorted.
    """
    n,m=len(predictions),len(labels)
    dp={(0,0):(0,0.,[])}
    for i in range(n+1):
        for j in range(m+1):
            if not (i or j):continue
            choices=[]
            if i:choices.append(dp[i-1,j])
            if j:choices.append(dp[i,j-1])
            if i and j:
                err=abs(predictions[i-1]['t_s']-labels[j-1]['t_s'])
                if err<=tolerance:
                    count,cost,pairs=dp[i-1,j-1]
                    choices.append((count+1,cost+err,pairs+[(i-1,j-1)]))
            dp[i,j]=max(choices,key=lambda x:(x[0],-x[1]))
    return dp[n,m][2]


def evaluate(events, labels):
    pairs=match_times(events,labels)
    used_p={i for i,j in pairs};used_l={j for i,j in pairs}
    wrong=[];unknown=[];correct=0;errors=[]
    for i,j in pairs:
        p,l=events[i],labels[j];errors.append(abs(p['t_s']-l['t_s']))
        record=dict(predicted_t_s=p['t_s'],label_t_s=l['t_s'],shot=l['shot'],
                    proposed=p['hitter']['proposed'],expected=l['evaluation_hitter'])
        if p['hitter']['proposed'] is None:unknown.append(record)
        elif p['hitter']['proposed']==l['evaluation_hitter']:correct+=1
        else:wrong.append(record)
    return dict(labeled_contacts=len(labels),predicted_contacts=len(events),matched=len(pairs),
                missed=len(labels)-len(pairs),extra=len(events)-len(pairs),
                precision=len(pairs)/len(events) if events else None,
                recall=len(pairs)/len(labels) if labels else None,
                correct_hitter=correct,wrong_hitter=wrong,unknown_hitter=unknown,
                median_timing_error_s=float(np.median(errors)) if errors else None,
                missed_labels=[l for j,l in enumerate(labels) if j not in used_l],
                extra_events=[p for i,p in enumerate(events) if i not in used_p])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir',type=Path,default=Path('.'))
    p.add_argument('--predictions',type=Path,default=Path('data/vision/ball_hitter_predictions'))
    p.add_argument('--state',type=Path,default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--contacts',type=Path,default=Path('data/vision/contact_labels_chicago0725.csv'))
    p.add_argument('--rallies',nargs='+',type=int,default=[6,7,8,9,10])
    p.add_argument('--out',type=Path,default=Path('data/vision/auto_contact_run1'))
    a=p.parse_args();results=[]
    if a.out.exists():raise ValueError('Output folder already exists; choose a new --out')
    for rally in a.rallies:
        pose=find_pose(a.pose_dir,rally)
        z,names,report=build(pose,a.state,a.contacts,rally)
        if z['hw'].tolist()!=[720,1280] or abs(float(z['fps'][0])-60)>.01:
            raise ValueError('Expected original 1280x720 60fps poses')
        path=a.predictions/f'r{rally}'/'predictions.csv';ball=read_ball(path)
        # Detector functions receive no contact timestamps or hitter labels.
        distance=distance_stream(z,names,ball)
        events=pick_contacts(ball[:,0],distance)
        for e in events:e['hitter']=propose(z,names,ball,e['t_s'])
        labels=[]
        for e in report['contacts']:
            if not ball[0,0] <= e['t_s'] <= ball[-1,0]:
                raise ValueError(f'Rally {rally}: incomplete prediction window. Regenerate with --contact-windows.')
            target=e['labeled_hitter'];correction=None
            if rally==10 and e['shot']==2 and abs(e['t_s']-296.922)<.002 and target=='Emma Nelson':
                target='Ting Chieh Wei';correction='User adjudication; source label preserved'
            labels.append(dict(shot=e['shot'],t_s=e['t_s'],evaluation_hitter=target,
                               source_hitter=e['labeled_hitter'],correction=correction))
        score=evaluate(events,labels)
        results.append(dict(rally=rally,score=score,events=events,
                            prediction_span_s=[float(ball[0,0]),float(ball[-1,0])],
                            provenance={**report['provenance'],str(path):hashlib.sha256(path.read_bytes()).hexdigest()}))
        print('Rally',rally,{k:score[k] for k in ('labeled_contacts','predicted_contacts','matched','missed','extra','correct_hitter')},flush=True)
    a.out.mkdir(parents=True)
    (a.out/'report.json').write_text(json.dumps(dict(
        scope='Automatic contact timing in existing clips; human identities; development data, not held-out',
        limitations=['Clip boundaries originally depend on contact labels',
                     'Near-paddle passage, held balls, and detector jumps can create false contacts',
                     'Unknown hitter does not remove a predicted contact from scoring',
                     'No camera-cut detector or whole-match segmentation'],
        parameters=dict(radius=.4,rise=.1,separation_s=.20,local_window_s=.15,matching_tolerance_s=.15),
        rallies=results),indent=2))
    print('Send',a.out/'report.json')


if __name__=='__main__':main()
