"""Diagnostic ball/paddle proximity at human contact times. No training.

Uses top-1 ball detections, not human ball labels. Always reports abstentions.
Thresholds are provisional, not validated probabilities or acceptance gates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from make_player_hitter_audit import find_pose
from player_hitter_review import build


def read_ball(path):
    out = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            t, x, y = map(float, (r['t_s'], r['predicted_x'], r['predicted_y']))
            if not np.isfinite([t, x, y]).all() or not (0 <= x < 1280 and 0 <= y < 720):
                raise ValueError('Invalid ball coordinates; expected original 1280x720 pixels')
            if abs(float(r['frame']) / 60 - t) > .002:
                raise ValueError('Ball timestamps do not match 60fps source frames')
            out.append((t, x, y))
    a = np.asarray(sorted(out), dtype=float).reshape(-1, 3)
    if not len(a) or len(np.unique(a[:, 0])) != len(a):
        raise ValueError('Empty predictions or duplicate timestamps')
    return a


def propose(z, names, ball, tc, radius=.4, margin=.15):
    # Require named observations on both sides of the contact. This catches
    # track resets near cuts, but is NOT a general camera-cut detector.
    people = sorted(set(names.values()))
    for name in people:
        tids = [tid for tid, nm in names.items() if nm == name]
        ts = z['t'][np.isin(z['track'], tids)]
        if not ((ts >= tc - .15) & (ts <= tc - .05)).any() or not ((ts >= tc + .05) & (ts <= tc + .15)).any():
            return dict(proposed=None, reason='incomplete_identity_context', evidence={})
    evidence = {name: [] for name in people}
    times = np.unique(z['t'])
    for t, bx, by in ball[np.abs(ball[:, 0] - tc) <= .12]:
        j = np.searchsorted(times, t)
        near = [i for i in (j - 1, j) if 0 <= i < len(times)]
        pt = times[min(near, key=lambda i: abs(times[i] - t))]
        if abs(pt - t) > .51 / 60:
            continue
        per_name = {}
        for i in np.flatnonzero(z['t'] == pt):
            name = names.get(int(z['track'][i]))
            if name is None:
                continue
            k, c, b = z['kpt'][i], z['kpc'][i], z['box'][i]
            distances = []
            for wrist, elbow in ((9, 7), (10, 8)):
                if min(c[wrist], c[elbow]) <= .3:
                    continue
                # Distance to wrist-to-estimated-paddle segment; neither point
                # is an actual paddle detection. Compare at the same instant.
                start = k[wrist]; delta = .5 * (k[wrist] - k[elbow])
                u = np.clip(np.dot(np.array([bx, by]) - start, delta) / max(float(np.dot(delta, delta)), 1e-9), 0, 1)
                px = float(np.linalg.norm(np.array([bx, by]) - (start + u * delta)))
                distances.append((px / max(float(b[3] - b[1]), 20), px))
            if distances:
                d, px = min(distances)
                record = dict(t_s=float(t), distance_body_heights=d, distance_px=px)
                if name not in per_name or d < per_name[name]['distance_body_heights']:
                    per_name[name] = record
        for name, record in per_name.items():
            evidence[name].append(record)
    scores = {}
    for name, samples in evidence.items():
        nearest = sorted(samples, key=lambda x:x['distance_body_heights'])[:3]
        if len(nearest) == 3:
            scores[name] = float(np.median([x['distance_body_heights'] for x in nearest]))
    ordered = sorted(scores, key=scores.get)
    result = dict(proposed=None, reason='insufficient_ball_or_pose', scores=scores, evidence=evidence)
    if len(ordered) != 4:
        return result
    first, second = ordered[:2]
    if scores[first] > radius:
        result['reason'] = 'ball_far_from_paddles'
    elif scores[second] - scores[first] < margin:
        result['reason'] = 'ambiguous_players'
    else:
        result.update(proposed=first, reason='proximity_support')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir', type=Path, default=Path('.'))
    p.add_argument('--predictions', type=Path, default=Path('data/vision/ball_hitter_predictions'))
    p.add_argument('--rallies', nargs='+', type=int, default=[6, 7, 8, 9, 10])
    p.add_argument('--state', type=Path, default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--contacts', type=Path, default=Path('data/vision/contact_labels_chicago0725.csv'))
    p.add_argument('--out', type=Path, default=Path('data/vision/ball_hitter_run1'))
    a = p.parse_args()
    results = []
    for rally in a.rallies:
        pose = find_pose(a.pose_dir, rally)
        z, names, report = build(pose, a.state, a.contacts, rally)
        if z['hw'].tolist() != [720,1280] or abs(float(z['fps'][0])-60)>.01:
            raise ValueError('Expected 1280x720 60fps poses')
        path = a.predictions / f'r{rally}' / 'predictions.csv'
        ball = read_ball(path)
        events = []
        for e in report['contacts']:
            covered = ball[0,0] <= e['t_s']-.12 and ball[-1,0] >= e['t_s']+.12
            target = e['labeled_hitter']; correction = None
            # Explicit user adjudication for this exact Chicago event only.
            if rally == 10 and e['shot'] == 2 and abs(e['t_s']-296.922)<.002 and target=='Emma Nelson':
                target='Ting Chieh Wei'; correction='User video review confirmed Ting Chieh Wei; source CSV preserved'
            q = (propose(z, names, ball, e['t_s']) if covered else
                 dict(proposed=None, reason='prediction_window_missing', evidence={}))
            events.append(dict(**e, evaluation_hitter=target, correction=correction,
                               ball_result=q, ball_agrees=q['proposed']==target,
                               reach_agrees=e['proposed_hitter']==target))
        assigned = [e for e in events if e['ball_result']['proposed'] is not None]
        summary = dict(total=len(events), assigned=len(assigned), abstained=len(events)-len(assigned),
                       ball_correct=sum(e['ball_agrees'] for e in events),
                       reach_correct=sum(e['reach_agrees'] for e in events))
        results.append(dict(rally=rally, summary=summary, events=events,
                            provenance={**report['provenance'], str(path):hashlib.sha256(path.read_bytes()).hexdigest()}))
        print('Rally',rally,summary)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(dict(scope='Development comparison at known contacts, human identities; no held-out claim',
        limitations=['Top-1 ball can be a distractor','Paddle position is an elbow/wrist extrapolation',
                     'Identity-context check is not camera-cut detection','No ball trajectory or motion-direction feature yet'],
        parameters=dict(window_s=.12,min_samples=3,max_distance_body_heights=.4,min_margin=.15),rallies=results),indent=2))
    print('Send',a.out/'report.json')


if __name__=='__main__':
    main()
