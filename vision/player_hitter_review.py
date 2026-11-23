"""Replay saved pose tracks and a simple hitter baseline at HUMAN contact times.

No model inference or training. Names are existing human track assignments.
Run from repo root: python3 vision/player_hitter_review.py --npz PATH --video PATH
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def build(npz, state, contacts, rally):
    with np.load(npz, allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    required = {'t', 'track', 'box', 'kpt', 'kpc', 'hw', 'fps'}
    if not required <= z.keys() or not len(z['t']):
        raise ValueError('Missing or empty pose arrays')
    n = len(z['t'])
    for key, shape in [('track', (n,)), ('box', (n, 4)),
                       ('kpt', (n, 17, 2)), ('kpc', (n, 17))]:
        if z[key].shape != shape or not np.isfinite(z[key]).all():
            raise ValueError('Invalid pose array: ' + key)
    if not np.isfinite(z['t']).all() or np.any(np.diff(z['t']) < 0):
        raise ValueError('Pose timestamps must be finite and sorted')
    names = {}
    with open(state, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['rally_cum']) == rally and row['kind'] == 'track_assign':
                tid = int(float(row['t_s']))
                if tid in names and names[tid] != row['player']:
                    raise ValueError('Conflicting human track assignments')
                names[tid] = row['player']
    if len(set(names.values())) != 4:
        raise ValueError('Need existing human assignments for all four players')
    features, tracks = {}, []
    for tid in np.unique(z['track']):
        mask = z['track'] == tid
        t, k, c, b = (z[key][mask] for key in ('t', 'kpt', 'kpc', 'box'))
        if len(np.unique(t)) != len(t):
            raise ValueError('Duplicate timestamp within a track')
        reach = np.full(len(t), np.nan)
        h = np.maximum(b[:, 3] - b[:, 1], 20)
        for i in range(len(t)):
            vals = [float(np.linalg.norm(k[i, w] - k[i, s]) / h[i])
                    for w, s in ((9, 5), (10, 6)) if min(c[i, w], c[i, s]) > .3]
            if vals:
                reach[i] = max(vals)
        if int(tid) in names:
            features.setdefault(names[int(tid)], []).append((t, reach))
        tracks.append(dict(track=int(tid), name=names.get(int(tid), 'Unassigned'),
                           detections=len(t), start_s=float(t[0]), end_s=float(t[-1]),
                           max_gap_s=float(np.diff(t).max()) if len(t) > 1 else None,
                           ankle_available_fraction=float((c[:, 15:17] > .3).any(axis=1).mean())))
    events = []
    with open(contacts, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['rally_cum']) != rally or row['contact'] != '1':
                continue
            tc = float(row['t_refined_s'] or row['t_tap_s'])
            if not z['t'][0] <= tc <= z['t'][-1]:
                raise ValueError('Contact outside pose time range; check rally/file')
            scores = {}
            for name, fragments in features.items():
                local, base = [], []
                for t, reach in fragments:
                    v = reach[(t >= tc - .25) & (t <= tc + .10)]
                    local.extend(v[np.isfinite(v)].tolist())
                    base.extend(reach[np.isfinite(reach)].tolist())
                if len(local) >= 3 and len(base) >= 10:
                    scores[name] = float(np.percentile(local, 90) / max(np.median(base), 1e-6))
            proposed = max(scores, key=scores.get) if len(scores) == 4 else None
            events.append(dict(shot=int(row['shot_index']), t_s=tc,
                               labeled_hitter=row['hitter_name'], source=row['source'],
                               proposed_hitter=proposed, scores=scores,
                               agrees=proposed == row['hitter_name']))
    if not events:
        raise ValueError('No labeled contacts for this rally')
    events.sort(key=lambda e: e['t_s'])
    report = dict(rally=rally, scope='Development diagnostic at human contact times; not contact detection or held-out evaluation',
                  identity_source='Existing human track_assign labels; persistence needs visual verification',
                  method='Arm reach: local p90 / whole-track median; window -0.25 to +0.10 seconds; no side restriction',
                  provenance={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (npz, state, contacts)},
                  tracks=tracks, contacts=events, agrees=sum(e['agrees'] for e in events), total=len(events))
    return z, names, report


def render(video, out, z, names, report):
    import cv2
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or [h, w] != z['hw'].tolist() or abs(fps - float(z['fps'][0])) > .01:
            raise ValueError('Video size/FPS differs from saved poses')
        start, end = max(0, int(np.floor(z['t'][0] * fps))), int(np.floor(z['t'][-1] * fps))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h + 100))
        if not writer.isOpened():
            raise ValueError('Cannot open MP4 writer')
        times = np.unique(z['t'])
        colors = [(80, 220, 255), (255, 180, 70), (180, 255, 100), (255, 100, 220)]
        palette = {tid: colors[i % 4] for i, tid in enumerate(sorted(names))}
        for frame in range(end + 1):
            ok, img = cap.read()
            if not ok:
                raise ValueError('Video ended before pose window')
            if frame < start:
                continue
            t = frame / fps
            j = np.searchsorted(times, t)
            candidates = [i for i in (j - 1, j) if 0 <= i < len(times)]
            pt = times[min(candidates, key=lambda i: abs(times[i] - t))]
            # Poses use an offset resampling grid; never imply exact native-frame alignment.
            if abs(pt - t) <= .51 / fps:
                for i in np.flatnonzero(z['t'] == pt):
                    tid = int(z['track'][i]); col = palette.get(tid, (150, 150, 150))
                    x0, y0, x1, y1 = np.rint(z['box'][i]).astype(int)
                    cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
                    cv2.putText(img, f'{tid}: {names.get(tid, "Unassigned")}',
                                (max(0, x0), max(18, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .46, col, 1)
                    for key in (9, 10, 15, 16):
                        if z['kpc'][i, key] > .3:
                            xy = tuple(np.rint(z['kpt'][i, key]).astype(int))
                            cv2.circle(img, xy, 5, col, 1)
            canvas = np.zeros((h + 100, w, 3), np.uint8); canvas[:h] = img
            event = min(report['contacts'], key=lambda e: abs(e['t_s'] - t))
            lines = [f'R{report["rally"]} | {t:.3f}s | human track names; nearest saved pose (within half a frame)']
            if abs(event['t_s'] - t) <= .25:
                lines += [f'Contact {event["shot"]} HUMAN LABEL: {event["labeled_hitter"]}',
                          f'ARM-REACH PROPOSAL: {event["proposed_hitter"] or "Unknown"} | {"agrees" if event["agrees"] else "REVIEW"}']
            else:
                lines += ['Hollow wrist/ankle markers | grey = unassigned track | no automatic contact detection']
            for j, line in enumerate(lines):
                cv2.putText(canvas, line, (10, h + 24 + j * 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
            writer.write(canvas)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz', required=True, type=Path)
    p.add_argument('--video', type=Path, help='Optional: render review MP4 without rerunning pose inference')
    p.add_argument('--rally', type=int, default=1)
    p.add_argument('--state', type=Path, default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--contacts', type=Path, default=Path('data/vision/contact_labels_chicago0725.csv'))
    p.add_argument('--out', type=Path, default=Path('data/vision/player_hitter_review_r1'))
    a = p.parse_args()
    z, names, report = build(a.npz, a.state, a.contacts, a.rally)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2))
    if a.video:
        render(a.video, a.out / 'review.mp4', z, names, report)
    print(f'{report["agrees"]}/{report["total"]} agree with labeled hitters. Development diagnostic only.')
    print('Wrote', a.out)


if __name__ == '__main__':
    main()
