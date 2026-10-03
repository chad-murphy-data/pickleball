"""Offline candidate-path experiment on existing diagnostic exports. No model fitting."""
import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


@dataclass
class Settings:
    motion_scale: float = 32.0  # source pixels per 1/60 second
    strength_weight: float = 0.5
    gap_cost: float = 3.0
    gap_switch_cost: float = 0.75

    def validate(self):
        for value in asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Selector settings must be finite and positive')


def select_path(frames, candidates, fps, settings):
    """Exact first-order dynamic programming, with an explicit null state.

    No velocity extrapolation, acceleration penalty, or imputed coordinates.
    Direction changes incur only displacement cost. Gaps reset association.
    These are heuristic costs, not calibrated probabilities of ball presence.
    """
    settings.validate()
    if not math.isfinite(fps) or fps <= 0 or not frames:
        raise ValueError('Need frames and positive FPS')
    if any(b <= a for a,b in zip(frames,frames[1:])):
        raise ValueError('Frames must be strictly increasing')
    states = []
    for frame in frames:
        cs = candidates.get(frame, [])
        for c in cs:
            if not all(math.isfinite(c[k]) for k in ('x','y','peak_mass')) or not 0 <= c['peak_mass'] <= 1:
                raise ValueError('Invalid candidate coordinate or mass')
        states.append(cs + [None])
    costs, back = [], []
    for i, current in enumerate(states):
        new_costs, parents = [], []
        for c in current:
            emission = settings.gap_cost if c is None else -settings.strength_weight * math.log(max(c['peak_mass'], 1e-12))
            if i == 0:
                new_costs.append(emission)
                parents.append(-1)
                continue
            scale = settings.motion_scale * (frames[i]-frames[i-1]) * 60 / fps
            options = []
            for j, previous in enumerate(states[i-1]):
                if c is None and previous is None:
                    transition = 0
                elif c is None or previous is None:
                    transition = settings.gap_switch_cost
                else:
                    transition = (math.hypot(c['x']-previous['x'], c['y']-previous['y'])/scale)**2
                options.append(costs[j] + transition)
            parent = min(range(len(options)), key=options.__getitem__)
            new_costs.append(emission + options[parent])
            parents.append(parent)
        costs = new_costs
        back.append(parents)
    index = min(range(len(costs)), key=costs.__getitem__)
    total_cost = costs[index]
    result = []
    for i in range(len(states)-1, -1, -1):
        result.append(states[i][index])
        index = back[i][index]
    return list(reversed(result)), total_cost


def load_candidates(path, first, last):
    candidates = {}
    keys = set()
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            frame, rank = int(row['frame']), int(row['rank'])
            if not first <= frame <= last or rank < 1 or (frame,rank) in keys:
                raise ValueError('Invalid or duplicate candidate frame/rank')
            keys.add((frame,rank))
            candidates.setdefault(frame,[]).append(dict(rank=rank, **{k:float(row[k]) for k in ('x','y','peak_mass')}))
    for cs in candidates.values():
        cs.sort(key=lambda c:c['rank'])
    return candidates


def render(source, destination, selections, fps):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(source))
    proc = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {source}')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        with (destination.parent/'encoder.log').open('w') as log:
            proc = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24',
                '-s',f'{width}x{height+70}','-r',str(fps),'-i','-','-an','-c:v','libx264',
                '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],
                stdin=subprocess.PIPE, stderr=log)
            for i, chosen in enumerate(selections):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f'Review MP4 ended early at frame offset {i}')
                canvas = np.zeros((height+70,width,3), dtype=np.uint8)
                canvas[:height] = bgr
                if chosen is not None:
                    cv2.circle(canvas,(round(chosen['x']),round(chosen['y'])),28,(0,255,255),2,cv2.LINE_AA)
                status = 'GAP - no candidate selected (not proof of absence)' if chosen is None else f'Selected candidate rank {chosen["rank"]}'
                for j,line in enumerate(['Yellow outer ring: selected path. Magenta: raw winner. Cyan: alternatives.',status]):
                    cv2.putText(canvas,line,(10,height+25+30*j),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                proc.stdin.write(canvas.tobytes())
            if cap.read()[0]:
                raise ValueError('Review MP4 has more frames than its report')
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('Encoding failed; see encoder.log')
    finally:
        cap.release()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()


def run(args):
    settings = Settings(args.motion_scale,args.strength_weight,args.gap_cost,args.gap_switch_cost)
    settings.validate()
    review = Path(args.review)
    report = json.loads((review/'report.json').read_text())
    fps, playback = float(report['fps']), float(report.get('playback_speed',1))
    if not math.isfinite(fps*playback) or min(fps,playback) <= 0:
        raise ValueError('Invalid report FPS/playback speed')
    if not args.csv_only and not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg is required for rendering')
    jobs = []
    for key, window in report['windows'].items():
        if not re.fullmatch(r'\d+(?:_window\d+)?',key):
            raise ValueError('Unexpected review window key')
        first,last = int(window['start_frame']),int(window['end_frame'])
        frames = list(range(first,last+1))
        candidates = load_candidates(review/f'r{key}'/'candidates.csv',first,last)
        selected,total_cost = select_path(frames,candidates,fps,settings)
        jobs.append((key,frames,candidates,selected,total_cost))
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    summaries = {}
    for key,frames,candidates,selected,total_cost in jobs:
        folder = out/f'r{key}'
        folder.mkdir()
        rows = []
        for frame,c in zip(frames,selected):
            raw = next((x for x in candidates.get(frame,[]) if x['rank']==1),None)
            rows.append(dict(frame=frame,t_s=frame/fps,status='gap' if c is None else 'candidate_selected',
                selected_rank='' if c is None else c['rank'],selected_x='' if c is None else c['x'],
                selected_y='' if c is None else c['y'],selected_mass='' if c is None else c['peak_mass'],
                raw_x='' if raw is None else raw['x'],raw_y='' if raw is None else raw['y']))
        with (folder/'selected.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summaries[key] = dict(frames=len(frames),gap_frames=sum(c is None for c in selected),
            rank1_frames=sum(c is not None and c['rank']==1 for c in selected),
            alternative_frames=sum(c is not None and c['rank']!=1 for c in selected),path_cost=total_cost)
        if not args.csv_only:
            render(review/f'r{key}'/'overlay.mp4',folder/'selected.mp4',selected,fps*playback)
        print(key,summaries[key],flush=True)
    output = dict(settings=asdict(settings),source_report=report,windows=summaries,
        notes=['Offline full-window lookahead, not a live tracker.',
               'Gap costs are heuristic; gap does not establish ball absence.',
               'No velocity extrapolation or interpolation. Gaps reset continuity.',
               'Persistent distractors can form cheap paths. Rank changes are not accuracy gains.',
               'Each review window is independent. Short-window edges lack outside context.'])
    (out/'report.json').write_text(json.dumps(output,indent=2))
    print(f'Wrote {out}; inputs and checkpoint unchanged',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--motion-scale',type=float,default=32)
    p.add_argument('--strength-weight',type=float,default=.5)
    p.add_argument('--gap-cost',type=float,default=3)
    p.add_argument('--gap-switch-cost',type=float,default=.75)
    p.add_argument('--csv-only',action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
