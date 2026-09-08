"""Small, deliberately in-sample temporal ball localization experiment.

No existing tracker or historical gates are changed. Run --help for commands.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_labels(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty labels')
    seen = set()
    for r in rows:
        if r['schema'] != 'ball-label-v2.1':
            raise ValueError('Use verified v2.1 labels; v2.0 recovery requires visual review')
        frame = int(r['source_frame'])
        if frame != int(r['displayed_frame']):
            raise ValueError('Requested/decoded frame mismatch')
        key = (r['video_name'], frame)
        if key in seen:
            raise ValueError('Duplicate decoded frame')
        seen.add(key)
        if r['visibility'] not in ('V', 'S', 'I', 'N'):
            raise ValueError('Invalid visibility')
        if r['visibility'] != 'N':
            if not all(math.isfinite(float(r[k])) for k in ('x', 'y')):
                raise ValueError('Nonfinite coordinate')
        r['frame'] = frame
        # User clarification applies only to this identified source/rally/run.
        absent = (r['video_name'] == 'full_match.mp4.webm' and r['rally'] == '18'
                  and 40 <= int(r['sample_index']) <= 51 and 27414 <= frame <= 27447)
        r['presence'] = ('absent' if absent else 'unknown') if r['visibility'] == 'N' else ('visible' if r['visibility'] in ('V', 'S') else 'occluded_inferred')
    if len({(r['video_name'], r['fps'], r['rally']) for r in rows}) != 1:
        raise ValueError('Supply one video and rally per run')
    return sorted(rows, key=lambda r: r['frame'])


def prepare(args):
    import cv2
    import numpy as np
    rows = read_labels(args.labels)
    video = Path(args.video)
    if video.name != rows[0]['video_name']:
        raise ValueError('Video filename differs from label source')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if abs(fps - float(rows[0]['fps'])) > .01:
            raise ValueError('Video/label FPS mismatch')
        needed = {r['frame'] + d for r in rows for d in range(-2, 3)}
        if min(needed) < 0:
            raise ValueError('Context extends before video')
        frames = {}
        # Sequential decoding from zero avoids approximate compressed-video seeks.
        for i in range(max(needed) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f'Video ended before required frame {i}')
            if i in needed:
                frames[i] = cv2.cvtColor(cv2.resize(bgr, (640, 360)), cv2.COLOR_BGR2RGB)
            if i % 6000 == 0:
                print(f'Decoded {i}/{max(needed)}', flush=True)
        for r in rows:
            if r['visibility'] != 'N' and not (0 <= float(r['x']) < width and 0 <= float(r['y']) < height):
                raise ValueError('Coordinate outside source image')
        ordered = sorted(frames)
        np.savez_compressed(out / 'frames.npz', ids=np.array(ordered), rgb=np.stack([frames[i] for i in ordered]))
        meta = dict(rows=rows, source_width=width, source_height=height,
                    input_width=640, input_height=360, fps=fps,
                    video=str(video.resolve()), labels_sha256=hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
                    scope='One-rally in-sample debugging; not held-out evaluation',
                    context_offsets=[-2, -1, 0, 1, 2])
        (out / 'manifest.json').write_text(json.dumps(meta, indent=2))
        print(f'Prepared {len(rows)} labels and {len(frames)} unique context frames in {out}')
    finally:
        cap.release()


def train(args):
    import random
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    data = Path(args.data)
    meta = json.loads((data / 'manifest.json').read_text())
    archive = np.load(data / 'frames.npz')
    frames = dict(zip(archive['ids'].tolist(), archive['rgb']))
    rows = [r for r in meta['rows'] if r['visibility'] in ('V', 'S')][:args.limit]
    if not rows:
        raise ValueError('No V/S targets')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # Full-image spatial softmax avoids the all-background MSE solution.
    model = nn.Sequential(nn.Conv2d(15, 24, 5, stride=2, padding=2), nn.ReLU(),
                          nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 1, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    def batch(selected):
        x = np.stack([np.concatenate([frames[r['frame'] + d] for d in range(-2, 3)], axis=2).transpose(2, 0, 1) for r in selected])
        return torch.tensor(x, dtype=torch.float32, device=device) / 255
    yy, xx = torch.meshgrid(torch.arange(90, device=device), torch.arange(160, device=device), indexing='ij')
    targets = []
    for r in rows:
        cx = float(r['x']) / meta['source_width'] * 160
        cy = float(r['y']) / meta['source_height'] * 90
        sigma = 1.5 if r['visibility'] == 'S' else .75
        g = torch.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma*sigma))
        targets.append(g.flatten() / g.sum())
    targets = torch.stack(targets)
    print(f'{device}: memorizing {len(rows)} V/S samples; I/N excluded', flush=True)
    history = []
    for step in range(args.steps):
        indices = random.sample(range(len(rows)), min(args.batch_size, len(rows)))
        logits = model(batch([rows[i] for i in indices])).flatten(1)
        loss = -(targets[indices] * F.log_softmax(logits, dim=1)).sum(1).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            history.append(dict(step=step+1, loss=float(loss.item())))
            print(history[-1], flush=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for r in rows:
            peak = int(model(batch([r])).flatten().argmax().item())
            x = peak % 160 / 160 * meta['source_width']
            y = peak // 160 / 90 * meta['source_height']
            error = math.hypot(x-float(r['x']), y-float(r['y']))
            predictions.append(dict(frame=r['frame'], visibility=r['visibility'], sampling_reason=r['sampling_reason'],
                                    target_x=float(r['x']), target_y=float(r['y']), predicted_x=x, predicted_y=y, error_px=error))
    errors = np.array([p['error_px'] for p in predictions])
    report = dict(scope=meta['scope'], device=device, seed=args.seed, steps=args.steps, samples=len(rows),
                  median_error_px=float(np.median(errors)), p90_error_px=float(np.percentile(errors, 90)),
                  within_px={str(k):float((errors<=k).mean()) for k in (5, 10, 20)}, history=history)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    torch.save(dict(state_dict=model.state_dict(), manifest=meta, args=vars(args)), out / 'model.pt')
    print(json.dumps({k:v for k,v in report.items() if k != 'history'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--video', required=True)
    prep.add_argument('--labels', required=True)
    prep.add_argument('--out', required=True)
    tr = subs.add_parser('train')
    tr.add_argument('--data', required=True)
    tr.add_argument('--out', required=True)
    tr.add_argument('--steps', type=int, default=300)
    tr.add_argument('--limit', type=int, default=32)
    tr.add_argument('--batch-size', type=int, default=4)
    tr.add_argument('--lr', type=float, default=.001)
    tr.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if args.command == 'train' and min(args.steps, args.limit, args.batch_size, args.lr) <= 0:
        p.error('Training parameters must be positive')
    (prepare if args.command == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
