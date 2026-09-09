"""Render saved temporal-model predictions, without optimization or retraining."""
import argparse
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from ball_temporal_baseline import build_model


def distinct_candidates(heatmap, width, height, count=5, separation=16):
    """Rank local maxima; suppress nearby peaks in original-video pixels."""
    import numpy as np
    a = np.asarray(heatmap)
    if a.ndim != 2 or not np.isfinite(a).all() or count < 1 or separation < 0:
        raise ValueError('Invalid heatmap or candidate settings')
    h, w = a.shape
    pad = np.pad(a, 1, constant_values=-np.inf)
    maxima = np.ones(a.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            maxima &= a >= pad[dy:dy+h, dx:dx+w]
    indices = np.flatnonzero(maxima)
    indices = indices[np.argsort(-a.flat[indices], kind='stable')]
    selected = []
    for i in indices:
        x, y = (int(i) % w)/w*width, (int(i)//w)/h*height
        if any(math.hypot(x-c['x'], y-c['y']) <= separation for c in selected):
            continue
        selected.append(dict(rank=len(selected)+1, x=x, y=y, peak_mass=float(a.flat[i])))
        if len(selected) == count:
            break
    return selected


def review_windows(rally, specs, fps):
    result = {}
    for i, spec in enumerate(specs, 1):
        lo, hi = map(float, spec.split(':'))
        if not all(math.isfinite(t) for t in (lo,hi)) or lo < 2/fps or hi <= lo:
            raise ValueError('Review ranges must be increasing finite source-video seconds')
        result[f'{rally}_window{i}'] = (math.ceil(lo*fps), math.floor(hi*fps))
    return result


def training_frames(checkpoint):
    selected = [r for r in checkpoint['manifest']['rows'] if r['visibility'] in ('V', 'S')]
    selected = selected[:checkpoint['args']['limit']]
    targets = {int(r['frame']) for r in selected}
    context = {f + d for f in targets for d in range(-2, 3)}
    return targets, context


def exposure(frame, targets, context):
    if frame in targets:
        return 'training_target'
    if frame in context:
        return 'training_context_only'
    return 'unseen_center_same_video'


def windows_for(rallies, manifest, contacts_path, pre_seconds=1.0, post_seconds=1.0):
    if not all(math.isfinite(x) and x >= 0 for x in (pre_seconds, post_seconds)):
        raise ValueError('Window padding must be finite and nonnegative')
    rows = manifest['rows']
    trained_rallies = {int(r['rally']) for r in rows}
    fps = manifest['fps']
    result = {}
    for rally in rallies:
        if rally in trained_rallies:
            rally_rows=[r for r in rows if int(r['rally'])==rally]
            result[rally] = (min(int(r['frame']) for r in rally_rows), max(int(r['frame']) for r in rally_rows))
        else:
            with Path(contacts_path).open(newline='') as f:
                contacts = [float(r.get('t_refined_s') or r['t_tap_s']) for r in csv.DictReader(f)
                            if int(r['rally_cum']) == rally and r.get('contact', '1') != '0'
                            and r.get('source') in ('manual', 'divergent', 'prefill')]
            if not contacts:
                raise ValueError(f'No accepted contact timestamps for rally {rally}')
            # This is a contact-centered review window, not a verified rally-end boundary.
            result[rally] = (math.ceil((min(contacts)-pre_seconds)*fps), math.floor((max(contacts)+post_seconds)*fps))
    if any(a < 2 or b < a for a, b in result.values()):
        raise ValueError('Invalid window or insufficient preceding context')
    return result


def human_fields(row):
    if row is None:
        return dict(visibility='', presence='', target_x='', target_y='')
    vis = row['visibility']
    presence = row.get('presence', 'unknown' if vis == 'N' else 'visible' if vis in ('V', 'S') else 'occluded_inferred')
    return dict(visibility=vis, presence=presence,
                target_x=row['x'] if vis != 'N' else '', target_y=row['y'] if vis != 'N' else '')


def raw_predictions(stream, first, last, predict):
    """A single decode pass; predict only when all five ordered frames exist."""
    ring = deque(maxlen=5)
    completed = 0
    for index, frame in enumerate(stream):
        if index < first-2:
            continue
        ring.append(frame)
        center = index-2
        if len(ring) == 5 and first <= center <= last:
            yield center, ring[2], predict(list(ring), center)
            completed += 1
        if index >= last+2:
            break
    if completed != last-first+1:
        raise ValueError('Video ended before the complete requested context was decoded')


def run(args):
    import cv2
    import numpy as np
    import torch
    if not shutil.which('ffmpeg'):
        raise ValueError('ffmpeg is required to write a QuickTime-compatible H.264 MP4')
    video = Path(args.video)
    # Our checkpoint contains only tensors and primitive metadata. No unrestricted pickle load.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = checkpoint['manifest']
    if video.name != manifest['rows'][0]['video_name']:
        raise ValueError('This command expects the same source video as training')
    if (manifest['input_width'], manifest['input_height'], manifest['context_offsets']) != (640, 360, [-2,-1,0,1,2]):
        raise ValueError('Unsupported model input configuration')
    windows = (review_windows(args.rallies[0], args.review_seconds, manifest['fps']) if args.review_seconds
               else windows_for(args.rallies, manifest, args.contacts, args.pre_seconds, args.post_seconds))
    targets, context = training_frames(checkpoint)
    labels = {int(r['frame']): r for r in manifest['rows']}
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    cap = cv2.VideoCapture(str(video))
    encoders, handles = {}, []
    success = False
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open source video')
        width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (width, height) != (manifest['source_width'], manifest['source_height']) or abs(fps-manifest['fps']) > .01:
            raise ValueError('Source geometry/FPS differs from checkpoint')
        if width % 2 or height % 2:
            raise ValueError('H.264 overlay requires even source dimensions')
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        fields = ['frame', 't_s', 'exposure', 'context_overlaps_training', 'predicted_x', 'predicted_y',
                  'peak_mass', 'visibility', 'presence', 'target_x', 'target_y', 'error_px']
        writers, candidate_writers, counts, errors = {}, {}, {}, {}
        for rally in windows:
            folder = out / f'r{rally}'
            folder.mkdir()
            log = (folder/'encoder.log').open('w')
            handle = (folder/'predictions.csv').open('w', newline='')
            handles.extend([handle, log])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writers[rally] = writer
            candidate_file = (folder/'candidates.csv').open('w', newline='')
            handles.append(candidate_file)
            candidate_writer = csv.DictWriter(candidate_file, fieldnames=['frame','t_s','rank','x','y','peak_mass'])
            candidate_writer.writeheader()
            candidate_writers[rally] = candidate_writer
            counts[rally] = dict(frames=0, training_target=0, training_context_only=0, unseen_center_same_video=0,
                                 context_overlaps_training=0, explicitly_absent_labels=0, unknown_labels=0)
            errors[rally] = []
            encoders[rally] = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{width}x{height+100}', '-r', str(fps*args.playback_speed), '-i', '-', '-an',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(folder/'overlay.mp4')], stdin=subprocess.PIPE, stderr=log)
        print(f'{device}: loaded checkpoint; no retraining. Windows: {windows}', flush=True)

        def stream():
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    return
                if i % 6000 == 0:
                    print(f'Decoded {i} source frames', flush=True)
                i += 1
                yield bgr

        def predict(frames, center):
            if not any(a <= center <= b for a,b in windows.values()):
                return None
            rgb = [cv2.cvtColor(cv2.resize(f, (640,360)), cv2.COLOR_BGR2RGB) for f in frames]
            x = np.concatenate(rgb, axis=2).transpose(2,0,1)[None]
            with torch.inference_mode():
                logits = model(torch.tensor(x, dtype=torch.float32, device=device)/255).flatten()
                if not bool(torch.isfinite(logits).all().item()):
                    raise ValueError('Nonfinite prediction')
                peak = int(logits.argmax().item())
                mass = float(torch.softmax(logits, dim=0)[peak].item())
                heatmap = torch.softmax(logits, dim=0).reshape(90,160).cpu().numpy()
            candidates = distinct_candidates(heatmap, width, height, args.candidates, args.candidate_separation)
            return peak % 160 / 160 * width, peak // 160 / 90 * height, mass, candidates

        first, last = min(a for a,b in windows.values()), max(b for a,b in windows.values())
        for center, frame, prediction in raw_predictions(stream(), first, last, predict):
            if prediction is None:
                continue
            x,y,mass,candidates = prediction
            human = human_fields(labels.get(center))
            category = exposure(center, targets, context)
            overlap = bool(set(range(center-2, center+3)) & context)
            error = math.hypot(x-float(human['target_x']), y-float(human['target_y'])) if human['visibility'] in ('V','S') else ''
            for rally,(a,b) in windows.items():
                if not a <= center <= b:
                    continue
                writers[rally].writerow(dict(frame=center, t_s=center/fps, exposure=category,
                    context_overlaps_training=int(overlap), predicted_x=x, predicted_y=y,
                    peak_mass=mass, error_px=error, **human))
                for candidate in candidates:
                    candidate_writers[rally].writerow(dict(frame=center, t_s=center/fps, **candidate))
                counts[rally]['frames'] += 1
                counts[rally][category] += 1
                counts[rally]['context_overlaps_training'] += int(overlap)
                counts[rally]['explicitly_absent_labels'] += int(human['presence']=='absent')
                counts[rally]['unknown_labels'] += int(human['visibility']=='N' and human['presence']=='unknown')
                if error != '':
                    errors[rally].append(error)
                canvas = np.zeros((height+100,width,3), dtype=np.uint8)
                canvas[:height] = frame
                # Keep the ball pixels unobscured: no center dot, cross, fill, or label.
                cv2.circle(canvas, (round(x),round(y)), 18, (255,0,255), 2, cv2.LINE_AA)
                for candidate in candidates[1:]:
                    cx, cy = round(candidate['x']), round(candidate['y'])
                    cv2.circle(canvas, (cx,cy), 18, (255,255,0), 1, cv2.LINE_AA)
                    # Put ranks outside the clear center; keep edge labels inside the image.
                    tx, ty = min(width-20, max(0,cx+21)), min(height-5, max(15,cy-21))
                    cv2.putText(canvas, str(candidate['rank']), (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,0), 1, cv2.LINE_AA)
                if human['target_x'] != '':
                    color = (0,220,0) if human['visibility'] in ('V','S') else (0,165,255)
                    cv2.circle(canvas, (round(float(human['target_x'])),round(float(human['target_y']))), 12, color, 2, cv2.LINE_AA)
                text = [f'R{rally}  frame {center}  {center/fps:.3f}s  {category}',
                        f'Magenta: rank 1  Cyan: alternatives  Green: V/S  Orange: inferred  Human: {human["presence"] or "unlabeled"}',
                        f'Peak mass {mass:.4f} is NOT presence confidence. Model always predicts a position.']
                for j,line in enumerate(text):
                    cv2.putText(canvas, line, (10,height+25+j*30), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
                encoders[rally].stdin.write(canvas.tobytes())
            if (center-first) % 120 == 0:
                print(f'Predicted through source frame {center}/{last}', flush=True)
        for rally, proc in encoders.items():
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f'Encoding failed; see r{rally}/encoder.log')
        reports = {}
        for rally,(a,b) in windows.items():
            e = errors[rally]
            reports[str(rally)] = dict(start_frame=a, end_frame=b, **counts[rally],
                labeled_VS_count=len(e), median_error_px=float(np.median(e)) if e else None,
                max_error_px=max(e) if e else None,
                scope='Same-video qualitative review; no cross-video generalization claim')
        report = dict(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                      device=device, fps=fps, windows=reports,
                      candidates=args.candidates, candidate_separation_px=args.candidate_separation,
                      playback_speed=args.playback_speed, review_seconds=args.review_seconds,
                      contact_window_padding=dict(pre_seconds=args.pre_seconds, post_seconds=args.post_seconds),
                      notes=['Sequential CFR zero-origin decode matches preparation assumptions; visual alignment still needs checking.',
                             'Unknown and inferred labels are excluded from localization metrics.',
                             'Absent labels describe annotated samples only; no gap interpolation.',
                             'Unlabeled rallies have no accuracy score. Contact windows use the recorded padding, not verified point boundaries.',
                             'Five-frame input includes two future frames. MP4 has no audio.'])
        (out/'report.json').write_text(json.dumps(report, indent=2))
        print(f'Wrote overlays, per-frame CSVs and report.json to {out}', flush=True)
        success = True
    finally:
        cap.release()
        for proc in encoders.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for handle in handles:
            handle.close()
        if not success:
            print('Overlay did not complete. Any new output folder contains partial results; use a new --out on retry.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rallies', nargs='+', type=int, default=[18,19])
    p.add_argument('--pre-seconds', type=float, default=1.0, help='Padding before first contact for non-training rallies')
    p.add_argument('--post-seconds', type=float, default=1.0, help='Padding after last contact for non-training rallies')
    p.add_argument('--contacts', default=str(Path(__file__).resolve().parent.parent/'data/vision/contact_labels_chicago0725.csv'))
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    p.add_argument('--candidates', type=int, default=1, help='Number of spatially distinct heatmap peaks, at most 10')
    p.add_argument('--candidate-separation', type=float, default=16, help='Peak suppression radius in source pixels')
    p.add_argument('--review-seconds', nargs='+', help='Source-video start:end ranges, with exactly one --rallies value')
    p.add_argument('--playback-speed', type=float, default=1, help='0.25 makes quarter-speed video; source timestamps are unchanged')
    args = p.parse_args()
    if not 1 <= args.candidates <= 10 or not math.isfinite(args.candidate_separation) or args.candidate_separation < 0:
        p.error('Use 1–10 candidates and finite, nonnegative separation')
    if not math.isfinite(args.playback_speed) or not 0 < args.playback_speed <= 1:
        p.error('Playback speed must be greater than zero and at most one')
    if args.review_seconds and len(args.rallies) != 1:
        p.error('--review-seconds requires exactly one rally')
    run(args)


if __name__ == '__main__':
    main()
