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


def windows_for(rallies, manifest, contacts_path):
    rows = manifest['rows']
    trained_rally = int(rows[0]['rally'])
    fps = manifest['fps']
    result = {}
    for rally in rallies:
        if rally == trained_rally:
            result[rally] = (min(int(r['frame']) for r in rows), max(int(r['frame']) for r in rows))
        else:
            with Path(contacts_path).open(newline='') as f:
                contacts = [float(r.get('t_refined_s') or r['t_tap_s']) for r in csv.DictReader(f)
                            if int(r['rally_cum']) == rally and r.get('contact', '1') != '0'
                            and r.get('source') in ('manual', 'divergent', 'prefill')]
            if not contacts:
                raise ValueError(f'No accepted contact timestamps for rally {rally}')
            # This is a contact-centered review window, not a verified rally-end boundary.
            result[rally] = (math.ceil((min(contacts)-1)*fps), math.floor((max(contacts)+1)*fps))
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
    windows = windows_for(args.rallies, manifest, args.contacts)
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
        writers, counts, errors = {}, {}, {}
        for rally in windows:
            folder = out / f'r{rally}'
            folder.mkdir()
            log = (folder/'encoder.log').open('w')
            handle = (folder/'predictions.csv').open('w', newline='')
            handles.extend([handle, log])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writers[rally] = writer
            counts[rally] = dict(frames=0, training_target=0, training_context_only=0, unseen_center_same_video=0,
                                 context_overlaps_training=0, explicitly_absent_labels=0, unknown_labels=0)
            errors[rally] = []
            encoders[rally] = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{width}x{height+100}', '-r', str(fps), '-i', '-', '-an',
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
            return peak % 160 / 160 * width, peak // 160 / 90 * height, mass

        first, last = min(a for a,b in windows.values()), max(b for a,b in windows.values())
        for center, frame, prediction in raw_predictions(stream(), first, last, predict):
            if prediction is None:
                continue
            x,y,mass = prediction
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
                if human['target_x'] != '':
                    color = (0,220,0) if human['visibility'] in ('V','S') else (0,165,255)
                    cv2.circle(canvas, (round(float(human['target_x'])),round(float(human['target_y']))), 12, color, 2, cv2.LINE_AA)
                text = [f'R{rally}  frame {center}  {center/fps:.3f}s  {category}',
                        f'Magenta: prediction  Green: V/S label  Orange: inferred  Human: {human["presence"] or "unlabeled"}',
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
                      notes=['Sequential CFR zero-origin decode matches preparation assumptions; visual alignment still needs checking.',
                             'Unknown and inferred labels are excluded from localization metrics.',
                             'Absent labels describe annotated samples only; no gap interpolation.',
                             'Unlabeled rallies have no accuracy score. Review windows use contacts plus one second.',
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
    p.add_argument('--contacts', default=str(Path(__file__).resolve().parent.parent/'data/vision/contact_labels_chicago0725.csv'))
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    run(p.parse_args())


if __name__ == '__main__':
    main()
