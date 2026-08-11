"""Does a LEARNED ball detector actually fix the wall?  CPU test, no GPU.

`vision/ball_recall.py` established that the colour detector fails at
DETECTION, not association: the ball is in roughly 20-25% of rally frames
and simply absent from the rest, so no amount of linking can help.  The
proposed fix is a TrackNet-class model, which eats three consecutive frames
and emits a heat map — learning the ball from MOTION plus appearance rather
than colour, which is what survives motion blur and a pale court.

Before renting anything, test the claim.  The model is small enough to run
on CPU: a 26 s rally is minutes, not hours.

THE POINT OF THIS SCRIPT
    It writes the SAME CSV schema as `track_match.py` — frame, t_s, x_img,
    y_img, area, strength, x_ft, y_ft — so its output drops into
    `ball_recall.py` and `shots.py` with no changes at all.  The comparison
    is then apples to apples on a yardstick that already exists and that
    was built before anyone knew which detector would win.

WEIGHTS
    Tennis, not pickleball: an unofficial PyTorch TrackNet trained on
    broadcast tennis (github.com/yastrebksv/TrackNet, weights ~43 MB).
    Ball size, speed regime and camera geometry are close enough that
    transfer is plausible; it is not guaranteed, and this script is exactly
    how we find out.  Model definition and weights are fetched on first run
    into `raw/tracknet/` (gitignored) rather than vendored, so authorship
    stays with the original repo.

IF TRANSFER IS POOR, LABELS ARE ALREADY FREE
    The colour detector's net-crossing tracks are HIGH PRECISION even
    though they are low recall — 9,101 ball positions from one Chicago
    match, obtained with zero manual annotation.  That is a fine-tuning set
    for the domain gap, and it costs nothing but the compute already spent.

    python vision/tracknet_probe.py --video clip.mp4 --court court.json \\
        --out pfx --t0 34 --t1 58
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import court as C                                            # noqa: E402

CACHE = ROOT / "raw" / "tracknet"
MODEL_URL = "https://raw.githubusercontent.com/yastrebksv/TrackNet/main/model.py"
WEIGHTS_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
W, H = 640, 360                        # the network's fixed input size


def ensure_assets():
    CACHE.mkdir(parents=True, exist_ok=True)
    mp = CACHE / "model.py"
    wp = CACHE / "weights.pt"
    if not mp.exists():
        urllib.request.urlretrieve(MODEL_URL, mp)
    if not wp.exists():
        import gdown

        gdown.download(id=WEIGHTS_ID, output=str(wp), quiet=True)
    return mp, wp


def load_model(device="cpu"):
    import importlib.util

    import torch

    mp, wp = ensure_assets()
    spec = importlib.util.spec_from_file_location("tracknet_model", mp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.BallTrackerNet()
    net.load_state_dict(torch.load(wp, map_location=device))
    net.to(device).eval()
    return net


def postprocess(prob, thresh=0.5, min_area=2, max_area=200):
    """Heat map -> one ball position, or None.

    The reference implementation runs a Hough circle transform and then
    discards any frame returning more than one circle.  Connected
    components are simpler, need no OpenCV, and keep the strongest blob
    instead of throwing the frame away — which matters here because the
    whole question is how often the ball is found at all.
    """
    m = prob > thresh
    if not m.any():
        return None
    lab, n = ndimage.label(m)
    if not n:
        return None
    areas = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
    peaks = ndimage.maximum(prob, lab, range(1, n + 1))
    best, bi = -1.0, None
    for i in range(n):
        if not (min_area <= areas[i] <= max_area):
            continue
        if peaks[i] > best:
            best, bi = peaks[i], i + 1
    if bi is None:
        return None
    cy, cx = ndimage.center_of_mass(m, lab, bi)
    return float(cx), float(cy), int(areas[bi - 1]), float(best)


def decode(path, w, h):
    cmd = [C.FF, "-v", "error", "-i", str(path), "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-vf", f"scale={w}:{h}", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    n = w * h * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.kill()


def run(video, court_json, out, t0=None, t1=None, thresh=0.5, device="cpu",
        progress=200):
    import json

    import torch

    d = json.load(open(court_json))
    w0, h0, fps, dur = C.probe(video)
    # the court fit may have been made at a different resolution
    Hm = np.diag([W / d["w"], H / d["h"], 1.0]) @ np.array(d["H_court_to_img"])
    Hi = np.linalg.inv(Hm)

    net = load_model(device)
    f0 = int((t0 or 0) * fps)
    f1 = int(t1 * fps) if t1 else None

    rows = []
    buf = []
    t_start = time.time()
    n_done = n_hit = 0
    for i, fr in enumerate(decode(video, W, H)):
        buf.append(fr)
        if len(buf) > 3:
            buf.pop(0)
        if len(buf) < 3 or i < f0:
            continue
        if f1 and i > f1:
            break
        # Channel order matches the reference: current, prev, prev-prev —
        # and BGR, not RGB.  The reference reads frames with OpenCV, which
        # is BGR, so the weights expect it.  Measured on 48 mid-rally
        # frames: BGR 46% detections, RGB 17%.  Feeding RGB does not fail
        # loudly, it just quietly makes the model three times worse.
        x = np.concatenate([b[..., ::-1] for b in (buf[2], buf[1], buf[0])],
                           axis=2)
        x = np.ascontiguousarray(np.rollaxis(x.astype(np.float32) / 255.0,
                                             2, 0))[None]
        with torch.no_grad():
            o = net(torch.from_numpy(x).float().to(device))
        # The head is a 256-way per-pixel classifier over heat-map
        # intensity, not a binary mask, so the predicted intensity is the
        # argmax class — same as the reference implementation.
        prob = (o.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.float32)
                / 255.0)
        r = postprocess(prob, thresh=thresh)
        n_done += 1
        if r:
            n_hit += 1
            cx, cy, area, peak = r
            ft = C.apply_H(Hi, np.array([[cx, cy]]))[0]
            rows.append([i, f"{i/fps:.3f}", f"{cx:.1f}", f"{cy:.1f}", area,
                         f"{peak:.3f}", f"{ft[0]:.2f}", f"{ft[1]:.2f}"])
        if progress and n_done % progress == 0:
            el = time.time() - t_start
            print(f"   {n_done} frames  {n_hit} hits ({100*n_hit/n_done:.0f}%)"
                  f"  {n_done/el:.1f} fps  eta "
                  f"{((f1 or i) - i)/max(n_done/el,1e-9)/60:.1f} min",
                  flush=True)

    p = f"{out}_ball.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "x_img", "y_img", "area", "strength",
                    "x_ft", "y_ft"])
        w.writerows(rows)
    el = time.time() - t_start
    print(f"\nTrackNet: {n_hit}/{n_done} frames with a ball "
          f"({100*n_hit/max(n_done,1):.1f}%), {el/60:.1f} min "
          f"({n_done/max(el,1e-9):.1f} fps on {device})")
    print(f"wrote {p}")
    return {"frames": n_done, "hits": n_hit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    run(a.video, a.court, a.out, t0=a.t0, t1=a.t1, thresh=a.thresh,
        device=a.device)


if __name__ == "__main__":
    main()
