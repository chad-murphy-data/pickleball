"""Pixels to feet: fit the court once, then everything downstream is physical.

Every measurement the earlier probes produced was in pixels, which made all
of them camera-specific and most of them uninterpretable.  "22 px of vertical
reversal" is not a fact about pickleball.  With a homography, the same
detections become feet and mph, the kitchen becomes a line rather than a
guessed pixel band, and out-of-bounds candidates can simply be discarded —
which is where most of the false positives were coming from.

THE FIT
    The camera is effectively fixed for a whole broadcast, so this is a
    once-per-video problem and can afford to be slow and careful.

    1. Median background over sampled frames — players and ball vanish.
    2. A LINE RESPONSE, not a white mask.  A court line is bright AND
       brighter than what sits a few pixels to either side of it.  Sponsor
       text and the white apron logos are bright but wide, so the two-sided
       test rejects them; this matters more than it sounds, because this
       broadcast paints DOORDASH across the apron in letters taller than the
       kitchen line is thick.
    3. Initialise the four outer corners from the playing-surface colour
       blob (the inner blue is lighter than the apron blue).
    4. Refine by direct search on the four corners, maximising how much line
       response the projected COURT MODEL lands on.  The model is all ten
       real segments — sidelines, baselines, both kitchen lines, both
       centrelines — so the fit is driven by the court's whole internal
       geometry and not just its outline.  A four-corner fit that is right
       on the outline but wrong in perspective scores badly, because the
       kitchen and centre lines fall in the wrong place.

    Quality is reported as the median distance, in FEET, from each projected
    model line to the nearest actual line pixel.  That number is the thing
    to quote, and it is what makes downstream claims auditable.

COORDINATES
    Court is 20 ft wide by 44 ft long, x in [0,20], y in [0,44], net at
    y=22, kitchen lines at y=15 and y=29.  x is measured from the left
    sideline as the camera sees it, y from the far baseline.

    python vision/court.py --video <mp4> [--out court.json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()

W_FT, L_FT = 20.0, 44.0
NET_Y, KITCHEN = 22.0, 7.0

# the ten segments a pickleball court actually paints
MODEL_LINES = [
    ((0, 0), (20, 0)),           # far baseline
    ((0, 44), (20, 44)),         # near baseline
    ((0, 0), (0, 44)),           # left sideline
    ((20, 0), (20, 44)),         # right sideline
    ((0, 15), (20, 15)),         # far kitchen line
    ((0, 29), (20, 29)),         # near kitchen line
    ((10, 0), (10, 15)),         # far centreline
    ((10, 29), (10, 44)),        # near centreline
]


# --------------------------------------------------------------------------
# video io
# --------------------------------------------------------------------------
def probe(path):
    out = subprocess.run([FF, "-i", str(path)], capture_output=True,
                         text=True).stderr
    import re

    m = re.search(r", (\d+)x(\d+)", out)
    f = re.search(r"([\d.]+) fps", out)
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    fps = float(f.group(1)) if f else 30.0
    dur = (int(d.group(1)) * 3600 + int(d.group(2)) * 60
           + float(d.group(3))) if d else None
    return w, h, fps, dur


def sample_frames(path, n=40, w=None, h=None, t0=0.0, t1=None):
    """n frames spread across the video, as float32 RGB."""
    w0, h0, fps, dur = probe(path)
    w, h = w or w0, h or h0
    t1 = t1 or (dur or 60.0)
    step = max((t1 - t0) / n, 0.5)
    out = []
    for i in range(n):
        t = t0 + i * step
        if t >= t1:
            break
        p = subprocess.run(
            [FF, "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-vf", f"scale={w}:{h}", "-"], capture_output=True)
        if len(p.stdout) == w * h * 3:
            out.append(np.frombuffer(p.stdout, np.uint8).reshape(h, w, 3))
    if not out:
        raise SystemExit("could not sample any frames")
    return np.stack(out)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def line_response(bg, width=4):
    """Bright, and brighter than the pixels `width` away on BOTH sides.

    The two-sided test is what separates a 3 px court line from a 30 px
    sponsor letter: the letter's interior is bright but its neighbours are
    bright too, so it scores zero.
    """
    g = bg.mean(axis=2).astype(np.float32)
    out = np.zeros_like(g)
    for ax in (0, 1):
        a = np.roll(g, width, axis=ax)
        b = np.roll(g, -width, axis=ax)
        r = np.minimum(g - a, g - b)          # darker neighbour dominates
        out = np.maximum(out, r)
    out[:width + 1, :] = out[-width - 1:, :] = 0
    out[:, :width + 1] = out[:, -width - 1:] = 0
    return np.clip(out, 0, None) * (g > 90)


def surface_mask(bg, tol=60.0, min_frac=0.008):
    """The playing surface, as a convex hull of its colour.

    Distance from the modal centre-of-frame colour, NOT a hand-set blue
    threshold, so it ports across venues.  Measured on MLP Chicago the
    surface is (96,145,236) and the surrounding apron is (52,77,124): far
    enough apart that the tolerance is not delicate.  An earlier version
    also accepted "any saturated colour" to catch the maroon kitchen and
    swallowed the entire apron with it — that clause is gone.

    The kitchen is recovered geometrically instead: a differently coloured
    kitchen SPLITS the blue into a far half and a near half, so the mask is
    the CONVEX HULL of every large blue component.  A perspective court is
    convex, so the hull is the court, whatever colour its middle is painted.
    """
    from scipy import ndimage
    from scipy.spatial import ConvexHull, Delaunay

    f = bg.astype(np.float32)
    h, w = f.shape[:2]
    c = f[int(h * .35):int(h * .75), int(w * .25):int(w * .75)]
    seed = np.median(c.reshape(-1, 3), axis=0)
    m = np.linalg.norm(f - seed, axis=2) < tol

    lab, n = ndimage.label(m)
    if n == 0:
        raise SystemExit("no court-coloured region found")
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
    keep = {i + 1 for i, s in enumerate(sizes) if s > min_frac * h * w}
    if not keep:
        keep = {int(np.argmax(sizes)) + 1}
    m = np.isin(lab, list(keep))

    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1)
    hull = ConvexHull(P)
    tri = Delaunay(P[hull.vertices])
    gy, gx = np.mgrid[0:h, 0:w]
    inside = tri.find_simplex(np.stack([gx.ravel(), gy.ravel()], 1)) >= 0
    return inside.reshape(h, w), seed


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def homography(src, dst):
    """DLT, least squares over >=4 correspondences."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_H(H, pts):
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))])
    q = p @ H.T
    return q[:, :2] / q[:, 2:3]


def init_corners(mask):
    """Extreme points of the surface blob, in image order TL TR BR BL.

    A perspective court is a convex quad, so its corners are the blob points
    that maximise the four diagonal directions.
    """
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(np.float32)
    out = []
    for vx, vy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        out.append(P[np.argmax(P[:, 0] * vx + P[:, 1] * vy)])
    return np.array(out, float)


def model_points(step=0.5):
    """Dense samples along every model line, in court feet."""
    pts = []
    for (x0, y0), (x1, y1) in MODEL_LINES:
        n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 2)
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], 1))
    return np.vstack(pts)


def peak_distance(resp, pct=95.0):
    """Distance (px) from every pixel to the nearest detected line pixel."""
    from scipy import ndimage

    pos = resp[resp > 0]
    thr = np.percentile(pos, pct) if pos.size else 1.0
    return ndimage.distance_transform_edt(~(resp > thr)), thr


def score(corners, dist, mpts, tol=2.0):
    """Fraction of the projected model that lands ON a detected line.

    Sharper than mean brightness, and that matters: with a soft objective
    the optimiser happily rescaled the court until its eight lines sat on
    eight unrelated bright apron edges.  Requiring each model point to be
    within `tol` px of an actual detected line makes those impostor fits
    score near zero, because sponsor bands do not reproduce the court's
    internal spacing.
    """
    H = homography([(0, 0), (W_FT, 0), (W_FT, L_FT), (0, L_FT)], corners)
    q = apply_H(H, mpts)
    h, w = dist.shape
    xi = np.round(q[:, 0]).astype(int)
    yi = np.round(q[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if ok.sum() < len(mpts) * 0.75:
        return -1.0
    hit = dist[yi[ok], xi[ok]] < tol
    return float(hit.sum()) / len(mpts)


def refine(corners, resp, iters=300, seed=0, leash=0.06):
    """Direct search on the 8 corner coordinates.

    `leash` caps how far a corner may travel from its colour-derived start,
    as a fraction of image diagonal.  The colour init is trustworthy in
    position and only roughly right in detail, so the search should polish
    it, never relocate it.
    """
    dist, _ = peak_distance(resp)
    mpts = model_points()
    c0 = corners.copy()
    diag = float(np.hypot(*resp.shape))
    lim = leash * diag
    best, bs = c0.copy(), score(c0, dist, mpts)
    rng = np.random.default_rng(seed)
    stepsz = 20.0
    while stepsz > 0.2:
        improved = False
        for _ in range(iters):
            cand = best.copy()
            i = rng.integers(0, 4)
            cand[i] += rng.normal(0, stepsz, 2)
            if np.linalg.norm(cand[i] - c0[i]) > lim:
                continue
            s = score(cand, dist, mpts)
            if s > bs:
                best, bs, improved = cand, s, True
        stepsz *= 0.8 if improved else 0.5
    return best, bs


def residual_ft(corners, resp, thresh=None):
    """Median distance from projected model lines to real line pixels, FEET.

    This is the honest quality number: it is measured against detected
    image evidence, not against the optimiser's own objective.
    """
    from scipy import ndimage

    H = homography([(0, 0), (W_FT, 0), (W_FT, L_FT), (0, L_FT)], corners)
    thresh = thresh or np.percentile(resp[resp > 0], 88) if (resp > 0).any() else 1
    peaks = resp > thresh
    # distance (px) from every pixel to the nearest line pixel
    dist = ndimage.distance_transform_edt(~peaks)
    mpts = model_points(0.25)
    q = apply_H(H, mpts)
    h, w = resp.shape
    xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
    dpx = dist[yi, xi]
    # convert px -> ft locally: 1 ft along x at each model point
    q2 = apply_H(H, mpts + np.array([1.0, 0.0]))
    scale = np.linalg.norm(q2 - q, axis=1)            # px per ft, varies
    good = scale > 1e-6
    return float(np.median(dpx[good] / scale[good])), float(
        np.mean((dpx[good] / scale[good]) < 0.5))


# --------------------------------------------------------------------------
def fit(video, n=40, t0=0.0, t1=None, out=None, overlay=None, size=None):
    w0, h0, fps, dur = probe(video)
    w, h = size or (w0, h0)
    frames = sample_frames(video, n=n, w=w, h=h, t0=t0, t1=t1)
    bg = np.median(frames, axis=0)
    mask, seed = surface_mask(bg)
    # Confine the line search to the court itself.  Every real court line is
    # inside the surface hull (the boundary lines sit on its edge, hence the
    # dilation); every sponsor logo that was luring the optimiser off-court
    # is outside it.  Without this the refinement reliably scored a skewed
    # quad higher than the correct one by parking model lines on apron text.
    from scipy import ndimage

    resp = line_response(bg) * ndimage.binary_dilation(mask, iterations=6)
    c0 = init_corners(mask)
    c1, s1 = refine(c0, resp)
    s0 = score(c0, peak_distance(resp)[0], model_points())
    med_ft, within = residual_ft(c1, resp)
    H = homography([(0, 0), (W_FT, 0), (W_FT, L_FT), (0, L_FT)], c1)
    res = {
        "video": str(video), "w": w, "h": h, "fps": fps, "duration_s": dur,
        "corners_img": c1.tolist(), "H_court_to_img": H.tolist(),
        "H_img_to_court": np.linalg.inv(H).tolist(),
        "score": s1, "score_init": s0, "residual_ft_median": med_ft,
        "frac_within_half_ft": within, "n_frames_sampled": len(frames),
    }
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    if overlay:
        draw_overlay(bg, c1, overlay)
    return res


def draw_overlay(bg, corners, path):
    H = homography([(0, 0), (W_FT, 0), (W_FT, L_FT), (0, L_FT)], corners)
    im = bg.astype(np.float32).copy()
    q = apply_H(H, model_points(0.08))
    h, w = im.shape[:2]
    xi = np.round(q[:, 0]).astype(int)
    yi = np.round(q[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            X = np.clip(xi[ok] + dx, 0, w - 1)
            Y = np.clip(yi[ok] + dy, 0, h - 1)
            im[Y, X] = (0, 255, 60)
    subprocess.run([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                    "rgb24", "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1",
                    str(path)],
                   input=np.clip(im, 0, 255).astype(np.uint8).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out")
    ap.add_argument("--overlay")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    a = ap.parse_args()
    size = (a.width, a.height) if a.width and a.height else None
    r = fit(a.video, n=a.frames, t0=a.t0, t1=a.t1, out=a.out,
            overlay=a.overlay, size=size)
    print(f"court fit on {Path(a.video).name}  ({r['w']}x{r['h']} @ {r['fps']}fps)")
    print(f"  corners  {[[round(v) for v in c] for c in r['corners_img']]}")
    print(f"  residual {r['residual_ft_median']:.2f} ft median   "
          f"{100*r['frac_within_half_ft']:.0f}% of model within 0.5 ft")
    if a.out:
        print(f"  wrote {a.out}")
    if a.overlay:
        print(f"  wrote {a.overlay}")


if __name__ == "__main__":
    main()
