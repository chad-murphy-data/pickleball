"""Gate B probe: extract SWING events and AUDIO POPS from the broadcast.

The hypothesis (model/vision_adjudication.md §Gate B): a paddle contact is
recoverable from PLAYER motion — a wrist-speed burst on a pose skeleton —
gated on a coincident audio pop, with the ball never tracked at all.
People are person-scale (the branch measured that player detection works
whenever players are in frame), pretrained pose is in-domain for humans,
and swings + pops have complementary failure modes (vision-weak = small
fast counters, audio-weak = soft dink pops).

WHAT THIS SCRIPT DELIBERATELY IS NOT
    It is not the pipeline.  It is the measurement that decides whether
    the pipeline gets built.  So it stays permissive and dumb on purpose:
    it emits EVERY wrist-speed local maximum above a low floor and EVERY
    audio-flux peak above a low floor, with magnitudes, and leaves all
    thresholding to vision/swing_score.py — which selects the operating
    point on LABEL-FREE criteria (side alternation + contact rate) so the
    hand labels are only ever touched once, for the final score.
    Attribution here is SIDE-level (near/far by image geometry): that is
    what the gate needs (alternation + sequence alignment).  Naming the
    player within a team is lineup.py's job and is already measured at
    99.25% from referee logs alone — it does not need re-proving here.

RUN (laptop or GPU box; ~40% of the VOD is decoded, rally windows only)
    pip install ultralytics imageio-ffmpeg
    git checkout claude/vision-branch-accounts-e1s6yk
    python vision/swing_probe.py --video full_match.mp4 --smoke   # 2 rallies + a debug frame
    python vision/swing_probe.py --video full_match.mp4           # the real run
    # CPU overnight: add --fast   (yolov8n-pose, 640px, 20 fps)
    # GPU (~20-30 min on a T4/RTX4000): defaults are already sized for it

    Outputs (small CSVs, safe to commit):
      data/vision/swing_probe_swings.csv   one row per wrist-speed peak
      data/vision/swing_probe_pops.csv     one row per audio onset peak
      data/vision/swing_probe_meta.json    params + runtime, for the record

    Resumable: already-finished rallies are skipped on rerun.

SELF-TEST (no torch, no video — runs anywhere, including CI)
    python vision/swing_probe.py --selftest
    Plants swings in synthetic keypoint streams and pops in synthetic
    audio flux, and asserts the SAME peak-extraction code paths recover
    them.  A detector that cannot recover a planted signal cannot be
    trusted to report a real one (house pattern: audio_contacts
    --selftest, gap_exploit --inject).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = ROOT / "data/vision/rally_windows_chicago0725.csv"

# COCO-17 keypoint indices (ultralytics pose head)
L_WRIST, R_WRIST = 9, 10

# ---------------------------------------------------------------- ffmpeg


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"                     # hope it's on PATH


def decode_window(video, t0, dur, fps, width):
    """Yield BGR frames for one rally window. -ss before -i = fast seek."""
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{max(0, t0):.3f}",
           "-i", str(video), "-t", f"{dur:.3f}",
           "-f", "rawvideo", "-pix_fmt", "bgr24",       # BGR: the TrackNet
           "-vf", f"scale={width}:-2", "-r", str(fps), "-"]  # RGB bug, once
    probe = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{max(0, t0):.3f}",
         "-i", str(video), "-t", "0.04", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-vf", f"scale={width}:-2", "-frames:v", "1", "-"],
        capture_output=True)
    if not probe.stdout:
        raise SystemExit(f"ffmpeg produced no frames at t={t0:.1f}s — "
                         f"wrong file? ({probe.stderr.decode()[:200]})")
    h = len(probe.stdout) // (width * 3)
    n = width * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n * 4)
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, width, 3)
    finally:
        p.kill()


def decode_audio(video, sr=32000):
    """Mono float32 track for the whole file, one pass."""
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(video), "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, np.float32), sr


# ------------------------------------------------------- pose -> swings


def side_split(bottoms):
    """near/far by 1-D 2-means on box-bottom y. Returns threshold."""
    b = np.sort(np.asarray(bottoms, float))
    if len(b) < 2:
        return None
    best, thr = -1.0, None
    for i in range(1, len(b)):
        lo, hi = b[:i], b[i:]
        gap = hi.min() - lo.max()
        if gap > best:
            best, thr = gap, (lo.max() + hi.min()) / 2
    return thr


class SlotTracker:
    """Four slots: (near|far) x (left|right). Per frame, persons are
    assigned side by box-bottom y and left/right by center x within side.
    Crossovers within a team swap left/right — harmless at side level,
    which is all the gate scores."""

    def __init__(self, fps):
        self.fps = fps
        self.hist = {}          # slot -> (t, wrist_xy, box_h)
        self.rows = []          # emitted per-frame wrist speeds

    def feed(self, t, persons):
        """persons: list of dicts {bottom, cx, box_h, wrists (2,2), kpc}."""
        if not persons:
            return
        thr = side_split([p["bottom"] for p in persons])
        for p in persons:
            p["side"] = "near" if (thr is None or p["bottom"] >= thr) else "far"
        for side in ("near", "far"):
            grp = sorted([p for p in persons if p["side"] == side],
                         key=lambda p: -p["box_h"])[:2]
            grp.sort(key=lambda p: p["cx"])
            for lr, p in zip(("L", "R"), grp):
                slot = f"{side}{lr}"
                prev = self.hist.get(slot)
                v = np.nan
                if prev is not None and t - prev[0] <= 2.5 / self.fps:
                    dt = t - prev[0]
                    d = np.linalg.norm(p["wrists"] - prev[1], axis=1)
                    v = float(np.nanmax(d)) / max(dt * self.fps, 1e-9) \
                        / max(p["box_h"], 1e-9)       # box-heights per frame
                self.hist[slot] = (t, p["wrists"], p["box_h"])
                self.rows.append((t, slot, v, p["box_h"], p["kpc"]))


def strongest_first(cands, refractory):
    """Peak suppression keeping the STRONGEST within each refractory
    window, not the first — a first-wins refractory lets a small noise
    bump just before a real peak eat it (found by the selftest: a z=1.6
    ripple 24 ms ahead of a z=300 planted pop suppressed it)."""
    keep = []
    for c in sorted(cands, key=lambda x: -x[1]):
        if all(abs(c[0] - k[0]) >= refractory for k in keep):
            keep.append(c)
    keep.sort()
    return keep


def speed_peaks(rows, fps, floor=0.06, refractory=0.30):
    """Local maxima of the per-slot wrist-speed series above a LOW floor.
    Real thresholding happens in the scorer; this just drops noise."""
    out = []
    by = {}
    for t, slot, v, bh, kc in rows:
        by.setdefault(slot, []).append((t, v, bh, kc))
    for slot, seq in by.items():
        seq.sort()
        t = np.array([x[0] for x in seq])
        v = np.array([np.nan_to_num(x[1]) for x in seq])
        if len(v) > 2:                       # 3-point smooth
            v = np.convolve(v, [0.25, 0.5, 0.25], mode="same")
        cands = [(t[i], float(v[i]), seq[i][2], seq[i][3])
                 for i in range(1, len(v) - 1)
                 if v[i] >= floor and v[i] >= v[i - 1] and v[i] >= v[i + 1]]
        out.extend((tt, slot, vv, bh, kc)
                   for tt, vv, bh, kc in strongest_first(cands, refractory))
    out.sort()
    return out


def court_halfwidth(y_frac):
    """Allowed |cx - 0.5W| as a fraction of W, at a person's feet row.

    The court is a trapezoid in the image, so a FIXED x-gate cannot both
    reject the line officials (who stand at mid-frame rows, cx ~0.06 and
    ~0.96 in the Chicago smoke frame) and keep a server hugging the
    sideline at the near baseline, where the court spans nearly the full
    frame.  Piecewise-linear band, calibrated on the smoke frame: narrow
    where the far court is (sidelines ~0.34-0.66 there), the measured
    ~0.16-0.83 at the near kitchen row, wide open at the frame bottom —
    officials never stand down there, and that is exactly where wide
    baseline stances live."""
    pts = [(0.34, 0.24), (0.60, 0.36), (0.85, 0.50)]
    if y_frac <= pts[0][0]:
        return pts[0][1]
    if y_frac >= pts[-1][0]:
        return pts[-1][1]
    for (ya, wa), (yb, wb) in zip(pts, pts[1:]):
        if ya <= y_frac <= yb:
            f = (y_frac - ya) / (yb - ya)
            return wa + f * (wb - wa)
    return pts[-1][1]


def keep_person(box, kps, kpc, H, W):
    """Court players, not crowd or officials: feet in the lower ~2/3, box
    tall enough, wrists actually seen, and center-x inside the
    perspective-shaped court band (see court_halfwidth)."""
    x0, y0, x1, y1 = box
    bh = y1 - y0
    cx = (x0 + x1) / 2
    if y1 < 0.34 * H or bh < 0.07 * H:
        return None
    if abs(cx / W - 0.5) > court_halfwidth(y1 / H):
        return None
    wr = kps[[L_WRIST, R_WRIST]]
    wc = kpc[[L_WRIST, R_WRIST]]
    if np.nanmax(wc) < 0.25:
        return None
    return {"bottom": float(y1), "cx": float(cx),
            "box_h": float(bh), "wrists": wr, "kpc": float(np.nanmax(wc))}


# ---------------------------------------------------------- audio pops


def pop_peaks(audio, sr, floor_z=1.5, refractory=0.055,
              band=(2000.0, 9000.0), nfft=1024, hop=256):
    """Band-limited spectral-flux onsets, z-scored against a rolling
    median — the retuned recipe from the POC (above the commentary band,
    high frequencies, short refractory)."""
    if len(audio) < nfft * 4:
        return []
    win = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    nf = (len(audio) - nfft) // hop
    prev = None
    flux = np.zeros(nf, np.float32)
    for i in range(nf):
        seg = audio[i * hop:i * hop + nfft] * win
        mag = np.abs(np.fft.rfft(seg))[sel]
        if prev is not None:
            flux[i] = np.sum(np.maximum(0, mag - prev))
        prev = mag
    t = (np.arange(nf) * hop + nfft / 2) / sr
    # rolling median/MAD z-score, 20 s window
    w = int(20.0 * sr / hop)
    z = np.zeros_like(flux)
    for i in range(nf):
        lo, hi = max(0, i - w // 2), min(nf, i + w // 2)
        med = np.median(flux[lo:hi])
        mad = np.median(np.abs(flux[lo:hi] - med)) + 1e-9
        z[i] = (flux[i] - med) / (1.4826 * mad)
    cands = [(float(t[i]), float(z[i])) for i in range(1, nf - 1)
             if z[i] >= floor_z and z[i] >= z[i - 1] and z[i] >= z[i + 1]]
    return strongest_first(cands, refractory)


# ------------------------------------------------------------- windows


def load_windows(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"cum": int(r["rally_cum"]), "game": int(r["game"]),
                     "t0": float(r["t0s"]), "t1": float(r["t1s"]),
                     "core": r["core"] == "1"})
    rows.sort(key=lambda x: x["cum"])
    return rows


def rally_of(t, windows, pad=1.5):
    for w in windows:
        if w["t0"] - pad <= t <= w["t1"] + pad:
            return w["cum"]
    return None


# ------------------------------------------------------------ the run


def run(a):
    from ultralytics import YOLO          # lazy: --selftest needs no torch

    windows = load_windows(a.windows)
    if a.smoke:
        windows = [w for w in windows if w["core"]][:2]
    elif a.rallies:
        want = {int(x) for x in a.rallies.split(",")}
        windows = [w for w in windows if w["cum"] in want]

    out_sw = Path(f"{a.out}_swings.csv")
    out_pp = Path(f"{a.out}_pops.csv")
    done = set()
    if out_sw.exists() and not a.smoke:
        done = {int(r["rally_cum"]) for r in csv.DictReader(open(out_sw))}
        print(f"resume: {len(done)} rallies already extracted")

    model = YOLO(a.model)
    t_start = time.time()

    # ---- audio, one pass over the whole file --------------------------
    if not out_pp.exists() or a.smoke:
        print("audio pass...", flush=True)
        audio, sr = decode_audio(a.video)
        pops = pop_peaks(audio, sr)
        with open(out_pp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_video", "rally_cum", "z"])
            n = 0
            for t, z in pops:
                cum = rally_of(t, windows if a.smoke else load_windows(a.windows))
                if cum is not None:
                    w.writerow([f"{t:.3f}", cum, f"{z:.2f}"])
                    n += 1
        print(f"  {n} in-window pops -> {out_pp}")

    # ---- video, rally windows only ------------------------------------
    new = not out_sw.exists()
    fh = open(out_sw, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["t_video", "rally_cum", "slot", "v_boxh_per_frame",
                     "box_h_px", "kp_conf"])
    todo = [w for w in windows if w["cum"] not in done]
    for k, wnd in enumerate(todo):
        pad0, pad1 = 2.0, 1.5
        trk = SlotTracker(a.fps)
        n_frames = 0
        for i, frame in enumerate(decode_window(
                a.video, wnd["t0"] - pad0, (wnd["t1"] - wnd["t0"]) + pad0 + pad1,
                a.fps, a.width)):
            t = wnd["t0"] - pad0 + i / a.fps
            H, W = frame.shape[:2]
            res = model(frame, imgsz=a.imgsz, verbose=False,
                        device=(a.device or None), conf=0.35)[0]
            persons = []
            dbg_rows = [] if (a.smoke and i == int(a.fps * 3)) else None
            if res.keypoints is not None and len(res.boxes):
                kxy = res.keypoints.xy.cpu().numpy()
                kcf = (res.keypoints.conf.cpu().numpy()
                       if res.keypoints.conf is not None
                       else np.ones(kxy.shape[:2], np.float32))
                for box, kps, kpc in zip(
                        res.boxes.xyxy.cpu().numpy(), kxy, kcf):
                    p = keep_person(box, kps, kpc, H, W)
                    if p:
                        persons.append(p)
                    if dbg_rows is not None:
                        dbg_rows.append((box, kps, p))
            trk.feed(t, persons)
            n_frames += 1
            if dbg_rows is not None:
                # green = fed to the tracker, red = rejected by the filter:
                # the picture must show the filter's decision, not the raw
                # model output, or officials look like a problem after
                # they've already been solved.
                png = Path(f"{a.out}_debugframe_r{wnd['cum']}.png")
                try:
                    import cv2
                    dbg = frame.copy()
                    # draw the gate's boundary so the frame shows exactly
                    # who would be cut where
                    for yy in range(int(0.34 * H), H - 1, 6):
                        hw = court_halfwidth(yy / H)
                        for xx in (int((0.5 - hw) * W), int((0.5 + hw) * W)):
                            if 0 <= xx < W:
                                dbg[yy:yy + 3, max(0, xx - 1):xx + 1] = \
                                    (0, 220, 220)
                    for box, kps, p in dbg_rows:
                        x0, y0, x1, y1 = map(int, box[:4])
                        ok = p is not None
                        cv2.rectangle(dbg, (x0, y0), (x1, y1),
                                      (0, 200, 0) if ok else (0, 0, 230),
                                      2 if ok else 1)
                        if ok:
                            for wx, wy in kps[[L_WRIST, R_WRIST]]:
                                cv2.circle(dbg, (int(wx), int(wy)), 5,
                                           (0, 165, 255), -1)
                    cv2.imwrite(str(png), dbg)
                    print(f"  debug frame -> {png} "
                          f"(green=tracked, red=rejected)")
                except Exception:
                    pass
        peaks = speed_peaks(trk.rows, a.fps)
        for t, slot, v, bh, kc in peaks:
            wr.writerow([f"{t:.3f}", wnd["cum"], slot, f"{v:.4f}",
                         f"{bh:.0f}", f"{kc:.2f}"])
        fh.flush()
        el = time.time() - t_start
        rate = (k + 1) / el
        print(f"rally #{wnd['cum']:>3} ({k + 1}/{len(todo)})  "
              f"{n_frames} frames  {len(peaks)} swing peaks  "
              f"eta {(len(todo) - k - 1) / max(rate, 1e-9) / 60:.0f} min",
              flush=True)
    fh.close()

    meta = {"video": str(a.video), "model": a.model, "imgsz": a.imgsz,
            "fps": a.fps, "width": a.width, "device": a.device or "auto",
            "rallies": len(todo), "runtime_s": round(time.time() - t_start, 1),
            "smoke": a.smoke}
    Path(f"{a.out}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {(time.time() - t_start) / 60:.1f} min. "
          f"Next: python vision/swing_score.py")


# ------------------------------------------------------------ selftest


def selftest():
    """Plant swings + pops in synthetic streams; the SAME extraction code
    must recover them."""
    rng = np.random.default_rng(7)
    fps = 30.0
    planted = [3.0, 5.5, 8.2, 12.0]

    # --- swings: 4 slots of keypoint jitter, bursts on nearL ------------
    trk = SlotTracker(fps)
    for i in range(int(15 * fps)):
        t = i / fps
        persons = []
        for side, bottom, bh in (("near", 600, 260), ("far", 380, 130)):
            for lr, cx in (("L", 400), ("R", 900)):
                wr = np.array([[cx, bottom - bh * 0.5],
                               [cx + 20, bottom - bh * 0.5]], float)
                wr += rng.normal(0, bh * 0.006, wr.shape)      # idle jitter
                if side == "near" and lr == "L":
                    for tp in planted:
                        if abs(t - tp) < 0.10:                 # 6-frame burst
                            wr[0, 0] += bh * 0.5 * np.sin((t - tp) * 30)
                persons.append({"bottom": bottom + rng.normal(0, 2),
                                "cx": cx, "box_h": bh, "wrists": wr,
                                "kpc": 0.9})
        trk.feed(t, persons)
    peaks = speed_peaks(trk.rows, fps)
    strong = [(t, s, v) for t, s, v, *_ in peaks if v > 0.12]
    hits = sum(any(abs(t - tp) < 0.25 for t, s, v in strong
                   if s == "nearL") for tp in planted)
    junk = [x for x in strong if x[1] != "nearL"]
    print(f"swings: {hits}/{len(planted)} planted recovered on the right "
          f"slot; {len(junk)} strong peaks elsewhere")
    assert hits == len(planted), "planted swings not recovered"
    assert len(junk) <= 1, "idle jitter produced strong phantom swings"

    # --- pops: noise + planted broadband clicks ------------------------
    sr = 32000
    audio = rng.normal(0, 0.02, sr * 15).astype(np.float32)
    for tp in planted:
        i = int(tp * sr)
        click = (rng.normal(0, 1.0, 96) * np.hanning(96)).astype(np.float32)
        audio[i:i + 96] += click
    pops = pop_peaks(audio, sr)
    strongp = [(t, z) for t, z in pops if z > 4]
    hitsp = sum(any(abs(t - tp) < 0.05 for t, z in strongp) for tp in planted)
    print(f"pops:   {hitsp}/{len(planted)} planted recovered; "
          f"{len(strongp)} strong onsets total")
    assert hitsp == len(planted), "planted pops not recovered"
    assert len(strongp) <= len(planted) + 2, "noise floor produced phantom pops"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--windows", default=str(WINDOWS))
    ap.add_argument("--out", default=str(ROOT / "data/vision/swing_probe"))
    ap.add_argument("--model", default="yolov8s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--device", default="")
    ap.add_argument("--rallies", help="comma-separated rally_cum subset")
    ap.add_argument("--smoke", action="store_true",
                    help="2 core rallies + a pose debug frame, then stop")
    ap.add_argument("--fast", action="store_true",
                    help="CPU preset: yolov8n-pose, imgsz 640, 20 fps")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.video:
        ap.error("--video is required (or use --selftest)")
    if a.fast:
        a.model, a.imgsz, a.fps = "yolov8n-pose.pt", 640, 20.0
    run(a)


if __name__ == "__main__":
    main()
