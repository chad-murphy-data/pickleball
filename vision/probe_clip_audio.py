"""Audio postmortem, run against the same probed clip WITH the rally windows
known frame-exactly (RUN 2026-08-10). Findings, in order of importance:

  * THE TRANSIENTS EXIST. The spectrogram of the fast exchange (t 45.5-50.5)
    shows unmistakable full-height broadband stripes at shot cadence. Paddle
    pops DO survive this broadcast mix. Both earlier "detector failed"
    verdicts were about the DETECTOR and its validation, not the signal.
  * Standalone detection is still the wrong architecture, for two measured
    reasons: (a) APPLAUSE IS ALSO IMPULSIVE — the between-rally gap is
    acoustically hotter than the rallies in the high bands (6-12k flux
    ratio 0.52), so density-vs-deadtime validation can never pass; and
    (b) during a fast exchange real transients arrive every ~0.3-0.5 s, so
    any +/-0.25 s window "contains a transient" — my z-score control came
    back 52% positive at RANDOM in-rally moments, which is saturation, not
    a null. A statistic that cannot even define its null is not a detector.
  * The correct role for audio: TIMING REFINEMENT. Vision localises a
    contact to +/-2-3 frames; the largest broadband stripe within that
    +/-100 ms is the contact to ~3 ms. Search problem, not detection.
"""
import csv
import subprocess

import numpy as np

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
import os, sys
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/vision/clip.mp4"
S = os.path.dirname(os.path.abspath(CLIP)) or "."
# Rally windows IN CLIP TIME for the probed clip (from the scorebug-flip
# anchor). Edit for a different clip.
R29, GAP, R30 = (3.07, 29.07), (29.07, 34.07), (34.07, 58.07)

# ---- 1. ball-candidate density in vs out of rallies -------------------
cands = [(int(r["frame"]), float(r["x"]), float(r["y"]), int(r["area"]))
         for r in csv.DictReader(open(f"{S}/ball_candidates.csv"))]
t = np.array([c[0] / 30.0 for c in cands])


def dens(win):
    n = np.sum((t >= win[0]) & (t < win[1]))
    return n / (win[1] - win[0])


print("ball-candidate density (candidates/s):")
print(f"   rally 29 {dens(R29):5.1f}   gap {dens(GAP):5.1f}   "
      f"rally 30 {dens(R30):5.1f}   pre/post {np.sum((t<3.07)|(t>=58.07))/5.0:5.1f}")

# ---- 2. audio ---------------------------------------------------------
cmd = [FF, "-v", "error", "-i", CLIP, "-f", "s16le",
       "-ac", "1", "-ar", "44100", "-"]
raw = subprocess.run(cmd, capture_output=True).stdout
x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
sr = 44100
print(f"\naudio: {len(x)/sr:.1f}s")

# spectrogram, full band
nfft, hop = 1024, 256
win = np.hanning(nfft).astype(np.float32)
nfr = 1 + (len(x) - nfft) // hop
idx = np.arange(nfr)[:, None] * hop + np.arange(nfft)[None, :]
mag = np.abs(np.fft.rfft(x[idx] * win, axis=1)).astype(np.float32)
freqs = np.fft.rfftfreq(nfft, 1 / sr)
tt = (np.arange(nfr) * hop + nfft / 2) / sr

# flux in a few bands
def flux(flo, fhi):
    kb = (freqs >= flo) & (freqs < fhi)
    d = np.diff(mag[:, kb], axis=0)
    f = np.maximum(d, 0).sum(axis=1)
    return np.concatenate([[0], f])

bands = {"0.2-1k": (200, 1000), "1-3k": (1000, 3000),
         "3-6k": (3000, 6000), "6-12k": (6000, 12000)}
fl = {k: flux(*v) for k, v in bands.items()}

def wmask(w):
    return (tt >= w[0]) & (tt < w[1])

print("\nspectral-flux p95 by band, in-rally vs gap "
      "(ratio >1 means rallies are acoustically hotter):")
for k, f in fl.items():
    inr = np.percentile(f[wmask(R29) | wmask(R30)], 95)
    gap = np.percentile(f[wmask(GAP)], 95)
    print(f"   {k:7s} in {inr:8.1f}   gap {gap:8.1f}   ratio {inr/max(gap,1e-9):5.2f}")

# ---- 3. transients AT known contact moments ---------------------------
# fast ball tracks start/end where a shot launched/received the ball.
byf = {}
for c in cands:
    byf.setdefault(c[0], []).append(c)
# rebuild quick tracks same as probe_ball (nearest within 45px/frame gap<=3)
tracks, open_tr = [], []
for i in range(1800):
    cur = byf.get(i, [])
    used, nxt = set(), []
    for tr in open_tr:
        li, lx, ly = tr[-1][0], tr[-1][1], tr[-1][2]
        if i - li > 3:
            tracks.append(tr)
            continue
        best, bj = 45.0 * (i - li), None
        for j, c in enumerate(cur):
            if j not in used:
                d = ((c[1] - lx) ** 2 + (c[2] - ly) ** 2) ** 0.5
                if d < best:
                    best, bj = d, j
        if bj is not None:
            used.add(bj)
            tr.append(cur[bj])
        nxt.append(tr)
    for j, c in enumerate(cur):
        if j not in used:
            nxt.append([c])
    open_tr = nxt
tracks.extend(open_tr)
fast = []
for tr in tracks:
    if len(tr) < 10:
        continue
    xs = np.array([c[1] for c in tr])
    ys = np.array([c[2] for c in tr])
    sp = np.hypot(np.diff(xs), np.diff(ys)).mean() * 30
    if sp > 80:
        fast.append((tr[0][0] / 30.0, tr[-1][0] / 30.0, sp, len(tr)))
print(f"\nfast tracks (>=10 frames, >80 px/s): {len(fast)}")
events = sorted({round(e, 2) for a, b, _, _ in fast for e in (a, b)
                 if R29[0] < e < R30[1]})
print(f"contact-adjacent moments (track starts/ends): {events}")

# z-score of full-band flux at those moments vs the rally background
full = flux(200, 12000)
rmask = wmask(R29) | wmask(R30)
mu, sd = np.median(full[rmask]), np.percentile(full[rmask], 75) - np.median(full[rmask])
print("\nfull-band flux z at contact-adjacent moments (max within +/-0.25s):")
hits = 0
for e in events:
    m = (tt >= e - 0.25) & (tt <= e + 0.25)
    z = (full[m].max() - mu) / max(sd, 1e-9)
    hits += z > 4
    print(f"   t={e:6.2f}s   z={z:6.1f}")
print(f"-> {hits}/{len(events)} moments show a z>4 transient")

# control: same statistic at random in-rally moments away from events
rng = np.random.default_rng(7)
ctrl = []
for _ in range(200):
    e = rng.uniform(*R30)
    if all(abs(e - v) > 0.6 for v in events):
        m = (tt >= e - 0.25) & (tt <= e + 0.25)
        ctrl.append((full[m].max() - mu) / max(sd, 1e-9))
print(f"control (random in-rally moments): median z {np.median(ctrl):.1f}, "
      f"p90 {np.percentile(ctrl,90):.1f}, share z>4: {np.mean(np.array(ctrl)>4):.0%}")

# ---- 4. spectrogram image around the fastest track --------------------
a, b = 45.5, 50.5
m = (tt >= a) & (tt <= b)
kb = freqs <= 12000
im = 20 * np.log10(mag[m][:, kb].T + 1e-6)
im -= im.max()
im = np.clip((im + 60) / 60 * 255, 0, 255)
im = im[::-1]                                   # low freq at bottom
Ht, Wt = im.shape
scale = max(1, 640 // Wt)
im2 = np.repeat(np.repeat(im, 1, 0), scale, 1)
Hh, Ww = im2.shape
p = subprocess.run([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                    "gray", "-s", f"{Ww}x{Hh}", "-i", "-", "-frames:v", "1",
                    f"{S}/spectrogram_45_50.png"],
                   input=im2.astype(np.uint8).tobytes())
print(f"\nwrote spectrogram_45_50.png ({Ww}x{Hh}), window {a}-{b}s")
