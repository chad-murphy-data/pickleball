"""Court coverage per player — the model behind vision/coverage_spec.md.

Consumes the solved layers ONLY: pose tracks (pose_extract npz), court
homography (court.py json), and referee-log identity (lineup.py +
coverage_windows.py).  No ball, no contact detection, no training.

    python vision/coverage.py --scan-camera vod.mp4 --cam-out cam.csv
    python vision/coverage.py --run --pose-dir data/vision/pose_x \
        --court court.json --windows coverage_windows_x.csv \
        --lineup data/vision/lineup_<id8>.csv --cam cam.csv \
        --vod <vod-id> --event "<name>" --date YYYY-MM-DD
    python vision/coverage.py --selftest

PRE-REGISTERED DEFINITIONS (frozen 2026-08-16, before the first real
number was computed; the coverage_spec.md proposals made exact).  The
identity/hygiene INSTRUMENT may be tuned against the overlay
verification; these metric formulas may not be tuned after real
coverage numbers exist.

  * Foot point: ankle midpoint (COCO 15/16) when both ankles have
    confidence >= 0.3; one confident ankle if only one; else box
    bottom-center.
  * Rally-active span: [t_serve + 2.0 s, t1].  The 2 s serve phase is
    reported separately (ellipse only), never mixed into rally-phase
    metrics.  t_serve comes from the anchor finder (below); rallies
    where the finder fails are DROPPED from metrics (in start-marked
    logs the logged duration ~= the lead, so no arithmetic fallback can
    place the serve — ambiguous spans are dropped, never guessed).
  * Coverage area = 90% Gaussian occupancy ellipse of rally-phase foot
    positions: pi * 4.605 * sqrt(det Sigma) ft^2, detection-confidence-
    weighted mean/cov, one 3-sigma Mahalanobis trim pass then refit.
  * Width share (the w observable): per frame where both partners are
    observed, the partners' lateral midpoint m maps the court's 20 ft
    of width into the player's territory: share = m/20 if the player is
    left of the partner else (20-m)/20; time-averaged over the rally
    phase.  Partners sum to 1 by construction.  width_share_kitchen =
    the same restricted to frames where BOTH partners stand within
    4 ft of their kitchen line.  Secondary: width_share_area =
    own ellipse area / (own + partner's).
  * Depth = |court y - 22| (distance from the net, ft): p05/p95 over
    the rally phase; kitchen trips = transitions per rally from depth-
    from-own-kitchen > 10 ft to <= 4 ft.
  * Distance / speed: positions smoothed with a 0.5 s box filter
    first (raw 10 Hz foot noise would dominate — the selftest plants
    0.3 ft jitter on a static player and requires the static verdict);
    distance = smoothed path length per rally; static fraction =
    smoothed speed < 1.5 ft/s.
  * Speed gate: court-plane displacement over 23 ft/s (7 m/s) drops the
    later detection (keypoint teleports).
  * Identity: serve-anchor chain (spec step 5).  The serving end is the
    end whose pair stands DEEP at the serve (two-bounce rule keeps the
    serving pair back; the receiving end has exactly one player at the
    kitchen).  Receiver = the deep player on the receiving end; server
    = the serving-end player DIAGONAL to the receiver (cross-court rule
    — depth alone coin-flips under stacking), depth as tie-break.
    Report-only checks, never inputs: the diagonal margin, the lineup
    state machine's R/L halves (scored only where that machine's own
    receiver_ok agrees locally — PPA logs desync it in runs) and, in
    mixed, the taller-male height prior.  Names ride track ids; a named
    track that dies may hand its name to an unnamed track appearing
    within 1.2 s and 6 ft on the same side of the net (flagged,
    confidence x0.8; the dying track's earlier detections keep their
    name).  Rallies below confidence 0.5 are DROPPED from metrics,
    never guessed; every drop is counted with a reason.
  * DreamBreakers: excluded structurally — DBs are separate singles
    match_ids whose 2-player frames can never satisfy the 4-player
    serving configuration, so they land in the drop ledger even if a
    DB timeline is ever pointed at this.  (House rule: DBs never enter
    doubles models.)
  * Deferred, documented: the serve-vs-receive coverage cut (needs
    per-rally serving-team accumulators; add when the first real VOD
    shows the base metrics are sound).  Publication gates (split-half
    battery, min-matches, no single-match leaderboards) live in
    coverage_spec.md and apply DOWNSTREAM of these per-game rows —
    nothing this module emits is publishable by itself.

Outputs (committed): data/coverage_players.csv (one row per
player-game) + data/coverage_events.csv (per-VOD ledger).  Existing
rows for the same keys are replaced, so re-runs are idempotent.
Pose npz / overlay renders stay local (broadcast-imagery rule).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

W_FT, L_FT, NET_Y = 20.0, 44.0, 22.0
KITCHEN_Y = {"near": 29.0, "far": 15.0}       # near end owns y in [22,44]
BASELINE_Y = {"near": 44.0, "far": 0.0}

CONF_ANKLE = 0.3
SERVE_PHASE_S = 2.0
SPEED_GATE_FTS = 23.0          # 7 m/s
STATIC_FTS = 1.5
SMOOTH_S = 0.5
KITCHEN_BAND_FT = 4.0
ELLIPSE_CHI2 = 4.605           # chi^2_2 at 90%
DEEP_FT = 9.0                  # serving-end pair stands at least this deep
RCV_KITCHEN_FT = 4.0           # receiving end has a player this close
OFFCOURT_FT = 7.0              # projected foot beyond court+this = dropped
CONF_MIN = 0.5
AF_CONF = 0.80        # confidence stamped on anchor-free rallies: above
                      # CONF_MIN so they are kept, below a clean geometric
                      # anchor so they never outrank one
HANDOFF_DT, HANDOFF_FT = 1.2, 6.0
CAM_NCC_MAIN = 0.55
MIN_FRAMES_PLAYER_GAME = 150   # ~15 s of observation before a row is real


# ---------------------------------------------------------------- court


def load_court(path):
    c = json.loads(Path(path).read_text())
    return {"H": np.array(c["H_img_to_court"], float),
            "w": c["w"], "h": c["h"],
            "residual_ft": c.get("residual_ft_median"),
            "within_half_ft": c.get("frac_within_half_ft")}


def project(court, pts_px, frame_w, frame_h):
    """Pixel points (N,2) in a frame of (frame_w,frame_h) -> court feet.
    The homography was fit at court['w']x['h']; rescale first."""
    p = np.asarray(pts_px, float).reshape(-1, 2).copy()
    p[:, 0] *= court["w"] / frame_w
    p[:, 1] *= court["h"] / frame_h
    q = np.hstack([p, np.ones((len(p), 1))]) @ court["H"].T
    return q[:, :2] / q[:, 2:3]


def foot_point(box, kpt, kpc):
    """(x_px, y_px, source).  Ankles when confident, else box bottom."""
    la, ra = kpt[15], kpt[16]
    ok_l, ok_r = kpc[15] >= CONF_ANKLE, kpc[16] >= CONF_ANKLE
    if ok_l and ok_r:
        return (la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2, "ankles"
    if ok_l:
        return la[0], la[1], "ankle"
    if ok_r:
        return ra[0], ra[1], "ankle"
    return (box[0] + box[2]) / 2, box[3], "box"


# ---------------------------------------------------------- camera gate


def _stream_grey(video, fps, w, h):
    from scorebug_windows import ffmpeg_bin
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(video),
           "-vf", f"scale={w}:{h},fps={fps}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 64)
    n = w * h
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8)
    finally:
        p.kill()


def scan_camera(video, out_csv, fps=10.0, w=160, h=90):
    """Two cheap streaming passes over the whole VOD: is each moment the
    main (static, elevated) camera?  Pass 1 medians a sparse sample (the
    main camera dominates a broadcast, so the median IS the main view);
    pass 2 streams NCC of every downscaled grey frame against it —
    cuts and close-ups decorrelate hard."""
    sample = [f.astype(np.float32) for f in _stream_grey(video, 0.5, w, h)]
    if not sample:
        raise SystemExit("could not decode any frames for the camera scan")
    med = np.median(np.stack(sample), axis=0)
    del sample
    mz = med - med.mean()
    mn = np.sqrt((mz * mz).sum()) + 1e-9
    nccs = []
    with open(out_csv, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["t_s", "ncc", "is_main"])
        for i, fr in enumerate(_stream_grey(video, fps, w, h)):
            f = fr.astype(np.float32)
            fz = f - f.mean()
            v = float((fz * mz).sum()
                      / ((np.sqrt((fz * fz).sum()) + 1e-9) * mn))
            nccs.append(v)
            wcsv.writerow([f"{i / fps:.2f}", f"{v:.3f}",
                           int(v >= CAM_NCC_MAIN)])
    ncc = np.array(nccs)
    frac = float((ncc >= CAM_NCC_MAIN).mean())
    print(f"wrote {out_csv}: {len(ncc)} samples, {frac:.1%} main-camera "
          f"(ncc median {np.median(ncc):.3f}, threshold {CAM_NCC_MAIN})")
    return frac


def load_camera(path, fps=10.0):
    """-> (t array, is_main array); None only when no path was given.
    A path that was EXPLICITLY passed but doesn't exist is an error —
    a typo'd --cam must not silently run the whole VOD ungated."""
    if not path:
        return None
    if not Path(path).exists():
        raise SystemExit(f"--cam {path} does not exist "
                         f"(run --scan-camera first, or pass no --cam "
                         f"with --no-cam-gate)")
    t, m = [], []
    for r in csv.DictReader(open(path)):
        t.append(float(r["t_s"]))
        m.append(int(r["is_main"]))
    return np.array(t), np.array(m, bool)


def is_main_at(cam, t):
    if cam is None:
        return True
    ts, ms = cam
    i = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
    return bool(ms[i])


# ------------------------------------------------------------ rally I/O


class Det:
    __slots__ = ("t", "track", "side", "conf", "xy", "src", "h_px", "h_ft",
                 "kpt", "kpc", "box")

    def __init__(self, t, track, side, conf, xy, src, h_px, kpt, kpc, box,
                 h_ft=0.0):
        self.t, self.track, self.side, self.conf = t, track, side, conf
        self.xy, self.src, self.h_px = xy, src, h_px
        self.kpt, self.kpc, self.box = kpt, kpc, box
        self.h_ft = h_ft


def load_rally(npz_path, court, cam):
    """npz -> speed-gated, camera-gated, on-court detections + counters.

    Track side sanity: a track whose npz near/far side disagrees with
    its court-y side (players never cross the net) is excluded from
    anchoring and metrics — its detections are counted as dropped."""
    z = np.load(npz_path)
    n = len(z["t"])
    drops = {"camera": 0, "offcourt": 0, "speed": 0, "side": 0}
    dets = []
    if n == 0:
        return dets, drops
    H_img, W_img = int(z["hw"][0]), int(z["hw"][1])
    feet = np.empty((n, 2))
    srcs = []
    for i in range(n):
        x, y, src = foot_point(z["box"][i], z["kpt"][i], z["kpc"][i])
        feet[i] = (x, y)
        srcs.append(src)
    ct = project(court, feet, W_img, H_img)
    # local px-per-ft at each foot, from the inverse homography (for
    # perspective-corrected person height: partners split across 10+ ft
    # of depth during serves, so raw image heights are incomparable —
    # measured on the mixed final, where the deep receiver imaged
    # smaller than the male at the kitchen and the gender check
    # inverted)
    Hinv = np.linalg.inv(court["H"])
    sx, sy = W_img / court["w"], H_img / court["h"]

    def to_img(pts):
        q = np.hstack([pts, np.ones((len(pts), 1))]) @ Hinv.T
        q = q[:, :2] / q[:, 2:3]
        q[:, 0] *= sx
        q[:, 1] *= sy
        return q

    # LATERAL px-per-ft: a standing body's image height scales like
    # f/Z, exactly as a 1-ft segment ACROSS the court does; the along-
    # court axis is foreshortening-dominated and over-corrects far
    # players ~3x (measured: 18-ft receivers)
    pxft = np.linalg.norm(to_img(ct + np.array([1.0, 0.0])) - to_img(ct),
                          axis=1)
    pxft = np.maximum(pxft, 1e-6)
    last = {}                      # track -> (t, xy) for the speed gate
    for i in np.argsort(z["t"], kind="stable"):
        t = float(z["t"][i])
        if not is_main_at(cam, t):
            drops["camera"] += 1
            continue
        x, y = ct[i]
        if not (-OFFCOURT_FT <= x <= W_FT + OFFCOURT_FT
                and -OFFCOURT_FT - 3 <= y <= L_FT + OFFCOURT_FT + 3):
            drops["offcourt"] += 1
            continue
        tr = int(z["track"][i])
        prev = last.get(tr)
        if prev is not None and t > prev[0]:
            v = float(np.hypot(x - prev[1][0], y - prev[1][1])) / (t - prev[0])
            if v > SPEED_GATE_FTS:
                drops["speed"] += 1
                continue
        last[tr] = (t, (x, y))
        h_px = float(z["box"][i][3] - z["box"][i][1])
        dets.append(Det(t, tr, int(z["side"][i]), float(z["conf"][i]),
                        (float(x), float(y)), srcs[i], h_px,
                        z["kpt"][i], z["kpc"][i], z["box"][i],
                        h_ft=h_px / float(pxft[i])))
    # Side comes from COURT GEOMETRY here, not the npz height clusters:
    # coverage has the homography, and on multi-angle broadcasts (PPA
    # Indoor Nationals cuts between TWO elevated court angles plus
    # close-ups mid-window) the height clustering collapses — one
    # close-up torso out-heights every real player and all four get
    # "far".  Track-median court y is the side; the npz label survives
    # only as a disagreement DIAGNOSTIC in the ledger ("side" count —
    # high values flag footage where the pose-side channel is unusable,
    # they no longer cost detections).
    ys = defaultdict(list)
    npz_side = {}
    for d in dets:
        ys[d.track].append(d.xy[1])
        npz_side[d.track] = d.side
    court_side = {tr: 0 if float(np.median(yy)) > NET_Y else 1
                  for tr, yy in ys.items()}
    for d in dets:
        if npz_side[d.track] >= 0 and npz_side[d.track] != court_side[d.track]:
            drops["side"] += 1          # diagnostic only
        d.side = court_side[d.track]
    return dets, drops


def by_frame(dets):
    fr = defaultdict(list)
    for d in dets:
        fr[round(d.t, 3)].append(d)
    return dict(sorted(fr.items()))


# ------------------------------------------------------- serve anchoring


def track_positions_at(dets, t_lo, t_hi, min_n=2):
    """track -> (median court x, median court y, n dets) over
    [t_lo, t_hi] (sided only); n feeds serving_config's subset weights."""
    acc = defaultdict(list)
    for d in dets:
        if t_lo <= d.t <= t_hi and d.side >= 0:
            acc[d.track].append(d.xy)
    return {tr: (float(np.median([p[0] for p in v])),
                 float(np.median([p[1] for p in v])), len(v))
            for tr, v in acc.items() if len(v) >= min_n}


def d_kitchen(xy, end):
    return abs(xy[1] - KITCHEN_Y[end])


def serving_config(pos, weight=None):
    """Score the two serving-end hypotheses on a position snapshot.

    pos: track -> (x, y) (extra tuple elements tolerated).  Returns
    (end, margin, roles) where roles = dict(server=track,
    srv_partner=track, receiver=track, rcv_kitchen=track); or
    (None, 0, {}) when the snapshot cannot say.  The margin is the
    weakest of the geometric facts the winning hypothesis asserts, in
    feet — 0 or negative means unconvinced.

    More than 2 candidates per side is NORMAL on real footage (track
    fragments, a ballkid) — the best 2+2 SUBSET by margin wins, with
    detection-count weights as the tie-break so a fragment never beats
    a persistent track (measured on the mixed final: the strict ==2
    rule refused whole rallies whose anchor slice held one fragment)."""
    from itertools import combinations
    near_all = {tr: p for tr, p in pos.items() if p[1] > NET_Y}
    far_all = {tr: p for tr, p in pos.items() if p[1] <= NET_Y}
    if len(near_all) < 2 or len(far_all) < 2:
        return None, 0.0, {}
    w = weight or {}

    def top(side):
        return sorted(side, key=lambda tr: -w.get(tr, 1))[:5]

    best = (None, 0.0, {})
    best_score = -1e18
    for np_ in combinations(top(near_all), 2):
        for fp_ in combinations(top(far_all), 2):
            sub = {tr: pos[tr] for tr in np_ + fp_}
            end, margin, roles = _config4(sub)
            if end is None:
                continue
            score = margin + 1e-3 * sum(w.get(tr, 1) for tr in sub)
            if score > best_score:
                best_score = score
                best = (end, margin, roles)
    return best


def _config4(pos):
    """The exact 2+2 configuration test (see serving_config)."""
    near = {tr: p for tr, p in pos.items() if p[1] > NET_Y}
    far = {tr: p for tr, p in pos.items() if p[1] <= NET_Y}
    if len(near) != 2 or len(far) != 2:
        return None, 0.0, {}

    def hyp(srv_end):
        rcv_end = "far" if srv_end == "near" else "near"
        S = near if srv_end == "near" else far
        R = far if srv_end == "near" else near
        dS = {tr: d_kitchen(p, srv_end) for tr, p in S.items()}
        dR = {tr: d_kitchen(p, rcv_end) for tr, p in R.items()}
        rcv_kitchen = min(dR, key=dR.get)
        receiver = max(dR, key=dR.get)
        # The serve is cross-court by rule, so the server stands DIAGONAL
        # to the receiver (opposite lateral half).  Under stacking BOTH
        # serving-end players can sit on the diagonal half — then the
        # rule court's lateral bounds decide (the server must serve from
        # within the diagonal half's x-range; the classic stack partner
        # hugs or crosses the sideline extension).  A residual depth
        # tie-break is flagged ambiguous so it can never ride at full
        # confidence.
        rx = R[receiver][0]
        lo, hi = (0.0, W_FT / 2) if rx > W_FT / 2 else (W_FT / 2, W_FT)
        diag = [tr for tr, p in S.items()
                if (p[0] - W_FT / 2) * (rx - W_FT / 2) < 0]
        amb = False
        if len(diag) == 1:
            server = diag[0]
        else:
            cand = diag or list(S)
            inside = [tr for tr in cand if lo <= S[tr][0] <= hi]
            if len(inside) == 1:
                server = inside[0]     # stack partner outside the lines
            else:
                server = max(cand, key=lambda tr: dS[tr])
                amb = True             # stacked; geometry cannot say
        margin = min(min(dS.values()) - DEEP_FT,          # both srv deep
                     RCV_KITCHEN_FT - dR[rcv_kitchen],    # one rcv at NVZ
                     dR[receiver] - DEEP_FT)              # receiver deep
        roles = {"server": server,
                 "srv_partner": next(tr for tr in S if tr != server),
                 "receiver": receiver, "rcv_kitchen": rcv_kitchen,
                 "srv_amb": amb}
        return margin, roles

    m_near, r_near = hyp("near")
    m_far, r_far = hyp("far")
    if m_near <= 0 and m_far <= 0:
        return None, 0.0, {}
    if m_near >= m_far:
        return "near", m_near - max(m_far, 0.0), r_near
    return "far", m_far - max(m_near, 0.0), r_far


def serving_config3(pos):
    """Relaxed 2+1 / 1+2 serving-configuration test for the ANCHOR
    FINDER only.  Extraction drops a player in many real pre-serve
    freezes (measured on the Chicago pins: 5/15 freezes ran with 3 or
    fewer tracks the whole way), so the strict 2+2 test never sees the
    true freeze and the finder falls back onto a late impostor.  This
    is weaker evidence by construction — find_serve accepts such runs
    only with a positive post-freeze signature (return rush / play
    onset) and a longer minimum run, and identity resolution still
    demands the full 2+2."""
    from itertools import combinations
    near_all = {tr: p for tr, p in pos.items() if p[1] > NET_Y}
    far_all = {tr: p for tr, p in pos.items() if p[1] <= NET_Y}
    if not near_all or not far_all:
        return None, 0.0, {}

    def score(S, R, srv_end):
        rcv_end = "far" if srv_end == "near" else "near"
        dS = {tr: d_kitchen(p, srv_end) for tr, p in S.items()}
        dR = {tr: d_kitchen(p, rcv_end) for tr, p in R.items()}
        facts = [min(dS.values()) - DEEP_FT]      # visible servers deep
        roles = {"srv_amb": True}
        if len(R) == 2:
            rcv_kitchen = min(dR, key=dR.get)
            receiver = max(dR, key=dR.get)
            facts += [RCV_KITCHEN_FT - dR[rcv_kitchen],
                      dR[receiver] - DEEP_FT]
            roles["receiver"] = receiver
            roles["rcv_kitchen"] = rcv_kitchen
        else:
            (tr,) = R
            k_fact = RCV_KITCHEN_FT - dR[tr]      # fits the NVZ role...
            r_fact = dR[tr] - DEEP_FT             # ...or the deep role
            if r_fact >= k_fact:
                facts.append(r_fact)
                roles["receiver"] = tr
            else:
                facts.append(k_fact)
                roles["rcv_kitchen"] = tr
        return min(facts), roles

    best = (None, 0.0, {})
    best_m = 0.0
    for srv_end in ("near", "far"):
        S_all = near_all if srv_end == "near" else far_all
        R_all = far_all if srv_end == "near" else near_all
        for ns, nr in ((2, 1), (1, 2)):
            if len(S_all) < ns or len(R_all) < nr:
                continue
            for Sc in combinations(S_all, ns):
                for Rc in combinations(R_all, nr):
                    m, roles = score({tr: S_all[tr] for tr in Sc},
                                     {tr: R_all[tr] for tr in Rc}, srv_end)
                    if m > best_m:
                        best_m = m
                        best = (srv_end, m, roles)
    return best


def find_serve(dets, t0, t1, lead_s):
    """The anchor-frame finder (the spec's one new algorithmic piece).

    Serve = the END of the last stretch where (a) the frame's positions
    form a valid serving configuration and (b) all-player motion is
    low; motion rises when the serve is struck.  Scans [t0, t1 - 3].
    Returns (t_serve, quality, srv_end) — quality 0 means fallback
    (t0 + lead clamped) and the rally should be treated as low-trust."""
    frames = [(t, ds) for t, ds in by_frame(dets).items() if t <= t1 - 3.0]
    if len(frames) < 4:
        t = min(max(t0 + (lead_s or 0.0), t0), t1 - 3.0)
        return t, 0.0, None
    ts = [t for t, _ in frames]
    pos_seq, ok_seq = [], []
    for t, ds in frames:
        pos = {d.track: d.xy for d in ds if d.side >= 0}
        end, margin, roles = serving_config(pos)
        partial = False
        if end is None:
            end, margin, roles = serving_config3(pos)
            partial = end is not None
        pos_seq.append(pos)
        ok_seq.append((end, margin, roles, partial))
    # median motion (ft/s) over a ~0.5 s baseline: real movement
    # integrates over the baseline while projected-foot jitter does not,
    # so slow post-serve convergence separates cleanly from stillness
    # (frame-to-frame speed at 10 Hz is jitter-dominated — measured in
    # the selftest, which plants 0.25 ft noise)
    mot = []
    j = 0
    for k in range(len(frames)):
        while ts[k] - ts[j] > 0.65:
            j += 1
        jj = j if ts[k] - ts[j] >= 0.35 else max(0, k - 1)
        dt = ts[k] - ts[jj]
        common = set(pos_seq[k]) & set(pos_seq[jj])
        if not common or dt <= 0:
            mot.append(np.inf if k else 0.0)
            continue
        mot.append(float(np.median([np.hypot(
            pos_seq[k][tr][0] - pos_seq[jj][tr][0],
            pos_seq[k][tr][1] - pos_seq[jj][tr][1]) / dt
            for tr in common])))
    # quiescent serving-stance frames.  The quiet gate adapts to THIS
    # window's stillness floor (p10 of motion — pre-serve frames
    # dominate the low tail), so far-court jitter doesn't flicker the
    # gate mid-play and clean footage doesn't over-admit slow drift.
    finite = [m for m in mot if np.isfinite(m)]
    thr = float(np.clip(2.5 * np.percentile(finite, 10), 0.9, 2.5)) \
        if finite else 0.9
    good = [k for k in range(len(frames))
            if ok_seq[k][0] is not None and mot[k] < thr]
    if not good:
        t = min(max(t0 + (lead_s or 0.0), t0), t1 - 3.0)
        return t, 0.0, None
    # LAST run of good frames = this rally's pre-serve freeze (the
    # window reaches back over the previous rally, whose own freeze can
    # be longer — the last freeze before the flip is structurally this
    # rally's, because this rally's play fills the span up to the flip).
    # Noise guard: a last "run" under 3 frames falls back to the
    # longest — UNLESS the short run starts at the very first visible
    # frame: replay-heavy broadcasts cut back to the main camera with
    # the freeze already underway, amputating its start (measured on
    # PPA Indoor Nationals), and a 2-frame freeze at the span's opening
    # edge is the freeze's END, which is all the anchor needs.
    runs, cur = [], [good[0]]
    for k in good[1:]:
        if ts[k] - ts[cur[-1]] <= 0.7:
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)

    def surge(run):
        """A real serve is followed by the RETURN RUSH: the deep
        receiver moves net-ward within ~4 s of the freeze end (the
        two-bounce rule keeps the SERVING pair back, so the surge is
        one-sided and receiver-specific).  A quiet serving-shaped span
        5+ s before the actual serve — the next rally's forming freeze
        leaking into the window tail, the measured +10..27 s late tail
        on the Chicago pins — has no surge and is rejected."""
        k_end = run[-1]
        end, _, roles, _p = ok_seq[k_end]
        rcv = roles.get("receiver")
        if rcv is None or rcv not in pos_seq[k_end]:
            return False
        y_tau = pos_seq[k_end][rcv][1]
        s = 1.0 if end == "near" else -1.0
        tau = ts[k_end]
        best = 0.0
        for k in range(k_end + 1, len(frames)):
            if ts[k] - tau > 4.0:
                break
            if rcv in pos_seq[k]:
                best = max(best, s * (pos_seq[k][rcv][1] - y_tau))
        return best >= 4.5

    def sustained(run):
        """Play-onset fallback when no receiver track survives to be
        watched: rally motion (median ~3+ ft/s) follows a real serve
        within its first 4 s; pre-serve milling stays at walking pace
        (measured medians 3.4 vs 1.3 on the Chicago failures).  Too
        few post-freeze frames (window edge) counts as NO evidence."""
        k_end = run[-1]
        tau = ts[k_end]
        ms = [mot[k] for k in range(k_end + 1, len(frames))
              if ts[k] - tau <= 4.0 and np.isfinite(mot[k])]
        return len(ms) >= 6 and float(np.median(ms)) >= 2.5

    def hold_deep(run):
        """Two-bounce corollary: after a REAL serve the serving pair
        stays deep through the return (the serve and return must both
        bounce), so for ~1.5 s no serving-side track may appear inside
        the court.  After a mid-rally lull the 'serving-shaped' side is
        free to advance immediately — the measured thief (a 30-frame
        lull, +12.5 s late) fails exactly this.  Tracks that vanish
        give no evidence either way."""
        k_end = run[-1]
        end = ok_seq[k_end][0]
        side = [tr for tr, p in pos_seq[k_end].items()
                if (p[1] > NET_Y) == (end == "near")]
        tau = ts[k_end]
        for k in range(k_end + 1, len(frames)):
            dt = ts[k] - tau
            if dt > 1.5:
                break
            if dt < 0.3:
                continue
            for tr in side:
                if tr in pos_seq[k] and \
                        d_kitchen(pos_seq[k][tr], end) < DEEP_FT - 2.0:
                    return False
        return True

    def follow(run):
        rcv = ok_seq[run[-1]][2].get("receiver")
        rcv_gone = rcv is None or rcv not in pos_seq[run[-1]]
        sig = surge(run) or (rcv_gone and sustained(run))
        return sig and hold_deep(run)

    def is_partial(run):
        return sum(1 for k in run if ok_seq[k][3]) * 2 > len(run)

    # Walk candidates latest-first; take the first with a post-freeze
    # signature.  Partial-config runs (serving_config3) may ONLY win
    # this way and need a longer freeze; with no signature anywhere,
    # fall back to the OLD rule over runs rebuilt from strict-config
    # frames alone (bit-identical to the pre-signature behavior), so
    # the discriminators can move the anchor but never lose one.
    best = None
    fallback = False
    for run in reversed(runs):
        ml = 6 if is_partial(run) else (2 if run[0] == 0 else 3)
        if len(run) >= ml and follow(run):
            best = run
            break
    if best is None:
        fallback = True
        good_f = [k for k in good if not ok_seq[k][3]]
        if not good_f:
            t = min(max(t0 + (lead_s or 0.0), t0), t1 - 3.0)
            return t, 0.0, None
        runs_f, cur = [], [good_f[0]]
        for k in good_f[1:]:
            if ts[k] - ts[cur[-1]] <= 0.7:
                cur.append(k)
            else:
                runs_f.append(cur)
                cur = [k]
        runs_f.append(cur)
        min_len = 2 if runs_f[-1][0] == 0 else 3
        best = runs_f[-1] if len(runs_f[-1]) >= min_len \
            else max(runs_f, key=len)
    k_end = best[-1]
    margin = float(np.median([ok_seq[k][1] for k in best]))
    qual = min(1.0, len(best) / 8.0) * min(1.0, max(margin, 0.0) / 2.0 + 0.5)
    if is_partial(best):
        qual *= 0.75
    if fallback:
        # no post-freeze signature confirmed this anchor — the measured
        # late tail lives exactly here, so it rides at half trust
        qual *= 0.5
    ends = [ok_seq[k][0] for k in best]
    end = max(set(ends), key=ends.count)
    return ts[k_end], qual, end


# ------------------------------------------------------------- identity


def lineup_reliability(lineup_rows, id8, game, rally_in_game, radius=3):
    """The state machine self-scores (receiver_ok); its halves are only
    trusted where the surrounding rallies agree — PPA logs can desync
    the machine in long runs (measured 65.2% on c4eb30d0 vs 99.25% on
    MLP 2026), and those runs are visible in receiver_ok itself."""
    ok = [int(r["receiver_ok"]) for r in lineup_rows
          if r.get("_id8", id8) == id8 and r["game"] == str(game)
          and abs(int(r["rally"]) - int(rally_in_game)) <= radius]
    return float(np.mean(ok)) if ok else 0.0


def anchor_identity(dets, t_serve, win, lin, genders, heights=None):
    """Resolve all four names at the serve.  Returns
    (names {track: uuid}, conf, checks dict) or (None, 0, checks)."""
    checks = {"diagonal": "", "halves": "", "gender": ""}
    pos = track_positions_at(dets, t_serve - 0.6, t_serve + 0.4)
    wts = {tr: p[2] for tr, p in pos.items() if len(p) > 2}
    end, margin, roles = serving_config(pos, wts)
    if end is None:
        return None, 0.0, dict(checks, reason="no_serving_config")
    server_uuid = win["server_uuid"].lower()
    receiver_uuid = win["receiver_uuid"].lower()
    if lin is None:
        return None, 0.0, dict(checks, reason="no_lineup_row")
    teamA = {lin["team_A_R"], lin["team_A_L"]}
    teamB = {lin["team_B_R"], lin["team_B_L"]}
    if server_uuid in teamA:
        srv_partner_uuid = next(p for p in teamA if p != server_uuid)
    elif server_uuid in teamB:
        srv_partner_uuid = next(p for p in teamB if p != server_uuid)
    else:
        return None, 0.0, dict(checks, reason="server_not_in_lineup")
    rcv_team = teamB if receiver_uuid in teamB else teamA
    if receiver_uuid not in rcv_team:
        return None, 0.0, dict(checks, reason="receiver_not_in_lineup")
    rcv_partner_uuid = next(p for p in rcv_team if p != receiver_uuid)

    names = {roles["server"]: server_uuid,
             roles["srv_partner"]: srv_partner_uuid,
             roles["receiver"]: receiver_uuid,
             roles["rcv_kitchen"]: rcv_partner_uuid}

    conf = min(1.0, 0.55 + 0.15 * min(margin, 3.0))
    # diagonal: server and receiver on opposite lateral halves
    sx = pos[roles["server"]][0]
    rx = pos[roles["receiver"]][0]
    diag_ok = (sx - W_FT / 2) * (rx - W_FT / 2) < 0
    checks["diagonal"] = int(diag_ok)
    conf *= 1.0 if diag_ok else 0.75
    if roles.get("srv_amb"):
        checks["stacked"] = 1          # depth tie-break had to decide
        conf *= 0.75
    # mixed-doubles gender/height prior: within each pair the male's
    # detections should image taller (redundant check, never an input)
    if genders and heights:
        agree = tested = 0
        for pair in (({roles["server"]: names[roles["server"]],
                       roles["srv_partner"]: names[roles["srv_partner"]]}),
                     ({roles["receiver"]: names[roles["receiver"]],
                       roles["rcv_kitchen"]: names[roles["rcv_kitchen"]]})):
            trs = list(pair)
            g = [genders.get(pair[tr], "?") for tr in trs]
            if set(g) == {"M", "F"} and all(tr in heights for tr in trs):
                tested += 1
                male = trs[g.index("M")]
                female = trs[g.index("F")]
                agree += heights[male] > heights[female]
        if tested:
            # REPORT-ONLY, no confidence effect: rally-median box height
            # measures STANCE, not stature — kitchen players crouch all
            # rally, so the tallest man on tour measures shortest when
            # he plays the NVZ (measured on the mixed final).  A crouch-
            # proof estimator (skeleton torso+femur segment lengths)
            # is the upgrade path if this check is ever to bite.
            checks["gender"] = f"{agree}/{tested}"
    return names, conf, checks


def carry_names(dets, names0, conf0):
    """Names ride track ids; strict-gated handoff when a named track
    dies and a new unnamed one appears nearby on the SAME side of the
    net.  Returns per-detection (uuid or None, conf, handoff_flag)
    aligned with dets (time order).

    Track lifespans are contiguous (pose_extract ids never resurrect),
    so a per-track name is time-correct as long as handoffs never strip
    a dying track's name — the receiving track gets the name IN
    ADDITION; only future handoff ELIGIBILITY moves (a name hands off
    at most once per break, but the old track's earlier detections keep
    it)."""
    all_names = dict(names0)           # per-detection lookup, never shrinks
    open_names = dict(names0)          # still eligible to hand off
    conf = {tr: conf0 for tr in names0}
    last_seen, first_seen = {}, {}
    for d in dets:
        first_seen.setdefault(d.track, (d.t, d.xy))
        last_seen[d.track] = (d.t, d.xy)
    unnamed = sorted((t_xy[0], tr) for tr, t_xy in first_seen.items()
                     if tr not in all_names)
    handoff = set()
    for t_new, tr_new in unnamed:
        best = None
        xy_new = first_seen[tr_new][1]
        for tr_old, uuid in list(open_names.items()):
            if tr_old not in last_seen:
                # a name for a track with no detections in this rally
                # (stale identity ledger).  It has no position, so it
                # cannot hand off; skipping beats crashing.
                continue
            t_old, xy_old = last_seen[tr_old]
            if t_old >= t_new or t_new - t_old > HANDOFF_DT:
                continue
            if (xy_old[1] - NET_Y) * (xy_new[1] - NET_Y) <= 0:
                continue               # never hand a name across the net
            dist = float(np.hypot(xy_new[0] - xy_old[0],
                                  xy_new[1] - xy_old[1]))
            if dist > HANDOFF_FT:
                continue
            if best is None or dist < best[0]:
                best = (dist, tr_old, uuid)
        if best is not None:
            _, tr_old, uuid = best
            all_names[tr_new] = uuid
            conf[tr_new] = conf[tr_old] * 0.8
            handoff.add(tr_new)
            del open_names[tr_old]
            open_names[tr_new] = uuid
    out = []
    for d in dets:
        u = all_names.get(d.track)
        out.append((u, conf.get(d.track, 0.0), d.track in handoff))
    return out


# -------------------------------------------------------------- metrics


def smooth_series(ts, xs, span=SMOOTH_S):
    """Box filter over +-span/2 seconds (irregular sampling tolerated)."""
    ts = np.asarray(ts, float)
    xs = np.asarray(xs, float)
    out = np.empty_like(xs)
    j0 = 0
    for i, t in enumerate(ts):
        lo, hi = t - span / 2, t + span / 2
        while ts[j0] < lo:
            j0 += 1
        j1 = i
        while j1 + 1 < len(ts) and ts[j1 + 1] <= hi:
            j1 += 1
        out[i] = xs[j0:j1 + 1].mean(axis=0)
    return out


def ellipse_area(pts, wts):
    """90% Gaussian occupancy ellipse, ft^2 (weighted, one 3-sigma trim)."""
    P = np.asarray(pts, float)
    w = np.asarray(wts, float)
    if len(P) < 20:
        return np.nan
    for _ in range(2):
        mu = (P * w[:, None]).sum(0) / w.sum()
        d = P - mu
        cov = (d * w[:, None]).T @ d / w.sum()
        cov += np.eye(2) * 1e-6
        m = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
        keep = m < 9.0
        if keep.all():
            break
        P, w = P[keep], w[keep]
        if len(P) < 20:
            return np.nan
    det = float(np.linalg.det(cov))
    return float(np.pi * ELLIPSE_CHI2 * np.sqrt(max(det, 0.0)))


class PlayerGame:
    """Accumulates one player's rally-phase observations over a game."""

    def __init__(self):
        self.pts = []          # (x, y, weight)
        self.serve_pts = []
        self.rallies = set()
        self.dist = 0.0
        self.static_n = 0
        self.speed_n = 0
        self.trips = 0
        self.handoff_n = 0
        self.af_rallies = set()   # rallies named without a serve anchor
        self.af_det_n = 0
        self.det_n = 0
        self.conf_sum = 0.0
        self.share_num = 0.0   # width share accumulator (frame count)
        self.share_n = 0
        self.share_k_num = 0.0
        self.share_k_n = 0
        self.partner_area_pts = None   # filled at game level

    def add_rally(self, ts, xy, conf, wts, handoffs, phase_mask, serve_mask,
                  end, cum=None, anchor_free=False):
        """ts sorted; xy (n,2); phase_mask = rally-phase frames."""
        self.rallies.add(True)
        if anchor_free:
            self.af_rallies.add(cum)
            self.af_det_n += int(phase_mask.sum())
        for i in np.nonzero(phase_mask)[0]:
            self.pts.append((xy[i, 0], xy[i, 1], wts[i]))
        for i in np.nonzero(serve_mask)[0]:
            self.serve_pts.append((xy[i, 0], xy[i, 1], wts[i]))
        self.det_n += int(phase_mask.sum())
        self.conf_sum += float(conf * phase_mask.sum())
        self.handoff_n += int((handoffs & phase_mask).sum())
        if phase_mask.sum() >= 3:
            sm = smooth_series(ts[phase_mask], xy[phase_mask])
            d = np.linalg.norm(np.diff(sm, axis=0), axis=1)
            dt = np.diff(ts[phase_mask])
            v = np.divide(d, dt, out=np.zeros_like(d), where=dt > 0)
            self.dist += float(d.sum())
            self.static_n += int((v < STATIC_FTS).sum())
            self.speed_n += len(v)
            dk = np.abs(sm[:, 1] - KITCHEN_Y[end])
            deep = dk > 10.0
            close = dk <= KITCHEN_BAND_FT
            was_deep = False
            for i in range(len(dk)):
                if deep[i]:
                    was_deep = True
                elif close[i] and was_deep:
                    self.trips += 1
                    was_deep = False


def rally_share(ts_a, xy_a, ts_b, xy_b, ends_y):
    """Width-share sums for player a vs partner b on common frames.
    Returns (sum_share_a, n, sum_share_kitchen_a, n_kitchen)."""
    ta = {round(t, 3): i for i, t in enumerate(ts_a)}
    s = sk = 0.0
    n = nk = 0
    for j, t in enumerate(ts_b):
        i = ta.get(round(t, 3))
        if i is None:
            continue
        xa, ya = xy_a[i]
        xb, yb = xy_b[j]
        m = (xa + xb) / 2
        share = m / W_FT if xa < xb else (W_FT - m) / W_FT
        s += share
        n += 1
        if (abs(ya - ends_y) <= KITCHEN_BAND_FT
                and abs(yb - ends_y) <= KITCHEN_BAND_FT):
            sk += share
            nk += 1
    return s, n, sk, nk


# ------------------------------------------------------------ pipeline


def load_lineup(paths):
    """One or more lineup_<id8>.csv (comma-separated for matchup VODs,
    whose windows carry per-rally match_ids).  Keyed by
    (match id8, game, rally); rows carry _id8 for reliability lookups."""
    rows_all, by_key, ids = [], {}, []
    for p in str(paths).split(","):
        p = Path(p.strip())
        id8 = p.stem.replace("lineup_", "")[:8]
        ids.append(id8)
        for r in csv.DictReader(open(p)):
            for k in ("server_uuid", "receiver_uuid", "team_A_R",
                      "team_A_L", "team_B_R", "team_B_L"):
                r[k] = r[k].lower()
            r["_id8"] = id8
            rows_all.append(r)
            by_key[(id8, r["game"], r["rally"])] = r
    return rows_all, by_key, ids


def lineup_for(win, by_key, ids):
    """The lineup row for a windows row: matchup windows carry match_id;
    single-match windows fall back to the only lineup file given."""
    id8 = (win.get("match_id") or "")[:8]
    if not id8 and len(ids) == 1:
        id8 = ids[0]
    return by_key.get((id8, win["game"], win["rally_in_game"])), id8


def load_windows(path):
    return {int(r["rally_cum"]): r for r in csv.DictReader(open(path))}


MIN_END_SEG = 3       # rallies a mid-game end segment must hold to be
                      # believed; below this a lone odd rally is an
                      # identity error, not a switch


def fit_end_segments(per_rally, min_seg=MIN_END_SEG):
    """Split one game's rallies into at most TWO end segments.

    Teams change ends mid-game under rules that are consistent WITHIN a
    league and differ BETWEEN them (user, 2026-08-19): MLP switches at 6
    in every game, PPA only in a decider.  Rather than hard-code a
    league, fit the switch from the data — the side (near/far) channel
    is box-height clustering and the names are geometry, so neither owes
    anything to the appearance model this map goes on to constrain.

    per_rally: [(cum, key)] in any order, key = whatever identifies the
    near-side pair.  Returns {cum: segment index 0/1}.

    A segment must hold min_seg rallies to count.  Without that floor a
    single mis-identified rally at a game boundary manufactures a
    spurious switch — exactly what rally 107 of the mixed final would
    have done (one rally disagreeing with the eleven after it).
    """
    seq = sorted(per_rally)
    if len(seq) < 2 * min_seg:
        return {c: 0 for c, _ in seq}
    keys = [k for _, k in seq]
    best, best_cut = -1, None
    for cut in range(min_seg, len(keys) - min_seg + 1):
        a_k = Counter(keys[:cut]).most_common(1)[0]
        b_k = Counter(keys[cut:]).most_common(1)[0]
        if a_k[0] == b_k[0]:
            continue                      # not a switch, just noise
        if a_k[1] < min_seg or b_k[1] < min_seg:
            continue                      # the floor is on the EVIDENCE
        score = a_k[1] + b_k[1]           # for each end, not on where
                                          # the cut happens to fall
        if score > best:
            best, best_cut = score, cut
    flat = Counter(keys).most_common(1)[0][1]
    if best_cut is None or best <= flat:
        return {c: 0 for c, _ in seq}     # one end all game
    return {c: (0 if i < best_cut else 1) for i, (c, _) in enumerate(seq)}


def player_meta():
    g, nm = {}, {}
    f = DATA / "players.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            pid = r["player_id"].lower()
            g[pid] = r["gender"]
            nm[pid] = r["full_name"]
    return g, nm


def run(a, collect=None, write=True):
    """write=False runs the whole chain but persists NOTHING.

    run() writing the committed CSVs as a side effect is a real footgun
    for collect-only callers: a diagnostic script run with a placeholder
    --vod appended 13 junk rows to data/coverage_players.csv under
    vod="X" (the upsert keys on vod, so they did not even overwrite).
    Anything that only wants the collect hook should pass write=False.
    """
    """collect: optional callback handed rally_tracks_by_game after the
    resolution loop — lets sibling instruments (coverage_dominance.py)
    compute on EXACTLY the frame set the shipped metrics used, without
    duplicating the identity chain.  Never alters this run's output."""
    court = load_court(a.court)
    cam = load_camera(a.cam)
    if cam is None and not getattr(a, "no_cam_gate", False):
        raise SystemExit("--run needs --cam (or --no-cam-gate to run "
                         "ungated — the ledger will say so)")
    windows = load_windows(a.windows)
    lineup_rows, lineup_by, lineup_ids = load_lineup(a.lineup)
    genders, names = player_meta()
    pose_dir = Path(a.pose_dir)
    swaps = defaultdict(list)
    n_swapped = 0
    if getattr(a, "swaps", ""):
        for r in csv.DictReader(open(a.swaps)):
            if r["swap"] == "1" and float(r["unanimity"]) >= 0.8:
                swaps[int(r["rally_cum"])].append(
                    tuple(r["team"].split("|")))
        print(f"swap ledger: {sum(len(v) for v in swaps.values())} "
              f"team-rally swaps loaded from {a.swaps}")
    track_map = defaultdict(lambda: defaultdict(list))
    n_tm = Counter()
    if getattr(a, "track_map", ""):
        for r in csv.DictReader(open(a.track_map)):
            track_map[int(r["rally_cum"])][int(r["track"])].append(
                (float(r["t0"]), float(r["t1"]), r["uuid"], r["action"]))
        print(f"track map: {sum(len(v) for m in track_map.values() for v in m.values())} "
              f"spans loaded from {a.track_map}")
    anchor_free = defaultdict(dict)
    n_anchor_free = 0
    if getattr(a, "anchor_free", ""):
        for r in csv.DictReader(open(a.anchor_free)):
            anchor_free[int(r["rally_cum"])][int(r["track"])] = \
                r["player_uuid"]
        print(f"anchor-free ledger: {len(anchor_free)} rallies, "
              f"{sum(len(v) for v in anchor_free.values())} track names "
              f"from {a.anchor_free}")
    # backend provenance: fleet numbers are not trusted until the spec's
    # pre-named A/B guard has run (vision/coverage_ab.py; ViTPose wins
    # disagreements) — so every row records which backend produced it
    backend = "unknown"
    mp = pose_dir / "meta.json"
    if mp.exists():
        backend = json.loads(mp.read_text()).get("backend", "unknown")
    if backend == "unknown":
        print("WARNING: pose dir carries no backend provenance "
              "(meta.json missing/keyless) — rows record 'unknown'")
    spot = {}
    if a.spotcheck and Path(a.spotcheck).exists():
        for r in csv.DictReader(open(a.spotcheck)):
            if r.get("watched", "0") == "1":
                spot[int(r["rally_cum"])] = int(r.get("swaps_seen", 0) or 0)

    games = defaultdict(lambda: defaultdict(PlayerGame))
    endmap_obs = []                   # (id8, game, cum, serving team, end)
    dropped = defaultdict(int)        # rally-level drops, by reason
    stale_ledger = defaultdict(int)   # ledger entries refused as stale
    det_drops = defaultdict(int)      # detection-level gate drops
    frames_kept = 0
    anchor_offsets = []
    halves_ok = halves_tested = 0
    gender_agree = gender_tested = 0
    n_covered = 0
    rally_tracks_by_game = defaultdict(list)

    for cum in sorted(windows):
        win = windows[cum]
        npz = pose_dir / f"r{cum:04d}.npz"
        if not npz.exists():
            dropped["no_pose"] += 1
            continue
        if win["approx"] == "1":
            dropped["approx_window"] += 1
            continue
        if win["outcome"] not in ("point", "sideout", "second"):
            dropped["outcome"] += 1
            continue
        dets, drops = load_rally(npz, court, cam)
        for k, v in drops.items():
            det_drops[k] += v
        frames_kept += len(dets)
        if not dets:
            dropped["no_detections"] += 1
            continue
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        lead = float(win["lead_s"]) if win.get("lead_s") else 0.0
        t_serve, qual, srv_end = find_serve(dets, t0, t1, lead)
        # ANCHOR-FREE fallback (coverage_anchorfree --emit): a rally whose
        # serve the broadcast never showed has no anchor, but the players
        # are still identifiable from an appearance model trained on the
        # rallies that DO have serves.  Only games clearing that module's
        # Gate A reach this ledger, and the rally is flagged so every
        # downstream number can be recomputed without them.
        af_avail = anchor_free.get(cum)
        # STALE-LEDGER GUARD.  The identity ledgers (--anchor-free,
        # --track-map) are keyed on pose TRACK IDS, and those ids live
        # only inside the gitignored pose dir: re-extracting poses
        # renumbers them, so a ledger built against an older extraction
        # points at different people.  Refuse a mismatched entry WHOLE --
        # applying the subset that happens to resolve is exactly the
        # silent mis-bind this guard exists to prevent.  (Found
        # 2026-08-20: a mismatch crashed carry_names with a KeyError,
        # which at least failed loudly; a partial match would not have.)
        present_tracks = {d.track for d in dets}
        if af_avail is not None and any(tr not in present_tracks
                                        for tr in af_avail):
            stale_ledger["anchor_free"] += 1
            af_avail = None
        tm_ok = True
        if cum in track_map and any(tr not in present_tracks
                                    for tr in track_map[cum]):
            stale_ledger["track_map"] += 1
            tm_ok = False
        af_names = af_avail if qual == 0.0 else None
        if qual == 0.0 and af_names is None:
            # no anchor found: the fallback guess cannot place the serve
            # (in start-marked logs dur ~= lead, so t0+lead sits at the
            # rally END).  Ambiguous spans are dropped, not guessed.
            dropped["anchor_not_found"] += 1
            continue
        if af_names is None:
            anchor_offsets.append(t_serve - t0)
        lin, id8 = lineup_for(win, lineup_by, lineup_ids)
        heights = defaultdict(list)
        for d in dets:
            heights[d.track].append(d.h_ft or d.h_px)
        heights = {tr: float(np.median(v)) for tr, v in heights.items()}
        if af_names is not None:
            names_map, conf, checks = dict(af_names), AF_CONF, {}
        else:
            names_map, conf, checks = anchor_identity(
                dets, t_serve, win, lin, genders, heights)
            if (names_map is None or conf < CONF_MIN) and af_avail:
                # an anchor exists but geometry cannot resolve the
                # configuration (stacking, a missing track).  Appearance
                # does not need the serving geometry, so it can still
                # name the rally — same ledger, same Gate A.
                names_map, conf, checks = dict(af_avail), AF_CONF, {}
                af_names = af_avail
        if names_map is None or conf < CONF_MIN:
            dropped["identity_" + checks.get("reason", "lowconf")] += 1
            continue
        if af_names is not None:
            n_anchor_free += 1
        # appearance-audited anchor swaps (coverage_appearance --audit):
        # the geometry chain's server/partner pick can swap a TEAM's two
        # names for a whole rally (stacking ambiguity; measured 12/63 +
        # 19/63 on the mixed final).  The ledger says which rallies —
        # swap them back before any frame is attributed.
        for (ua, ub) in swaps.get(cum, ()):
            inv = {u: tr for tr, u in names_map.items()}
            if ua in inv and ub in inv:
                names_map[inv[ua]], names_map[inv[ub]] = ub, ua
                n_swapped += 1
        # lineup-halves consistency (report-only): predicted lateral half
        # of the server vs observed, weighted by the machine's local
        # receiver_ok agreement around this rally
        if lin is not None and af_names is None:
            rel = lineup_reliability(lineup_rows, id8, win["game"],
                                     win["rally_in_game"])
            if rel >= 0.99:
                pos = track_positions_at(dets, t_serve - 0.6, t_serve + 0.4)
                srv_tr = next(tr for tr, u in names_map.items()
                              if u == win["server_uuid"].lower())
                if srv_tr in pos:
                    # the serving team's R half is image-left for the far
                    # end and image-right for the near end (camera behind
                    # the near end; both bits over-identified per game —
                    # here we only score agreement, we never consume it)
                    end = "near" if pos[srv_tr][1] > NET_Y else "far"
                    want_right_img = (end == "near") == \
                        (lin["server_half"] == "R")
                    got_right_img = pos[srv_tr][0] > W_FT / 2
                    halves_tested += 1
                    halves_ok += want_right_img == got_right_img
        if checks.get("gender"):
            agree, tested = checks["gender"].split("/")
            gender_agree += int(agree)
            gender_tested += int(tested)

        if af_names is None:
            endmap_obs.append((id8, win["game"], int(cum),
                               "A" if win["server_uuid"].lower() in
                               (lin["team_A_R"], lin["team_A_L"]) else "B",
                               srv_end or "?"))
        assign = carry_names(sorted(dets, key=lambda d: d.t), names_map, conf)
        dets_sorted = sorted(dets, key=lambda d: d.t)
        # stage-2 appearance track map (coverage_appearance --stage2):
        # rebind wrongly-carried names, name grey tracks, honor splits —
        # applied AFTER the carry so the audit trail is span-exact
        if cum in track_map and tm_ok:
            tm = track_map[cum]
            assign = list(assign)
            action = [None] * len(assign)
            for i, d in enumerate(dets_sorted):
                for (ta, tb, uu, act) in tm.get(d.track, ()):
                    if ta <= d.t <= tb:
                        if assign[i][0] != uu:
                            n_tm[act] += 1
                        assign[i] = (uu, assign[i][1], False)
                        action[i] = act
                        break
            # per-instant uniqueness: corrections must never create two
            # bodies under one name (measured failure: rescues colliding
            # with the carried track interleaved two players' positions
            # and collapsed the pair arrays).  A rescue only fills
            # absence; a rebind outranks the carried name it corrects.
            by_t = defaultdict(list)
            for i, d in enumerate(dets_sorted):
                if assign[i][0] is not None:
                    by_t[(round(d.t, 3), assign[i][0])].append(i)
            for (_t, _u), idxs in by_t.items():
                if len(idxs) < 2:
                    continue
                keep = None
                if any(action[i] in ("rebind", "split") for i in idxs):
                    for i in idxs:
                        if action[i] in ("rebind", "split"):
                            keep = i
                            break
                else:
                    for i in idxs:
                        if action[i] is None:
                            keep = i
                            break
                if keep is None:
                    keep = idxs[0]
                for i in idxs:
                    if i != keep:
                        assign[i] = (None, assign[i][1], False)
                        n_tm["dedup_dropped"] += 1
        per_uuid = defaultdict(lambda: ([], [], [], []))  # ts, xy, w, hand
        for d, (u, c, hand) in zip(dets_sorted, assign):
            if u is None:
                det_drops["unnamed"] += 1    # per-DETECTION, so it lives
                continue                     # with the gate counts
            per_uuid[u][0].append(d.t)
            per_uuid[u][1].append(d.xy)
            per_uuid[u][2].append(d.conf)
            per_uuid[u][3].append(hand)
        # key player-games by (match, game): an MLP matchup VOD holds
        # several single-game matches whose game numbers all collide
        game = (win.get("match_id") or a.match_id or id8, win["game"])
        n_covered += 1
        rally_data = {}
        for u, (ts, xy, wts, hand) in per_uuid.items():
            ts = np.array(ts)
            xy = np.array(xy)
            wts = np.array(wts)
            hand = np.array(hand, bool)
            if af_names is None:
                phase = ts >= t_serve + SERVE_PHASE_S
                serve_m = (ts >= t_serve) & (ts < t_serve + SERVE_PHASE_S)
            else:
                # ANCHOR-FREE: the serve instant is unknown by
                # construction (find_serve's qual-0 fallback is the very
                # guess the drop rule exists to refuse), so the frozen
                # "first SERVE_PHASE_S after the serve" mask cannot be
                # evaluated.  Excluding the first SERVE_PHASE_S of the
                # RETAINED frames instead is deliberately conservative:
                # if the camera was away at the serve it discards good
                # mid-rally play rather than risk admitting serve-stance
                # frames into an occupancy statistic.  No serve-phase
                # ellipse is claimed for these rallies.
                phase = ts >= (ts[0] + SERVE_PHASE_S) if len(ts) else ts
                serve_m = np.zeros(len(ts), bool)
            end = "near" if float(np.median(xy[:, 1])) > NET_Y else "far"
            games[game][u].add_rally(ts, xy, conf, wts, hand, phase,
                                     serve_m, end, cum=cum,
                                     anchor_free=af_names is not None)
            rally_data[u] = (ts[phase], xy[phase], end)
        rally_tracks_by_game[game].append((cum, rally_data, lin))

    if collect is not None:
        collect(rally_tracks_by_game)

    # width shares per game (needs partner pairing per rally)
    for game, rallies in rally_tracks_by_game.items():
        for cum, rd, lin in rallies:
            if lin is None:
                continue
            for ta, tb in ((("team_A_R", "team_A_L")),
                           (("team_B_R", "team_B_L"))):
                ua, ub = lin[ta], lin[tb]
                if ua not in rd or ub not in rd:
                    continue
                ts_a, xy_a, end_a = rd[ua]
                ts_b, xy_b, end_b = rd[ub]
                if end_a != end_b or not len(ts_a) or not len(ts_b):
                    continue
                ky = KITCHEN_Y[end_a]
                s, n, sk, nk = rally_share(ts_a, xy_a, ts_b, xy_b, ky)
                pg_a, pg_b = games[game][ua], games[game][ub]
                pg_a.share_num += s
                pg_a.share_n += n
                pg_a.share_k_num += sk
                pg_a.share_k_n += nk
                pg_b.share_num += n - s
                pg_b.share_n += n
                pg_b.share_k_num += nk - sk
                pg_b.share_k_n += nk

    # ---- rows -----------------------------------------------------------
    partner_of = {}
    for game, players in games.items():
        for r in rally_tracks_by_game[game]:
            lin = r[2]
            if lin:
                partner_of[(game, lin["team_A_R"])] = lin["team_A_L"]
                partner_of[(game, lin["team_A_L"])] = lin["team_A_R"]
                partner_of[(game, lin["team_B_R"])] = lin["team_B_L"]
                partner_of[(game, lin["team_B_L"])] = lin["team_B_R"]
    prow = []
    n_thin = 0
    for game, players in sorted(games.items()):
        for u, pg in sorted(players.items()):
            if pg.det_n < MIN_FRAMES_PLAYER_GAME:
                n_thin += 1
                continue
            pts = np.array([(x, y) for x, y, _ in pg.pts])
            wts = np.array([w for _, _, w in pg.pts])
            area = ellipse_area(pts[:, :2], wts)
            sa = (ellipse_area(np.array([(x, y) for x, y, _ in pg.serve_pts]),
                               np.array([w for _, _, w in pg.serve_pts]))
                  if len(pg.serve_pts) >= 20 else np.nan)
            depth = np.abs(pts[:, 1] - NET_Y)
            part = partner_of.get((game, u))
            area_part = np.nan
            if part in players and players[part].pts:
                pp = np.array([(x, y) for x, y, _ in players[part].pts])
                pw = np.array([w for _, _, w in players[part].pts])
                area_part = ellipse_area(pp, pw)
            share_area = (area / (area + area_part)
                          if np.isfinite(area) and np.isfinite(area_part)
                          and (area + area_part) > 0 else np.nan)
            n_r = sum(1 for (c, rd, _) in rally_tracks_by_game[game]
                      if u in rd)
            prow.append({
                "vod": a.vod, "event": a.event, "date": a.date,
                "match_id": game[0], "game": game[1],
                "backend": backend,
                "player_uuid": u, "player": names.get(u, u[:8]),
                "gender": genders.get(u, "?"),
                "partner_uuid": part or "",
                "n_rallies": n_r, "frames": pg.det_n,
                "ellipse_area_ft2": f"{area:.1f}",
                "ellipse_area_serve_ft2": (f"{sa:.1f}"
                                           if np.isfinite(sa) else ""),
                "width_share": (f"{pg.share_num / pg.share_n:.4f}"
                                if pg.share_n else ""),
                "width_share_kitchen": (
                    f"{pg.share_k_num / pg.share_k_n:.4f}"
                    if pg.share_k_n >= 40 else ""),
                "width_share_area": (f"{share_area:.4f}"
                                     if np.isfinite(share_area) else ""),
                "depth_p05_ft": f"{np.percentile(depth, 5):.1f}",
                "depth_p95_ft": f"{np.percentile(depth, 95):.1f}",
                "dist_per_rally_ft": (f"{pg.dist / max(n_r, 1):.1f}"),
                "static_frac": (f"{pg.static_n / pg.speed_n:.3f}"
                                if pg.speed_n else ""),
                "kitchen_trips_per_rally": f"{pg.trips / max(n_r, 1):.2f}",
                "identity_conf": f"{pg.conf_sum / max(pg.det_n, 1):.3f}",
                "anchor_free_rallies": len(pg.af_rallies),
                "anchor_free_frac": (f"{pg.af_det_n / pg.det_n:.3f}"
                                     if pg.det_n else ""),
                "handoff_frac": f"{pg.handoff_n / max(pg.det_n, 1):.3f}",
                "identity_err_rate": (
                    f"{sum(v > 0 for v in spot.values()) / len(spot):.3f}"
                    if spot else ""),
            })

    # end-map consistency: within a game teams keep ends (<=1 switch,
    # at 6 in deciders), so per-rally serving-end inferences must fit a
    # piecewise-constant team->end map; violations = identity errors
    endmap_viol = endmap_n = 0
    by_g = defaultdict(list)
    for id8_, g, cum, team, end in endmap_obs:
        if end in ("near", "far"):
            by_g[(id8_, g)].append((cum, (team == "A") == (end == "near")))
    for g, seq in by_g.items():
        seq.sort()
        b = [x for _, x in seq]
        best = 0
        for cut in range(len(b) + 1):
            for v0 in (True, False):
                ok = sum(1 for i, x in enumerate(b)
                         if x == (v0 if i < cut else not v0))
                best = max(best, ok, len(b) - ok)
        endmap_viol += len(b) - best
        endmap_n += len(b)

    ecount = {
        "vod": a.vod, "event": a.event, "date": a.date,
        "match_id": a.match_id,
        "court_residual_ft": (f"{court['residual_ft']:.3f}"
                              if court["residual_ft"] is not None else ""),
        "court_within_half_ft": (f"{court['within_half_ft']:.2f}"
                                 if court["within_half_ft"] is not None
                                 else ""),
        "n_rallies_timeline": len(windows),
        "n_rallies_covered": n_covered,
        "n_rallies_dropped": sum(dropped.values()),
        "rally_drop_reasons": ";".join(f"{k}:{v}" for k, v in
                                       sorted(dropped.items())),
        "backend": backend,
        "frames_kept": frames_kept,
        "det_gate_drops": ";".join(f"{k}:{v}" for k, v in
                                   sorted(det_drops.items())),
        "camera_gate": ("OFF" if cam is None
                        else f"{float(cam[1].mean()):.2f}_main"),
        "anchor_offset_med_s": (f"{np.median(anchor_offsets):.1f}"
                                if anchor_offsets else ""),
        "endmap_check": (f"{endmap_n - endmap_viol}/{endmap_n}"
                         if endmap_n else ""),
        "halves_check": (f"{halves_ok}/{halves_tested}"
                         if halves_tested else ""),
        "gender_check": (f"{gender_agree}/{gender_tested}"
                         if gender_tested else ""),
        "spotcheck_rallies": len(spot),
        "spotcheck_swaps": sum(1 for v in spot.values() if v > 0),
        "thin_player_games": n_thin,
    }
    if write:
        upsert_csv(DATA / "coverage_players.csv", prow,
                   ("vod", "match_id", "game", "player_uuid"))
        upsert_csv(DATA / "coverage_events.csv", [ecount],
                   ("vod", "match_id"))
    print(f"covered {n_covered}/{len(windows)} rallies; "
          f"dropped: {dict(dropped)}")
    if swaps:
        print(f"anchor-identity swaps applied: {n_swapped}")
    if track_map:
        print(f"track-map overrides applied (detections): {dict(n_tm)}")
    if endmap_n:
        print(f"end-map consistency {endmap_n - endmap_viol}/{endmap_n}")
    if halves_tested:
        print(f"lineup-halves check {halves_ok}/{halves_tested}")
    if gender_tested:
        print(f"mixed gender-height check {gender_agree}/{gender_tested}")
    print(f"{len(prow)} player-game rows -> data/coverage_players.csv"
          if write else
          f"{len(prow)} player-game rows computed (write=False, "
          f"nothing persisted)")
    return prow, ecount


def upsert_csv(path, rows, key_cols):
    if not rows:
        return
    old = []
    if path.exists():
        old = list(csv.DictReader(open(path)))
    keys = {tuple(r[k] for k in key_cols) for r in rows}
    keep = [r for r in old
            if tuple(str(r.get(k, "")) for k in key_cols) not in
            {tuple(str(v) for v in k) for k in keys}]
    allr = keep + [{k: str(v) for k, v in r.items()} for r in rows]
    # column union (new schema first) so re-runs with an evolved schema
    # never silently drop the older rows' columns
    cols = list(dict.fromkeys(
        list(rows[0].keys()) + (list(old[0].keys()) if old else [])))
    for r in allr:
        for c in cols:
            r.setdefault(c, "")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allr)


# ------------------------------------------------- anchor validation


def validate_anchor(a):
    """Anchor-finder error vs the hand-stamped Chicago serve pins — the
    spec's pre-scale-out validation, runnable wherever the Gate C pose
    npzs exist (data/vision/pose/, extracted from the Chicago VOD; fit
    the court once with court.py and pass its json).

        python vision/coverage.py --validate-anchor \
            --pose-dir data/vision/pose --court chicago_court.json

    Ground truth: shot_labels_chicago0725.csv serve_time_s (rallies
    1-16; rally 3 excluded — its pin marks a broadcast replay, see
    data/vision/pin_realignment.md).  CAVEAT on reading the numbers:
    these label windows open only 1.5 s before the serve, so the
    visible pre-serve freeze is short — treat the error distribution as
    an UPPER bound for real coverage windows, whose pre-serve spans run
    6-20 s."""
    court = load_court(a.court)
    labels = ROOT / "data/vision/shot_labels_chicago0725.csv"
    v4 = Path(a.windows) if a.windows else \
        ROOT / "data/vision/rally_windows_chicago0725_v4.csv"
    pins = {}
    for r in csv.DictReader(open(labels)):
        cum = int(r["rally_cum"])
        if r.get("serve_time_s") and 1 <= cum <= 16 and cum != 3:
            pins[cum] = float(r["serve_time_s"])
    wins = {int(r["rally_cum"]): (float(r["t0s"]), float(r["t1s"]))
            for r in csv.DictReader(open(v4))}
    errs, missing, quals = [], [], []
    for cum, t_pin in sorted(pins.items()):
        npz = Path(a.pose_dir) / f"r{cum:04d}.npz"
        if not npz.exists() or cum not in wins:
            missing.append(cum)
            continue
        dets, _ = load_rally(npz, court, None)
        t0, t1 = wins[cum]
        t_found, qual, _ = find_serve(dets, t0, t1, 0.0)
        quals.append(qual)
        if qual > 0:
            errs.append(t_found - t_pin)
    if missing:
        print(f"missing pose/window for rallies {missing} — extract "
              f"them first (pose_extract.py --rallies "
              f"{','.join(str(m) for m in missing)})")
    if not errs:
        raise SystemExit("no rallies validated")
    e = np.array(errs)
    print(f"anchor finder vs {len(e)} hand-stamped serves "
          f"({sum(q == 0 for q in quals)} finder-failures dropped):")
    print(f"  error  median {np.median(e):+.2f}s  "
          f"IQR [{np.percentile(e, 25):+.2f}, {np.percentile(e, 75):+.2f}]  "
          f"max|.| {np.abs(e).max():.2f}s")
    print(f"  within 1s: {(np.abs(e) <= 1).sum()}/{len(e)}   "
          f"within 2s: {(np.abs(e) <= 2).sum()}/{len(e)}")
    print("  (upper bound: label windows open only 1.5 s pre-serve; "
          "coverage windows give the finder 6-20 s of freeze)")


# ------------------------------------------------------------ selftest


def synth_court():
    """Identity-ish homography: 30 px per ft, y flipped like an image."""
    H_c2i = np.array([[30.0, 0, 100], [0, 30.0, 50], [0, 0, 1]])
    return {"H": np.linalg.inv(H_c2i), "w": 1280, "h": 1440,
            "residual_ft": 0.05, "within_half_ft": 0.99}


def selftest():
    rng = np.random.default_rng(5)
    court = synth_court()

    # ---- projection round trip ---------------------------------------
    pts_ft = np.array([[0, 0], [20, 44], [10, 22], [3.5, 29.0]])
    px = np.hstack([pts_ft, np.ones((4, 1))]) @ np.linalg.inv(court["H"]).T
    px = px[:, :2] / px[:, 2:3]
    back = project(court, px, court["w"], court["h"])
    assert np.allclose(back, pts_ft, atol=1e-6)
    half = project(court, px / 2, court["w"] / 2, court["h"] / 2)
    assert np.allclose(half, pts_ft, atol=1e-6), "frame rescale broken"
    print("  projection + frame-size rescale OK")

    # ---- foot point ---------------------------------------------------
    kpt = np.zeros((17, 2), np.float32)
    kpc = np.zeros(17, np.float32)
    box = np.array([100, 100, 160, 300], np.float32)
    x, y, src = foot_point(box, kpt, kpc)
    assert src == "box" and x == 130 and y == 300
    kpt[15], kpt[16] = (120, 290), (140, 294)
    kpc[15] = kpc[16] = 0.9
    x, y, src = foot_point(box, kpt, kpc)
    assert src == "ankles" and x == 130 and y == 292
    print("  foot point (ankles/box fallback) OK")

    # ---- serving configuration ---------------------------------------
    # near team serving: both near players deep (y ~ 43-46), far team:
    # one at kitchen (y ~ 16), receiver deep DIAGONAL to the server
    pos = {1: (14.0, 45.5), 2: (6.0, 43.0),      # near pair, deep
           3: (13.0, 16.0), 4: (7.0, 1.0)}      # far: kitchen + receiver
    end, margin, roles = serving_config(pos)
    assert end == "near" and roles["server"] == 1 and roles["receiver"] == 4 \
        and roles["rcv_kitchen"] == 3, (end, roles)
    # receiving-side flip
    pos2 = {1: (14.0, 30.5), 2: (6.0, 43.5),     # near: kitchen + deep
            3: (7.0, 0.5), 4: (13.0, 2.0)}      # far pair deep
    end2, _, roles2 = serving_config(pos2)
    assert end2 == "far" and roles2["rcv_kitchen"] == 1, (end2, roles2)
    # STACKED serve: both serving-end players deep on the diagonal half,
    # partner parked OUTSIDE the sideline extension -> lateral bounds
    # resolve it (not depth), unambiguously
    pos3 = {1: (15.0, 45.5), 2: (20.8, 45.9),    # partner outside + deeper
            3: (13.0, 16.0), 4: (7.0, 1.0)}
    end3, _, roles3 = serving_config(pos3)
    assert end3 == "near" and roles3["server"] == 1 \
        and not roles3["srv_amb"], (end3, roles3)
    # stack with BOTH inside the lines: geometry cannot say -> depth
    # tie-break, flagged ambiguous
    pos4 = {1: (15.0, 44.5), 2: (12.0, 46.0),
            3: (13.0, 16.0), 4: (7.0, 1.0)}
    _, _, roles4 = serving_config(pos4)
    assert roles4["srv_amb"] and roles4["server"] == 2, roles4
    print("  serving-end + roles OK both ways; stacked serve resolved "
          "by sideline bound, residual ambiguity flagged")

    # ---- synthetic rally: anchor finder + identity + metrics ---------
    fps = 10.0
    t_serve_true = 6.0
    t0, t1 = 0.0, 26.0
    uuids = {1: "srv", 2: "srvp", 3: "rcvp", 4: "rcv"}
    genders = {"srv": "M", "srvp": "F", "rcv": "F", "rcvp": "M"}
    start = {1: (14.0, 45.5), 2: (6.0, 43.0),       # server near-right,
             3: (13.0, 16.0), 4: (7.0, 1.0)}        # receiver DIAGONAL
    # after the serve everyone converges to the kitchen over ~6 s and
    # then holds with jitter; the server hangs slightly left-of-partner
    target = {1: (13.0, 31.0), 2: (5.0, 30.5),
              3: (14.0, 14.0), 4: (6.0, 13.5)}
    rows = []
    heights = {1: 240.0, 2: 200.0, 3: 235.0, 4: 205.0}
    for k in range(int((t1 - t0) * fps)):
        t = t0 + k / fps
        for tr in (1, 2, 3, 4):
            if t < t_serve_true:
                x, y = start[tr]
                x += rng.normal(0, 0.12)
                y += rng.normal(0, 0.12)
            else:
                f = min((t - t_serve_true) / 6.0, 1.0)
                x = start[tr][0] + f * (target[tr][0] - start[tr][0])
                y = start[tr][1] + f * (target[tr][1] - start[tr][1])
                x += rng.normal(0, 0.25)
                y += rng.normal(0, 0.25)
            side = 0 if y > NET_Y else 1
            rows.append(Det(t, tr, side, 0.9, (x, y), "ankles",
                            heights[tr], None, None, None))
    ts, qual, end = find_serve(rows, t0, t1, lead_s=6.0)
    assert abs(ts - t_serve_true) < 1.2, f"anchor finder off: {ts}"
    assert qual > 0.4 and end == "near"
    print(f"  anchor finder: t_serve {ts:.1f} vs true {t_serve_true} "
          f"(qual {qual:.2f}) OK")

    # the window-overlap trap: prepend the PREVIOUS rally's pre-serve
    # freeze (LONGER than this rally's) + its play; the finder must
    # anchor on the LAST freeze, not the longest
    prev = []
    for k in range(int(10 * fps)):     # 10 s: 6 s freeze + 4 s play
        t = -10.0 + k / fps
        for tr in (1, 2, 3, 4):
            if t < -4.0:               # previous rally's long freeze
                x, y = start[tr]
                x += rng.normal(0, 0.1)
                y += rng.normal(0, 0.1)
            else:                      # previous rally's play: motion
                x = start[tr][0] + rng.normal(0, 0.2)
                y = start[tr][1] + (t + 4.0) * 3.0 * (1 if tr < 3 else -1)
            side = 0 if y > NET_Y else 1
            prev.append(Det(t, tr + 10, side, 0.9, (x, y), "ankles",
                            heights[tr], None, None, None))
    ts2, qual2, _ = find_serve(prev + rows, -10.0, t1, lead_s=6.0)
    assert abs(ts2 - t_serve_true) < 1.2, \
        f"finder anchored on the previous rally's freeze: {ts2}"
    print(f"  anchor finder ignores the previous rally's longer freeze "
          f"({ts2:.1f}s) OK")

    win = {"server_uuid": "SRV", "receiver_uuid": "RCV",
           "game": "1", "rally_in_game": "1"}
    lin = {"team_A_R": "srv", "team_A_L": "srvp",
           "team_B_R": "rcv", "team_B_L": "rcvp",
           "server_half": "R", "game": "1", "rally": "1",
           "receiver_ok": "1"}
    nm, conf, checks = anchor_identity(rows, ts, win, lin, genders,
                                       heights)
    assert nm == {1: "srv", 2: "srvp", 4: "rcv", 3: "rcvp"}, nm
    assert conf >= CONF_MIN and checks["diagonal"] == 1
    assert checks["gender"] == "2/2"
    print(f"  identity chain: all four named, conf {conf:.2f}, "
          f"diagonal + gender checks pass")

    # handoff: track 4 dies at t=14, track 9 appears at 14.3 nearby
    rows2 = [d for d in rows if not (d.track == 4 and d.t > 14.0)]
    for d in rows:
        if d.track == 4 and d.t > 14.3:
            rows2.append(Det(d.t, 9, d.side, d.conf, d.xy, d.src,
                             d.h_px, None, None, None))
    rows2.sort(key=lambda d: d.t)
    assign = carry_names(rows2, nm, conf)
    got = {(d.track, u) for d, (u, c, h) in zip(rows2, assign)}
    assert (9, "rcv") in got, "handoff failed"
    assert all(u == "rcv" for d, (u, c, h) in zip(rows2, assign)
               if d.track == 9)
    print("  name handoff across a track break OK")

    # STALE LEDGER: a name for a track id with no detections in this
    # rally must be ignored, not crash.  Identity ledgers are keyed on
    # pose track ids that only exist inside the gitignored pose dir, so
    # a re-extraction makes exactly this shape of input (2026-08-20:
    # it raised KeyError in carry_names).
    nm_stale = dict(nm)
    nm_stale[9999] = "ghost"
    assign = carry_names(rows2, nm_stale, conf)
    assert all(u != "ghost" for (u, c, h) in assign), \
        "a name for an absent track leaked onto real detections"
    got2 = {(d.track, u) for d, (u, c, h) in zip(rows2, assign)}
    assert (9, "rcv") in got2, "stale entry broke a legitimate handoff"
    print("  stale ledger entry (named track with no detections) ignored OK")

    # speed gate: teleporting detection dropped
    class Z:                            # minimal npz stand-in
        pass
    # (exercised through load_rally in integration; unit: direct check)
    # ellipse area on a known Gaussian: pi * chi2 * sigma^2
    sig = 2.0
    pts = rng.normal(0, sig, (4000, 2))
    a = ellipse_area(pts, np.ones(len(pts)))
    want = np.pi * ELLIPSE_CHI2 * sig * sig
    assert abs(a - want) / want < 0.08, (a, want)
    print(f"  ellipse area {a:.1f} vs analytic {want:.1f} OK")

    # width share: A at x centered 6, B at 14 -> boundary 10 -> 0.5/0.5;
    # A at 2, B at 10 -> boundary 6 -> A 0.30, B 0.70
    tsx = np.arange(100) / 10.0
    xa = np.stack([np.full(100, 2.0), np.full(100, 30.0)], 1)
    xb = np.stack([np.full(100, 10.0), np.full(100, 30.0)], 1)
    s, n, sk, nk = rally_share(tsx, xa, tsx, xb, 29.0)
    assert abs(s / n - 0.30) < 1e-9 and nk == 100
    print("  width share definition OK (0.30/0.70 split reproduced)")

    # smoothing: static point with noise -> static_frac high
    tt = np.arange(60) / 10.0
    noisy = np.full((60, 2), 5.0) + rng.normal(0, 0.3, (60, 2))
    sm = smooth_series(tt, noisy)
    v = np.linalg.norm(np.diff(sm, axis=0), axis=1) / np.diff(tt)
    assert np.median(v) < STATIC_FTS, "smoothing insufficient for static"
    print("  static detection under 10 Hz foot noise OK")
    # end-segment fitting: a real mid-game switch is found, a lone odd
    # rally is NOT promoted to a segment (the rally-107 failure mode)
    switch = [(i, "AB" if i < 10 else "CD") for i in range(20)]
    segs = fit_end_segments(switch)
    assert segs[0] == 0 and segs[19] == 1, segs
    assert sum(v == 0 for v in segs.values()) == 10, segs
    outlier = [(0, "CD")] + [(i, "AB") for i in range(1, 12)]
    segs2 = fit_end_segments(outlier)
    assert set(segs2.values()) == {0}, "a lone odd rally faked a switch"
    steady = [(i, "AB") for i in range(20)]
    assert set(fit_end_segments(steady).values()) == {0}
    short = [(i, "AB" if i < 2 else "CD") for i in range(8)]
    assert set(fit_end_segments(short).values()) == {0}, \
        "a 2-rally segment is below the floor and must not count"
    print("  end segments: switch found, lone outlier + short segment "
          "rejected")
    # write=False must persist nothing: guards collect-only callers
    import inspect
    src = inspect.getsource(run)
    assert "if write:" in src, "run() lost its write guard"
    assert src.index("if write:") < src.index('coverage_players.csv"'), \
        "the players upsert is no longer inside the write guard"
    assert "write=True" in inspect.signature(run).__str__().replace(" ", "") \
        or run.__defaults__ == (None, True), run.__defaults__
    print("  run(write=False) guard in place")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan-camera", type=Path, metavar="VIDEO")
    ap.add_argument("--cam-out", type=Path, default=Path("camera_mask.csv"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--validate-anchor", action="store_true",
                    help="anchor-finder error vs the Chicago serve pins "
                         "(needs --pose-dir of Gate C npzs + --court)")
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--no-cam-gate", action="store_true",
                    help="explicitly run without the main-camera gate "
                         "(recorded as OFF in the events ledger)")
    ap.add_argument("--spotcheck", default="")
    ap.add_argument("--swaps", default="",
                    help="identity_swaps CSV from coverage_appearance "
                         "--audit; anchor names swap back per ledger")
    ap.add_argument("--anchor-free", default="",
                    help="identity_anchorfree_<vod>.csv — names rallies "
                         "whose serve was never shown, from an appearance "
                         "model trained on the rallies that DO have "
                         "serves (coverage_anchorfree.py). Flagged "
                         "identity_source=appearance downstream.")
    ap.add_argument("--track-map", default="",
                    help="identity_track_map CSV from coverage_appearance "
                         "--stage2; per-span rebind/rescue/split")
    ap.add_argument("--vod", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--match-id", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.scan_camera:
        scan_camera(a.scan_camera, a.cam_out)
    elif a.validate_anchor:
        if not (a.pose_dir and a.court):
            ap.error("--validate-anchor needs --pose-dir and --court")
        validate_anchor(a)
    elif a.run:
        for req in ("pose_dir", "court", "windows", "lineup", "vod"):
            if not getattr(a, req):
                ap.error(f"--run needs --{req.replace('_', '-')}")
        run(a)
    else:
        ap.error("pick --selftest, --scan-camera or --run")


if __name__ == "__main__":
    main()
