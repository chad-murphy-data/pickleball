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
from collections import defaultdict
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
    """-> (t array, is_main array) or None."""
    if not path or not Path(path).exists():
        return None
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
    __slots__ = ("t", "track", "side", "conf", "xy", "src", "h_px", "kpt",
                 "kpc", "box")

    def __init__(self, t, track, side, conf, xy, src, h_px, kpt, kpc, box):
        self.t, self.track, self.side, self.conf = t, track, side, conf
        self.xy, self.src, self.h_px = xy, src, h_px
        self.kpt, self.kpc, self.box = kpt, kpc, box


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
        dets.append(Det(t, tr, int(z["side"][i]), float(z["conf"][i]),
                        (float(x), float(y)), srcs[i],
                        float(z["box"][i][3] - z["box"][i][1]),
                        z["kpt"][i], z["kpc"][i], z["box"][i]))
    # side sanity per track
    ys = defaultdict(list)
    sides = {}
    for d in dets:
        ys[d.track].append(d.xy[1])
        sides[d.track] = d.side
    bad = set()
    for tr, yy in ys.items():
        court_side = 0 if float(np.median(yy)) > NET_Y else 1
        if sides.get(tr, -1) >= 0 and sides[tr] != court_side:
            bad.add(tr)
    if bad:
        kept = [d for d in dets if d.track not in bad]
        drops["side"] += len(dets) - len(kept)
        dets = kept
    return dets, drops


def by_frame(dets):
    fr = defaultdict(list)
    for d in dets:
        fr[round(d.t, 3)].append(d)
    return dict(sorted(fr.items()))


# ------------------------------------------------------- serve anchoring


def track_positions_at(dets, t_lo, t_hi, min_n=2):
    """track -> median court position over [t_lo, t_hi] (sided only)."""
    acc = defaultdict(list)
    for d in dets:
        if t_lo <= d.t <= t_hi and d.side >= 0:
            acc[d.track].append(d.xy)
    return {tr: (float(np.median([p[0] for p in v])),
                 float(np.median([p[1] for p in v])))
            for tr, v in acc.items() if len(v) >= min_n}


def d_kitchen(xy, end):
    return abs(xy[1] - KITCHEN_Y[end])


def serving_config(pos):
    """Score the two serving-end hypotheses on a position snapshot.

    pos: track -> (x, y).  Returns (end, margin, roles) where roles =
    dict(server=track, srv_partner=track, receiver=track,
    rcv_kitchen=track); or (None, 0, {}) when the snapshot cannot say
    (missing players, nobody at the kitchen, ...).  The margin is the
    weakest of the geometric facts the winning hypothesis asserts, in
    feet — 0 or negative means unconvinced."""
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
        # to the receiver (opposite lateral half).  Under stacking the
        # partner can be exactly as deep as the server, so depth alone
        # coin-flips — the diagonal is the rule-determined discriminator;
        # depth breaks ties only when both serving-end players sit on the
        # receiver's own half (server mid-crossover, rare).
        rx = R[receiver][0]
        diag = [tr for tr, p in S.items()
                if (p[0] - W_FT / 2) * (rx - W_FT / 2) < 0]
        server = diag[0] if len(diag) == 1 else max(dS, key=dS.get)
        margin = min(min(dS.values()) - DEEP_FT,          # both srv deep
                     RCV_KITCHEN_FT - dR[rcv_kitchen],    # one rcv at NVZ
                     dR[receiver] - DEEP_FT)              # receiver deep
        roles = {"server": server,
                 "srv_partner": next(tr for tr in S if tr != server),
                 "receiver": receiver, "rcv_kitchen": rcv_kitchen}
        return margin, roles

    m_near, r_near = hyp("near")
    m_far, r_far = hyp("far")
    if m_near <= 0 and m_far <= 0:
        return None, 0.0, {}
    if m_near >= m_far:
        return "near", m_near - max(m_far, 0.0), r_near
    return "far", m_far - max(m_near, 0.0), r_far


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
        pos_seq.append(pos)
        ok_seq.append((end, margin))
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
    # Noise guard: a last "run" under 3 frames falls back to the longest.
    runs, cur = [], [good[0]]
    for k in good[1:]:
        if ts[k] - ts[cur[-1]] <= 0.7:
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)
    best = runs[-1] if len(runs[-1]) >= 3 else max(runs, key=len)
    k_end = best[-1]
    margin = float(np.median([ok_seq[k][1] for k in best]))
    qual = min(1.0, len(best) / 8.0) * min(1.0, max(margin, 0.0) / 2.0 + 0.5)
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
    end, margin, roles = serving_config(pos)
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
            checks["gender"] = f"{agree}/{tested}"
            if agree < tested:
                conf *= 0.8
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
        self.det_n = 0
        self.conf_sum = 0.0
        self.share_num = 0.0   # width share accumulator (frame count)
        self.share_n = 0
        self.share_k_num = 0.0
        self.share_k_n = 0
        self.partner_area_pts = None   # filled at game level

    def add_rally(self, ts, xy, conf, wts, handoffs, phase_mask, serve_mask,
                  end):
        """ts sorted; xy (n,2); phase_mask = rally-phase frames."""
        self.rallies.add(True)
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


def player_meta():
    g, nm = {}, {}
    f = DATA / "players.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            pid = r["player_id"].lower()
            g[pid] = r["gender"]
            nm[pid] = r["full_name"]
    return g, nm


def run(a):
    court = load_court(a.court)
    cam = load_camera(a.cam)
    windows = load_windows(a.windows)
    lineup_rows, lineup_by, lineup_ids = load_lineup(a.lineup)
    genders, names = player_meta()
    pose_dir = Path(a.pose_dir)
    spot = {}
    if a.spotcheck and Path(a.spotcheck).exists():
        for r in csv.DictReader(open(a.spotcheck)):
            if r.get("watched", "0") == "1":
                spot[int(r["rally_cum"])] = int(r.get("swaps_seen", 0) or 0)

    games = defaultdict(lambda: defaultdict(PlayerGame))
    endmap_obs = []                   # (id8, game, cum, serving team, end)
    dropped = defaultdict(int)        # rally-level drops, by reason
    det_drops = defaultdict(int)      # detection-level gate drops
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
        if not dets:
            dropped["no_detections"] += 1
            continue
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        lead = float(win["lead_s"]) if win.get("lead_s") else 0.0
        t_serve, qual, srv_end = find_serve(dets, t0, t1, lead)
        if qual == 0.0:
            # no anchor found: the fallback guess cannot place the serve
            # (in start-marked logs dur ~= lead, so t0+lead sits at the
            # rally END).  Ambiguous spans are dropped, not guessed.
            dropped["anchor_not_found"] += 1
            continue
        anchor_offsets.append(t_serve - t0)
        lin, id8 = lineup_for(win, lineup_by, lineup_ids)
        heights = defaultdict(list)
        for d in dets:
            heights[d.track].append(d.h_px)
        heights = {tr: float(np.median(v)) for tr, v in heights.items()}
        names_map, conf, checks = anchor_identity(
            dets, t_serve, win, lin, genders, heights)
        if names_map is None or conf < CONF_MIN:
            dropped["identity_" + checks.get("reason", "lowconf")] += 1
            continue
        # lineup-halves consistency (report-only): predicted lateral half
        # of the server vs observed, weighted by the machine's local
        # receiver_ok agreement around this rally
        if lin is not None:
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

        endmap_obs.append((id8, win["game"], int(cum),
                           "A" if win["server_uuid"].lower() in
                           (lin["team_A_R"], lin["team_A_L"]) else "B",
                           srv_end or "?"))
        assign = carry_names(sorted(dets, key=lambda d: d.t), names_map, conf)
        dets_sorted = sorted(dets, key=lambda d: d.t)
        per_uuid = defaultdict(lambda: ([], [], [], []))  # ts, xy, w, hand
        for d, (u, c, hand) in zip(dets_sorted, assign):
            if u is None:
                dropped["unnamed_dets"] += 1
                continue
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
            phase = ts >= t_serve + SERVE_PHASE_S
            serve_m = (ts >= t_serve) & (ts < t_serve + SERVE_PHASE_S)
            end = "near" if float(np.median(xy[:, 1])) > NET_Y else "far"
            games[game][u].add_rally(ts, xy, conf, wts, hand, phase,
                                     serve_m, end)
            rally_data[u] = (ts[phase], xy[phase], end)
        rally_tracks_by_game[game].append((cum, rally_data, lin))

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
        "det_gate_drops": ";".join(f"{k}:{v}" for k, v in
                                   sorted(det_drops.items())),
        "camera_gate": ("off" if cam is None
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
    upsert_csv(DATA / "coverage_players.csv", prow,
               ("vod", "match_id", "game", "player_uuid"))
    upsert_csv(DATA / "coverage_events.csv", [ecount], ("vod",))
    print(f"covered {n_covered}/{len(windows)} rallies; "
          f"dropped: {dict(dropped)}")
    if endmap_n:
        print(f"end-map consistency {endmap_n - endmap_viol}/{endmap_n}")
    if halves_tested:
        print(f"lineup-halves check {halves_ok}/{halves_tested}")
    if gender_tested:
        print(f"mixed gender-height check {gender_agree}/{gender_tested}")
    print(f"{len(prow)} player-game rows -> data/coverage_players.csv")
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
    print("  serving-end + roles OK both ways")

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
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan-camera", type=Path, metavar="VIDEO")
    ap.add_argument("--cam-out", type=Path, default=Path("camera_mask.csv"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--spotcheck", default="")
    ap.add_argument("--vod", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--match-id", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.scan_camera:
        scan_camera(a.scan_camera, a.cam_out)
    elif a.run:
        for req in ("pose_dir", "court", "windows", "lineup", "vod"):
            if not getattr(a, req):
                ap.error(f"--run needs --{req.replace('_', '-')}")
        run(a)
    else:
        ap.error("pick --selftest, --scan-camera or --run")


if __name__ == "__main__":
    main()
