"""Verification overlay — the identity layer's human instrument.

Renders an annotated copy of rally footage the user can just watch:
boxes + skeletons on all four players, each labeled with the RESOLVED
NAME (team-colored), track id and near/far side, the foot point, and a
schematic-court inset where the four projected dots move in court feet
(the inset verifies homography AND identity at once — court coordinates
are where every coverage metric lives).  Labels DIM when identity
confidence is low, so the eye goes straight to the machine's guesses;
excluded spans (pre-serve, serve phase, non-main camera, dropped
rallies) carry visible banners.

The one failure coverage cannot self-detect is a within-side partner
swap mid-rally; a name label jumping between partners is instantly
visible to a human who knows the players.  --sample N makes the check a
MEASUREMENT: N random rallies per game plus a spotcheck CSV template
(rally, watched, swaps_seen) whose filled-in copy coverage.py folds
into data/coverage_players.csv as the identity error rate.

    python vision/coverage_overlay.py --video vod.mp4 \
        --pose-dir data/vision/pose_x --court court.json \
        --windows coverage_windows_x.csv --lineup lineup_<id8>.csv \
        --out overlay.mp4 [--rallies 5,6,7 | --sample 10] [--half-speed]
    python vision/coverage_overlay.py --selftest

Overlay videos are broadcast-derived imagery: LOCAL ONLY, never
committed (same rule as data/vision/*.png).  Uses the identical gated
detections and identity code path as coverage.py — the overlay verifies
the exact frames the metrics are computed from.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import (CONF_MIN, L_FT, NET_Y, SERVE_PHASE_S, W_FT,
                      anchor_identity, by_frame, carry_names, find_serve,
                      is_main_at, lineup_for, load_camera, load_court,
                      load_lineup, load_rally, load_windows, player_meta)
from swing_probe import decode_window, ffmpeg_bin

# COCO-17 skeleton
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

TEAM_COLORS = {"A": (60, 80, 235), "B": (235, 160, 40)}   # BGR: red / blue
DIM = (140, 140, 140)
INSET_W, INSET_H, INSET_PAD = 120, 240, 12


def team_of(uuid, lin):
    if lin is None or uuid is None:
        return None
    if uuid in (lin["team_A_R"], lin["team_A_L"]):
        return "A"
    if uuid in (lin["team_B_R"], lin["team_B_L"]):
        return "B"
    return None


def draw_inset(frame, positions, H_img, W_img):
    """Schematic 20x44 court, top-right; positions = [(x_ft, y_ft, color,
    initial)]."""
    import cv2
    x0 = W_img - INSET_W - INSET_PAD
    y0 = INSET_PAD
    cv2.rectangle(frame, (x0, y0), (x0 + INSET_W, y0 + INSET_H),
                  (30, 30, 30), -1)
    def to_px(xf, yf):
        return (x0 + int(xf / W_FT * (INSET_W - 12)) + 6,
                y0 + int(yf / L_FT * (INSET_H - 12)) + 6)
    white = (220, 220, 220)
    cv2.rectangle(frame, to_px(0, 0), to_px(W_FT, L_FT), white, 1)
    for y in (15.0, NET_Y, 29.0):
        cv2.line(frame, to_px(0, y), to_px(W_FT, y),
                 white if y != NET_Y else (120, 220, 255), 1)
    cv2.line(frame, to_px(W_FT / 2, 0), to_px(W_FT / 2, 15.0), white, 1)
    cv2.line(frame, to_px(W_FT / 2, 29.0), to_px(W_FT / 2, L_FT), white, 1)
    for xf, yf, color, init in positions:
        xf = float(np.clip(xf, -1.5, W_FT + 1.5))
        yf = float(np.clip(yf, -3.0, L_FT + 3.0))
        p = to_px(xf, yf)
        cv2.circle(frame, p, 5, color, -1)
        cv2.putText(frame, init, (p[0] + 6, p[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_banner(frame, text, color=(0, 0, 200)):
    import cv2
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, H - 34), (W, H), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def draw_detection(frame, det, name, color, dim, court):
    import cv2
    c = DIM if dim else color
    x0, y0, x1, y1 = [int(v) for v in det.box[:4]]
    cv2.rectangle(frame, (x0, y0), (x1, y1), c, 1 if dim else 2)
    if det.kpt is not None:
        for a, b in EDGES:
            if det.kpc[a] >= 0.3 and det.kpc[b] >= 0.3:
                pa = tuple(int(v) for v in det.kpt[a])
                pb = tuple(int(v) for v in det.kpt[b])
                cv2.line(frame, pa, pb, c, 1, cv2.LINE_AA)
    # foot point (pixel space: reproject court xy is overkill — redraw
    # from the same foot_point the metrics used is what load_rally did;
    # here the projected court position is shown in the inset instead)
    fy = int(det.box[3])
    fx = int((det.box[0] + det.box[2]) / 2)
    cv2.circle(frame, (fx, fy), 3, c, -1)
    side = {0: "N", 1: "F"}.get(det.side, "?")
    label = f"{name or '?'} t{det.track}/{side}"
    cv2.putText(frame, label, (x0, max(14, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1 if dim else 2,
                cv2.LINE_AA)


def render_rally(video, cum, win, dets, assign_by_id, conf, t_serve,
                 status, lin, names_full, court, fps, width, writer,
                 cam=None, legend=None):
    """Draw one rally's frames into the ffmpeg writer."""
    import cv2
    t0, t1 = float(win["t0s"]), float(win["t1s"])
    frames_dets = by_frame(dets)
    det_times = np.array(sorted(frames_dets))
    for i, frame in enumerate(decode_window(video, t0, t1 - t0, fps, width)):
        t = t0 + i / fps
        frame = frame.copy()       # decode_window yields read-only buffers
        H_img, W_img = frame.shape[:2]
        cv2.putText(frame, f"rally {cum}  game {win['game']}  "
                           f"{win['start_score']}  t={t:7.1f}s",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 2, cv2.LINE_AA)
        if legend:
            # jersey-description lines in each team's LABEL color, so
            # the coder can tie label color to what the players wear;
            # right-aligned under the court inset — top-left belongs to
            # the broadcast's own scorebug (measured collision)
            y = INSET_PAD + INSET_H + 24
            for k, (txt, col) in enumerate(legend):
                if txt:
                    (tw, _), _ = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.putText(frame, txt, (W_img - 12 - tw, y + 20 * k),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2,
                                cv2.LINE_AA)
        near = []
        if len(det_times):
            j = int(np.argmin(np.abs(det_times - t)))
            if abs(det_times[j] - t) <= 0.6 / fps + 1e-9:
                near = frames_dets[det_times[j]]
        inset = []
        for d in near:
            u = assign_by_id.get(id(d), (None, 0.0, False))[0]
            tm = team_of(u, lin)
            color = TEAM_COLORS.get(tm, DIM)
            dim = (u is None or conf < CONF_MIN
                   or assign_by_id.get(id(d), (None, 0.0, False))[2])
            nm = (names_full.get(u) or u[:8]).split()[-1] if u else None
            draw_detection(frame, d, nm, color, dim, court)
            inset.append((d.xy[0], d.xy[1],
                          DIM if dim else color, (nm or "?")[:1]))
        draw_inset(frame, inset, H_img, W_img)
        if status == "dropped":
            draw_banner(frame, "IDENTITY LOW CONFIDENCE - RALLY DROPPED")
        elif status == "anchor":
            draw_banner(frame, "SERVE ANCHOR NOT FOUND - RALLY DROPPED")
        elif status == "approx":
            draw_banner(frame, "WINDOW APPROX (replay/missed flip) - "
                               "EXCLUDED")
        elif not is_main_at(cam, t):
            draw_banner(frame, "NON-MAIN CAMERA - frames excluded",
                        (0, 160, 255))
        elif t < t_serve:
            draw_banner(frame, "PRE-SERVE - excluded from coverage",
                        (0, 200, 200))
        elif t < t_serve + SERVE_PHASE_S:
            draw_banner(frame, "SERVE PHASE - reported separately",
                        (0, 200, 200))
        writer.stdin.write(frame.tobytes())


def run(a):
    court = load_court(a.court)
    cam = load_camera(a.cam)
    windows = load_windows(a.windows)
    lineup_rows, lineup_by, lineup_ids = load_lineup(a.lineup)
    genders, names_full = player_meta()
    pose_dir = Path(a.pose_dir)
    swaps = {}
    if getattr(a, "swaps", ""):
        import csv as _csv
        from collections import defaultdict as _dd
        swaps = _dd(list)
        for r in _csv.DictReader(open(a.swaps)):
            if r["swap"] == "1" and float(r["unanimity"]) >= 0.8:
                swaps[int(r["rally_cum"])].append(
                    tuple(r["team"].split("|")))
        print(f"swap ledger: {sum(len(v) for v in swaps.values())} "
              f"team-rally swaps")
    track_map = {}
    if getattr(a, "track_map", ""):
        import csv as _csv
        from collections import defaultdict as _dd
        track_map = _dd(lambda: _dd(list))
        for r in _csv.DictReader(open(a.track_map)):
            track_map[int(r["rally_cum"])][int(r["track"])].append(
                (float(r["t0"]), float(r["t1"]), r["uuid"], r["action"]))
        print(f"track map loaded")

    have = sorted(int(p.stem[1:]) for p in pose_dir.glob("r*.npz"))
    if a.rallies:
        todo = [int(x) for x in a.rallies.split(",") if int(x) in windows]
    elif a.sample:
        rng = np.random.default_rng(20260816)
        by_game = {}
        for c in have:
            if c in windows and windows[c]["approx"] == "0":
                by_game.setdefault(windows[c]["game"], []).append(c)
        todo = sorted(c for g, cs in by_game.items()
                      for c in rng.permutation(cs)[:a.sample])
        root = Path(__file__).resolve().parent.parent
        tpl = Path(a.spotcheck_out) if a.spotcheck_out else (
            root / f"data/vision/coverage_spotcheck_{a.vod or 'vod'}.csv")
        with open(tpl, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rally_cum", "game", "watched", "swaps_seen",
                        "notes"])
            for c in todo:
                w.writerow([c, windows[c]["game"], 0, "", ""])
        print(f"spotcheck template -> {tpl} (fill watched=1 + swaps_seen "
              f"while watching; coverage.py --spotcheck folds it in)")
    else:
        todo = [c for c in have if c in windows]
    todo = [c for c in todo if (pose_dir / f"r{c:04d}.npz").exists()]
    if not todo:
        raise SystemExit("no rallies to render (pose npz missing?)")
    print(f"rendering {len(todo)} rallies -> {a.out}")

    # writer geometry from the first NON-EMPTY npz (an empty rally npz
    # stores hw = (0, 0), which would hand ffmpeg '-s 1280x0')
    fps, width, height = 10.0, 1280, 720
    for c in todo:
        z0 = np.load(pose_dir / f"r{c:04d}.npz")
        if int(z0["hw"][0]) > 0:
            fps = float(z0["fps"][0])
            height, width = int(z0["hw"][0]), int(z0["hw"][1])
            break
    out_fps = fps / 2 if a.half_speed else fps
    writer = subprocess.Popen(
        [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
         "-r", str(out_fps), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         str(a.out)], stdin=subprocess.PIPE)

    for k, cum in enumerate(todo):
        win = windows[cum]
        dets, drops = load_rally(pose_dir / f"r{cum:04d}.npz", court, cam)
        lin, _id8 = lineup_for(win, lineup_by, lineup_ids)
        status = ""
        t_serve = float(win["t0s"])
        conf = 0.0
        assign_by_id = {}
        if win["approx"] == "1":
            status = "approx"
        elif dets:
            t0, t1 = float(win["t0s"]), float(win["t1s"])
            lead = float(win["lead_s"]) if win.get("lead_s") else 0.0
            t_serve, qual, _ = find_serve(dets, t0, t1, lead)
            # identical inputs to the metrics path: per-track MEDIAN
            # heights (the overlay must verify the frames being scored)
            hts = {}
            for d in dets:
                hts.setdefault(d.track, []).append(d.h_ft or d.h_px)
            hts = {tr: float(np.median(v)) for tr, v in hts.items()}
            nm, conf, checks = anchor_identity(
                dets, t_serve, win, lin, genders, hts)
            if nm is not None:
                # appearance-audited anchor swaps (same ledger the
                # metrics run consumes via coverage.run --swaps)
                for (ua, ub) in swaps.get(cum, ()):
                    inv = {u: tr for tr, u in nm.items()}
                    if ua in inv and ub in inv:
                        nm[inv[ua]], nm[inv[ub]] = ub, ua
            if qual == 0.0:
                status = "anchor"
            elif nm is None or conf < CONF_MIN:
                status = "dropped"
            else:
                srt = sorted(dets, key=lambda d: d.t)
                tm = track_map.get(cum, {}) if track_map else {}
                for d, tup in zip(srt, carry_names(srt, nm, conf)):
                    for (ta, tb, uu, _act) in tm.get(d.track, ()):
                        if ta <= d.t <= tb:
                            tup = (uu, tup[1], False)
                            break
                    assign_by_id[id(d)] = tup
        legend = [(a.legend_a, TEAM_COLORS["A"]),
                  (a.legend_b, TEAM_COLORS["B"])] \
            if (a.legend_a or a.legend_b) else None
        render_rally(a.video, cum, win, dets, assign_by_id, conf, t_serve,
                     status, lin, names_full, court, fps, width, writer,
                     cam, legend)
        print(f"  rally {cum} ({k + 1}/{len(todo)}) "
              f"{'[' + status + ']' if status else 'ok'}", flush=True)
    writer.stdin.close()
    writer.wait()
    print(f"wrote {a.out} (LOCAL ONLY - broadcast imagery, never commit)")


# ------------------------------------------------------------ selftest


def selftest():
    import cv2                                              # noqa: F401
    from coverage import Det
    frame = np.zeros((720, 1280, 3), np.uint8)
    d = Det(1.0, 3, 0, 0.9, (10.0, 30.0), "ankles", 200.0,
            np.random.rand(17, 2).astype(np.float32) * 200 + 100,
            np.full(17, 0.9, np.float32),
            np.array([100, 100, 200, 300], np.float32))
    draw_detection(frame, d, "Alshon", TEAM_COLORS["A"], False, None)
    assert frame.sum() > 0, "nothing drawn"
    bright = frame.copy()
    frame2 = np.zeros((720, 1280, 3), np.uint8)
    draw_detection(frame2, d, "Alshon", TEAM_COLORS["A"], True, None)
    assert frame2.sum() < bright.sum(), "dimmed draw should be fainter"
    draw_inset(bright, [(5.0, 40.0, TEAM_COLORS["A"], "A"),
                        (15.0, 2.0, TEAM_COLORS["B"], "B")], 720, 1280)
    corner = bright[INSET_PAD:INSET_PAD + INSET_H,
                    1280 - INSET_W - INSET_PAD:1280 - INSET_PAD]
    assert corner.sum() > 0, "inset not drawn"
    draw_banner(bright, "TEST")
    assert bright[-20:, :, :].sum() > 0
    lin = {"team_A_R": "u1", "team_A_L": "u2",
           "team_B_R": "u3", "team_B_L": "u4"}
    assert team_of("u2", lin) == "A" and team_of("u3", lin) == "B"
    assert team_of("zz", lin) is None
    print("SELFTEST OK (drawing + team colors + dimming + inset)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--out", default="coverage_overlay.mp4")
    ap.add_argument("--rallies", help="comma-separated rally_cum")
    ap.add_argument("--sample", type=int,
                    help="N random confident rallies per game + "
                         "spotcheck template")
    ap.add_argument("--spotcheck-out", default="")
    ap.add_argument("--vod", default="")
    ap.add_argument("--swaps", default="",
                    help="identity_swaps CSV (coverage_appearance "
                         "--audit)")
    ap.add_argument("--track-map", default="",
                    help="identity_track_map CSV (--stage2)")
    ap.add_argument("--legend-a", default="",
                    help="jersey-description line drawn in team A's "
                         "label color (match-specific, so a flag)")
    ap.add_argument("--legend-b", default="",
                    help="same for team B")
    ap.add_argument("--half-speed", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("video", "pose_dir", "court", "windows", "lineup"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_', '-')} required")
    run(a)


if __name__ == "__main__":
    main()
