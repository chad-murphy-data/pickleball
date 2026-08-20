"""Blind VLM LOCALIZATION test — the one that measures the binder.

The 2026-08-19 hitter test handed the model pre-located contacts, so it
measured RECOGNITION GIVEN PLACEMENT (95% side, 85% four-way, 75%
pace). Placement is the thing that actually binds: the decoded pipeline
matches only 45.7% of labeled contacts, and the whole temporal-model
program exists to fix that. This script builds the test that asks the
unanswered question — user proposal 2026-08-19, "was there a shot or
not, strip by strip".

Each question is a 3x3 GRID of 9 frames at 0.15 s spacing (1.2 s span),
cropped to the playing area. Grid rather than a vertical strip because
images are downscaled to ~1568 px on the long edge: a 9-frame vertical
stack renders each frame ~30 px tall and is useless, while a square-ish
grid keeps players ~80-100 px. 0.15 s sampling resolves a firefight
(contacts ~0.45 s apart land 3 cells apart) and is far finer than the
0.5 s matching tolerance.

WINDOWS ARE NOT CENTRED ON CONTACTS — that is the entire point. Starts
are drawn uniformly from [first_contact - PRE_PAD, last_contact], so
some windows land in pre-serve dead time and contain ZERO contacts.
Those are not filler: a scan that hallucinates shots in dead time is
useless, and only 0-contact windows measure that.

Read order is left-to-right, top-to-bottom: cell 1 = t0, cell 9 =
t0 + 1.2 s.

Usage (Mac, flat folder):
    python3 vlm_localize_sample.py --video full_match.mp4.webm
      -> vlm_loc/w01.png ..            paste these into the thread
      -> vlm_loc/ANSWER_KEY_LOC.csv    keep back until the calls are in
    python3 vlm_localize_sample.py --selftest      (no video needed)
"""
import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

from fastslow_check import classify_type
from swing_probe import ffmpeg_bin

LABELS = "contact_labels_chicago0725.csv"
SPLIT = "label_split.csv"
STEP_S = 0.15
PRE_PAD = 4.0          # sample this far before the serve (dead time)
LONG_EDGE = 1568       # where the API downscales to; see vlm_pack.py
MARK_BGR = (255, 0, 255)   # magenta: not the ball, not the court


def load_train_contacts(labels_path, split_path):
    """{rally_cum: [(t, hitter, pace_class), ...]} for TRAIN rallies,
    real contacts only (whiffs excluded — they are not shots)."""
    p = Path(split_path)
    if not p.exists():
        raise SystemExit(f"{split_path} not found — the split is "
                         f"mandatory; holdout must never be cut.")
    train = {int(r["rally_cum"]) for r in csv.DictReader(open(p))
             if r["split"] == "train"}
    out = {}
    for r in csv.DictReader(open(labels_path)):
        cum = int(r["rally_cum"])
        if cum not in train or r.get("contact", "1") == "0":
            continue
        out.setdefault(cum, []).append(
            (float(r["t_refined_s"] or r["t_tap_s"]), r["hitter_name"],
             classify_type(r["shot_type"])))
    for v in out.values():
        v.sort()
    return out


def sample_windows(contacts, n, seed, dur):
    """[(rally_cum, t0)] drawn uniformly over each rally's span extended
    PRE_PAD seconds earlier. Rallies are weighted by span so long
    rallies contribute proportionally, as a real scan would."""
    rng = random.Random(seed)
    spans = []
    for cum, cs in contacts.items():
        lo, hi = cs[0][0] - PRE_PAD, cs[-1][0]
        if hi > lo:
            spans.append((cum, lo, hi, hi - lo))
    total = sum(s[3] for s in spans)
    out = []
    for _ in range(n):
        x = rng.uniform(0, total)
        for cum, lo, hi, w in spans:
            if x <= w:
                out.append((cum, lo + rng.uniform(0, max(hi - lo, 1e-9))))
                break
            x -= w
    return out


def contacts_in(contacts, cum, t0, dur):
    """Contacts inside [t0, t0+dur), as (offset_s, hitter, pace)."""
    return [(round(t - t0, 3), h, p) for t, h, p in contacts[cum]
            if t0 <= t < t0 + dur]


def cut_grid(video, t0, out_path, grid, crop, markers=False):
    """grid x grid cells at STEP_S spacing, assembled to exactly the
    LONG_EDGE the API downscales to, so the file IS what the model sees.

    cv2 rather than ffmpeg: the marker arm has to draw on the frames,
    and routing every arm through one renderer keeps the ladder's rungs
    comparable. Frames are read sequentially — seeking per cell is
    several times slower and the marker arm needs the whole window
    decoded anyway.

    MARKERS COME FROM THE TRACKER, NOT THE CANDIDATE SET. Drawing the
    top-12 raw candidates was tried first and is actively harmful: they
    cluster on player limb motion and crowd movement, i.e. exactly
    where the eye already looks, and the ball's marker is lost among
    them. The tracker's surviving segments give ONE mark per frame that
    follows a ball-like path. Marks are drawn AFTER the downscale — a
    1 px stroke at source would vanish at 0.29x, which is the whole
    problem they exist to solve.

    Returns a per-cell '1'/'0' string of WHERE THE TRACKER PUT A MARK
    (empty when markers are off). Stored raw in the answer key rather
    than as a derived per-contact flag, so the tolerance used to join
    marks to contacts stays a scoring-time choice. This is what makes
    a miss attributable: a contact the tracker HAD and the reader still
    missed indicts the reading, one the tracker never had indicts the
    tracker, and a contact placed with no mark nearby was carried by
    posture alone."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cw_f, ch_f, cx_f, cy_f = crop
    x0, y0 = int(cx_f * W), int(cy_f * H)
    cw, ch = int(cw_f * W), int(ch_f * H)
    n_cells = grid * grid
    f_lo = max(int(round(t0 * fps)) - 1, 0)
    n_f = int(round((n_cells - 1) * STEP_S * fps)) + 3

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    buf = []
    for _ in range(n_f):
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(fr[y0:y0 + ch, x0:x0 + cw])
    cap.release()
    if len(buf) < n_f:
        raise SystemExit(f"ran off the end of the video at t0={t0:.2f}s")

    pos = {}
    if markers:
        from ball_candidates import candidates
        from ball_track import track_all
        cand = [candidates(buf[i - 1], buf[i], buf[i + 1])
                for i in range(1, len(buf) - 1)]
        # keyword, not positional: an older ball_track.py on disk has
        # (cand, max_tracks, fps) and silently takes fps as max_tracks
        for tr in track_all(cand, fps=fps):
            for f, x, y, seen in tr:
                if seen:
                    pos.setdefault(f, (x, y))

    cell_w = LONG_EDGE // grid
    cell_h = int(round(cell_w * ch / cw))
    sx, sy = cell_w / cw, cell_h / ch
    cells, marked = [], []
    for i in range(n_cells):
        fi = int(round(i * STEP_S * fps))       # index into cand/pos
        small = cv2.resize(buf[fi + 1], (cell_w, cell_h),
                           interpolation=cv2.INTER_AREA)
        marked.append("1" if fi in pos else "0")
        if fi in pos:
            x, y = pos[fi]
            cv2.circle(small, (int(x * sx), int(y * sy)), 8,
                       MARK_BGR, 1)
        cells.append(cv2.copyMakeBorder(small, 2, 2, 2, 2,
                                        cv2.BORDER_CONSTANT, value=(128,) * 3))
    g = np.vstack([np.hstack(cells[r * grid:(r + 1) * grid])
                   for r in range(grid)])
    cv2.imwrite(str(out_path), g)
    return "".join(marked) if markers else ""


def load_used(paths):
    """[(rally_cum, t0, span)] already spent on an earlier draw, so a new
    arm can be drawn off FRESH video instead of rescoring seen windows."""
    used = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            used.append((int(r["rally_cum"]), float(r["t0_s"]),
                         float(r.get("span_s") or 1.2)))
    return used


def overlaps(cum, t0, dur, used):
    return any(c == cum and t0 < ut + us and ut < t0 + dur
               for c, ut, us in used)


def selftest():
    cs = {1: [(10.0, "A", "slow"), (11.0, "B", "slow"), (11.4, "A", "fast"),
              (11.8, "B", "fast")],
          2: [(50.0, "C", "slow"), (58.0, "D", "slow")]}
    dur = (3 * 3 - 1) * STEP_S
    assert abs(dur - 1.2) < 1e-9, dur
    # packing: cells scale as n^2, delivered pixels per cell as 1/n
    for n in (3, 4, 5, 6):
        assert abs((n * n - 1) * STEP_S - {3: 1.2, 4: 2.25, 5: 3.6,
                                           6: 5.25}[n]) < 1e-9, n
        assert LONG_EDGE // n * n <= LONG_EDGE
    # exclusion is half-open interval overlap, per rally
    used = [(1, 10.0, 1.2)]
    assert overlaps(1, 10.5, 1.2, used) and overlaps(1, 9.5, 1.2, used)
    assert not overlaps(1, 11.2, 1.2, used)
    assert not overlaps(2, 10.5, 1.2, used)      # different rally
    # containment is half-open and offsets are relative
    got = contacts_in(cs, 1, 11.0, 1.2)
    assert [g[0] for g in got] == [0.0, 0.4, 0.8], got
    assert contacts_in(cs, 1, 6.0, 1.2) == []          # dead time
    assert len(contacts_in(cs, 1, 10.9, 1.2)) == 3
    # windows stay inside their extended span, and dead-time windows do occur
    w = sample_windows(cs, 400, 7, dur)
    assert len(w) == 400
    for cum, t0 in w:
        lo, hi = cs[cum][0][0] - PRE_PAD, cs[cum][-1][0]
        assert lo - 1e-9 <= t0 <= hi + 1e-9, (cum, t0)
    assert any(not contacts_in(cs, c, t, dur) for c, t in w), \
        "no zero-contact windows drawn — false-positive arm would be blind"
    assert any(len(contacts_in(cs, c, t, dur)) >= 2 for c, t in w)
    # long rally 2 (span 12s) should out-draw short rally 1 (span 5.4s)
    n2 = sum(1 for c, _ in w if c == 2)
    assert n2 > 200, n2
    # determinism, and a different seed really moves
    assert sample_windows(cs, 20, 7, dur) == sample_windows(cs, 20, 7, dur)
    assert sample_windows(cs, 20, 8, dur) != sample_windows(cs, 20, 7, dur)
    print("selftest: windows, containment, dead-time coverage, "
          "span weighting, determinism OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out-dir", default="vlm_loc")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--grid", type=int, default=3,
                    help="cells per side; 3=the 93%% test, 6=4x cheaper")
    ap.add_argument("--markers", action="store_true",
                    help="draw the free classical TRACKER's ball position "
                         "on each cell — the arm that could rescue the "
                         "ball at high packing (slower: it tracks first)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="an earlier ANSWER_KEY to draw FRESH of; repeatable")
    ap.add_argument("--crop", default="0.70,0.85,0.15,0.10",
                    help="w,h,x,y as fractions of the source frame — the "
                         "playing area; widen if players get clipped")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.video:
        raise SystemExit("--video is required (or use --selftest)")

    cells = a.grid * a.grid
    dur = (cells - 1) * STEP_S
    crop = tuple(float(x) for x in a.crop.split(","))
    contacts = load_train_contacts(a.labels, a.split)
    used = load_used(a.exclude)
    # over-draw, then reject windows that overlap spent video
    wins, seen = [], 0
    for cum, t0 in sample_windows(contacts, a.n * 40, a.seed, dur):
        seen += 1
        if overlaps(cum, t0, dur, used) or overlaps(cum, t0, dur, wins):
            continue
        wins.append((cum, t0, dur))
        if len(wins) == a.n:
            break
    if len(wins) < a.n:
        print(f"only {len(wins)} non-overlapping windows left in TRAIN "
              f"at this packing — scoring what there is")

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    key = out / "ANSWER_KEY_LOC.csv"
    with open(key, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "rally_cum", "t0_s", "span_s", "grid",
                    "markers", "n_contacts", "offsets_s", "hitters",
                    "paces", "marked_cells"])
        for i, (cum, t0, _d) in enumerate(wins, 1):
            name = f"w{i:02d}.png"
            cs = contacts_in(contacts, cum, t0, dur)
            print(f"  {name}  rally {cum} @ {t0:.2f}s")
            bits = cut_grid(a.video, t0, out / name, a.grid, crop,
                            a.markers)
            w.writerow([name, cum, f"{t0:.3f}", f"{dur:.2f}", a.grid,
                        int(a.markers), len(cs),
                        "|".join(f"{o:.2f}" for o, _, _ in cs),
                        "|".join(h for _, h, _ in cs),
                        "|".join(p for _, _, p in cs), bits])
    # NOTE: the realized contacts-per-window distribution is deliberately
    # NOT printed. The 2026-08-19 run printed it and had to disclose the
    # contamination — a scorer who knows the count distribution has a
    # prior on how many shots to call.
    print(f"\n{len(wins)} windows in {out}/  ({dur:.2f}s each, "
          f"{cells} cells at {STEP_S}s, grid {a.grid}x{a.grid}, "
          f"markers {'on' if a.markers else 'off'}, seed {a.seed})")
    print(f"covers {len(wins) * dur:.0f}s of TRAIN video"
          + (f", drawn clear of {len(used)} spent windows" if used else ""))
    print(f"Paste w*.png. Do NOT paste {key.name} until the calls are in.")


if __name__ == "__main__":
    main()
