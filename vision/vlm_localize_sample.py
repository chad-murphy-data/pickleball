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
N_CELLS, STEP_S = 9, 0.15
PRE_PAD = 4.0          # sample this far before the serve (dead time)


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


def cut_grid(video, t0, out_path, width, crop):
    """Nine seeks -> cropped, scaled, 1px-padded cells -> 3x3 grid."""
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error"]
    for i in range(N_CELLS):
        cmd += ["-ss", f"{max(t0 + i * STEP_S, 0.0):.3f}", "-i", str(video)]
    cw, ch, cx, cy = crop
    parts = [f"[{i}:v]crop=iw*{cw}:ih*{ch}:iw*{cx}:ih*{cy},"
             f"scale={width}:-2,pad=iw+4:ih+4:2:2:gray[c{i}]"
             for i in range(N_CELLS)]
    for r in range(3):
        parts.append("".join(f"[c{r*3+j}]" for j in range(3)) +
                     f"hstack=inputs=3[r{r}]")
    parts.append("[r0][r1][r2]vstack=inputs=3")
    cmd += ["-filter_complex", ";".join(parts), "-frames:v", "1",
            str(out_path)]
    subprocess.run(cmd, check=True)


def selftest():
    cs = {1: [(10.0, "A", "slow"), (11.0, "B", "slow"), (11.4, "A", "fast"),
              (11.8, "B", "fast")],
          2: [(50.0, "C", "slow"), (58.0, "D", "slow")]}
    dur = (N_CELLS - 1) * STEP_S
    assert abs(dur - 1.2) < 1e-9, dur
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
    ap.add_argument("--width", type=int, default=520,
                    help="per-cell width in the 3x3 grid")
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
    try:
        subprocess.run([ffmpeg_bin(), "-version"], capture_output=True,
                       check=True)
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("no usable ffmpeg (same resolver as pose_extract)")

    dur = (N_CELLS - 1) * STEP_S
    crop = tuple(float(x) for x in a.crop.split(","))
    contacts = load_train_contacts(a.labels, a.split)
    wins = sample_windows(contacts, a.n, a.seed, dur)
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    key = out / "ANSWER_KEY_LOC.csv"
    hist = {}
    with open(key, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window", "rally_cum", "t0_s", "n_contacts",
                    "offsets_s", "hitters", "paces"])
        for i, (cum, t0) in enumerate(wins, 1):
            name = f"w{i:02d}.png"
            cs = contacts_in(contacts, cum, t0, dur)
            hist[len(cs)] = hist.get(len(cs), 0) + 1
            print(f"  {name}  rally {cum} @ {t0:.2f}s  "
                  f"({len(cs)} contacts)")
            cut_grid(a.video, t0, out / name, a.width, crop)
            w.writerow([name, cum, f"{t0:.3f}", len(cs),
                        "|".join(f"{o:.2f}" for o, _, _ in cs),
                        "|".join(h for _, h, _ in cs),
                        "|".join(p for _, _, p in cs)])
    print(f"\n{len(wins)} windows in {out}/  ({dur:.1f}s each, "
          f"{N_CELLS} cells at {STEP_S}s, seed {a.seed})")
    print(f"contacts-per-window distribution: "
          f"{dict(sorted(hist.items()))}")
    print(f"Paste w*.png. Do NOT paste {key.name} until the calls are in.")


if __name__ == "__main__":
    main()
