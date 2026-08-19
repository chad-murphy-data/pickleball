"""Blind VLM hitter-call test — frame sampler (EXPLORATION, not a verdict).

Question (user, 2026-08-19): could an AI API / vision-language model do
what the pose pipeline can't? The measured walls say no for DETECTION —
the ball is findable in only 64% of in-play frames, and the binder is
contact PLACEMENT, which is temporal localization over 145k frames. But
"which player just hit it" is a RECOGNITION question, and recognition is
what these models are for. Never tested; nothing in the record had tried.

The point of THIS script is to make that test honest. It cuts a blind
set: N labeled contacts sampled at random from TRAIN rallies, balanced
fast/slow, each rendered as a 3-frame strip (t-0.1, t, t+0.1 — a single
frame at contact is often motion-blurred). Strips are numbered in
SHUFFLED order and the answer key goes to a separate file that must not
be shared until scoring.

WHAT IS EXCLUDED, and why it matters:
  - serves and returns. Their hitter is already known from the referee
    logs via the lineup state machine (99.25%, no camera needed), and a
    serve is trivially inferable from score parity anyway (odd score =>
    served from the left, which pins the server and the receiver by
    diagonal). Including them would inflate the result with cases no
    model is needed for. This was demonstrated live on 2026-08-19: a
    pre-serve screenshot was called correctly off court geometry alone.
  - whiffs (contact = 0). Not hits.
  - nothing else. Third shots stay IN — the third-shot drive is the
    gap-invisible case the tempo classifier misses, so it is exactly
    the population of interest.

Sampling is seeded and printed, so the draw is reproducible and cannot
be quietly re-rolled until it flatters anyone.

Usage (Mac, flat folder):
    python3 vlm_frame_sample.py --video full_match.mp4.webm
      -> vlm_test/q01.png .. qNN.png   paste these into the thread
      -> vlm_test/ANSWER_KEY.csv       keep; do NOT paste until scoring

Scoring: the caller answers each strip with a court position
(near-left / near-right / far-left / far-right, camera view) plus a
fast/slow guess. You convert position -> name by eye (you have the
video open) and score against the key. Chance is 25% on the 4-way
call, 50% if the team is treated as given.
"""
import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

from fastslow_check import classify_type

LABELS = "contact_labels_chicago0725.csv"
SPLIT = "label_split.csv"
OFFSETS = (-0.10, 0.0, 0.10)


def train_rallies(path):
    """rally_cum set marked train. Missing file => refuse (the whole
    point of the split is that it is never optional)."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{path} not found — the train/holdout split is "
                         f"mandatory; holdout frames must never be cut.")
    return {int(r["rally_cum"]) for r in csv.DictReader(open(p))
            if r["split"] == "train"}


def candidates(labels_path, train):
    """Paced, non-opening, real contacts from train rallies."""
    out = []
    for r in csv.DictReader(open(labels_path)):
        if r.get("contact", "1") == "0":
            continue
        cum = int(r["rally_cum"])
        if cum not in train:
            continue
        cls = classify_type(r["shot_type"])
        if cls not in ("fast", "slow"):
            continue            # drops serves/returns/lunges/untyped
        out.append({"cum": cum, "shot": int(r["shot_index"]),
                    "t": float(r["t_refined_s"] or r["t_tap_s"]),
                    "hitter": r["hitter_name"], "type": r["shot_type"],
                    "pace": cls})
    return out


def draw(cands, n, seed):
    """Balanced fast/slow draw, then shuffled into question order."""
    rng = random.Random(seed)
    by = {"fast": [c for c in cands if c["pace"] == "fast"],
          "slow": [c for c in cands if c["pace"] == "slow"]}
    picked = []
    for pace in ("fast", "slow"):
        want = n // 2
        pool = sorted(by[pace], key=lambda c: (c["cum"], c["shot"]))
        if len(pool) < want:
            print(f"note: only {len(pool)} {pace} contacts available "
                  f"(wanted {want}) — taking all of them")
            want = len(pool)
        picked += rng.sample(pool, want)
    rng.shuffle(picked)
    return picked


def cut_strip(video, t, out_path, width):
    """Three seeks, scaled and vstacked into one image."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for off in OFFSETS:
        cmd += ["-ss", f"{max(t + off, 0.0):.3f}", "-i", str(video)]
    chain = ";".join(f"[{i}:v]scale={width}:-2[s{i}]"
                     for i in range(len(OFFSETS)))
    stack = "".join(f"[s{i}]" for i in range(len(OFFSETS)))
    cmd += ["-filter_complex",
            f"{chain};{stack}vstack=inputs={len(OFFSETS)}",
            "-frames:v", "1", str(out_path)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out-dir", default="vlm_test")
    ap.add_argument("--n", type=int, default=20,
                    help="questions to cut (balanced fast/slow)")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--width", type=int, default=1280,
                    help="per-frame width in the stacked strip")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH")
    if not Path(a.video).exists():
        raise SystemExit(f"{a.video} not found")

    train = train_rallies(a.split)
    cands = candidates(a.labels, train)
    if not cands:
        raise SystemExit("no eligible contacts — is the labels file right?")
    picked = draw(cands, a.n, a.seed)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    key = out / "ANSWER_KEY.csv"
    with open(key, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "rally_cum", "shot_index", "t_s",
                    "hitter_name", "shot_type", "pace"])
        for i, c in enumerate(picked, 1):
            name = f"q{i:02d}.png"
            print(f"  {name}  (cutting t={c['t']:.2f}s)")
            cut_strip(a.video, c["t"], out / name, a.width)
            w.writerow([name, c["cum"], c["shot"], f"{c['t']:.3f}",
                        c["hitter"], c["type"], c["pace"]])

    n_f = sum(1 for c in picked if c["pace"] == "fast")
    print(f"\n{len(picked)} strips in {out}/ "
          f"({n_f} fast / {len(picked) - n_f} slow), seed {a.seed}, "
          f"drawn from {len(cands)} eligible train contacts across "
          f"{len({c['cum'] for c in picked})} rallies.")
    print(f"Paste q*.png into the thread. Do NOT paste {key.name} "
          f"until the calls are in — it is the whole test.")


if __name__ == "__main__":
    main()
