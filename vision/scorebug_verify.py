"""Score-STATE verification of rally windows — flips identified by value.

The flip-train aligner (scorebug_windows -> coverage_windows) matches
detected scorebug flips to log rallies by GAP STRUCTURE alone, and on a
replay-heavy broadcast that leaves most windows approx-flagged: spacing
cannot prove which flip belongs to which rally (82/141 unstable on the
PPA Indoor Nationals mixed final).  This reads the scorebug's STATE —
the score digits as pixels — around every flip and checks identity by
VALUE: the state after rally k's flip must equal the state after every
other event that shows the same score, and must differ from every
other score.  No OCR: states are clustered by pixel distance, and the
cluster sequence is matched to the log's known score sequence.

Free extra: a flip whose after-state equals an EARLIER score's cluster
is a broadcast REPLAY of that earlier rally end — the coverage spec's
replay trap, detected mechanically by value.

    python vision/scorebug_verify.py --video vod.mp4 \
        --diff sb_diff.csv --timeline rally_timeline_<id8>.csv \
        --windows coverage_windows_<vod>.csv --out <v2.csv>
    python vision/scorebug_verify.py --selftest

Verdicts per rally: VERIFIED (flip state matches its score's cluster),
CORRECTED (previous flip rejected; a unique nearby flip carries the
right before/after states — window re-timed), UNKNOWN (score's cluster
never established, or no candidate flip).  The output windows CSV sets
approx=0 for VERIFIED/CORRECTED and keeps approx=1 otherwise; existing
confident windows that FAIL verification are demoted to approx=1 (the
state is stronger evidence than spacing stability).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorebug_windows import FLIP_LAG_S, detect_flips, ffmpeg_bin

CROP_W, CROP_H = 256, 96
PRE_S, POST_S = 1.2, 1.2
MATCH_TOL_S = 0.8          # window t1 -> flip association tolerance
SEARCH_S = 90.0            # correction search radius


def grab_crop(video, t, w=CROP_W, h=CROP_H):
    """Grayscale scorebug-corner crop at time t (SB geometry from the
    validated scan: top-left 42% x 17%)."""
    p = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{max(t, 0):.2f}",
         "-i", str(video), "-frames:v", "1",
         "-vf", f"crop=iw*0.42:ih*0.17:0:0,scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True)
    if len(p.stdout) < w * h:
        return None
    return np.frombuffer(p.stdout[:w * h], np.uint8).reshape(h, w)


def digit_mask(flip_pairs, quiet_pairs, top_frac=0.02):
    """Auto-locate the score digits: pixels that change ACROSS flips but
    not during quiet spans (kills clocks, animations, static names)."""
    fd = np.mean([np.abs(a.astype(np.int16) - b.astype(np.int16))
                  for a, b in flip_pairs], axis=0)
    qd = np.mean([np.abs(a.astype(np.int16) - b.astype(np.int16))
                  for a, b in quiet_pairs], axis=0) if quiet_pairs else 0
    sig = np.clip(fd - 1.5 * qd, 0, None)
    thr = np.percentile(sig, 100 * (1 - top_frac))
    mask = sig >= max(thr, 8.0)
    if mask.sum() < 40:
        mask = sig >= np.percentile(sig, 99)
    return mask


def state_vec(crop, mask):
    v = crop[mask].astype(np.float32)
    return v


def cluster_states(vecs, thr):
    """Greedy leader clustering by mean absolute pixel distance."""
    cents, out = [], []
    for v in vecs:
        best, bd = -1, 1e18
        for i, c in enumerate(cents):
            d = float(np.abs(v - c[0] / c[1]).mean())
            if d < bd:
                best, bd = i, d
        if best >= 0 and bd < thr:
            cents[best][0] += v
            cents[best][1] += 1
            out.append(best)
        else:
            cents.append([v.copy(), 1])
            out.append(len(cents) - 1)
    return out


def calibrate_thr(video, flips, mask, rng):
    """Same-state noise floor: crop pairs 0.4 s apart just BEFORE
    flips (where the bug is provably up — the same place the 'before'
    states come from).  MEDIAN, not a high percentile: on a dense flip
    train some samples land in bug animations and a p90 there inflated
    the floor 40x (measured: threshold 120 -> 2 clusters -> garbage)."""
    ds = []
    for _ in range(30):
        i = int(rng.integers(1, len(flips)))
        t = flips[i][0] - PRE_S - 0.4
        if i and t - flips[i - 1][0] < 2.0:
            continue
        a, b = grab_crop(video, t), grab_crop(video, t + 0.4)
        if a is None or b is None:
            continue
        ds.append(float(np.abs(state_vec(a, mask)
                               - state_vec(b, mask)).mean()))
    noise = float(np.median(ds)) if ds else 1.5
    return max(4.0 * noise, 6.0)


def run(a):
    flips = detect_flips(a.diff)
    print(f"{len(flips)} flips")
    wins = list(csv.DictReader(open(a.windows)))
    tl = list(csv.DictReader(open(a.timeline)))
    scores = {int(w["rally_cum"]): r["end_score"]
              for w, r in zip(wins, tl)}
    games = {int(w["rally_cum"]): w["game"] for w in wins}

    # crops around every flip (cached: crop grabs are the slow pass)
    cache = Path(str(a.out) + ".crops.npz")
    before, after = {}, {}
    if cache.exists():
        z = np.load(cache)
        before = {int(k[1:]): z[k] for k in z.files if k[0] == "b"}
        after = {int(k[1:]): z[k] for k in z.files if k[0] == "a"}
        print(f"crops from cache ({len(before)}/{len(after)})")
    if len(before) < len(flips) - 5:
        print("grabbing crops (2 per flip)...")
        for i, (t, z_) in enumerate(flips):
            if i not in before:
                b = grab_crop(a.video, t - PRE_S)
                if b is not None:
                    before[i] = b
            if i not in after:
                f = grab_crop(a.video, t + POST_S)
                if f is not None:
                    after[i] = f
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(flips)}", flush=True)
        np.savez_compressed(
            cache, **{f"b{i}": v for i, v in before.items()},
            **{f"a{i}": v for i, v in after.items()})
    rng = np.random.default_rng(11)
    fp = [(before[i], after[i]) for i in range(len(flips))
          if i in before and i in after]
    qp = []
    for _ in range(20):
        i = int(rng.integers(0, len(flips) - 1))
        t = flips[i][0] + 4.5
        if flips[i + 1][0] - t < 2.0:
            continue
        x, y = grab_crop(a.video, t), grab_crop(a.video, t + 0.6)
        if x is not None and y is not None:
            qp.append((x, y))
    mask = digit_mask(fp, qp)
    print(f"digit mask: {int(mask.sum())} px")
    thr = calibrate_thr(a.video, flips, mask, rng)
    print(f"cluster threshold {thr:.1f} grey levels")

    idx = sorted(set(before) | set(after))
    vecs, keys = [], []
    for i in idx:
        for which, store in (("b", before), ("a", after)):
            if i in store:
                vecs.append(state_vec(store[i], mask))
                keys.append((i, which))
    labels = cluster_states(vecs, thr)
    lab = dict(zip(keys, labels))
    n_cl = len(set(labels))
    print(f"{n_cl} distinct scorebug states across {len(vecs)} crops")
    if n_cl < 10:
        raise SystemExit(
            f"clustering collapsed to {n_cl} states — a real match has "
            f"dozens of scores; the threshold or mask is wrong for this "
            f"bug. NOT emitting verdicts on a broken instrument.")

    # score -> cluster map from currently-confident rallies
    F = [f[0] for f in flips]
    def flip_near(t1):
        j = int(np.argmin(np.abs(np.array(F) - (t1 + FLIP_LAG_S))))
        return j if abs(F[j] - (t1 + FLIP_LAG_S)) <= MATCH_TOL_S else None
    votes = defaultdict(Counter)
    for w in wins:
        cum = int(w["rally_cum"])
        if w["approx"] != "0":
            continue
        j = flip_near(float(w["t1s"]))
        if j is not None and (j, "a") in lab:
            votes[(games[cum], scores[cum])][lab[(j, "a")]] += 1
    score_cl = {k: c.most_common(1)[0][0] for k, c in votes.items()
                if c.most_common(1)[0][1] >= max(1, sum(c.values()) // 2)}
    # scores seen once are still usable; ambiguous (tied) ones dropped
    print(f"{len(score_cl)} score states pinned from confident rallies")

    n_ver = n_cor = n_dem = n_unk = 0
    out_rows = []
    replay_flips = set()
    # replay detection: a flip whose after-state matches a score pinned
    # EARLIER in video time than that score's real end
    pinned_time = {}
    for w in wins:
        cum = int(w["rally_cum"])
        key = (games[cum], scores[cum])
        j = flip_near(float(w["t1s"]))
        if w["approx"] == "0" and j is not None and key in score_cl:
            pinned_time[score_cl[key]] = min(
                pinned_time.get(score_cl[key], 1e18), F[j])
    for j, (t, z) in enumerate(flips):
        cl = lab.get((j, "a"))
        if cl is not None and cl in pinned_time and t > pinned_time[cl] + 5.0:
            replay_flips.add(j)

    for w in wins:
        w = dict(w)
        cum = int(w["rally_cum"])
        key = (games[cum], scores[cum])
        exp = score_cl.get(key)
        j = flip_near(float(w["t1s"]))
        got = lab.get((j, "a")) if j is not None else None
        prev_key = (games[cum], scores[cum - 1]) if cum - 1 in scores \
            and games.get(cum - 1) == games[cum] else None
        exp_prev = score_cl.get(prev_key) if prev_key else None
        verdict = "unknown"
        if exp is not None and got is not None:
            if got == exp:
                verdict = "verified"
            else:
                # search for the right flip by value
                cand = [k for k, (tf, _) in enumerate(flips)
                        if abs(tf - (float(w["t1s"]) + FLIP_LAG_S)) <= SEARCH_S
                        and lab.get((k, "a")) == exp
                        and k not in replay_flips
                        and (exp_prev is None
                             or lab.get((k, "b")) == exp_prev)]
                if len(cand) == 1:
                    dur = float(w["dur_s"])
                    t1 = flips[cand[0]][0] - FLIP_LAG_S
                    w["t1s"] = f"{t1:.1f}"
                    w["t0s"] = f"{max(t1 - dur, 0.0):.1f}"
                    verdict = "corrected"
                else:
                    verdict = "mismatch"
        if verdict == "verified":
            n_ver += 1
            w["approx"] = "0"
        elif verdict == "corrected":
            n_cor += 1
            w["approx"] = "0"
        elif verdict == "mismatch":
            n_dem += 1
            w["approx"] = "1"
        else:
            n_unk += 1          # keep the spacing verdict as-is
        w["state_check"] = verdict
        out_rows.append(w)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)
    n_conf = sum(1 for w in out_rows if w["approx"] == "0")
    print(f"verified {n_ver}, corrected {n_cor}, demoted {n_dem}, "
          f"unknown {n_unk}; confident windows now {n_conf}/{len(out_rows)}")
    print(f"replay-echo flips detected by value: {len(replay_flips)}")
    print(f"wrote {a.out}")


# ------------------------------------------------------------ selftest


def selftest():
    rng = np.random.default_rng(2)
    # 12 scores, video flips = true ends + replay echoes + junk
    H, W = 8, 30
    mask = np.zeros((H, W), bool)
    mask[:, :10] = True

    def crop_for(state):
        rng2 = np.random.default_rng(state)
        base = np.zeros((H, W), np.uint8)
        base[:, :10] = rng2.integers(0, 255, (H, 10))
        return base

    def noisy(c):
        return np.clip(c.astype(int) + rng.integers(-2, 3, c.shape),
                       0, 255).astype(np.uint8)

    vecs, keys = [], []
    true_flips = []
    t = 0.0
    # true sequence: states 1..12; replay of state 3 after state 6;
    # junk flip with unique state 99 after state 8
    seq = [1, 2, 3, 4, 5, 6, (3, "replay"), 7, 8, (99, "junk"), 9, 10, 11, 12]
    for s in seq:
        st = s[0] if isinstance(s, tuple) else s
        t += 30.0
        true_flips.append((t, st, s[1] if isinstance(s, tuple) else "real"))
    for j, (tf, st, kind) in enumerate(true_flips):
        prev = true_flips[j - 1][1] if j else 0
        vecs.append(state_vec(noisy(crop_for(prev)), mask))
        keys.append((j, "b"))
        vecs.append(state_vec(noisy(crop_for(st)), mask))
        keys.append((j, "a"))
    labels = cluster_states(vecs, thr=8.0)
    lab = dict(zip(keys, labels))
    # replay after-state must cluster WITH the original state 3
    j3 = next(j for j, (tf, st, k) in enumerate(true_flips)
              if st == 3 and k == "real")
    jr = next(j for j, (tf, st, k) in enumerate(true_flips) if k == "replay")
    assert lab[(j3, "a")] == lab[(jr, "a")], "replay state not matched"
    jj = next(j for j, (tf, st, k) in enumerate(true_flips) if k == "junk")
    assert lab[(jj, "a")] != lab[(j3, "a")], "junk state collided"
    n_cl = len(set(labels))
    assert n_cl >= 13, f"clusters collapsed: {n_cl}"
    print(f"  clustering: {n_cl} states, replay matched to original, "
          f"junk distinct OK")
    # digit-mask: flips change digits; quiet pairs change a fake clock
    clock = np.zeros((H, W), np.uint8)
    fp = [(crop_for(1), crop_for(2)), (crop_for(2), crop_for(3))]
    qa = crop_for(1).copy()
    qb = crop_for(1).copy()
    qa[:, 20:] = 100
    qb[:, 20:] = 200            # clock region flickers without flips
    m = digit_mask(fp, [(qa, qb)])
    assert m[:, :10].sum() > 0.5 * m.sum(), "mask missed the digits"
    assert not m[:, 20:].any(), "mask includes the clock"
    print("  digit-mask isolates score pixels from a clock OK")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--diff", type=Path)
    ap.add_argument("--timeline", type=Path)
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("video", "diff", "timeline", "windows", "out"):
        if not getattr(a, req):
            ap.error(f"--{req} required")
    run(a)


if __name__ == "__main__":
    main()
