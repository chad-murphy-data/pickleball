"""What a VLM scan actually costs, and how far it can be packed.

User question 2026-08-20: "I like the accuracy, don't like the cost."

The 93%-recall localization test used a 3x3 grid of 9 frames at 0.15 s.
Pricing that naively says the images dominate and the bill scales with
video length. It does — but not the way it looks, because of one fact
about how images are billed:

    EVERY GRID COSTS THE SAME NUMBER OF TOKENS.

Images are downscaled to ~1568 px on the long edge before tokenisation
(tokens ~ w*h/750), and the shipped 3x3 grid is 1572 px wide — already
at the cap. A 4x4 or 6x6 grid of the same crop downscales to the same
1568 px and therefore costs the same ~2239 tokens while covering 1.8x
or 4x more video. Packing is nearly free in tokens and is paid for in
PIXELS PER FRAME, which is a different currency: 3x3 gives each frame
0.58x of the source crop, 6x6 gives 0.29x.

So the cost question is really a resolution question, and it splits
into two channels that do NOT degrade at the same rate. This module
prices the grid (cost_table) and renders the ladder (render_ladder) so
the degradation can be looked at rather than assumed.

    python3 vlm_pack.py --cost
    python3 vlm_pack.py --video full_match.mp4.webm --rally 1
    python3 vlm_pack.py --selftest
"""
import argparse
import csv
from pathlib import Path

LABELS = "contact_labels_chicago0725.csv"
SPLIT = "label_split.csv"
CROP = (0.70, 0.85, 0.15, 0.10)     # same playing-area crop as the test
STEP_S = 0.15
LONG_EDGE = 1568                    # where images get downscaled to
IMG_TOK = LONG_EDGE * 1071 / 750    # the 3x3 grid's delivered aspect
PROMPT_TOK = 400                    # fixed instructions, uncached
CONTACTS_PER_S = 1.13               # measured: 311 contacts / 275 s span
# (input $/Mtok, output $/Mtok) at list; batch halves both
TIERS = {"top": (15.0, 75.0), "mid": (3.0, 15.0), "small": (1.0, 5.0)}


def cost_table(rally_s, tiers=TIERS, batch=True):
    """[(n, cells, span_s, images, Mtok_in, {tier: $})] for n x n grids.

    rally_s = seconds of RALLY time, not wall clock: the referee logs
    give rally spans, so dead time is never tiled and never billed."""
    out = []
    for n in (3, 4, 5, 6):
        cells = n * n
        span = cells * STEP_S               # advance per window when tiling
        images = rally_s / span
        t_in = images * (IMG_TOK + PROMPT_TOK) / 1e6
        t_out = (rally_s * CONTACTS_PER_S * 12 + images * 20) / 1e6
        div = 2.0 if batch else 1.0
        cost = {k: (t_in * i + t_out * o) / div for k, (i, o) in tiers.items()}
        out.append((n, cells, span, images, t_in, cost))
    return out


def cell_px(n, src_w=896, src_h=612):
    """Delivered pixels per frame in an n x n grid, and the scale factor
    against the source crop. This is the currency packing is paid in."""
    w = LONG_EDGE // n
    return w, int(round(w * src_h / src_w)), w / src_w


def train_rallies(split_path=SPLIT):
    p = Path(split_path)
    if not p.exists():
        raise SystemExit(f"{split_path} not found — the split is mandatory")
    return {int(r["rally_cum"]) for r in csv.DictReader(open(p))
            if r["split"] == "train"}


def render_ladder(video, rally, labels_path, split_path, out_dir, offset=0.0):
    """Write pack_{n}x{n}.png for n in 3..6, each downscaled to exactly
    what the API delivers, all starting at the same instant. TRAIN
    rallies only — this is a look-at-it probe, and holdout burns on use.

    Images are broadcast-derived: they stay local, never committed."""
    import cv2
    import numpy as np
    if rally not in train_rallies(split_path):
        raise SystemExit(f"rally {rally} is not TRAIN — refusing")
    ts = sorted(float(r["t_refined_s"] or r["t_tap_s"]) - offset
                for r in csv.DictReader(open(labels_path))
                if int(r["rally_cum"]) == rally
                and r.get("contact", "1") == "1")
    if not ts:
        raise SystemExit(f"no contacts for rally {rally}")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cw_f, ch_f, cx_f, cy_f = CROP
    x0, y0 = int(cx_f * W), int(cy_f * H)
    cw, ch = int(cw_f * W), int(ch_f * H)
    t0 = ts[0] - 0.3
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for n in (3, 4, 5, 6):
        cells = []
        for i in range(n * n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t0 + i * STEP_S) * fps)))
            ok, fr = cap.read()
            if not ok:
                break
            cells.append(fr[y0:y0 + ch, x0:x0 + cw])
        if len(cells) < n * n:
            print(f"{n}x{n}: ran off the end of the video")
            continue
        w, h, scale = cell_px(n, cw, ch)
        small = [cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
                 for c in cells]
        grid = np.vstack([np.hstack(small[r * n:(r + 1) * n])
                          for r in range(n)])
        out = Path(out_dir) / f"pack_{n}x{n}.png"
        cv2.imwrite(str(out), grid)
        span = (n * n - 1) * STEP_S
        got = sum(1 for t in ts if t0 <= t < t0 + span + STEP_S)
        print(f"{n}x{n}: cell {w}x{h} ({scale:.2f}x source), "
              f"grid {grid.shape[1]}x{grid.shape[0]}, span {span:.2f}s, "
              f"{got} contacts -> {out}")
    cap.release()


def print_cost(rally_s):
    rows = cost_table(rally_s)
    print(f"one grid = {IMG_TOK:.0f} image tokens no matter how many cells "
          f"it holds\n(all of them downscale to {LONG_EDGE} px on the long "
          f"edge)\n")
    hdr = f"{'grid':>5} {'cells':>6} {'span':>7} {'cell px':>9} {'imgs':>7} "
    print(hdr + "  ".join(f"{k + ' tier':>11}" for k in TIERS))
    for n, cells, span, imgs, _t, cost in rows:
        w, h, sc = cell_px(n)
        print(f"{n}x{n:<3} {cells:>6} {span:>6.2f}s {w:>4}x{h:<4} {imgs:>7.0f} "
              + "  ".join(f"${cost[k]:>10.2f}" for k in TIERS))
    print(f"\nper {rally_s / 60:.0f} min of RALLY time, batch pricing. "
          f"Dead time is never\ntiled: rally spans come from the referee "
          f"logs, which are free.")


def selftest():
    # the invariant the whole argument rests on: cost per unit of video
    # falls as the square of the grid side, because tokens per image
    # are FLAT and span grows with cells
    rows = {n: r for n, *r in
            [(n, c, s, i, t, k) for n, c, s, i, t, k in cost_table(3600)]}
    for n in (4, 5, 6):
        ratio = rows[3][2] / rows[n][2]        # images(3x3) / images(nxn)
        assert abs(ratio - (n / 3) ** 2) < 1e-9, (n, ratio)
    # and the currency it is paid in: pixels per frame fall linearly
    for n in (3, 4, 5, 6):
        w, h, sc = cell_px(n)
        assert abs(w * n - LONG_EDGE) <= n, (n, w)
        assert 0.28 < sc < 0.60, (n, sc)
    assert cell_px(3)[0] > cell_px(6)[0] * 1.9
    # monotone in both directions, and batch really is half
    c = {n: k for n, _c, _s, _i, _t, k in cost_table(2820)}
    for t in TIERS:
        assert c[3][t] > c[4][t] > c[5][t] > c[6][t], t
    assert c[3]["top"] > c[3]["mid"] > c[3]["small"]
    full = {n: k for n, _c, _s, _i, _t, k in cost_table(2820, batch=False)}
    assert abs(full[3]["top"] - 2 * c[3]["top"]) < 1e-9
    # a rally-time figure, not a wall-clock one
    assert abs(cost_table(2820)[0][3] - 2820 / 1.35) < 1.0
    print("selftest: packing scales as n^2, pixels as 1/n, pricing "
          "monotone, batch = half OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--rally", type=int, default=1)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out-dir", default="vlm_pack")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds to subtract if --video is a clip, not "
                         "the full match")
    ap.add_argument("--rally-min", type=float, default=47.0,
                    help="minutes of rally time in a match")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    print_cost(a.rally_min * 60)
    if a.video:
        print()
        render_ladder(a.video, a.rally, a.labels, a.split, a.out_dir,
                      a.offset)
    elif not a.cost:
        print("\n(pass --video to also render the packing ladder)")


if __name__ == "__main__":
    main()
