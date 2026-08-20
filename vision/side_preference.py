#!/usr/bin/env python3
"""Do players hold a fixed side? Tile one mid-rally frame per rally.

The claim (user, 2026-08-20): at the top level a player occupies one
half of their team's court for the whole match. If true it is a much
stronger identity constraint than appearance, because it survives
matching kit -- and it is what rescues the ANCHOR-FREE rallies, where
the broadcast never showed the serve and appearance is currently the
only channel.

This does not try to name anyone. It renders one row per rally, near
band and far band side by side, so a human who knows the players can
scan a column for a break. A swap is loud; uniformity is the finding.

READ THE MIRROR BEFORE CONCLUDING ANYTHING: near-side players have
their backs to the camera, so their left/right MATCHES the viewer's.
Far-side players face the camera, so theirs is REVERSED. A player who
holds "her right" appears on opposite sides of the image at the two
ends, and the ends swap at the change-over -- which is why this reports
per game, and why a raw pixel-x is not the quantity of interest.

Local-only: writes broadcast-derived imagery, never commit the output.
"""
import argparse, csv, sys
from collections import defaultdict


def rally_times(labels, split, want_split="train"):
    keep = {int(r["rally_cum"]) for r in csv.DictReader(open(split))
            if r["split"].strip() == want_split} if split else None
    by = defaultdict(list)
    for r in csv.DictReader(open(labels)):
        if r["contact"] != "1":
            continue
        c = int(r["rally_cum"])
        if keep is not None and c not in keep:
            continue
        t = r["t_refined_s"].strip() or r["t_tap_s"].strip()
        if t:
            by[c].append((float(t), r["hitter_name"], r["game"]))
    return {c: sorted(v) for c, v in by.items()}


def teams_from_alternation(rt):
    """Doubles sides alternate strictly, so consecutive shots are on
    OPPOSITE sides and shots two apart are on the SAME side. Pool every
    rally into one graph and 2-colour it: free team assignment with no
    video, no model, and no timestamps.

    POOL, DO NOT WORK RALLY BY RALLY. A single rally often cannot name
    all four players -- 5 of the 19 Chicago train rallies never see one
    of them touch the ball (three are 2-shot rallies; rally 7 runs nine
    shots and Jones never hits). That is missing data, not an
    alternation failure, and treating it as one was an early bug here.
    Measured on the same 19 rallies / 229 contacts: 0 consecutive-
    same-hitter violations, 0 colouring contradictions."""
    diff, same, viol = set(), set(), 0
    for _c, shots in sorted(rt.items()):
        n = [x[1] for x in shots]
        for i in range(len(n) - 1):
            if n[i] == n[i + 1]:
                viol += 1
            else:
                diff.add(frozenset((n[i], n[i + 1])))
        for i in range(len(n) - 2):
            if n[i] != n[i + 2]:
                same.add(frozenset((n[i], n[i + 2])))
    players = sorted({p for e in diff | same for p in e})
    if not players:
        return None, None, viol, 0
    col = {players[0]: 0}
    for _ in range(len(players) + 2):
        for e in diff:
            a, b = tuple(e)
            if a in col and b not in col: col[b] = 1 - col[a]
            elif b in col and a not in col: col[a] = 1 - col[b]
        for e in same:
            a, b = tuple(e)
            if a in col and b not in col: col[b] = col[a]
            elif b in col and a not in col: col[a] = col[b]
    bad = sum(1 for e in diff if col[tuple(e)[0]] == col[tuple(e)[1]])
    bad += sum(1 for e in same if col[tuple(e)[0]] != col[tuple(e)[1]])
    t0 = sorted(p for p in players if col[p] == 0)
    t1 = sorted(p for p in players if col[p] == 1)
    return t0, t1, viol, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--labels", default="data/vision/contact_labels_chicago0725.csv")
    ap.add_argument("--split", default="data/vision/label_split.csv")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="video_t = match_t - offset")
    ap.add_argument("--out-prefix", default="side_pref")
    ap.add_argument("--per-page", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.5,
                    help="where in the rally to sample (0.5 = midpoint)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    rt = rally_times(a.labels, a.split)
    if a.selftest:
        t0, t1, viol, bad = teams_from_alternation(rt)
        n = sum(len(v) for v in rt.values())
        print(f"  {len(rt)} rallies / {n} contacts")
        print(f"  consecutive-same-hitter violations: {viol}")
        print(f"  2-colouring contradictions:         {bad}")
        print(f"  team 0: {t0}")
        print(f"  team 1: {t1}")
        assert viol == 0, "alternation violated"
        assert bad == 0, "teams not 2-colourable"
        assert len(t0) == 2 and len(t1) == 2, "expected 2v2"
        singles = sum(1 for v in rt.values()
                      if len({x[1] for x in v}) < 4)
        print(f"  rallies that alone cannot name all four: {singles}"
              f"/{len(rt)}  <- why pooling is required")
        print("selftest: ALL OK")
        return

    if not a.video:
        raise SystemExit("--video required (or --selftest)")
    import cv2, numpy as np
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(3)); H = int(cap.get(4))
    BANDS = (("FAR", 0.10, 0.30), ("NEAR", 0.42, 0.80))
    rows, page, n = [], 1, 0
    for c, shots in sorted(rt.items()):
        t = shots[0][0] + a.frac * (shots[-1][0] - shots[0][0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t - a.offset) * fps)))
        ok, fr = cap.read()
        if not ok:
            continue
        strip = []
        for name, y0, y1 in BANDS:
            band = fr[int(y0*H):int(y1*H), int(0.15*W):int(0.85*W)]
            band = cv2.resize(band, (740, 150), interpolation=cv2.INTER_AREA)
            cv2.putText(band, f"r{c} g{shots[0][2]} {name}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            strip.append(band)
        rows.append(np.hstack(strip)); n += 1
        if len(rows) == a.per_page:
            cv2.imwrite(f"{a.out_prefix}_{page:02d}.png", np.vstack(rows))
            print(f"  wrote {a.out_prefix}_{page:02d}.png ({len(rows)} rallies)")
            rows, page = [], page + 1
    if rows:
        cv2.imwrite(f"{a.out_prefix}_{page:02d}.png", np.vstack(rows))
        print(f"  wrote {a.out_prefix}_{page:02d}.png ({len(rows)} rallies)")
    cap.release()
    print(f"\n{n} rallies sampled at {a.frac:.0%} through each rally.\n"
          "Scan each column for a break in left-right order. NEAR is\n"
          "un-mirrored (backs to camera), FAR is mirrored (facing it).")


if __name__ == "__main__":
    main()
