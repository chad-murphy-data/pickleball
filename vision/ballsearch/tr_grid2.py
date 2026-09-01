"""Mini-grid 2: hardening ladder + label-free adaptive selection.

All label-free machinery; labels only score the outcome. Human bars
use the ball_grade convention (full label span).
"""
import sys, csv, math
from pathlib import Path

sys.path.insert(0, "/home/user/pickleball/vision")
import ball_decoder as bd
from make_ball_audit import detect_events, score_events, load_impacts

DATA = Path("/home/user/pickleball/data/vision")
RALLIES = [6, 7, 8]

CTX = {}
for r in RALLIES:
    byf, t0 = bd.load_candidates(r)
    imps, dead = load_impacts(rally=r)
    f_min = round((imps[0] - 0.3 - t0) * bd.FPS)
    f_max = round((dead + 0.3 - t0) * bd.FPS)
    byf = {f: c for f, c in byf.items() if f_min <= f <= f_max}
    oflags = bd.out_of_court_flags(byf, bd.court_hull())
    labels = [row for row in csv.DictReader(open(DATA / f"ball_path_r{r}.csv"))
              if row["x"]]
    hum_pts = [(float(row["t_s"]), float(row["x"]), float(row["y"]))
               for row in labels if row["vis"] == "V"]
    span = (imps[0] - 1.0, dead)
    hobs, _, hpct, _ = score_events(detect_events(hum_pts), imps, span)
    CTX[r] = dict(byf=byf, t0=t0, oflags=oflags, imps=imps, span=span,
                  hum=(hobs, hpct))


def ladder_decode(c, schedule, ang):
    """Hardening ladder: each round decodes fwd+rev at schedule[i],
    accumulating turn anchors; returns final fwd refined + fwd/rev
    event agreement of the FINAL round (label-free diagnostic)."""
    bd.TIMING_ANGLE = ang
    byf, t0, oflags = c["byf"], c["t0"], c["oflags"]
    anc = []
    fwd = bd.decode(byf, None, oflags, None)
    rev = bd.decode_reversed(byf, None, oflags, None)
    for hard in schedule:
        for path in (fwd, rev):
            for a in bd.turn_anchor_list(bd.refine_arcs(path, t0)):
                if all(abs(a[0] - b[0]) > 0.08 or
                       math.hypot(a[1] - b[1], a[2] - b[2]) > 20
                       for b in anc):
                    anc.append(a)
        af = bd.anchor_flags(byf, t0, anc)
        fwd = bd.decode(byf, None, oflags, af, hard=hard)
        rev = bd.decode_reversed(byf, None, oflags, af, hard=hard)
    fr = bd.refine_arcs(fwd, t0)
    rr = bd.refine_arcs(rev, t0)
    ef, er = detect_events(fr), detect_events(rr)
    matched = sum(1 for e in ef if any(abs(e - e2) <= 0.10 for e2 in er))
    agree = matched / max(len(ef), len(er), 1)
    return fr, agree, len(ef)


def score(c, ref):
    obs, _, pct, _ = score_events(detect_events(ref), c["imps"], c["span"])
    return obs, pct


def show(name, results):
    row, npass = [], 0
    for r in RALLIES:
        obs, pct = results[r]
        ho, hp = CTX[r]["hum"]
        ok = obs >= ho - 1e-9 and pct >= hp - 1e-9
        npass += ok
        row.append(f"r{r} {100*obs:5.1f}@{100*pct:4.1f}{'*' if ok else ' '}")
    print(f"{name:<34} | " + " | ".join(row) + f" | pass {npass}/3",
          flush=True)


if __name__ == "__main__":
    # A. ladders
    for sched, ang in ((["1.5", "2.5"], 40), (["1.5", "2.5"], 60),
                       (["1.0", "2.5"], 60), (["1.5", "4.0"], 60),
                       (["1.5", "2.5", "4.0"], 60)):
        s = [float(x) for x in sched]
        res = {r: score(CTX[r], ladder_decode(CTX[r], s, ang)[0])
               for r in RALLIES}
        show(f"ladder {'-'.join(sched)} ang={ang}", res)

    # B. adaptive: single-hard runs, pick per rally by fwd/rev agreement
    cand_hards = (1.5, 2.5, 4.0)
    per = {r: {} for r in RALLIES}
    for h in cand_hards:
        for r in RALLIES:
            ref, agree, nev = ladder_decode(CTX[r], [h], 60)
            per[r][h] = (score(CTX[r], ref), agree, nev)
            print(f"  [diag] r{r} hard={h}: "
                  f"{100*per[r][h][0][0]:.1f}@{100*per[r][h][0][1]:.1f} "
                  f"agree={agree:.2f} nev={nev}", flush=True)
    res = {}
    for r in RALLIES:
        best = max(cand_hards, key=lambda h: per[r][h][1])
        res[r] = per[r][best][0]
        print(f"  adaptive pick r{r}: hard={best}")
    show("adaptive by fwd/rev agreement", res)
