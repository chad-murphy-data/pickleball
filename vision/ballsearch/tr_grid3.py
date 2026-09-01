"""Mini-grid 3: intersection anchors (fwd AND rev must see the turn)."""
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


def intersect_anchors(fwd_ref, rev_ref):
    fa = bd.turn_anchor_list(fwd_ref)
    ra = bd.turn_anchor_list(rev_ref)
    return [a for a in fa
            if any(abs(a[0] - b[0]) <= 0.10 for b in ra)]


def run(c, schedule, mode):
    byf, t0, oflags = c["byf"], c["t0"], c["oflags"]
    anc = []
    fwd = bd.decode(byf, None, oflags, None)
    rev = bd.decode_reversed(byf, None, oflags, None)
    for hard in schedule:
        fr, rr = bd.refine_arcs(fwd, t0), bd.refine_arcs(rev, t0)
        new = (intersect_anchors(fr, rr) if mode == "int"
               else bd.turn_anchor_list(fr) + bd.turn_anchor_list(rr))
        for a in new:
            if all(abs(a[0] - b[0]) > 0.08 or
                   math.hypot(a[1] - b[1], a[2] - b[2]) > 20 for b in anc):
                anc.append(a)
        af = bd.anchor_flags(byf, t0, anc)
        fwd = bd.decode(byf, None, oflags, af, hard=hard)
        rev = bd.decode_reversed(byf, None, oflags, af, hard=hard)
    return bd.refine_arcs(fwd, t0)


def show(name, res):
    row, npass = [], 0
    for r in RALLIES:
        obs, pct = res[r]
        ho, hp = CTX[r]["hum"]
        ok = obs >= ho - 1e-9 and pct >= hp - 1e-9
        npass += ok
        row.append(f"r{r} {100*obs:5.1f}@{100*pct:4.1f}{'*' if ok else ' '}")
    print(f"{name:<30} | " + " | ".join(row) + f" | pass {npass}/3",
          flush=True)


if __name__ == "__main__":
    bd.TIMING_ANGLE = 60.0
    for name, sched, mode in (
            ("single 2.5 int", [2.5], "int"),
            ("ladder 1.5-2.5 int", [1.5, 2.5], "int"),
            ("ladder 1.5-4.0 int", [1.5, 4.0], "int")):
        res = {}
        for r in RALLIES:
            ref = run(CTX[r], sched, mode)
            obs, _, pct, _ = score_events(detect_events(ref),
                                          CTX[r]["imps"], CTX[r]["span"])
            res[r] = (obs, pct)
        show(name, res)
