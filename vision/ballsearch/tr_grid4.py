"""Mini-grid 4: anchor angle threshold for long-rally robustness.
Single round, hard=2.5 fixed; angle in {60 (frozen), 40, 30}; panel
now includes rally 1 (dev, free)."""
import sys, csv
from pathlib import Path

sys.path.insert(0, "/home/user/pickleball/vision")
import ball_decoder as bd
from make_ball_audit import detect_events, score_events, load_impacts

DATA = Path("/home/user/pickleball/data/vision")
RALLIES = [1, 6, 7, 8]

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
    print(f"rally {r}: human {100*hobs:.1f}@{100*hpct:.1f}", flush=True)

for ang in (60.0, 40.0, 30.0):
    bd.TIMING_ANGLE = ang
    row, npass = [], 0
    for r in RALLIES:
        c = CTX[r]
        _, ref = bd.timing_decode(c["byf"], None, c["oflags"], c["t0"], [])
        obs, _, pct, _ = score_events(detect_events(ref), c["imps"],
                                      c["span"])
        ho, hp = c["hum"]
        ok = obs >= ho - 1e-9 and pct >= hp - 1e-9
        npass += ok
        row.append(f"r{r} {100*obs:5.1f}@{100*pct:4.1f}{'*' if ok else ' '}")
    print(f"ang={ang:<4} | " + " | ".join(row) + f" | pass {npass}/4",
          flush=True)
