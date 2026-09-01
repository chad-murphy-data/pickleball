"""Two-regime timing-stream grid search on train rallies 6/7/8.

Scores ONLY check 2 (the timing stream) against the human-matched bar.
Position stream is untouched by these knobs.
"""
import sys, csv, math
from pathlib import Path

sys.path.insert(0, "/home/user/pickleball/vision")
import ball_decoder as bd
from make_ball_audit import detect_events, score_events, load_impacts

DATA = Path("/home/user/pickleball/data/vision")
RALLIES = [6, 7, 8]

# ---- per-rally fixed context (cache once)
CTX = {}
for r in RALLIES:
    byf, t0 = bd.load_candidates(r)
    imps, dead = load_impacts(rally=r)
    serve, end = imps[0], dead
    f_min = round((serve - 0.3 - t0) * bd.FPS)
    f_max = round((end + 0.3 - t0) * bd.FPS)
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
    print(f"rally {r}: human {100*hobs:.1f}@{100*hpct:.1f}")

def run(cfg):
    for k, v in cfg.items():
        setattr(bd, k, v)
    row = []
    npass = 0
    for r in RALLIES:
        c = CTX[r]
        _, ref = bd.timing_decode(c["byf"], None, c["oflags"], c["t0"], [])
        obs, _, pct, _ = score_events(detect_events(ref), c["imps"], c["span"])
        ho, hp = c["hum"]
        ok = obs >= ho - 1e-9 and pct >= hp - 1e-9
        npass += ok
        row.append(f"r{r} {100*obs:5.1f}@{100*pct:4.1f}{'*' if ok else ' '}")
    print(f"hard={cfg['TIMING_HARD']:<4} ang={cfg['TIMING_ANGLE']:<4} "
          f"rounds={cfg['TIMING_ROUNDS']} | " + " | ".join(row)
          + f" | pass {npass}/3", flush=True)
    return npass

if __name__ == "__main__":
    grid = []
    for hard in (1.5, 2.5, 4.0):
        for ang in (40.0, 60.0, 80.0):
            for rounds in (1, 2):
                grid.append(dict(TIMING_HARD=hard, TIMING_ANGLE=ang,
                                 TIMING_ROUNDS=rounds))
    for cfg in grid:
        run(cfg)
