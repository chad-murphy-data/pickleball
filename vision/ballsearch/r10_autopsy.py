"""Pre-registered autopsy arms on spent rally 10 (ball_gate.md)."""
import sys, csv, gzip, math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/user/pickleball/vision")
import ball_decoder as bd
from make_ball_audit import detect_events, score_events, load_impacts

DATA = Path("/home/user/pickleball/data/vision")
R = 10
SERVE, END = 294.30, 318.45

labels = [r for r in csv.DictReader(open(DATA / f"ball_path_r{R}.csv"))
          if r["x"]]
imps, dead = load_impacts(rally=R)
span = (imps[0] - 1.0, dead)

# ---- ARM 3 first (frame checks): candidate recall at +/-1 frame
byf, t0 = bd.load_candidates(R)
hit = tot = 0
for r in labels:
    if r["vis"] != "V":
        continue
    tot += 1
    f0 = round((float(r["t_s"]) - t0) * bd.FPS)
    cx, cy = float(r["x"]), float(r["y"])
    if any(math.hypot(x - cx, y - cy) <= 25.0
           for df in (-1, 0, 1) for x, y in byf.get(f0 + df, ())):
        hit += 1
print(f"ARM 3 frame/clock: candidate V recall {100*hit/tot:.1f}% "
      f"({hit}/{tot}) at +/-1 frame — alignment "
      f"{'SANE' if hit/tot > 0.5 else 'SUSPECT'}", flush=True)

# ---- ARM 4 companion: recall inside vs outside the lob windows
LOBS = [(300.85, 301.25), (307.85, 308.35)]   # owner-flagged region(s)
for lo, hi in LOBS:
    h = t2 = 0
    for r in labels:
        if r["vis"] != "V" or not (lo <= float(r["t_s"]) <= hi):
            continue
        t2 += 1
        f0 = round((float(r["t_s"]) - t0) * bd.FPS)
        cx, cy = float(r["x"]), float(r["y"])
        h += any(math.hypot(x - cx, y - cy) <= 25.0
                 for df in (-1, 0, 1) for x, y in byf.get(f0 + df, ()))
    print(f"  lob window {lo}-{hi}: candidate recall "
          f"{h}/{t2}", flush=True)

# ---- ARM 1: oracle substitution — user's clicks through the SAME
# decoder (position stream config, oracle candidates, no junk)
obyf = defaultdict(list)
for r in labels:
    if r["vis"] in ("V", "S"):
        f = round((float(r["t_s"]) - t0) * bd.FPS)
        obyf[f].append((float(r["x"]), float(r["y"])))
f_min = round((SERVE - 0.3 - t0) * bd.FPS)
f_max = round((END + 0.3 - t0) * bd.FPS)
obyf = {f: c for f, c in obyf.items() if f_min <= f <= f_max}
oflags = bd.out_of_court_flags(obyf, bd.court_hull())
vis_o = bd.decode(obyf, None, oflags, None)
ref_o = bd.refine_arcs(vis_o, t0)
pf = {}
for t, x, y in ref_o:
    pf[round((t - t0) * bd.FPS)] = (x, y)
h = tot = 0
p_lo, p_hi = imps[0], imps[-1] + 0.5
for r in labels:
    if r["vis"] != "V" or not (p_lo <= float(r["t_s"]) <= p_hi):
        continue
    tot += 1
    f0 = round((float(r["t_s"]) - t0) * bd.FPS)
    best = min((math.hypot(pf[g][0] - float(r["x"]),
                           pf[g][1] - float(r["y"]))
                for g in (f0 - 1, f0, f0 + 1) if g in pf), default=1e9)
    h += int(best <= 25.0)
obs, _, pct, _ = score_events(detect_events(ref_o), imps, span)
print(f"ARM 1 oracle substitution: V {100*h/tot:.1f}% ({h}/{tot}); "
      f"turns {100*obs:.1f}% at pct {100*pct:.0f} — decoder on clean "
      f"input {'SOUND' if h/tot > 0.85 else 'IMPLICATED'}", flush=True)

# ---- bounce geography: human events not near contact taps = bounces;
# how many fall in each rally third + near the lob windows
evs_h = detect_events([(float(r["t_s"]), float(r["x"]), float(r["y"]))
                       for r in labels if r["vis"] == "V"])
bounces_h = [e for e in evs_h
             if all(abs(e - i) > 0.25 for i in imps)]
print(f"human path events {len(evs_h)}, non-contact (bounce-ish) "
      f"{len(bounces_h)}: "
      + ", ".join(f"{e:.2f}" for e in bounces_h), flush=True)
print("ARM 2 person ablation: N/A — no manual track assigns exist "
      "for r10 (automated channel was the only person channel).",
      flush=True)
