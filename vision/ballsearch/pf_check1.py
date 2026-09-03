"""CHECK 1 of ball_gate.md, scored on the PATH-FIRST track.

Why this exists (2026-09-03): the graded battery (`vision/ball_grade.py`)
runs the decoder stack — candidates -> hitter_chain -> ball_decoder ->
replication.  The adopted incumbent is path-first (pathfirst_gate.md),
and it has never been scored on the gate's own checks.  So "how far is
the tracker from clearing the seal" currently has no answer.  This
closes half of that: CHECK 1, on train rallies, through the SAME frozen
scorer ball_grade uses (vision/gate_checks.check1).

MEASUREMENT ONLY, on TRAIN rallies.  Not a grade, not a verdict, no
seal touched.  Bars are printed for reference and are not loosened
anywhere.  Changing which stack the seal grades needs its own
pre-registration before r20 is spent; this is the train read that
informs whether that registration is worth writing.

The panel, the tolerances (V 25 px, S 40 px), the +-1 frame lookup and
the I/N exclusion all come from gate_checks, i.e. from ball_grade.
The only thing added is the CLAIM count: an unclaimed frame is charged
as a miss by the frozen rule, and path-first is a precision-first
instrument with real coverage gaps, so hit rate alone cannot separate
a coverage failure from a placement failure.

    python3 pf_check1.py [rallies...]        # default: the train set
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import gate_checks as gc                                    # noqa: E402
import pathfirst as pf                                      # noqa: E402
from make_ball_audit import load_impacts                    # noqa: E402

SP = Path(__file__).parent
DATA = Path("/home/user/pickleball/data/vision")
TRAIN = [2, 3, 4, 5, 6, 7, 17]
FPS = 60.0


def per_frame_of(track, t0):
    """path-first {frame: (x, y)} -> gate {frame: (t, x, y)}."""
    return {f: (t0 + f / FPS, x, y) for f, (x, y) in track.items()}


def main(rallies):
    cell = json.loads((SP / "pathfirst_tune.json").read_text())
    cell = {k: cell[k] for k in ("p_seed", "s_min", "gap")}
    print(f"CHECK 1 (ball_gate.md) on the PATH-FIRST track, frozen cell {cell}")
    print("bars: PASS >= 70% V, FAIL < 40% V   |   train measurement, no seal\n")
    print(f"{'rally':>5} {'V hit':>12} {'rate':>7} {'claimed':>8} "
          f"{'of-claimed':>11} {'S hit':>11} {'rate':>7} {'flights':>8}")
    rows = []
    for r in rallies:
        ctx = pf.context(r)
        res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
        t0 = ctx["t0"]
        pfm = per_frame_of(res["track"], t0)
        labels = list(csv.DictReader(open(DATA / f"ball_path_r{r}.csv")))
        imps, _dead = load_impacts(rally=r, prefill_ok=True)
        c1 = gc.check1(pfm, labels, imps, t0, FPS)
        v, s = c1["V"], c1["S"]
        # of-claimed: placement accuracy where the tracker actually spoke
        ofc = 100 * v["hit"] / max(v["claimed"], 1)
        print(f"{r:>5} {v['hit']:>6}/{v['tot']:<5} {v['rate']:>6.1f}% "
              f"{v['claimed']:>8} {ofc:>10.1f}% "
              f"{s['hit']:>5}/{s['tot']:<5} {s['rate']:>6.1f}% "
              f"{len(res['chosen']):>8}")
        rows.append(dict(rally=r, v_hit=v["hit"], v_tot=v["tot"],
                         v_rate=v["rate"], v_claimed=v["claimed"],
                         of_claimed=ofc, s_hit=s["hit"], s_tot=s["tot"],
                         s_rate=s["rate"], flights=len(res["chosen"]),
                         verdict=("PASS" if c1["pass"] else
                                  "FAIL" if c1["fail"] else "MIDDLE")))
    vh = sum(x["v_hit"] for x in rows)
    vt = sum(x["v_tot"] for x in rows)
    vc = sum(x["v_claimed"] for x in rows)
    print(f"\npooled V {vh}/{vt} = {100*vh/max(vt,1):.1f}%  "
          f"claimed {vc}/{vt} = {100*vc/max(vt,1):.1f}%  "
          f"of-claimed {100*vh/max(vc,1):.1f}%")
    print("per-rally CHECK 1 verdicts: "
          + "  ".join(f"r{x['rally']} {x['verdict']}" for x in rows))
    (SP / "pf_check1.json").write_text(json.dumps(rows, indent=1))
    print("wrote", SP / "pf_check1.json")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or TRAIN)
