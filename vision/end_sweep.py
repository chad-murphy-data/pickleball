#!/usr/bin/env python3
"""Sweep end-of-point rules against a recorded dump. No video, no pose.

WHY THIS EXISTS (user, 2026-08-22): "could we record all events
(including crossings) and sweep for something that would work instead
of rerunning?" Five window rules have been tried against this footage -
last-confident-candidate, candidate-gap, contact-gap-p99,
self-calibrated crossing gap, resume - and every one of them cost a
full run of the pose stack to discover it was wrong. The one that
finally got tuned properly (the resume rule) was only tunable because
the CONTACT LABELS were already sitting on disk in a CSV, so a Monte
Carlo could be run against them in two minutes.

Crossings and motion were not on disk. Now they are:
    python vision/touch_attribute.py ... --dump-events dump.json
writes every per-rally observation a rule could read, and this script
sweeps rules against it in seconds.

THE HONEST NUMBER IS THE LEAVE-ONE-OUT ONE. This dump holds ~15
labelled rallies. Sweeping seven rule families over a few hundred
parameter settings on 15 rallies WILL produce a rule with a beautiful
minimum, and that minimum is mostly selection. So every rule is scored
twice: in-sample, and leave-one-out (pick the parameters on 14
rallies, score the 15th, rotate). Where those two diverge, the
in-sample number is a mirage and the LOO number is the forecast. This
is the same discipline as select-then-verify in the clutch work.

THE LOSS IS ASYMMETRIC ON PURPOSE. Closing the window early destroys
contacts no later stage can recover; closing late only admits junk the
DP already has to filter. Early seconds are therefore charged 10x. If
you disagree with 10, change EARLY_COST - but change it deliberately
and say so, because "min" and "median" rank rules differently and the
whole point of writing the preference down is that it stops being
re-litigated per run.
"""
import argparse
import itertools
import json
import statistics

EARLY_COST = 10.0          # seconds of play destroyed, per second
LATE_COST = 1.0            # seconds of dead time admitted, per second


def loss(d):
    """Cost of ending a rally at true_last + d seconds."""
    return -d * EARLY_COST if d < 0 else d * LATE_COST


# ---------------------------------------------------------------- rules
# Each takes (rally, params, meta) and returns an end time or None.
# None means ABSTAIN, which is not free: a rule that abstains on half
# the rallies has no window on those and is reported as such rather
# than being quietly scored on the half it likes.

def r_last_cross(r, p, meta):
    """The last crossing, full stop. The benchmark to beat: +3.97s
    median, -0.79 min on the 2026-08-22d run."""
    cts = r.get("crossings") or []
    return cts[-1] if cts else None


def r_gap(r, p, meta):
    """Last crossing before a fixed gap."""
    cts = r.get("crossings") or []
    if not cts:
        return None
    stop = cts[0]
    for i in range(1, len(cts)):
        if cts[i] - cts[i - 1] > p["gap"]:
            break
        stop = cts[i]
    return stop


def r_selfcal(r, p, meta):
    """Gap threshold scaled by THIS rally's own crossing spacing - a
    rally whose crossings are dense should end at a small break, one
    whose crossings are sparse needs a big one."""
    cts = r.get("crossings") or []
    if len(cts) < 2:
        return cts[-1] if cts else None
    g = sorted(cts[i] - cts[i - 1] for i in range(1, len(cts)))
    cut = max(p["floor"], p["mult"] * g[len(g) // 2])
    stop = cts[0]
    for i in range(1, len(cts)):
        if cts[i] - cts[i - 1] > cut:
            break
        stop = cts[i]
    return stop


def r_resume(r, p, meta):
    """A gap only ends the rally if play never resumes (user,
    2026-08-21). Tuned to gap=4.0/look=8.0/need=1 by Monte Carlo on the
    contact labels; swept here against the real crossings."""
    cts = r.get("crossings") or []
    if not cts:
        return None
    last, i = cts[0], 1
    while i < len(cts):
        if cts[i] - cts[i - 1] <= p["gap"]:
            last = cts[i]
            i += 1
            continue
        resumed = [t for t in cts[i:] if t <= cts[i - 1] + p["look"]]
        if len(resumed) >= p["need"]:
            last = cts[i]
            i += 1
            continue
        return last
    return last


def r_cross_pad(r, p, meta):
    """Last crossing plus a fixed pad. The 2026-08-22e run said the
    resume rule collapses to the last crossing at every sane setting -
    after the point the ball gets picked up and flights just STOP, so
    there was nothing for a gap rule to defend against. That leaves one
    early close (r14, -0.79s, one contact destroyed) and the question
    of whether a small pad that rescues it is worth +pad of junk
    everywhere else. That is a loss-function question, so it goes in
    the sweep, not in an argument. fb=1 falls back to motion when the
    rally has no crossings at all - without it the family abstains on
    any crossing-less rally and is disqualified from the winner's
    table."""
    cts = r.get("crossings") or []
    if cts:
        return cts[-1] + p["pad"]
    if p.get("fb"):
        return r_motion(r, {"frac": 0.45, "hold": 1.0, "tail": 1.2},
                        meta)
    return None


def r_motion(r, p, meta):
    """rally_end_motion, replayed from the dumped series. Identical
    logic, so a parameter found here transfers to the real detector
    without a re-derivation."""
    m = r.get("motion")
    if not m or len(m["t"]) < 10:
        return None
    ts, sp = m["t"], m["speed"]
    t0 = ts[0]
    early = sorted(s for t, s in zip(ts, sp) if t <= t0 + 3.0)
    if not early:
        return None
    ref = early[len(early) // 2]
    if ref <= 0:
        return None
    thr = p["frac"] * ref
    last = None
    for i, (t, s) in enumerate(zip(ts, sp)):
        if s < thr:
            continue
        quiet = [(tt, ss) for tt, ss in zip(ts[i + 1:], sp[i + 1:])
                 if tt <= t + p["hold"]]
        if quiet and all(ss < thr for _t, ss in quiet):
            last = t
            break
        last = t
    return None if last is None else last + p["tail"]


def r_retreat(r, p, meta):
    """THE USER'S OTHER IDEA (2026-08-22): the point is over when the
    players walk away from the kitchen - "all four moving away from the
    kitchen probably (3/4?)".

    Net side is taken from the dumped net_y where the ball channel
    supplied one, so "away" is a real direction rather than a guess;
    without it the rule abstains rather than inventing an orientation.
    A player is retreating when their distance from the net line has
    grown over `win` seconds. Fires at the last instant FEWER than
    `k` players were retreating - i.e. the last moment somebody was
    still closing on the ball."""
    m, net = r.get("motion"), r.get("net_y")
    if not m or net is None or len(m["t"]) < 10:
        return None
    ts = m["t"]
    step = (ts[-1] - ts[0]) / max(1, len(ts) - 1)
    back = max(1, int(round(p["win"] / step))) if step > 0 else 1
    tids = sorted(m["xy"])
    if len(tids) != 4:
        return None
    last = None
    for i in range(back, len(ts)):
        n_ret = 0
        for tid in tids:
            a, b = m["xy"][tid][i - back], m["xy"][tid][i]
            if a is None or b is None:
                continue
            if abs(b[1] - net) > abs(a[1] - net) + p["eps"]:
                n_ret += 1
        if n_ret < p["k"]:
            last = ts[i]
    return None if last is None else last + p["tail"]


def r_later(r, p, meta):
    """Later of resume and motion. Two instruments that fail in
    OPPOSITE directions - crossings go blind when the ball is lost mid
    rally, motion runs long through the walk-off - so taking the later
    one is the conservative combination: it can only be early if BOTH
    are early at once."""
    a = r_resume(r, {"gap": p["gap"], "look": p["look"], "need": 1}, meta)
    b = r_motion(r, {"frac": p["frac"], "hold": 1.0, "tail": 0.0}, meta)
    vals = [v for v in (a, b) if v is not None]
    return max(vals) + p["tail"] if vals else None


def r_earlier(r, p, meta):
    """Earlier of the two. The aggressive combination - included only
    so the sweep can price it rather than have it asserted away."""
    a = r_resume(r, {"gap": p["gap"], "look": p["look"], "need": 1}, meta)
    b = r_motion(r, {"frac": p["frac"], "hold": 1.0, "tail": 0.0}, meta)
    vals = [v for v in (a, b) if v is not None]
    return min(vals) + p["tail"] if vals else None


def _grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, c)) for c in itertools.product(*kw.values())]


RULES = {
    "last_cross": (r_last_cross, [{}]),
    "gap": (r_gap, _grid(gap=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])),
    "selfcal": (r_selfcal, _grid(floor=[2.0, 2.5, 3.0, 4.0],
                                 mult=[2.0, 3.0, 4.0, 5.0])),
    "resume": (r_resume, _grid(gap=[2.0, 3.0, 4.0, 5.0, 6.0],
                               look=[5.0, 8.0, 10.0, 12.0],
                               need=[1, 2])),
    "cross_pad": (r_cross_pad, _grid(pad=[0.0, 0.5, 0.8, 1.0, 1.5,
                                          2.0, 3.0],
                                     fb=[0, 1])),
    "motion": (r_motion, _grid(frac=[0.30, 0.45, 0.60, 0.75],
                               hold=[0.6, 1.0, 1.5],
                               tail=[0.0, 0.6, 1.2])),
    "retreat": (r_retreat, _grid(k=[2, 3, 4], win=[0.6, 1.0, 1.6],
                                 eps=[0.0, 2.0, 5.0],
                                 tail=[0.0, 0.6, 1.2])),
    "later": (r_later, _grid(gap=[3.0, 4.0, 5.0], look=[8.0, 10.0],
                             frac=[0.45, 0.60], tail=[0.0, 0.6])),
    "earlier": (r_earlier, _grid(gap=[3.0, 4.0, 5.0], look=[8.0, 10.0],
                                 frac=[0.45, 0.60], tail=[0.0, 0.6])),
}


# --------------------------------------------------------------- scoring

def evaluate(fn, params, rallies, meta):
    """[(rally_id, d)] for rallies where the rule fires AND truth
    exists. Abstentions are dropped here and counted by the caller -
    never scored as zero, which would reward a rule for staying
    silent."""
    out = []
    for rid, r in sorted(rallies.items()):
        tru = max(r.get("true_contacts") or [], default=None)
        if tru is None:
            continue
        t = fn(r, params, meta)
        if t is None:
            continue
        out.append((rid, t - tru))
    return out


def summarize(ds):
    xs = sorted(d for _r, d in ds)
    n = len(xs)
    return {
        "n": n,
        "min": xs[0], "max": xs[-1], "med": xs[n // 2],
        "mean_loss": statistics.fmean(loss(x) for x in xs),
        "within2": sum(1 for x in xs if abs(x) <= 2.0),
        "early": sum(1 for x in xs if x < 0),
    }


def best_params(fn, grid, rallies, meta, n_need):
    """Lowest mean loss, with a rule that abstains on any scoreable
    rally disqualified outright - a window that does not exist is not
    a window."""
    best = None
    for params in grid:
        ds = evaluate(fn, params, rallies, meta)
        if len(ds) < n_need:
            continue
        s = summarize(ds)
        if best is None or s["mean_loss"] < best[0]:
            best = (s["mean_loss"], params, s)
    return best


def loo(fn, grid, rallies, meta):
    """Pick parameters on every rally but one, score the one. The only
    number here that forecasts anything."""
    ids = sorted(rallies)
    held = []
    for rid in ids:
        rest = {k: v for k, v in rallies.items() if k != rid}
        n_need = sum(1 for v in rest.values() if v.get("true_contacts"))
        b = best_params(fn, grid, rest, meta, n_need)
        if b is None:
            continue
        ds = evaluate(fn, b[1], {rid: rallies[rid]}, meta)
        if ds:
            held.append(ds[0])
    return held


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", nargs="?", default="dump.json")
    ap.add_argument("--rule", default=None,
                    help="sweep only this rule, and print every "
                         "parameter setting rather than the winner")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    with open(a.dump) as fh:
        blob = json.load(fh)
    rallies = {int(k): v for k, v in blob["rallies"].items()}
    meta = {k: v for k, v in blob.items() if k != "rallies"}
    scoreable = {k: v for k, v in rallies.items() if v.get("true_contacts")}
    n_x = sum(1 for v in scoreable.values() if v.get("crossings"))
    n_m = sum(1 for v in scoreable.values() if v.get("motion"))
    print(f"dump: {blob.get('build', '?')}")
    print(f"  {len(rallies)} rallies, {len(scoreable)} with truth, "
          f"{n_x} with crossings, {n_m} with motion")
    print(f"  loss: early {EARLY_COST}/s, late {LATE_COST}/s")

    if a.rule:
        fn, grid = RULES[a.rule]
        rows = []
        for params in grid:
            ds = evaluate(fn, params, scoreable, meta)
            if ds:
                rows.append((summarize(ds), params))
        rows.sort(key=lambda r: r[0]["mean_loss"])
        print(f"\n{a.rule}: {len(rows)} settings that fire")
        print(f"    {'loss':>7}{'n':>4}{'min':>8}{'med':>8}{'max':>8}"
              f"{'<=2s':>6}{'early':>7}  params")
        for s, params in rows:
            print(f"    {s['mean_loss']:>7.2f}{s['n']:>4}{s['min']:>+8.2f}"
                  f"{s['med']:>+8.2f}{s['max']:>+8.2f}"
                  f"{s['within2']:>6}{s['early']:>7}  {params}")
        return

    print("\nBEST SETTING PER RULE (in-sample - optimistic by "
          "construction)")
    print(f"    {'rule':<12}{'loss':>7}{'n':>4}{'min':>8}{'med':>8}"
          f"{'max':>8}{'<=2s':>6}  params")
    order = []
    for name, (fn, grid) in RULES.items():
        b = best_params(fn, grid, scoreable, meta, len(scoreable))
        if b is None:
            print(f"    {name:<12}  abstains on at least one rally")
            continue
        _l, params, s = b
        order.append((s["mean_loss"], name))
        print(f"    {name:<12}{s['mean_loss']:>7.2f}{s['n']:>4}"
              f"{s['min']:>+8.2f}{s['med']:>+8.2f}{s['max']:>+8.2f}"
              f"{s['within2']:>6}  {params}")

    print("\nLEAVE-ONE-OUT (parameters chosen without the rally they "
          "are scored on)")
    print("  this is the forecast. A rule whose LOO is much worse than "
          "its in-sample\n  row was tuned to these 15 rallies and will "
          "not survive the holdout.")
    print(f"    {'rule':<12}{'loss':>7}{'n':>4}{'min':>8}{'med':>8}"
          f"{'max':>8}{'<=2s':>6}{'early':>7}")
    for name, (fn, grid) in RULES.items():
        held = loo(fn, grid, scoreable, meta)
        if not held:
            continue
        s = summarize(held)
        print(f"    {name:<12}{s['mean_loss']:>7.2f}{s['n']:>4}"
              f"{s['min']:>+8.2f}{s['med']:>+8.2f}{s['max']:>+8.2f}"
              f"{s['within2']:>6}{s['early']:>7}")

    print("\n  min is the number that matters: closing early destroys "
          "contacts no later\n  stage can recover, closing late only "
          "admits junk the DP already filters.\n  Benchmarks from the "
          "2026-08-22d run: motion +8.97 med / +3.57 min, last "
          "crossing\n  +3.97 / -0.79, last-before-gap disqualified at "
          "-17.78 min.")


def selftest():
    # a rule that abstains must not be scored on the subset it likes
    rs = {1: {"crossings": [0.0, 1.0], "true_contacts": [0.0, 1.0]},
          2: {"crossings": [], "true_contacts": [0.0, 5.0]}}
    ds = evaluate(r_last_cross, {}, rs, {})
    assert len(ds) == 1 and ds[0][0] == 1, ds
    assert best_params(r_last_cross, [{}], rs, {}, 2) is None

    # asymmetry: one second early must cost ten times one second late
    assert loss(-1.0) == 10.0 * loss(1.0)

    # resume: a gap that never resumes ends it; one that does, does not
    r = {"crossings": [0.0, 1.0, 2.0, 15.0], "true_contacts": [2.0]}
    p = {"gap": 4.0, "look": 8.0, "need": 1}
    assert r_resume(r, p, {}) == 2.0
    r2 = {"crossings": [0.0, 1.0, 2.0, 9.0]}
    assert r_resume(r2, p, {}) == 9.0

    # cross_pad: pad shifts the last crossing; no crossings -> abstain
    # unless fb, in which case motion answers
    assert r_cross_pad({"crossings": [1.0, 4.0]}, {"pad": 0.8}, {}) == 4.8
    assert r_cross_pad({"crossings": []}, {"pad": 0.8}, {}) is None

    # retreat, on a rally shape: all four CLOSE on the net for the
    # first two seconds, then all four walk off. The end is the
    # turnaround, not the end of the series.
    _ts = [i * 0.2 for i in range(30)]
    _y = [100.0 - 5.0 * i if i <= 10 else 50.0 + 5.0 * (i - 10)
          for i in range(30)]
    m = {"t": _ts, "speed": [1.0] * 30,
         "xy": {str(i): [[0.0, v] for v in _y] for i in range(4)}}
    # abstains without a net line rather than guessing an orientation
    assert r_retreat({"motion": m, "net_y": None},
                     {"k": 3, "win": 1.0, "eps": 0.0, "tail": 0.0},
                     {}) is None
    out = r_retreat({"motion": m, "net_y": 0.0},
                    {"k": 3, "win": 1.0, "eps": 0.0, "tail": 0.0}, {})
    assert out is not None and 2.0 <= out <= 3.2, out
    # and when nobody ever closes - the window opened after the point
    # was already over - it abstains instead of naming the first sample
    _off = {"t": _ts, "speed": [1.0] * 30,
            "xy": {str(i): [[0.0, 100.0 + j] for j in range(30)]
                   for i in range(4)}}
    assert r_retreat({"motion": _off, "net_y": 0.0},
                     {"k": 3, "win": 1.0, "eps": 0.0, "tail": 0.0},
                     {}) is None

    # motion replay reproduces the shape of the real detector: a rally
    # that goes quiet halfway ends halfway, plus the tail
    ts = [i * 0.2 for i in range(40)]
    sp = [10.0 if t < 4.0 else 0.5 for t in ts]
    got = r_motion({"motion": {"t": ts, "speed": sp, "xy": {}}},
                   {"frac": 0.45, "hold": 1.0, "tail": 1.2}, {})
    assert got is not None and abs(got - (3.8 + 1.2)) < 1e-6, got

    # later/earlier bracket their inputs
    rr = {"crossings": [0.0, 1.0, 2.0, 20.0],
          "motion": {"t": ts, "speed": sp, "xy": {}}}
    pp = {"gap": 4.0, "look": 8.0, "frac": 0.45, "tail": 0.0}
    lo_, hi_ = r_earlier(rr, pp, {}), r_later(rr, pp, {})
    assert lo_ <= hi_, (lo_, hi_)

    # LOO never lets a rally choose its own parameters: a grid with one
    # setting that is perfect on rally 3 alone must not win there
    rs3 = {1: {"crossings": [0.0, 1.0], "true_contacts": [1.0]},
           2: {"crossings": [0.0, 1.0], "true_contacts": [1.0]},
           3: {"crossings": [0.0, 9.0], "true_contacts": [9.0]}}
    held = loo(r_gap, _grid(gap=[2.0, 20.0]), rs3, {})
    d3 = dict(held)[3]
    assert d3 < 0, held      # 1 and 2 pick the tight gap; 3 pays for it
    print("selftest OK: abstention accounting, asymmetric loss, resume, "
          "retreat (abstains without a net line), motion replay, "
          "later/earlier bracketing, leave-one-out isolation")


if __name__ == "__main__":
    main()
