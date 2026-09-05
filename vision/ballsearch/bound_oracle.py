"""Bound oracle: which stage loses the bounces?  (2026-09-04)

Every pair of contacts makes ONE call (bounce or not).  The bounce
counter is therefore only as good as its flight BOUNDARIES, and the
autopsy (bounce_autopsy.py) said 13 of 35 human bounces on r7/r9/r10/r17.
This script swaps each stage for its oracle, one at a time, and grades
the same 35 bounces.  Human contacts are an ORACLE for the bounds only:
nothing is trained, nothing is tuned, no seal is touched.

Grid.  Bounds source x demotion policy, tracked ball observations
throughout (the tracked path is as accurate as the hand-labeled one on
the failing flights: median 4.5 px, p90 9 px -- measured before this):

  --bounds tracked     the claimed bounds exactly as the pipeline makes them
  --bounds recall      tracked + the human contacts the claim MISSED (17/79)
  --bounds precision   tracked minus the bounds that match NO human contact
                       (34 of them; the removed times become bounce markers,
                       which is what the demotion step does with a bound)
  --bounds human       the human contacts (both fixed)
  --demotion none | shipped | validated
       shipped   = ball_replicate.crossing_demotion (drop a bound whose
                   following flight never crosses the net, 3 rounds)
       validated = shipped, but only when the merged flight [k-1, k+1]
                   is itself _plausible (no new knob)
  --first-pass         fit_segment per flight only, no consensus sweeps
  --policy raw | dedup raw = anchors as ball_replicate.main and the
                   autopsy pass them; dedup = ball_grade check 3
                   (dedupe_anchors, and every unclaimed turn is a
                   bounce marker -- check 3 skips bounce_shaped)

RESULTS, human bounces matched (+-0.30 s) out of 35 (raw policy):

    bounds \\ demotion    none   shipped   validated
    tracked               11      13*        13        * = the autopsy's 13
    recall (contacts in)  14      14
    precision (junk out)  14      14
    human                 25      20                   (first pass only: 23)

  Per rally r7/r9/r10/r17: shipped 3/6/2/2, human/none 3/9/9/4.
  Check 3's own anchor policy (--policy dedup): tracked none 13 /
  shipped 12 / validated 13, contacts 57/79, junk 22.
  What the grid says: the fitter recovers 25/35 when it is handed the
  right flights; removing demotion (11) or validating it (13) does not
  move the shipped number; fixing contact recall alone (14) or junk
  alone (14) barely does; fixing both does (25).  Read by FLIGHTS: a
  bounce needs an intact flight (both contacts within MATCH_S, no bound
  between).  13 of the 35 bounce-holding flights are intact in the
  claimed bounds (16 after demotion); on intact flights the counter
  hits 11/16, on broken ones 2/19.  Of the 22 broken: 9 missing end +
  junk inside, 7 junk only, 6 missing end only -- the two defects break
  the same flights, so either fix alone is +3 and both are +14.

The 34 junk bounds (tracked, raw policy), by what they sit on:
    8  ON a human bounce (within 0.05 s; the bounce-turn was claimed as
       a contact -- the flight is split exactly at the bounce, so neither
       piece can hold it)
    8  duplicate of a contact another bound already matched
    7  mistimed claim of a contact that is otherwise MISSED (0.27-0.41 s
       off; MATCH_S is 0.25)
   11  far from anything
  The bounce-shape sign test cannot veto the claims: 60% of REAL contact
  bounds are also fall-then-rise (dinks and low volleys), vs 88% of the
  on-a-bounce ones.

Shipped demotion on tracked bounds removes 12 bounds over the four
rallies, 5 of them REAL contacts (r9 258.77/259.12, r10 306.02,
r17 431.48, and r10 308.75 under validated); on the human flights its
premise fails 7 times in 71 (dinks whose drawn path never reaches the
net).  It is net +2 on the tracked bounds only because it also removes
junk (r7: three junk bounds, one of them on the bounce at 170.16).  On
human bounds every one of its 11 removals is a real contact (25 -> 20);
on junk-free tracked bounds all 7 are (14 -> 14).

Robustness: court3d.fit_arc raises LinAlgError (singular matrix) on a
2-observation piece; this script guards it (rms=inf -> rejected).  The
shipped fitter does not.

    python3 vision/ballsearch/bound_oracle.py --rally 7 --bounds recall --demotion none
    python3 vision/ballsearch/bound_oracle.py --summary
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ball_replicate as br                                    # noqa: E402
import ball_decoder as bdec                                    # noqa: E402
import court3d as c3                                           # noqa: E402
from make_ball_audit import detect_events                      # noqa: E402
from claim_lab import load as c3load                           # noqa: E402
from bounce_autopsy import tracked as autopsy_tracked          # noqa: E402

RAL = [7, 9, 10, 17]
KINDS = ["tracked", "recall", "precision", "anchors", "anchors2", "human"]
ON_BOUNCE_S = 0.05

_fit_arc = c3.fit_arc


def fit_arc_safe(*a, **k):
    try:
        return _fit_arc(*a, **k)
    except np.linalg.LinAlgError:
        th0 = k.get("theta0", a[3] if len(a) > 3 else None)
        return (np.array(th0 if th0 is not None else c3.default_inits()[0],
                         float), float("inf"))


c3.fit_arc = fit_arc_safe


# ------------------------------------------------------------- grading

def seg_index(bounds, ts):
    for k in range(len(bounds) - 1):
        if bounds[k] <= ts <= bounds[k + 1]:
            return k
    return None


def grade(segs, bounds, h_bnc):
    """bounce_autopsy's buckets (minus CAPPED): matched / NO WINDOW /
    NO SEG / NOT OK / WRONG TIME / CALLED ARC, per human bounce."""
    called = {k: float(s["ts"]) for k, s in enumerate(segs)
              if s and s.get("ok") and s["kind"] == "bounce"}
    out = {}
    for ts in h_bnc:
        k = seg_index(bounds, ts)
        if k is None:
            b = "NO WINDOW"
        elif k in called and abs(called[k] - ts) <= br.BOUNCE_MATCH_S:
            b = "matched"
        elif segs[k] is None:
            b = "NO SEG"
        elif not segs[k].get("ok"):
            b = "NOT OK"
        elif k in called:
            b = "WRONG TIME"
        else:
            b = "CALLED ARC"
        out[ts] = b
    return out


def human_bounces(c):
    return [float(s["ts"]) for s in c["h_segs"]
            if s and s.get("ok") and s["kind"] == "bounce"]


# -------------------------------------------------------------- bounds

def predem(c, policy):
    """The tracked side BEFORE demotion, under either anchor policy.
    Fresh decode (the c3 cache's turns predate the fitter commits and a
    cache-based claim does not reproduce tracked_side's bound times);
    cached per rally because the decode is minutes."""
    r = c["rally"]
    cp = HERE / f"bound_oracle_predem_r{r}.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            d = pickle.load(f)
    else:
        serve, end = c["imps"][0], c["dead"]
        byf, t0 = bdec.load_candidates(r)
        f_min = round((serve - 0.3 - t0) * bdec.FPS)
        f_max = round((end + 0.3 - t0) * bdec.FPS)
        byf = {f: cc for f, cc in byf.items() if f_min <= f <= f_max}
        oflags = bdec.out_of_court_flags(byf, bdec.court_hull())
        visited = bdec.decode(byf, None, oflags, None)
        pts = [(t0 + f / bdec.FPS, x, y) for f, x, y in visited]
        _, timing_ref = bdec.timing_decode(byf, None, oflags, t0, [])
        turns = [e for e in detect_events(timing_ref)
                 if serve - 0.3 <= e < end - 0.05]
        angs = br.turn_angles(timing_ref, turns)
        obs = [(tt, x, y, 1.0) for tt, x, y in pts]
        d = dict(obs=obs, turns=turns, timing_ref=timing_ref, claims={})
        dd = br.dedupe_anchors(c["anchors"], c["zs"],
                               br.track_sides(c["floors"]), turns)
        for name, anchors in (("raw", c["anchors"]), ("dedup", dd)):
            matched = br.claim_bounds(turns, angs, timing_ref, anchors)
            claimed = set(matched)
            d["claims"][name] = dict(
                anchors=anchors, bounds=matched + [end],
                evs_shaped=[e for e in turns if e not in claimed
                            and br.bounce_shaped(timing_ref, e)],
                evs_all=[e for e in turns if e not in claimed])
        with open(cp, "wb") as f:
            pickle.dump(d, f)
    cl = d["claims"][policy]
    evs = cl["evs_shaped"] if policy == "raw" else cl["evs_all"]
    return d["obs"], list(cl["bounds"]), list(evs), cl["anchors"]


def is_real(b, imps):
    return any(abs(hc - b) <= br.MATCH_S for hc in imps)


def make_bounds(kind, t_bounds, t_evs, imps, anchors=(), zs=()):
    """Returns (bounds, evs, note)."""
    if kind == "tracked":
        return list(t_bounds), list(t_evs), ""
    if kind == "recall":
        missed = [hc for hc in imps
                  if not any(abs(hc - b) <= br.MATCH_S for b in t_bounds)]
        return (sorted(t_bounds + missed), list(t_evs),
                f"inserted {len(missed)} missed contacts")
    if kind == "precision":
        inner = t_bounds[1:-1]
        gone = [b for b in inner if not is_real(b, imps)]
        keep = [t_bounds[0]] + [b for b in inner if is_real(b, imps)] \
            + [t_bounds[-1]]
        return keep, sorted(t_evs + gone), f"removed {len(gone)} junk bounds"
    if kind == "human":
        return list(imps) + [t_bounds[-1]], list(t_evs), "human contacts"
    if kind in ("anchors", "anchors2"):
        # LABEL-FREE claim-step change (2026-09-05): a pose anchor that
        # claimed no turn and has no bound within MATCH_S sets its own
        # bound. Motivated by miss anatomy: 10 of the 17 missed contacts
        # have an anchor within MATCH_S and no turn (6 have no anchor,
        # 3 have both and lost the claim). anchors2 = same, z >= 2.0
        # (one pre-stated threshold, not tuned).
        zmin = 2.0 if kind == "anchors2" else -1e9
        add = [a[0] for a, z in zip(anchors, zs) if z >= zmin
               and not any(abs(a[0] - b) <= br.MATCH_S for b in t_bounds)
               and t_bounds[0] < a[0] < t_bounds[-1]]
        return (sorted(t_bounds + add), list(t_evs),
                f"added {len(add)} anchor-only bounds")
    raise ValueError(kind)


# ------------------------------------------------------------ demotion

def validated_demotion(P, obs, bounds, evs, floors, anchors, rounds=3):
    """crossing_demotion, but a non-crossing bound is dropped only when
    the merged flight [k-1, k+1] is itself plausible."""
    log = []
    for _ in range(rounds):
        pa = br.bound_anchor_positions(bounds, anchors, floors)
        segs, cons = br.reconstruct(P, obs, bounds, evs, pa, corridor=True)
        demote = None
        for k, seg in enumerate(segs):
            if seg is None or not seg["ok"] or k == 0:
                continue
            ys = np.array([p[2] for p in c3.sample_path(seg)])
            if ys.min() < c3.NET_Y < ys.max():
                continue
            t0, t1 = bounds[k - 1], bounds[k + 1]
            mo = [o for o in obs
                  if t0 + br.END_TRIM_S <= o[0] <= t1 - br.END_TRIM_S]
            if len(mo) < 5:
                log.append((bounds[k], "kept: merge too thin"))
                continue
            corr = (pa[k - 1], pa[k + 1] if k + 1 < len(pa) else None)
            m = c3.fit_segment(P, mo, t0, t1, evs, corridor=corr)
            if br._plausible(m):
                demote = k
                log.append((bounds[k], f"demoted: merge ok rms {m['rms']:.1f}"))
                break
            log.append((bounds[k], f"kept: merge not plausible rms {m['rms']:.1f}"))
        if demote is None:
            return segs, cons, bounds, evs, log
        evs = sorted(evs + [bounds[demote]])
        bounds = bounds[:demote] + bounds[demote + 1:]
    pa = br.bound_anchor_positions(bounds, anchors, floors)
    segs, cons = br.reconstruct(P, obs, bounds, evs, pa, corridor=True)
    return segs, cons, bounds, evs, log


def first_pass(P, obs, bounds, evs, pa):
    segs = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        o = [x for x in obs if a + br.END_TRIM_S <= x[0] <= b - br.END_TRIM_S]
        if len(o) < 5:
            segs.append(None)
            continue
        s = c3.fit_segment(P, o, a, b, evs,
                           corridor=(pa[k], pa[k + 1] if k + 1 < len(pa) else None))
        s["ok"] = br._plausible(s)
        segs.append(s)
    return segs


# ---------------------------------------------------------------- cell

def run_cell(r, kind, demotion, policy, fp):
    c = c3load(r)
    P, floors, imps = c["P"], c["floors"], list(c["imps"])
    h_bnc = human_bounces(c)
    _, segs_a, bounds_a, _ = autopsy_tracked(c)
    base = grade(segs_a, bounds_a, h_bnc)
    obs, t_bounds, t_evs, anchors = predem(c, policy)
    bounds, evs, note = make_bounds(kind, t_bounds, t_evs, imps,
                                    anchors, c["zs"] if policy == "raw"
                                    else [0.0] * len(anchors))
    t = time.time()
    log = []
    if fp:
        pa = br.bound_anchor_positions(bounds, anchors, floors)
        segs, bounds2 = first_pass(P, obs, bounds, evs, pa), list(bounds)
    elif demotion == "none":
        pa = br.bound_anchor_positions(bounds, anchors, floors)
        segs, _ = br.reconstruct(P, obs, list(bounds), list(evs), pa,
                                 corridor=True)
        bounds2 = list(bounds)
    elif demotion == "shipped":
        segs, _, bounds2, _ = br.crossing_demotion(
            P, obs, list(bounds), list(evs), floors, anchors)
    else:
        segs, _, bounds2, _, log = validated_demotion(
            P, obs, list(bounds), list(evs), floors, anchors)
    g = grade(segs, bounds2, h_bnc)
    m = sum(1 for v in g.values() if v == "matched")
    b0 = sum(1 for v in base.values() if v == "matched")
    tag = f"{kind}/{demotion}{'/first-pass' if fp else ''}[{policy}]"
    print(f"r{r} {tag}: {len(bounds)} -> {len(bounds2)} bounds; matched "
          f"{b0} -> {m} of {len(h_bnc)}  ({time.time()-t:.0f}s)  {note}")
    print("   buckets:", dict(Counter(g.values())))
    for gb in [b for b in bounds if b not in bounds2]:
        hc = min(imps, key=lambda h: abs(h - gb))
        print(f"   demoted {gb:.2f}: nearest human contact {hc:.2f} "
              f"(d {abs(hc-gb):.2f}s) -> "
              f"{'REAL' if abs(hc-gb) <= br.MATCH_S else 'junk'}")
    for tb, why in log:
        print(f"   [{tb:.2f}] {why}")
    for ts, v in g.items():
        if v != base[ts]:
            print(f"     {ts:8.2f}: {base[ts]} -> {v}")
    cm = sum(1 for hc in imps if any(abs(hc - b) <= br.MATCH_S for b in bounds2))
    sp = sum(1 for b in bounds2[:-1] if not is_real(b, imps))
    print(f"   contacts matched {cm}/{len(imps)}, junk bounds {sp}")
    out = dict(rally=r, bounds=kind, demotion=demotion, first_pass=fp,
               policy=policy, base=b0, after=m, n=len(h_bnc),
               buckets=dict(Counter(g.values())), contacts=[cm, len(imps)],
               junk=sp)
    name = f"bound_oracle_{kind}_{demotion}{'_fp' if fp else ''}_{policy}_r{r}.json"
    with open(HERE / name, "w") as f:
        json.dump(out, f)
    return out


def junk_anatomy(policy="raw"):
    """Where the junk bounds sit: on a bounce / duplicate / mistimed / far."""
    kinds = Counter()
    for r in RAL:
        c = c3load(r)
        imps = list(c["imps"])
        h_bnc = human_bounces(c)
        _, bounds, _, _ = predem(c, policy)
        inner = bounds[1:-1]
        matched = {hc for hc in imps
                   if any(abs(hc - b) <= br.MATCH_S for b in inner)}
        for b in inner:
            if is_real(b, imps):
                continue
            hc = min(imps, key=lambda h: abs(h - b))
            dc = abs(hc - b)
            db = min(abs(b - x) for x in h_bnc) if h_bnc else 9.0
            if db <= ON_BOUNCE_S:
                k = "on a human bounce"
            elif dc <= 0.5 and hc in matched:
                k = "duplicate of a matched contact"
            elif dc <= 0.5:
                k = "mistimed claim of a missed contact"
            else:
                k = "far from anything"
            kinds[k] += 1
    print(f"junk bounds [{policy}]: {sum(kinds.values())}")
    for k, n in kinds.most_common():
        print(f"  {n:>3}  {k}")


def intact_flights(bounds, imps, h_bnc):
    """How many bounce-holding human flights survive in these bounds:
    both contacts matched within MATCH_S and no bound in between. The
    grading number for claim-step work (2026-09-05): a bounce is only
    findable inside an intact flight (shipped: 11/16 intact vs 2/19
    broken), so this moves BEFORE the bounce count does, and needs no
    fit -- seconds, not minutes."""
    n = 0
    for ts in h_bnc:
        prev = max([c for c in imps if c <= ts], default=None)
        nxt = min([c for c in imps if c > ts], default=None)
        if prev is None or nxt is None:
            continue
        ok = (any(abs(prev - b) <= br.MATCH_S for b in bounds)
              and any(abs(nxt - b) <= br.MATCH_S for b in bounds)
              and not any(prev + br.MATCH_S < b < nxt - br.MATCH_S
                          for b in bounds))
        n += ok
    return n


def intact_table(policy="raw", kinds=KINDS):
    """Pre-demotion intact flights / contacts matched / junk per bounds
    variant. Measured 2026-09-05 (raw policy):
        tracked    13/35   62/79   35 junk
        recall     19/35   79/79   35
        precision  19/35   62/79    1
        anchors     8/35   67/79   59   <- anchor-only bounds: DEAD
        anchors2   12/35   66/79   45   <- (z >= 2) also dead
        human      34/35   79/79    0
    Each bound defect alone lifts intact 13 -> 19; both -> 34. The
    anchor-only claim (the obvious fix for the 10 'anchor present, no
    turn' misses) buys 4-5 contacts for 10-24 junk bounds and LOSES
    intact flights -- pose anchors are too fake-heavy to bound alone."""
    print(f"{'bounds':10s} {'intact':>7} {'contacts':>9} {'junk':>5}   per rally")
    for kind in kinds:
        tot, per = [0, 0, 0, 0], []
        for r in RAL:
            c = c3load(r)
            imps, hb = list(c["imps"]), human_bounces(c)
            _, tb, te, anch = predem(c, policy)
            zs = c["zs"] if policy == "raw" else [0.0] * len(anch)
            b, _, _ = make_bounds(kind, tb, te, imps, anch, zs)
            i = intact_flights(b, imps, hb)
            cm = sum(1 for hc in imps if is_real(hc, b))
            j = sum(1 for x in b[:-1] if not is_real(x, imps))
            tot[0] += i; tot[1] += len(hb); tot[2] += cm; tot[3] += j
            per.append(f"r{r} {i}/{len(hb)}")
        print(f"{kind:10s} {tot[0]:>3}/{tot[1]:<3} {tot[2]:>5}/79 "
              f"{tot[3]:>5}   " + "  ".join(per))


def summary(policy="raw"):
    cells = {}
    for p in HERE.glob(f"bound_oracle_*_{policy}_r*.json"):
        d = json.load(open(p))
        key = (d["bounds"], d["demotion"] + ("/fp" if d["first_pass"] else ""))
        cells.setdefault(key, {})[d["rally"]] = d
    print(f"human bounces matched of 35, policy={policy}  (per rally r7/r9/r10/r17)")
    print(f"{'bounds':10s} {'demotion':14s} {'total':>5}   per rally")
    for (kind, dem), by in sorted(cells.items()):
        if set(by) != set(RAL):
            print(f"{kind:10s} {dem:14s}   ...  {sorted(by)}")
            continue
        tot = sum(by[r]["after"] for r in RAL)
        per = " ".join(f"{by[r]['after']:>2}" for r in RAL)
        cm = sum(by[r]["contacts"][0] for r in RAL)
        sp = sum(by[r]["junk"] for r in RAL)
        print(f"{kind:10s} {dem:14s} {tot:>5}   {per}   contacts {cm}/79 junk {sp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", type=int)
    ap.add_argument("--bounds", default="tracked",
                    choices=KINDS)
    ap.add_argument("--demotion", default="none",
                    choices=["none", "shipped", "validated"])
    ap.add_argument("--first-pass", action="store_true")
    ap.add_argument("--policy", default="raw", choices=["raw", "dedup"])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--anatomy", action="store_true")
    ap.add_argument("--intact", action="store_true",
                    help="intact-flight table per bounds variant (no fits)")
    a = ap.parse_args()
    if a.summary:
        summary(a.policy)
        return
    if a.anatomy:
        junk_anatomy(a.policy)
        return
    if a.intact:
        intact_table(a.policy)
        return
    for r in ([a.rally] if a.rally else RAL):
        run_cell(r, a.bounds, a.demotion, a.policy, a.first_pass)


if __name__ == "__main__":
    main()
