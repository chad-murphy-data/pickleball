"""B2b — inference for the switch-vs-placebo difference-in-differences.

    python model/weather_review/b2b_did.py <rebuilt_splits.csv>

The identifying idea: the pre/post-6 swing statistic can be computed for
EVERY sideout-to-11 game, but the ends only actually change at 6 in
MLP games and PPA deciders. PPA games 1-2 of a best-of-3 give the same
statistic with NO end change — a placebo with the same mechanics, the
same weather exposure and no decider selection. The wind effect
attributable to the end change is

  DiD = [z2(windy) - z2(calm) | switch] - [z2(windy) - z2(calm) | no switch]

with both inner contrasts computed within event (paired). Inference:
(1) event cluster bootstrap, 4000 draws; (2) an exact randomization test
that reshuffles the wind label across games WITHIN each event x arm cell,
holding the cell's windy/calm counts fixed — exact under the sharp null
"wind changes nothing in either arm"; (3) the continuous analogue, a
within-event slope of z2 on match-hour wind estimated separately in each
arm, differenced.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2b_lib as L  # noqa: E402

REF = "OUTDOOR calm <8"


def build(matches, source, gmeta, strict, scope):
    out = []
    for (mid, gn), r in source.items():
        m = matches.get(mid)
        g = gmeta.get((mid, gn))
        if not m or not g or g["scoring_format"] != "sideout_11":
            continue
        if m["tour"] == "MLP":
            is_switch = gn == 1
            if gn != 1:
                continue
        else:
            is_switch = ((m["best_of"] == 3 and gn == 3) or
                         (m["best_of"] == 5 and gn == 5))
        if scope == "switch" and not is_switch:
            continue
        if scope == "noswitch" and (is_switch or m["tour"] == "MLP"):
            continue
        if strict:
            if r["seq_ok"] != "1" or r["boundary_ok"] != "1":
                continue
            if sorted([int(r["fa"]), int(r["fb"])]) != \
                    sorted([int(g["t1_score"]), int(g["t2_score"])]):
                continue
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        grp = L.group_of(m)
        if not grp:
            continue
        _, _, z2 = L.zsq(int(r["pa_pre"]), pre, int(r["pa_post"]), post)
        out.append((m["event"], grp, m["wind"], m["setting"], z2))
    return out


def cells(units, grp):
    """{event: (windy_vals, calm_vals)} restricted to events with both."""
    T, C = defaultdict(list), defaultdict(list)
    for e, g, w, s, z in units:
        if g == grp:
            T[e].append([z])
        elif g == REF:
            C[e].append([z])
    return {e: (T[e], C[e]) for e in T if e in C}


def did(sw, ns, grp, weight="fe"):
    a, b = cells(sw, grp), cells(ns, grp)
    if not a or not b:
        return float("nan"), None, None
    return L.paired_diff(a, weight) - L.paired_diff(b, weight), a, b


def main():
    reb = {(r["match_id"], int(r["game_number"])): r
           for r in L.read_csv(sys.argv[1])}
    gmeta = {}
    for g in L.read_csv(L.ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        gmeta[(g["match_id"], int(g["game_number"]))] = g
    arms = L.label_arms()
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# B2b — switch vs no-switch placebo: DiD inference\n")
    say("| labels | strict | bin | switch paired-FE | placebo paired-FE | "
        "DiD [boot 95%] | randomization p (1-sided) | events |")
    say("|---|---|---|---|---|---|---|---|")
    for lab in ("published", "corrected_all", "corrected_hi"):
        matches = L.load_matches(arms[lab])
        for strict in (False, True):
            sw = build(matches, reb, gmeta, strict, "switch")
            ns = build(matches, reb, gmeta, strict, "noswitch")
            for grp in ("OUTDOOR moderate 8-14", "OUTDOOR windy 14+"):
                est, a, b = did(sw, ns, grp)
                if a is None:
                    continue
                s_in = L.paired_diff(a, "fe")
                n_in = L.paired_diff(b, "fe")
                evs = sorted(set(a) | set(b))

                def stat(keys):
                    p1 = {e: a[e] for e in keys if e in a}
                    p2 = {e: b[e] for e in keys if e in b}
                    if not p1 or not p2:
                        return float("nan")
                    return L.paired_diff(p1, "fe") - L.paired_diff(p2, "fe")
                lo, hi = L.cluster_boot({e: e for e in evs},
                                        lambda ss: stat(list(ss)), n=4000)
                rng = random.Random(4242)
                hits = 0
                REPS = 3000
                for _ in range(REPS):
                    pa, pb = {}, {}
                    for cell, dst in ((a, pa), (b, pb)):
                        for e, (t, c) in cell.items():
                            pool = list(t) + list(c)
                            rng.shuffle(pool)
                            dst[e] = (pool[:len(t)], pool[len(t):])
                    v = L.paired_diff(pa, "fe") - L.paired_diff(pb, "fe")
                    if v >= est:
                        hits += 1
                p1v = (hits + 1) / (REPS + 1)
                say(f"| {lab} | {'yes' if strict else 'no'} | "
                    f"{grp.replace('OUTDOOR ', '')} | {s_in:+.3f} | "
                    f"{n_in:+.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                    f"| {p1v:.3f} | {len(evs)} |")

    # ---- continuous analogue: within-event wind slope, per arm, differenced
    say("\n## Continuous analogue — within-event slope of z2 on match-hour "
        "wind (outdoor only), by arm\n")
    say("| labels | switch slope /10mph [95% CI] | placebo slope /10mph "
        "[95% CI] | difference [95% CI] |")
    say("|---|---|---|---|")
    for lab in ("published", "corrected_all"):
        matches = L.load_matches(arms[lab])
        sw = [u for u in build(matches, reb, gmeta, False, "switch")
              if u[3] == "outdoor"]
        ns = [u for u in build(matches, reb, gmeta, False, "noswitch")
              if u[3] == "outdoor"]
        byev = defaultdict(lambda: ([], []))
        for e, g, w, s, z in sw:
            byev[e][0].append((w, z))
        for e, g, w, s, z in ns:
            byev[e][1].append((w, z))

        def sl(cells_, idx):
            num = den = 0.0
            for c in cells_:
                pts = c[idx]
                if len(pts) < 2:
                    continue
                mw = sum(w for w, _ in pts) / len(pts)
                mz = sum(z for _, z in pts) / len(pts)
                for w, z in pts:
                    num += (w - mw) * (z - mz)
                    den += (w - mw) ** 2
            return num / den if den else float("nan")
        allc = list(byev.values())
        s1, s2 = sl(allc, 0), sl(allc, 1)
        b1 = L.cluster_boot(byev, lambda ss: sl(ss, 0), n=3000)
        b2 = L.cluster_boot(byev, lambda ss: sl(ss, 1), n=3000)
        bd = L.cluster_boot(byev, lambda ss: sl(ss, 0) - sl(ss, 1), n=3000)
        say(f"| {lab} | {s1*10:+.3f} [{b1[0]*10:+.3f}, {b1[1]*10:+.3f}] "
            f"| {s2*10:+.3f} [{b2[0]*10:+.3f}, {b2[1]*10:+.3f}] "
            f"| {(s1-s2)*10:+.3f} [{bd[0]*10:+.3f}, {bd[1]*10:+.3f}] |")

    (HERE / "b2b_did.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/weather_review/b2b_did.md")


if __name__ == "__main__":
    main()
