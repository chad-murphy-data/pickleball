"""B2b — corrected-label + PAIRED re-run of the court-END analyses.

    python model/weather_review/b2b_end_effects.py

Leaves model/end_effects.py untouched. Recomputes Design B (point-share
swing across the mid-game switch at 6) and Design C (serve-rally win-rate
swing across the same switch) with

  (1) an explicit WITHIN-EVENT PAIRED contrast alongside the published
      unpaired dummy contrast, under three transparent weightings;
  (2) four indoor/outdoor label maps (published heuristic, corrected-all,
      corrected-high, audited-high-only);
  (3) the winprob.py serve-state simulated null for the excess.

All RNGs seeded. Writes model/weather_review/b2b_end_effects.md.
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # model/  -> end_effects
import b2b_lib as L                            # noqa: E402
from end_effects import sim_game               # noqa: E402  (reused, not modified)
sys.path.insert(0, str(L.ROOT / "web"))
from sitelib.winprob import serve_probs        # noqa: E402

REF = "OUTDOOR calm <8"
ORDER = ["INDOOR", "OUTDOOR calm <8", "OUTDOOR moderate 8-14", "OUTDOOR windy 14+"]


# ------------------------------------------------------------ build units
def design_b_units(matches, splits):
    """[(event, [z2], meta)] over the published Design-B game set."""
    out = []
    for r in splits:
        m = matches.get(r["match_id"])
        if not m:
            continue
        gn = int(r["game_number"])
        if m["tour"] == "MLP":
            if gn != 1:
                continue
        elif not (m["best_of"] == 3 and gn == 3) and \
                not (m["best_of"] == 5 and gn == 5):
            continue
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        _, noise, z2 = L.zsq(int(r["pa_pre"]), pre, int(r["pa_post"]), post)
        grp = L.group_of(m)
        if not grp:
            continue
        out.append((m["event"], [z2],
                    {"grp": grp, "wind": m["wind"], "eta": m["eta"] or 0.0,
                     "tour": m["tour"], "mid": r["match_id"], "gn": gn,
                     "outdoor": m["setting"] == "outdoor"}))
    return out


def design_c_units(matches, splits):
    out = []
    for r in splits:
        m = matches.get(r["match_id"])
        if not m:
            continue
        gn = int(r["game_number"])
        if m["tour"] == "MLP":
            if gn != 1:
                continue
        elif not (m["best_of"] == 3 and gn == 3) and \
                not (m["best_of"] == 5 and gn == 5):
            continue
        grp = L.group_of(m)
        if not grp:
            continue
        vals = []
        for side in ("a", "b"):
            rp, wp = int(r[f"r{side}_pre"]), int(r[f"w{side}_pre"])
            rq, wq = int(r[f"r{side}_post"]), int(r[f"w{side}_post"])
            if rp < 5 or rq < 5:
                continue
            _, noise, z2 = L.zsq(wp, rp, wq, rq)
            if noise <= 0:
                continue
            vals.append(z2)
        if not vals:
            continue
        out.append((m["event"], vals,
                    {"grp": grp, "wind": m["wind"], "eta": m["eta"] or 0.0,
                     "tour": m["tour"], "mid": r["match_id"], "gn": gn,
                     "outdoor": m["setting"] == "outdoor"}))
    return out


# ------------------------------------------------------------ sim null
def sim_null_b(units, reps=40, seed=20260729):
    rng = random.Random(seed)
    out = []
    for ev, vals, meta in units:
        kA, kB = serve_probs(meta["eta"])
        s = n = 0.0
        for _ in range(reps):
            pre, post, _, _, _ = sim_game(kA, kB, rng)
            n1, n2 = sum(pre), sum(post)
            if n1 < 5 or n2 < 5:
                continue
            _, noise, z = L.zsq(pre[0], n1, post[0], n2)
            s += z if noise > 0 else 0.0
            n += 1
        out.append(s / n if n else float("nan"))
    return out


def sim_null_c(units, reps=25, seed=20260730):
    rng = random.Random(seed)
    out = []
    for ev, vals, meta in units:
        kA, kB = serve_probs(meta["eta"])
        s = n = 0.0
        for _ in range(reps):
            _, _, _, _, sv = sim_game(kA, kB, rng)
            for t in (0, 1):
                rp, wp = sv["pre"][t]
                rq, wq = sv["post"][t]
                if rp < 5 or rq < 5:
                    continue
                _, noise, z = L.zsq(wp, rp, wq, rq)
                if noise > 0:
                    s += z
                    n += 1
        out.append(s / n if n else float("nan"))
    return out


# ------------------------------------------------------------ reporting
def group_table(say, units, sims, title):
    say(f"\n**{title} — per-group level and excess over the simulated null**\n")
    say("| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |")
    say("|---|---|---|---|---|")
    for grp in ORDER:
        sel = [(u, s) for u, s in zip(units, sims) if u[2]["grp"] == grp
               and s == s]
        if not sel:
            continue
        cl = defaultdict(list)
        for u, s in sel:
            cl[u[0]].append((u[1], s))
        flat = [p for v in cl.values() for p in v]
        mz = lambda smp: sum(v for pl, _ in smp for v in pl) / \
            sum(len(pl) for pl, _ in smp)
        mex = lambda smp: (sum(v for pl, _ in smp for v in pl) /
                           sum(len(pl) for pl, _ in smp)) - \
            (sum(s for _, s in smp) / len(smp))
        z = mz(flat)
        zlo, zhi = L.cluster_boot(cl, lambda ss: mz([p for s in ss for p in s]))
        ex = mex(flat)
        elo, ehi = L.cluster_boot(cl, lambda ss: mex([p for s in ss for p in s]))
        nullz = sum(s for _, s in flat) / len(flat)
        nobs = sum(len(pl) for pl, _ in flat)
        say(f"| {grp} | {nobs} | {z:.3f} [{zlo:.3f}, {zhi:.3f}] | {nullz:.3f} "
            f"| {ex:+.3f} [{elo:+.3f}, {ehi:+.3f}] |")


def contrast_table(say, units, title, perm=True):
    say(f"\n**{title} — contrast vs {REF}: UNPAIRED (published design) vs "
        "PAIRED (within-event)**\n")
    say("| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] "
        "| paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | "
        "perm p (1-sided / 2-sided) |")
    say("|---|---|---|---|---|---|---|---|")
    ref_units = [(e, v) for e, v, m in units if m["grp"] == REF]
    for grp in ORDER:
        if grp == REF:
            continue
        trt = [(e, v) for e, v, m in units if m["grp"] == grp]
        if not trt:
            continue
        # --- unpaired, exactly the published estimator/clustering
        merged = defaultdict(list)
        for e, v in ref_units:
            merged["R:" + e].append(("ref", v))
        for e, v in trt:
            merged["T:" + e].append(("trt", v))

        def diff(sample):
            t = [x for tag, pl in sample if tag == "trt" for x in pl]
            r = [x for tag, pl in sample if tag == "ref" for x in pl]
            if not t or not r:
                return float("nan")
            return sum(t) / len(t) - sum(r) / len(r)
        up = diff([p for v in merged.values() for p in v])
        ulo, uhi = L.cluster_boot(merged,
                                  lambda ss: diff([p for s in ss for p in s]))
        # --- paired
        pe = L.paired_events(trt, ref_units)
        if not pe:
            continue
        p_fe = L.paired_diff(pe, "fe")
        p_att = L.paired_diff(pe, "att")
        p_unit = L.paired_diff(pe, "unit")
        def pboot(w):
            return L.cluster_boot(
                {e: e for e in pe},
                lambda ss: L.paired_diff(
                    {f"{i}": pe[e] for i, e in enumerate(ss)}, w))
        plo, phi = pboot("fe")
        alo, ahi = pboot("att")
        nlo, nhi = pboot("unit")
        tlo, thi, se, G = L.t_interval(pe, "fe")
        if perm:
            _, p1, p2 = L.perm_test(pe, "fe", n=3000)
            ptxt = f"{p1:.3f} / {p2:.3f}"
        else:
            ptxt = "-"
        say(f"| {grp} - calm | {up:+.3f} [{ulo:+.3f}, {uhi:+.3f}] "
            f"| {p_fe:+.3f} [{plo:+.3f}, {phi:+.3f}] "
            f"| [{tlo:+.3f}, {thi:+.3f}] "
            f"| {p_att:+.3f} [{alo:+.3f}, {ahi:+.3f}] "
            f"| {p_unit:+.3f} [{nlo:+.3f}, {nhi:+.3f}] "
            f"| {G} | {ptxt} |")


def within_event_slope(say, units, tag):
    """Dose-response immune to between-event composition: OLS of z2 on
    match-hour wind after demeaning BOTH within event (= event fixed
    effects). Outdoor games only."""
    rows = [(e, m["wind"], v) for e, vals, m in units if m["outdoor"]
            for v in vals]
    by = defaultdict(list)
    for e, w, v in rows:
        by[e].append((w, v))

    def sl(evs):
        num = den = 0.0
        for pts in evs:
            if len(pts) < 2:
                continue
            mw = sum(w for w, _ in pts) / len(pts)
            mv = sum(v for _, v in pts) / len(pts)
            for w, v in pts:
                num += (w - mw) * (v - mv)
                den += (w - mw) ** 2
        return num / den if den else float("nan")
    est = sl(list(by.values()))
    lo, hi = L.cluster_boot(by, sl)
    n_eff = sum(len(v) for v in by.values() if len(v) > 1)
    say(f"\n{tag}: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, "
        f"{n_eff} obs / {sum(1 for v in by.values() if len(v) > 1)} events): "
        f"{est*10:+.3f} per +10 mph [{lo*10:+.3f}, {hi*10:+.3f}]")


def main():
    out = []
    say = lambda s="": (print(s), out.append(s))
    arms = L.label_arms()
    splits_b = L.read_csv(L.ROOT / "data/decider_splits.csv")
    splits_c = L.read_csv(L.ROOT / "data/decider_serve_splits.csv")

    say("# B2b — paired + corrected-label re-run of the end-effect designs\n")
    say("Pre-specified signal: a real bad-end effect makes the PAIRED "
        "pre/post swing VARIANCE exceed its sampling/serve-clustering null, "
        "and that excess GROWS with wind. Concretely: (i) mean z2 in the "
        "outdoor windy bin above its simulated null, and (ii) the "
        "windy-minus-calm contrast positive with a CI excluding 0, in the "
        "PRE-SPECIFIED within-event paired form, surviving the corrected "
        "venue labels. A contrast that flips sign or loses its CI when the "
        "labels are fixed is not a signal.\n")

    say("## Pairing estimator (stated before results)\n")
    say("Treatment arm = games in wind bin g at event e; control arm = games "
        "in the OUTDOOR calm bin at the SAME event e. Only events "
        "contributing games to both arms enter. Per event\n")
    say("    d_e = mean z2(treated games at e) - mean z2(calm games at e)\n")
    say("and the reported contrast is a weighted mean of d_e. Three "
        "weightings are reported because the weighting is a real choice:\n")
    say("* **FE** w_e = n_t n_c/(n_t+n_c) - the inverse-variance weight under "
        "homoskedastic per-game z2; algebraically the OLS coefficient on the "
        "wind dummy in a regression with event fixed effects. **Pre-specified "
        "primary**: it is the efficient estimator, and it is fixed by the "
        "design, not by the data.\n")
    say("* **ATT** w_e = n_t - 'effect for the average windy game observed' "
        "(the weighting used in the phase-1 audit).\n")
    say("* **unit** w_e = 1 - 'average event-level effect'.\n")
    say("Why the weighting cannot manufacture the result: all three weight "
        "vectors are functions of arm SIZES only, fixed before any z2 is "
        "looked at, and none can change the sign of a common effect - they "
        "only trade efficiency against which events dominate. The honest "
        "check is that they are reported side by side; a result that lives in "
        "only one weighting is a composition artifact, not an effect. "
        "Inference: (a) cluster bootstrap resampling EVENTS from the paired "
        "set, 2000 draws; (b) a t interval on the G event-level d_e with G-1 "
        "df (small-G honest); (c) a within-event permutation test that "
        "reshuffles the wind label across each event's games holding n_t, n_c "
        "fixed - exact under the sharp null.\n")
    say("Residual confounding the pairing does NOT remove: within an event, "
        "windy games are a non-random subset of DAYS and HOURS (and court "
        "assignments). The paired contrast is 'windy vs calm at the same "
        "tournament', not 'same match under two winds'.\n")

    for arm in ("published", "corrected_all", "corrected_hi", "audited_hi"):
        matches = L.load_matches(arms[arm])
        ub = design_b_units(matches, splits_b)
        uc = design_c_units(matches, splits_c)
        say(f"\n---\n\n## Label arm: `{arm}`  "
            f"(Design B n={len(ub)} games, Design C n="
            f"{sum(len(v) for _, v, _ in uc)} team-halves)\n")
        by = defaultdict(int)
        for _, _, m in ub:
            by[m["grp"]] += 1
        say("Design B group sizes: " +
            ", ".join(f"{g}={by[g]}" for g in ORDER if by[g]) + "\n")
        sb = sim_null_b(ub)
        group_table(say, ub, sb, "Design B (point share)")
        contrast_table(say, ub, "Design B (point share)")
        within_event_slope(say, ub, "Design B")
        sc = sim_null_c(uc)
        group_table(say, uc, sc, "Design C (serve-rally rate)")
        contrast_table(say, uc, "Design C (serve-rally rate)")
        within_event_slope(say, uc, "Design C")

    (HERE / "b2b_end_effects.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/weather_review/b2b_end_effects.md")


if __name__ == "__main__":
    main()
