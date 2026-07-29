"""B2b parts 3 + 4 — decider SELECTION and STRICT correction handling.

    python model/weather_review/b2b_selection_strict.py <rebuilt_splits.csv>

Part 3 (selection / collider): the committed Design B conditions PPA on
DECIDERS. Deciders are the only PPA games that actually switch ends at 6,
but they are also a collider (a close match already happened). Every PPA
game has a recoverable "score first reaches 6" boundary, so the identical
statistic can be computed on NON-deciders, where NO end switch occurs.
That is a placebo arm with the same mechanics and the same selection-free
weather exposure:
  * decider-minus-non-decider mean z2 = the size of the selection effect;
  * difference-in-differences  [windy-calm | switch] - [windy-calm | no
    switch]  = the wind effect that is actually attributable to the end
    change rather than to wind-correlated variance in general.

Part 4 (integrity): rebuild every split from pb_rally with explicit
correction handling and compare against data/decider_splits.csv; then
re-run the paired windy contrast on the STRICT subset.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2b_lib as L  # noqa: E402

REF = "OUTDOOR calm <8"
ORDER = ["INDOOR", "OUTDOOR calm <8", "OUTDOOR moderate 8-14", "OUTDOOR windy 14+"]


def ci(pe, w="fe", seed=7):
    return L.cluster_boot({e: e for e in pe},
                          lambda ss: L.paired_diff(
                              {f"{i}": pe[e] for i, e in enumerate(ss)}, w),
                          seed=seed)


def paired_block(say, units, tag):
    """units: [(event, [vals], meta)] with meta['grp']."""
    ref = [(e, v) for e, v, m in units if m["grp"] == REF]
    say(f"\n| contrast ({tag}) | n_t | unpaired | paired-FE [95% CI] "
        "| paired-ATT [95% CI] | events | perm p (1s) |")
    say("|---|---|---|---|---|---|---|")
    for grp in ORDER:
        if grp == REF:
            continue
        trt = [(e, v) for e, v, m in units if m["grp"] == grp]
        if not trt or not ref:
            continue
        up = L.unpaired_diff(trt, ref)
        pe = L.paired_events(trt, ref)
        if not pe:
            continue
        fe, att = L.paired_diff(pe, "fe"), L.paired_diff(pe, "att")
        flo, fhi = ci(pe, "fe")
        alo, ahi = ci(pe, "att")
        _, p1, _ = L.perm_test(pe, "fe", n=2000)
        nt = sum(len(v) for _, v in trt)
        say(f"| {grp} - calm | {nt} | {up:+.3f} | {fe:+.3f} [{flo:+.3f}, "
            f"{fhi:+.3f}] | {att:+.3f} [{alo:+.3f}, {ahi:+.3f}] "
            f"| {len(pe)} | {p1:.3f} |")


def main():
    reb_path = sys.argv[1]
    reb = {(r["match_id"], int(r["game_number"])): r
           for r in L.read_csv(reb_path)}
    out = []
    say = lambda s="": (print(s), out.append(s))

    say("# B2b parts 3-4 — decider selection + strict correction handling\n")

    # ---------------------------------------------------------------- games
    gmeta = {}
    for g in L.read_csv(L.ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        gmeta[(g["match_id"], int(g["game_number"]))] = g

    # ------------------------------------------- part 4a: integrity audit
    com = {(r["match_id"], int(r["game_number"])): r
           for r in L.read_csv(L.ROOT / "data/decider_splits.csv")}
    say("## Part 4a — integrity of data/decider_splits.csv vs a fresh "
        "pb_rally rebuild\n")
    n_com = len(com)
    agree = miss = diff_counts = 0
    bad_seq = bad_bound = bad_score = 0
    for k, r in com.items():
        rr = reb.get(k)
        if rr is None:
            miss += 1
            continue
        same = all(int(r[c]) == int(rr[c])
                   for c in ("pa_pre", "pb_pre", "pa_post", "pb_post"))
        # side labels may be mirrored; accept the mirror too
        mirror = (int(r["pa_pre"]) == int(rr["pb_pre"]) and
                  int(r["pb_pre"]) == int(rr["pa_pre"]) and
                  int(r["pa_post"]) == int(rr["pb_post"]) and
                  int(r["pb_post"]) == int(rr["pa_post"]))
        if same or mirror:
            agree += 1
        else:
            diff_counts += 1
        if rr["seq_ok"] == "0":
            bad_seq += 1
        if rr["boundary_ok"] == "0":
            bad_bound += 1
        g = gmeta.get(k)
        if g:
            got = sorted([int(rr["fa"]), int(rr["fb"])])
            want = sorted([int(g["t1_score"]), int(g["t2_score"])])
            if got != want:
                bad_score += 1
    say(f"- committed rows: {n_com}; reproduced exactly (or mirrored): "
        f"{agree} ({agree/n_com:.1%}); differ: {diff_counts}; not in rebuild: "
        f"{miss}")
    say(f"- of the committed rows, the fresh rebuild flags "
        f"{bad_seq} ({bad_seq/n_com:.1%}) with a score-sequence "
        f"correction/rewind, {bad_bound} ({bad_bound/n_com:.1%}) with the "
        f"switch boundary NOT at exactly 6, and {bad_score} "
        f"({bad_score/n_com:.1%}) whose derived final score disagrees with "
        f"games.csv\n")

    # ------------------------------------------------- build unit sets
    matches_pub = L.load_matches(L.label_arms()["published"])
    matches_cor = L.load_matches(L.label_arms()["corrected_all"])

    def build(matches, source, strict, scope):
        """scope: 'switch' (MLP g1 + PPA deciders), 'noswitch' (PPA
        non-deciders), 'all'."""
        ub, uc = [], []
        for k, r in source.items():
            mid, gn = k
            m = matches.get(mid)
            g = gmeta.get(k)
            if not m or not g or g["scoring_format"] != "sideout_11":
                continue
            if m["tour"] == "MLP":
                is_switch = (gn == 1)
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
                if r.get("seq_ok") != "1" or r.get("boundary_ok") != "1":
                    continue
                got = sorted([int(r["fa"]), int(r["fb"])])
                want = sorted([int(g["t1_score"]), int(g["t2_score"])])
                if got != want:
                    continue
            grp = L.group_of(m)
            if not grp:
                continue
            meta = {"grp": grp, "wind": m["wind"], "tour": m["tour"],
                    "switch": is_switch}
            pre = int(r["pa_pre"]) + int(r["pb_pre"])
            post = int(r["pa_post"]) + int(r["pb_post"])
            if pre >= 5 and post >= 5:
                _, noise, z2 = L.zsq(int(r["pa_pre"]), pre,
                                     int(r["pa_post"]), post)
                ub.append((m["event"], [z2], meta))
            if "ra_pre" in r:
                vals = []
                for s in ("a", "b"):
                    rp, wp = int(r[f"r{s}_pre"]), int(r[f"w{s}_pre"])
                    rq, wq = int(r[f"r{s}_post"]), int(r[f"w{s}_post"])
                    if rp < 5 or rq < 5:
                        continue
                    _, noise, z2 = L.zsq(wp, rp, wq, rq)
                    if noise > 0:
                        vals.append(z2)
                if vals:
                    uc.append((m["event"], vals, meta))
        return ub, uc

    # ------------------------------- part 3: switch vs no-switch placebo
    say("## Part 3 — decider conditioning: switch games vs the no-switch "
        "placebo\n")
    say("Statistic identical in both arms (point share before vs after the "
        "score first reaches 6). In SWITCH games the ends actually change "
        "there; in NO-SWITCH games (PPA games 1-2 of a best-of-3) nothing "
        "changes, so any wind-driven excess there cannot be an end effect.\n")
    for lbl, matches in (("published labels", matches_pub),
                         ("corrected labels", matches_cor)):
        sw, _ = build(matches, reb, False, "switch")
        nsw, _ = build(matches, reb, False, "noswitch")
        say(f"\n### {lbl}\n")
        say("| arm | n games | mean z2 | by group: " +
            " / ".join(g.replace("OUTDOOR ", "") for g in ORDER) + " |")
        say("|---|---|---|---|")
        for tag, U in (("SWITCH (MLP all + PPA deciders)", sw),
                       ("NO-SWITCH placebo (PPA non-deciders)", nsw)):
            vals = [v for _, vv, _ in U for v in vv]
            per = []
            for g in ORDER:
                s = [v for _, vv, m in U if m["grp"] == g for v in vv]
                per.append(f"{sum(s)/len(s):.3f} (n={len(s)})" if s else "-")
            say(f"| {tag} | {len(U)} | {sum(vals)/len(vals):.3f} | "
                + " / ".join(per) + " |")
        # PPA-only decider vs non-decider (removes tour composition)
        ppa_d = [v for _, vv, m in sw if m["tour"] == "PPA" for v in vv]
        ppa_n = [v for _, vv, m in nsw for v in vv]
        say(f"\nPPA only, decider minus non-decider mean z2: "
            f"{sum(ppa_d)/len(ppa_d) - sum(ppa_n)/len(ppa_n):+.3f} "
            f"({len(ppa_d)} vs {len(ppa_n)} games) — this is the SELECTION "
            f"(collider) magnitude, not an end effect.")
        say("\n**Wind contrasts inside each arm** (same estimators as part 1):")
        paired_block(say, sw, "SWITCH")
        paired_block(say, nsw, "NO-SWITCH placebo")
        # difference in differences, paired within event
        say("\n**Difference-in-differences** (windy-calm in switch games "
            "minus windy-calm in no-switch games), event-paired both sides:")
        for grp in ORDER:
            if grp == REF:
                continue
            def arm(U):
                t = [(e, v) for e, v, m in U if m["grp"] == grp]
                c = [(e, v) for e, v, m in U if m["grp"] == REF]
                if not t or not c:
                    return None
                pe = L.paired_events(t, c)
                return pe if pe else None
            a1, a2 = arm(sw), arm(nsw)
            if not a1 or not a2:
                continue
            evs = sorted(set(a1) | set(a2))
            def did(keys):
                p1 = {e: a1[e] for e in keys if e in a1}
                p2 = {e: a2[e] for e in keys if e in a2}
                if not p1 or not p2:
                    return float("nan")
                return L.paired_diff(p1, "fe") - L.paired_diff(p2, "fe")
            est = did(evs)
            lo, hi = L.cluster_boot({e: e for e in evs},
                                    lambda ss: did(list(ss)))
            say(f"- DiD {grp} vs calm: {est:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                f"({len(evs)} events)")

    # ------------------------------------------- part 4b: strict re-run
    say("\n## Part 4b — the paired windy contrast under STRICT correction "
        "handling\n")
    say("STRICT = fresh pb_rally rebuild AND no score-sequence "
        "correction/rewind AND switch boundary exactly at 6 AND derived "
        "final score equals games.csv.\n")
    for lbl, matches in (("published labels", matches_pub),
                         ("corrected labels", matches_cor)):
        say(f"\n### {lbl}\n")
        for strict in (False, True):
            ub, uc = build(matches, reb, strict, "switch")
            say(f"\n**{'STRICT' if strict else 'rebuilt, all rows'}** "
                f"(Design B n={len(ub)}, Design C n="
                f"{sum(len(v) for _, v, _ in uc)})")
            paired_block(say, ub, "Design B")
            paired_block(say, uc, "Design C")

    (HERE / "b2b_selection_strict.md").write_text("\n".join(out) + "\n")
    print("\nwrote model/weather_review/b2b_selection_strict.md")


if __name__ == "__main__":
    main()
