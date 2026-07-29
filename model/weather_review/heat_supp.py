"""TASK B1, part 3 — supplements: joint controls, the published binned
statistic under corrected labels, and power/MDE translations.

    python model/weather_review/heat_supp.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib.race import sigmoid, game_win_prob  # noqa: E402
from heat_test import (read_csv, ols, boot, demean, fmt, load,  # noqa: E402
                       build_games)
from heat_robust import add_h, symmetrize  # noqa: E402

NBOOT = 1200
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def main():
    setting_audit, setting_heur, hourly, start, v2 = load()
    rows, meta, per_match = build_games(setting_audit, hourly, start, v2)
    by = defaultdict(list)
    for r in rows:
        by[r["setting"]].append(r)

    say("# TASK B1 part 3 — joint controls, corrected-label binned stat, "
        "power\n")

    # ---- 1. event FE + hour control, symmetrized -----------------------
    say("## 1. Everything at once: event fixed effects + hour-of-day "
        "controls, odd (symmetrized) spec\n")
    say("| setting | games | d (skill x heat) [95% CI] |")
    say("|---|---|---|")
    for name in ("outdoor", "indoor"):
        rs = add_h(by[name])
        sym = []
        for r in symmetrize(rs):
            q = dict(r)
            q["hr"] = (r["hour"] - 14.0) / 6.0
            q["shr"] = q["skill"] * q["hr"]
            q["shr2"] = q["skill"] * q["hr"] ** 2
            q["cell"] = r["ev"]
            sym.append(q)
        dm = demean(sym, ["y", "skill", "sh", "shr", "shr2"])
        b = ols(dm, "y", ["skill", "sh", "shr", "shr2"])
        cis, _ = boot(dm, "y", ["skill", "sh", "shr", "shr2"], n=NBOOT)
        say(f"| {name} | {len(dm)//2} | {fmt(b[2], cis[2], 3)} |")
    say("")

    # ---- 2. published binned statistic under audited labels ------------
    say("## 2. The PUBLISHED statistic (favourite obs-pred edge by temp "
        "bin, match hour) recomputed with AUDITED venue labels\n")
    say("Published outdoor (heuristic labels): -0.039 / -0.041 / -0.044 / "
        "-0.049 across <70 / 70-82 / 82-92 / 92+ F. The overall -4 pp is a "
        "level miscalibration of the race DP, not weather; only the DRIFT "
        "across bins is a heat test.\n")
    bins = [("<70F", -99, 70), ("70-82F", 70, 82), ("82-92F", 82, 92),
            ("92F+", 92, 999)]
    say("| setting | bin | games | pred fav % | obs fav % | edge (obs-pred) "
        "[95% CI] | edge minus <70F bin [95% CI] |")
    say("|---|---|---|---|---|---|---|")
    rng = random.Random(5)
    for name in ("outdoor", "indoor"):
        rs = []
        for r in by[name]:
            skill = r["skill"]
            if abs(skill) < 1e-9:
                continue
            eta = math.log((skill + 0.5) / (0.5 - skill))
            p = game_win_prob(abs(eta))
            rs.append({"ev": r["ev"], "T": r["T"], "p": p,
                       "won": 1.0 if (r["y"] > 0) == (skill > 0) else 0.0})
        clustered = defaultdict(list)
        for r in rs:
            clustered[r["ev"]].append(r)
        keys = list(clustered)
        # bootstrap all bins jointly so the difference has a CI
        draws = defaultdict(list)
        for _ in range(400):
            s = []
            for _ in keys:
                s.extend(clustered[rng.choice(keys)])
            edges = {}
            for lab, lo, hi in bins:
                sub = [r for r in s if lo <= r["T"] < hi]
                if len(sub) < 30:
                    edges[lab] = None
                    continue
                edges[lab] = (sum(r["won"] for r in sub) / len(sub)
                              - sum(r["p"] for r in sub) / len(sub))
            for lab, _, _ in bins:
                if edges[lab] is not None and edges["<70F"] is not None:
                    draws[lab].append((edges[lab],
                                       edges[lab] - edges["<70F"]))
        for lab, lo, hi in bins:
            sub = [r for r in rs if lo <= r["T"] < hi]
            if len(sub) < 30:
                continue
            pred = sum(r["p"] for r in sub) / len(sub)
            obs = sum(r["won"] for r in sub) / len(sub)
            e = sorted(d[0] for d in draws[lab])
            df = sorted(d[1] for d in draws[lab])
            elo, ehi = e[int(.025 * len(e))], e[int(.975 * len(e))]
            dmean = sum(df) / len(df)
            dlo, dhi = df[int(.025 * len(df))], df[int(.975 * len(df))]
            say(f"| {name} | {lab} | {len(sub)} | {pred:.3f} | {obs:.3f} | "
                f"{obs-pred:+.3f} [{elo:+.3f}, {ehi:+.3f}] | "
                f"{fmt(dmean, (dlo, dhi), 3)} |")
    say("")

    # ---- 3. attenuation bound ------------------------------------------
    say("## 3. Attenuation bound from the actual-vs-planned start split\n")
    say("Planned-start rows carry extra hour error, so their slope should be "
        "smaller; the ratio bounds how much the pooled estimate is "
        "attenuated.\n")
    say("| arm | games | d (skill x heat) [95% CI] |")
    say("|---|---|---|")
    rs = add_h(by["outdoor"])
    for lab, f in (("actual start time", lambda r: r["actual"]),
                   ("planned start time", lambda r: not r["actual"])):
        sub = symmetrize([r for r in rs if f(r)])
        if len(sub) < 600:
            say(f"| {lab} | {len(sub)//2} | (too few) |")
            continue
        b = ols(sub, "y", ["skill", "sh"])
        cis, _ = boot(sub, "y", ["skill", "sh"], n=NBOOT)
        say(f"| {lab} | {len(sub)//2} | {fmt(b[2], cis[2], 3)} |")
    say("")

    # ---- 4. duration MDE -------------------------------------------------
    say("## 4. What the duration nulls can still hide (H3 power)\n")
    say("From heat_test.py section 4 (within event x format cells, outdoor):"
        "\n")
    say("| outcome | mean | CI per +10F | largest effect still allowed at "
        "+20F |")
    say("|---|---|---|---|")
    tbl = [("rallies per match", 79.35, -0.426, 1.388),
           ("points per match", 35.64, -0.290, 0.347),
           ("rallies per point", 2.21, -0.001, 0.025),
           ("games per match", 1.99, -0.017, 0.004),
           ("3-game rate (bo3)", 0.283, -0.0181, 0.0088)]
    for lab, mean, lo, hi in tbl:
        say(f"| {lab} | {mean:.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{2*lo:+.3f} to {2*hi:+.3f} "
            f"({200*lo/mean:+.1f}% to {200*hi/mean:+.1f}%) |")
    say("")

    # ---- 5. real-world translation of the PRIMARY (odd-spec) estimate ---
    say("## 5. Real-world translation of the primary estimate\n")
    say("Primary spec = antisymmetric (odd) outdoor fit: "
        "share = b*skill + d*skill*h. Below, a favourite is described by "
        "its 75F game win probability; the +20F column applies d*skill*2 "
        "to the expected point share and re-runs the race DP.\n")
    rs = symmetrize(add_h(by["outdoor"]))
    b = ols(rs, "y", ["skill", "sh"])
    cis, nev = boot(rs, "y", ["skill", "sh"], n=NBOOT)
    d, dlo, dhi = b[2], cis[2][0], cis[2][1]
    say(f"Outdoor d = {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}] "
        f"(n = {len(rs)//2} games, {nev} events); b = {b[1]:.3f}.\n")
    say("| favourite at 75F | share at 75F | win prob at 95F (point est) "
        "| (CI low = most leveling allowed) | (CI high) |")
    say("|---|---|---|---|---|")
    for target in (0.60, 0.75, 0.90, 0.97):
        lo_e, hi_e = 0.0, 4.0
        for _ in range(60):
            mid = (lo_e + hi_e) / 2
            if game_win_prob(mid) < target:
                lo_e = mid
            else:
                hi_e = mid
        eta = (lo_e + hi_e) / 2
        skill = sigmoid(eta) - 0.5
        cells = []
        for dd in (d, dlo, dhi):
            s_new = min(0.98, max(0.02, sigmoid(eta) + dd * skill * 2.0))
            p_new = game_win_prob(math.log(s_new / (1 - s_new)))
            cells.append(f"{p_new:.3f} ({100*(p_new-target):+.2f} pp)")
        say(f"| {target:.0%} | {sigmoid(eta):.3f} | " + " | ".join(cells)
            + " |")
    say("")
    say("---\n*model/weather_review/heat_supp.py*")
    (ROOT / "model/weather_review/heat_supp.md").write_text(
        "\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/heat_supp.md")


if __name__ == "__main__":
    main()
