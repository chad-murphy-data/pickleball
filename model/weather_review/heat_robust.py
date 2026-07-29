"""TASK B1, part 2 — try to BREAK the heat result found by heat_test.py.

    python model/weather_review/heat_robust.py

heat_test.py found the OPPOSITE of the pre-specified physiology: outdoors the
skill x heat interaction is POSITIVE (favorites convert skill slightly BETTER
in heat), d = +0.031 [+0.007, +0.060] per +10 F, while the indoor arm is
-0.015 [-0.047, +0.006]. A wrong-signed "signal" earns more scepticism than a
right-signed one, so this script attacks it:

  A. Antisymmetry. share and skill both flip when the team labels flip, so the
     truth must be an ODD function. If mean(skill) != 0 the intercept and heat
     main effect are entangled with the interaction. Refit on the symmetrized
     (doubled, sign-flipped) sample with no intercept and no heat main effect.
  B. Functional form. share ~ skill is concave (bounded). If high-|skill|
     mismatches cluster in cool morning rounds, a LINEAR fit manufactures a
     positive skill x heat term out of nothing. Add skill^3 (+ its heat
     interaction), restrict to a narrow skill band, and estimate the slope
     separately inside |skill| terciles.
  C. Non-parametric dose-response: fit b (skill slope) inside temperature
     bins; a real effect should be monotone.
  D. Round/stage composition: within (event x stage) fixed effects.
  E. Venue labels: audited-vs-heuristic on the SAME events; high-confidence
     only; mixed and unknown arms.
  F. Splits: tour, year, actual-vs-planned start times.
  G. The real-world quantity: does the FAVORITE WIN more games in heat?
     Logistic P(favorite wins game) ~ |skill| + h + |skill|*h.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib.race import sigmoid, team_eta, game_win_prob  # noqa: E402
from heat_test import (read_csv, ols, boot, demean, fmt, load,  # noqa: E402
                       build_games, _suff, _solve)

NBOOT = 1200
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def add_h(rows, ref=75.0, key="T"):
    out = []
    for r in rows:
        q = dict(r)
        q["h"] = (r[key] - ref) / 10.0
        q["sh"] = q["skill"] * q["h"]
        q["s3"] = q["skill"] ** 3
        q["s3h"] = q["s3"] * q["h"]
        out.append(q)
    return out


def symmetrize(rows):
    """Duplicate every game with team labels flipped: y -> -y, skill -> -skill.
    Makes the design exactly odd, so intercept/heat-main-effect vanish and the
    interaction cannot borrow from them."""
    out = []
    for r in rows:
        out.append(r)
        q = dict(r)
        q["y"] = -r["y"]
        q["skill"] = -r["skill"]
        q["sh"] = -r["sh"]
        q["s3"] = -r["s3"]
        q["s3h"] = -r["s3h"]
        out.append(q)
    return out


def logit_fit(rows, xkeys, init=None, maxiter=30):
    """Binomial logit on 0/1 rows (weights via 'n'/'wins' if present)."""
    p = len(xkeys) + 1
    beta = list(init) if init else [0.0] * p
    for _ in range(maxiter):
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        for r in rows:
            x = [1.0] + [r[k] for k in xkeys]
            z = sum(b * xx for b, xx in zip(beta, x))
            pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            res = r["y"] - pr
            w = pr * (1 - pr)
            for i in range(p):
                grad[i] += res * x[i]
                for j in range(p):
                    hess[i][j] += w * x[i] * x[j]
        step = _solve(hess, grad, p)
        if step is None:
            return None
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return beta


def main():
    setting_audit, setting_heur, hourly, start, v2 = load()
    rows, meta, per_match = build_games(setting_audit, hourly, start, v2)
    ov = {r["event_id"]: r for r in read_csv("data/venue_overrides.csv")}
    games_meta = {}
    for g in read_csv("data/games.csv"):
        games_meta[g["game_id"]] = g
    stage_of = {}
    for g in read_csv("data/games.csv"):
        stage_of[g["match_id"]] = (g["stage"] or "?").split("—")[0].strip()

    by = defaultdict(list)
    for r in rows:
        by[r["setting"]].append(r)
    out_rows = add_h(by["outdoor"])
    ind_rows = add_h(by["indoor"])

    say("# TASK B1 part 2 — attacking the positive heat interaction\n")
    ms = sum(r["skill"] for r in out_rows) / len(out_rows)
    say(f"Mean skill (team-1 minus team-2 expected share) outdoors: {ms:+.4f} "
        f"— {'NOT ' if abs(ms) > 0.002 else ''}centred, so the intercept and "
        "heat main effect can entangle with the interaction (test A).\n")

    # ---- A. antisymmetric specification --------------------------------
    say("## A. Antisymmetric (symmetrized) specification\n")
    say("Doubling with flipped labels forces the fit to be odd in skill: "
        "y = b*skill + d*skill*h (intercept and heat main effect are exactly "
        "0 by construction).\n")
    say("| setting | spec | b (skill) | d (skill x heat) [95% CI] |")
    say("|---|---|---|---|")
    for name, rs in (("outdoor", out_rows), ("indoor", ind_rows)):
        for spec, xk in (("with intercept + heat main (published form)",
                          ["skill", "h", "sh"]),
                         ("symmetrized, odd only", ["skill", "sh"])):
            src = rs if "published" in spec else symmetrize(rs)
            b = ols(src, "y", xk)
            cis, _ = boot(src, "y", xk, n=NBOOT)
            i = xk.index("sh") + 1
            say(f"| {name} | {spec} | {b[1]:.3f} | {fmt(b[i], cis[i], 3)} |")
    say("")

    # ---- B. functional form --------------------------------------------
    say("## B. Functional form — is the interaction just concavity x "
        "composition?\n")
    say("share ~ skill is concave near the bounds. If cool hours hold the "
        "big mismatches, a linear model reads that as skill x heat. "
        "skill^3 (odd, so antisymmetry survives) absorbs the curvature.\n")
    say("| setting | spec | games | d (skill x heat) [95% CI] |")
    say("|---|---|---|---|")
    for name, rs in (("outdoor", out_rows), ("indoor", ind_rows)):
        sym = symmetrize(rs)
        for spec, xk in (("odd linear", ["skill", "sh"]),
                         ("odd + skill^3", ["skill", "sh", "s3"]),
                         ("odd + skill^3 + skill^3 x heat",
                          ["skill", "sh", "s3", "s3h"])):
            b = ols(sym, "y", xk)
            cis, _ = boot(sym, "y", xk, n=NBOOT)
            say(f"| {name} | {spec} | {len(rs)} | {fmt(b[2], cis[2], 3)} |")
    say("")

    say("### B2. Inside |skill| terciles (each band is nearly linear)\n")
    say("| setting | band | games | mean|skill| | b | d (skill x heat) "
        "[95% CI] |")
    say("|---|---|---|---|---|---|")
    for name, rs in (("outdoor", out_rows), ("indoor", ind_rows)):
        cuts = sorted(abs(r["skill"]) for r in rs)
        q1, q2 = cuts[len(cuts) // 3], cuts[2 * len(cuts) // 3]
        bands = [("low |skill| (<%.3f)" % q1, lambda a: a < q1),
                 ("mid", lambda a: q1 <= a < q2),
                 ("high |skill| (>=%.3f)" % q2, lambda a: a >= q2)]
        for lab, f in bands:
            sub = symmetrize([r for r in rs if f(abs(r["skill"]))])
            b = ols(sub, "y", ["skill", "sh"])
            cis, _ = boot(sub, "y", ["skill", "sh"], n=NBOOT)
            ma = sum(abs(r["skill"]) for r in sub) / len(sub)
            say(f"| {name} | {lab} | {len(sub)//2} | {ma:.3f} | {b[1]:.3f} "
                f"| {fmt(b[2], cis[2], 3)} |")
    say("")

    # ---- C. dose-response ----------------------------------------------
    say("## C. Non-parametric dose-response: skill slope b inside "
        "temperature bins\n")
    say("A real effect should climb monotonically; a composition artefact "
        "need not.\n")
    say("| setting | temp bin | games | b (skill slope) [95% CI] |")
    say("|---|---|---|---|")
    bins = [("<60F", -99, 60), ("60-70F", 60, 70), ("70-80F", 70, 80),
            ("80-90F", 80, 90), ("90F+", 90, 999)]
    for name, rs in (("outdoor", out_rows), ("indoor", ind_rows)):
        for lab, lo, hi in bins:
            sub = symmetrize([r for r in rs if lo <= r["T"] < hi])
            if len(sub) < 400:
                say(f"| {name} | {lab} | {len(sub)//2} | (too few) |")
                continue
            b = ols(sub, "y", ["skill"])
            cis, _ = boot(sub, "y", ["skill"], n=NBOOT)
            say(f"| {name} | {lab} | {len(sub)//2} | "
                f"{b[1]:.3f} [{cis[1][0]:.3f}, {cis[1][1]:.3f}] |")
    say("")

    # ---- D. stage composition ------------------------------------------
    say("## D. Round/stage composition: within (event x stage) fixed "
        "effects\n")
    say("| setting | games | d (skill x heat) [95% CI] |")
    say("|---|---|---|")
    for name, rs in (("outdoor", out_rows), ("indoor", ind_rows)):
        sym = []
        for r in symmetrize(rs):
            q = dict(r)
            q["cell"] = r["ev"] + "|" + stage_of.get(r["match_id"], "?")
            sym.append(q)
        dm = demean(sym, ["y", "skill", "sh"])
        b = ols(dm, "y", ["skill", "sh"])
        cis, _ = boot(dm, "y", ["skill", "sh"], n=NBOOT)
        say(f"| {name} | {len(dm)//2} | {fmt(b[2], cis[2], 3)} |")
    say("")

    # ---- E. venue labels ------------------------------------------------
    say("## E. Venue labels\n")
    audited = set(ov)
    say("Same events, audited vs heuristic label (isolates relabelling from "
        "sample change); then confidence and the mixed/unknown arms.\n")
    say("| arm | games | events | d (skill x heat) [95% CI] |")
    say("|---|---|---|---|")
    rows_h, _, _ = build_games(setting_heur, hourly, start, v2)
    arms = []
    arms.append(("audited events, AUDITED label = outdoor",
                 [r for r in rows if r["setting"] == "outdoor"
                  and r["ev"] in audited]))
    arms.append(("audited events, HEURISTIC label = outdoor",
                 [r for r in rows_h if r["setting"] == "outdoor"
                  and r["ev"] in audited]))
    arms.append(("audited outdoor, HIGH confidence only",
                 [r for r in rows if r["setting"] == "outdoor"
                  and ov.get(r["ev"], {}).get("confidence") == "high"]))
    arms.append(("unaudited events (heuristic outdoor)",
                 [r for r in rows if r["setting"] == "outdoor"
                  and r["ev"] not in audited]))
    arms.append(("mixed-venue events",
                 [r for r in rows if r["setting"] == "mixed"]))
    arms.append(("unknown-venue events",
                 [r for r in rows if r["setting"] == "unknown"]))
    for lab, rs in arms:
        if len(rs) < 300:
            say(f"| {lab} | {len(rs)} | — | (too few) |")
            continue
        sym = symmetrize(add_h(rs))
        b = ols(sym, "y", ["skill", "sh"])
        cis, nev = boot(sym, "y", ["skill", "sh"], n=NBOOT)
        say(f"| {lab} | {len(rs)} | {nev} | {fmt(b[2], cis[2], 3)} |")
    say("")

    # ---- F. splits -------------------------------------------------------
    say("## F. Splits (outdoor, audited labels, symmetrized odd spec)\n")
    say("| split | games | events | d (skill x heat) [95% CI] |")
    say("|---|---|---|---|")
    splits = [("PPA only", lambda r: r["tour"] == "PPA"),
              ("MLP only", lambda r: r["tour"] == "MLP"),
              ("ACTUAL start times only", lambda r: r["actual"]),
              ("PLANNED start times only", lambda r: not r["actual"])]
    yrs = sorted({games_meta[k]["date"][:4] for k in games_meta})
    for y in yrs:
        splits.append((f"season {y}",
                       lambda r, y=y: r["match_id"] in per_match
                       and per_match[r["match_id"]][0]["date"][:4] == y))
    for lab, f in splits:
        sub = [r for r in out_rows if f(r)]
        if len(sub) < 300:
            say(f"| {lab} | {len(sub)} | — | (too few) |")
            continue
        sym = symmetrize(sub)
        b = ols(sym, "y", ["skill", "sh"])
        cis, nev = boot(sym, "y", ["skill", "sh"], n=NBOOT)
        say(f"| {lab} | {len(sub)} | {nev} | {fmt(b[2], cis[2], 3)} |")
    say("")

    # ---- G. the real-world quantity -------------------------------------
    say("## G. Does the FAVOURITE actually WIN more in heat?\n")
    say("Logistic P(favourite wins the game) = a + b*|skill| + c*h + "
        "d*|skill|*h. This is the quantity a reader cares about; d > 0 = "
        "fewer upsets in heat. Cluster bootstrap over events, "
        f"{NBOOT // 4} draws.\n")
    say("| setting | games | favourite win rate | d (|skill| x heat) "
        "[95% CI] | upset shift, 75F -> 95F, median favourite |")
    say("|---|---|---|---|---|")
    rng = random.Random(11)
    for name, rs in (("outdoor", out_rows), ("indoor", ind_rows)):
        L = []
        for r in rs:
            a = abs(r["skill"])
            if a < 1e-6:
                continue
            L.append({"ev": r["ev"],
                      "y": 1.0 if (r["y"] > 0) == (r["skill"] > 0) else 0.0,
                      "a": a, "h": r["h"], "ah": a * r["h"]})
        L = [r for r in L if r["y"] in (0.0, 1.0)]
        beta = logit_fit(L, ["a", "h", "ah"])
        clustered = defaultdict(list)
        for r in L:
            clustered[r["ev"]].append(r)
        keys = list(clustered)
        draws = []
        for _ in range(NBOOT // 4):
            s = []
            for _ in keys:
                s.extend(clustered[rng.choice(keys)])
            bb = logit_fit(s, ["a", "h", "ah"], init=beta, maxiter=6)
            if bb:
                draws.append(bb[3])
        draws.sort()
        lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
        wr = sum(r["y"] for r in L) / len(L)
        amed = sorted(r["a"] for r in L)[len(L) // 2]
        p75 = sigmoid(beta[0] + beta[1] * amed + beta[2] * 0.0)
        p95 = sigmoid(beta[0] + beta[1] * amed + beta[2] * 2.0
                      + beta[3] * amed * 2.0)
        say(f"| {name} | {len(L)} | {wr:.3f} | {fmt(beta[3], (lo, hi), 3)} "
            f"| {p75:.3f} -> {p95:.3f} ({100*(p95-p75):+.2f} pp) |")
    say("")

    say("---\n*Deterministic; cluster bootstrap over events "
        f"({NBOOT} draws, seeded). model/weather_review/heat_robust.py*")
    (ROOT / "model/weather_review/heat_robust.md").write_text(
        "\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/heat_robust.md")


if __name__ == "__main__":
    main()
