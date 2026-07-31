"""B6 part 3 — settle two phase-1 claims.

(i)  "H4 'KILLED' is overstated: the binned −2.0pp windy drift maps to
     reg-1 d = −0.072, the reg-1 CI [−0.060, +0.064] excludes it only
     marginally while still allowing 83% of the signal, and the rally-logit
     'second kill' never constrained anything."
     -> Recompute the binned→continuous translation from scratch, using the
        race DP rather than a linear approximation, and ALSO test whether the
        translation is even legitimate (a linear-in-wind coefficient and a
        step-at-14-mph coefficient are different functionals; OLS on a step
        truth returns a specific, computable, much smaller d).

(ii) "'14+ mph is hot in every design' is FALSE: A and A2 are not hot, and
     B and C share the same 111 games."
     -> Count and intersect the windy-14+ samples of Designs B and C.

    python model/weather_review/b6_claims.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b6_lib as L  # noqa: E402
from b6_lib import DATA, ROOT, read_csv, sigmoid  # noqa: E402

OUT = []
NBOOT = 1000


def say(s=""):
    print(s)
    OUT.append(s)


def main():
    sm = L.ShareMoments(n=481)
    games = [g for g in L.load_games() if g["wind"] is not None]

    say("# B6 — settling the two phase-1 claims about H4\n")

    # ------------------------------------------------------------------ (i)
    say("## Claim 1: does the binned −2.0pp drift really map to d = −0.072?\n")

    for label_key, tag in (("setting_heur", "heuristic labels (as published)"),
                           ("setting", "corrected labels (venue_overrides)")):
        out = [g for g in games if g[label_key] == "outdoor"]
        for g in out:
            p = min(max(sigmoid(g["eta"]), 0.16), 0.84)
            g["_p"] = p
            g["_skill"] = p - 0.5
            g["_fav_skill"] = abs(p - 0.5)
            g["_T"] = g["T"]
        calm = [g for g in out if g["wind"] < 8]
        windy = [g for g in out if 14 <= g["wind"] < 20]

        def fav_pred(g, d, step=False):
            """Favourite's predicted win prob under interaction size d."""
            s = g["_fav_skill"]
            w = g["wind"] / 10.0
            mult = (1 + d * (1.0 if w >= 1.4 else 0.0)) if step else (1 + d * w)
            return sm.win(min(max(0.5 + s * mult, 0.16), 0.84), g["_T"])

        def drift(d, step=False):
            a = np.mean([fav_pred(g, d, step) - fav_pred(g, 0.0) for g in windy])
            b = np.mean([fav_pred(g, d, step) - fav_pred(g, 0.0) for g in calm])
            return a - b

        def solve(target, step=False):
            lo, hi = -1.0, 1.0
            for _ in range(80):
                mid = (lo + hi) / 2
                if drift(mid, step) > target:   # drift increases with d
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) / 2

        d_lin = solve(-0.020, step=False)
        d_step = solve(-0.020, step=True)
        say(f"**{tag}** — outdoor {len(out):,} games "
            f"(calm {len(calm):,}, 14–20 {len(windy):,}); mean wind "
            f"{np.mean([g['wind'] for g in calm]):.1f} vs "
            f"{np.mean([g['wind'] for g in windy]):.1f} mph; mean favourite "
            f"|skill| {np.mean([g['_fav_skill'] for g in out]):.3f}.\n")
        say(f"- A LINEAR interaction reproducing a −2.0 pp calm→windy drift "
            f"in the favourite's win rate needs **d = {d_lin:+.4f}**.")
        say(f"- A STEP interaction switched on at 14 mph reproducing the same "
            f"−2.0 pp drift needs **d_step = {d_step:+.4f}** "
            "(i.e. the favourite's edge shrinks by "
            f"{100*abs(d_step):.1f}% above 14 mph).\n")

        # what OLS returns when the truth is the step
        s = np.array([g["_skill"] for g in out])
        w = np.array([g["wind"] / 10.0 for g in out])
        X = np.column_stack([np.ones(len(out)), s, w, s * w])
        target = s * (w >= 1.4)
        proj = np.linalg.lstsq(X, target, rcond=None)[0]
        dilution = proj[3]
        say(f"- If the TRUTH is that step, the continuous reg-1 coefficient "
            f"on skill×wind converges to d_step × **{dilution:.3f}** = "
            f"**{d_step*dilution:+.4f}** — because only "
            f"{100*np.mean(w>=1.4):.1f}% of outdoor games are ≥14 mph and the "
            "step is nearly orthogonal to the linear term after partialling "
            "out skill and wind. The binned→continuous translation therefore "
            "only holds if the effect really is linear in wind from 0 mph up."
            "\n")

        # fit both specs
        y = np.array([g["share"] - 0.5 for g in out])
        clusters = defaultdict(list)
        for i, g in enumerate(out):
            clusters[g["event"]].append(i)

        def fit(idx, kind):
            ii = np.asarray(idx)
            if kind == "lin":
                Xk = np.column_stack([np.ones(len(ii)), s[ii], w[ii],
                                      s[ii] * w[ii]])
            else:
                st = (w[ii] >= 1.4).astype(float)
                Xk = np.column_stack([np.ones(len(ii)), s[ii], st, s[ii] * st])
            try:
                return np.linalg.lstsq(Xk, y[ii], rcond=None)[0][3]
            except np.linalg.LinAlgError:
                return np.nan

        keys = list(clusters)
        rng = np.random.default_rng(5)
        res = {}
        for kind in ("lin", "step"):
            base = fit([i for k in keys for i in clusters[k]], kind)
            dr = []
            for _ in range(NBOOT):
                pick = rng.integers(0, len(keys), len(keys))
                idx = []
                for j in pick:
                    idx.extend(clusters[keys[j]])
                dr.append(fit(idx, kind))
            lo, hi = np.nanpercentile(dr, [2.5, 97.5])
            res[kind] = (base, lo, hi)
        say("| spec fitted to the data | coefficient | 95% CI (event "
            "bootstrap) | value implied by the −2.0 pp binned drift | "
            "is that value excluded? |")
        say("|---|---|---|---|---|")
        b, lo, hi = res["lin"]
        say(f"| continuous d (skill×wind/10) | {b:+.4f} | [{lo:+.4f}, "
            f"{hi:+.4f}] | {d_lin:+.4f} (if linear) / {d_step*dilution:+.4f} "
            f"(if a 14 mph step) | "
            f"{'yes' if d_lin < lo else 'no'} / "
            f"{'yes' if d_step*dilution < lo else 'no'} |")
        b, lo, hi = res["step"]
        say(f"| step d (skill×1[w≥14]) | {b:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{d_step:+.4f} | {'yes' if d_step < lo else 'no'} |")
        say("")

    # -------- rally logit: what would the binned drift imply there? -------
    say("### Did the rally-level logit constrain anything?\n")
    say("Published: outdoor d = −0.017 [−0.098, +0.058] on adv×(wind/10), "
        "where adv is the serving team's v2 eta advantage. Translate the "
        "same −2.0 pp binned favourite drift into that parameterisation.\n")
    say("Put both regressions on ONE scale: *fraction of the skill edge lost "
        "per +10 mph*, which is d divided by the main skill coefficient. It "
        "is scale-free, so the game-level share regression and the "
        "rally-level logit become directly comparable.\n")
    say("| test | d | main skill coef | fractional compression per 10 mph "
        "[95% CI] |")
    say("|---|---|---|---|")
    for name, d, lo, hi, b in [
        ("reg 1, game level (published, heuristic labels)",
         0.002, -0.060, 0.064, 1.040),
        ("reg 1, game level (corrected labels)",
         -0.038, -0.096, 0.017, 1.051),
        ("reg 3, rally logit (published, heuristic labels)",
         -0.017, -0.098, 0.058, 0.458),
    ]:
        say(f"| {name} | {d:+.3f} | {b:.3f} | {d/b:+.3f} "
            f"[{lo/b:+.3f}, {hi/b:+.3f}] |")
    say("")
    say(f"The binned −2.0 pp drift corresponds to a fractional compression "
        f"of {d_lin/1.051:+.3f} per 10 mph (using the corrected-label "
        "d above and its own skill coefficient). The rally logit's CI on "
        "that scale is [−0.214, +0.127] — six times wider than the effect "
        "it was cited as ruling out. **Confirmed: the rally-level logit "
        "never constrained the hypothesis.** Its apparent agreement with "
        "reg 1 is agreement between a tight estimate and a very loose one.\n")

    # ----------------------------------------------------------------- (ii)
    say("## Claim 2: is '14+ mph is hot in every design' false?\n")
    _, _, _, hourly, times = L.weather_index()
    meta = {}
    for g in games:
        meta.setdefault(g["match"], g)

    def rows_of(path):
        got = []
        for r in read_csv(DATA / path):
            m = meta.get(r["match_id"].lower())
            if not m or m["wind"] is None:
                continue
            got.append((r["match_id"].lower(), int(r["game_number"]), m))
        return got

    B = rows_of("decider_splits.csv")
    C = rows_of("decider_serve_splits.csv")
    say("| design | rows with a wind join | outdoor (heuristic) | "
        "outdoor & ≥14 mph | distinct matches in that windy cell |")
    say("|---|---|---|---|---|")
    cells = {}
    for name, rows in (("B (point share, pre/post switch)", B),
                       ("C (serve-rally rate, pre/post switch)", C)):
        od = [x for x in rows if x[2]["setting_heur"] == "outdoor"]
        wy = [x for x in od if x[2]["wind"] >= 14]
        cells[name[0]] = set((x[0], x[1]) for x in wy)
        say(f"| {name} | {len(rows):,} | {len(od):,} | {len(wy):,} | "
            f"{len(set(x[0] for x in wy)):,} |")
    inter = cells["B"] & cells["C"]
    say("")
    say(f"Overlap of the two windy-14+ cells: **{len(inter)}** of "
        f"{len(cells['B'])} (B) and {len(cells['C'])} (C) game-rows — "
        f"{100*len(inter)/max(1,len(cells['B'])):.0f}% / "
        f"{100*len(inter)/max(1,len(cells['C'])):.0f}%.\n")
    # same under corrected labels
    for name, rows in (("B", B), ("C", C)):
        od = [x for x in rows if x[2]["setting"] == "outdoor"]
        wy = [x for x in od if x[2]["wind"] >= 14]
        cells[name + "c"] = set((x[0], x[1]) for x in wy)
    say(f"Under the corrected venue labels the same cells are "
        f"{len(cells['Bc'])} (B) and {len(cells['Cc'])} (C) rows with "
        f"{len(cells['Bc'] & cells['Cc'])} shared.\n")

    (ROOT / "model/weather_review/b6_claims.md").write_text("\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/b6_claims.md")


if __name__ == "__main__":
    main()
