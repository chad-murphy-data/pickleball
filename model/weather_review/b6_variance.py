"""B6 part 1 — the VARIANCE channel: does wind add noise at unchanged mean?

Every committed wind test asks whether wind moves the MEAN (point share,
serve rate, favourite edge).  Folklore ("wind causes upsets") is equally
consistent with an unchanged mean and a fatter spread: at the same skill
gap, windy games scatter further from their expectation, so favourites
convert less often even though their average score line is unchanged.

Pre-specified signal (written before looking):
  V1  mean z² (squared standardised share residual, exact race-DP moments)
      rises monotonically with outdoor match-hour wind, with the corrected-
      label INDOOR arm flat, in a within-event paired contrast.
  V2  the continuous z² slope on wind is > 0 outdoors and ~0 indoors.
  V3  at matched predicted win probability, favourites win LESS in wind.
  V4  the reliability slope beta of a logistic recalibration is < 1 in the
      windy outdoor bin and ~1 in calm (over-confidence = unmodelled noise).
Anything that shows up in the indoor arm as strongly counts as method noise,
not wind (house stance: indoor is a control, not a zero).

    python model/weather_review/b6_variance.py
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b6_lib as L  # noqa: E402
from b6_lib import ROOT, sigmoid  # noqa: E402

BINS = [(0, 8, "0–8"), (8, 14, "8–14"), (14, 20, "14–20"), (20, 99, "20+")]
NBOOT = 1000
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def build():
    sm = L.ShareMoments(n=481)
    games = L.load_games()
    rows = []
    for g in games:
        if g["wind"] is None or g["setting"] not in ("outdoor", "indoor"):
            continue
        p = sigmoid(g["eta"])
        p = min(max(p, 0.16), 0.84)
        mu, sd = sm.moments(p, g["T"])
        z = (g["share"] - mu) / sd
        rows.append(dict(ev=g["event"], setting=g["setting"], tour=g["tour"],
                         T=g["T"], wind=g["wind"], z=z, z2=z * z,
                         p=p, skill=p - 0.5, won=g["won"],
                         pwin=sm.win(p, g["T"]), match=g["match"],
                         src=g["wx_source"]))
    return rows, sm


def bin_of(w):
    for lo, hi, lab in BINS:
        if lo <= w < hi:
            return lab
    return None


def cboot(clusters, stat, n=NBOOT, seed=7):
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    base = stat([r for k in keys for r in clusters[k]])
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        s = []
        for i in pick:
            s.extend(clusters[keys[i]])
        v = stat(s)
        if v is not None and np.all(np.isfinite(v)):
            draws.append(v)
    d = np.array(draws, float)
    return base, np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


def main():
    rows, sm = build()
    say("# B6 — the variance channel: does wind add noise at unchanged mean?\n")
    say("*(model/weather_review/b6_variance.py; corrected venue labels from "
        "data/venue_overrides.csv; z = (observed point share − exact race-DP "
        "expected share)/race-DP sd, so the mechanical null is E[z²] = 1 and "
        "any excess is unmodelled dispersion — rating error plus serve "
        "clustering — which is common to all bins.)*\n")

    for setting in ("outdoor", "indoor"):
        sub = [r for r in rows if r["setting"] == setting]
        say(f"- {setting}: {len(sub):,} games, mean z² = "
            f"{np.mean([r['z2'] for r in sub]):.3f}")
    say("")

    # ---------------------------------------------------------- V1 binned --
    say("## V1. Mean z² by wind bin (unpaired and within-event paired)\n")
    say("| setting | bin | games | mean z² [95% CI] | Δ vs calm, UNPAIRED "
        "[95% CI] | Δ vs calm, PAIRED within event [95% CI] |")
    say("|---|---|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        sub = [r for r in rows if r["setting"] == setting]
        clusters = defaultdict(list)
        for r in sub:
            clusters[r["ev"]].append(r)
        for lo, hi, lab in BINS:
            inb = [r for r in sub if lo <= r["wind"] < hi]
            if len(inb) < 30:
                say(f"| {setting} | {lab} | {len(inb)} | — | — | — |")
                continue

            def stat(s, lo=lo, hi=hi):
                a = [r["z2"] for r in s if lo <= r["wind"] < hi]
                c = [r["z2"] for r in s if r["wind"] < 8]
                if len(a) < 5 or len(c) < 5:
                    return None
                # paired: only events supplying BOTH
                by = defaultdict(lambda: ([], []))
                for r in s:
                    if lo <= r["wind"] < hi:
                        by[r["ev"]][0].append(r["z2"])
                    if r["wind"] < 8:
                        by[r["ev"]][1].append(r["z2"])
                diffs, wts = [], []
                for ev, (aa, cc) in by.items():
                    if aa and cc:
                        diffs.append(np.mean(aa) - np.mean(cc))
                        wts.append(min(len(aa), len(cc)))
                pair = (np.average(diffs, weights=wts) if diffs else np.nan)
                return [np.mean(a), np.mean(a) - np.mean(c), pair]

            base, blo, bhi = cboot(clusters, stat)
            if lab == "0–8":
                say(f"| {setting} | {lab} | {len(inb):,} | {base[0]:.3f} "
                    f"[{blo[0]:.3f}, {bhi[0]:.3f}] | (reference) | "
                    "(reference) |")
            else:
                say(f"| {setting} | {lab} | {len(inb):,} | {base[0]:.3f} "
                    f"[{blo[0]:.3f}, {bhi[0]:.3f}] | {base[1]:+.3f} "
                    f"[{blo[1]:+.3f}, {bhi[1]:+.3f}] | {base[2]:+.3f} "
                    f"[{blo[2]:+.3f}, {bhi[2]:+.3f}] |")
    say("")

    # ------------------------------------------------------ V2 continuous --
    say("## V2. Continuous z² slope on wind (with composition controls)\n")
    say("OLS of z² on wind/10 plus controls for tour, race length and "
        "|skill| decile — so the slope cannot be a composition artefact of "
        "which matches happen to be played in wind.\n")

    def make_design(sub):
        sk = np.array([abs(r["skill"]) for r in sub])
        qs = np.quantile(sk, np.linspace(0, 1, 11)[1:-1])
        dec = np.searchsorted(qs, sk)
        cols = [np.ones(len(sub)), np.array([r["wind"] / 10 for r in sub])]
        cols.append(np.array([1.0 if r["tour"] == "MLP" else 0.0 for r in sub]))
        cols.append(np.array([1.0 if r["T"] == 15 else 0.0 for r in sub]))
        for k in range(1, 10):
            cols.append((dec == k).astype(float))
        return np.column_stack(cols)

    v2_slope = {}
    say("| setting | games | z² slope per +10 mph [95% CI] | "
        "same, no controls |")
    say("|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        sub = [r for r in rows if r["setting"] == setting]
        clusters = defaultdict(list)
        for r in sub:
            clusters[r["ev"]].append(r)

        def stat(s):
            X = make_design(s)
            y = np.array([r["z2"] for r in s])
            try:
                b1 = np.linalg.lstsq(X, y, rcond=None)[0][1]
                X0 = X[:, :2]
                b0 = np.linalg.lstsq(X0, y, rcond=None)[0][1]
            except np.linalg.LinAlgError:
                return None
            return [b1, b0]

        base, blo, bhi = cboot(clusters, stat)
        v2_slope[setting] = (base[0], blo[0], bhi[0])
        say(f"| {setting} | {len(sub):,} | {base[0]:+.4f} [{blo[0]:+.4f}, "
            f"{bhi[0]:+.4f}] | {base[1]:+.4f} [{blo[1]:+.4f}, "
            f"{bhi[1]:+.4f}] |")
    say("")

    # --------------------------------------------- V3 upsets at matched p --
    say("## V3. Upset rate at matched predicted probability\n")
    say("Games are stratified into 10 bins of the race-DP predicted win "
        "probability of team 1; within each stratum the observed win rate is "
        "compared calm (<8 mph) vs windy (≥14 mph). The reported number is "
        "the stratum-size-weighted mean of (observed − predicted) windy minus "
        "the same calm — i.e. the extra upset rate in wind at equal skill "
        "gap. Positive = favourites lose more in wind.\n")

    def upset_stat(s, wlo):
        edges = np.linspace(0.0, 1.0, 11)
        num = den = 0.0
        for k in range(10):
            a = [r for r in s if edges[k] <= r["pwin"] < edges[k + 1]]
            if not a:
                continue
            calm = [r for r in a if r["wind"] < 8]
            windy = [r for r in a if r["wind"] >= wlo]
            if len(calm) < 5 or len(windy) < 5:
                continue
            ec = np.mean([r["won"] - r["pwin"] for r in calm])
            ew = np.mean([r["won"] - r["pwin"] for r in windy])
            fav = 1.0 if edges[k] >= 0.5 else -1.0   # sign so + = more upsets
            wgt = min(len(calm), len(windy))
            num += wgt * (-fav) * (ew - ec)
            den += wgt
        return num / den if den else None

    say("| setting | windy threshold | extra upset rate in wind [95% CI] |")
    say("|---|---|---|")
    for setting in ("outdoor", "indoor"):
        sub = [r for r in rows if r["setting"] == setting]
        clusters = defaultdict(list)
        for r in sub:
            clusters[r["ev"]].append(r)
        for wlo in (14, 12):
            base, lo, hi = cboot(clusters, lambda s, w=wlo: upset_stat(s, w))
            say(f"| {setting} | ≥{wlo} mph | {100*base:+.2f} pp "
                f"[{100*lo:+.2f}, {100*hi:+.2f}] |")
    say("")

    # ------------------------------------------- V4 reliability / Brier ----
    say("## V4. Reliability slope and Brier decomposition by wind bin\n")
    say("Logistic recalibration logit P(win) = alpha + beta·logit(p_v2) per "
        "bin. beta < 1 means predictions are too confident for that bin — the "
        "signature of extra outcome noise. (The absolute level of beta is not "
        "interpretable here: v2 values are current-form applied "
        "retroactively, which inflates beta everywhere. The comparison "
        "ACROSS bins within a setting is the test.)\n")

    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def fit_beta(s):
        x = np.array([logit(r["pwin"]) for r in s])
        y = np.array([1.0 if r["won"] else 0.0 for r in s])
        b = np.zeros(2)
        X = np.column_stack([np.ones(len(s)), x])
        for _ in range(40):
            eta = X @ b
            p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
            W = p * (1 - p) + 1e-9
            g = X.T @ (y - p)
            H = X.T @ (X * W[:, None])
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return None
            b = b + step
            if np.max(np.abs(step)) < 1e-9:
                break
        return b[1]

    say("| setting | bin | games | reliability slope beta [95% CI] | "
        "Brier | Brier of a p=½ forecast |")
    say("|---|---|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        sub = [r for r in rows if r["setting"] == setting]
        clusters = defaultdict(list)
        for r in sub:
            clusters[r["ev"]].append(r)
        for lo_, hi_, lab in BINS:
            inb = [r for r in sub if lo_ <= r["wind"] < hi_]
            if len(inb) < 200:
                say(f"| {setting} | {lab} | {len(inb)} | (too few) | | |")
                continue

            def stat(s, lo_=lo_, hi_=hi_):
                a = [r for r in s if lo_ <= r["wind"] < hi_]
                if len(a) < 100:
                    return None
                bt = fit_beta(a)
                return None if bt is None else [bt]

            base, blo, bhi = cboot(clusters, stat)
            br = np.mean([(r["pwin"] - (1.0 if r["won"] else 0.0)) ** 2
                          for r in inb])
            say(f"| {setting} | {lab} | {len(inb):,} | {base[0]:.3f} "
                f"[{blo[0]:.3f}, {bhi[0]:.3f}] | {br:.4f} | "
                f"{np.mean([(0.5-(1.0 if r['won'] else 0.0))**2 for r in inb]):.4f} |")
    say("")

    # --------------------------------------------------------- power / MDE --
    say("## V5. What the variance null is worth (power translation)\n")
    say("Inflating the outcome sd by a factor f multiplies mean z² by f². "
        "For the reference favourite (v2 expected share 0.60, an 83.6% "
        "favourite in a race to 11), an sd inflation of f raises the upset "
        "rate to Phi(−(mu−½)/(f·sd)) — the table converts the CI edges of "
        "V2 into that currency at 20 mph vs 5 mph.\n")
    mu, sd = sm.moments(0.60, 11)
    # the normal approximation is the SAME map used for the perturbed rows,
    # so the reference must use it too or the change column inherits the
    # approximation error (the exact race-DP upset rate is 16.4%).
    base_up = 1 - 0.5 * (1 + math.erf(((mu - 0.5) / sd) / math.sqrt(2)))
    say(f"Reference: mu = {mu:.3f}, sd = {sd:.3f}, upset rate "
        f"{100*base_up:.1f}% (normal approximation; the exact race-DP value "
        f"is {100*(1-sm.win(0.60, 11)):.1f}% — the CHANGE column is the "
        "object, and both ends use the same approximation).\n")
    say("| z² slope per 10 mph | z² at 20 mph vs 5 mph | sd inflation f | "
        "upset rate | change |")
    say("|---|---|---|---|---|")
    outdoor = [r for r in rows if r["setting"] == "outdoor"]
    z2_base = float(np.mean([r["z2"] for r in outdoor]))
    b, blo, bhi = v2_slope["outdoor"]     # reuse V2's outdoor bootstrap
    for tag, slope in [("point estimate", b), ("lower CI edge", blo),
                       ("upper CI edge", bhi),
                       ("upper CI edge, de-attenuated (lambda_T = 0.941)",
                        bhi / 0.941)]:
        dz = slope * 1.5
        f = math.sqrt(max(0.05, (z2_base + dz) / z2_base))
        up = 1 - 0.5 * (1 + math.erf(((mu - 0.5) / (f * sd)) / math.sqrt(2)))
        say(f"| {tag} {slope:+.4f} | {dz:+.4f} | {f:.4f} | {100*up:.2f}% | "
            f"{100*(up-base_up):+.2f} pp |")
    say("")
    say(f"Mean outdoor z² = {z2_base:.3f} is the denominator: the excess over "
        "1.0 is rating error + serve clustering, present in every bin.\n")

    (ROOT / "model/weather_review/b6_variance.md").write_text("\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/b6_variance.md")


if __name__ == "__main__":
    main()
