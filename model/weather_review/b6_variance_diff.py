"""B6 part 1b — the outdoor MINUS indoor contrast for the variance channel.

b6_variance.py reports the two arms separately; the falsification logic needs
the DIFFERENCE with its own interval, bootstrapped over events jointly across
both arms (an event contributes to whichever arm it belongs to, so resampling
events resamples both arms coherently).

    python model/weather_review/b6_variance_diff.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b6_lib as L  # noqa: E402
from b6_lib import ROOT, sigmoid  # noqa: E402

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def main():
    sm = L.ShareMoments(n=481)
    rows = []
    for g in L.load_games():
        if g["wind"] is None or g["setting"] not in ("outdoor", "indoor"):
            continue
        p = min(max(sigmoid(g["eta"]), 0.16), 0.84)
        mu, sd = sm.moments(p, g["T"])
        z = (g["share"] - mu) / sd
        rows.append(dict(ev=g["event"], setting=g["setting"], z2=z * z,
                         wind=g["wind"], won=g["won"], pwin=sm.win(p, g["T"]),
                         T=g["T"], tour=g["tour"], skill=p - 0.5))

    clusters = defaultdict(list)
    for r in rows:
        clusters[r["ev"]].append(r)
    keys = list(clusters)

    def upset_excess(s, wlo):
        edges = np.linspace(0.0, 1.0, 11)
        num = den = 0.0
        for k in range(10):
            a = [r for r in s if edges[k] <= r["pwin"] < edges[k + 1]]
            calm = [r for r in a if r["wind"] < 8]
            windy = [r for r in a if r["wind"] >= wlo]
            if len(calm) < 5 or len(windy) < 5:
                continue
            ec = np.mean([r["won"] - r["pwin"] for r in calm])
            ew = np.mean([r["won"] - r["pwin"] for r in windy])
            sgn = -1.0 if edges[k] >= 0.5 else 1.0
            wgt = min(len(calm), len(windy))
            num += wgt * sgn * (ew - ec)
            den += wgt
        return num / den if den else np.nan

    def z2_slope(s):
        w = np.array([r["wind"] / 10 for r in s])
        y = np.array([r["z2"] for r in s])
        X = np.column_stack([np.ones(len(s)), w])
        return np.linalg.lstsq(X, y, rcond=None)[0][1]

    def stat(s):
        o = [r for r in s if r["setting"] == "outdoor"]
        i = [r for r in s if r["setting"] == "indoor"]
        if len(o) < 500 or len(i) < 500:
            return None
        u_o, u_i = upset_excess(o, 14), upset_excess(i, 14)
        s_o, s_i = z2_slope(o), z2_slope(i)
        return [u_o, u_i, u_o - u_i, s_o, s_i, s_o - s_i]

    base = stat(rows)
    rng = np.random.default_rng(77)
    draws = []
    for _ in range(1500):
        pick = rng.integers(0, len(keys), len(keys))
        s = []
        for j in pick:
            s.extend(clusters[keys[j]])
        v = stat(s)
        if v is not None and np.all(np.isfinite(v)):
            draws.append(v)
    d = np.array(draws)
    lo = np.percentile(d, 2.5, axis=0)
    hi = np.percentile(d, 97.5, axis=0)

    say("# B6 — variance channel, outdoor MINUS indoor (joint event bootstrap)\n")
    say(f"1500 resamples of the {len(keys)} events; both arms move together "
        "in every resample, so the difference row is a proper paired "
        "contrast, not a comparison of two independent intervals.\n")
    say("| quantity | outdoor | indoor | **outdoor − indoor** |")
    say("|---|---|---|---|")
    say(f"| extra upset rate at ≥14 mph, matched predicted probability | "
        f"{100*base[0]:+.2f} pp [{100*lo[0]:+.2f}, {100*hi[0]:+.2f}] | "
        f"{100*base[1]:+.2f} pp [{100*lo[1]:+.2f}, {100*hi[1]:+.2f}] | "
        f"**{100*base[2]:+.2f} pp [{100*lo[2]:+.2f}, {100*hi[2]:+.2f}]** |")
    say(f"| z² slope per +10 mph | {base[3]:+.4f} [{lo[3]:+.4f}, {hi[3]:+.4f}] "
        f"| {base[4]:+.4f} [{lo[4]:+.4f}, {hi[4]:+.4f}] | "
        f"**{base[5]:+.4f} [{lo[5]:+.4f}, {hi[5]:+.4f}]** |")
    say("")
    (ROOT / "model/weather_review/b6_variance_diff.md").write_text(
        "\n".join(OUT) + "\n")
    print("wrote model/weather_review/b6_variance_diff.md")


if __name__ == "__main__":
    main()
