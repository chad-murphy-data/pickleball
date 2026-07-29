"""B3 part 2 — dose-response, gusts, endgame contrast, and the MDE.

Reuses the (match x server-side) rally aggregate built by
fetch_rally_match_side.py and the cell builder in
rally_favorites_allmatches.py.

    python model/weather_review/rally_favorites_extras.py <scratch>

Adds to part 1:
  A. non-parametric wind bins — is the skill->rally slope b flat in wind?
     (a threshold effect that a linear interaction would miss, and the
     shape an attenuated-regressor story predicts)
  B. gusts instead of sustained wind (collected, never used)
  C. an explicit endgame contrast: adv x wind x late triple interaction
  D. minimum detectable effect, translated to win probability
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib.race import game_win_prob  # noqa: E402
import rally_favorites_allmatches as P1  # noqa: E402


def main():
    cells, _ = P1.build_cells()
    for c in cells:
        c["advw"] = c["adv"] * c["w"]
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# B3 part 2 — dose-response, gusts, endgame contrast, MDE\n")

    # ---- A. wind bins: fit b (skill slope) separately per bin ----------
    say("## A. non-parametric: skill->rally slope b within wind bins\n")
    say("logit p = a + b*adv fitted inside each bin. Compression means b "
        "falls as wind rises; a threshold effect shows as a drop in the top "
        "bin only. Cluster bootstrap over events.\n")
    bins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 40)]
    for setting in ("outdoor", "indoor"):
        say(f"### {setting}")
        for lo, hi in bins:
            sub = [c for c in cells if c["setting"] == setting
                   and lo <= c["wind"] < hi]
            if sum(c["n"] for c in sub) < 8000:
                say(f"    {lo:>2}-{hi:<2} mph: only "
                    f"{sum(c['n'] for c in sub)} rallies — skipped")
                continue
            beta, cis = P1.fit_and_ci(sub, ["adv"],
                                      f"    {lo}-{hi} mph", out, R=300)
        say("")

    # ---- B. gusts ------------------------------------------------------
    say("## B. gusts instead of sustained wind (same spec)\n")
    for c in cells:
        g = c["gust"]
        c["g"] = (g / 10.0) if g == g else None
        c["advg"] = (c["adv"] * c["g"]) if c["g"] is not None else None
    for setting in ("outdoor", "indoor"):
        sub = [c for c in cells if c["setting"] == setting and c["g"] is not None]
        P1.fit_and_ci(sub, ["adv", "g", "advg"], f"[{setting}] gust spec", out,
                      R=300)
    say("")

    # ---- C. endgame contrast, one model --------------------------------
    say("## C. endgame contrast in a single model (to-11 games)\n")
    say("logit p = a + b*adv + c*w + d*adv*w + e*late + f*late*w + "
        "g*late*adv + h*late*adv*w, late = leader score >= 9. h is the "
        "test: is the wind interaction bigger late in the game?\n")
    for setting in ("outdoor", "indoor"):
        sub = []
        for c in cells:
            if c["setting"] != setting or "11" not in c["fmt"]:
                continue
            for late, rng_ in ((0.0, range(0, 9)), (1.0, range(9, 12))):
                n = sum(c["buck"][i][0] for i in rng_)
                wv = sum(c["buck"][i][1] for i in rng_)
                if n < 3:
                    continue
                sub.append(dict(c, n=n, wins=wv, late=late,
                                latew=late * c["w"], lateadv=late * c["adv"],
                                lateadvw=late * c["advw"]))
        P1.fit_and_ci(sub, ["adv", "w", "advw", "late", "latew", "lateadv",
                            "lateadvw"], f"[{setting}] late contrast", out,
                      R=300)
    say("")

    # ---- D. MDE --------------------------------------------------------
    say("## D. what the outdoor interval still allows, in real units\n")
    b, d_lo, d_hi, d = 0.5016, -0.0481, +0.0134, -0.0191
    for w_mph, lab in ((15.0, "15 mph"), (20.0, "20 mph")):
        w = w_mph / 10.0
        for dd, lab2 in ((d, "point est"), (d_lo, "CI floor"), (d_hi, "CI ceil")):
            mult = (b + dd * w) / b
            say(f"  {lab} {lab2:>9}: effective skill slope x {mult:.3f}")
            for p0 in (0.65, 0.75, 0.90):
                eta = eta_for(p0)
                p1 = game_prob(eta * mult)
                say(f"      a {p0:.0%} favorite (game to 11) -> {p1:.1%} "
                    f"({100*(p1-p0):+.1f} pp)")
        say("")
    say("Ratings enter both b and d, so classical error in the v2 rating "
        "attenuates BOTH and largely cancels in the ratio d*w/b used above. "
        "Error in the WIND regressor does not cancel: it attenuates d "
        "toward zero, so the true compression could be larger than the "
        "interval by roughly 1/reliability of ERA5 grid wind as a proxy "
        "for on-court wind. If that reliability were as low as 0.6, the "
        "outdoor floor of -0.048 would correspond to a true -0.080.")

    (Path(__file__).parent / "rally_favorites_extras.txt").write_text(
        "\n".join(out) + "\n")
    print("\nwrote model/weather_review/rally_favorites_extras.txt")


def game_prob(eta: float, T: int = 11) -> float:
    return game_win_prob(eta, T)


def eta_for(p: float, T: int = 11) -> float:
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if game_prob(mid, T) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    main()
