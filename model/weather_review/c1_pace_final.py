"""C1c — the decisive pace specs + gusts.

    python model/weather_review/c1_pace_final.py

1. OUTDOOR-MINUS-INDOOR interaction (the falsification arm differenced
   inside one regression, so the contrast gets its own CI).
2. MUNDLAK decomposition: day-mean wind vs within-day deviation, both in
   one regression — the between/within split that section A of
   c1_pace_robust.py showed the whole result hinges on.
3. Rain controls (precipitation is the obvious day-level rival cause).
4. GUSTS: pace and score shape on gust speed, and on the gust-minus-
   sustained residual (pure gustiness).
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_texture import (build_score_rows, cluster_boot_ols, design,  # noqa
                        fmt, load_pace)
from c1_pace_robust import attach, keep  # noqa: E402

NB = 1000


def prep(rows):
    """add day-mean / within-day-deviation wind + gust, and precip."""
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["evday"]].append(r)
    for r in rows:
        r["precip"] = float(r["precip"] or 0.0)
    for d, rs in by_day.items():
        mw = sum(x["wind"] for x in rs) / len(rs)
        mg = sum(x["gust"] for x in rs) / len(rs)
        for x in rs:
            x["wind_day"] = mw
            x["wind_dev"] = x["wind"] - mw
            x["gust_day"] = mg
            x["gust_dev"] = x["gust"] - mg
    return rows


BASE = [("const", lambda r: 1.0),
        ("temp10", lambda r: r["temp"] / 10.0),
        ("inv_points", lambda r: 1.0 / r["points"]),
        ("first_game", lambda r: float(r["first_game"])),
        ("fmt15", lambda r: 1.0 if r["fmt"] == "sideout_15" else 0.0),
        ("gn3plus", lambda r: 1.0 if r["gn"] >= 3 else 0.0)]
HR = ("hr", lambda r: r["hour"])
EV = ("ev", lambda r: r["event_id"])


def fit(sub, extra_cols, want, y=None, cats=(HR, EV), nb=NB):
    X, names = design(sub, BASE + list(extra_cols), list(cats))
    if y is None:
        y = np.array([r["spp"] for r in sub])
    return cluster_boot_ols(X, y, [r["event_id"] for r in sub], names,
                            list(want), nboot=nb)


def head(t):
    print("\n" + "-" * 74 + "\n" + t + "\n" + "-" * 74, flush=True)


def main():
    only = set(sys.argv[1:]) or {"1", "2", "3", "4"}
    rows = prep(attach(load_pace()))
    out = keep(rows, "outdoor")
    ind = keep(rows, "indoor")
    both = out + ind
    for r in both:
        r["outd"] = 1.0 if r["corrected_all"] == "outdoor" else 0.0

    if "1" in only:
     head("1. OUTDOOR MINUS INDOOR — wind x outdoor interaction, one model")
     cols = [("wind10", lambda r: r["wind"] / 10.0),
             ("wind10_x_out", lambda r: r["wind"] / 10.0 * r["outd"]),
             ("temp10_x_out", lambda r: r["temp"] / 10.0 * r["outd"])]
     res = fit(both, cols, ["wind10", "wind10_x_out"])
     print("   n=%d (%d outdoor / %d indoor), %d events"
           % (len(both), len(out), len(ind),
              len({r["event_id"] for r in both})))
     print("   indoor slope        %s" % fmt(res["wind10"], "s/pt/10mph", 1, 3))
     print("   OUTDOOR - INDOOR    %s" % fmt(res["wind10_x_out"],
                                             "s/pt/10mph", 1, 3))

    if "2" in only:
     head("2. MUNDLAK — day-mean wind vs within-day deviation (outdoor)")
     cols = [("wind10_day", lambda r: r["wind_day"] / 10.0),
             ("wind10_dev", lambda r: r["wind_dev"] / 10.0)]
     for lab, sub in (("outdoor", out), ("indoor", ind)):
         res = fit(sub, cols, ["wind10_day", "wind10_dev"])
         sd_day = np.std([r["wind_day"] for r in sub])
         sd_dev = np.std([r["wind_dev"] for r in sub])
         print("   [%s] n=%d  sd(day-mean)=%.2f mph  sd(within-day)=%.2f mph"
               % (lab, len(sub), sd_day, sd_dev))
         print("      BETWEEN-day  %s" % fmt(res["wind10_day"], "s/pt/10mph", 1, 3))
         print("      WITHIN-day   %s" % fmt(res["wind10_dev"], "s/pt/10mph", 1, 3))

    if "3" in only:
     head("3. RAIN as a rival day-level cause (outdoor)")
     cols = [("wind10", lambda r: r["wind"] / 10.0),
             ("precip", lambda r: r["precip"])]
     res = fit(out, cols, ["wind10", "precip"])
     print("   + precip control    %s" % fmt(res["wind10"], "s/pt/10mph", 1, 3))
     print("   precip coefficient  %s" % fmt(res["precip"], "s/pt per inch", 1, 2))
     dry = [r for r in out if r["precip"] <= 0]
     res = fit(dry, [("wind10", lambda r: r["wind"] / 10.0)], ["wind10"])
     print("   dry hours only n=%d  %s" % (len(dry), fmt(res["wind10"],
                                                         "s/pt/10mph", 1, 3)))

    if "4" in only:
     head("4a. GUSTS — pace")
     for lab, sub in (("outdoor", out), ("indoor", ind)):
         r1 = fit(sub, [("gust10", lambda r: r["gust"] / 10.0)], ["gust10"])
         r2 = fit(sub, [("wind10", lambda r: r["wind"] / 10.0),
                        ("gustiness", lambda r: (r["gust"] - r["wind"]) / 10.0)],
                  ["wind10", "gustiness"])
         print("   [%s] gust alone       %s" % (lab, fmt(r1["gust10"],
                                                         "s/pt per +10 mph gust", 1, 3)))
         print("   [%s] sustained | gust %s" % (lab, fmt(r2["wind10"],
                                                         "s/pt/10mph", 1, 3)))
         print("   [%s] gustiness resid  %s" % (lab, fmt(r2["gustiness"],
                                                         "s/pt per +10 mph excess", 1, 3)))

     head("4b. GUSTS — score shape (blowout / deuce / margin), event+hour FE")
     srows = build_score_rows()
     for r in srows:
         if r["gust"] is None or (isinstance(r["gust"], float)
                                  and math.isnan(r["gust"])):
             r["gust"] = r["wind"]
     scols = [("gust10", lambda r: r["gust"] / 10.0),
              ("temp10", lambda r: r["temp"] / 10.0),
              ("gap", lambda r: r["eta"]), ("gap2", lambda r: r["eta"] ** 2)]
     for arm in ("outdoor", "indoor"):
         sub = [r for r in srows if r["arm"] == arm]
         X, names = design(sub, [("const", lambda r: 1.0)] + scols,
                           [("hr", lambda r: r["hour"]),
                            ("ev", lambda r: r["event_id"])])
         cl = [r["event_id"] for r in sub]
         print("   [%s] n=%d" % (arm, len(sub)))
         for v, unit, scale, dec in (("blowout", "pp/10mph gust", 100.0, 2),
                                     ("deuce", "pp/10mph gust", 100.0, 2),
                                     ("margin", "pts/10mph gust", 1.0, 3)):
             y = np.array([r[v] for r in sub])
             res = cluster_boot_ols(X, y, cl, names, ["gust10"], nboot=NB)
             print("      %-8s %s" % (v, fmt(res["gust10"], unit, scale, dec)))


if __name__ == "__main__":
     main()
