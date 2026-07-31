"""C1e — the same between/within and outdoor-minus-indoor discipline
applied to the one score-shape measure that leaned (deuce rate).

    python model/weather_review/c1_score_final.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_texture import build_score_rows, cluster_boot_ols, design, fmt  # noqa

NB = 1000


def main():
    rows = build_score_rows()
    by_day = defaultdict(list)
    for r in rows:
        by_day[(r["event_id"], r["day"])].append(r)
    for _, rs in by_day.items():
        m = sum(x["wind"] for x in rs) / len(rs)
        for x in rs:
            x["wind_day"] = m
            x["wind_dev"] = x["wind"] - m

    BASE = [("const", lambda r: 1.0), ("temp10", lambda r: r["temp"] / 10.0),
            ("gap", lambda r: r["eta"]), ("gap2", lambda r: r["eta"] ** 2)]
    CATS = [("hr", lambda r: r["hour"]), ("ev", lambda r: r["event_id"])]

    print("MUNDLAK split, score shape (event + hour FE, cluster boot / event)")
    cols = BASE + [("wind10_day", lambda r: r["wind_day"] / 10.0),
                   ("wind10_dev", lambda r: r["wind_dev"] / 10.0)]
    for arm in ("outdoor", "indoor"):
        sub = [r for r in rows if r["arm"] == arm]
        X, names = design(sub, cols, CATS)
        cl = [r["event_id"] for r in sub]
        print("\n[%s] n=%d  sd(day-mean)=%.2f  sd(within-day)=%.2f"
              % (arm, len(sub), np.std([r["wind_day"] for r in sub]),
                 np.std([r["wind_dev"] for r in sub])))
        for v, unit, sc, dec in (("deuce", "pp/10mph", 100.0, 2),
                                 ("blowout", "pp/10mph", 100.0, 2),
                                 ("margin", "pts/10mph", 1.0, 3)):
            y = np.array([r[v] for r in sub])
            res = cluster_boot_ols(X, y, cl, names,
                                   ["wind10_day", "wind10_dev"], nboot=NB)
            print("   %-8s BETWEEN %s" % (v, fmt(res["wind10_day"], unit, sc, dec)))
            print("   %-8s WITHIN  %s" % ("", fmt(res["wind10_dev"], unit, sc, dec)))

    print("\n\nOUTDOOR MINUS INDOOR interaction, score shape")
    both = [r for r in rows if r["arm"] in ("outdoor", "indoor")]
    for r in both:
        r["outd"] = 1.0 if r["arm"] == "outdoor" else 0.0
    cols = BASE + [("wind10", lambda r: r["wind"] / 10.0),
                   ("wind10_x_out", lambda r: r["wind"] / 10.0 * r["outd"]),
                   ("temp10_x_out", lambda r: r["temp"] / 10.0 * r["outd"])]
    X, names = design(both, cols, CATS)
    cl = [r["event_id"] for r in both]
    for v, unit, sc, dec in (("deuce", "pp/10mph", 100.0, 2),
                             ("blowout", "pp/10mph", 100.0, 2),
                             ("margin", "pts/10mph", 1.0, 3)):
        y = np.array([r[v] for r in both])
        res = cluster_boot_ols(X, y, cl, names, ["wind10", "wind10_x_out"],
                               nboot=NB)
        print("   %-8s indoor %s" % (v, fmt(res["wind10"], unit, sc, dec)))
        print("   %-8s OUT-IN %s" % ("", fmt(res["wind10_x_out"], unit, sc, dec)))


if __name__ == "__main__":
    main()
