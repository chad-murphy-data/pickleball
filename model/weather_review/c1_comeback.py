"""C1f — comeback frequency vs wind (the last score-shape measure).

    python model/weather_review/c1_comeback.py

A "comeback" = in a best-of-3 match that went past game 1, the team that
LOST game 1 wins the match. Wind folklore says scrambled matches; a real
texture effect should raise this at fixed skill gap. Outdoor arm on
corrected labels, indoor arm as control, cluster bootstrap over events.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1_lib import (ROOT, game_eta, label_arms, load_games, load_hourly,  # noqa
                    load_v2)
from c1_texture import build_match_hours, cluster_boot_ols, design, fmt  # noqa

NB = 1000


def main():
    hours, _ = build_match_hours()
    hourly = load_hourly()
    arms = label_arms()
    v2 = load_v2()
    rows = []
    for mid, gs in load_games().items():
        if len(gs) < 2 or gs[0]["best_of"] != "3":
            continue
        h = hours.get(mid)
        if not h:
            continue
        ev, hk, hr = h
        wx = hourly.get((ev, hk))
        if not wx or wx["wind"] is None or wx["temp"] is None:
            continue
        eta = game_eta(gs[0], v2)
        if eta is None:
            continue
        # team 1 orientation is stable across a match's rows in games.csv
        w1 = sum(1 for g in gs if int(g["t1_score"]) > int(g["t2_score"]))
        w2 = len(gs) - w1
        if w1 == w2:
            continue
        g1_t1 = int(gs[0]["t1_score"]) > int(gs[0]["t2_score"])
        m_t1 = w1 > w2
        rows.append({"event_id": ev, "hour": hr, "wind": wx["wind"],
                     "gust": wx["gust"] if wx["gust"] is not None else wx["wind"],
                     "temp": wx["temp"], "eta": abs(eta),
                     "comeback": 0.0 if g1_t1 == m_t1 else 1.0,
                     "arm": arms["corrected_all"].get(ev) or ""})
    BASE = [("const", lambda r: 1.0), ("temp10", lambda r: r["temp"] / 10.0),
            ("gap", lambda r: r["eta"]), ("gap2", lambda r: r["eta"] ** 2)]
    CATS = [("hr", lambda r: r["hour"]), ("ev", lambda r: r["event_id"])]
    for arm in ("outdoor", "indoor"):
        sub = [r for r in rows if r["arm"] == arm]
        if len(sub) < 300:
            print(f"[{arm}] n={len(sub)} too thin")
            continue
        cl = [r["event_id"] for r in sub]
        y = np.array([r["comeback"] for r in sub])
        for v, lab in (("wind", "sustained"), ("gust", "gust")):
            X, names = design(sub, BASE + [("w10", lambda r, v=v: r[v] / 10.0)],
                              CATS)
            res = cluster_boot_ols(X, y, cl, names, ["w10"], nboot=NB)
            print("[%s] n=%d  comeback rate %.3f   %s: %s"
                  % (arm, len(sub), y.mean(), lab,
                     fmt(res["w10"], "pp per +10 mph", 100.0, 2)))


if __name__ == "__main__":
    main()
