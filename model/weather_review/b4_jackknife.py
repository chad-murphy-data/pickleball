"""B4 part 1-2 — leave-one-EVENT-out influence on every tail-bin statistic
AND on the headline nulls (a null can be fragile too).

    python model/weather_review/b4_jackknife.py

Reproduces each published statistic exactly (same joins, same labels, same
bins as the committed model/weather_report.py, model/end_effects.py and
model/favorites_wind.py), then recomputes it with each contributing EVENT
deleted in turn. Reports:
  * full-sample estimate
  * jackknife spread (min / max over the leave-one-event-out replicates)
  * the top 3 most influential events, NAMED, with the signed change
  * concentration: share of the statistic's sample from the top event
  * for the favorites-wind outdoor null: leave-one-TOUR-out and
    leave-one-YEAR-out as well.

Deterministic; stdlib only. Writes model/weather_review/b4_jackknife.md
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(HERE))
from sitelib.race import game_win_prob, sigmoid, team_eta  # noqa: E402
import b2b_lib as L  # noqa: E402

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------- context
GEO = {r["event_id"]: r for r in read_csv(ROOT / "data/event_geo.csv")}
NAME = {e: f'{r["event_name"]} ({r["first_date"]}, {r["venue"] or r["city"]})'
        for e, r in GEO.items()}
SETTING = {e: r["setting"] for e, r in GEO.items()}


OV = {r["event_id"]: r for r in read_csv(ROOT / "data/venue_overrides.csv")}


def ename(e):
    o = OV.get(e)
    tag = (f'[audit: {o["setting"]}/{o["confidence"]}]' if o
           else "[UNAUDITED — heuristic label]")
    return f"{NAME.get(e, e)} {tag}"


# ------------------------------------------------------------- jackknife
def jackknife(units, stat, label, note=""):
    """units: list of (event_id, payload). stat(list_of_payloads)->float."""
    by_ev = defaultdict(list)
    for e, p in units:
        by_ev[e].append(p)
    allp = [p for _, p in units]
    full = stat(allp)
    infl = []
    for e in by_ev:
        rest = [p for e2, ps in by_ev.items() if e2 != e for p in ps]
        if len(rest) < 5:
            infl.append((e, float("nan"), len(by_ev[e])))
            continue
        v = stat(rest)
        infl.append((e, v - full, len(by_ev[e])))
    infl.sort(key=lambda t: -abs(t[1] if t[1] == t[1] else 0.0))
    vals = [full + d for _, d, _ in infl if d == d]
    say(f"\n### {label}")
    if note:
        say(f"*{note}*")
    top_share = max(n for _, _, n in infl) / len(allp)
    say(f"\nfull = **{full:+.4f}**   n = {len(allp)}   events = {len(by_ev)}"
        f"   LOEO range [{min(vals):+.4f}, {max(vals):+.4f}]"
        f"   biggest single event = {top_share*100:.0f}% of the sample")
    say("\n| rank | event dropped | n in stat | Δ estimate | estimate w/o it |")
    say("|---|---|---|---|---|")
    for i, (e, d, n) in enumerate(infl[:3], 1):
        say(f"| {i} | {ename(e)} | {n} | {d:+.4f} | {full+d:+.4f} |")
    return full, infl


def mean(v):
    return sum(v) / len(v) if v else float("nan")


# =====================================================================
# 1. Design B / Design C tail bins (end_effects.py)
# =====================================================================
def design_bc():
    arms = L.label_arms()
    matches = L.load_matches(arms["published"])
    # ---- Design B rows (exact end_effects logic)
    rows_b = defaultdict(list)
    for r in read_csv(ROOT / "data/decider_splits.csv"):
        m = matches.get(r["match_id"])
        if not m:
            continue
        gn = int(r["game_number"])
        if m["tour"] == "MLP":
            if gn != 1:
                continue
        elif not (m["best_of"] == 3 and gn == 3) and \
                not (m["best_of"] == 5 and gn == 5):
            continue
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        x = int(r["pa_pre"]) / pre
        y = int(r["pa_post"]) / post
        p_hat = (int(r["pa_pre"]) + int(r["pa_post"])) / (pre + post)
        noise = p_hat * (1 - p_hat) * (1 / pre + 1 / post)
        z2 = (x - y) ** 2 / noise if noise > 0 else 0.0
        grp = L.group_of(m)
        if grp:
            rows_b[grp].append((m["event"], {"z2": z2, "wind": m["wind"]}))
    # ---- Design C rows
    rows_c = defaultdict(list)
    for r in read_csv(ROOT / "data/decider_serve_splits.csv"):
        m = matches.get(r["match_id"])
        if not m:
            continue
        gn = int(r["game_number"])
        if m["tour"] == "MLP":
            if gn != 1:
                continue
        elif not (m["best_of"] == 3 and gn == 3) and \
                not (m["best_of"] == 5 and gn == 5):
            continue
        grp = L.group_of(m)
        if not grp:
            continue
        for side in ("a", "b"):
            rp, wp = int(r[f"r{side}_pre"]), int(r[f"w{side}_pre"])
            rq, wq = int(r[f"r{side}_post"]), int(r[f"w{side}_post"])
            if rp < 5 or rq < 5:
                continue
            sq, noise, z2 = L.zsq(wp, rp, wq, rq)
            if noise <= 0:
                continue
            rows_c[grp].append((m["event"], {"z2": z2, "wind": m["wind"]}))

    say("\n## 1a. Design B — point-share swing z², windy 14+ tail bin "
        "(published 1.95, n=111)")
    windy = "OUTDOOR windy 14+"
    calm = "OUTDOOR calm <8"
    mz = lambda s: mean([d["z2"] for d in s])
    jackknife(rows_b[windy], mz, "Design B mean z², OUTDOOR windy 14+")

    # Δ vs calm: tag payloads
    units = ([(e, ("t", d)) for e, d in rows_b[windy]]
             + [(e, ("r", d)) for e, d in rows_b[calm]])

    def dz(s):
        t = [d["z2"] for tag, d in s if tag == "t"]
        r = [d["z2"] for tag, d in s if tag == "r"]
        if not t or not r:
            return float("nan")
        return mean(t) - mean(r)
    jackknife(units, dz, "Design B Δ mean z², windy 14+ − outdoor calm "
              "(published +0.224 [−0.160, +0.585])",
              "an event can sit in BOTH arms (different match hours)")

    say("\n## 1b. Design C — serve-rate swing z², windy 14+ "
        "(published 1.34, n=162 team-halves)")
    jackknife(rows_c[windy], mz, "Design C mean z², OUTDOOR windy 14+")
    unitsc = ([(e, ("t", d)) for e, d in rows_c[windy]]
              + [(e, ("r", d)) for e, d in rows_c[calm]])
    jackknife(unitsc, dz, "Design C Δ mean z², windy 14+ − outdoor calm "
              "(published +0.190 [−0.067, +0.544])")

    # continuous slopes (nulls)
    def slope(s):
        pts = [(d["wind"], d["z2"]) for d in s]
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        den = sum((x - mx) ** 2 for x, _ in pts)
        return 10 * sum((x - mx) * (y - my) for x, y in pts) / den if den else 0.0
    out_b = [(e, d) for g, v in rows_b.items() if g.startswith("OUTDOOR")
             for e, d in v]
    out_c = [(e, d) for g, v in rows_c.items() if g.startswith("OUTDOOR")
             for e, d in v]
    say("\n## 1c. The continuous versions (published as near-nulls)")
    jackknife(out_b, slope, "Design B slope of z² on match-hour wind, outdoor "
              "(published +0.165 per +10 mph [−0.053, +0.364])")
    jackknife(out_c, slope, "Design C slope of z² on match-hour wind, outdoor "
              "(published +0.038 per +10 mph [−0.093, +0.166])")
    return matches


# =====================================================================
# 2. weather_report favourite-edge bins (tail bins + the "-6.0pp")
# =====================================================================
def fav_bins():
    wx = {(r["event_id"], r["date"]): r
          for r in read_csv(ROOT / "data/event_weather.csv")}
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        hourly[(r["event_id"], r["local_time"][:13])] = r
    start_hour = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    rows = []
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        w = wx.get((g["event_id"], g["date"]))
        st = SETTING.get(g["event_id"])
        if not w or st not in ("indoor", "outdoor"):
            continue
        try:
            wind_d = float(w["windspeed_10m_max"])
            tmax = float(w["temperature_2m_max"])
        except (TypeError, ValueError):
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        eta = team_eta(*vals)
        T = int(g["scoring_format"].rsplit("_", 1)[1])
        if T < 11:
            continue
        p1 = game_win_prob(eta, T)
        p_fav = max(p1, 1 - p1)
        fav_won = 1.0 if (int(g["t1_score"]) > int(g["t2_score"])) == (p1 >= 0.5) \
            else 0.0
        hw = hourly.get((g["event_id"], start_hour.get(g["match_id"], "")))
        rows.append({
            "ev": g["event_id"], "setting": st, "tour": g["tour"],
            "wind": wind_d, "tmax": tmax,
            "wind_h": float(hw["windspeed_10m"]) if hw else None,
            "temp_h": float(hw["temperature_2m"]) if hw else None,
            "p_fav": p_fav, "fav_won": fav_won})

    edge = lambda s: mean([d["fav_won"] for d in s]) - mean([d["p_fav"] for d in s])
    say("\n## 2. Favourite edge (obs − pred) tail bins, weather_report.py")
    cuts = [
        ("outdoor 20+ mph at match hour (published −0.047, n=82)",
         "outdoor", "wind_h", 20, 999),
        ("outdoor 14–20 mph at match hour — the '−6.0pp' result "
         "(published −0.060, n=1151)", "outdoor", "wind_h", 14, 20),
        ("outdoor 92+ °F at match hour (published −0.049, n=644)",
         "outdoor", "temp_h", 92, 999),
        ("outdoor 92+ °F daily max (published −0.055, n=1755)",
         "outdoor", "tmax", 92, 999),
        ("outdoor 20+ mph daily max (published −0.059, n=313)",
         "outdoor", "wind", 20, 999),
        ("INDOOR control 20+ mph at match hour (published −0.113, n=222)",
         "indoor", "wind_h", 20, 999),
        ("outdoor 0–8 mph at match hour — the calm REFERENCE "
         "(published −0.040, n=17139)", "outdoor", "wind_h", 0, 8),
    ]
    for label, st, key, lo, hi in cuts:
        sub = [(r["ev"], r) for r in rows
               if r["setting"] == st and r.get(key) is not None
               and lo <= r[key] < hi]
        jackknife(sub, edge, label)

    # difference of tail bin vs calm reference (the actual claim)
    say("\n### 2b. The CONTRAST that the claim rests on: tail bin − calm bin")
    for label, key, lo, hi in [
            ("outdoor 14–20 mph − outdoor calm <8 (match hour)", "wind_h", 14, 20),
            ("outdoor 20+ mph − outdoor calm <8 (match hour)", "wind_h", 20, 999),
            ("outdoor 92+ °F − outdoor <70 °F (match hour)", "temp_h", 92, 999)]:
        if key == "temp_h":
            ref = [(r["ev"], ("r", r)) for r in rows if r["setting"] == "outdoor"
                   and r.get(key) is not None and r[key] < 70]
        else:
            ref = [(r["ev"], ("r", r)) for r in rows if r["setting"] == "outdoor"
                   and r.get(key) is not None and r[key] < 8]
        trt = [(r["ev"], ("t", r)) for r in rows if r["setting"] == "outdoor"
               and r.get(key) is not None and lo <= r[key] < hi]

        def dedge(s):
            t = [d for tag, d in s if tag == "t"]
            r_ = [d for tag, d in s if tag == "r"]
            if not t or not r_:
                return float("nan")
            return edge(t) - edge(r_)
        jackknife(trt + ref, dedge, label)
    return rows


# =====================================================================
# 3. favorites_wind regression 1 — the headline NULL
# =====================================================================
def ols(rows, ykey, xkeys):
    p = len(xkeys) + 1
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for r in rows:
        x = [1.0] + [r[k] for k in xkeys]
        y = r[ykey]
        for i in range(p):
            xty[i] += x[i] * y
            for j in range(p):
                xtx[i][j] += x[i] * x[j]
    m = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r_: abs(m[r_][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            return None
        for r_ in range(p):
            if r_ != col:
                f = m[r_][col] / m[col][col]
                for c_ in range(col, p + 1):
                    m[r_][c_] -= f * m[col][c_]
    return [m[i][p] / m[i][i] for i in range(p)]


def fav_wind_reg1():
    hourly, start_hour = {}, {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = float(r["windspeed_10m"])
        except (TypeError, ValueError):
            pass
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    rows_by = defaultdict(list)
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        st = SETTING.get(g["event_id"])
        if st not in ("outdoor", "indoor"):
            continue
        wind = hourly.get((g["event_id"], start_hour.get(g["match_id"], "")))
        if wind is None:
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        eta = team_eta(*vals)
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        rows_by[st].append({"ev": g["event_id"], "tour": g["tour"],
                            "year": g["date"][:4],
                            "y": s1 / (s1 + s2) - 0.5,
                            "skill": sigmoid(eta) - 0.5, "w": wind / 10.0,
                            "sw": (sigmoid(eta) - 0.5) * wind / 10.0})

    def dcoef(s):
        c = ols(s, "y", ["skill", "w", "sw"])
        return c[3] if c else float("nan")

    def ratio(s):
        c = ols(s, "y", ["skill", "w", "sw"])
        return c[3] / c[1] if c else float("nan")

    say("\n## 3. favorites_wind regression 1 — is the NULL itself fragile?")
    for st, pub in (("outdoor", "+0.002 [−0.060, +0.064]"),
                    ("indoor", "−0.080 [−0.150, +0.020]")):
        units = [(r["ev"], r) for r in rows_by[st]]
        jackknife(units, dcoef, f"{st}: d (skill×wind per +10 mph), "
                                f"published {pub}")
        jackknife(units, ratio, f"{st}: d/b = fraction of the favourite's "
                                f"edge erased per +10 mph")

    say("\n### 3b. Leave-one-TOUR-out and leave-one-YEAR-out, outdoor d")
    say("\n| dropped | n | d | d/b |")
    say("|---|---|---|---|")
    rows = rows_by["outdoor"]
    full = ols(rows, "y", ["skill", "w", "sw"])
    say(f"| — (full) | {len(rows)} | {full[3]:+.4f} | {full[3]/full[1]:+.4f} |")
    for grp_key in ("tour", "year"):
        for v in sorted({r[grp_key] for r in rows}):
            sub = [r for r in rows if r[grp_key] != v]
            c = ols(sub, "y", ["skill", "w", "sw"])
            say(f"| {grp_key}={v} | {len(sub)} | {c[3]:+.4f} | "
                f"{c[3]/c[1]:+.4f} |")
    say("\n| kept ONLY | n | d | d/b |")
    say("|---|---|---|---|")
    for grp_key in ("tour", "year"):
        for v in sorted({r[grp_key] for r in rows}):
            sub = [r for r in rows if r[grp_key] == v]
            c = ols(sub, "y", ["skill", "w", "sw"])
            if c:
                say(f"| {grp_key}={v} | {len(sub)} | {c[3]:+.4f} | "
                    f"{c[3]/c[1]:+.4f} |")


# =====================================================================
# 4. H1 serve-rate slope (null) LOEO
# =====================================================================
def serve_rate_slope():
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        hourly[(r["event_id"], r["local_time"][:13])] = r
    start_hour = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]
    wx = {(r["event_id"], r["date"]): r
          for r in read_csv(ROOT / "data/event_weather.csv")}
    ev_of = {}
    for g in read_csv(ROOT / "data/games.csv"):
        ev_of.setdefault(g["match_id"], (g["event_id"], g["date"]))
    units = defaultdict(list)
    for r in read_csv(ROOT / "data/match_rally_summary.csv"):
        if r["discipline"] != "doubles" or int(r["n_rallies"]) < 20:
            continue
        ed = ev_of.get(r["match_id"])
        if not ed:
            continue
        ev, date = ed
        if SETTING.get(ev) != "outdoor":
            continue
        hw = hourly.get((ev, start_hour.get(r["match_id"], "")))
        dw = wx.get((ev, date))
        rec = {"nr": int(r["n_rallies"]),
               "rate": int(r["n_points"]) / int(r["n_rallies"])}
        if hw:
            rec["wind_h"] = float(hw["windspeed_10m"])
        if dw and dw["windspeed_10m_max"]:
            rec["wind_d"] = float(dw["windspeed_10m_max"])
        units[ev].append(rec)

    def wls(key):
        def f(s):
            sw = sx = sy = sxx = sxy = 0.0
            for r in s:
                if key not in r:
                    continue
                w, x, y = r["nr"], r[key], r["rate"]
                sw += w; sx += w * x; sy += w * y
                sxx += w * x * x; sxy += w * x * y
            den = sxx - sx * sx / sw if sw else 0.0
            return 10 * (sxy - sx * sy / sw) / den if den else float("nan")
        return f
    say("\n## 4. H1 serve-point-rate slope vs wind, outdoor (published null)")
    flat = [(e, r) for e, rs in units.items() for r in rs]
    jackknife([(e, r) for e, r in flat if "wind_h" in r], wls("wind_h"),
              "serve rate per +10 mph, match-hour wind "
              "(published +0.0030 [−0.0009, +0.0072])")
    jackknife([(e, r) for e, r in flat if "wind_d" in r], wls("wind_d"),
              "serve rate per +10 mph, daily max wind "
              "(published +0.0017 [−0.0024, +0.0061])")


def main():
    say("# B4 (1-2) — leave-one-event-out fragility of every tail-bin "
        "statistic AND of the headline nulls\n")
    say("Every statistic below is recomputed from the committed data with "
        "the SAME joins/labels/bins the published version used "
        "(heuristic event_geo labels, match-hour wind where the published "
        "test used it). 'Δ estimate' is the change when that one event is "
        "deleted. Cluster/CI work lives in the spec-curve script; this "
        "script is about INFLUENCE.\n")
    design_bc()
    fav_bins()
    fav_wind_reg1()
    serve_rate_slope()
    (HERE / "b4_jackknife.md").write_text("\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/b4_jackknife.md")


if __name__ == "__main__":
    main()
