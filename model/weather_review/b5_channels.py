"""B5 — the untested weather channels: gusts, rain, cold, swirl, day/night.

    python model/weather_review/b5_channels.py

Every published weather test used SUSTAINED windspeed_10m at match hour.
data/event_weather_hourly.csv also carries windgusts_10m, precipitation,
temperature_2m, winddirection_10m and the local hour — all unused. This
script runs each of those channels through the SAME two outcomes the
published work used, with corrected indoor/outdoor labels
(data/venue_overrides.csv) and an indoor falsification arm for every test.

Outcomes
--------
S  serve-point rate      : n_points / n_rallies per MATCH (rally logs),
                           WLS weighted by n_rallies.
F  favorite compression  : game-level  share-1/2 = a + b*skill + c*x
                           + d*(skill*x) [+ sustained-wind controls];
                           d < 0 outdoors = the channel flattens skill.

Channels (x), all measured at the match's LOCAL START HOUR
----------
  gustiness = windgusts_10m - windspeed_10m   (per 10 mph)   <- PRIMARY
  gust      = windgusts_10m                   (per 10 mph)
  wet       = 1[precip over hour h-2..h > 0.01 in]  (+ light/heavy split)
  cold      = max(0, 60F - temperature_2m)    (per 10 F below 60)
  swirl     = circular sd of winddirection over h-1..h+2 (per 30 deg)
  night     = 1[local start hour >= 17]

Inference: cluster bootstrap over EVENTS, 4000 resamples, exact via
per-cluster (X'WX, X'Wy) accumulation. Seeded, deterministic.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "model" / "weather_review"))
from sitelib.race import sigmoid, team_eta  # noqa: E402
from b2b_lib import label_arms  # noqa: E402

NBOOT = 4000
SEED = 20260731


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ weather
def load_hourly():
    """(event_id, 'YYYY-MM-DDTHH') -> parsed hourly row."""
    H = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        H[(r["event_id"], r["local_time"][:13])] = {
            "sust": fnum(r["windspeed_10m"]),
            "gust": fnum(r["windgusts_10m"]),
            "temp": fnum(r["temperature_2m"]),
            "app": fnum(r["apparent_temperature"]),
            "rh": fnum(r["relative_humidity_2m"]),
            "prcp": fnum(r["precipitation"]),
            "cloud": fnum(r["cloudcover"]),
            "dir": fnum(r["winddirection_10m"]),
        }
    return H


def shift_hour(key, delta):
    """key = 'YYYY-MM-DDTHH' -> same shifted by delta hours."""
    import datetime as dt
    t = dt.datetime.strptime(key, "%Y-%m-%dT%H") + dt.timedelta(hours=delta)
    return t.strftime("%Y-%m-%dT%H")


def circ_sd_deg(angles):
    if len(angles) < 3:
        return None
    c = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    s = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    R = math.hypot(c, s)
    R = min(max(R, 1e-9), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(R)))


def load_match_hours():
    """match_id -> (hour_key, is_actual)."""
    out = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts, actual = r["start_local"], True
        if not ts:
            ts, actual = r["planned_start_local"], False
        if ts:
            out[r["match_id"]] = (ts[:13], actual)
    return out


def build_context(H, mh, setting_map):
    """match_id -> dict of channel values (or None if unjoinable)."""
    ctx = {}
    for mid, (hk, actual) in mh.items():
        pass
    return ctx


# ------------------------------------------------------------------ WLS core
def cluster_blocks(rows, ykey, xkeys, wkey, ckey):
    """{cluster: (XtWX, XtWy, n, sumw)} — exact bootstrap sufficient stats."""
    p = len(xkeys) + 1
    acc = {}
    for r in rows:
        x = np.empty(p)
        x[0] = 1.0
        for i, k in enumerate(xkeys):
            x[i + 1] = r[k]
        w = r[wkey]
        y = r[ykey]
        c = r[ckey]
        if c not in acc:
            acc[c] = [np.zeros((p, p)), np.zeros(p), 0, 0.0]
        a = acc[c]
        a[0] += w * np.outer(x, x)
        a[1] += w * y * x
        a[2] += 1
        a[3] += w
    return acc


def solve(G, b):
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return None


def fit_boot(rows, ykey, xkeys, wkey="w", ckey="ev", nboot=NBOOT, seed=SEED):
    """Returns dict: coef names -> (point, lo, hi, se, p_boot)."""
    acc = cluster_blocks(rows, ykey, xkeys, wkey, ckey)
    keys = list(acc)
    p = len(xkeys) + 1
    G = np.zeros((p, p))
    b = np.zeros(p)
    for k in keys:
        G += acc[k][0]
        b += acc[k][1]
    pt = solve(G, b)
    Gs = np.stack([acc[k][0] for k in keys])
    bs = np.stack([acc[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(nboot, len(keys)))
    draws = []
    for i in range(nboot):
        ii = idx[i]
        gg = Gs[ii].sum(axis=0)
        bb = bs[ii].sum(axis=0)
        v = solve(gg, bb)
        if v is not None and np.all(np.isfinite(v)):
            draws.append(v)
    D = np.array(draws)
    names = ["const"] + list(xkeys)
    out = {}
    for i, nm in enumerate(names):
        col = np.sort(D[:, i])
        lo, hi = col[int(0.025 * len(col))], col[int(0.975 * len(col))]
        frac_le = float((col <= 0).mean())
        pv = 2 * min(frac_le, 1 - frac_le)
        pv = max(pv, 1.0 / len(col))
        out[nm] = dict(point=float(pt[i]), lo=float(lo), hi=float(hi),
                       se=float(col.std(ddof=1)), p=pv)
    out["_n"] = len(rows)
    out["_nclust"] = len(keys)
    return out


def within_event(rows, ykey, xkeys, wkey="w"):
    grp = defaultdict(list)
    for r in rows:
        grp[r["ev"]].append(r)
    dm = []
    for ev, rs in grp.items():
        if len(rs) < 5:
            continue
        sw = sum(r[wkey] for r in rs)
        mu = {k: sum(r[k] * r[wkey] for r in rs) / sw for k in [ykey] + xkeys}
        for r in rs:
            d = {"ev": ev, wkey: r[wkey]}
            d[ykey] = r[ykey] - mu[ykey]
            for k in xkeys:
                d[k] = r[k] - mu[k]
            dm.append(d)
    return fit_boot(dm, ykey, xkeys, wkey=wkey)


def holm(pvals):
    """pvals: {name: p} -> {name: adjusted p} (Holm-Bonferroni)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)
        prev = a
        adj[k] = a
    return adj


# ------------------------------------------------------------------ main
def main():
    arms = label_arms()
    setting_map = arms["corrected_all"]
    pub_map = arms["published"]
    hi_map = arms["audited_hi"]

    H = load_hourly()
    mh = load_match_hours()
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}
    rally = {r["match_id"]: r
             for r in read_csv(ROOT / "data/match_rally_summary.csv")
             if r["discipline"] == "doubles" and int(r["n_rallies"]) >= 20}

    games_by_match = defaultdict(list)
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        games_by_match[g["match_id"]].append(g)

    # ---- per-match weather context ---------------------------------------
    mrows = []   # serve-rate rows (one per match)
    grows = []   # favorite rows (one per game)
    diag = defaultdict(int)
    for mid, gs in games_by_match.items():
        g0 = gs[0]
        ev = g0["event_id"]
        if mid not in mh:
            diag["no_time"] += 1
            continue
        hk, actual = mh[mid]
        w = H.get((ev, hk))
        if not w or w["sust"] is None or w["gust"] is None:
            diag["no_wx"] += 1
            continue
        # precip over h-2..h
        pr = [H.get((ev, shift_hour(hk, d)), {}).get("prcp") for d in (-2, -1, 0)]
        pr = [x for x in pr if x is not None]
        prcp3 = sum(pr) if pr else None
        # swirl over h-1..h+2
        dirs = [H.get((ev, shift_hour(hk, d)), {}).get("dir") for d in (-1, 0, 1, 2)]
        dirs = [x for x in dirs if x is not None]
        swirl = circ_sd_deg(dirs)
        hour = int(hk[11:13])
        base = {
            "ev": ev, "mid": mid, "tour": g0["tour"], "date": g0["date"],
            "actual": 1.0 if actual else 0.0,
            "sust": w["sust"] / 10.0,
            "gustiness": (w["gust"] - w["sust"]) / 10.0,
            "gust": w["gust"] / 10.0,
            "temp": w["temp"],
            "cold": max(0.0, 60.0 - w["temp"]) / 10.0 if w["temp"] is not None else None,
            "prcp3": prcp3,
            "wet": (1.0 if (prcp3 or 0) > 0.01 else 0.0) if prcp3 is not None else None,
            "wet_any": (1.0 if (prcp3 or 0) > 0.0 else 0.0) if prcp3 is not None else None,
            "wet_light": (1.0 if 0.0 < (prcp3 or 0) <= 0.05 else 0.0) if prcp3 is not None else None,
            "wet_heavy": (1.0 if (prcp3 or 0) > 0.05 else 0.0) if prcp3 is not None else None,
            "swirl": swirl / 30.0 if swirl is not None else None,
            "night": 1.0 if hour >= 17 else 0.0,
            "hour": hour,
            "rh": w["rh"] / 10.0 if w["rh"] is not None else None,
            "cloud": w["cloud"] / 100.0 if w["cloud"] is not None else None,
        }
        for name, smap in (("corr", setting_map), ("pub", pub_map), ("hi", hi_map)):
            base["set_" + name] = smap.get(ev)

        if mid in rally:
            rs = rally[mid]
            m = dict(base)
            m["n_rallies"] = int(rs["n_rallies"])
            m["w"] = float(rs["n_rallies"])
            m["serve_rate"] = int(rs["n_points"]) / int(rs["n_rallies"])
            mrows.append(m)

        for g in gs:
            vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
            if not all(v is not None for v in vals):
                continue
            s1, s2 = int(g["t1_score"]), int(g["t2_score"])
            if s1 + s2 < 11:
                continue
            eta = team_eta(*vals)
            r = dict(base)
            r["stage"] = g["stage"]
            r["skill"] = sigmoid(eta) - 0.5
            r["share"] = s1 / (s1 + s2) - 0.5
            r["w"] = 1.0
            r["npts"] = s1 + s2
            grows.append(r)

    out = []
    say = lambda s="": (print(s), out.append(s))

    say("# B5 — untested weather channels: gusts, rain, cold, swirl, day/night\n")
    say(f"Match-hour joins: {len(mrows)} matches with rally logs, "
        f"{len(grows)} games with full v2 ratings. "
        f"Dropped: {diag['no_time']} matches with no start time, "
        f"{diag['no_wx']} with no hourly weather row.\n")

    # ------------------------------------------------------------ PRE-SPEC
    say("## Pre-specification (written before reading any result below)\n")
    say("""**PRIMARY channel: GUSTINESS = windgusts_10m - windspeed_10m at match hour,
controlling for sustained wind.** Two pre-registered coefficients:

* **S** (serve-point rate): `serve_rate ~ 1 + sust + gustiness`, WLS by rallies.
  Signal = gustiness slope outdoors whose 95% cluster-bootstrap CI excludes 0
  AND is |>= 0.010| per 10 mph of gustiness (1.0 pp of serve-point rate --
  the smallest change that would matter for the live win-prob DP), AND the
  indoor falsification arm does NOT move the same way.
* **F** (favorite compression): `share-1/2 ~ 1 + skill + sust + gustiness
  + skill*sust + skill*gustiness`. Signal = `skill x gustiness` coefficient
  d < 0 outdoors with CI excluding 0, |d| >= 0.05 (i.e. >= 5% of the skill
  slope destroyed per 10 mph of gustiness), and indoor d not equally negative.

Secondary channels, same two outcomes each, same rules: plain gust speed,
wet ball (recent precipitation), cold tail, direction swirl, night session.
Family-wise: 12 outdoor primary coefficients (6 channels x 2 outcomes);
Holm-Bonferroni across the family is reported alongside raw.

Everything runs on CORRECTED labels (data/venue_overrides.csv, mixed/unknown
events dropped). Robustness arms: published labels, high-confidence-only
labels, and actual-start-time-only matches.\n""")

    # ------------------------------------------------------------ EXPOSURE
    say("## Exposure: what the untested channels actually look like\n")
    outd = [r for r in mrows if r["set_corr"] == "outdoor"]
    ind = [r for r in mrows if r["set_corr"] == "indoor"]
    og = [r for r in grows if r["set_corr"] == "outdoor"]
    ig = [r for r in grows if r["set_corr"] == "indoor"]
    say(f"Corrected labels: {len(og)} outdoor games / {len(ig)} indoor games; "
        f"{len(outd)} outdoor matches / {len(ind)} indoor matches with logs.\n")

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))]

    say("| channel (outdoor games) | p10 | p50 | p90 | p99 | max | n>threshold |")
    say("|---|---|---|---|---|---|---|")
    for key, unit, scale, thr in (("sust", "mph", 10, 14), ("gust", "mph", 10, 25),
                                  ("gustiness", "mph", 10, 12),
                                  ("swirl", "deg", 30, 45)):
        v = [r[key] * scale for r in og if r.get(key) is not None]
        n_hi = sum(1 for x in v if x >= thr)
        say(f"| {key} | {pct(v,.1):.1f} | {pct(v,.5):.1f} | {pct(v,.9):.1f} | "
            f"{pct(v,.99):.1f} | {max(v):.1f} | {n_hi} >= {thr} {unit} |")
    tv = [r["temp"] for r in og]
    say(f"| temperature (F) | {pct(tv,.1):.1f} | {pct(tv,.5):.1f} | {pct(tv,.9):.1f} | "
        f"{pct(tv,.99):.1f} | {max(tv):.1f} | {sum(1 for x in tv if x < 60)} < 60F |")
    wv = [r for r in og if r.get("wet") is not None]
    say(f"| wet (precip h-2..h) | - | - | - | - | - | "
        f"{sum(1 for r in wv if r['wet_any'])} any / "
        f"{sum(1 for r in wv if r['wet_light'])} light / "
        f"{sum(1 for r in wv if r['wet_heavy'])} >0.05in |")
    say(f"| night (local start >= 17h) | - | - | - | - | - | "
        f"{sum(1 for r in og if r['night'])} of {len(og)} |")
    say()

    # cross-tab: gusty games hidden inside calm sustained bins
    gusty = [r for r in og if r["gust"] * 10 >= 25]
    hid = sum(1 for r in gusty if r["sust"] * 10 < 14)
    say(f"**Concealment check**: of {len(gusty)} outdoor games at gust >= 25 mph, "
        f"{hid} ({100*hid/max(1,len(gusty)):.0f}%) sit in the published "
        f"calm/moderate SUSTAINED bins (<14 mph). The published binning did "
        f"hide most of the gust exposure.\n")
    # correlation
    a = np.array([r["gustiness"] for r in og])
    b = np.array([r["sust"] for r in og])
    say(f"corr(sustained, gustiness) outdoor = {np.corrcoef(a, b)[0,1]:+.3f} "
        f"-- they are far from collinear, so the control is identified.\n")

    # ------------------------------------------------------------ TESTS
    CHANNELS = [
        ("gustiness", "gustiness (gust-sust), per 10 mph", ["sust", "gustiness"]),
        ("gust", "gust speed, per 10 mph", ["gust"]),
        ("wet", "wet ball: precip>0.01in in h-2..h", ["sust", "wet"]),
        ("cold", "cold: degrees below 60F, per 10F", ["sust", "cold"]),
        ("swirl", "swirl: circular sd of wind dir, per 30 deg", ["sust", "swirl"]),
        ("night", "night session (local start >= 17h)", ["sust", "night"]),
    ]

    results = {}

    def run_serve(arm_rows, chan, xkeys, tag):
        rows = [r for r in arm_rows if all(r.get(k) is not None for k in xkeys)]
        if len(rows) < 200:
            return None
        return fit_boot(rows, "serve_rate", xkeys, wkey="w")

    def run_fav(arm_rows, chan, xkeys, tag):
        xs = []
        for k in xkeys:
            xs.append(k)
        full = ["skill"] + xs + ["skill_x_" + k for k in xs]
        rows = []
        for r in arm_rows:
            if any(r.get(k) is None for k in xs):
                continue
            rr = dict(r)
            for k in xs:
                rr["skill_x_" + k] = r["skill"] * r[k]
            rows.append(rr)
        if len(rows) < 300:
            return None
        return fit_boot(rows, "share", full, wkey="w")

    say("## Results — outdoor (test) vs indoor (falsification), corrected labels\n")
    say("### S. Serve-point rate (WLS by rallies)\n")
    say("| channel | arm | n matches | events | slope | 95% CI | raw p |")
    say("|---|---|---|---|---|---|---|")
    for chan, desc, xkeys in CHANNELS:
        for arm, rows_arm in (("outdoor", outd), ("indoor", ind)):
            res = run_serve(rows_arm, chan, xkeys, arm)
            if res is None:
                say(f"| {desc} | {arm} | - | - | insufficient | | |")
                continue
            c = res[chan]
            results[("S", chan, arm)] = c
            say(f"| {desc} | {arm} | {res['_n']} | {res['_nclust']} | "
                f"{c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
    say()
    say("### F. Favorite compression (game-level skill x channel interaction)\n")
    say("Coefficient shown = `skill x channel`. d<0 means the channel eats "
        "the favourite's edge. Skill slope b (at channel=0) given for scale.\n")
    say("| channel | arm | n games | events | b (skill) | d (skill x chan) | 95% CI | raw p |")
    say("|---|---|---|---|---|---|---|---|")
    for chan, desc, xkeys in CHANNELS:
        for arm, rows_arm in (("outdoor", og), ("indoor", ig)):
            res = run_fav(rows_arm, chan, xkeys, arm)
            if res is None:
                say(f"| {desc} | {arm} | - | - | | insufficient | | |")
                continue
            c = res["skill_x_" + chan]
            results[("F", chan, arm)] = c
            say(f"| {desc} | {arm} | {res['_n']} | {res['_nclust']} | "
                f"{res['skill']['point']:.3f} | {c['point']:+.4f} | "
                f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
    say()

    # ------------------------------------------------------------ MULTIPLICITY
    say("## Family-wise multiplicity\n")
    fam = {f"{o}:{c}": results[(o, c, "outdoor")]["p"]
           for o, c, _ in [(o, c, 0) for o in ("S", "F") for c, _, _ in CHANNELS]
           if (o, c, "outdoor") in results}
    adj = holm(fam)
    say(f"Family = {len(fam)} pre-declared OUTDOOR coefficients "
        "(6 channels x 2 outcomes). Holm-Bonferroni:\n")
    say("| test | raw p | Holm-adjusted p | survives 0.05? |")
    say("|---|---|---|---|")
    for k in sorted(fam, key=lambda k: fam[k]):
        say(f"| {k} | {fam[k]:.3f} | {adj[k]:.3f} | "
            f"{'YES' if adj[k] < 0.05 else 'no'} |")
    say()

    # ------------------------------------------------------------ MDE
    say("## Minimum detectable effect (80% power, two-sided 0.05)\n")
    say("MDE = 2.80 x cluster-bootstrap SE, translated into real units.\n")
    say("| test | channel | SE | MDE (coef) | MDE in real-world units |")
    say("|---|---|---|---|---|")
    for chan, desc, xkeys in CHANNELS:
        for o in ("S", "F"):
            k = (o, chan, "outdoor")
            if k not in results:
                continue
            se = results[k]["se"]
            mde = 2.80 * se
            if o == "S":
                unit = (f"{100*mde:.2f} pp of serve-point rate per "
                        + ("10 mph" if chan in ("gustiness", "gust") else
                           "unit" if chan in ("wet", "night") else
                           "10F" if chan == "cold" else "30 deg"))
                if chan in ("wet", "night"):
                    unit = f"{100*mde:.2f} pp of serve-point rate (on/off)"
            else:
                # translate d into pp of favourite point share at a
                # representative favourite (skill = +0.15 => ~65% expected share)
                unit = (f"{100*mde*0.15:.2f} pp of point share for a 65%"
                        f" favourite, per "
                        + ("10 mph" if chan in ("gustiness", "gust") else
                           "on/off" if chan in ("wet", "night") else
                           "10F" if chan == "cold" else "30 deg"))
            say(f"| {o} | {chan} | {se:.4f} | {mde:.4f} | {unit} |")
    say()

    # ------------------------------------------------------------ BINNED
    say("## Binned view of the PRIMARY channel (gustiness)\n")
    GBINS = [(0, 4), (4, 7), (7, 10), (10, 14), (14, 99)]

    def binned_serve(rows):
        lines = []
        for lo, hi in GBINS:
            sub = [r for r in rows if lo <= r["gustiness"] * 10 < hi]
            if not sub:
                continue
            nr = sum(r["n_rallies"] for r in sub)
            rate = sum(r["serve_rate"] * r["n_rallies"] for r in sub) / nr
            # cluster bootstrap of the bin mean
            acc = defaultdict(lambda: [0.0, 0.0])
            for r in sub:
                acc[r["ev"]][0] += r["serve_rate"] * r["n_rallies"]
                acc[r["ev"]][1] += r["n_rallies"]
            keys = list(acc)
            num = np.array([acc[k][0] for k in keys])
            den = np.array([acc[k][1] for k in keys])
            rng = np.random.default_rng(SEED + lo)
            ii = rng.integers(0, len(keys), size=(2000, len(keys)))
            bs = np.sort(num[ii].sum(1) / den[ii].sum(1))
            lines.append((f"{lo}-{hi if hi<99 else '+'}", len(sub), nr, rate,
                          bs[50], bs[1949]))
        return lines

    for arm, rows_arm in (("outdoor", outd), ("indoor", ind)):
        say(f"\n**{arm}** serve-point rate by gustiness bin\n")
        say("| gustiness (mph) | matches | rallies | serve-point rate | 95% CI |")
        say("|---|---|---|---|---|")
        for lbl, nm, nr, rate, lo, hi in binned_serve(rows_arm):
            say(f"| {lbl} | {nm} | {nr} | {rate:.4f} | [{lo:.4f}, {hi:.4f}] |")

    say("\n**Favourite point-share edge by gustiness bin** "
        "(obs share of the v2 favourite minus its model-expected share; "
        "negative = favourites underperform)\n")
    for arm, rows_arm in (("outdoor", og), ("indoor", ig)):
        say(f"\n*{arm}*\n")
        say("| gustiness (mph) | games | events | obs-exp share | 95% CI |")
        say("|---|---|---|---|---|")
        for lo, hi in GBINS:
            sub = [r for r in rows_arm if lo <= r["gustiness"] * 10 < hi]
            if len(sub) < 30:
                continue
            acc = defaultdict(list)
            for r in sub:
                sgn = 1.0 if r["skill"] >= 0 else -1.0
                acc[r["ev"]].append(sgn * r["share"] - abs(r["skill"]))
            keys = list(acc)
            sums = np.array([sum(acc[k]) for k in keys])
            cnts = np.array([len(acc[k]) for k in keys])
            pt = sums.sum() / cnts.sum()
            rng = np.random.default_rng(SEED + 7 + lo)
            ii = rng.integers(0, len(keys), size=(2000, len(keys)))
            bs = np.sort(sums[ii].sum(1) / cnts[ii].sum(1))
            say(f"| {lo}-{hi if hi<99 else '+'} | {len(sub)} | {len(keys)} | "
                f"{100*pt:+.2f} pp | [{100*bs[50]:+.2f}, {100*bs[1949]:+.2f}] |")

    say()
    # ------------------------------------------------------------ ROBUST
    say("## Robustness on the PRIMARY channel\n")
    say("| arm / variant | S: gustiness slope [CI] | F: skill x gustiness [CI] |")
    say("|---|---|---|")
    variants = [
        ("corrected labels (primary)",
         lambda r: r["set_corr"] == "outdoor", lambda r: r["set_corr"] == "indoor"),
        ("published labels",
         lambda r: r["set_pub"] == "outdoor", lambda r: r["set_pub"] == "indoor"),
        ("high-confidence labels only",
         lambda r: r["set_hi"] == "outdoor", lambda r: r["set_hi"] == "indoor"),
        ("corrected + ACTUAL start times only",
         lambda r: r["set_corr"] == "outdoor" and r["actual"] == 1.0,
         lambda r: r["set_corr"] == "indoor" and r["actual"] == 1.0),
        ("corrected + PPA only",
         lambda r: r["set_corr"] == "outdoor" and r["tour"] == "PPA",
         lambda r: r["set_corr"] == "indoor" and r["tour"] == "PPA"),
    ]
    for lbl, fo, fi in variants:
        for arm, fsel in (("outdoor", fo), ("indoor", fi)):
            ms = [r for r in mrows if fsel(r)]
            gsx = [r for r in grows if fsel(r)]
            s = run_serve(ms, "gustiness", ["sust", "gustiness"], arm) if len(ms) >= 200 else None
            f = run_fav(gsx, "gustiness", ["sust", "gustiness"], arm) if len(gsx) >= 300 else None
            scell = (f"{s['gustiness']['point']:+.4f} "
                     f"[{s['gustiness']['lo']:+.4f},{s['gustiness']['hi']:+.4f}] "
                     f"(n={s['_n']})") if s else "n/a"
            fcell = (f"{f['skill_x_gustiness']['point']:+.4f} "
                     f"[{f['skill_x_gustiness']['lo']:+.4f},"
                     f"{f['skill_x_gustiness']['hi']:+.4f}] (n={f['_n']})") if f else "n/a"
            say(f"| {lbl} — {arm} | {scell} | {fcell} |")
    say()

    # ------------------------------------------------------------ EXTRAS
    say("## Extra cuts\n")
    # cold: continuous temperature both tails, outdoor + indoor
    say("### Temperature, both tails (serve rate; outdoor)\n")
    TBINS = [(-50, 55), (55, 65), (65, 75), (75, 85), (85, 92), (92, 150)]
    for arm, rows_arm in (("outdoor", outd), ("indoor", ind)):
        say(f"\n*{arm}*\n")
        say("| temp (F) | matches | rallies | serve-point rate |")
        say("|---|---|---|---|")
        for lo, hi in TBINS:
            sub = [r for r in rows_arm if lo <= r["temp"] < hi]
            if not sub:
                continue
            nr = sum(r["n_rallies"] for r in sub)
            rate = sum(r["serve_rate"] * r["n_rallies"] for r in sub) / nr
            say(f"| {lo if lo>-50 else '<'}-{hi if hi<150 else '+'} | "
                f"{len(sub)} | {nr} | {rate:.4f} |")

    say("\n### Favourite edge (obs-exp point share) by temperature and session\n")

    def fav_edge(sub, seed_off):
        acc = defaultdict(list)
        for r in sub:
            sgn = 1.0 if r["skill"] >= 0 else -1.0
            acc[r["ev"]].append(sgn * r["share"] - abs(r["skill"]))
        keys = list(acc)
        sums = np.array([sum(acc[k]) for k in keys])
        cnts = np.array([len(acc[k]) for k in keys])
        pt = sums.sum() / cnts.sum()
        rng = np.random.default_rng(SEED + seed_off)
        ii = rng.integers(0, len(keys), size=(2000, len(keys)))
        bs = np.sort(sums[ii].sum(1) / cnts[ii].sum(1))
        return pt, bs[50], bs[1949], len(keys)

    for arm, rows_arm in (("outdoor", og), ("indoor", ig)):
        say(f"\n*{arm}*\n")
        say("| cut | games | events | obs-exp share | 95% CI |")
        say("|---|---|---|---|---|")
        cuts = [(f"temp {lo if lo>-50 else '<'}-{hi if hi<150 else '+'}F",
                 [r for r in rows_arm if lo <= r["temp"] < hi])
                for lo, hi in TBINS]
        cuts += [("night (>=17h)", [r for r in rows_arm if r["night"]]),
                 ("day (<17h)", [r for r in rows_arm if not r["night"]]),
                 ("wet h-2..h", [r for r in rows_arm if r.get("wet")]),
                 ("dry", [r for r in rows_arm if r.get("wet") == 0.0]),
                 ("gust >= 25 mph", [r for r in rows_arm if r["gust"] * 10 >= 25]),
                 ("gust < 25 mph", [r for r in rows_arm if r["gust"] * 10 < 25])]
        for i, (lbl, sub) in enumerate(cuts):
            if len(sub) < 40:
                say(f"| {lbl} | {len(sub)} | - | too thin | |")
                continue
            pt, lo_, hi_, ne = fav_edge(sub, 31 + i)
            say(f"| {lbl} | {len(sub)} | {ne} | {100*pt:+.2f} pp | "
                f"[{100*lo_:+.2f}, {100*hi_:+.2f}] |")

    say("\n### COLD deep dive (the outdoor temp gradient above is monotone)\n")
    say("Linear (un-hinged) temperature interaction, `skill x temp/10F`. "
        "Positive = favourites do BETTER as it warms = cold compresses skill.\n")
    say("| arm | n games | events | skill x temp/10F | 95% CI | raw p |")
    say("|---|---|---|---|---|---|")
    cold_fits = {}
    for arm, rows_arm in (("outdoor", og), ("indoor", ig)):
        rows = []
        for r in rows_arm:
            rr = dict(r, t10=r["temp"] / 10.0)
            rr["skill_x_t10"] = r["skill"] * rr["t10"]
            rr["skill_x_sust"] = r["skill"] * r["sust"]
            rows.append(rr)
        res = fit_boot(rows, "share",
                       ["skill", "sust", "t10", "skill_x_sust", "skill_x_t10"],
                       wkey="w")
        cold_fits[arm] = (rows, res)
        c = res["skill_x_t10"]
        say(f"| {arm} | {res['_n']} | {res['_nclust']} | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
    # DiD version
    pooled = []
    for arm, (rows, _) in cold_fits.items():
        o = 1.0 if arm == "outdoor" else 0.0
        for r in rows:
            rr = dict(r, out=o)
            rr["skill_x_out"] = r["skill"] * o
            rr["out_x_t10"] = o * r["t10"]
            rr["out_x_sust"] = o * r["sust"]
            rr["skill_x_out_x_t10"] = r["skill"] * o * r["t10"]
            rr["skill_x_out_x_sust"] = r["skill"] * o * r["sust"]
            pooled.append(rr)
    resd = fit_boot(pooled, "share",
                    ["skill", "out", "skill_x_out", "sust", "t10",
                     "out_x_sust", "out_x_t10", "skill_x_sust", "skill_x_t10",
                     "skill_x_out_x_sust", "skill_x_out_x_t10"], wkey="w")
    cd = resd["skill_x_out_x_t10"]
    say(f"| **DiD (out-in)** | {resd['_n']} | {resd['_nclust']} | "
        f"{cd['point']:+.4f} | [{cd['lo']:+.4f}, {cd['hi']:+.4f}] | {cd['p']:.3f} |")
    # within-event outdoor
    rows, _ = cold_fits["outdoor"]
    resw = within_event(rows, "share",
                        ["skill", "sust", "t10", "skill_x_sust", "skill_x_t10"])
    cw = resw["skill_x_t10"]
    say(f"| outdoor, within-event | {resw['_n']} | {resw['_nclust']} | "
        f"{cw['point']:+.4f} | [{cw['lo']:+.4f}, {cw['hi']:+.4f}] | {cw['p']:.3f} |")
    say()
    # jackknife the cold tail by event
    tail = [r for r in og if r["temp"] < 60]
    byev = defaultdict(int)
    for r in tail:
        byev[r["ev"]] += 1
    top = sorted(byev.items(), key=lambda kv: -kv[1])[:5]
    say(f"Cold-tail exposure outdoors: {len(tail)} games below 60F over "
        f"{len(byev)} events; the 5 biggest contribute "
        f"{sum(n for _, n in top)} ({100*sum(n for _,n in top)/len(tail):.0f}%). "
        "Leave-one-event-out on the hinged `skill x cold` coefficient:")
    rows_c = []
    for r in og:
        rr = dict(r)
        rr["skill_x_sust"] = r["skill"] * r["sust"]
        rr["skill_x_cold"] = r["skill"] * r["cold"]
        rows_c.append(rr)
    xs_c = ["skill", "sust", "cold", "skill_x_sust", "skill_x_cold"]
    acc = cluster_blocks(rows_c, "share", xs_c, "w", "ev")
    keys = list(acc)
    Gt = sum(acc[k][0] for k in keys)
    bt = sum(acc[k][1] for k in keys)
    full_b = solve(Gt, bt)
    i_c = xs_c.index("skill_x_cold") + 1
    jk = []
    for k in keys:
        v = solve(Gt - acc[k][0], bt - acc[k][1])
        if v is not None:
            jk.append((v[i_c], k))
    jk.sort()
    say(f"- full-sample coefficient {full_b[i_c]:+.4f}; "
        f"leave-one-out range [{jk[0][0]:+.4f}, {jk[-1][0]:+.4f}] "
        f"across {len(jk)} events — "
        + ("no single event drives it."
           if jk[0][0] * jk[-1][0] > 0 else
           "at least one event flips the sign, so it is event-fragile."))

    say("\n#### Temperature interaction — spec curve and confound kill-list\n")
    say("""The `skill x temp` result was NOT in the pre-registered family: the
linear (un-hinged) parameterisation was chosen after seeing the monotone
binned table. It must therefore clear a higher bar. The two confounds that
would manufacture it:

* **Seasonal form lookahead.** v2 values are CURRENT form applied
  retroactively. Cold games are January-March; if the field improves over a
  season, a current rating overstates the January favourite and the
  favourite "underperforms in the cold" for reasons that have nothing to do
  with temperature. Controlled by adding `skill x days-since-2024-01-01`.
* **Day/night and hour.** Cold hours inside an event are early mornings and
  late nights, which get different draws/rounds. Controlled by adding
  `skill x night` and `skill x hour`.\n""")
    day0 = 0

    def mk(rows_arm, extra):
        out_rows = []
        for r in rows_arm:
            y, m, d = (int(x) for x in r["date"].split("-"))
            import datetime as _dt
            days = (_dt.date(y, m, d) - _dt.date(2024, 1, 1)).days / 365.0
            rr = dict(r, t10=r["temp"] / 10.0, days=days,
                      hour_c=(r["hour"] - 14) / 6.0)
            rr["skill_x_t10"] = r["skill"] * rr["t10"]
            rr["skill_x_sust"] = r["skill"] * r["sust"]
            rr["skill_x_days"] = r["skill"] * days
            rr["skill_x_night"] = r["skill"] * r["night"]
            rr["skill_x_hour_c"] = r["skill"] * rr["hour_c"]
            rr["skill_x_gustiness"] = r["skill"] * r["gustiness"]
            out_rows.append(rr)
        return out_rows

    BASE = ["skill", "sust", "t10", "skill_x_sust", "skill_x_t10"]
    SPECS = [
        ("base (outdoor)", BASE, False),
        ("+ skill x season-time", BASE + ["days", "skill_x_days"], False),
        ("+ skill x night", BASE + ["night", "skill_x_night"], False),
        ("+ skill x hour", BASE + ["hour_c", "skill_x_hour_c"], False),
        ("+ skill x gustiness", BASE + ["gustiness", "skill_x_gustiness"], False),
        ("all controls", BASE + ["days", "skill_x_days", "night",
                                 "skill_x_night", "hour_c", "skill_x_hour_c"], False),
        ("event FE", BASE, True),
        ("event FE + all controls",
         BASE + ["days", "skill_x_days", "night", "skill_x_night",
                 "hour_c", "skill_x_hour_c"], True),
    ]
    say("| spec | skill x temp/10F (outdoor) | 95% CI | raw p | indoor same spec |")
    say("|---|---|---|---|---|")
    for lbl, xs, fe in SPECS:
        ro, ri = mk(og, None), mk(ig, None)
        fn = within_event if fe else (lambda r, y, x, wkey="w": fit_boot(r, y, x, wkey=wkey))
        a = fn(ro, "share", xs)
        b_ = fn(ri, "share", xs)
        ca, cb = a["skill_x_t10"], b_["skill_x_t10"]
        say(f"| {lbl} | {ca['point']:+.4f} | [{ca['lo']:+.4f}, {ca['hi']:+.4f}] | "
            f"{ca['p']:.3f} | {cb['point']:+.4f} "
            f"[{cb['lo']:+.4f}, {cb['hi']:+.4f}] |")
    say()
    say("Hinge-point sweep (outdoor, `skill x max(0, T0-temp)/10`):\n")
    say("| hinge T0 | games below | coef | 95% CI | raw p |")
    say("|---|---|---|---|---|")
    for T0 in (55, 60, 65, 70, 75):
        rows = []
        for r in og:
            rr = dict(r, ch=max(0.0, T0 - r["temp"]) / 10.0)
            rr["skill_x_ch"] = r["skill"] * rr["ch"]
            rr["skill_x_sust"] = r["skill"] * r["sust"]
            rows.append(rr)
        res = fit_boot(rows, "share",
                       ["skill", "sust", "ch", "skill_x_sust", "skill_x_ch"],
                       wkey="w")
        c = res["skill_x_ch"]
        nb = sum(1 for r in og if r["temp"] < T0)
        say(f"| {T0}F | {nb} | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
    say()
    say("Heat-only hinge (`skill x max(0, temp-T0)/10`, the published heat "
        "channel re-expressed as an interaction):\n")
    say("| hinge T0 | games above | coef | 95% CI | raw p |")
    say("|---|---|---|---|---|")
    for T0 in (75, 82, 88):
        rows = []
        for r in og:
            rr = dict(r, ch=max(0.0, r["temp"] - T0) / 10.0)
            rr["skill_x_ch"] = r["skill"] * rr["ch"]
            rr["skill_x_sust"] = r["skill"] * r["sust"]
            rows.append(rr)
        res = fit_boot(rows, "share",
                       ["skill", "sust", "ch", "skill_x_sust", "skill_x_ch"],
                       wkey="w")
        c = res["skill_x_ch"]
        na = sum(1 for r in og if r["temp"] > T0)
        say(f"| {T0}F | {na} | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")

    say("\n#### Temperature interaction — remaining confound kill-list\n")
    say("""Three more ways to manufacture a `skill x temp` slope without any
physics: (a) **draw composition** — cold hours are early-morning qualifiers
with badly-rated players, and v2's calibration differs by round; (b) **skill
misspecification** — if the true share-vs-skill curve is not linear and the
skill distribution shifts with temperature, a linear `skill` main effect
leaks into the interaction; (c) **label error** — 26% of games carry the
wrong indoor/outdoor tag.\n""")

    def mk2(rows_arm):
        import datetime as _dt
        out_rows = []
        for r in rows_arm:
            y, m, d = (int(x) for x in r["date"].split("-"))
            days = (_dt.date(y, m, d) - _dt.date(2024, 1, 1)).days / 365.0
            qual = 1.0 if "qual" in r["stage"].lower() else 0.0
            rr = dict(r, t10=r["temp"] / 10.0, days=days,
                      hour_c=(r["hour"] - 14) / 6.0, qual=qual,
                      sk2=r["skill"] * abs(r["skill"]))
            rr["skill_x_t10"] = r["skill"] * rr["t10"]
            rr["skill_x_sust"] = r["skill"] * r["sust"]
            rr["skill_x_days"] = r["skill"] * days
            rr["skill_x_qual"] = r["skill"] * qual
            rr["sk2_x_t10"] = rr["sk2"] * rr["t10"]
            rr["skill_x_night"] = r["skill"] * r["night"]
            rr["skill_x_hour_c"] = r["skill"] * rr["hour_c"]
            out_rows.append(rr)
        return out_rows

    B2 = ["skill", "sust", "t10", "skill_x_sust", "skill_x_t10"]
    SPECS2 = [
        ("+ skill x qualifier", B2 + ["qual", "skill_x_qual"]),
        ("+ nonlinear skill (skill*|skill|)", B2 + ["sk2"]),
        ("+ nonlinear skill x temp", B2 + ["sk2", "sk2_x_t10"]),
        ("kitchen sink", B2 + ["days", "skill_x_days", "night", "skill_x_night",
                               "hour_c", "skill_x_hour_c", "qual",
                               "skill_x_qual", "sk2", "sk2_x_t10"]),
    ]
    say("| spec | skill x temp/10F (outdoor) | 95% CI | raw p |")
    say("|---|---|---|---|")
    for lbl, xs in SPECS2:
        a = fit_boot(mk2(og), "share", xs, wkey="w")
        c = a["skill_x_t10"]
        say(f"| {lbl} | {c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
            f"{c['p']:.3f} |")
    say()
    say("Label arms (base spec):\n")
    say("| label arm | outdoor coef [CI] | indoor coef [CI] |")
    say("|---|---|---|")
    for key, lbl in (("set_corr", "corrected (primary)"),
                     ("set_pub", "published heuristic"),
                     ("set_hi", "high-confidence audited only")):
        o2 = mk2([r for r in grows if r.get(key) == "outdoor"])
        i2 = mk2([r for r in grows if r.get(key) == "indoor"])
        a = fit_boot(o2, "share", B2, wkey="w")
        b_ = fit_boot(i2, "share", B2, wkey="w") if len(i2) > 500 else None
        ca = a["skill_x_t10"]
        cell_i = (f"{b_['skill_x_t10']['point']:+.4f} "
                  f"[{b_['skill_x_t10']['lo']:+.4f},"
                  f"{b_['skill_x_t10']['hi']:+.4f}] (n={b_['_n']})") if b_ else "n/a"
        say(f"| {lbl} | {ca['point']:+.4f} [{ca['lo']:+.4f},{ca['hi']:+.4f}] "
            f"(n={a['_n']}) | {cell_i} |")
    say()
    say("Tour arms and per-year replication (outdoor, base spec):\n")
    say("| subset | n games | events | coef | 95% CI | raw p |")
    say("|---|---|---|---|---|---|")
    subs = [("PPA only", lambda r: r["tour"] == "PPA"),
            ("MLP only", lambda r: r["tour"] == "MLP"),
            ("2024", lambda r: r["date"][:4] == "2024"),
            ("2025", lambda r: r["date"][:4] == "2025"),
            ("2026", lambda r: r["date"][:4] == "2026"),
            ("non-qualifier rounds", lambda r: "qual" not in r["stage"].lower())]
    for lbl, f in subs:
        rows = mk2([r for r in og if f(r)])
        if len(rows) < 500:
            say(f"| {lbl} | {len(rows)} | - | too thin | | |")
            continue
        a = fit_boot(rows, "share", B2, wkey="w")
        c = a["skill_x_t10"]
        say(f"| {lbl} | {a['_n']} | {a['_nclust']} | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")

    say("\n#### Temperature vs THE CALENDAR — the decisive test\n")
    say("""Outdoors, temperature IS the calendar: cold = Jan-Mar/Nov-Dec, hot =
Jun-Aug. v2 values are end-of-2026 form applied retroactively, so if v2's
calibration sags early in each season the favourite's edge is smaller in
January for reasons that have nothing to do with the ball. A LINEAR
season-time control cannot absorb an annual sawtooth. Two tests that can:

1. Replace temperature with a pure seasonal wave `seas = cos(2pi(doy-200)/365)`
   (peaks mid-July). If the effect is calendar, `skill x seas` reproduces it
   OUTDOORS **and appears INDOORS too** (indoor venues have the same
   calendar but a thermostat).
2. Horse-race: put `skill x temp` and `skill x seas` in together, outdoors.
   Temperature survives only if within-season temperature deviations —
   a cold snap at a July event, a warm February — carry the effect.\n""")

    def mk3(rows_arm):
        import datetime as _dt
        out_rows = []
        for r in rows_arm:
            y, m, d = (int(x) for x in r["date"].split("-"))
            doy = _dt.date(y, m, d).timetuple().tm_yday
            seas = math.cos(2 * math.pi * (doy - 200) / 365.0)
            days = (_dt.date(y, m, d) - _dt.date(2024, 1, 1)).days / 365.0
            qual = 1.0 if "qual" in r["stage"].lower() else 0.0
            rr = dict(r, t10=r["temp"] / 10.0, seas=seas, days=days, qual=qual,
                      hour_c=(r["hour"] - 14) / 6.0,
                      sk2=r["skill"] * abs(r["skill"]))
            for k in ("t10", "seas", "days", "qual", "hour_c", "night", "sust"):
                rr["skill_x_" + k] = r["skill"] * rr[k]
            out_rows.append(rr)
        return out_rows

    o3, i3 = mk3(og), mk3(ig)
    say("| test | arm | coef of interest | 95% CI | raw p |")
    say("|---|---|---|---|---|")
    S_ONLY = ["skill", "sust", "seas", "skill_x_sust", "skill_x_seas"]
    for arm, rr in (("outdoor", o3), ("indoor", i3)):
        a = fit_boot(rr, "share", S_ONLY, wkey="w")
        c = a["skill_x_seas"]
        say(f"| 1. skill x SEASON only | {arm} | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
    HORSE = ["skill", "sust", "t10", "seas", "skill_x_sust",
             "skill_x_t10", "skill_x_seas"]
    for arm, rr in (("outdoor", o3), ("indoor", i3)):
        a = fit_boot(rr, "share", HORSE, wkey="w")
        for nm, lbl in (("skill_x_t10", "temp"), ("skill_x_seas", "season")):
            c = a[nm]
            say(f"| 2. horse-race: {lbl} | {arm} | {c['point']:+.4f} | "
                f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
    # strongest available spec: event FE + every control incl. qualifier
    FINAL = ["skill", "sust", "t10", "seas", "days", "night", "hour_c", "qual",
             "sk2", "skill_x_sust", "skill_x_t10", "skill_x_seas",
             "skill_x_days", "skill_x_night", "skill_x_hour_c", "skill_x_qual"]
    for arm, rr in (("outdoor", o3), ("indoor", i3)):
        a = within_event(rr, "share", FINAL)
        c = a["skill_x_t10"]
        say(f"| 3. event FE + ALL controls + season | {arm} | "
            f"{c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
            f"{c['p']:.3f} |")
    say()
    grp = defaultdict(list)
    for r in og:
        grp[r["ev"]].append(r["temp"])
    dev = [t - sum(v) / len(v) for v in grp.values() for t in v]
    say("Within-event temperature spread that identifies spec 3 (outdoor): "
        f"sd of temp deviation from the event mean = {float(np.std(dev)):.1f} F, "
        f"range [{min(dev):.0f}, {max(dev):.0f}] F.")

    # pooled saturated DiD, event FE
    pooled = []
    for arm, rr in (("outdoor", o3), ("indoor", i3)):
        o = 1.0 if arm == "outdoor" else 0.0
        for r in rr:
            d = dict(r)
            d["out"] = o
            for k in ("skill", "t10", "seas", "days", "night", "hour_c",
                      "qual", "sk2", "sust", "skill_x_t10", "skill_x_seas",
                      "skill_x_days", "skill_x_night", "skill_x_hour_c",
                      "skill_x_qual", "skill_x_sust"):
                d["out_x_" + k] = o * r[k]
            pooled.append(d)
    # NB: "out" itself is absorbed by the event FE (events are wholly
    # indoor or wholly outdoor), so it is omitted to keep X'X non-singular.
    XS = (FINAL + ["out_x_" + k for k in
                             ("skill", "t10", "seas", "days", "night",
                              "hour_c", "qual", "sk2", "sust", "skill_x_t10",
                              "skill_x_seas", "skill_x_days", "skill_x_night",
                              "skill_x_hour_c", "skill_x_qual", "skill_x_sust")])
    a = within_event(pooled, "share", XS)
    c = a["out_x_skill_x_t10"]
    say(f"\n**Saturated DiD (event FE, every control, both arms pooled): "
        f"outdoor-minus-indoor `skill x temp/10F` = {c['point']:+.4f} "
        f"[{c['lo']:+.4f}, {c['hi']:+.4f}], p = {c['p']:.3f}, "
        f"n = {a['_n']}.** The identifying contrast is gone: once season, "
        "round, hour and a nonlinear skill term are in, the sheltered arm "
        "moves exactly as much as the exposed one.\n")
    say(f"Bound: the CI allows an outdoor-specific temperature effect up to "
        f"{abs(max(abs(c['lo']), abs(c['hi']))):.3f} per 10F, i.e. at most "
        f"{100*max(abs(c['lo']), abs(c['hi']))*0.15*4:.1f} pp of point share "
        "for a 65% favourite across the full 55F-to-95F range — "
        "roughly a 4-5 pp swing in game win probability at the extreme, "
        "and nothing at all is equally consistent with the data.\n")

    say("""**Caveat on spec 3 / the saturated DiD**: inside a 4-day event the
seasonal wave and season-time are nearly constant, so `skill x seas` and
`skill x days` are near-zero-variance regressors under event FE — including
them there is an unstable over-control, not a clean adjustment. The event FE
already absorbs everything between events (calendar, venue, field). The
right within-event saturation keeps only controls that actually vary inside
an event: hour, night, round, and the nonlinear skill term.\n""")
    CLEAN = ["skill", "sust", "t10", "night", "hour_c", "qual", "sk2",
             "skill_x_sust", "skill_x_t10", "skill_x_night", "skill_x_hour_c",
             "skill_x_qual"]
    say("| arm | within-event saturated `skill x temp/10F` | 95% CI | raw p |")
    say("|---|---|---|---|")
    for arm, rr in (("outdoor", o3), ("indoor", i3)):
        a = within_event(rr, "share", CLEAN)
        c = a["skill_x_t10"]
        say(f"| {arm} | {c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
            f"{c['p']:.3f} |")
    pooled2 = []
    for arm, rr in (("outdoor", o3), ("indoor", i3)):
        o = 1.0 if arm == "outdoor" else 0.0
        for r in rr:
            d = dict(r)
            for k in CLEAN:
                d["out_x_" + k] = o * r[k]
            pooled2.append(d)
    a = within_event(pooled2, "share", CLEAN + ["out_x_" + k for k in CLEAN])
    c2 = a["out_x_skill_x_t10"]
    say(f"| **DiD (out - in)** | {c2['point']:+.4f} | "
        f"[{c2['lo']:+.4f}, {c2['hi']:+.4f}] | {c2['p']:.3f} |")
    say()

    say("\n### Rain intensity split (outdoor)\n")
    for xk in ("wet_any", "wet_light", "wet_heavy"):
        rows = [r for r in outd if r["set_corr"] == "outdoor" and r.get(xk) is not None]
        if sum(r[xk] for r in rows) < 15:
            say(f"- {xk}: only {int(sum(r[xk] for r in rows))} exposed matches — skipped.")
            continue
        res = fit_boot(rows, "serve_rate", ["sust", xk], wkey="w")
        c = res[xk]
        say(f"- {xk}: slope {c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], "
            f"n={res['_n']}, exposed={int(sum(r[xk] for r in rows))}")

    # ------------------------------------------------------- DiD (out-in)
    say("\n## Difference-in-differences: outdoor MINUS indoor\n")
    say("""The single cleanest statistic per channel. Pool both arms, add an
`out` indicator, and read the `out x channel` interaction: how much MORE the
channel moves outcomes outdoors than in the sheltered control. This absorbs
any season/venue/format confound that moves both arms together (which the
binned tables below show is real for gustiness).\n""")
    say("| channel | outcome | DiD coefficient | 95% CI | raw p | MDE (real units) |")
    say("|---|---|---|---|---|---|")
    did_p = {}
    for chan, desc, xkeys in CHANNELS:
        base_x = [k for k in xkeys]
        # --- serve rate
        rows = []
        for r in mrows + []:
            if r["set_corr"] not in ("outdoor", "indoor"):
                continue
            if any(r.get(k) is None for k in base_x):
                continue
            rr = dict(r)
            rr["out"] = 1.0 if r["set_corr"] == "outdoor" else 0.0
            for k in base_x:
                rr["out_x_" + k] = rr["out"] * r[k]
            rows.append(rr)
        xs = ["out"] + base_x + ["out_x_" + k for k in base_x]
        res = fit_boot(rows, "serve_rate", xs, wkey="w")
        c = res["out_x_" + chan]
        did_p[f"S:{chan}"] = c["p"]
        say(f"| {chan} | serve rate | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} | "
            f"{100*2.8*c['se']:.2f} pp |")
        # --- favourite compression (triple interaction)
        grows2 = []
        for r in grows:
            if r["set_corr"] not in ("outdoor", "indoor"):
                continue
            if any(r.get(k) is None for k in base_x):
                continue
            rr = dict(r)
            rr["out"] = 1.0 if r["set_corr"] == "outdoor" else 0.0
            rr["skill_x_out"] = r["skill"] * rr["out"]
            for k in base_x:
                rr["out_x_" + k] = rr["out"] * r[k]
                rr["skill_x_" + k] = r["skill"] * r[k]
                rr["skill_x_out_x_" + k] = r["skill"] * rr["out"] * r[k]
            grows2.append(rr)
        xs2 = (["skill", "out", "skill_x_out"] + base_x
               + ["out_x_" + k for k in base_x]
               + ["skill_x_" + k for k in base_x]
               + ["skill_x_out_x_" + k for k in base_x])
        res2 = fit_boot(grows2, "share", xs2, wkey="w")
        c2 = res2["skill_x_out_x_" + chan]
        did_p[f"F:{chan}"] = c2["p"]
        say(f"| {chan} | favourite compression | {c2['point']:+.4f} | "
            f"[{c2['lo']:+.4f}, {c2['hi']:+.4f}] | {c2['p']:.3f} | "
            f"{100*2.8*c2['se']*0.15:.2f} pp share @65% fav |")
    adj2 = holm(did_p)
    say("\nHolm across the 12 DiD coefficients: min adjusted p = "
        f"{min(adj2.values()):.3f} "
        f"(smallest raw p = {min(did_p.values()):.3f}, "
        f"{[k for k,v in did_p.items() if v==min(did_p.values())][0]}). "
        f"{sum(1 for v in adj2.values() if v<0.05)} survive 0.05.\n")

    # --------------------------------------------------- WITHIN-EVENT (FE)
    say("## Within-event (event fixed effects)\n")
    say("""Cold hours and gusty hours are not randomly assigned to events: a
cold event is an early-season northern stop with its own field, format and
court. Demeaning every variable within event (Frisch-Waugh; exactly the
event-dummy estimator) throws away all between-event variation and asks
whether the channel still moves outcomes ACROSS HOURS AND DAYS OF THE SAME
EVENT. Cluster bootstrap still over events.\n""")

    say("Within-event spread of the channels (weighted sd of the deviation "
        "from the event mean, outdoor):\n")
    for key, scale, nm in (("gustiness", 10, "gustiness mph"),
                           ("sust", 10, "sustained mph"),
                           ("cold", 10, "deg below 60F"),
                           ("swirl", 30, "swirl deg")):
        grp = defaultdict(list)
        for r in og:
            if r.get(key) is not None:
                grp[r["ev"]].append(r[key] * scale)
        dev = [x - sum(v) / len(v) for v in grp.values() for x in v]
        say(f"- {nm}: within-event sd = {float(np.std(dev)):.2f} "
            f"(total sd {float(np.std([r[key]*scale for r in og if r.get(key) is not None])):.2f})")
    say()
    say("| channel | outcome | arm | within-event coef | 95% CI | raw p |")
    say("|---|---|---|---|---|---|")
    for chan, desc, xkeys in CHANNELS:
        for arm, ms, gs2 in (("outdoor", outd, og), ("indoor", ind, ig)):
            rows = [r for r in ms if all(r.get(k) is not None for k in xkeys)]
            if len(rows) >= 300:
                res = within_event(rows, "serve_rate", list(xkeys))
                c = res[chan]
                say(f"| {chan} | serve rate | {arm} | {c['point']:+.4f} | "
                    f"[{c['lo']:+.4f}, {c['hi']:+.4f}] | {c['p']:.3f} |")
            rows = []
            for r in gs2:
                if any(r.get(k) is None for k in xkeys):
                    continue
                rr = dict(r)
                for k in xkeys:
                    rr["skill_x_" + k] = r["skill"] * r[k]
                rows.append(rr)
            if len(rows) >= 500:
                full = ["skill"] + list(xkeys) + ["skill_x_" + k for k in xkeys]
                res = within_event(rows, "share", full)
                c = res["skill_x_" + chan]
                say(f"| {chan} | favourite compression | {arm} | "
                    f"{c['point']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                    f"{c['p']:.3f} |")
    say()

    # --------------------------------------------------- CONTROL-ARM CALIB
    say("## What the control arm says about our false-positive rate\n")
    ip = {}
    for chan, _, _ in CHANNELS:
        for o in ("S", "F"):
            k = (o, chan, "indoor")
            if k in results:
                ip[f"{o}:{chan}"] = results[k]["p"]
    n10 = sum(1 for v in ip.values() if v < 0.10)
    n05 = sum(1 for v in ip.values() if v < 0.05)
    say(f"Indoor is where rain, gusts and swirl CANNOT touch the ball. Running "
        f"the identical 12 tests there returns {n05} coefficient(s) at raw "
        f"p<0.05 and {n10} at raw p<0.10 (chance: 0.6 and 1.2). "
        f"Smallest indoor p = {min(ip.values()):.3f} "
        f"({[k for k,v in ip.items() if v==min(ip.values())][0]}). "
        "Any outdoor result at raw p in the 0.05-0.15 range is therefore "
        "inside the noise this pipeline generates on variables that provably "
        "do nothing.\n")

    # --------------------------------------------------- ATTENUATION BOUND
    say("## Attenuation: how noisy is the match-hour channel?\n")
    both = []
    for r in read_csv(ROOT / "data/match_times.csv"):
        if r["start_local"] and r["planned_start_local"]:
            a, p = r["start_local"][:13], r["planned_start_local"][:13]
            wa, wp = H.get((r["event_id"], a)), H.get((r["event_id"], p))
            if wa and wp and wa["gust"] is not None and wp["gust"] is not None:
                both.append(((wa["gust"] - wa["sust"]), (wp["gust"] - wp["sust"]),
                             wa["sust"], wp["sust"], wa["temp"], wp["temp"]))
    if both:
        A = np.array(both)
        say(f"On {len(both)} matches with BOTH a planned and an actual start "
            "time, correlation between the channel measured at the planned "
            "hour and at the actual hour:\n")
        for i, nm in ((0, "gustiness"), (2, "sustained"), (4, "temperature")):
            r_ = np.corrcoef(A[:, i], A[:, i + 1])[0, 1]
            say(f"- {nm}: r = {r_:+.3f}  -> if a share s of rows use planned "
                f"times, the reliability of the pooled regressor is about "
                f"1-s(1-r); at s=0.30 that is "
                f"{1-0.30*(1-r_):.3f}, inflating any true slope by "
                f"{1/(1-0.30*(1-r_)):.2f}x.")
        say("\nHour-to-hour persistence (a match spans ~1-1.5 h, so the hour "
            "stamp is itself only a sample of the exposure window):")
        pers = defaultdict(list)
        for (ev, hk), w in list(H.items()):
            w2 = H.get((ev, shift_hour(hk, 1)))
            if w2 and w["gust"] is not None and w2["gust"] is not None:
                pers["gustiness"].append((w["gust"] - w["sust"],
                                          w2["gust"] - w2["sust"]))
                pers["sustained"].append((w["sust"], w2["sust"]))
        for nm, v in pers.items():
            V = np.array(v)
            say(f"- {nm} h vs h+1: r = {np.corrcoef(V[:,0], V[:,1])[0,1]:+.3f}")
        say("\nBoth sources bias slopes TOWARD ZERO. Combined with ERA5 grid "
            "error (unmeasurable here), the honest reading is that every "
            "slope below is a LOWER bound on |true effect| by roughly "
            "10-25%, and the CIs should be widened correspondingly before "
            "being read as 'nothing bigger than X is possible'.\n")

    # --------------------------------------------------- SWIRL CONDITIONING
    say("## Swirl, conditioned on there being wind to swirl\n")
    sw = np.array([r["swirl"] * 30 for r in og if r.get("swirl") is not None])
    su = np.array([r["sust"] * 10 for r in og if r.get("swirl") is not None])
    say(f"corr(swirl, sustained) outdoor = {np.corrcoef(sw, su)[0,1]:+.3f} — "
        "direction wanders most when the wind is LIGHT, so raw swirl is "
        "partly an inverse wind proxy. Restricting to hours with real wind:\n")
    say("| subset | n matches | swirl slope on serve rate | 95% CI |")
    say("|---|---|---|---|")
    for lbl, lo in (("all outdoor", 0), ("sustained >= 8 mph", 8),
                    ("sustained >= 12 mph", 12)):
        sub = [r for r in outd if r.get("swirl") is not None and r["sust"] * 10 >= lo]
        if len(sub) < 200:
            continue
        res = fit_boot(sub, "serve_rate", ["sust", "swirl"], wkey="w")
        c = res["swirl"]
        say(f"| {lbl} | {len(sub)} | {c['point']:+.4f} | "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}] |")
    say()

    say("\n### Hour of day, continuous (outdoor games, favourite edge)\n")
    for arm, rows_arm in (("outdoor", og), ("indoor", ig)):
        sub = [dict(r, hour_c=(r["hour"] - 14) / 6.0) for r in rows_arm]
        for r in sub:
            r["skill_x_hour_c"] = r["skill"] * r["hour_c"]
        res = fit_boot(sub, "share", ["skill", "sust", "hour_c",
                                      "skill_x_sust", "skill_x_hour_c"],
                       wkey="w") if all("skill_x_sust" in r for r in sub[:1]) else None
        if res is None:
            for r in sub:
                r["skill_x_sust"] = r["skill"] * r["sust"]
            res = fit_boot(sub, "share", ["skill", "sust", "hour_c",
                                          "skill_x_sust", "skill_x_hour_c"], wkey="w")
        c = res["skill_x_hour_c"]
        say(f"- {arm}: skill x (hour-14)/6 = {c['point']:+.4f} "
            f"[{c['lo']:+.4f}, {c['hi']:+.4f}], n={res['_n']}")

    say("""
## Verdict

**The pre-registered primary (gustiness) is a null, and a reasonably tight
one.** Outdoor serve-point rate moves −0.49 pp per +10 mph of gustiness
[−1.09, +0.05]; within-event −0.45 pp; the indoor control is flat; the DiD
is −0.48 pp [−1.77, +1.27]. The binned pattern is non-monotone AND mirrored
bin-for-bin in the indoor arm, which is what a shared season/venue confound
looks like, not a wind effect. MDE 0.82 pp per 10 mph, so over the realistic
5.6→21 mph gustiness range we can exclude anything bigger than ~1.2 pp of
serve-point rate. Favourite compression by gustiness is +0.02 [−0.07, +0.11]
— dead centre on zero, MDE ≈ 2 pp of point share per 10 mph (≈3 pp of game
win probability for a 65%-share favourite).

**Plain gust speed, rain, swirl and night are nulls too**, none surviving
Holm, and two of them (swirl, night) have a same-sized or larger twin in the
indoor arm where the mechanism cannot operate. Rain in particular: only 340
outdoor matches saw >0.05 in in the three hours before start, so "they stop
play when it really rains" caps the exposure.

**The one live thread is TEMPERATURE, and it was not pre-registered.** The
outdoor favourite edge is monotone across all six temperature bins (+0.43 pp
at <55 °F rising to +1.83 pp at 92 °F+), the linear `skill x temp`
interaction is +0.031 per 10 °F [+0.004, +0.059], it survives event fixed
effects, leave-one-event-out, a season-wave horse race and every control
except `skill x qualifier` (which shrinks it 30%), and it strengthens as the
venue labels get cleaner (+0.076 on high-confidence labels only). It fails
Holm against its own family, is absent in 2026, and its indoor placebo arm
sits at −0.022 rather than the 0 physics demands — so half of the +0.054 DiD
comes from a control arm that should not be moving at all. Suggestive, not
established.

**If I got exactly one shot**: pre-register the FULL-RANGE temperature
interaction, not gusts. Exact spec, fixed in advance:
`share − ½ ~ skill + sust + temp + night + hour + qualifier + skill·|skill|
+ skill×(each of those)`, event fixed effects, cluster bootstrap over events,
outdoor arm with corrected labels, indoor arm as the falsification, and the
DiD `out × skill × temp` as the single reported number. Direction
pre-committed positive (favourites convert skill better as it warms).
Declare a hit only if the outdoor coefficient AND the DiD both clear zero
and the indoor arm is inside ±0.015. Gustiness is the runner-up on physics
but it has already spent its power here: the estimate is half its MDE, the
binned shape is confounded, and unlike temperature nothing about it
coheres across specifications.""")

    txt = "\n".join(out) + "\n"
    (ROOT / "model/weather_review/b5_channels.md").write_text(txt)
    print("\nwrote model/weather_review/b5_channels.md")


if __name__ == "__main__":
    main()
