"""VERIFIER 3 — independent bound on the TIMING attenuation lambda_T, using
the hourly-wind autocorrelation of the ERA5 series at the venue-hours that
actually host games.  This route needs NO per-game end stamps, so it is
independent of b6_attenuation's game-hour reconstruction.

lambda(k) = cov(w_h, w_{h+k}) / var(w_h) is the exact attenuation a regression
on w_h suffers if the outcome truly depends on w_{h+k}.  Any mixture of lags
gives lambda = sum_k pi_k lambda(k) (cov is linear in the second argument).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_b6_frame import build  # noqa: E402

DATA = Path(__file__).resolve().parent.parent.parent / "data"

hourly = {}
for r in csv.DictReader(open(DATA / "event_weather_hourly.csv")):
    try:
        hourly[(r["event_id"].lower(), r["local_time"][:13])] = \
            float(r["windspeed_10m"])
    except (TypeError, ValueError):
        pass


def shift(hk, n):
    return (datetime.strptime(hk, "%Y-%m-%dT%H")
            + timedelta(hours=n)).strftime("%Y-%m-%dT%H")


G = [g for g in build() if g["wind"] is not None and g["hk"]
     and g["setting"] == "outdoor"]
print("outdoor games with an hour key:", len(G))

base = np.array([g["wind"] for g in G])
print(f"var(w_meas) over the real outdoor sample = {base.var():.2f}")

print("\nlag  n     lambda(k)=cov(w_h,w_h+k)/var(w_h)   var(w_h - w_h+k)")
lams = {}
for k in (-3, -2, -1, 0, 1, 2, 3):
    a, b = [], []
    for g in G:
        v = hourly.get((g["event"], shift(g["hk"], k)))
        if v is None:
            continue
        a.append(g["wind"])
        b.append(v)
    a = np.array(a)
    b = np.array(b)
    lam = float(np.cov(a, b, bias=True)[0, 1] / a.var())
    lams[k] = lam
    print(f"{k:+d}   {len(a):6d}  {lam:.4f}                        {(a-b).var():.2f}")

# what mixture of lags would be needed to drag lambda_T below 0.90 / 0.80 ?
print("\nMixtures (pi_0 at lag 0, rest split evenly over +-1h / +-2h):")
for p0 in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
    for spread in (1, 2):
        lam = p0 * lams[0] + (1 - p0) * 0.5 * (lams[spread] + lams[-spread])
        print(f"  pi_0={p0:.1f} lag=+-{spread}h -> lambda_T = {lam:.3f}")

# planned-vs-actual channel, recomputed independently
rows = list(csv.DictReader(open(DATA / "match_times.csv")))
pa = []
for r in rows:
    if r["start_local"] and r["planned_start_local"]:
        ev = r["event_id"].lower()
        wa = hourly.get((ev, r["start_local"][:13]))
        wp = hourly.get((ev, r["planned_start_local"][:13]))
        if wa is not None and wp is not None:
            pa.append((wp, wa))
wp = np.array([x[0] for x in pa])
wa = np.array([x[1] for x in pa])
print(f"\nplanned-vs-actual: n={len(pa)}  lambda="
      f"{float(np.cov(wp,wa,bias=True)[0,1]/wp.var()):.4f}  "
      f"var(diff)={float((wp-wa).var()):.2f}  var(wp)={float(wp.var()):.2f}")
d = np.abs(np.array([
    (datetime.fromisoformat(r["start_local"].replace('Z', ''))
     - datetime.fromisoformat(r["planned_start_local"].replace('Z', '')))
    .total_seconds() / 3600
    for r in rows if r["start_local"] and r["planned_start_local"]]))
print("planned-vs-actual |delay| hours: mean %.2f  median %.2f  "
      ">=1h %.1f%%  >=2h %.1f%%" % (d.mean(), np.median(d),
                                    100 * (d >= 1).mean(), 100 * (d >= 2).mean()))
