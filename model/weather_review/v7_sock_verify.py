"""V7 adversarial verification of A7's named-player wind claims.

Independent re-derivation (written from the spec in model/wind_skill.py's
docstring, NOT copied from a7_provenance_wind_names.py) of the per-player
wind slopes for Jack Sock, Anna Leigh Waters, Tyra Black and Anna Bright,
plus THREE inference routes the published work never ran:

  R1  within-player game permutation  (the wind_skill.py route)
  R2  analytic OLS t / z              (closed form; no RNG at all)
  R3  event-CLUSTER inference         (cluster-robust se + a cluster
      permutation that keeps each event's residuals together)

R3 is the load-bearing one: a player's games are clustered in events,
every event has ONE wind value shared by all its games, and in doubles a
single game contributes the SAME residual to both partners.  Shuffling
wind across a player's games (R1) treats those games as exchangeable and
will understate the null spread if a player's form clusters by event.

Also: multiplicity context (where Sock's |z| sits among the 552 scanned
players, and how many players clear his nominal p).

Deterministic: every RNG seeded.  Read-only; writes nothing to data/.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402

MIN_GAMES = 40


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load():
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    v2rows = read_csv(ROOT / "data/v2_players.csv")
    v2 = {r["player_id"]: float(r["value_now_mean"]) for r in v2rows}
    names = {r["player_id"]: r["full_name"] for r in v2rows}
    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = float(r["windspeed_10m"])
        except (TypeError, ValueError):
            pass
    start_hour = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]

    # pid -> list of (wind, signed_resid, event_id, match_id, date)
    players = defaultdict(list)
    n_games = 0
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        if geo.get(g["event_id"]) != "outdoor":
            continue
        wind = hourly.get((g["event_id"], start_hour.get(g["match_id"], "")))
        if wind is None:
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        resid = s1 / (s1 + s2) - sigmoid(team_eta(*vals))
        n_games += 1
        for k in ("t1_p1", "t1_p2"):
            players[g[k]].append((wind, resid, g["event_id"], g["match_id"], g["date"]))
        for k in ("t2_p1", "t2_p2"):
            players[g[k]].append((wind, -resid, g["event_id"], g["match_id"], g["date"]))
    players = {p: v for p, v in players.items() if len(v) >= MIN_GAMES}
    return players, names, n_games


def slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, mx, my, den


def ols_t(xs, ys):
    """Closed-form OLS slope, iid se, t."""
    b, mx, my, sxx = slope(xs, ys)
    n = len(xs)
    ssr = sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(ssr / (n - 2) / sxx)
    return b, se, b / se, n


def cluster_se(xs, ys, clusters):
    """CR1 cluster-robust se on the OLS slope (clusters = event ids)."""
    b, mx, my, sxx = slope(xs, ys)
    by = defaultdict(float)
    for x, y, c in zip(xs, ys, clusters):
        e = y - my - b * (x - mx)
        by[c] += (x - mx) * e
    meat = sum(v * v for v in by.values())
    G = len(by)
    n = len(xs)
    adj = (G / (G - 1)) * ((n - 1) / (n - 2)) if G > 1 else 1.0
    var = adj * meat / (sxx ** 2)
    return b, math.sqrt(var), G


def perm_within(xs, ys, seed, nperm=20000):
    """R1: shuffle wind within player (games exchangeable)."""
    rng = random.Random(seed)
    b0 = slope(xs, ys)[0]
    w = list(xs)
    ge = 0
    draws = []
    for _ in range(nperm):
        rng.shuffle(w)
        s = slope(w, ys)[0]
        draws.append(s)
        ge += s >= b0
    draws.sort()
    lo = draws[int(0.025 * nperm)]
    hi = draws[int(0.975 * nperm)]
    return b0, (ge + 1) / (nperm + 1), lo, hi


def perm_cluster(rows, seed, nperm=20000):
    """R3: permute the WIND LABEL across the player's events, keeping every
    event's residual block intact.  This is the exchangeability the design
    actually supports: wind is assigned at the event level."""
    rng = random.Random(seed)
    ev = defaultdict(list)
    for w, y, e, *_ in rows:
        ev[e].append((w, y))
    keys = list(ev.keys())
    # each event's own wind values (may vary by match hour within event)
    blocks = [[w for w, _ in ev[k]] for k in keys]
    resid = [[y for _, y in ev[k]] for k in keys]
    xs = [w for b in blocks for w in b]
    ys = [y for r in resid for y in r]
    b0 = slope(xs, ys)[0]
    ge = 0
    draws = []
    order = list(range(len(keys)))
    for _ in range(nperm):
        rng.shuffle(order)
        px, py = [], []
        for slot, src in enumerate(order):
            wb = blocks[src]
            rb = resid[slot]
            # recycle/trim the donor event's winds onto this event's residuals
            for i, y in enumerate(rb):
                px.append(wb[i % len(wb)])
                py.append(y)
        s = slope(px, py)
        if s is None:
            continue
        draws.append(s[0])
        ge += s[0] >= b0
    draws.sort()
    return b0, (ge + 1) / (len(draws) + 1), draws[int(0.025 * len(draws))], \
        draws[int(0.975 * len(draws))], len(keys)


def main():
    players, names, n_games = load()
    print(f"outdoor games with match-hour wind + full ratings: {n_games}")
    print(f"players with >= {MIN_GAMES}: {len(players)}")

    # ---- population z scan (independent of wind_skill.py's own printout)
    zs = {}
    for pid, rows in players.items():
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        if len(xs) < 20 or sum((x - sum(xs) / len(xs)) ** 2 for x in xs) < 1e-9:
            continue
        zs[pid] = ols_t(xs, ys)

    targets = ["Jack Sock", "Anna Leigh Waters", "Tyra Hurricane Black",
               "Anna Bright"]
    print("\n=== per-player, three inference routes ===")
    results = {}
    for nm in targets:
        pid = next((p for p, x in names.items() if x == nm), None)
        if pid is None or pid not in players:
            print(f"{nm}: NOT FOUND / below cutoff")
            continue
        rows = players[pid]
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        evs = [r[2] for r in rows]
        b, se, t, n = ols_t(xs, ys)
        _, cse, G = cluster_se(xs, ys, evs)
        ct = b / cse
        # multiple seeds for the within-player permutation
        ps = [perm_within(xs, ys, sd, 20000)[1] for sd in (1, 2, 3)]
        b0, pw, lo, hi = perm_within(xs, ys, 1, 20000)
        bc, pc, clo, chi, nev = perm_cluster(rows, 11, 20000)
        wmin, wmax = min(xs), max(xs)
        print(f"\n{nm}  n_games={n}  n_events={G}  wind range {wmin:.1f}-{wmax:.1f} mph")
        print(f"  slope           {b*10:+.4f} share / +10 mph")
        print(f"  iid   se {se*10:.4f}  t={t:+.2f}  one-sided p={1-norm_cdf(t):.4f}")
        print(f"  CLUSTER(event) se {cse*10:.4f}  t={ct:+.2f}  "
              f"one-sided p={1-norm_cdf(ct):.4f}   (se ratio {cse/se:.2f}x)")
        print(f"  R1 within-player perm p={pw:.4f} (seeds {['%.4f'%q for q in ps]}) "
              f"null 95% [{lo*10:+.4f},{hi*10:+.4f}]")
        print(f"  R3 event-cluster perm p={pc:.4f}  null 95% "
              f"[{clo*10:+.4f},{chi*10:+.4f}]  ({nev} events)")
        results[nm] = dict(b=b, n=n, t=t, ct=ct, pw=pw, pc=pc, G=G)

    # ---- multiplicity context for Sock
    pid_sock = next((p for p, x in names.items() if x == "Jack Sock"), None)
    if pid_sock in zs:
        zsock = zs[pid_sock][2]
        allz = sorted((abs(v[2]) for v in zs.values()), reverse=True)
        rank = 1 + sum(1 for a in allz if a > abs(zsock))
        n_more = sum(1 for v in zs.values() if v[2] >= zsock)
        p1 = 1 - norm_cdf(zsock)
        print("\n=== multiplicity context ===")
        print(f"players scanned: {len(zs)}")
        print(f"Sock t = {zsock:+.3f}; |t| rank {rank} of {len(zs)}")
        print(f"players with t >= Sock's: {n_more}")
        print(f"players with |t| > 1.96: "
              f"{sum(1 for v in zs.values() if abs(v[2]) > 1.96)} "
              f"(expected under null ~{0.05*len(zs):.0f})")
        print(f"players with one-sided p <= Sock's ({p1:.3f}) by t: "
              f"{n_more} (expected ~{p1*len(zs):.1f})")
        print(f"Bonferroni-corrected p for Sock: {min(1.0, p1*len(zs)):.2f}")
        print(f"max |t| in scan: {allz[0]:.2f}")

    # ---- jackknife Sock by event: is the slope one tournament?
    if pid_sock and pid_sock in players:
        rows = players[pid_sock]
        ev = defaultdict(list)
        for r in rows:
            ev[r[2]].append(r)
        base = slope([r[0] for r in rows], [r[1] for r in rows])[0]
        print("\n=== Sock: leave-one-EVENT-out jackknife ===")
        print(f"full slope {base*10:+.4f}")
        drops = []
        for k in ev:
            keep = [r for r in rows if r[2] != k]
            if len(keep) < 20:
                continue
            s = slope([r[0] for r in keep], [r[1] for r in keep])
            if s:
                drops.append((s[0] * 10, k, len(ev[k])))
        drops.sort()
        for s, k, ng in drops[:3]:
            print(f"  drop {k[:8]} (n={ng}): {s:+.4f}")
        print("  ...")
        for s, k, ng in drops[-3:]:
            print(f"  drop {k[:8]} (n={ng}): {s:+.4f}")
        wind_by_ev = {k: sum(r[0] for r in v) / len(v) for k, v in ev.items()}
        res_by_ev = {k: sum(r[1] for r in v) / len(v) for k, v in ev.items()}
        pts = sorted(((wind_by_ev[k], res_by_ev[k], len(ev[k])) for k in ev))
        print("  event-mean (wind, mean resid, n):")
        for w, r, n in pts:
            print(f"    {w:5.1f} mph  {r:+.4f}  n={n}")


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


if __name__ == "__main__":
    main()
