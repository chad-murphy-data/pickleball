"""TASK B1 — the never-run continuous HEAT test, at match-hour resolution.

    python model/weather_review/heat_test.py            # ~2 min
    python model/weather_review/heat_test.py --quick    # fewer bootstrap draws

The published weather thread tested WIND continuously (model/favorites_wind.py)
but heat only via thin binned cuts ("favorites drift more negative at 92F+").
This is the continuous analogue, with the falsification arm that heat deserves
more than wind does (indoor venues are climate-controlled).

PRE-SPECIFIED (exertion physiology):
  H1 heat is a leveler  -> d < 0 in  share ~ a + b*skill + c*heat + d*skill*heat
                           OUTDOOR, and d_indoor ~ 0 (bigger outdoor-indoor
                           contrast than wind showed).
  H2 rally-level        -> favorite-minus-underdog serve-rally-win gap falls
                           with heat outdoors (c < 0); same sign in the
                           binomial-logit interaction.
  H3 fatigue lengthens  -> more rallies / more points / more 3-game matches at
                           fixed skill gap and fixed format, OUTDOOR.
  H4 not just clock     -> H1/H3 survive an hour-of-day control (skill x hour
                           and hour, hour^2 in the model).
Secondary channels: apparent_temperature (heat index), relative humidity,
precipitation.

Units: temperature in F, heat regressor h = (T - 75)/10, so coefficients read
"per +10 F above 75 F". All inference is a cluster bootstrap over EVENTS.
Venue labels: data/venue_overrides.csv (web-audited) first, event_geo heuristic
only for events the audit never covered; a heuristic-labels-only rerun is
printed as a sensitivity.
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
from sitelib.race import sigmoid, team_eta, game_win_prob  # noqa: E402

QUICK = "--quick" in sys.argv
NBOOT = 300 if QUICK else 1500
NBOOT_LOGIT = 80 if QUICK else 250
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def read_csv(path):
    with open(ROOT / path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- linear algebra
def _solve(xtx, xty, p):
    m = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r_: abs(m[r_][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-11:
            return None
        for r_ in range(p):
            if r_ != col:
                f = m[r_][col] / m[col][col]
                for c_ in range(col, p + 1):
                    m[r_][c_] -= f * m[col][c_]
    return [m[i][p] / m[i][i] for i in range(p)]


def _suff(rows, ykey, xkeys):
    """Per-cluster (X'X, X'y) sufficient statistics — OLS is a sum over them,
    so the cluster bootstrap is exact and costs O(#clusters) per draw."""
    p = len(xkeys) + 1
    acc = {}
    for r in rows:
        ev = r["ev"]
        a = acc.get(ev)
        if a is None:
            a = acc[ev] = ([[0.0] * p for _ in range(p)], [0.0] * p)
        xtx, xty = a
        x = [1.0] + [r[k] for k in xkeys]
        y = r[ykey]
        for i in range(p):
            xi = x[i]
            if xi:
                xty[i] += xi * y
                row = xtx[i]
                for j in range(p):
                    row[j] += xi * x[j]
    return acc, p


def ols(rows, ykey, xkeys):
    acc, p = _suff(rows, ykey, xkeys)
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for a, b in acc.values():
        for i in range(p):
            xty[i] += b[i]
            for j in range(p):
                xtx[i][j] += a[i][j]
    return _solve(xtx, xty, p)


def boot(rows, ykey, xkeys, n=NBOOT, seed=17):
    """Cluster bootstrap over EVENTS. Returns ([(lo, hi)...], n_events)."""
    acc, p = _suff(rows, ykey, xkeys)
    keys = list(acc)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        xtx = [[0.0] * p for _ in range(p)]
        xty = [0.0] * p
        for _ in keys:
            a, b = acc[keys[rng.randrange(len(keys))]]
            for i in range(p):
                xty[i] += b[i]
                ri, ai = xtx[i], a[i]
                for j in range(p):
                    ri[j] += ai[j]
        bb = _solve(xtx, xty, p)
        if bb:
            draws.append(bb)
    cis = []
    for i in range(p):
        v = sorted(d[i] for d in draws)
        cis.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))]))
    return cis, len(keys)


def demean(rows, keys, cellkey="cell"):
    """Within-cell demean the given numeric keys (= cell fixed effects)."""
    sums = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    for r in rows:
        c = r[cellkey]
        cnt[c] += 1
        for k in keys:
            sums[c][k] += r[k]
    out = []
    for r in rows:
        c = r[cellkey]
        if cnt[c] < 2:
            continue
        q = dict(r)
        for k in keys:
            q[k] = r[k] - sums[c][k] / cnt[c]
        out.append(q)
    return out


def fmt(b, ci, prec=4):
    return f"{b:+.{prec}f} [{ci[0]:+.{prec}f}, {ci[1]:+.{prec}f}]"


# ---------------------------------------------------------------- data loading
def load():
    ov = {r["event_id"]: r for r in read_csv("data/venue_overrides.csv")}
    geo = {r["event_id"]: r for r in read_csv("data/event_geo.csv")}
    setting_audit, setting_heur = {}, {}
    for eid, r in geo.items():
        setting_heur[eid] = r["setting"]
        o = ov.get(eid)
        setting_audit[eid] = o["setting"] if o else r["setting"]
    hourly = {}
    for r in read_csv("data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = {
                "temp": float(r["temperature_2m"]),
                "app": float(r["apparent_temperature"]),
                "rh": float(r["relative_humidity_2m"]),
                "precip": float(r["precipitation"]),
                "wind": float(r["windspeed_10m"]),
            }
        except (TypeError, ValueError):
            pass
    start = {}
    for r in read_csv("data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start[r["match_id"]] = (ts[:13], bool(r["start_local"]))
    v2 = {r["player_id"]: float(r["value_now_mean"])
          for r in read_csv("data/v2_players.csv")}
    return setting_audit, setting_heur, hourly, start, v2


def build_games(setting_map, hourly, start, v2, tempkey="temp"):
    """One row per GAME with weather at match hour, plus per-match metadata."""
    rows, meta, per_match = [], {}, defaultdict(list)
    for g in read_csv("data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        st = setting_map.get(g["event_id"])
        sh = start.get(g["match_id"])
        if not sh:
            continue
        w = hourly.get((g["event_id"], sh[0]))
        if w is None:
            continue
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        eta = team_eta(*vals)
        hour = int(sh[0][11:13])
        rows.append({
            "ev": g["event_id"], "match_id": g["match_id"], "setting": st,
            "y": s1 / (s1 + s2) - 0.5,
            "skill": sigmoid(eta) - 0.5,
            "eta": eta, "hour": hour, "actual": sh[1],
            "T": w[tempkey], "rh": w["rh"], "precip": w["precip"],
            "wind": w["wind"], "app": w["app"], "temp": w["temp"],
            "tour": g["tour"],
            "stratum": f'{g["tour"]}|{g["scoring_format"]}|{g["best_of"]}',
            "npts": s1 + s2,
        })
        meta[g["match_id"]] = (st, w, eta, g["event_id"], hour)
        per_match[g["match_id"]].append(g)
    return rows, meta, per_match


# ---------------------------------------------------------------- reg 1
def reg1(rows_by, label, tempkey="T", extra_note=""):
    say(f"### {label}\n")
    say("| setting | games | events | b (skill slope) | d (skill x heat) "
        "per +10F [95% CI] | c (heat main) | slope at 95F |")
    say("|---|---|---|---|---|---|---|")
    res = {}
    for setting in ("outdoor", "indoor"):
        rows = rows_by.get(setting) or []
        if len(rows) < 200:
            continue
        for r in rows:
            r["h"] = (r[tempkey] - 75.0) / 10.0
            r["sh"] = r["skill"] * r["h"]
        b = ols(rows, "y", ["skill", "h", "sh"])
        cis, nev = boot(rows, "y", ["skill", "h", "sh"])
        say(f"| {setting} | {len(rows)} | {nev} | {b[1]:.3f} | "
            f"{fmt(b[3], cis[3], 3)} | {fmt(b[2], cis[2], 4)} | "
            f"{b[1] + 2.0 * b[3]:.3f} |")
        res[setting] = (b, cis, len(rows), nev)
    say("")
    if extra_note:
        say(extra_note + "\n")
    return res


def main():
    setting_audit, setting_heur, hourly, start, v2 = load()
    say("# TASK B1 — continuous HEAT test at match-hour resolution\n")
    say("Pre-specified (exertion physiology): heat is a LEVELER "
        "(d < 0 outdoors, ~0 indoors) and heat LENGTHENS matches "
        "(more rallies/points/3-game matches at fixed skill gap).\n")

    rows, meta, per_match = build_games(setting_audit, hourly, start, v2)
    by = defaultdict(list)
    for r in rows:
        by[r["setting"]].append(r)
    temps = sorted(r["T"] for r in by["outdoor"])
    n = len(temps)
    say(f"Games with a match-hour weather join: {len(rows)} "
        f"(outdoor {len(by['outdoor'])}, indoor {len(by['indoor'])}, "
        f"mixed {len(by['mixed'])}, unknown {len(by['unknown'])}).")
    say(f"Outdoor temperature (F): min {temps[0]:.0f}, p10 "
        f"{temps[n//10]:.0f}, median {temps[n//2]:.0f}, p90 "
        f"{temps[9*n//10]:.0f}, max {temps[-1]:.0f}; "
        f"{sum(1 for t in temps if t >= 90)} games at 90F+, "
        f"{sum(1 for t in temps if t >= 95)} at 95F+.")
    act = sum(1 for r in rows if r["actual"]) / len(rows)
    say(f"Match-hour joins using an ACTUAL start time: {act:.0%} "
        "(rest use planned start — classical measurement error, which "
        "attenuates every slope below toward zero).\n")

    # ---------------- 1. game level, pooled -----------------------------
    say("## 1. Game level: share - 1/2 = a + b*skill + c*h + d*skill*h\n")
    say("h = (temperature_2m at match hour - 75F)/10. d is the test: "
        "negative = heat compresses the favorite's conversion of skill "
        "into points.\n")
    r1 = reg1({k: [dict(r) for r in v] for k, v in by.items()},
              "1a. Pooled (audited venue labels), temperature_2m")

    # within-event fixed effects
    ev_rows = defaultdict(list)
    for r in rows:
        q = dict(r)
        q["h"] = (q["T"] - 75.0) / 10.0
        q["sh"] = q["skill"] * q["h"]
        q["cell"] = q["ev"]
        ev_rows[q["setting"]].append(q)
    say("### 1b. Within-EVENT (event fixed effects; identified off "
        "hour-to-hour and day-to-day heat swings inside one event, so no "
        "venue/season/field confound can enter)\n")
    say("| setting | games | events | b (skill) | d (skill x heat) [95% CI] |")
    say("|---|---|---|---|---|")
    r1b = {}
    for setting in ("outdoor", "indoor"):
        dm = demean(ev_rows[setting], ["y", "skill", "h", "sh"])
        b = ols(dm, "y", ["skill", "h", "sh"])
        cis, nev = boot(dm, "y", ["skill", "h", "sh"])
        say(f"| {setting} | {len(dm)} | {nev} | {b[1]:.3f} | "
            f"{fmt(b[3], cis[3], 3)} |")
        r1b[setting] = (b, cis, len(dm))
    say("")

    # apparent temperature
    say("### 1c. Same, using APPARENT temperature (heat index: temp + "
        "humidity + wind + radiation — the exertion-relevant variable)\n")
    app_rows = {}
    for setting in ("outdoor", "indoor"):
        app_rows[setting] = [dict(r, T=r["app"]) for r in by[setting]]
    r1c = reg1(app_rows, "apparent_temperature, pooled")

    # heuristic-label sensitivity
    rows_h, _, _ = build_games(setting_heur, hourly, start, v2)
    byh = defaultdict(list)
    for r in rows_h:
        byh[r["setting"]].append(r)
    say("### 1d. Sensitivity: the OLD heuristic venue labels (what every "
        "published test used; ~26% of games are mislabeled)\n")
    r1d = reg1(byh, "heuristic labels, temperature_2m")

    # ---------------- 2. hour-of-day control ----------------------------
    say("## 2. Is it heat, or is it the afternoon? (H4)\n")
    say("Adds hour-of-day as a control AND as its own interaction with "
        "skill, so d is identified off heat variation at a GIVEN hour "
        "(different days / different venues).\n")
    say("| setting | games | d (skill x heat) alone | d (skill x heat) with "
        "hour controls | skill x hour |")
    say("|---|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        rs = []
        for r in by[setting]:
            q = dict(r)
            q["h"] = (q["T"] - 75.0) / 10.0
            q["sh"] = q["skill"] * q["h"]
            q["hr"] = (q["hour"] - 14.0) / 6.0
            q["hr2"] = q["hr"] ** 2
            q["shr"] = q["skill"] * q["hr"]
            rs.append(q)
        b0 = ols(rs, "y", ["skill", "h", "sh"])
        b1 = ols(rs, "y", ["skill", "h", "sh", "hr", "hr2", "shr"])
        cis, _ = boot(rs, "y", ["skill", "h", "sh", "hr", "hr2", "shr"])
        say(f"| {setting} | {len(rs)} | {b0[3]:+.3f} | "
            f"{fmt(b1[3], cis[3], 3)} | {fmt(b1[6], cis[6], 3)} |")
    say("")

    # ---------------- 3. rally-level favorite gap -----------------------
    say("## 3. Rally level: favorite-minus-underdog serve-rally win gap "
        "vs heat (H2)\n")
    say("Sample = data/decider_serve_splits.csv (PPA deciders + all MLP "
        "games) — close-match-selected, so the heat SLOPE is the object, "
        "not the level. v2 only labels who the favorite is (|eta| >= 0.1).\n")
    gap = defaultdict(list)
    logit_rows = defaultdict(list)
    for r in read_csv("data/decider_serve_splits.csv"):
        m = meta.get(r["match_id"])
        if not m:
            continue
        st, w, eta, ev, hour = m
        h = (w["temp"] - 75.0) / 10.0
        ra = int(r["ra_pre"]) + int(r["ra_post"])
        wa = int(r["wa_pre"]) + int(r["wa_post"])
        rb = int(r["rb_pre"]) + int(r["rb_post"])
        wb = int(r["wb_pre"]) + int(r["wb_post"])
        if ra >= 8 and rb >= 8 and abs(eta) >= 0.1:
            d = (wa / ra - wb / rb) if eta > 0 else (wb / rb - wa / ra)
            gap[st].append({"ev": ev, "y": d, "h": h})
        for side, sgn in (("a", 1.0), ("b", -1.0)):
            nn = int(r[f"r{side}_pre"]) + int(r[f"r{side}_post"])
            ww = int(r[f"w{side}_pre"]) + int(r[f"w{side}_post"])
            if nn >= 4:
                logit_rows[st].append({"ev": ev, "n": nn, "wins": ww,
                                       "adv": sgn * eta, "h": h})
    say("| setting | games | mean gap | heat slope per +10F [95% CI] |")
    say("|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        rs = gap[setting]
        b = ols(rs, "y", ["h"])
        cis, _ = boot(rs, "y", ["h"])
        mg = sum(r["y"] for r in rs) / len(rs)
        say(f"| {setting} | {len(rs)} | {mg:+.3f} | {fmt(b[1], cis[1])} |")
    say("")

    def fit_logit(rs, init=None, maxiter=30):
        beta = list(init) if init else [0.0] * 4
        for _ in range(maxiter):
            grad = [0.0] * 4
            hess = [[0.0] * 4 for _ in range(4)]
            for r in rs:
                x = (1.0, r["adv"], r["h"], r["adv"] * r["h"])
                z = sum(b_ * x_ for b_, x_ in zip(beta, x))
                p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
                res = r["wins"] - r["n"] * p
                wgt = r["n"] * p * (1 - p)
                for i in range(4):
                    grad[i] += res * x[i]
                    for j in range(4):
                        hess[i][j] += wgt * x[i] * x[j]
            m = [hess[i][:] + [grad[i]] for i in range(4)]
            for col in range(4):
                piv = max(range(col, 4), key=lambda r_: abs(m[r_][col]))
                m[col], m[piv] = m[piv], m[col]
                if abs(m[col][col]) < 1e-10:
                    return None
                for r_ in range(4):
                    if r_ != col:
                        f = m[r_][col] / m[col][col]
                        for c_ in range(col, 5):
                            m[r_][c_] -= f * m[col][c_]
            step = [m[i][4] / m[i][i] for i in range(4)]
            beta = [b_ + s_ for b_, s_ in zip(beta, step)]
            if max(abs(s_) for s_ in step) < 1e-9:
                break
        return beta

    say("### 3b. Rally-level binomial logit: logit P(server wins rally) = "
        "a + b*adv + c*h + d*adv*h\n")
    say("| setting | rallies | b (adv) | d (adv x heat) [95% CI] |")
    say("|---|---|---|---|")
    rng = random.Random(23)
    for setting in ("outdoor", "indoor"):
        rs = logit_rows[setting]
        beta = fit_logit(rs)
        clustered = defaultdict(list)
        for r in rs:
            clustered[r["ev"]].append(r)
        keys = list(clustered)
        draws = []
        for _ in range(NBOOT_LOGIT):
            s = []
            for _ in keys:
                s.extend(clustered[rng.choice(keys)])
            bb = fit_logit(s, init=beta, maxiter=6)
            if bb:
                draws.append(bb[3])
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[int(0.975 * len(draws))]
        say(f"| {setting} | {sum(r['n'] for r in rs)} | {beta[1]:.3f} | "
            f"{beta[3]:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    say("")

    # ---------------- 4. duration / fatigue -----------------------------
    say("## 4. Does heat LENGTHEN matches? (H3 — the channel the wind "
        "work never had)\n")
    summ = {r["match_id"]: r for r in read_csv("data/match_rally_summary.csv")}
    dur = defaultdict(list)
    for mid, gs in per_match.items():
        s = summ.get(mid)
        m = meta[mid]
        st, w, eta, ev, hour = m
        stratum = f'{gs[0]["tour"]}|{gs[0]["scoring_format"]}|{gs[0]["best_of"]}'
        base = {"ev": ev, "cell": ev + "|" + stratum,
                "h": (w["temp"] - 75.0) / 10.0,
                "gap": abs(eta),
                "hr": (hour - 14.0) / 6.0,
                "ngames": float(len(gs)),
                "dec": 1.0 if (gs[0]["best_of"] == "3" and len(gs) == 3) else 0.0,
                "best_of": gs[0]["best_of"]}
        if s and s["score_check"] == "ok":
            try:
                base["nral"] = float(s["n_rallies"])
                base["npts"] = float(s["n_points"])
                base["rpp"] = float(s["n_rallies"]) / float(s["n_points"])
            except (ValueError, ZeroDivisionError):
                pass
        dur[st].append(base)

    say("Within (event x format) cells, so format/venue/season are "
        "differenced out; identified off heat swings inside one event. "
        "Controls: |skill gap|. Positive = heat lengthens.\n")
    for ykey, ylabel, needs in (
            ("nral", "rallies per match", True),
            ("npts", "points per match", True),
            ("rpp", "rallies per point", True),
            ("ngames", "games per match", False)):
        say(f"**{ylabel}**\n")
        say("| setting | matches | mean | heat slope per +10F [95% CI] | "
            "+ hour control |")
        say("|---|---|---|---|---|")
        for setting in ("outdoor", "indoor"):
            rs = [dict(r) for r in dur[setting] if (ykey in r)]
            if len(rs) < 200:
                continue
            mean = sum(r[ykey] for r in rs) / len(rs)
            dm = demean(rs, [ykey, "h", "gap", "hr"])
            b = ols(dm, ykey, ["h", "gap"])
            cis, _ = boot(dm, ykey, ["h", "gap"])
            b2 = ols(dm, ykey, ["h", "gap", "hr"])
            cis2, _ = boot(dm, ykey, ["h", "gap", "hr"], seed=31)
            say(f"| {setting} | {len(rs)} | {mean:.2f} | "
                f"{fmt(b[1], cis[1], 3)} | {fmt(b2[1], cis2[1], 3)} |")
        say("")

    say("**3-game rate (best-of-3 matches only; 1 = went to a decider)**\n")
    say("| setting | matches | rate | heat slope per +10F [95% CI] |")
    say("|---|---|---|---|")
    for setting in ("outdoor", "indoor"):
        rs = [dict(r) for r in dur[setting] if r["best_of"] == "3"]
        if len(rs) < 200:
            continue
        rate = sum(r["dec"] for r in rs) / len(rs)
        dm = demean(rs, ["dec", "h", "gap"])
        b = ols(dm, "dec", ["h", "gap"])
        cis, _ = boot(dm, "dec", ["h", "gap"])
        say(f"| {setting} | {len(rs)} | {rate:.3f} | {fmt(b[1], cis[1], 4)} |")
    say("")

    # ---------------- 5. secondary channels -----------------------------
    say("## 5. Secondary channels: humidity, apparent temp, precipitation\n")
    say("Same reg-1 form with the channel in place of h (units: RH per "
        "+10 pct pts, apparent per +10F, precip per +1 mm/h).\n")
    say("| channel | setting | games | d (skill x channel) [95% CI] | "
        "main effect c [95% CI] |")
    say("|---|---|---|---|---|")
    chans = (("app", "apparent temp /10F", lambda r: (r["app"] - 75.0) / 10.0),
             ("rh", "humidity /10pp", lambda r: (r["rh"] - 50.0) / 10.0),
             ("precip", "precip mm/h", lambda r: r["precip"]))
    for key, lab, f in chans:
        for setting in ("outdoor", "indoor"):
            rs = []
            for r in by[setting]:
                q = dict(r)
                q["h"] = f(r)
                q["sh"] = q["skill"] * q["h"]
                rs.append(q)
            b = ols(rs, "y", ["skill", "h", "sh"])
            cis, _ = boot(rs, "y", ["skill", "h", "sh"], n=max(400, NBOOT // 3))
            say(f"| {lab} | {setting} | {len(rs)} | {fmt(b[3], cis[3], 3)} | "
                f"{fmt(b[2], cis[2], 4)} |")
    say("")

    # ---------------- 6. what does the CI allow, in real units ----------
    say("## 6. Translation: what the outdoor CI still allows\n")
    b, cis, nrows, nev = r1["outdoor"]
    d, dlo, dhi = b[3], cis[3][0], cis[3][1]
    say(f"Outdoor d = {d:+.3f} share per unit skill per +10F, 95% CI "
        f"[{dlo:+.3f}, {dhi:+.3f}] (n = {nrows} games, {nev} events).")
    # a clear favorite: eta such that game win prob ~ 0.80
    for target in (0.75, 0.90):
        lo_e, hi_e = 0.0, 3.0
        for _ in range(60):
            mid = (lo_e + hi_e) / 2
            if game_win_prob(mid) < target:
                lo_e = mid
            else:
                hi_e = mid
        eta = (lo_e + hi_e) / 2
        skill = sigmoid(eta) - 0.5
        for dd, name in ((d, "point est"), (dlo, "CI low"), (dhi, "CI high")):
            dshare = dd * skill * 2.0          # +20F above 75F -> 95F
            # map share back to an eta and then to a game win prob
            s_new = sigmoid(eta) + dshare
            s_new = min(0.99, max(0.01, s_new))
            eta_new = math.log(s_new / (1 - s_new))   # p = sigmoid(eta)
            p_new = game_win_prob(eta_new)
            say(f"  - a {target:.0%} favorite at 75F -> {p_new:.1%} at 95F "
                f"({name}: share shift {dshare*100:+.2f} pp)")
    say("")
    say("---")
    say("*Deterministic: all RNGs seeded. Written by "
        "model/weather_review/heat_test.py.*")

    (ROOT / "model/weather_review/heat_test.md").write_text("\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/heat_test.md")


if __name__ == "__main__":
    main()
