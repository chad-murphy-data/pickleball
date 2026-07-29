"""B3 — the rally-level favorites x wind test WITHOUT the decider collider.

model/favorites_wind.py's regression 3 (the rally-level arm) is fit on
data/decider_serve_splits.csv, i.e. deciding games + MLP games only: a
sample selected on the match having stayed close, which is a collider on
the very outcome (favorite failing to pull away) the test is about.

This re-runs the identical specification on EVERY doubles match with a
referee rally log — 993k rallies, 13k matches — using the corrected venue
labels (data/venue_overrides.csv) instead of the heuristic ones.

    python model/weather_review/fetch_rally_match_side.py <scratch>   # once
    python model/weather_review/rally_favorites_allmatches.py <scratch>

Specification (same as favorites_wind reg 3):
    logit P(server side wins this rally) = a + b*adv + c*w + d*adv*w
    adv = serving team's v2 eta advantage (signed, per-point logit)
    w   = match-hour wind speed / 10 mph
    d < 0  =  wind erodes the better team's rally edge (the hypothesis)
Fit on (wins, attempts) per match-side, which is the exact Bernoulli
likelihood of the rally series because the covariates are constant within
a match-side.  Cluster bootstrap over EVENTS.

Also reported (mechanism channels never tested before):
  * side-out rate and second-server rate vs wind (does wind make serving
    harder / make serves change hands faster?)
  * the same interaction split by score state (early / mid / endgame),
    to see whether any wind effect concentrates late in games.
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
from sitelib.race import team_eta  # noqa: E402

SCRATCH = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
R_BOOT = 500
SEED = 20260729


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- data ----
def load():
    v2 = {r["player_id"].lower(): float(r["value_now_mean"])
          for r in read_csv(ROOT / "data/v2_players.csv")}

    # corrected venue labels; fall back to the heuristic where unaudited
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    ovr, conf = {}, {}
    for r in read_csv(ROOT / "data/venue_overrides.csv"):
        ovr[r["event_id"]] = r["setting"]
        conf[r["event_id"]] = r["confidence"]

    hourly = {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        try:
            hourly[(r["event_id"], r["local_time"][:13])] = (
                float(r["windspeed_10m"]), float(r["windgusts_10m"] or "nan"),
                float(r["temperature_2m"]))
        except (TypeError, ValueError):
            pass
    hour, actual = {}, {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            hour[r["match_id"]] = ts[:13]
            actual[r["match_id"]] = bool(r["start_local"])

    # match -> teams / event / format, from games.csv (first non-DB game)
    match = {}
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        m = g["match_id"]
        if m in match:
            match[m]["ngames"] += 1
            continue
        match[m] = {
            "ev": g["event_id"], "date": g["date"], "tour": g["tour"],
            "fmt": g["scoring_format"], "ngames": 1,
            "t1": (g["t1_p1"].lower(), g["t1_p2"].lower()),
            "t2": (g["t2_p1"].lower(), g["t2_p2"].lower()),
        }
    return v2, geo, ovr, conf, hourly, hour, actual, match


def build_cells():
    v2, geo, ovr, conf, hourly, hour, actual, match = load()
    side_player = defaultdict(dict)
    for r in read_csv(SCRATCH / "rally_side_player.csv"):
        side_player[r["match_id"]][r["side"]] = r["player_uuid"].lower()

    agg = defaultdict(dict)
    for r in read_csv(SCRATCH / "rally_match_side.csv"):
        agg[r["match_id"]][r["side"]] = r

    cells, drop = [], defaultdict(int)
    for m, sides in agg.items():
        meta = match.get(m)
        if not meta:
            drop["no_game_row"] += 1
            continue
        vals = [v2.get(p) for p in meta["t1"] + meta["t2"]]
        if not all(v is not None for v in vals):
            drop["no_rating"] += 1
            continue
        eta = team_eta(*vals)            # team1 advantage, per-point logit
        h = hour.get(m)
        wx = hourly.get((meta["ev"], h)) if h else None
        if wx is None:
            drop["no_weather"] += 1
            continue
        wind, gust, temp = wx
        setting = ovr.get(meta["ev"], geo.get(meta["ev"]))
        confid = conf.get(meta["ev"], "heuristic")
        for side, row in sides.items():
            who = side_player[m].get(side)
            if who is None:
                drop["no_side_map"] += 1
                continue
            if who in meta["t1"]:
                sgn = 1.0
            elif who in meta["t2"]:
                sgn = -1.0
            else:
                drop["side_player_mismatch"] += 1
                continue
            n = int(row["n"])
            if n < 4:
                continue
            cells.append({
                "m": m, "ev": meta["ev"], "tour": meta["tour"],
                "date": meta["date"], "fmt": meta["fmt"],
                "ngames": meta["ngames"], "setting": setting,
                "conf": confid, "actual_time": actual.get(m, False),
                "n": n, "wins": int(row["w"]),
                "n_sideout": int(row["n_sideout"]),
                "n_second": int(row["n_second"]),
                "n1": int(row["n1"]), "w1": int(row["w1"]),
                "n2": int(row["n2"]), "w2": int(row["w2"]),
                "buck": [(int(row[f"b{i}"]), int(row[f"c{i}"]))
                         for i in range(16)],
                "adv": sgn * eta, "wind": wind, "gust": gust, "temp": temp,
                "w": wind / 10.0,
            })
    return cells, drop


# ------------------------------------------------------------- fitting ----
def fit_logit(rows, k, beta0=None, iters=30):
    """Newton-Raphson binomial logit. rows = (wins, n, x1..x_{k-1}) tuples,
    intercept implicit at position 0."""
    beta = list(beta0) if beta0 else [0.0] * k
    exp = math.exp
    for _ in range(iters):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for r in rows:
            wins, n = r[0], r[1]
            z = beta[0]
            for i in range(1, k):
                z += beta[i] * r[1 + i]
            if z > 30:
                z = 30.0
            elif z < -30:
                z = -30.0
            p = 1.0 / (1.0 + exp(-z))
            res = wins - n * p
            wgt = n * p * (1.0 - p)
            for i in range(k):
                xi = 1.0 if i == 0 else r[1 + i]
                grad[i] += res * xi
                hi = hess[i]
                for j in range(k):
                    xj = 1.0 if j == 0 else r[1 + j]
                    hi[j] += wgt * xi * xj
        step = solve(hess, grad, k)
        if step is None:
            return None
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return beta


def solve(a, b, k):
    m = [a[i][:] + [b[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            return None
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, k + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][k] / m[i][i] for i in range(k)]


def boot(rows_by_ev, k, beta_hat, R=R_BOOT, seed=SEED):
    keys = list(rows_by_ev)
    rng = random.Random(seed)
    draws = []
    for _ in range(R):
        s = []
        for _ in keys:
            s.extend(rows_by_ev[rng.choice(keys)])
        b = fit_logit(s, k, beta0=beta_hat, iters=8)
        if b:
            draws.append(b)
    out = []
    for i in range(k):
        v = sorted(d[i] for d in draws)
        out.append((v[int(0.025 * len(v))], v[int(0.975 * len(v))],
                    (sum(d[i] for d in draws) / len(draws))))
    return out, len(draws)


def design(cells, keys, ykey="wins", nkey="n"):
    """rows of (wins, n, x1...) with x from keys."""
    return [tuple([c[ykey], c[nkey]] + [c[k] for k in keys]) for c in cells]


def fit_and_ci(cells, keys, label, out, R=R_BOOT):
    rows = design(cells, keys)
    k = len(keys) + 1
    beta = fit_logit(rows, k)
    by_ev = defaultdict(list)
    for c, r in zip(cells, rows):
        by_ev[c["ev"]].append(r)
    cis, nd = boot(by_ev, k, beta, R=R)
    names = ["const"] + keys
    out.append(f"{label}: n_cells={len(cells)} rallies={sum(c['n'] for c in cells)} "
               f"events={len(by_ev)} (boot {nd}/{R})")
    for nm, b, (lo, hi, mn) in zip(names, beta, cis):
        out.append(f"    {nm:>10} = {b:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    return beta, cis


# ------------------------------------------------------------------ main --
def main():
    cells, drop = build_cells()
    out = []
    say = lambda s="": (print(s), out.append(s))
    say("# B3 — rally-level favorites x wind on ALL matches (no decider collider)\n")
    say(f"cells built: {len(cells)}   dropped: {dict(drop)}")
    tot = sum(c["n"] for c in cells)
    say(f"rallies in scope: {tot}")
    bys = defaultdict(int)
    for c in cells:
        bys[(c["setting"], c["conf"])] += c["n"]
    say("rallies by (setting, label confidence): "
        + ", ".join(f"{k}={v}" for k, v in sorted(bys.items(), key=str)))
    say("")

    def arm(name, pred):
        return [c for c in cells if pred(c)]

    # ---------- 1. the headline replication, all matches ----------------
    say("## 1. logit P(server wins rally) = a + b*adv + c*w + d*adv*w\n")
    keys = ["adv", "w", "advw"]
    for c in cells:
        c["advw"] = c["adv"] * c["w"]
    res = {}
    for setting in ("outdoor", "indoor"):
        sub = arm(setting, lambda c, s=setting: c["setting"] == s)
        res[setting] = fit_and_ci(sub, keys, f"[{setting}] ALL matches", out)
        print()
    say("")

    # high/medium-confidence labels only
    say("### label-confidence sensitivity (high+medium override only)")
    for setting in ("outdoor", "indoor"):
        sub = arm(setting, lambda c, s=setting: c["setting"] == s
                  and c["conf"] in ("high", "medium"))
        fit_and_ci(sub, keys, f"[{setting}] high/med labels", out, R=300)
    say("")

    # actual (not planned) start times only
    say("### start-time sensitivity (actual start times only)")
    for setting in ("outdoor", "indoor"):
        sub = arm(setting, lambda c, s=setting: c["setting"] == s
                  and c["actual_time"])
        fit_and_ci(sub, keys, f"[{setting}] actual-time", out, R=300)
    say("")

    # decider-like subsample, to show the collider's footprint
    say("### collider check: matches that WENT the distance vs matches that did not")
    for setting in ("outdoor", "indoor"):
        for lab, pred in (("went-distance",
                           lambda c: c["ngames"] >= 3 or c["tour"] == "MLP"),
                          ("straight-games",
                           lambda c: not (c["ngames"] >= 3 or c["tour"] == "MLP"))):
            sub = arm(lab, lambda c, s=setting, p=pred: c["setting"] == s and p(c))
            if len(sub) > 200:
                fit_and_ci(sub, keys, f"[{setting}] {lab}", out, R=300)
    say("")

    # ---------- 2. mechanism: side-out rate, second-server rate ---------
    say("## 2. mechanism channels: does wind change k / the side-out rate?\n")
    say("serve-win rate is 1 - (rate the serve changes hands); a wind effect "
        "on serving shows up as c < 0 with no need for any rating.\n")
    for setting in ("outdoor", "indoor"):
        sub = arm(setting, lambda c, s=setting: c["setting"] == s)
        fit_and_ci(sub, ["w"], f"[{setting}] serve-win ~ wind", out, R=300)
    say("")
    say("second-server rate = P(rally is served by the 2nd server) — rises if "
        "first serves are lost more often; and P(win | 2nd server).")
    for setting in ("outdoor", "indoor"):
        sub = [dict(c, sec=c["n_second"]) for c in cells if c["setting"] == setting]
        # P(this serve rally is a 2nd-server rally)
        cc = [dict(c, wins=c["n2"], n=c["n"]) for c in cells
              if c["setting"] == setting]
        fit_and_ci(cc, ["w"], f"[{setting}] P(2nd-server rally) ~ wind", out, R=300)
        cc2 = [dict(c, wins=c["w2"], n=c["n2"]) for c in cells
               if c["setting"] == setting and c["n2"] >= 4]
        fit_and_ci(cc2, ["w"], f"[{setting}] P(win | 2nd server) ~ wind", out, R=300)
        cc1 = [dict(c, wins=c["w1"], n=c["n1"]) for c in cells
               if c["setting"] == setting and c["n1"] >= 4]
        fit_and_ci(cc1, ["w"], f"[{setting}] P(win | 1st server) ~ wind", out, R=300)
    say("")

    # ---------- 3. score state ------------------------------------------
    say("## 3. score state: is any wind effect concentrated late in games?\n")
    say("stratum by leader score at the start of the rally (to-11 games only "
        "for a clean threshold): early 0-5, mid 6-8, endgame 9+.\n")
    strata = {"early(0-5)": range(0, 6), "mid(6-8)": range(6, 9),
              "endgame(9+)": range(9, 12)}
    for setting in ("outdoor", "indoor"):
        for sname, rng_ in strata.items():
            sub = []
            for c in cells:
                if c["setting"] != setting or "11" not in c["fmt"]:
                    continue
                n = sum(c["buck"][i][0] for i in rng_)
                wv = sum(c["buck"][i][1] for i in rng_)
                if n < 3:
                    continue
                sub.append(dict(c, n=n, wins=wv))
            if len(sub) > 200:
                fit_and_ci(sub, keys, f"[{setting}] {sname}", out, R=300)
        say("")

    (Path(__file__).parent / "rally_favorites_allmatches.txt").write_text(
        "\n".join(out) + "\n")
    print("\nwrote model/weather_review/rally_favorites_allmatches.txt")


if __name__ == "__main__":
    main()
