"""Frozen analysis code for model/momentum_prereg.md. Run ONCE on the
pre-registered sample after `--selftest` passes on synthetic data.

    python model/momentum_test.py --selftest    # synthetic iid validation
    python model/momentum_test.py               # THE run (writes results md)

Everything here is specified by the pre-registration; do not tune after
unblinding. Rally construction mirrors scraper/harvest_logs.py's
validated tally (56/58 exact score reconciliation).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(ROOT / "web"))
from harvest_logs import RAW, _point_delta          # noqa: E402
from sitelib.race import team_eta                   # noqa: E402

RALLY, POINT, SIDEOUT, SECOND = 12, 14, 16, 23
TIMEOUTS = (18, 35)
SEED = 20260717
N_SAMPLE = 2500


# ---------- sample (frozen) ----------

def sample_matches():
    vals = load_values()
    pairs, fmts = {}, {}
    for r in csv.DictReader(open(ROOT / "data" / "games.csv")):
        m = r["match_id"].lower()
        if m not in pairs:
            pairs[m] = ([r["t1_p1"].lower(), r["t1_p2"].lower()],
                        [r["t2_p1"].lower(), r["t2_p2"].lower()])
        fmts.setdefault(m, {})[int(r["game_number"])] = \
            15 if r["scoring_format"] == "sideout_15" else 11
    pop = []
    for r in csv.DictReader(open(ROOT / "data" / "match_rally_summary.csv")):
        if r["discipline"] == "doubles" and r["score_check"] == "ok":
            m = r["match_id"]
            if m in pairs and all(u in vals for s in pairs[m] for u in s):
                pop.append(m)
    sample = random.Random(SEED).sample(sorted(pop), N_SAMPLE)
    return sample, pairs, fmts, vals


def load_values():
    return {r["player_id"].lower(): float(r["value_now_mean"])
            for r in csv.DictReader(open(ROOT / "data" / "v2_players.csv"))}


# ---------- rally table ----------

def match_rallies(rows, side1, side2, eta_m, T_by_game):
    """Yield dict per resolved rally, per the pre-registration."""
    rows = sorted(rows, key=lambda x: x.get("log_index", 0))
    s1 = set(side1)
    # team uuid -> side for timeout attribution (learned as in the tally)
    team_side = {}
    for r in rows:
        if r.get("log_type") == POINT:
            d, team = _point_delta(r)
            if d > 0 and team:
                srv = (r.get("server_uuid") or "").lower()
                if srv in s1:
                    team_side.setdefault(team, 0)
                elif srv in set(side2):
                    team_side.setdefault(team, 1)

    out = []
    game = None
    poss_srv, poss_n = None, 0
    y_hist = []                      # resolved y_A this game
    pend_to = set()                  # sides with timeout since last rally
    i = 0
    while i < len(rows):
        r = rows[i]
        t = r.get("log_type")
        if t in TIMEOUTS:
            key = ("timeout_log", "additional_timeout_log")
            team = None
            for k in key:
                if isinstance(r.get(k), dict):
                    team = (r[k].get("team_uuid") or "").lower()
            if team in team_side:
                pend_to.add(team_side[team])
            i += 1
            continue
        if t != RALLY:
            i += 1
            continue
        g = r.get("game_number")
        if g != game:
            game, poss_srv, poss_n, y_hist, pend_to = g, None, 0, [], set()
        # outcome = next 14/16/23 row
        j = i + 1
        outcome = None
        while j < len(rows):
            tj = rows[j].get("log_type")
            if tj in (POINT, SIDEOUT, SECOND):
                outcome = rows[j]
                break
            if tj == RALLY:
                break
            j += 1
        if outcome is None:
            i += 1
            continue
        srv = (r.get("server_uuid") or "").lower()
        try:
            ss, sr, num = (int(x) for x in
                           r["start_score_current_game_string"].split("-"))
        except (KeyError, ValueError):
            i = j + 1
            continue
        if outcome.get("log_type") == POINT:
            d, _ = _point_delta(outcome)
            if d <= 0:                       # correction: unresolved
                poss_srv, poss_n = None, 0
                i = j + 1
                continue
            y_srv = 1
        else:
            y_srv = 0
        side = 0 if srv in s1 else 1
        if srv != poss_srv:
            poss_srv, poss_n = srv, 0
        T = T_by_game.get(g, 11)
        y_A = y_srv if side == 0 else 1 - y_srv
        out.append(dict(
            y_srv=y_srv, y_A=y_A, side=side, srv2=1 if num == 2 else 0,
            eta_srv=eta_m if side == 0 else -eta_m, eta_m=eta_m,
            margin=max(-7, min(7, ss - sr)),
            poss1=1 if poss_n == 1 else 0, poss2p=1 if poss_n >= 2 else 0,
            trail5c=(sum(y_hist[-5:]) / 5 - 0.5) if len(y_hist) >= 5 else None,
            lag1_A=y_hist[-1] if y_hist else None,
            to_recv=1 if (1 - side) in pend_to else 0,
            to_srv=1 if side in pend_to else 0,
            gp_srv=1 if (ss >= T - 1 and ss - sr >= 1) else 0,
            gp_recv=1 if (sr >= T - 1 and sr - ss >= 1) else 0,
            state=("A1", "A2", "B1", "B2")[side * 2 + (num == 2)],
        ))
        pend_to = set()
        poss_n = poss_n + 1 if y_srv else 0
        if not y_srv:
            poss_srv = None
        y_hist.append(y_A)
        i = j + 1
    return out


# ---------- estimation ----------

def fit_logit(X, y, clusters):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p) + 1e-12
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p) + 1e-12)[:, None]))
    resid = (y - p)[:, None] * X
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(clusters):
        s = resid[clusters == c].sum(axis=0)
        meat += np.outer(s, s)
    V = bread @ meat @ bread
    return beta, V


def ape(X, beta, V, col, delta=1.0, base=0.0):
    """Average partial effect of moving column `col` base→base+delta,
    delta-method CI."""
    X0, X1 = X.copy(), X.copy()
    X0[:, col] = base
    X1[:, col] = base + delta
    p0 = 1 / (1 + np.exp(-X0 @ beta))
    p1 = 1 / (1 + np.exp(-X1 @ beta))
    eff = float(np.mean(p1 - p0))
    grad = (X1 * (p1 * (1 - p1))[:, None] - X0 * (p0 * (1 - p0))[:, None]
            ).mean(axis=0)
    se = float(np.sqrt(grad @ V @ grad))
    z = eff / se if se > 0 else 0.0
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return eff, se, z, pval


def run_models(rallies, clusters):
    R = rallies
    idx = np.arange(len(R))
    cl = np.asarray(clusters)

    def col(name):
        return np.array([r[name] for r in R], float)

    ones = np.ones(len(R))
    res = {}

    # M1 base: y_srv ~ 1 srv2 eta_srv margin poss1 poss2p
    X1 = np.column_stack([ones, col("srv2"), col("eta_srv"), col("margin"),
                          col("poss1"), col("poss2p")])
    b, V = fit_logit(X1, col("y_srv"), cl)
    res["H1 phi_poss (poss2p)"] = ape(X1, b, V, 5)

    # M2: y_A ~ 1 + state dummies + eta_m + trail5c   (full-window subset)
    keep = np.array([r["trail5c"] is not None for r in R])
    R2 = [r for r in R if r["trail5c"] is not None]
    st = {s: np.array([1.0 if r["state"] == s else 0.0 for r in R2])
          for s in ("A2", "B1", "B2")}
    X2 = np.column_stack([np.ones(len(R2)), st["A2"], st["B1"], st["B2"],
                          np.array([r["eta_m"] for r in R2]),
                          np.array([r["trail5c"] for r in R2])])
    b2, V2 = fit_logit(X2, np.array([r["y_A"] for r in R2], float), cl[keep])
    res["H2 phi5 (trail5c +0.2)"] = ape(X2, b2, V2, 5, delta=0.2)

    keepl = np.array([r["lag1_A"] is not None for r in R])
    R2l = [r for r in R if r["lag1_A"] is not None]
    stl = {s: np.array([1.0 if r["state"] == s else 0.0 for r in R2l])
           for s in ("A2", "B1", "B2")}
    X2l = np.column_stack([np.ones(len(R2l)), stl["A2"], stl["B1"], stl["B2"],
                           np.array([r["eta_m"] for r in R2l]),
                           np.array([float(r["lag1_A"]) for r in R2l])])
    b2l, V2l = fit_logit(X2l, np.array([r["y_A"] for r in R2l], float), cl[keepl])
    res["H2s phi1 (lag1_A)"] = ape(X2l, b2l, V2l, 5)

    # M3: M1 + timeouts
    X3 = np.column_stack([X1, col("to_recv"), col("to_srv")])
    b3, V3 = fit_logit(X3, col("y_srv"), cl)
    res["H3 tau_recv"] = ape(X3, b3, V3, 6)
    res["H3s tau_srv"] = ape(X3, b3, V3, 7)

    # M4: M1 + game points
    X4 = np.column_stack([X1, col("gp_srv"), col("gp_recv")])
    b4, V4 = fit_logit(X4, col("y_srv"), cl)
    res["H4 beta_gp (gp_srv)"] = ape(X4, b4, V4, 6)
    res["H4s beta_gp_recv"] = ape(X4, b4, V4, 7)
    return res


# ---------- selftest (synthetic iid — run before unblinding) ----------

def selftest():
    rng = random.Random(7)
    R, cl = [], []
    k = 0.445
    for m in range(800):
        eta = rng.gauss(0, 0.5)
        oddsp = math.exp(eta)
        okA = (k / (1 - k)) * math.sqrt(oddsp)
        okB = (k / (1 - k)) / math.sqrt(oddsp)
        kA, kB = okA / (1 + okA), okB / (1 + okB)
        for g in (1, 2):
            a = b_ = 0
            side, num = rng.choice([0, 1]), 2
            poss_srv_n, y_hist, pend = 0, [], set()
            while not ((a >= 11 and a - b_ >= 2) or (b_ >= 11 and b_ - a >= 2)):
                for s in (0, 1):
                    if rng.random() < 0.015:
                        pend.add(s)
                pwin = kA if side == 0 else kB
                y_srv = 1 if rng.random() < pwin else 0
                ss, sr = (a, b_) if side == 0 else (b_, a)
                y_A = y_srv if side == 0 else 1 - y_srv
                R.append(dict(
                    y_srv=y_srv, y_A=y_A, side=side, srv2=1 if num == 2 else 0,
                    eta_srv=eta if side == 0 else -eta, eta_m=eta,
                    margin=max(-7, min(7, ss - sr)),
                    poss1=1 if poss_srv_n == 1 else 0,
                    poss2p=1 if poss_srv_n >= 2 else 0,
                    trail5c=(sum(y_hist[-5:]) / 5 - 0.5) if len(y_hist) >= 5 else None,
                    lag1_A=y_hist[-1] if y_hist else None,
                    to_recv=1 if (1 - side) in pend else 0,
                    to_srv=1 if side in pend else 0,
                    gp_srv=1 if (ss >= 10 and ss - sr >= 1) else 0,
                    gp_recv=1 if (sr >= 10 and sr - ss >= 1) else 0,
                    state=("A1", "A2", "B1", "B2")[side * 2 + (num == 2)]))
                cl.append(m)
                pend = set()
                y_hist.append(y_A)
                if y_srv:
                    if side == 0:
                        a += 1
                    else:
                        b_ += 1
                    poss_srv_n += 1
                else:
                    poss_srv_n = 0
                    if num == 1:
                        num = 2
                    else:
                        side, num = 1 - side, 1
    res = run_models(R, cl)
    print(f"selftest: {len(R)} synthetic rallies, all true effects = 0")
    ok = True
    for name, (eff, se, z, p) in res.items():
        flag = "" if abs(z) < 3.29 else "  <-- FAIL (|z| >= 3.29)"
        if abs(z) >= 3.29:
            ok = False
        print(f"  {name:26s} APE {eff*100:+.2f}pp  z={z:+.2f}{flag}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# ---------- the run ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    sample, pairs, fmts, vals = sample_matches()
    R, cl, dropped = [], [], 0
    for m in sample:
        p = RAW / "match_logs" / m[:2] / f"{m}.json"
        try:
            rows = json.loads(p.read_text()).get("data") or []
        except FileNotFoundError:
            dropped += 1
            continue
        s1, s2 = pairs[m]
        eta = team_eta(vals[s1[0]], vals[s1[1]], vals[s2[0]], vals[s2[1]])
        rs = match_rallies(rows, s1, s2, eta, fmts.get(m, {}))
        if not rs:
            dropped += 1
            continue
        R.extend(rs)
        cl.extend([m] * len(rs))
    counts = dict(matches=len(sample) - dropped, dropped=dropped,
                  rallies=len(R),
                  timeouts_recv=sum(r["to_recv"] for r in R),
                  timeouts_srv=sum(r["to_srv"] for r in R),
                  gp_srv=sum(r["gp_srv"] for r in R),
                  poss2p=sum(r["poss2p"] for r in R))
    print("counts:", counts)
    res = run_models(R, np.asarray(cl))

    lines = ["# Momentum pre-registration — RESULTS (run once)", "",
             f"Sample: {counts['matches']} matches ({counts['dropped']} dropped), "
             f"{counts['rallies']:,} resolved rallies; timeout events "
             f"recv/srv {counts['timeouts_recv']}/{counts['timeouts_srv']}; "
             f"game-point rallies {counts['gp_srv']:,}; "
             f"possession-3rd+ rallies {counts['poss2p']:,}.", "",
             "| estimand | APE | 99% CI | z | p |", "|---|---|---|---|---|"]
    for name, (eff, se, z, pv) in res.items():
        lo, hi = eff - 2.576 * se, eff + 2.576 * se
        lines.append(f"| {name} | {eff*100:+.2f}pp | "
                     f"[{lo*100:+.2f}, {hi*100:+.2f}] | {z:+.2f} | {pv:.2g} |")
        print(lines[-1])
    (ROOT / "model" / "momentum_results.md").write_text("\n".join(lines) + "\n")
    print("wrote model/momentum_results.md")


if __name__ == "__main__":
    main()
