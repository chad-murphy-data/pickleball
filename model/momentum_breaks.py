"""EXPLORATION (not a pre-registration) — do different break types affect
momentum? Scans sample A only; the untouched pool is reserved for
confirming anything that pings. Reports EVERY cell with its denominator,
per the working rule (log what you tried, not just what crossed).

Two tests x three break types x several windows:
  T1 ICING: matched-moment adjusted effect of a break on the receiving
     side's share of the next W rallies (W in 3,5,10). Same design as H3c
     but with the treatment being a timeout / challenge / 6-pt switchover.
  T2 FRAGILITY: adjusted lag-1 rally autocorrelation, estimated separately
     for pairs with no break between vs a break between (by type). Tests
     whether real within-run momentum exists in continuous play and is
     reset by interruptions (which pooled H1/H2 would dilute).

Nothing here writes to receipts.json. Output: model/momentum_breaks_scan.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))
from momentum_test import RAW, sample_matches, _point_delta   # noqa: E402
from momentum_h3b import ols_cr0                              # noqa: E402
from sitelib.race import team_eta                             # noqa: E402

RALLY, POINT, SIDEOUT, SECOND = 12, 14, 16, 23
TIMEOUTS, SWITCH, CHALLENGE = (18, 35), (17,), (2, 37, 45)
WINDOWS = (3, 5, 10)


def extract(rows, side1):
    """Break-aware rally sequence per game. Each rally records which break
    types occurred in the gap immediately before it."""
    rows = sorted(rows, key=lambda x: x.get("log_index", 0))
    s1 = set(side1)
    games, cur, game = [], [], None
    pend = set()
    i = 0
    while i < len(rows):
        r = rows[i]
        t = r.get("log_type")
        if t in TIMEOUTS:
            pend.add("timeout"); i += 1; continue
        if t in SWITCH:
            pend.add("switch"); i += 1; continue
        if t in CHALLENGE:
            pend.add("challenge"); i += 1; continue
        if t != RALLY:
            i += 1; continue
        g = r.get("game_number")
        if g != game:
            if cur:
                games.append(cur)
            cur, game, pend = [], g, set()
        j, outcome = i + 1, None
        while j < len(rows):
            tj = rows[j].get("log_type")
            if tj in (POINT, SIDEOUT, SECOND):
                outcome = rows[j]; break
            if tj == RALLY:
                break
            j += 1
        if outcome is None:
            i += 1; continue
        try:
            ss, sr, num = (int(x) for x in
                           r["start_score_current_game_string"].split("-"))
        except (KeyError, ValueError):
            i = j + 1; continue
        if outcome.get("log_type") == POINT:
            d, _ = _point_delta(outcome)
            if d <= 0:
                pend = set(); i = j + 1; continue
            y_srv = 1
        else:
            y_srv = 0
        side = 0 if (r.get("server_uuid") or "").lower() in s1 else 1
        cur.append(dict(side=side, y_srv=y_srv, num=num,
                        srv_score=ss, rcv_score=sr,
                        margin=max(-7, min(7, ss - sr)),
                        brk_timeout=int("timeout" in pend),
                        brk_switch=int("switch" in pend),
                        brk_challenge=int("challenge" in pend),
                        brk_any=int(bool(pend))))
        pend = set()
        i = j + 1
    if cur:
        games.append(cur)
    return games


def rally_win(r, P):
    return 1 if (r["side"] == P) == bool(r["y_srv"]) else 0


# ---------- T1: icing, per break type ----------

def icing(all_games, eta_by, W):
    """rows: [treat_to, treat_sw, treat_ch, share_after, share_pre,
    margin_recv, srv2, eta_recv, match]"""
    rows, cl = [], []
    for mid, games in all_games.items():
        eta = eta_by[mid]
        for game in games:
            n = len(game)
            for t in range(W, n - W + 1):
                r = game[t]
                P = 1 - r["side"]
                after = sum(rally_win(x, P) for x in game[t:t + W]) / W
                pre = sum(rally_win(x, P) for x in game[t - W:t]) / W
                rows.append([r["brk_timeout"], r["brk_switch"],
                             r["brk_challenge"], after, pre,
                             -r["margin"], r["num"] == 2,
                             eta if P == 0 else -eta])
                cl.append(mid)
    m = np.asarray(rows, float)
    cl = np.asarray(cl)
    res = {}
    for bi, name in ((0, "timeout"), (1, "switch"), (2, "challenge")):
        # treated = this break only; control = no break of any kind
        other = [c for c in (0, 1, 2) if c != bi]
        treat = m[:, bi] == 1
        ctrl = (m[:, 0] + m[:, 1] + m[:, 2]) == 0
        keep = treat | ctrl
        mm, cc = m[keep], cl[keep]
        tcol = (mm[:, bi] == 1).astype(float)
        X = np.column_stack([np.ones(len(mm)), tcol, mm[:, 4], mm[:, 5],
                             mm[:, 6], mm[:, 7]])
        beta, V = ols_cr0(X, mm[:, 3], cc)
        th, se = beta[1], math.sqrt(V[1, 1])
        z = th / se if se else 0.0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        naive = mm[tcol == 1, 3].mean() - mm[tcol == 1, 4].mean()
        res[name] = (th, se, z, p, int(tcol.sum()), int((tcol == 0).sum()),
                     naive)
    return res


# ---------- T2: fragility (adjusted lag-1 autocorrelation) ----------

def fragility(all_games, eta_by):
    """Adjusted lag-1: does winning rally t predict winning rally t+1,
    separately for no-break vs break-between pairs? Team-1 perspective,
    controlling for current serve state + eta."""
    buckets = defaultdict(lambda: ([], []))   # name -> (Xrows, y), + cl sep
    clus = defaultdict(list)
    for mid, games in all_games.items():
        eta = eta_by[mid]
        for game in games:
            for a, b in zip(game, game[1:]):
                yA = 1.0 if (b["side"] == 0) == bool(b["y_srv"]) else 0.0
                lagA = 1.0 if (a["side"] == 0) == bool(a["y_srv"]) else 0.0
                state = (b["side"] * 2 + (b["num"] == 2))   # 0..3
                row = [1.0, lagA,
                       float(state == 1), float(state == 2), float(state == 3),
                       eta]
                names = ["no_break"] if not b["brk_any"] else []
                if b["brk_timeout"]:
                    names.append("after_timeout")
                if b["brk_switch"]:
                    names.append("after_switch")
                if b["brk_challenge"]:
                    names.append("after_challenge")
                if b["brk_any"]:
                    names.append("after_any_break")
                for nm in names:
                    buckets[nm][0].append(row)
                    buckets[nm][1].append(yA)
                    clus[nm].append(mid)
    out = {}
    for nm, (X, y) in buckets.items():
        X, y, cl = np.asarray(X, float), np.asarray(y, float), np.asarray(clus[nm])
        if len(y) < 200:
            out[nm] = (None, None, None, None, len(y))
            continue
        beta, V = _logit_cr(X, y, cl)
        # APE of lag on P(win): delta at lag 0->1 averaged
        p0 = 1 / (1 + np.exp(-(X.copy() * [1, 0, 1, 1, 1, 1] @ beta
                               if False else None))) if False else None
        eff, se = _lag_ape(X, y, beta, V, cl)
        z = eff / se if se else 0.0
        out[nm] = (eff, se, z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))),
                   len(y))
    return out


def _logit_cr(X, y, clusters):
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ beta))
        Wd = p * (1 - p) + 1e-12
        H = X.T @ (X * Wd[:, None])
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
    return beta, bread @ meat @ bread


def _lag_ape(X, y, beta, V, cl):
    X0, X1 = X.copy(), X.copy()
    X0[:, 1] = 0.0
    X1[:, 1] = 1.0
    p0 = 1 / (1 + np.exp(-X0 @ beta))
    p1 = 1 / (1 + np.exp(-X1 @ beta))
    eff = float(np.mean(p1 - p0))
    grad = (X1 * (p1 * (1 - p1))[:, None] - X0 * (p0 * (1 - p0))[:, None]).mean(0)
    return eff, float(np.sqrt(grad @ V @ grad))


def main():
    sample, pairs, fmts, vals = sample_matches()
    all_games, eta_by = {}, {}
    for mid in sample:
        p = RAW / "match_logs" / mid[:2] / f"{mid}.json"
        try:
            rows = json.loads(p.read_text()).get("data") or []
        except FileNotFoundError:
            continue
        s1, s2 = pairs[mid]
        all_games[mid] = extract(rows, s1)
        eta_by[mid] = team_eta(vals[s1[0]], vals[s1[1]], vals[s2[0]], vals[s2[1]])

    L = ["# Break-type momentum scan — EXPLORATION (sample A, 2,500 matches)",
         "", "Not a pre-registration. All cells reported with denominators; "
         "nothing here is an asserted finding without a pre-registered shot "
         "on the untouched pool. ~%d break estimands below — under the null "
         "expect ~1 in 20 to cross p<.05 by chance." % 24, ""]

    L.append("## T1 — Icing: adjusted effect of a break on the receiver's "
             "next-W-rally share (matched moments)")
    L.append("")
    L.append("| break | W | theta | 99% CI | z | p | n_treat | n_ctrl | naive Δ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for W in WINDOWS:
        res = icing(all_games, eta_by, W)
        for name in ("timeout", "switch", "challenge"):
            th, se, z, p, nt, nc, nv = res[name]
            L.append(f"| {name} | {W} | {th*100:+.2f}pp | "
                     f"[{(th-2.576*se)*100:+.2f}, {(th+2.576*se)*100:+.2f}] | "
                     f"{z:+.2f} | {p:.2g} | {nt} | {nc} | {nv*100:+.1f}pp |")
    L.append("")
    L.append("## T2 — Fragility: adjusted lag-1 rally autocorrelation, by "
             "whether a break separates the two rallies")
    L.append("")
    L.append("| pair type | lag-1 APE | 99% CI | z | p | n_pairs |")
    L.append("|---|---|---|---|---|---|")
    frag = fragility(all_games, eta_by)
    for nm in ("no_break", "after_any_break", "after_timeout",
               "after_switch", "after_challenge"):
        if nm not in frag:
            continue
        eff, se, z, p, n = frag[nm]
        if eff is None:
            L.append(f"| {nm} | (n={n} too few) |||||")
        else:
            L.append(f"| {nm} | {eff*100:+.2f}pp | "
                     f"[{(eff-2.576*se)*100:+.2f}, {(eff+2.576*se)*100:+.2f}] | "
                     f"{z:+.2f} | {p:.2g} | {n} |")
    (ROOT / "model" / "momentum_breaks_scan.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote model/momentum_breaks_scan.md")


if __name__ == "__main__":
    main()
