"""Clutch, measured properly — leverage covariance, both channels, whole archive.

This supersedes the estimator in `model/big_points.py --clutch` (see
`model/clutch.md` for that first pass). Three things changed, each of which
buys real statistical power or removes a real confound:

  1. WHOLE ARCHIVE, NOT ONE SPRING.  The first pass read raw/match_logs for
     Jan-May 2026 only: 162,942 serving rallies. The warehouse (pb_rally)
     holds 1,514,518 rallies back to 2024-01 — 917k PPA doubles, 83k MLP
     doubles, 513k PPA singles. ~9x the sample.

  2. BOTH CHANNELS.  The first pass measured serving rallies only, on the
     grounds that "in doubles you can't pin a return rally on one of two
     receivers". But pb_rally carries receiver_uuid per rally — the referee
     log names the man who returned the serve, and it alternates within a
     side. Return rallies are therefore just as individually attributable as
     serve rallies (both are "this player struck the ball, with a partner on
     court"). Using both doubles the exposure AND, because the two sets are
     disjoint, hands us a genuinely independent split for reliability.
     PPA singles is a third arm with no partner at all.

  3. WITHIN-PLAYER-WITHIN-GAME IDENTIFICATION.  The first pass compared a
     rally outcome to the *matchup's* expected serve-win rate. That leaves a
     confound: a player who happens to serve at high-leverage moments AND is
     simply better than their partner scores as "clutch" without any
     moment-specific ability. Here both leverage and outcome are demeaned
     inside each (player x game x channel) cell, so the estimate is a pure
     within-player covariance. It is mechanically immune to how good the
     player is, how good their team was that day, which games they played,
     and how much high-leverage exposure they get. It can only answer:
     *among this player's own rallies in this game, do they win the big ones
     more than the small ones?*

Estimator. For rally r, LEVERAGE L_r = |P(win game | win rally) -
P(win game | lose rally)|, exact from the serve-aware side-out DP
(web/sitelib/winprob.py) at eta = 0 with the measured league k, so leverage
is a property of the SITUATION (score + serve state), never of the players.
Scaled to units of 1 SD across all rallies. Per player p:

    b_p = sum_g Sxy_g / sum_g Sxx_g

over that player's (game, channel) cells, where Sxy/Sxx are the within-cell
covariance and leverage sum-of-squares. b_p reads as *extra probability of
winning a rally per +1 SD of leverage*.

Null. Conditional permutation: shuffle leverage among a player's own rallies
inside each cell. That null is exact (it conditions on the realised leverage
multiset and outcome multiset, so it needs no assumption about the score path,
which is endogenous), and its variance is closed-form —
Var = sum_g Sxx_g * Syy_g / (n_g - 1) — so no Monte Carlo is required.

Shrinkage. Per-player (b_p, se_p) go into a normal-normal empirical Bayes
model, b_p ~ N(theta_p, se_p^2), theta_p ~ N(mu, tau^2), tau fit by marginal
ML. tau is the population spread of true clutch; the posterior mean is what
gets ranked. This is what lets us name players honestly: a big raw number on
few rallies shrinks to nothing, a moderate one on 20k rallies survives.

Run:
    python model/clutch_leverage.py --fetch     # refresh raw/rally_cache.csv
    python model/clutch_leverage.py             # measure + validate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = ROOT / "raw"
sys.path.insert(0, str(ROOT / "web"))

from sitelib.winprob import ServeDP, A1, A2, B1, K_DOUBLES  # noqa: E402

CACHE = Path(os.environ.get("CLUTCH_RALLY_CSV", RAW / "rally_cache.csv"))

# Games whose reconstructed final score is outside this band are log
# corruption (a handful run to 50+) or retirements, and are dropped.
MIN_WIN_SCORE, MAX_WIN_SCORE = 11, 20


# ----------------------------------------------------------------- fetch --


def fetch(out: Path = CACHE) -> int:
    """Date-partitioned pull of pb_rally (PostgREST caps pages at 1000 rows,
    and deep OFFSET times out, so we page inside each match_date)."""
    import datetime as dt
    import gzip
    import threading
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    base = os.environ.get(
        "SUPABASE_URL", "https://nwgxyytowbluuykbdcfc.supabase.co").rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
    if not key:
        raise SystemExit("set SUPABASE_ANON_KEY (anon key is public read)")
    cols = ("match_id,discipline,tour,match_date,game_number,rally_number,"
            "server_uuid,receiver_uuid,server_side,server_number,outcome,won,"
            "server_score,receiver_score")

    def get(q):
        req = urllib.request.Request(f"{base}/rest/v1{q}")
        req.add_header("apikey", key)
        req.add_header("Authorization", "Bearer " + key)
        req.add_header("Accept", "text/csv")
        req.add_header("Accept-Encoding", "gzip")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return raw.decode("utf-8")
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    start, end = dt.date(2024, 1, 1), dt.date.today() + dt.timedelta(days=1)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days)]
    lock, rows = threading.Lock(), []

    def one(d):
        off, got_all = 0, []
        while True:
            txt = get(f"/pb_rally?select={cols}&match_date=eq.{d.isoformat()}"
                      f"&order=match_id.asc,game_number.asc,rally_number.asc"
                      f"&limit=1000&offset={off}")
            lines = txt.splitlines()
            if len(lines) <= 1:
                break
            got_all.extend(lines[1:])
            off += len(lines) - 1
            if len(lines) - 1 < 1000:
                break
        with lock:
            rows.extend(got_all)

    hdr = get(f"/pb_rally?select={cols}&limit=1").splitlines()[0]
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(one, days))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(hdr + "\n")
        fh.write("\n".join(rows) + "\n")
    return len(rows)


# ------------------------------------------------------------- leverage --


def doubles_leverage(k: float, T: int) -> dict:
    """lev[(a, b, server_number)] for a game to T between equal sides."""
    dp = ServeDP(0.0, k, T)
    out = {}
    cap = T + 20
    for a in range(cap):
        for b in range(cap):
            for sn in (1, 2):
                st = A1 if sn == 1 else A2
                win = dp.p(a + 1, b, st)
                lose = dp.p(a, b, A2) if sn == 1 else dp.p(a, b, B1)
                out[(a, b, sn)] = abs(win - lose)
    return out


def singles_table(k: float, T: int, cap: int) -> dict:
    """f[(a, b)] = P(the serving side wins) in side-out singles to T.

    f(a,b) = k f(a+1,b) + (1-k)(1 - f(b,a)); f(a,b) and f(b,a) close a
    2-cycle at equal total score, so solve the pair algebraically and induct
    downward on a+b."""
    f = {}

    def done(a, b):
        if a >= T and a - b >= 2:
            return 1.0
        if b >= T and b - a >= 2:
            return 0.0
        return None

    def get(a, b):
        d = done(a, b)
        if d is not None:
            return d
        if a >= cap or b >= cap:
            return 0.5
        return f[(a, b)]

    q = 1.0 - k
    for tot in range(2 * cap - 2, -1, -1):
        for a in range(max(0, tot - cap + 1), min(cap - 1, tot) + 1):
            b = tot - a
            if b < 0 or b > cap - 1 or done(a, b) is not None:
                continue
            # f(a,b) = k*get(a+1,b) + q - q*f(b,a)
            # f(b,a) = k*get(b+1,a) + q - q*f(a,b)
            ca = k * get(a + 1, b) + q
            cb = k * get(b + 1, a) + q
            f[(a, b)] = (ca - q * cb) / (1 - q * q)
    return f


def singles_leverage(k: float, T: int) -> dict:
    cap = T + 20
    f = singles_table(k, T, cap)

    def val(a, b):
        if a >= T and a - b >= 2:
            return 1.0
        if b >= T and b - a >= 2:
            return 0.0
        return f.get((a, b), 0.5)

    out = {}
    for a in range(cap):
        for b in range(cap):
            win = val(a + 1, b)
            lose = 1.0 - val(b, a)      # side out: opponent now serves
            out[(a, b, 0)] = abs(win - lose)
    return out


# ----------------------------------------------------------------- load --


def load(path: Path = CACHE):
    """Read the rally cache, drop unusable rows/games, resolve each game's
    target score, and attach leverage. Returns a dict of numpy arrays."""
    if not path.exists():
        raise SystemExit(f"no rally cache at {path}; run with --fetch")

    mid, disc, tour, date, gnum = [], [], [], [], []
    srv, rcv, snum, won, sa, sb, sside = [], [], [], [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not r["server_score"] or not r["receiver_score"]:
                continue
            if not r["server_uuid"] or not r["receiver_uuid"]:
                continue
            sn = r["server_number"]
            if r["discipline"] == "doubles":
                if sn not in ("1", "2"):
                    continue
                sn = int(sn)
            else:
                sn = 0
            mid.append(r["match_id"])
            disc.append(r["discipline"])
            tour.append(r["tour"])
            date.append(r["match_date"])
            gnum.append(int(r["game_number"]))
            srv.append(r["server_uuid"].lower())
            rcv.append(r["receiver_uuid"].lower())
            snum.append(sn)
            won.append(int(r["won"]))
            sa.append(int(r["server_score"]))
            sb.append(int(r["receiver_score"]))
            sside.append(int(r["server_side"]) if r["server_side"] else 0)

    n = len(mid)
    d = dict(
        match=np.array(mid), disc=np.array(disc), tour=np.array(tour),
        date=np.array(date), game=np.array(gnum, dtype=np.int32),
        server=np.array(srv), receiver=np.array(rcv),
        snum=np.array(snum, dtype=np.int8), won=np.array(won, dtype=np.int8),
        a=np.array(sa, dtype=np.int16), b=np.array(sb, dtype=np.int16),
        side=np.array(sside, dtype=np.int8))

    # --- game ids and target resolution -------------------------------
    gkey = np.char.add(np.char.add(d["match"], "#"), d["game"].astype(str))
    guniq, gidx = np.unique(gkey, return_inverse=True)
    ng = len(guniq)

    after_hi = np.maximum(d["a"] + d["won"], d["b"])
    hi = np.zeros(ng, dtype=np.int32)
    np.maximum.at(hi, gidx, after_hi)

    # to-11 vs to-15: a to-11 game can only reach 15 as 15-13, and to-15
    # games are a small, real PPA-Challenger population (see CLAUDE.md);
    # 15 is assigned to-15, which mislabels the rare 15-13 deuce.
    tgt = np.where(hi <= 14, 11, 15).astype(np.int32)
    ok_game = (hi >= MIN_WIN_SCORE) & (hi <= MAX_WIN_SCORE)

    d["gidx"] = gidx
    d["target"] = tgt[gidx]
    # a handful of logs carry negative or runaway running scores (referee
    # corrections rewound past zero); they are unusable, not informative.
    sane = (d["a"] >= 0) & (d["b"] >= 0) & (d["a"] < MAX_WIN_SCORE + 15) \
        & (d["b"] < MAX_WIN_SCORE + 15)
    keep = ok_game[gidx] & sane

    for k_ in list(d):
        d[k_] = d[k_][keep]
    d["gidx"] = np.unique(d["gidx"], return_inverse=True)[1]

    # --- leverage -----------------------------------------------------
    is_s = d["disc"] == "singles"
    k_dbl = float(d["won"][~is_s].mean())
    k_sgl = float(d["won"][is_s].mean()) if is_s.any() else 0.5
    d["k_doubles"], d["k_singles"] = k_dbl, k_sgl

    tables = {}
    for T in (11, 15):
        tables[("doubles", T)] = doubles_leverage(k_dbl, T)
        tables[("singles", T)] = singles_leverage(k_sgl, T)

    lev = np.empty(len(d["a"]), dtype=np.float64)
    for T in (11, 15):
        for dd in ("doubles", "singles"):
            m = (d["target"] == T) & ((d["disc"] == "singles") == (dd == "singles"))
            if not m.any():
                continue
            tab = tables[(dd, T)]
            lev[m] = [tab[(int(x), int(y), int(s))]
                      for x, y, s in zip(d["a"][m], d["b"][m], d["snum"][m])]
    d["lev_raw"] = lev
    d["lev"] = lev / lev.std()
    return d


# ------------------------------------------------------- the estimator --


def cells(gi, x, y):
    """Within-cell sums for every group: n, Sxx, Syy, Sxy."""
    m = gi.max() + 1
    n = np.bincount(gi, minlength=m).astype(np.float64)
    sx = np.bincount(gi, weights=x, minlength=m)
    sy = np.bincount(gi, weights=y, minlength=m)
    sxx = np.bincount(gi, weights=x * x, minlength=m)
    syy = np.bincount(gi, weights=y * y, minlength=m)
    sxy = np.bincount(gi, weights=x * y, minlength=m)
    with np.errstate(invalid="ignore", divide="ignore"):
        Sxx = sxx - sx * sx / n
        Syy = syy - sy * sy / n
        Sxy = sxy - sx * sy / n
    return gi, n, Sxx, Syy, Sxy


def player_stats(players, cellcode, lev, out, npl):
    """Aggregate the within-cell covariance up to one row per player.

    `players` are integer player codes, `cellcode` an integer id unique to
    (player, game, channel). Returns arrays of length `npl`, indexed by
    player code."""
    key = np.unique(cellcode, return_inverse=True)[1]
    gi, n, Sxx, Syy, Sxy = cells(key, lev, out)

    # one player per cell, so map cells -> player by first occurrence
    first = np.zeros(gi.max() + 1, dtype=np.int64)
    first[gi[::-1]] = np.arange(len(gi), dtype=np.int64)[::-1]
    cell_player = players[first]

    use = (n >= 2) & (Sxx > 0)
    V = np.zeros_like(Sxx)
    V[use] = Sxx[use] * Syy[use] / (n[use] - 1.0)

    agg = lambda w: np.bincount(cell_player[use], weights=w[use], minlength=npl)
    return {"U": agg(Sxy), "SSL": agg(Sxx), "V": agg(V), "n": agg(n),
            "cells": np.bincount(cell_player[use], minlength=npl).astype(float)}


def combine(*stats):
    """Sum independent channels (disjoint rally sets) player-by-player."""
    out = {k: sum(s[k] for s in stats)
           for k in ("U", "SSL", "V", "n", "cells")}
    return out


def slopes(st, min_rallies=0):
    ok = (st["SSL"] > 0) & (st["V"] > 0) & (st["n"] >= min_rallies)
    b = np.where(ok, st["U"] / np.where(ok, st["SSL"], 1), np.nan)
    se = np.where(ok, np.sqrt(np.where(ok, st["V"], 1)) / np.where(ok, st["SSL"], 1), np.nan)
    return b, se, ok


# ------------------------------------------------------ empirical Bayes --


def eb(b, se):
    """Normal-normal marginal ML for (mu, tau); returns posterior mean/sd."""
    v = se ** 2

    def nll(mu, t2):
        s = v + t2
        return 0.5 * np.sum(np.log(s) + (b - mu) ** 2 / s)

    lo, hi = 0.0, max(1e-12, float(np.var(b)) * 4)
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        mu1 = np.sum(b / (v + m1)) / np.sum(1 / (v + m1))
        mu2 = np.sum(b / (v + m2)) / np.sum(1 / (v + m2))
        if nll(mu1, m1) < nll(mu2, m2):
            hi = m2
        else:
            lo = m1
    t2 = 0.5 * (lo + hi)
    mu = np.sum(b / (v + t2)) / np.sum(1 / (v + t2))
    shrink = t2 / (t2 + v)
    post = mu + shrink * (b - mu)
    post_sd = np.sqrt(shrink * v)
    return mu, math.sqrt(max(t2, 0.0)), post, post_sd, shrink


def tau_ci(b, se, reps=400, seed=7):
    """Parametric bootstrap CI for tau (resample players)."""
    rng = np.random.default_rng(seed)
    out = []
    n = len(b)
    for _ in range(reps):
        j = rng.integers(0, n, n)
        try:
            out.append(eb(b[j], se[j])[1])
        except Exception:
            pass
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ------------------------------------------------------------- helpers --


def names():
    nm, gd = {}, {}
    with open(DATA / "players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            nm[u] = r["full_name"]
            gd[u] = r.get("gender", "")
    return nm, gd


def v2_values():
    out = {}
    with open(DATA / "v2_players.csv") as fh:
        for r in csv.DictReader(fh):
            out[r["player_id"].lower()] = float(r["value_now_mean"])
    return out


def wcorr(x, y, w=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if w is not None:
        w = w[m]
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        cxy = np.average((x - mx) * (y - my), weights=w)
        return cxy / math.sqrt(np.average((x - mx) ** 2, weights=w)
                              * np.average((y - my) ** 2, weights=w))
    return float(np.corrcoef(x, y)[0, 1])


def boot_corr(x, y, reps=2000, seed=11):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = [np.corrcoef(x[j], y[j])[0, 1]
           for j in (rng.integers(0, len(x), len(x)) for _ in range(reps))]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def wls(x, y, w):
    """Weighted slope/intercept of y on x, plus the slope's se."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    sxx = np.sum(w * (x - mx) ** 2)
    slope = np.sum(w * (x - mx) * (y - my)) / sxx
    resid = y - (my + slope * (x - mx))
    s2 = np.sum(w * resid ** 2) / (len(x) - 2)
    return float(slope), float(math.sqrt(s2 / sxx)), int(len(x))


# ------------------------------------------------------------- channels --


class Frame:
    """Rally table reduced to integer codes, ready for grouped estimation."""

    def __init__(self, d):
        allu = np.concatenate([d["server"], d["receiver"]])
        self.uuids, inv = np.unique(allu, return_inverse=True)
        n = len(d["server"])
        self.srv_code = inv[:n]
        self.rcv_code = inv[n:]
        self.npl = len(self.uuids)
        self.d = d
        self.year = np.array([s[:4] for s in d["date"]])

    def stats(self, mask, channel):
        """channel 'S' = serving rallies, 'R' = receiving rallies."""
        d = self.d
        who = self.srv_code[mask] if channel == "S" else self.rcv_code[mask]
        y = d["won"][mask].astype(np.float64)
        if channel == "R":
            y = 1.0 - y
        cell = who.astype(np.int64) * (d["gidx"].max() + 1) + d["gidx"][mask]
        return player_stats(who, cell, d["lev"][mask], y, self.npl)

    def permuted(self, mask, channel, rng):
        """Same statistic with leverage shuffled inside each cell — the exact
        null. Used to verify the closed-form permutation variance."""
        d = self.d
        who = self.srv_code[mask] if channel == "S" else self.rcv_code[mask]
        y = d["won"][mask].astype(np.float64)
        if channel == "R":
            y = 1.0 - y
        cell = who.astype(np.int64) * (d["gidx"].max() + 1) + d["gidx"][mask]
        lev = d["lev"][mask]
        order = np.lexsort((rng.random(len(cell)), cell))
        back = np.lexsort((np.arange(len(cell)), cell))
        shuffled = np.empty_like(lev)
        shuffled[back] = lev[order]
        return player_stats(who, cell, shuffled, y, self.npl)


# ---------------------------------------------------------------- main --


MIN_RALLIES = 400


def report(d, out_json=None, out_csv=None):
    nm, gd = names()
    val = v2_values()
    F = Frame(d)
    dbl = d["disc"] == "doubles"
    sgl = d["disc"] == "singles"
    y26 = F.year == "2026"

    def pack(masks_channels):
        return combine(*[F.stats(m, c) for m, c in masks_channels])

    S = {
        "dbl_S": F.stats(dbl, "S"), "dbl_R": F.stats(dbl, "R"),
        "sgl_S": F.stats(sgl, "S"), "sgl_R": F.stats(sgl, "R"),
    }
    S["dbl"] = combine(S["dbl_S"], S["dbl_R"])
    S["sgl"] = combine(S["sgl_S"], S["sgl_R"])
    S["all"] = combine(S["dbl"], S["sgl"])
    S["dbl_pre26"] = pack([(dbl & ~y26, "S"), (dbl & ~y26, "R")])
    S["dbl_26"] = pack([(dbl & y26, "S"), (dbl & y26, "R")])
    S["all_pre26"] = pack([(~y26, "S"), (~y26, "R")])
    S["all_26"] = pack([(y26, "S"), (y26, "R")])

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 72)
    say("CLUTCH BY LEVERAGE COVARIANCE — full archive, both channels")
    say("=" * 72)
    say(f"rallies {len(d['a']):,}   games {d['gidx'].max()+1:,}   "
        f"matches {len(np.unique(d['match'])):,}")
    say(f"doubles {int(dbl.sum()):,}  singles {int(sgl.sum()):,}   "
        f"k_doubles {d['k_doubles']:.4f}  k_singles {d['k_singles']:.4f}")
    say(f"leverage: mean {d['lev_raw'].mean():.4f} sd {d['lev_raw'].std():.4f} "
        f"max {d['lev_raw'].max():.4f} (win-prob swing per rally)")
    say()

    # --- 0. does the closed-form null variance hold? ------------------
    rng = np.random.default_rng(SEED)
    pv = []
    for i in range(3):
        st = combine(F.permuted(dbl, "S", rng), F.permuted(dbl, "R", rng))
        b0, se0, ok0 = slopes(st, MIN_RALLIES)
        pv.append(float(np.var((b0 / se0)[ok0])))
    say("[0] NULL CALIBRATION (leverage shuffled within each player-game)")
    say(f"    var(z) under the exact permutation null: "
        f"{', '.join(f'{v:.3f}' for v in pv)}   (target 1.000)")
    say()

    # --- 1. existence -------------------------------------------------
    say("[1] EXISTENCE — is the field wider than chance?")
    hdr = f"    {'arm':<22}{'players':>8}{'var(z)':>9}{'tau':>9}{'tau 95% CI':>18}"
    say(hdr)
    tau_store = {}
    for key, label in (("dbl", "doubles (S+R)"), ("dbl_S", "doubles serve only"),
                       ("dbl_R", "doubles return only"), ("sgl", "singles (S+R)"),
                       ("all", "everything")):
        b, se, ok = slopes(S[key], MIN_RALLIES)
        bb, ss = b[ok], se[ok]
        mu, tau, post, psd, shr = eb(bb, ss)
        lo, hi = tau_ci(bb, ss)
        tau_store[key] = (mu, tau, lo, hi)
        say(f"    {label:<22}{ok.sum():>8}{np.var(bb/ss):>9.3f}"
            f"{tau:>9.4f}   [{lo:.4f}, {hi:.4f}]")
    say("    tau = population sd of TRUE clutch, in extra rally-win")
    say("    probability per +1 sd of leverage.")
    say()

    # --- 2. reliability: three independent splits ---------------------
    say("[2] RELIABILITY — does one slice of a player predict another?")
    say(f"    {'split':<34}{'n':>6}{'r':>8}{'95% CI':>18}{'calib slope':>14}")

    def rel(k1, k2, label, minr=MIN_RALLIES):
        b1, s1, o1 = slopes(S[k1], minr)
        b2, s2, o2 = slopes(S[k2], minr)
        o = o1 & o2
        if o.sum() < 15:
            say(f"    {label:<34}{o.sum():>6}   (too few players)")
            return None
        mu, tau, post, psd, shr = eb(b1[o], s1[o])
        r = float(np.corrcoef(b1[o], b2[o])[0, 1])
        lo, hi = boot_corr(b1[o], b2[o])
        sl, sle, n = wls(post, b2[o], 1.0 / s2[o] ** 2)
        say(f"    {label:<34}{o.sum():>6}{r:>8.3f}   [{lo:+.3f}, {hi:+.3f}]"
            f"   {sl:>6.2f} ± {sle:.2f}")
        return r, sl, sle

    rel("dbl_S", "dbl_R", "serve vs return (doubles, disjoint)")
    rel("dbl_pre26", "dbl_26", "2024-25 vs 2026 (doubles)")
    rel("sgl_S", "sgl_R", "serve vs return (singles, disjoint)")
    rel("dbl", "sgl", "doubles vs singles (same player)", minr=300)
    rel("all_pre26", "all_26", "2024-25 vs 2026 (all rallies)")
    say("    calib slope = weighted regression of the held-out estimate on the")
    say("    SHRUNK estimate from the other slice. 1.00 = honestly scaled.")
    say()

    # --- 3. skill confound -------------------------------------------
    b, se, ok = slopes(S["all"], MIN_RALLIES)
    b, se = b[ok], se[ok]
    mu, tau, post, psd, shr = eb(b, se)
    uu = F.uuids[ok]
    v = np.array([val.get(u, np.nan) for u in uu])
    have = np.isfinite(v)
    say("[3] IS IT JUST BEING GOOD?")
    say(f"    corr(shrunk clutch, v2 rating) = "
        f"{wcorr(post[have], v[have]):+.3f}  (n={have.sum()})")
    say()

    # --- 4. the leaderboard ------------------------------------------
    nr = S["all"]["n"][ok]
    gnd = np.array([gd.get(u, "") for u in uu])
    nam = np.array([nm.get(u, u[:8]) for u in uu])
    zsh = post / psd
    order = np.argsort(-post)

    def table(sel, title, top=12, asc=False):
        say(title)
        say(f"    {'#':>2} {'player':<24}{'rallies':>9}{'raw b':>9}"
            f"{'shrunk':>9}{'95% CI':>20}")
        idx = np.argsort(post[sel]) if asc else np.argsort(-post[sel])
        ii = np.where(sel)[0][idx][:top]
        for rank, i in enumerate(ii, 1):
            lo, hi = post[i] - 1.96 * psd[i], post[i] + 1.96 * psd[i]
            say(f"    {rank:>2} {nam[i]:<24}{int(nr[i]):>9,}{b[i]:>+9.4f}"
                f"{post[i]:>+9.4f}   [{lo:+.4f},{hi:+.4f}]")
        say()

    big = nr >= 3000
    table(big & (gnd == "M"), "[4] MOST CLUTCH — men (>=3,000 rallies)")
    table(big & np.isin(gnd, ("F", "W")),
          "    MOST CLUTCH — women (>=3,000 rallies)")

    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["player_id", "name", "gender", "rallies", "cells",
                        "b_raw", "se", "z", "clutch_shrunk", "shrunk_sd",
                        "shrinkage", "v2_value"])
            for i in np.argsort(-post):
                w.writerow([uu[i], nam[i], gnd[i], int(nr[i]),
                            int(S["all"]["cells"][ok][i]),
                            f"{b[i]:.6f}", f"{se[i]:.6f}",
                            f"{b[i]/se[i]:.3f}", f"{post[i]:.6f}",
                            f"{psd[i]:.6f}", f"{shr[i]:.3f}",
                            "" if not np.isfinite(v[i]) else f"{v[i]:.4f}"])
        say(f"wrote {out_csv}")

    if out_json:
        with open(out_json, "w") as fh:
            json.dump({"rallies": int(len(d["a"])),
                       "games": int(d["gidx"].max() + 1),
                       "k_doubles": d["k_doubles"],
                       "k_singles": d["k_singles"],
                       "tau": {k: {"mu": t[0], "tau": t[1],
                                   "ci": [t[2], t[3]]}
                               for k, t in tau_store.items()},
                       "null_var_z": pv}, fh, indent=1)
    return lines


SEED = 20260803

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="refresh raw/rally_cache.csv from Supabase")
    a = ap.parse_args()
    if a.fetch:
        print(f"fetched {fetch():,} rallies -> {CACHE}")
    report(load())
    print()
    print("!" * 72)
    print("THESE NUMBERS ARE UNCORRECTED AND ABOUT 75% ARTIFACT.")
    print("Side-out scoring ends every service run with exactly one loss, at")
    print("the run's highest leverage, which manufactures this covariance at a")
    print("true effect of zero — and the fake effect grows with the player's")
    print("own rally-win rate, so the leaderboard above is mostly a ranking of")
    print("how good people are. See model/clutch_mechanical.py for the proof")
    print("and model/clutch_report.py for the corrected answer.")
    print("!" * 72)
