"""B6 part 1c — within-event PAIRED version of the extra-upset test, plus the
reliability-slope contrast.

Phase 1's strongest methodological complaint about the weather thread was
that its windy-vs-calm contrasts were UNPAIRED, although nearly every windy
event also supplies calm games.  This runs the upset test in paired form:

    y_i = (won_i − p_i) · s_i        s_i = +1 if team 1 is the underdog,
                                          −1 if team 1 is the favourite
    y_i = alpha_event + gamma_decile + beta · 1[wind_i >= W] + e_i

so beta is the extra upset rate in wind measured WITHIN events and WITHIN
deciles of the predicted win probability.  Event effects are removed by
within-transformation, decile effects by dummies.  Second block: the
logistic reliability slope in windy 14+ minus the same in calm <8, per arm —
beta < 1 means over-confidence, and a NEGATIVE contrast is the variance
channel's signature.  Cluster bootstrap over events, seeded.

    python model/weather_review/b6_upset_paired.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b6_lib as L  # noqa: E402
from b6_lib import ROOT, sigmoid  # noqa: E402

NBOOT = 600
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


class Frame:
    """Column-store so a bootstrap resample is pure integer indexing."""

    def __init__(self):
        sm = L.ShareMoments(n=481)
        ev, setting, wind, y, pwin, won = [], [], [], [], [], []
        for g in L.load_games():
            if g["wind"] is None or g["setting"] not in ("outdoor", "indoor"):
                continue
            p = min(max(sigmoid(g["eta"]), 0.16), 0.84)
            pw = sm.win(p, g["T"])
            s = 1.0 if pw < 0.5 else -1.0
            w = 1.0 if g["won"] else 0.0
            ev.append(g["event"])
            setting.append(g["setting"])
            wind.append(g["wind"])
            pwin.append(pw)
            won.append(w)
            y.append((w - pw) * s)
        keys = sorted(set(ev))
        kid = {k: i for i, k in enumerate(keys)}
        self.ev = np.array([kid[e] for e in ev])
        self.nev = len(keys)
        self.out = np.array([s == "outdoor" for s in setting])
        self.wind = np.array(wind, float)
        self.y = np.array(y, float)
        self.pwin = np.array(pwin, float)
        self.won = np.array(won, float)
        self.lg = np.log(self.pwin / (1 - self.pwin))
        # decile dummies of the predicted win prob, fixed once
        qs = np.quantile(self.pwin, np.linspace(0, 1, 11)[1:-1])
        self.dec = np.searchsorted(qs, self.pwin)
        self.by_event = [np.flatnonzero(self.ev == i) for i in range(self.nev)]


def paired_beta(F, idx, wlo):
    """Within-event, within-decile OLS coefficient on the windy dummy."""
    if idx.size < 200:
        return np.nan
    ev = F.ev[idx]
    cols = [(F.wind[idx] >= wlo).astype(float)]
    for k in range(10):
        cols.append((F.dec[idx] == k).astype(float))
    X = np.column_stack(cols)
    y = F.y[idx].copy()
    # within-event demeaning, vectorised
    uniq, inv = np.unique(ev, return_inverse=True)
    cnt = np.bincount(inv, minlength=uniq.size).astype(float)
    for j in range(X.shape[1]):
        m = np.bincount(inv, weights=X[:, j], minlength=uniq.size) / cnt
        X[:, j] -= m[inv]
    m = np.bincount(inv, weights=y, minlength=uniq.size) / cnt
    y -= m[inv]
    try:
        return float(np.linalg.lstsq(X, y, rcond=None)[0][0])
    except np.linalg.LinAlgError:
        return np.nan


def rel_slope(F, idx):
    if idx.size < 150:
        return np.nan
    X = np.column_stack([np.ones(idx.size), F.lg[idx]])
    yv = F.won[idx]
    b = np.zeros(2)
    for _ in range(40):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        try:
            step = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (yv - p))
        except np.linalg.LinAlgError:
            return np.nan
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(b[1])


def stat(F, idx):
    o = idx[F.out[idx]]
    i = idx[~F.out[idx]]
    v = []
    for wlo in (14, 12):
        a = paired_beta(F, o, wlo)
        b = paired_beta(F, i, wlo)
        v += [a, b, a - b]
    for arm in (o, i):
        bw = rel_slope(F, arm[F.wind[arm] >= 14])
        bc = rel_slope(F, arm[F.wind[arm] < 8])
        v.append(bw - bc)
    v.append(v[-2] - v[-1])
    return v


def main():
    F = Frame()
    allidx = np.arange(F.y.size)
    base = stat(F, allidx)
    rng = np.random.default_rng(1234)
    draws = []
    for _ in range(NBOOT):
        pick = rng.integers(0, F.nev, F.nev)
        idx = np.concatenate([F.by_event[j] for j in pick])
        v = stat(F, idx)
        if np.all(np.isfinite(v)):
            draws.append(v)
    d = np.array(draws)
    lo = np.percentile(d, 2.5, axis=0)
    hi = np.percentile(d, 97.5, axis=0)

    say("# B6 — extra upset rate in wind, WITHIN-EVENT paired\n")
    say(f"y = (win − predicted win prob) signed so positive = the underdog "
        f"beat its prediction. Event and predicted-probability-decile effects "
        f"removed; the coefficient is on 1[wind ≥ W]. {len(draws)} usable "
        f"resamples of the {F.nev} events (cluster bootstrap, seed 1234).\n")
    say("| setting | threshold | games ≥ W | paired extra upset rate "
        "[95% CI] |")
    say("|---|---|---|---|")
    labels = [("outdoor", 14, 0), ("indoor", 14, 1),
              ("**outdoor − indoor**", 14, 2),
              ("outdoor", 12, 3), ("indoor", 12, 4),
              ("**outdoor − indoor**", 12, 5)]
    for name, wlo, k in labels:
        if name.startswith("**"):
            n = int((F.wind >= wlo).sum())
        elif name == "outdoor":
            n = int(((F.wind >= wlo) & F.out).sum())
        else:
            n = int(((F.wind >= wlo) & ~F.out).sum())
        say(f"| {name} | ≥{wlo} mph | {n:,} | {100*base[k]:+.2f} pp "
            f"[{100*lo[k]:+.2f}, {100*hi[k]:+.2f}] |")
    say("")
    say("## Reliability-slope contrast (windy 14+ minus calm <8)\n")
    say("Logistic recalibration slope beta of the outcome on logit(p_v2). "
        "beta < 1 = over-confident. A NEGATIVE contrast means predictions get "
        "relatively MORE over-confident in wind — the variance channel's "
        "signature. Absolute levels are not interpretable (retroactive "
        "current-form ratings inflate beta everywhere); the contrast is.\n")
    say("| arm | Δbeta (windy − calm) [95% CI] |")
    say("|---|---|")
    for name, k in (("outdoor", 6), ("indoor", 7),
                    ("**outdoor − indoor**", 8)):
        say(f"| {name} | {base[k]:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}] |")
    say("")
    (ROOT / "model/weather_review/b6_upset_paired.md").write_text(
        "\n".join(OUT) + "\n")
    print("wrote model/weather_review/b6_upset_paired.md")


if __name__ == "__main__":
    main()
