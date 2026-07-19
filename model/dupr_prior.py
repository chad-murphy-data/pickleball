"""DUPR-informed prior spec for v2 (the 'heavy but correct' unknown-player fix).

Every player's base value in fit_v2 currently uses a zero-mean prior, so a
low-data player shrinks toward the FIELD AVERAGE. That throws away a real
signal: DUPR. This module fits the per-gender DUPR->value line from
well-observed players and writes data/dupr_prior.json:

  * coef: per-gender (and pooled) intercept+slope, so ANY player with a DUPR
    gets a value-scale prior mean; used both to set fit_v2's prior means
    (SRM2_DUPR_PRIOR=1) and to impute never-seen players at prediction time.
  * resid_sd: scatter around the line -> how wide the prior should be.
  * mu: per-UUID prior mean for every player we have a (non-glitch) DUPR for.

Design decisions (see session notes):
  * Per-gender lines, NOT a flat 5.5 (the gender slopes differ enough that a
    single number lands above-average for women / below for men) and NOT the
    field-lowest DUPR (unstable + systematically harsh for a main-draw entry).
  * No-DUPR players fall back to the per-gender average (mu = 0 here, i.e.
    the model's existing field-average prior) and lean on wide uncertainty.
  * Glitched DUPR (<=3.65 reset default) is dropped, per the house rules.

    python model/dupr_prior.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DYN_MIN = 60
GLITCH = 3.65


def load():
    val, gen, games = {}, {}, {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        u = r["player_id"].lower()
        val[u] = float(r["value_now_mean"])
        gen[u] = r["gender"]
        games[u] = int(r["games"])
    dupr = {}
    for r in csv.DictReader((DATA / "platform_ratings.csv").open()):
        try:
            dupr[r["player_id"].lower()] = float(r["platform_rating_latest"])
        except (ValueError, KeyError):
            pass
    return val, gen, games, dupr


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) or 1.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = (sum(e * e for e in res) / max(n - 2, 1)) ** 0.5
    return a, b, mx, sd, n


def fit_line(val, gen, games, dupr, want):
    xs, ys = [], []
    for u, d in dupr.items():
        if u not in val or d <= GLITCH or games.get(u, 0) < DYN_MIN:
            continue
        if want is not None and gen.get(u) != want:
            continue
        xs.append(d); ys.append(val[u])
    return ols(xs, ys)


def main():
    val, gen, games, dupr = load()
    coef, resid, center = {}, {}, {}
    for key, want in (("F", "F"), ("M", "M"), ("pooled", None)):
        a, b, mx, sd, n = fit_line(val, gen, games, dupr, want)
        coef[key] = [round(a, 4), round(b, 4)]
        resid[key] = round(sd, 4)
        center[key] = round(mx, 3)

    def line(g):
        return coef[g] if g in coef else coef["pooled"]

    # per-UUID prior mean for everyone with a usable DUPR
    mu = {}
    for u, d in dupr.items():
        if d <= GLITCH:
            continue
        a, b = line(gen.get(u, "pooled"))
        mu[u] = round(a + b * d, 4)

    out = {
        "note": "Per-gender DUPR->value prior. mu[uuid] is the value-scale "
                "prior mean; players absent here default to 0 (field average). "
                "resid_sd is the scatter around each line (min prior width). "
                "Glitched DUPR (<=3.65) dropped.",
        "coef": coef, "resid_sd": resid, "dupr_center": center,
        "glitch_below": GLITCH, "n_with_prior": len(mu), "mu": mu,
    }
    (DATA / "dupr_prior.json").write_text(json.dumps(out, indent=1))
    print(f"wrote data/dupr_prior.json  ({len(mu)} players get a DUPR prior)")
    for g in ("F", "M"):
        a, b = coef[g]
        print(f"  {g}: value = {a:+.3f} + {b:.3f}*DUPR  (resid_sd {resid[g]}, "
              f"center {center[g]})")


if __name__ == "__main__":
    main()
