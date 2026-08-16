"""Does partnership SURVIVORSHIP hide chemistry?

    python model/chem_survival.py             # full battery -> printed report + JSON
    python model/chem_survival.py --fast      # fewer replicates (smoke)

THE HYPOTHESIS (user, 2026-08-16).  Finding 2 says chemistry is small
(sd_d = 0.053 logit ~ 0.013 share ~ 0.25 pts).  One story for WHY we
can't find it: bad partnerships don't last.  Pairs that click keep
playing; pairs that don't dissolve after a handful of games, so the
negative tail never accumulates enough games to be seen and the fitted
sd underestimates the true spread across pairings people TRY.
analysis.md already carries the seed of this: the unshrunk fixed-effects
check found survivor dyads' t-stats mean +0.44 ("hints at survivorship")
-- never adjudicated.

TWO MECHANISMS, SEPARATED UP FRONT:

  (M1) Selection on OBSERVED outcomes: teams drop pairs that lose.  The
       lost games are still in the archive; only the pair's FUTURE games
       are censored.  Survivor-only summaries are range-restricted and
       backward-looking survivor means are biased by the very luck that
       kept the pair alive (the stopped-mean bias -- analysis.md's +0.44
       is exactly that statistic), but the games that were played are
       clean draws.
  (M2) Selection on PRIVATE information: practice / eye test.  Pairs
       with bad true chemistry get few or zero official games regardless
       of official results; zero-game pairs are invisible and nothing
       here can measure them.  What can be measured is the tried-and-
       dissolved mass, which is large (arm 1), plus one quasi-
       experiment: MLP gender doubles, where the pair is FORCED by the
       2M+2W roster and cannot dissolve within a season.

THE DISCRIMINATING INSTRUMENT (arm 2).  The FORWARD tenure curve: the
mean cleaned residual of a pair's NEXT game as a function of how many
games they have already played together.  Continuation decisions can
only condition on PAST games, and under no-chemistry the past carries no
information about the next game's residual, so the curve is FLAT AT ZERO
under any outcome-based selection, however aggressive -- no simulation
needed for that null.  If chemistry is real and selection retains
good-chemistry pairs, survivors are an enriched sample and the curve
RISES with tenure.  Two panels: ALL games, and the PRIMARY all-four-
players-dynamic panel (every player >= 60 career games, i.e. carries a
v2 monthly curve): the all-games bucket-0 estimate is confounded by
thin-player rating shrinkage (never-seen players enter at the prior
mean), which the dynamic panel removes.  v2's beta_new (+0.088 logit
over an eventually->=15-game dyad's first 6 games, finding 5) lives in
the 0 / 1-5 buckets; the pre-specified ENRICHMENT contrast is buckets
>=12 vs 6-11, clear of the newness window.  A within-dyad shuffle
(permute which of a dyad's games sit at which tenure slot) splits the
curve into composition vs within-pair-order parts for description; under
selection each part is individually biased (survivor luck sits in the
composition and its mirror image in the order part) and only their sum
-- the observed forward curve -- has the analytic null.

THE ESTIMATOR LESSONS THIS SCRIPT LEARNED THE HARD WAY (kept in full
because every one of these wrong turns produced a CONFIDENT number, and
the next session will be tempted to "simplify" back into one of them).

  (i)   Rating error is constant within a player and nests inside every
        one of their dyads, so per-player offsets must be stripped --
        else offsets-of-rating-error pose as chemistry and as tenure
        structure.
  (ii)  But offsets fit on ALL of a player's games CHASE the player's
        dyad effects (the see-saw rule as an estimator bug): injected
        iid chemistry came back with lambda ~ 2-4 through naive offsets
        while a no-recleaning probe returned 0.94.  Fix: LEAVE-OWN-
        DYAD-OUT (LODO) offsets -- a dyad's residuals are never cleaned
        with information from the dyad itself.
  (iii) A single fit's offset-estimation ERROR is a constant shared by
        both halves of any within-dyad product, landing in it as
        +Var(error) -- it inflated the 3-5-game class ~20x and the ALL
        line by ~+600e-6.  Fix: CROSS-FITTING -- partition events, clean
        each half with the other half's nuisance fit, so the two halves'
        cleaning errors come from disjoint games.  The un-crossfit line
        is kept as a printed receipt.
  (iv)  v2's monthly walk TRACKS form.  Selection keeps lucky pairs,
        both partners' values absorb that luck, and their later games
        underperform the inflated values: a NEGATIVE bias that poses as
        anti-chemistry in short-lag products (random event halves came
        out at -350e-6 -- an impossible negative "variance") and as a
        transient dip in the forward curve just past the selection
        gates (the 12-19 bucket), healing as the walk re-converges.
        Injections cannot see this bias (injected effects bypass the
        tracker), so it must be handled by DESIGN, not calibration:
        the quantitative reads are TRACKING-CONVERGED -- the ERA-SPLIT
        product (halves >= 6 months apart, second factor entirely
        post-selection so the stopping rule cannot bias it either) and
        the 40+ tenure bucket.  Static chemistry itself is NOT eaten by
        the walk: v2 fits values jointly with static dyad terms, so a
        persistent pair effect sits in d_ij (excluded from this
        script's predictions), not in the values; only transient luck
        gets tracked.  Per-(player, quarter) offsets would absorb the
        tracking state directly and were tried -- but after LODO
        exclusion the cells are too thin, four noisy offsets enter
        every residual, and the cleaned variance TRIPLED.  Rejected.
  (v)   Thin players' offsets are ridge-dominated, their rating error
        survives cleaning, and it inflated the small-dyad classes.  The
        primary population is IDENTIFIABLE pairs: both players dynamic
        AND >= 20 out-of-dyad panel games each (the house rule
        "chemistry is only identified through partner variation" turned
        into a filter).

SUPPORTING ARMS.
  arm 1  panorama: how much of the archive is short-lived pairings.
  arm 3  dissolution hazard: P(pair plays another event) as a function
         of results so far -- measures how strong outcome-selection IS
         and feeds the simulation.
  arm 4  split-half moments: split each dyad's EVENT-BLOCKS into two
         random halves (averaged over R re-splits); E[m_h1 * m_h2] =
         PERSISTENT chemistry variance, with no noise-variance model
         (game, match and event shocks never straddle an event-random
         split; serial form-tracking by the monthly value walk averages
         out across random splits, unlike interleaved or contiguous
         splits which it biases).  Dyad-weighted (every tried pair
         counts once, doomed ones included) vs games-weighted (survivor-
         tilted).  Both remain biased by selection (halves share the
         "survived" conditioning) so both are read against the arm-5
         null, never against zero.  NOTE the estimand: across-EVENT
         persistent pair effect.  v2's sd_d cannot distinguish a career
         pair effect from pair-form within one event (most dyads play
         1-2 events, so the columns nearly coincide); this estimator
         deliberately measures only the persistent part, which is the
         part the survivorship story is about.
  arm 5  lifecycle simulation: dyads arrive as observed, chemistry drawn
         at a grid of true sds, games generated with measured game/
         match/event noise, continuation decided by the arm-3 FITTED
         hazard, each sim dyad capped by its real counterpart's
         REMAINING CALENDAR (#events in its tour on/after its first
         event -- without this the sim over-extends lives, since real
         late-arriving pairs are right-censored by the archive end).
         Scored by the same estimators.  Maps (true sd, observed
         selection) -> expected curve and moments, including the
         stopped-mean bias at sd = 0.
  arm 6  injection into the REAL panel (house rule: a null without a
         measured floor means nothing).  (i) iid per-dyad chemistry ->
         recovery lambda of the split-half moment through the full LODO
         pipeline.  (ii) tenure-CORRELATED chemistry with the arm-5
         enrichment shape -> recovery lambda of the curve estimator.
  arm 7  cross-season persistence of v1's per-season dyad estimates
         (independently fitted seasons, so value errors do not straddle
         the join), against the shrinkage-attenuation ceiling implied by
         each file's own posterior sds.

SCALE.  Cleaned SHARE units throughout: observed point share minus the
race-model share from month-of-game v2 values + pooled gamma, stripped
of per-division antisymmetric gap slopes and LODO ridge player offsets
(same nuisance structure as gap_exploit.py).  share x 4 ~ per-point
logit; share x ~17 (mean points per game) ~ points-per-game (v1 scale).
v2's sd_d = 0.053 logit = 0.0132 share is "finding 2" on this scale.

Traps carried: winner-first row order randomised (fixed seed);
DreamBreakers, forfeits, sub-5-point games excluded; UUIDs lowercased;
dyads keyed as unordered pairs exactly as fit_v2 keys them.

Stdlib only (no numpy in this environment); full battery ~5 min.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import GAMMA, sigmoid  # noqa: E402

DATA = ROOT / "data"
OUT_JSON = ROOT / "model" / "chem_survival_summary.json"

RIDGE_A = 25.0                 # same offset shrinkage as gap_exploit.py
SEED = 20260816
CENSOR_DAYS = 180              # a pair gets this long to show a next event
V2_SD_D_SHARE = 0.0529 / 4     # finding 2's chemistry sd on the share scale
SPLIT_R = 20                   # random event-half re-splits averaged per dyad
MIN_OUTSIDE = 20               # out-of-dyad games each partner needs (ident.)

BUCKETS = [(0, 0, "0"), (1, 5, "1-5"), (6, 11, "6-11"),
           (12, 19, "12-19"), (20, 39, "20-39"), (40, 10**9, "40+")]
REF_BUCKET = "6-11"
FIT_BUCKETS = [b for b in BUCKETS if b[2] != REF_BUCKET]
ENRICH_BUCKETS = ["12-19", "20-39", "40+"]

TEN_CLASSES = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 14, "6-14"),
               (15, 29, "15-29"), (30, 59, "30-59"), (60, 10**9, "60+")]


def bucket_of(t):
    for lo, hi, name in BUCKETS:
        if lo <= t <= hi:
            return name
    return BUCKETS[-1][2]


def class_of(n):
    for lo, hi, name in TEN_CLASSES:
        if lo <= n <= hi:
            return name
    return TEN_CLASSES[-1][2]


# ---------------------------------------------------------------- loading ---

def load_values():
    traj = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "v2_trajectories.csv")):
        traj[r["player_id"].lower()][r["month"]] = float(r["value_mean"])
    months = {p: sorted(d) for p, d in traj.items()}
    static = {}
    for r in csv.DictReader(open(DATA / "v2_players.csv")):
        static[r["player_id"].lower()] = float(r["value_now_mean"])
    dyn = set(traj)

    def value(pid, month):
        d = traj.get(pid)
        if d:
            if month in d:
                return d[month]
            ms = months[pid]
            return d[ms[0]] if month < ms[0] else d[ms[-1]]
        return static.get(pid)

    return value, dyn


def load_games():
    games = []
    for g in csv.DictReader(open(DATA / "games.csv")):
        if g["is_dreambreaker"] in ("True", "1") or g["is_forfeit"] in ("True", "1"):
            continue
        if g["scoring_format"] not in ("sideout_11", "sideout_15"):
            continue
        try:
            s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        except ValueError:
            continue
        if s1 + s2 < 5:
            continue
        games.append(g)
    games.sort(key=lambda g: (g["date"], g["event_id"], g["match_id"],
                              int(g["game_number"] or 1)))
    return games


def build(seed=SEED):
    """Universe tenure bookkeeping + residual panel.  Tenure counts every
    clean game the pair played (including games later dropped from the
    residual panel for missing values), so 'games together' means what it
    says."""
    value, dyn = load_values()
    games = load_games()

    dyad_seq = defaultdict(int)
    dyad_tot = defaultdict(int)
    dyad_dates = defaultdict(list)
    dyad_tours = defaultdict(lambda: defaultdict(int))
    dyad_div = defaultdict(lambda: defaultdict(int))
    for g in games:
        for keyc in (("t1_p1", "t1_p2"), ("t2_p1", "t2_p2")):
            k = tuple(sorted((g[keyc[0]].lower(), g[keyc[1]].lower())))
            dyad_tot[k] += 1
            dyad_dates[k].append(g["date"])
            dyad_tours[k][g["tour"]] += 1
            dyad_div[k][g["context"]] += 1

    rng = random.Random(seed)
    rows, skipped = [], 0
    for g in games:
        month = g["date"][:7]
        ps = [(g[k] or "").lower() for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        kA0 = tuple(sorted(ps[:2]))
        kB0 = tuple(sorted(ps[2:]))
        tenA0, tenB0 = dyad_seq[kA0], dyad_seq[kB0]
        dyad_seq[kA0] += 1
        dyad_seq[kB0] += 1

        vs = [value(p, month) for p in ps]
        if any(v is None for v in vs) or not all(ps):
            skipped += 1
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        gap_a, gap_b = abs(vs[0] - vs[1]), abs(vs[2] - vs[3])
        eta = (vs[0] + vs[1] + GAMMA * gap_a) - (vs[2] + vs[3] + GAMMA * gap_b)
        kA, kB, tenA, tenB = kA0, kB0, tenA0, tenB0
        if rng.random() < 0.5:           # neutralise the t1-winner-first bias
            ps = ps[2:] + ps[:2]
            gap_a, gap_b = gap_b, gap_a
            s1, s2 = s2, s1
            eta = -eta
            kA, kB, tenA, tenB = kB0, kA0, tenB0, tenA0
        rows.append({
            "players": ps, "div": g["context"], "tour": g["tour"],
            "date": g["date"], "match": g["match_id"], "event": g["event_id"],
            "dyadA": kA, "dyadB": kB, "tenA": tenA, "tenB": tenB,
            "dyn4": all(p in dyn for p in ps),
            "gap_a": gap_a, "gap_b": gap_b,
            "share": s1 / (s1 + s2), "pred": sigmoid(eta),
            "w": float(s1 + s2),
        })
    meta = {"dyad_tot": dict(dyad_tot), "dyad_dates": dyad_dates,
            "dyad_tours": dyad_tours, "dyad_div": dyad_div, "dyn": dyn,
            "n_universe": len(games), "skipped": skipped,
            "archive_end": games[-1]["date"]}
    return rows, meta


# --------------------------------------------------------------- cleaning ---

def converge_nuisance(rows, resid, use_idx, n_iter=6):
    """Converge per-division antisymmetric gap slopes + ridge per-player
    (career) offsets on a SUBSET of rows, and pre-compute the per-(player,
    own-dyad) num/den contributions so leave-own-dyad-out offsets can be
    read off at convergence.

    Granularity note, learned the expensive way: per-(player, QUARTER)
    offsets would also absorb the value-walk's tracking state (see the
    tracking discussion in the module docstring) and were tried -- but
    after LODO exclusion the cells hold a handful of games, four such
    offsets enter every residual, and the cleaned variance TRIPLED
    (27.7e-3 -> 87.6e-3 share^2), drowning every arm.  Career-player
    offsets keep the estimator stable; the tracking bias they leave in is
    handled by USING TRACKING-CONVERGED READS (the 40+ tenure bucket, the
    era-split product) rather than by finer offsets."""
    a = defaultdict(float)
    beta = defaultdict(float)
    for _ in range(n_iter):
        acc = defaultdict(lambda: [0.0, 0.0])
        for i in use_idx:
            row = rows[i]
            pl = sum(a[p] for p in row["players"][:2]) - \
                sum(a[p] for p in row["players"][2:])
            x, y, w = row["gap_a"] - row["gap_b"], resid[i] - pl, row["w"]
            s = acc[row["div"]]
            s[0] += w * x * x
            s[1] += w * x * y
        for d, s in acc.items():
            beta[d] = s[1] / s[0] if s[0] > 1e-12 else 0.0
        num, den = defaultdict(float), defaultdict(float)
        for i in use_idx:
            row = rows[i]
            base = resid[i] - beta[row["div"]] * (row["gap_a"] - row["gap_b"])
            pl = sum(a[p] for p in row["players"][:2]) - \
                sum(a[p] for p in row["players"][2:])
            for k, p in enumerate(row["players"]):
                s = 1.0 if k < 2 else -1.0
                num[p] += row["w"] * s * (base - pl + s * a[p])
                den[p] += row["w"]
        a = defaultdict(float)
        for p in num:
            a[p] = num[p] / (den[p] + RIDGE_A)

    cnum = defaultdict(float)
    cden = defaultdict(float)
    num_tot = defaultdict(float)
    den_tot = defaultdict(float)
    for i in use_idx:
        row = rows[i]
        base = resid[i] - beta[row["div"]] * (row["gap_a"] - row["gap_b"])
        pl = sum(a[p] for p in row["players"][:2]) - \
            sum(a[p] for p in row["players"][2:])
        for k, p in enumerate(row["players"]):
            s = 1.0 if k < 2 else -1.0
            d = row["dyadA"] if k < 2 else row["dyadB"]
            contrib = row["w"] * s * (base - pl + s * a[p])
            cnum[(p, d)] += contrib
            cden[(p, d)] += row["w"]
            num_tot[p] += contrib
            den_tot[p] += row["w"]
    return {"beta": beta, "cnum": cnum, "cden": cden,
            "num_tot": num_tot, "den_tot": den_tot}


def clean_with(rows, resid, fit, i):
    """LODO-clean row i using a (possibly out-of-sample) nuisance fit:
    each side's offsets are the fit's, with that side's own-dyad
    contributions (as seen by the fit) removed."""
    row = rows[i]

    def a_lodo(p, d):
        n = fit["num_tot"].get(p, 0.0) - fit["cnum"].get((p, d), 0.0)
        dd = fit["den_tot"].get(p, 0.0) - fit["cden"].get((p, d), 0.0)
        return n / (dd + RIDGE_A)

    pl = (a_lodo(row["players"][0], row["dyadA"]) +
          a_lodo(row["players"][1], row["dyadA"]) -
          a_lodo(row["players"][2], row["dyadB"]) -
          a_lodo(row["players"][3], row["dyadB"]))
    return (resid[i] - fit["beta"][row["div"]] *
            (row["gap_a"] - row["gap_b"]) - pl)


def fit_global(rows, resid):
    """Single-fit LODO cleaning of every row (used by the curve and hazard
    arms, where residuals enter as MEANS: LODO offset noise is mean-zero
    there.  The moment arm, which multiplies a dyad's residuals, must use
    crossfit_products instead -- a single fit's offset error is a constant
    shared by both halves of the product and lands in it as +Var(error)."""
    fit = converge_nuisance(rows, resid, range(len(rows)))
    return [clean_with(rows, resid, fit, i) for i in range(len(rows))]


def crossfit_products(rows, resid, ident, rng, partitions=5, era=False):
    """Cross-fitted split-half products: partition EVENTS into two halves,
    clean each half's games with the nuisance fit of the OTHER half (still
    leave-own-dyad-out), take each identifiable dyad's product of half
    means.  Offset-estimation errors of the two halves come from disjoint
    games, so the product's null bias from shared cleaning noise is zero
    by construction; chemistry (and see-saw leakage of chemistry, which
    the injection lambda measures) is what remains.  Averaged over
    `partitions` random partitions; era=True instead uses the fixed
    2024-25 x 2026 partition (the long-lag secondary)."""
    events = sorted({r["event"] for r in rows})
    acc = defaultdict(lambda: [0.0, 0])
    n_games = defaultdict(int)
    for r in rows:
        n_games[r["dyadA"]] += 1
        n_games[r["dyadB"]] += 1
    parts = 1 if era else partitions
    for _ in range(parts):
        if era:
            half = {ev: 0 for ev in events}
            for r in rows:
                if r["date"] >= "2026":
                    half[r["event"]] = 1
        else:
            half = {ev: rng.randrange(2) for ev in events}
        idx0 = [i for i, r in enumerate(rows) if half[r["event"]] == 0]
        idx1 = [i for i, r in enumerate(rows) if half[r["event"]] == 1]
        fit0 = converge_nuisance(rows, resid, idx0)
        fit1 = converge_nuisance(rows, resid, idx1)
        sums = defaultdict(lambda: [0.0, 0, 0.0, 0])   # s0, n0, s1, n1
        for i, r in enumerate(rows):
            h = half[r["event"]]
            e = clean_with(rows, resid, fit1 if h == 0 else fit0, i)
            for d, sgn in ((r["dyadA"], 1.0), (r["dyadB"], -1.0)):
                if d not in ident:
                    continue
                s = sums[d]
                if h == 0:
                    s[0] += sgn * e
                    s[1] += 1
                else:
                    s[2] += sgn * e
                    s[3] += 1
        for d, s in sums.items():
            if s[1] and s[3]:
                t = (s[0] / s[1]) * (s[2] / s[3])
                acc[d][0] += t
                acc[d][1] += 1
    return {d: {"t": v[0] / v[1], "t_era": None, "n": n_games[d],
                "nparts": v[1]}
            for d, v in acc.items() if v[1] > 0}


# ------------------------------------------------- arm 1: panorama ---------

def shift_date(d, days):
    import datetime
    dt = datetime.date.fromisoformat(d) + datetime.timedelta(days=days)
    return dt.isoformat()


def panorama(meta):
    tot = meta["dyad_tot"]
    cut = shift_date(meta["archive_end"], -CENSOR_DAYS)
    by_class = defaultdict(lambda: [0, 0])
    dissolved_short = active_short = 0
    for k, n in tot.items():
        c = class_of(n)
        by_class[c][0] += 1
        by_class[c][1] += n
        if n <= 5:
            if max(meta["dyad_dates"][k]) <= cut:
                dissolved_short += 1
            else:
                active_short += 1
    n_dyads = len(tot)
    n_sides = sum(tot.values())
    out = {"n_dyads": n_dyads, "classes": {}}
    for lo, hi, name in TEN_CLASSES:
        d, g = by_class[name]
        out["classes"][name] = {
            "dyads": d, "dyad_pct": 100 * d / n_dyads,
            "games": g, "game_side_pct": 100 * g / n_sides}
    out["short_dissolved"] = dissolved_short
    out["short_active_tail"] = active_short
    return out


# --------------------------------------- arm 2: forward tenure curve -------

def curve_design(rows, tenA=None, tenB=None):
    tenA = tenA if tenA is not None else [r["tenA"] for r in rows]
    tenB = tenB if tenB is not None else [r["tenB"] for r in rows]
    names = [b[2] for b in FIT_BUCKETS]
    pos = {n: j for j, n in enumerate(names)}
    X = []
    for i in range(len(rows)):
        x = [0.0] * len(names)
        ba, bb = bucket_of(tenA[i]), bucket_of(tenB[i])
        if ba in pos:
            x[pos[ba]] += 1.0
        if bb in pos:
            x[pos[bb]] -= 1.0
        X.append(x)
    return X, names


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def inv(A):
    n = len(A)
    return [solve(A, [1.0 if j == i else 0.0 for j in range(n)])
            for i in range(n)]


def wls(rows, e, X, idx):
    k = len(X[0])
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for i in idx:
        w = rows[i]["w"]
        xi = X[i]
        for p in range(k):
            if xi[p] == 0.0:
                continue
            b[p] += w * xi[p] * e[i]
            for q in range(k):
                if xi[q] != 0.0:
                    A[p][q] += w * xi[p] * xi[q]
    theta = solve(A, b)
    return theta, A


def cluster_se(rows, e, X, theta, A, idx):
    """Multi-membership cluster-robust covariance (each game belongs to its
    two dyads): M = sum_dyad g g' - sum_(dyadpair) g g' gives every obs
    pair sharing >= 1 dyad weight exactly 1 by inclusion-exclusion."""
    k = len(theta)
    Ainv = inv(A)
    gd = defaultdict(lambda: [0.0] * k)
    gp = defaultdict(lambda: [0.0] * k)
    for i in idx:
        row = rows[i]
        resid = e[i] - sum(X[i][p] * theta[p] for p in range(k))
        s = row["w"] * resid
        for p in range(k):
            if X[i][p] != 0.0:
                gd[row["dyadA"]][p] += s * X[i][p]
                gd[row["dyadB"]][p] += s * X[i][p]
                gp[(row["dyadA"], row["dyadB"])][p] += s * X[i][p]
    M = [[0.0] * k for _ in range(k)]
    for gset, sign in ((gd, 1.0), (gp, -1.0)):
        for g in gset.values():
            for p in range(k):
                if g[p] == 0.0:
                    continue
                for q in range(k):
                    M[p][q] += sign * g[p] * g[q]
    T = [[sum(Ainv[p][r] * M[r][q] for r in range(k)) for q in range(k)]
         for p in range(k)]
    V = [[sum(T[p][r] * Ainv[r][q] for r in range(k)) for q in range(k)]
         for p in range(k)]
    return V


def contrast(theta, V, names, num_buckets, weights):
    wsum = sum(weights.get(b, 0) for b in num_buckets)
    c = [0.0] * len(names)
    for b in num_buckets:
        c[names.index(b)] = weights.get(b, 0) / wsum
    est = sum(c[p] * theta[p] for p in range(len(c)))
    var = sum(c[p] * V[p][q] * c[q] for p in range(len(c)) for q in range(len(c)))
    return est, math.sqrt(max(var, 0.0))


def tenure_curve(rows, clean, idx, rng, shuffles=200):
    X, names = curve_design(rows)
    theta, A = wls(rows, clean, X, idx)
    V = cluster_se(rows, clean, X, theta, A, idx)
    se = [math.sqrt(max(V[p][p], 0.0)) for p in range(len(theta))]

    occ = defaultdict(int)
    for i in idx:
        occ[bucket_of(rows[i]["tenA"])] += 1
        occ[bucket_of(rows[i]["tenB"])] += 1
    enrich, enrich_se = contrast(theta, V, names, ENRICH_BUCKETS, occ)
    b0 = theta[names.index("0")]
    b0_se = se[names.index("0")]

    slots = defaultdict(list)
    for i in idx:
        slots[rows[i]["dyadA"]].append((i, 0, rows[i]["tenA"]))
        slots[rows[i]["dyadB"]].append((i, 1, rows[i]["tenB"]))
    null_thetas = []
    tenA = [rows[i]["tenA"] for i in range(len(rows))]
    tenB = [rows[i]["tenB"] for i in range(len(rows))]
    for _ in range(shuffles):
        tA = list(tenA)
        tB = list(tenB)
        for kk, lst in slots.items():
            ts = [t for _, _, t in lst]
            rng.shuffle(ts)
            for (i, side, _), t in zip(lst, ts):
                if side == 0:
                    tA[i] = t
                else:
                    tB[i] = t
        Xs, _ = curve_design(rows, tA, tB)
        th, _ = wls(rows, clean, Xs, idx)
        null_thetas.append(th)
    return {"names": names, "theta": theta, "se": se, "occ": dict(occ),
            "enrich": enrich, "enrich_se": enrich_se,
            "b0": b0, "b0_se": b0_se,
            "shuffle_theta": null_thetas}


# ------------------------------------------- arm 3: dissolution hazard -----

def dyad_side_rows(rows, clean, keep=None):
    out = defaultdict(list)
    for i, r in enumerate(rows):
        if keep is None or r["dyadA"] in keep:
            out[r["dyadA"]].append((r["date"], r["event"], r["match"],
                                    clean[i], r["tour"]))
        if keep is None or r["dyadB"] in keep:
            out[r["dyadB"]].append((r["date"], r["event"], r["match"],
                                    -clean[i], r["tour"]))
    for k in out:
        out[k].sort(key=lambda t: (t[0], t[1], t[2]))
    return out


def dyad_blocks_of(lst):
    blocks = []
    for d, ev, m, e, tour in lst:
        if not blocks or blocks[-1]["event"] != ev:
            blocks.append({"event": ev, "last": d, "first": d, "es": [],
                           "matches": [], "tour": tour})
        blocks[-1]["es"].append(e)
        blocks[-1]["last"] = d
        if not blocks[-1]["matches"] or blocks[-1]["matches"][-1][0] != m:
            blocks[-1]["matches"].append([m, 0])
        blocks[-1]["matches"][-1][1] += 1
    return blocks


KCAP = 12


def hazard_blocks(sides, archive_end):
    cut = shift_date(archive_end, -CENSOR_DAYS)
    recs = []
    for k, lst in sides.items():
        blocks = dyad_blocks_of(lst)
        cum_e = cum_n = 0.0
        for j, b in enumerate(blocks):
            cum_e += sum(b["es"])
            cum_n += len(b["es"])
            if b["last"] > cut:
                continue
            recs.append({"dyad": k, "k": j + 1,
                         "mres": max(-0.2, min(0.2, cum_e / cum_n)),
                         "games": cum_n, "tour": b["tour"],
                         "cont": 1 if j + 1 < len(blocks) else 0})
    return recs


def hazard_feats(r):
    x = [0.0] * (2 * KCAP + 3)
    kk = min(r["k"], KCAP) - 1
    base = 0 if r["tour"] == "PPA" else KCAP
    x[base + kk] = 1.0
    x[2 * KCAP] = r["mres"]
    x[2 * KCAP + 1] = r["mres"] if r["tour"] == "MLP" else 0.0
    x[2 * KCAP + 2] = math.log1p(r["games"])
    return x


def fit_hazard(recs):
    X = [hazard_feats(r) for r in recs]
    y = [r["cont"] for r in recs]
    p_dim = 2 * KCAP + 3
    beta = [0.0] * p_dim
    H = None
    for _ in range(30):
        g = [0.0] * p_dim
        H = [[0.0] * p_dim for _ in range(p_dim)]
        for xi, yi in zip(X, y):
            eta = sum(b * v for b, v in zip(beta, xi))
            mu = sigmoid(eta)
            r = yi - mu
            wv = mu * (1 - mu)
            for a in range(p_dim):
                if xi[a] == 0.0:
                    continue
                g[a] += r * xi[a]
                for b2 in range(p_dim):
                    if xi[b2] != 0.0:
                        H[a][b2] += wv * xi[a] * xi[b2]
        for a in range(p_dim):
            H[a][a] += 1e-6
            g[a] -= 1e-6 * beta[a]
        step = solve(H, g)
        beta = [beta[a] + step[a] for a in range(p_dim)]
        if max(abs(s) for s in step) < 1e-8:
            break
    Hinv = inv(H)
    return {"beta": beta,
            "se_mres": math.sqrt(max(Hinv[2 * KCAP][2 * KCAP], 0.0)),
            "se_mres_mlp": math.sqrt(max(Hinv[2 * KCAP + 1][2 * KCAP + 1], 0.0))}


def hazard_terciles(recs):
    strata = defaultdict(list)
    for r in recs:
        strata[(r["tour"], min(r["k"], 6))].append(r)
    agg = defaultdict(lambda: [0, 0])
    for lst in strata.values():
        lst.sort(key=lambda r: r["mres"])
        n = len(lst)
        for i, r in enumerate(lst):
            t = min(2, i * 3 // max(n, 1))
            agg[(r["tour"], t)][0] += r["cont"]
            agg[(r["tour"], t)][1] += 1
    return {f"{tour}_t{t}": (c / n if n else None, n)
            for (tour, t), (c, n) in sorted(agg.items())}


# -------------------------------------- arm 4: split-half moments ----------

def split_half(sides, rng, R=SPLIT_R):
    """Random EVENT-half products averaged over R re-splits, plus an era
    split (2024-25 x 2026).  Unweighted within dyad so the estimator is
    identical to the simulator's."""
    per_dyad = {}
    for k, lst in sides.items():
        blocks = dyad_blocks_of(lst)
        if len(blocks) < 2:
            continue
        bs = [(sum(b["es"]), len(b["es"]), b["first"]) for b in blocks]
        acc = 0.0
        for _ in range(R):
            order = list(range(len(bs)))
            rng.shuffle(order)
            h = [[0.0, 0], [0.0, 0]]
            for pos, j in enumerate(order):
                s, n, _ = bs[j]
                h[pos % 2][0] += s
                h[pos % 2][1] += n
            acc += (h[0][0] / h[0][1]) * (h[1][0] / h[1][1])
        t_rand = acc / R
        e1 = [(s, n) for s, n, d in bs if d < "2026"]
        e2 = [(s, n) for s, n, d in bs if d >= "2026"]
        t_era = None
        if e1 and e2:
            t_era = (sum(s for s, _ in e1) / sum(n for _, n in e1)) * \
                    (sum(s for s, _ in e2) / sum(n for _, n in e2))
        per_dyad[k] = {"t": t_rand, "t_era": t_era,
                       "n": len(lst), "nb": len(blocks)}
    return per_dyad


def moment_agg(per_dyad, sel, field="t"):
    vals = [(per_dyad[k][field], per_dyad[k]["n"]) for k in sel
            if per_dyad[k][field] is not None]
    if not vals:
        return None
    d = sum(t for t, _ in vals) / len(vals)
    g = sum(t * n for t, n in vals) / sum(n for _, n in vals)
    return d, g, len(vals)


def moment_stats(per_dyad, classify, boots, rng, field="t"):
    keys = list(per_dyad)
    groups = defaultdict(list)
    for k in keys:
        groups["ALL"].append(k)
        for lab in classify(k):
            groups[lab].append(k)
    out = {}
    for lab, sel in groups.items():
        base = moment_agg(per_dyad, sel, field)
        if base is None:
            continue
        bs_d, bs_g = [], []
        for _ in range(boots):
            samp = [sel[rng.randrange(len(sel))] for _ in range(len(sel))]
            r = moment_agg(per_dyad, samp, field)
            if r:
                bs_d.append(r[0])
                bs_g.append(r[1])
        bs_d.sort()
        bs_g.sort()

        def ci(v):
            return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]
        out[lab] = {"n_dyads": base[2],
                    "tau2_dyad": base[0], "tau2_dyad_ci": ci(bs_d),
                    "tau2_games": base[1], "tau2_games_ci": ci(bs_g)}
    return out


def variance_components(sides):
    """Nested empirics for the simulator: game / match / event noise from
    covariance ladders within dyads (static chemistry cancels in the
    DIFFERENCES between rungs, so these are chemistry-free)."""
    out = {}
    for tour in ("PPA", "MLP"):
        tv = tn = 0.0
        wm = [0.0, 0]
        we = [0.0, 0]
        be = [0.0, 0]
        for k, lst in sides.items():
            sub = [(ev, m, e) for d, ev, m, e, t in lst if t == tour]
            if not sub:
                continue
            for _, _, e in sub:
                tv += e * e
                tn += 1
            bym = defaultdict(list)
            for ev, m, e in sub:
                bym[(ev, m)].append(e)
            for es in bym.values():
                for i in range(len(es)):
                    for j in range(i + 1, len(es)):
                        wm[0] += es[i] * es[j]
                        wm[1] += 1
            byev = defaultdict(list)
            for (ev, m), es in bym.items():
                byev[ev].append(sum(es) / len(es))
            for ms in byev.values():
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        we[0] += ms[i] * ms[j]
                        we[1] += 1
            evs = [sum(ms) / len(ms) for ms in byev.values()]
            for i in range(len(evs)):
                for j in range(i + 1, len(evs)):
                    be[0] += evs[i] * evs[j]
                    be[1] += 1
        var_g = tv / max(tn, 1)
        c_wm = wm[0] / max(wm[1], 1)
        c_we = we[0] / max(we[1], 1)
        c_be = be[0] / max(be[1], 1)
        out[tour] = {"var_game": var_g,
                     "cov_within_match": c_wm, "cov_within_event": c_we,
                     "cov_between_event": c_be,
                     "v_match": max(c_wm - c_we, 0.0),
                     "v_event": max(c_we - c_be, 0.0),
                     "v_game": max(var_g - c_wm, 1e-6)}
    return out


# ------------------------------------------ arm 5: lifecycle simulation ----

def sim_empirics(sides, rows):
    """Block structures per tour + one lifecycle seat per real dyad,
    carrying that dyad's tour and CALENDAR AVAILABILITY: the number of
    real events in its tour on/after its own first event (the archive's
    right-censoring, which the hazard alone cannot express)."""
    ev_dates = defaultdict(dict)
    for r in rows:
        d = ev_dates[r["tour"]]
        if r["event"] not in d or r["date"] < d[r["event"]]:
            d[r["event"]] = r["date"]
    ev_sorted = {t: sorted(d.values()) for t, d in ev_dates.items()}

    structs = defaultdict(list)
    seats = []
    for k, lst in sides.items():
        n_ppa = sum(1 for x in lst if x[4] == "PPA")
        tour = "PPA" if n_ppa >= len(lst) / 2 else "MLP"
        blocks = dyad_blocks_of(lst)
        for b in blocks:
            structs[tour].append([n for _, n in b["matches"]])
        first = blocks[0]["first"]
        dates = ev_sorted[tour]
        lo, hi = 0, len(dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if dates[mid] < first:
                lo = mid + 1
            else:
                hi = mid
        seats.append((tour, max(1, len(dates) - lo)))
    return structs, seats


def simulate(tau, structs, seats, vc, hz_beta, rng, max_blocks=40, split_R=8):
    occ_sum = defaultdict(float)
    occ_n = defaultdict(int)
    per_dyad = []
    for tour, avail in seats:
        st = structs[tour]
        sg = math.sqrt(vc[tour]["v_game"])
        sm = math.sqrt(vc[tour]["v_match"])
        sev = math.sqrt(vc[tour]["v_event"])
        cap = min(avail, max_blocks)
        c = rng.gauss(0.0, tau) if tau > 0 else 0.0
        ten = 0
        cum_e = cum_n = 0.0
        blocks = []
        k = 0
        while True:
            k += 1
            ue = rng.gauss(0.0, sev)
            bsum, bn = 0.0, 0
            for msize in st[rng.randrange(len(st))]:
                um = rng.gauss(0.0, sm)
                for _ in range(msize):
                    e = c + ue + um + rng.gauss(0.0, sg)
                    occ_sum[bucket_of(ten)] += c
                    occ_n[bucket_of(ten)] += 1
                    ten += 1
                    bsum += e
                    bn += 1
            blocks.append((bsum, bn))
            cum_e += bsum
            cum_n += bn
            if k >= cap:
                break
            rec = {"k": k, "mres": max(-0.2, min(0.2, cum_e / cum_n)),
                   "games": cum_n, "tour": tour}
            p = sigmoid(sum(b * v for b, v in zip(hz_beta, hazard_feats(rec))))
            if rng.random() > p:
                break
        if len(blocks) >= 2:
            # SEQUENTIAL halves (first half of the pair's events vs the
            # rest), mirroring the real ERA product's structure: the
            # second factor postdates the first, so the stopping rule's
            # bias enters the same way it does in the data.
            cut2 = len(blocks) // 2
            h = [[0.0, 0], [0.0, 0]]
            for j, (s, n) in enumerate(blocks):
                hh = 0 if j < cut2 else 1
                h[hh][0] += s
                h[hh][1] += n
            per_dyad.append(((h[0][0] / h[0][1]) * (h[1][0] / h[1][1]),
                             int(cum_n)))
    curve = {b: (occ_sum[b] / occ_n[b] if occ_n[b] else 0.0)
             for _, _, b in BUCKETS}
    t_d = sum(t for t, n in per_dyad) / max(len(per_dyad), 1)
    t_g = (sum(t * n for t, n in per_dyad) /
           max(sum(n for _, n in per_dyad), 1))
    num_w = sum(occ_n.get(b, 0) for b in ENRICH_BUCKETS)
    enrich = (sum(curve[b] * occ_n.get(b, 0) for b in ENRICH_BUCKETS) / num_w
              - curve[REF_BUCKET]) if num_w else 0.0
    return {"curve": curve, "tau2_dyad": t_d, "tau2_games": t_g,
            "enrich": enrich, "occ_n": dict(occ_n),
            "n_split_dyads": len(per_dyad)}


# -------------------------------------------------- arm 6 helpers ----------

def draw_dyad_effects(rows, cval):
    return [cval.get(r["dyadA"], 0.0) - cval.get(r["dyadB"], 0.0)
            for r in rows]


def all_dyads(rows):
    s = set()
    for r in rows:
        s.add(r["dyadA"])
        s.add(r["dyadB"])
    return s


# ------------------------------------- arm 7: v1 cross-season persistence --

def v1_persistence(rng, min_games=10, perms=500):
    def load(year):
        out = {}
        p = DATA / f"results_dyads_{year}.csv"
        if not p.exists():
            return out, 0.47
        sds = []
        for r in csv.DictReader(open(p)):
            key = (frozenset((r["p1_name"], r["p2_name"])), r["context"])
            out[key] = (float(r["chemistry_mean"]), float(r["chemistry_sd"]),
                        int(r["games"]))
            sds.append(float(r["chemistry_sd"]))
        sds.sort()
        tau_file = sds[int(0.9 * len(sds))] if sds else 0.47
        return out, tau_file
    seasons = {}
    taus = {}
    for y in (2024, 2025, 2026):
        seasons[y], taus[y] = load(y)
    res = {}
    for y1, y2 in ((2024, 2025), (2025, 2026)):
        pairs = []
        for k, (c1, s1, n1) in seasons[y1].items():
            if k in seasons[y2]:
                c2, s2, n2 = seasons[y2][k]
                if min(n1, n2) >= min_games:
                    pairs.append((c1, c2, min(n1, n2), s1, s2))
        if len(pairs) < 10:
            continue

        def wcorr(ps):
            sw = sum(p[2] for p in ps)
            mx = sum(p[2] * p[0] for p in ps) / sw
            my = sum(p[2] * p[1] for p in ps) / sw
            sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in ps)
            sxx = sum(p[2] * (p[0] - mx) ** 2 for p in ps)
            syy = sum(p[2] * (p[1] - my) ** 2 for p in ps)
            return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
        obs = wcorr(pairs)
        # attenuation ceiling if chemistry were fully real and persistent:
        # posterior mean ~ shrink * (c + noise), shrink_y = 1 - (postsd/tau_y)^2
        # with tau_y inferred from the file's own sd ceiling (season fits used
        # different priors; the 90th-pct posterior sd ~ the no-data sd).
        sh = []
        for c1, c2, w, s1, s2 in pairs:
            k1 = max(1e-3, 1 - (s1 / taus[y1]) ** 2)
            k2 = max(1e-3, 1 - (s2 / taus[y2]) ** 2)
            sh.append((math.sqrt(k1 * k2), w))
        ceil = sum(s * w for s, w in sh) / sum(w for _, w in sh)
        null = []
        base = [(p[0], p[2]) for p in pairs]
        ys = [(p[1], p[2]) for p in pairs]
        for _ in range(perms):
            rng.shuffle(ys)
            null.append(wcorr([(a, b, min(w1, w2), 0, 0)
                               for (a, w1), (b, w2) in zip(base, ys)]))
        null.sort()
        res[f"{y1}->{y2}"] = {
            "n_pairs": len(pairs), "r": obs, "r_ceiling": ceil,
            "tau_files": [taus[y1], taus[y2]],
            "null_95": [null[int(0.025 * len(null))],
                        null[int(0.975 * len(null))]],
            "p": (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)}
    return res


# ------------------------------------------------------------------ main ---

CURVE_ORDER = ["0", "1-5", "12-19", "20-39", "40+"]


def fmt_curve(names, theta, se=None):
    parts = []
    for b in CURVE_ORDER:
        j = names.index(b)
        s = f"{b}: {theta[j] * 1000:+.2f}"
        if se:
            s += f"±{se[j] * 1000:.2f}"
        parts.append(s)
    return "  ".join(parts) + "   (x10^-3 share, vs 6-11)"


def pct_list(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(q * len(xs))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    shuffles = 40 if args.fast else 200
    boots = 400 if args.fast else 2000
    sim_reps0 = 40 if args.fast else 150
    sim_reps = 15 if args.fast else 50
    inj_reps = 3 if args.fast else 6

    rows, meta = build()
    idx_all = list(range(len(rows)))
    idx_dyn = [i for i in idx_all if rows[i]["dyn4"]]
    print(f"panel: {len(rows)} games ({meta['skipped']} skipped for missing "
          f"values) of {meta['n_universe']} universe games; "
          f"{len(idx_dyn)} all-dynamic games; archive ends {meta['archive_end']}")

    # identifiable dyads: both players dynamic AND >= MIN_OUTSIDE panel
    # games outside this dyad (chemistry is only identified through
    # partner variation -- the house rule as a filter)
    pgames = defaultdict(int)
    dgames = defaultdict(int)
    for r in rows:
        for k, p in enumerate(r["players"]):
            pgames[p] += 1
            dgames[(p, r["dyadA"] if k < 2 else r["dyadB"])] += 1
    dyn = meta["dyn"]

    def identifiable(d):
        return all(p in dyn and pgames[p] - dgames[(p, d)] >= MIN_OUTSIDE
                   for p in d)

    # ---------------- arm 1: panorama ------------------------------------
    pano = panorama(meta)
    print(f"\nARM 1 — PANORAMA ({pano['n_dyads']} dyads ever tried)")
    for lo, hi, name in TEN_CLASSES:
        c = pano["classes"][name]
        print(f"  {name:>6s} games: {c['dyads']:5d} dyads ({c['dyad_pct']:4.1f}%)"
              f"  carrying {c['game_side_pct']:4.1f}% of game-sides")
    print(f"  short (<=5 games) dyads dissolved vs still-active-window: "
          f"{pano['short_dissolved']} vs {pano['short_active_tail']}")

    # ---------------- cleaning (LODO) ------------------------------------
    resid = [r["share"] - r["pred"] for r in rows]
    clean = fit_global(rows, resid)
    mu = sum(clean) / len(clean)
    sd = math.sqrt(sum(x * x for x in clean) / len(clean))
    print(f"\nLODO-cleaned residual: mean {mu:+.5f}, sd {sd:.4f} (share units)")

    # ---------------- arm 2: forward curve -------------------------------
    curves = {}
    for lab, idx in (("all", idx_all), ("dyn4", idx_dyn)):
        tc = tenure_curve(rows, clean, idx, rng, shuffles)
        curves[lab] = tc
        print(f"\nARM 2 — FORWARD TENURE CURVE [{lab} panel, {len(idx)} games]")
        print("  observed  " + fmt_curve(tc["names"], tc["theta"], tc["se"]))
        if tc["shuffle_theta"]:
            sh_mean = [sum(t[j] for t in tc["shuffle_theta"]) /
                       len(tc["shuffle_theta"]) for j in range(len(tc["names"]))]
            tc["shuffle_mean"] = sh_mean
            print("  shuffle   " + fmt_curve(tc["names"], sh_mean) +
                  "  <- dyad-composition part")
            print("  obs-shuf  " + "  ".join(
                f"{b}: {(tc['theta'][tc['names'].index(b)] - sh_mean[tc['names'].index(b)]) * 1000:+.2f}"
                for b in CURVE_ORDER) + "  <- within-pair-order part")
        print(f"  ENRICHMENT (>=12 vs 6-11): {tc['enrich'] * 1000:+.3f} ± "
              f"{tc['enrich_se'] * 1000:.3f} x10^-3 share "
              f"(z = {tc['enrich'] / tc['enrich_se']:+.2f}; analytic null 0)   "
              f"scratch bucket 0: {tc['b0'] * 1000:+.2f} ± {tc['b0_se'] * 1000:.2f}")

    # ---------------- arm 3: hazard --------------------------------------
    sides_all = dyad_side_rows(rows, clean)
    ident = {k for k in sides_all if identifiable(k)}
    sides_id = {k: v for k, v in sides_all.items() if k in ident}
    print(f"\nidentifiable dyads (both dynamic, >= {MIN_OUTSIDE} out-of-dyad "
          f"games): {len(ident)} of {len(sides_all)}")

    recs_all = hazard_blocks(sides_all, meta["archive_end"])
    recs_id = hazard_blocks(sides_id, meta["archive_end"])
    hz_all = fit_hazard(recs_all)
    hz = fit_hazard(recs_id)
    print(f"\nARM 3 — DISSOLUTION HAZARD (censor {CENSOR_DAYS}d)")
    for tag, h, rr in (("all pairs", hz_all, recs_all),
                       ("identifiable", hz, recs_id)):
        print(f"  [{tag}, {len(rr)} blocks] P(another event) ~ mres: beta = "
              f"{h['beta'][2 * KCAP]:+.2f} ± {h['se_mres']:.2f} /share (PPA)  "
              f"MLP interact {h['beta'][2 * KCAP + 1]:+.2f} ± "
              f"{h['se_mres_mlp']:.2f}")
    terc = hazard_terciles(recs_id)
    print("  [identifiable] continuation by within-(tour,k) mres tercile: " +
          "  ".join(f"{k}: {v[0]:.3f} (n={v[1]})" for k, v in terc.items()
                    if v[0] is not None))

    # ---------------- arm 4: split-half moments --------------------------
    xf_parts = 3 if args.fast else 4
    per_dyad = crossfit_products(rows, resid, ident, rng, xf_parts)
    per_dyad_era = crossfit_products(rows, resid, ident, rng, era=True)
    per_dyad_1fit = split_half(sides_id, rng)     # shared-error receipt only
    tot = meta["dyad_tot"]
    tours = meta["dyad_tours"]
    divs = meta["dyad_div"]
    mean_pts = sum(r["w"] for r in rows) / len(rows)

    def classify(k):
        labs = [f"final_{class_of(tot[k])}"]
        tor = tours[k]
        dv = max(divs[k], key=divs[k].get)
        if tor.get("MLP", 0) == 0:
            labs.append("PPA_only")
            if dv in ("mens", "womens"):
                labs.append("PPA_gender")
        elif tor.get("PPA", 0) == 0:
            labs.append("MLP_only")
            if dv in ("mens", "womens"):
                labs.append("MLP_gender_FORCED")
        else:
            labs.append("cross_tour")
        labs.append(f"div_{dv}")
        return labs

    mo = moment_stats(per_dyad_era, classify, boots, rng)
    mo_rand = moment_stats(per_dyad, classify, boots, rng)
    mo_1fit = moment_stats(per_dyad_1fit, classify, boots, rng)
    print(f"\nARM 4 — SPLIT-HALF MOMENTS, PRIMARY = ERA-CROSSFIT "
          f"(2024-25 x 2026 halves, each era cleaned with the other era's "
          f"nuisance fit; tau^2 = E[m_era1 x m_era2]; identifiable pairs, "
          f"{mo['ALL']['n_dyads']} spanning both eras; share^2 x10^-6; "
          f"finding-2 tau^2 = {(V2_SD_D_SHARE ** 2) * 1e6:.0f})")

    def show(mm, lab, tag=""):
        if lab not in mm:
            return
        m = mm[lab]

        def t2s(v):
            return f"{v * 1e6:+7.1f}"
        print(f"  {lab:>18s}{tag} (n={m['n_dyads']:5d})  "
              f"dyad-wt {t2s(m['tau2_dyad'])} "
              f"[{t2s(m['tau2_dyad_ci'][0])},{t2s(m['tau2_dyad_ci'][1])}]   "
              f"games-wt {t2s(m['tau2_games'])} "
              f"[{t2s(m['tau2_games_ci'][0])},{t2s(m['tau2_games_ci'][1])}]")
    for lab in (["ALL"] + [f"final_{n}" for _, _, n in TEN_CLASSES] +
                ["PPA_only", "MLP_only", "cross_tour", "PPA_gender",
                 "MLP_gender_FORCED", "div_mens", "div_womens", "div_mixed"]):
        show(mo, lab)
    print("  --- estimator receipts (ALL line under alternative moments) ---")
    show(mo_rand, "ALL", " random-halves (tracking-biased low: adjacent-"
         "quarter lags where the walk's luck-overshoot has not healed)")
    show(mo_1fit, "ALL", " 1-fit (offset-error-biased high: shared "
         "cleaning noise lands in the product)")
    print(f"  (share->points x{mean_pts:.1f}; share->logit x4; "
          f"tau^2=175e-6 <=> sd {V2_SD_D_SHARE:.4f} share = 0.053 logit = "
          f"{V2_SD_D_SHARE * mean_pts:.2f} pts)")

    vc = variance_components(sides_id)
    for tour, v in vc.items():
        print(f"  {tour}: var_game {v['var_game'] * 1e3:.2f}e-3  "
              f"v_event {v['v_event'] * 1e6:.0f}e-6  "
              f"v_match {v['v_match'] * 1e6:.0f}e-6  "
              f"cov_between_event(=tau^2+sel bias) "
              f"{v['cov_between_event'] * 1e6:+.1f}e-6")

    structs, seats = sim_empirics(sides_id, rows)

    # ---------------- arm 5: lifecycle sim -------------------------------
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    n_seats = defaultdict(int)
    for t, _ in seats:
        n_seats[t] += 1
    print(f"\nARM 5 — LIFECYCLE SIM (identifiable-pair seats {dict(n_seats)}; "
          f"hazard as fitted; calendar-capped; tau in multiples of finding-2 "
          f"sd {V2_SD_D_SHARE:.4f})")
    sim_out = {}
    for mult in grid:
        tau = mult * V2_SD_D_SHARE
        reps = sim_reps0 if mult == 0.0 else sim_reps
        stats = [simulate(tau, structs, seats, vc, hz["beta"], rng)
                 for _ in range(reps)]
        enr = [s["enrich"] for s in stats]
        t_d = [s["tau2_dyad"] for s in stats]
        t_g = [s["tau2_games"] for s in stats]
        sim_out[mult] = {
            "tau_share": tau,
            "enrich_mean": sum(enr) / len(enr),
            "enrich_95": [pct_list(enr, 0.025), pct_list(enr, 0.975)],
            "tau2_dyad_mean": sum(t_d) / len(t_d),
            "tau2_dyad_95": [pct_list(t_d, 0.025), pct_list(t_d, 0.975)],
            "tau2_games_mean": sum(t_g) / len(t_g),
            "tau2_games_95": [pct_list(t_g, 0.025), pct_list(t_g, 0.975)],
            "curve_mean": {b: sum(s["curve"][b] for s in stats) / len(stats)
                           for _, _, b in BUCKETS},
            "occ_mean": {b: sum(s["occ_n"].get(b, 0) for s in stats) / len(stats)
                         for _, _, b in BUCKETS},
        }
        s = sim_out[mult]
        print(f"  tau = {mult:3.1f}x: enrich {s['enrich_mean'] * 1000:+.3f} "
              f"[{s['enrich_95'][0] * 1000:+.3f},{s['enrich_95'][1] * 1000:+.3f}] "
              f"e-3   tau2 dyad-wt {s['tau2_dyad_mean'] * 1e6:+7.1f} "
              f"[{s['tau2_dyad_95'][0] * 1e6:+.0f},{s['tau2_dyad_95'][1] * 1e6:+.0f}]"
              f"  games-wt {s['tau2_games_mean'] * 1e6:+7.1f} "
              f"[{s['tau2_games_95'][0] * 1e6:+.0f},{s['tau2_games_95'][1] * 1e6:+.0f}]")
    occ_real = defaultdict(int)
    for k in ident:
        lst = sides_id[k]
        for t_i in range(len(lst)):
            occ_real[bucket_of(t_i)] += 1
    tot_r = sum(occ_real.values())
    occ_sim = sim_out[0.0]["occ_mean"]
    tot_s = sum(occ_sim.values())
    print("  calibration (identifiable side-games % per tenure bucket, real "
          "vs tau=0 sim): "
          + "  ".join(f"{b}: {100 * occ_real[b] / tot_r:.1f}/"
                      f"{100 * occ_sim[b] / tot_s:.1f}"
                      for _, _, b in BUCKETS))

    # ---------------- arm 6: injections ----------------------------------
    print("\nARM 6 — INJECTIONS (through the full crossfit-LODO pipeline)")
    tau_star = 2 * V2_SD_D_SHARE
    base_tg = mo["ALL"]["tau2_games"]
    base_td = mo["ALL"]["tau2_dyad"]
    dyset = all_dyads(rows)
    lam_g, lam_d, lam_raw = [], [], []
    for rep in range(inj_reps):
        cval = {d: rng.gauss(0.0, tau_star) for d in dyset}
        add = draw_dyad_effects(rows, cval)
        clean_probe = [clean[i] + add[i] for i in range(len(rows))]
        pd_p = split_half(dyad_side_rows(rows, clean_probe, ident), rng)
        a_p = moment_agg(pd_p, list(pd_p))
        lam_raw.append((a_p[1] - mo_1fit["ALL"]["tau2_games"]) / tau_star ** 2)
        resid_i = [resid[i] + add[i] for i in range(len(rows))]
        pd_i = crossfit_products(rows, resid_i, ident, rng, era=True)
        a_i = moment_agg(pd_i, list(pd_i))
        lam_g.append((a_i[1] - base_tg) / tau_star ** 2)
        lam_d.append((a_i[0] - base_td) / tau_star ** 2)
    lam_mom_g = sum(lam_g) / len(lam_g)
    lam_mom_d = sum(lam_d) / len(lam_d)
    print(f"  iid c_d ~ N(0, {tau_star:.4f}): ERA-crossfit recovery lambda "
          f"games-wt {lam_mom_g:.2f} dyad-wt {lam_mom_d:.2f} "
          f"(no-recleaning probe {sum(lam_raw) / len(lam_raw):.2f}; "
          f"{inj_reps} reps)")

    def planted_enrich(cval):
        s = defaultdict(float)
        n = defaultdict(int)
        for i in idx_dyn:
            r = rows[i]
            for ten, d in ((r["tenA"], r["dyadA"]), (r["tenB"], r["dyadB"])):
                b = bucket_of(ten)
                s[b] += cval.get(d, 0.0)
                n[b] += 1
        num_n = sum(n[b] for b in ENRICH_BUCKETS)
        hi = sum(s[b] for b in ENRICH_BUCKETS) / num_n
        return hi - s[REF_BUCKET] / n[REF_BUCKET]

    inj_curve = {}
    for mult in (1.0, 2.0):
        smc = sim_out[mult]
        xs, ys, ws = [], [], []
        for (lo2, hi2, b) in BUCKETS:
            n = smc["occ_mean"][b]
            if n <= 0:
                continue
            mid = lo2 if hi2 > 10 ** 8 else (lo2 + hi2) / 2
            xs.append(math.log1p(mid + 3))
            ys.append(smc["curve_mean"][b])
            ws.append(n)
        mx = sum(w * x for w, x in zip(ws, xs)) / sum(ws)
        my = sum(w * y for w, y in zip(ws, ys)) / sum(ws)
        a_slope = (sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
                   / max(sum(w * (x - mx) ** 2 for w, x in zip(ws, xs)), 1e-12))
        tt = mult * V2_SD_D_SHARE
        ls = {d: math.log1p(tot.get(d, 1)) for d in dyset}
        mean_l = sum(ls.values()) / len(ls)
        curve_reps = 1 if args.fast else 3
        pls, recs_c = [], []
        for _ in range(curve_reps):
            cval = {d: a_slope * (ls[d] - mean_l) + rng.gauss(0.0, tt * 0.8)
                    for d in dyset}
            planted = planted_enrich(cval)
            add = draw_dyad_effects(rows, cval)
            resid_i = [resid[i] + add[i] for i in range(len(rows))]
            clean_i = fit_global(rows, resid_i)
            tc_i = tenure_curve(rows, clean_i, idx_dyn, rng, shuffles=0)
            pls.append(planted)
            recs_c.append(tc_i["enrich"] - curves["dyn4"]["enrich"])
        planted = sum(pls) / len(pls)
        rec = sum(recs_c) / len(recs_c)
        lam_c = rec / planted if planted else None
        inj_curve[mult] = {"a_slope": a_slope, "planted_enrich": planted,
                           "sim_world_enrich": smc["enrich_mean"],
                           "recovered_delta": rec, "lambda_curve": lam_c}
        lam_txt = f"{lam_c:.2f}" if lam_c is not None else "n/a"
        print(f"  tenure-linked at {mult:.0f}x (slope {a_slope * 1000:+.3f}"
              f"e-3/log-game, {curve_reps} reps): planted enrich "
              f"{planted * 1000:+.3f}e-3 -> recovered delta "
              f"{rec * 1000:+.3f}e-3 (lambda_curve {lam_txt})")

    # ---------------- arm 7: v1 persistence ------------------------------
    v1p = v1_persistence(rng)
    print("\nARM 7 — v1 CROSS-SEASON PERSISTENCE (independent per-season "
          "fits, dyads >=10 games both seasons, weighted r)")
    for k, v in v1p.items():
        print(f"  {k}: r = {v['r']:+.3f}  ceiling-if-fully-real "
              f"{v['r_ceiling']:+.3f} (file taus {v['tau_files'][0]:.2f}/"
              f"{v['tau_files'][1]:.2f})  null95 [{v['null_95'][0]:+.3f},"
              f"{v['null_95'][1]:+.3f}]  p = {v['p']:.3f}  (n = {v['n_pairs']})")

    # ---------------- reconciliation -------------------------------------
    print("\nRECONCILIATION (observed vs sim worlds, identifiable pairs)")
    obs_e = curves["dyn4"]["enrich"]
    obs_se = curves["dyn4"]["enrich_se"]
    lam_c_avg = [v["lambda_curve"] for v in inj_curve.values()
                 if v["lambda_curve"] is not None]
    lam_c_avg = sum(lam_c_avg) / len(lam_c_avg) if lam_c_avg else 1.0
    print(f"  curve: observed (dyn4) {obs_e * 1000:+.3f} ± {obs_se * 1000:.3f}"
          f" e-3; lambda_curve {lam_c_avg:.2f}")
    excl_c = []
    for mult in grid[1:]:
        pred = sim_out[mult]["enrich_mean"] * lam_c_avg
        z = (pred - obs_e) / obs_se
        tag = "EXCLUDED" if z > 1.96 else "allowed"
        excl_c.append((mult, pred, z, tag))
        print(f"    tau={mult:.1f}x -> predicted {pred * 1000:+.3f}e-3, "
              f"z vs observed {z:+.2f}  {tag}")
    print(f"  moments: observed games-wt {base_tg * 1e6:+.1f}e-6 "
          f"(dyad-wt {base_td * 1e6:+.1f}); sim null bias games-wt "
          f"{sim_out[0.0]['tau2_games_mean'] * 1e6:+.1f} (dyad-wt "
          f"{sim_out[0.0]['tau2_dyad_mean'] * 1e6:+.1f}); lambda "
          f"{lam_mom_g:.2f}/{lam_mom_d:.2f}")
    excl_m = []
    for mult in grid[1:]:
        pred_g = (sim_out[0.0]["tau2_games_mean"] +
                  lam_mom_g * (mult * V2_SD_D_SHARE) ** 2)
        lo_ci, hi_ci = mo["ALL"]["tau2_games_ci"]
        tag = "EXCLUDED" if pred_g > hi_ci else "allowed"
        excl_m.append((mult, pred_g, tag))
        print(f"    tau={mult:.1f}x -> predicted games-wt "
              f"{pred_g * 1e6:+.1f}e-6 vs observed CI "
              f"[{lo_ci * 1e6:+.1f},{hi_ci * 1e6:+.1f}]  {tag}")
    # implied point estimate
    imp_g = (base_tg - sim_out[0.0]["tau2_games_mean"]) / max(lam_mom_g, 0.1)
    imp_d = (base_td - sim_out[0.0]["tau2_dyad_mean"]) / max(lam_mom_d, 0.1)
    print(f"  implied persistent tau^2: games-wt {imp_g * 1e6:+.1f}e-6 "
          f"({math.sqrt(max(imp_g, 0)) / V2_SD_D_SHARE:.2f}x finding-2)  "
          f"dyad-wt {imp_d * 1e6:+.1f}e-6 "
          f"({math.sqrt(max(imp_d, 0)) / V2_SD_D_SHARE:.2f}x)")

    # ---------------- summary dump ---------------------------------------
    summary = {
        "n_games_panel": len(rows), "n_games_dyn4": len(idx_dyn),
        "n_universe": meta["n_universe"], "n_identifiable": len(ident),
        "panorama": pano, "cleaned_sd": sd,
        "curves": {lab: {kk: tc[kk] for kk in
                         ("names", "theta", "se", "occ", "enrich",
                          "enrich_se", "b0", "b0_se", "shuffle_mean")
                         if kk in tc}
                   for lab, tc in curves.items()},
        "hazard_all": {"beta_mres_ppa": hz_all["beta"][2 * KCAP],
                       "se": hz_all["se_mres"],
                       "beta_mres_mlp_interact": hz_all["beta"][2 * KCAP + 1],
                       "n_blocks": len(recs_all)},
        "hazard_identifiable": {"beta_mres_ppa": hz["beta"][2 * KCAP],
                                "se": hz["se_mres"],
                                "beta_mres_mlp_interact": hz["beta"][2 * KCAP + 1],
                                "terciles": terc, "n_blocks": len(recs_id)},
        "moments_era_primary": mo,
        "moments_random_tracking_biased": {"ALL": mo_rand.get("ALL")},
        "moments_1fit_offset_biased": {"ALL": mo_1fit.get("ALL")},
        "variance_components": vc,
        "mean_points_per_game": mean_pts,
        "sim": {str(k): v for k, v in sim_out.items()},
        "injection": {"lambda_moment_games": lam_mom_g,
                      "lambda_moment_dyad": lam_mom_d,
                      "lambda_raw_probe": sum(lam_raw) / len(lam_raw),
                      "tenure_linked": inj_curve},
        "v1_persistence": v1p,
        "reconciliation": {"curve_exclusions": excl_c,
                           "moment_exclusions": excl_m,
                           "implied_tau2_games": imp_g,
                           "implied_tau2_dyad": imp_d},
        "reference": {"v2_sd_d_share": V2_SD_D_SHARE,
                      "v2_sd_d_logit": 0.0529, "beta_new_logit": 0.0879},
        "seed": args.seed, "fast": args.fast,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
