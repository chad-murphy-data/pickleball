"""Load the model CSVs + games.csv and aggregate everything the site needs.

Single pass over games.csv (~36k rows); pure stdlib.  All identity is by
player UUID (lowercased upstream) — names are display-only, per house rules.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .race import GAMMA, race_dist, sigmoid, team_eta, value_points

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"


def _target(fmt: str) -> int:
    return 15 if fmt.endswith("15") else 11


def month_key(date: str) -> str:
    return date[:7]


class Player:
    __slots__ = ("pid", "name", "gender", "games", "value", "sd", "dynamic",
                 "pts", "pts_lo", "pts_hi", "rank", "traj", "dupr", "dupr_hist",
                 "stats", "form_delta", "last_date", "dupr_asof", "dupr_glitch")

    def __init__(self, pid, name, gender, games, value, sd, dynamic):
        self.pid, self.name, self.gender = pid, name, gender
        self.games, self.value, self.sd, self.dynamic = games, value, sd, dynamic
        self.pts = value_points(value)
        self.pts_lo = value_points(value - 1.645 * sd)
        self.pts_hi = value_points(value + 1.645 * sd)
        self.rank = None
        self.traj = []          # [(month, mean, sd)]
        self.dupr = None
        self.dupr_hist = []     # [(date, rating)]
        self.stats = None
        self.form_delta = None  # 6-month value change (logit)
        self.last_date = None
        self.dupr_asof = None   # date of the latest synced rating snapshot
        self.dupr_glitch = None # last credible value when latest is an artifact


class PStats:
    __slots__ = ("w", "l", "pf", "pa", "by_year_tour", "by_context",
                 "deciding", "overtime", "blowout_w", "blowout_l",
                 "cur_streak", "best_streak", "partners", "log")

    def __init__(self):
        self.w = self.l = self.pf = self.pa = 0
        self.by_year_tour = defaultdict(lambda: [0, 0, 0, 0])   # (year,tour) -> w,l,pf,pa
        self.by_context = defaultdict(lambda: [0, 0])           # context -> w,l
        self.deciding = [0, 0]
        self.overtime = [0, 0]
        self.blowout_w = self.blowout_l = 0
        self.cur_streak = 0     # signed: +wins / -losses
        self.best_streak = 0
        self.partners = defaultdict(lambda: [0, 0, 0, 0])       # pid -> g,w,pf,pa
        self.log = []           # per-game dicts, chronological


def load_players():
    """v2_players.csv + platform ratings; returns {pid: Player}."""
    players = {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        players[r["player_id"]] = Player(
            r["player_id"], r["full_name"] or r["player_id"][:8],
            r["gender"], int(r["games"]),
            float(r["value_now_mean"]), float(r["value_now_sd"]),
            r["dynamic"] == "1")
    for r in csv.DictReader((DATA / "platform_ratings.csv").open()):
        if r["player_id"] in players:
            players[r["player_id"]].dupr = float(r["platform_rating_latest"])
    for r in csv.DictReader((DATA / "v2_trajectories.csv").open()):
        p = players.get(r["player_id"])
        if p is not None:
            p.traj.append((r["month"], float(r["value_mean"]), float(r["value_sd"])))
    for p in players.values():
        p.traj.sort()
        if len(p.traj) >= 7:
            p.form_delta = p.traj[-1][1] - p.traj[-7][1]
    return players


def load_games():
    """Non-forfeit, non-DreamBreaker games, chronological."""
    games = []
    for g in csv.DictReader((DATA / "games.csv").open()):
        if g["is_forfeit"] != "False" or g["is_dreambreaker"] != "False":
            continue
        games.append(g)
    games.sort(key=lambda g: (g["date"], g["match_id"], int(g["game_number"])))
    return games


def month_values(players):
    """(pid, month) -> posterior mean value; static players fall back to
    their single value.  Months outside a trajectory clamp to its edges."""
    mv = {}
    for p in players.values():
        for m, mean, _sd in p.traj:
            mv[(p.pid, m)] = mean
    return mv


def value_at(players, mv, pid, month):
    p = players.get(pid)
    if p is None:
        return None
    if not p.traj:
        return p.value
    v = mv.get((pid, month))
    if v is not None:
        return v
    first, last = p.traj[0], p.traj[-1]
    return first[1] if month < first[0] else last[1]


def expected_share(players, mv, g):
    """Model point share for team1, from monthly values + weakest link.
    In-sample/descriptive (full-fit values) — labeled as such on the site."""
    m = month_key(g["date"])
    vs = [value_at(players, mv, g[c], m) for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
    if any(v is None for v in vs):
        return None
    return sigmoid(team_eta(*vs))


def aggregate(players, games):
    """One pass over games: per-player stats + site-wide records material."""
    mv = month_values(players)
    stats = defaultdict(PStats)
    pair_hist = defaultdict(lambda: [0, 0, 0])   # pairkey -> games, wins, streak
    pair_best = {}                               # pairkey -> best streak
    upsets = []                                  # (win_prob_of_winner, game)
    flagged = set()
    if (DATA / "flags.csv").exists():
        for r in csv.DictReader((DATA / "flags.csv").open()):
            if "swap" in (r.get("reason") or ""):
                flagged.add(r.get("game_id") or "")
    marathons = []                               # (total_points, game)
    ratings = json.load((DATA / "per_match_ratings.json").open()) \
        if (DATA / "per_match_ratings.json").exists() else {}
    seen_rating_date = set()

    for g in games:
        t = _target(g["scoring_format"])
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 == s2:
            continue
        team1 = (g["t1_p1"], g["t1_p2"])
        team2 = (g["t2_p1"], g["t2_p2"])
        t1_won = s1 > s2
        exp = expected_share(players, mv, g)
        year = g["date"][:4]
        deciding = (g["best_of"], g["game_number"]) in (("3", "3"), ("5", "5"))
        winner_score = max(s1, s2)
        ot = winner_score > t
        total = s1 + s2

        # site-wide records material
        if total >= 2 * t:                      # e.g. 12-10 or beyond at T=11
            marathons.append((total, g))
        if exp is not None and g["game_id"] not in flagged:
            w_exp = exp if t1_won else 1.0 - exp
            if w_exp < 0.42 and all(players.get(u) and players[u].dynamic
                                    for u in team1 + team2):
                pw = race_dist(round(w_exp, 4), t)["p_win"]
                # sub-0.2% "upsets" are almost always data quirks (injury
                # retirements, unrecorded lineup changes), not tennis miracles
                if 0.002 <= pw < 0.20:
                    upsets.append((pw, g))

        for pair, won in ((tuple(sorted(team1)), t1_won),
                          (tuple(sorted(team2)), not t1_won)):
            h = pair_hist[pair]
            h[0] += 1
            if won:
                h[1] += 1
                h[2] += 1
                if h[2] > pair_best.get(pair, 0):
                    pair_best[pair] = h[2]
            else:
                h[2] = 0

        # per-match DUPR snapshots -> per-player rating history
        mr = ratings.get(g["match_id"])

        for side, mine, theirs, won, my_pair, opp_pair in (
                (1, s1, s2, t1_won, team1, team2),
                (2, s2, s1, not t1_won, team2, team1)):
            e = exp if side == 1 else (None if exp is None else 1.0 - exp)
            for pid in my_pair:
                if pid not in players:
                    continue
                st = stats[pid]
                st.w += won; st.l += (not won)
                st.pf += mine; st.pa += theirs
                yt = st.by_year_tour[(year, g["tour"])]
                yt[0] += won; yt[1] += (not won); yt[2] += mine; yt[3] += theirs
                bc = st.by_context[g["context"] or "?"]
                bc[0] += won; bc[1] += (not won)
                if deciding:
                    st.deciding[0 if won else 1] += 1
                if ot:
                    st.overtime[0 if won else 1] += 1
                if t == 11 and theirs <= 5 and won:
                    st.blowout_w += 1
                if t == 11 and mine <= 5 and not won:
                    st.blowout_l += 1
                st.cur_streak = st.cur_streak + 1 if won and st.cur_streak >= 0 \
                    else (1 if won else (st.cur_streak - 1 if st.cur_streak <= 0 else -1))
                if st.cur_streak > st.best_streak:
                    st.best_streak = st.cur_streak
                partner = my_pair[0] if my_pair[1] == pid else my_pair[1]
                pt = st.partners[partner]
                pt[0] += 1; pt[1] += won; pt[2] += mine; pt[3] += theirs
                st.log.append({
                    "date": g["date"], "event": g["event_name"], "tour": g["tour"],
                    "context": g["context"], "partner": partner,
                    "opp": opp_pair, "score": f"{mine}-{theirs}",
                    "share": mine / total, "exp": e, "won": won, "ot": ot,
                })
                players[pid].last_date = g["date"]
                if mr and pid in mr and (pid, g["date"]) not in seen_rating_date:
                    seen_rating_date.add((pid, g["date"]))
                    players[pid].dupr_hist.append((g["date"], mr[pid]))

    for pid, st in stats.items():
        if pid in players:
            players[pid].stats = st
    for p in players.values():
        p.dupr_hist.sort()

    upsets.sort(key=lambda x: x[0])
    marathons.sort(key=lambda x: -x[0])
    return {
        "pair_games": {k: (v[0], v[1]) for k, v in pair_hist.items()},
        "pair_best_streak": pair_best,
        "upsets": upsets[:15],
        "marathons": marathons[:12],
    }


def finalize_dupr(players):
    """Latest synced rating per player, with an as-of date and a reset-
    artifact screen: a rating that falls to DUPR's ~3.5 reset default after
    a >=5.0 history is a recording artifact, not a skill measurement (one
    known case: a 6.13 player re-appearing at 3.50021 mid-season).  Artifact
    ratings are nulled for comparisons; the last credible value is kept for
    the footnote."""
    for p in players.values():
        if not p.dupr_hist:
            continue
        p.dupr_asof, p.dupr = p.dupr_hist[-1]
        peak = max(r for _, r in p.dupr_hist)
        if p.dupr <= 3.65 and peak >= 5.0:
            credible = [r for _, r in p.dupr_hist if r > 3.65]
            p.dupr_glitch = credible[-1] if credible else None
            p.dupr = None


def infer_missing_genders(players):
    """A handful of players lack a gender in players.csv; their game contexts
    (mens/womens) pin it down."""
    for p in players.values():
        if p.gender in ("M", "F") or p.stats is None:
            continue
        m = p.stats.by_context.get("mens", [0, 0])
        w = p.stats.by_context.get("womens", [0, 0])
        if sum(m) > sum(w):
            p.gender = "M"
        elif sum(w) > sum(m):
            p.gender = "F"
        else:
            # mixed-only player: partner in mixed is the opposite gender
            votes = {"M": 0, "F": 0}
            for pid2, (g, *_x) in p.stats.partners.items():
                pg = players.get(pid2)
                if pg and pg.gender in votes:
                    votes["F" if pg.gender == "M" else "M"] += g
            if votes["M"] != votes["F"]:
                p.gender = max(votes, key=votes.get)


ACTIVE_SINCE = "2026-01-01"


def is_active(p):
    return (p.last_date or "") >= ACTIVE_SINCE


def rank_players(players):
    """Gender ranks among dynamic players active this season, by current
    value.  Inactive players keep rank None ('—' on the site) so the
    rankings read as current form, not a hall of fame."""
    for gender in ("M", "F"):
        pool = sorted((p for p in players.values()
                       if p.dynamic and p.gender == gender and is_active(p)),
                      key=lambda p: -p.value)
        for i, p in enumerate(pool, 1):
            p.rank = i


def load_dyads():
    """Pair chemistry by name pair (v2_dyads.csv is name-keyed upstream)."""
    chem = {}
    for r in csv.DictReader((DATA / "v2_dyads.csv").open()):
        chem[frozenset((r["p1_name"], r["p2_name"]))] = (
            float(r["chem_logit_mean"]), float(r["chem_logit_sd"]))
    return chem


def load_receipts():
    return json.loads((MODEL / "receipts.json").read_text())


