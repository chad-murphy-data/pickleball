"""MLP 2026 season awards: Most Improved, MVP (matchup WPA), Under the Radar (GWAE).

Prints the three leaderboards behind model/mlp_awards_2026.md.  Stdlib-only.

  python model/mlp_awards.py

Definitions
-----------
Most Improved   delta in v1 season value (points per game) 2025 -> 2026,
                min 20 games in each season, MLP-2026-active players only.
MVP (WPA)       game wins weighted by how much they moved the team's matchup
                win probability.  Matchups reconstructed from games.csv
                (no matchup id exists there: group by event/date/stage, then
                split by roster overlap; order games by start time).  Future
                games are coin flips; matchups open at 50/50; first to 3 of 4
                wins, 2-2 goes to a DreamBreaker (excluded, nets to zero).
                Both players on court bank the full swing, so a roster's
                summed WPA equals its net matchups-decided-in-games record.
GWAE            games won above expectation: sum of (won - p_hat) where p_hat
                is the model win prob from month-of-game v2 values (team
                strength = sum + gamma*|gap|, gamma = -0.18; per-point
                p = sigma(diff); exact race-to-11, overtime branch lumped).
"""
import csv
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
GAMMA = -0.18
GWAE_MIN_GAMES = 25
SEASON = "2026-01-01"

sig = lambda x: 1 / (1 + math.exp(-x))


def p_game(p, T=11):
    """P(win a race to T) at per-point prob p; win-by-2 overtime lumped."""
    q = 1 - p
    pw = sum(math.comb(T - 1 + b, b) * p**T * q**b for b in range(T - 1))
    pl = sum(math.comb(T - 1 + a, a) * q**T * p**a for a in range(T - 1))
    return pw + (1 - pw - pl) * (p * p / (p * p + q * q))


@lru_cache(None)
def matchup_p(a, b):
    """P(team wins matchup | game score a-b), 50/50 games, 2-2 -> DB coinflip."""
    if a >= 3:
        return 1.0
    if b >= 3:
        return 0.0
    if a + b >= 4:
        return 0.5
    return 0.5 * matchup_p(a + 1, b) + 0.5 * matchup_p(a, b + 1)


def load():
    players = {r["player_id"]: r for r in csv.DictReader(open(DATA / "v2_players.csv"))}
    games = [r for r in csv.DictReader(open(DATA / "games.csv"))
             if r["tour"] == "MLP" and r["date"] >= SEASON and r["is_forfeit"] == "False"]
    times = {r["match_id"]: r for r in csv.DictReader(open(DATA / "match_times.csv"))}
    return players, games, times


def name(players, pid):
    p = players.get(pid)
    return f"{p['full_name']} ({p['gender']})" if p else pid[:8]


# --------------------------------------------------------------- most improved
def most_improved(players, games, min_games=20):
    active = set()
    for g in games:
        active |= {g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]}
    yearly = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "yearly_values.csv")):
        yearly[r["player_id"]][r["year"]] = r
    rows = []
    for pid in active:
        y = yearly.get(pid, {})
        if "2025" not in y or "2026" not in y:
            continue
        a, b = y["2025"], y["2026"]
        if int(a["games"]) < min_games or int(b["games"]) < min_games:
            continue
        rows.append((float(b["value"]) - float(a["value"]), pid, a, b))
    rows.sort(reverse=True)
    return rows


# ------------------------------------------------- matchup reconstruction + WPA
CTX_ORDER = {"womens": 0, "mens": 1, "mixed": 2}


def _components(games):
    """Group MLP games into matchups: same event/date/stage, linked rosters."""
    buckets = defaultdict(list)
    for g in games:
        buckets[(g["event_id"], g["date"], g["stage"])].append(g)
    comps = []
    for gs in buckets.values():
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for g in gs:
            ps = [g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]]
            for p in ps:
                parent.setdefault(p, p)
            for p in ps[1:]:
                parent[find(ps[0])] = find(p)
        grp = defaultdict(list)
        for g in gs:
            grp[find(g["t1_p1"])].append(g)
        comps.extend(grp.values())
    return comps


def _split_teams(comp):
    """Two rosters via union-find on partnerships; None if not exactly two."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    allp = set()
    for g in comp:
        for pair in ((g["t1_p1"], g["t1_p2"]), (g["t2_p1"], g["t2_p2"])):
            for p in pair:
                parent.setdefault(p, p)
                allp.add(p)
            parent[find(pair[0])] = find(pair[1])
    roots = {find(p) for p in allp}
    if len(roots) != 2:
        return None
    root_a = sorted(roots)[0]
    return lambda p: find(p) == root_a


def wpa(games, times):
    stats = defaultdict(lambda: {"wpa": 0.0, "w": 0, "n": 0, "clinch": 0, "save": 0,
                                 "lev": 0.0})
    used = dropped = 0
    for comp in _components(games):
        on_a = _split_teams(comp) if len(comp) in (3, 4) else None
        if on_a is None:
            dropped += 1
            continue
        used += 1
        comp.sort(key=lambda g: (times.get(g["match_id"], {}).get("start_local")
                                 or "9999", CTX_ORDER[g["context"]], g["match_id"]))
        a = b = 0
        for g in comp:
            t1_won = int(g["t1_score"]) > int(g["t2_score"])
            a_won = t1_won == on_a(g["t1_p1"])
            swing = (matchup_p(a + 1, b) if a_won else matchup_p(a, b + 1)) - matchup_p(a, b)
            winners = ((g["t1_p1"], g["t1_p2"]) if t1_won else (g["t2_p1"], g["t2_p2"]))
            facing_elim = (b == 2 and a_won) or (a == 2 and not a_won)
            a, b = (a + 1, b) if a_won else (a, b + 1)
            clinched = max(a, b) == 3
            for p in (g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]):
                s = stats[p]
                s["wpa"] += swing if on_a(p) else -swing
                s["n"] += 1
                s["lev"] += abs(swing)
                won = (p in winners)
                s["w"] += won
                if won and clinched:
                    s["clinch"] += 1
                if won and facing_elim:
                    s["save"] += 1
    return stats, used, dropped


# ------------------------------------------------------------------------ GWAE
def gwae(players, games):
    traj = defaultdict(dict)
    for r in csv.DictReader(open(DATA / "v2_trajectories.csv")):
        traj[r["player_id"]][r["month"]] = float(r["value_mean"])

    def val_at(pid, month):
        t = traj.get(pid)
        if t:
            ms = sorted(m for m in t if m <= month)
            return t[ms[-1]] if ms else t[sorted(t)[0]]
        p = players.get(pid)
        return float(p["value_now_mean"]) if p else 0.0

    out = defaultdict(lambda: {"gwae": 0.0, "w": 0, "n": 0})
    for g in games:
        month = g["date"][:7]
        t1 = (g["t1_p1"], g["t1_p2"])
        t2 = (g["t2_p1"], g["t2_p2"])
        v1 = [val_at(p, month) for p in t1]
        v2 = [val_at(p, month) for p in t2]
        s1 = sum(v1) + GAMMA * abs(v1[0] - v1[1])
        s2 = sum(v2) + GAMMA * abs(v2[0] - v2[1])
        p1 = p_game(sig(s1 - s2))
        w1 = int(g["t1_score"]) > int(g["t2_score"])
        for p in t1:
            out[p]["gwae"] += (1 if w1 else 0) - p1
            out[p]["w"] += w1
            out[p]["n"] += 1
        for p in t2:
            out[p]["gwae"] += (0 if w1 else 1) - (1 - p1)
            out[p]["w"] += not w1
            out[p]["n"] += 1
    return out


def main():
    players, games, times = load()
    print(f"MLP {SEASON[:4]} doubles games (no DB/forfeit): {len(games)}\n")

    print("=== MOST IMPROVED: v1 season value 2025 -> 2026 (pts/game, min 20 games each) ===")
    for delta, pid, a, b in most_improved(players, games)[:10]:
        print(f"  {name(players, pid):32s} {float(a['value']):+5.2f} -> {float(b['value']):+5.2f}"
              f"   delta {delta:+.2f}   games {a['games']}->{b['games']}"
              f"   now #{b['gender_rank']} {b['gender']}")

    stats, used, dropped = wpa(games, times)
    print(f"\n=== MVP: MATCHUP WPA ({used} matchups reconstructed, {dropped} odd groupings dropped) ===")
    for pid, s in sorted(stats.items(), key=lambda kv: -kv[1]["wpa"])[:14]:
        print(f"  {name(players, pid):32s} {s['wpa']:+5.2f}   {s['w']:2d}-{s['n']-s['w']:<2d}"
              f"   clinchers {s['clinch']:2d}   elim-saves {s['save']}"
              f"   avg leverage {s['lev']/s['n']:.3f}")

    g = gwae(players, games)
    print(f"\n=== UNDER THE RADAR: GWAE (min {GWAE_MIN_GAMES} games) ===")
    for pid, s in sorted(g.items(), key=lambda kv: -kv[1]["gwae"]):
        if s["n"] < GWAE_MIN_GAMES:
            continue
        print(f"  {name(players, pid):32s} {s['gwae']:+5.1f}   {s['w']:2d}-{s['n']-s['w']:<2d}"
              f"   (expected {s['w']-s['gwae']:.1f} wins)")


if __name__ == "__main__":
    main()
