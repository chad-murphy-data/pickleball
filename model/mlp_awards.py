"""MLP 2026 season awards: Most Improved, MVP (matchup WPA), Under the Radar (GWAE).

Prints every number behind model/mlp_awards_2026.md.  Stdlib-only.

  python model/mlp_awards.py

Definitions
-----------
Most Improved   MLP-only improvement 2025 -> 2026 among full-time MLP players
                (>= 25 MLP games in EACH season), measured two independent
                ways: (A) change in combined serve+return rally point share
                in MLP play (data/player_serve_rallies.csv, >= 500 rallies
                each season), and (B) change in an opponent/partner-adjusted
                ridge margin fit run on nothing but that season's MLP games
                (anchored so the returning cohort nets to zero, which removes
                pool drift).  Winners are where the two boards converge.
MVP (WPA)       game wins weighted by how much they moved the team's matchup
                win probability.  Matchup structure (which games, what order)
                comes from MLP's own records: data/mlp_matchups_2026.csv,
                built by scraper/mlp_matchups.py from the open BFF -- 286
                completed matchups covering all 1,111 games.  Future games
                are coin flips; matchups open at 50/50; first to 3 game wins,
                2-2 goes to a DreamBreaker (excluded, nets to zero).  Walkover
                / dead games (completed_type 6) advance the matchup score with
                nobody credited.  Both players on court bank the full swing,
                so a roster's summed within-team WPA equals its net
                matchups-decided-in-games record.
GWAE            games won above expectation: sum of (won - p_hat) where p_hat
                is the model win prob from month-of-game v2 values (team
                strength = sum + gamma*|gap|, gamma = -0.18; per-point
                p = sigma(diff); exact race-to-11, overtime branch lumped).
"""
import csv
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import race_dist  # noqa: E402  (stdlib-only module)

GAMMA = -0.18
GWAE_MIN_GAMES = 25
IMPROVED_MIN_MLP_GAMES = 25
SEASON = "2026-01-01"

sig = lambda x: 1 / (1 + math.exp(-x))


def p_game(p, T=11):
    """P(win a race to T) at per-point prob p; win-by-2 overtime lumped
    (exact for iid points: matches the holdout DP to 1e-15)."""
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
    mlp = [r for r in csv.DictReader(open(DATA / "games.csv"))
           if r["tour"] == "MLP" and r["is_forfeit"] == "False"
           and r["is_dreambreaker"] == "False"]
    games = [r for r in mlp if r["date"] >= SEASON]
    table = list(csv.DictReader(open(DATA / "mlp_matchups_2026.csv")))
    return players, mlp, games, table


def name(players, pid):
    p = players.get(pid)
    return f"{p['full_name']} ({p['gender']})" if p else pid[:8]


def on_court(g):
    return (g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"])


# --------------------------------------------------------------- most improved
def mlp_game_counts(mlp_games):
    counts = defaultdict(lambda: defaultdict(int))
    for g in mlp_games:
        for p in on_court(g):
            counts[g["date"][:4]][p] += 1
    return counts


def improved_rally(cohort, min_rallies=500):
    """(A) delta in MLP combined serve+return rally point share 2025 -> 2026."""
    sr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in csv.DictReader(open(DATA / "player_serve_rallies.csv")):
        if r["tour"] == "MLP" and r["discipline"] == "doubles" \
                and r["year"] in ("2025", "2026"):
            cell = sr[r["year"]][r["player_uuid"].lower()]
            cell[0] += int(r["serve_wins"]) + int(r["return_wins"])
            cell[1] += int(r["serve_rallies"]) + int(r["return_rallies"])
    rows = []
    for p in cohort:
        a, b = sr["2025"].get(p), sr["2026"].get(p)
        if not a or not b or a[1] < min_rallies or b[1] < min_rallies:
            continue
        s25, s26 = a[0] / a[1], b[0] / b[1]
        se = (s25 * (1 - s25) / a[1] + s26 * (1 - s26) / b[1]) ** 0.5
        rows.append((s26 - s25, s25, s26, se, p))
    rows.sort(reverse=True)
    return rows


def improved_fit(mlp_games, cohort, lam=8.0):
    """(B) delta in an MLP-games-only ridge margin fit (pts/game),
    anchored so the returning full-time cohort's mean delta is zero."""
    def fit(year):
        gs = [g for g in mlp_games if g["date"][:4] == year]
        idx = {}
        for g in gs:
            for p in on_court(g):
                idx.setdefault(p, len(idx))
        n = len(idx)
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n
        for g in gs:
            m = int(g["t1_score"]) - int(g["t2_score"])
            xs = [(idx[g["t1_p1"]], 1), (idx[g["t1_p2"]], 1),
                  (idx[g["t2_p1"]], -1), (idx[g["t2_p2"]], -1)]
            for i, si in xs:
                b[i] += si * m
                for j, sj in xs:
                    A[i][j] += si * sj
        for i in range(n):
            A[i][i] += lam
        for c in range(n):                       # gaussian elimination
            piv = max(range(c, n), key=lambda r: abs(A[r][c]))
            A[c], A[piv] = A[piv], A[c]
            b[c], b[piv] = b[piv], b[c]
            for r in range(c + 1, n):
                f = A[r][c] / A[c][c]
                if f:
                    for k in range(c, n):
                        A[r][k] -= f * A[c][k]
                    b[r] -= f * b[c]
        x = [0.0] * n
        for r in range(n - 1, -1, -1):
            x[r] = (b[r] - sum(A[r][k] * x[k] for k in range(r + 1, n))) / A[r][r]
        return {p: x[i] for p, i in idx.items()}

    th25, th26 = fit("2025"), fit("2026")
    deltas = {p: th26[p] - th25[p] for p in cohort if p in th25 and p in th26}
    anchor = sum(deltas.values()) / len(deltas)
    rows = sorted(((d - anchor, th25[p], th26[p], p) for p, d in deltas.items()),
                  reverse=True)
    return rows, anchor


# ------------------------------------------------------------------ matchup WPA
def wpa(games, table):
    """WPA from MLP's own matchup records (order + walkovers included)."""
    bygid = {g["match_id"].lower(): g for g in games}
    matchups = defaultdict(list)
    for r in table:
        matchups[r["matchup_id"]].append(r)
    stats = defaultdict(lambda: {"wpa": 0.0, "w": 0, "n": 0, "clinch": 0,
                                 "save": 0, "lev": 0.0})
    team = defaultdict(lambda: {"w": 0, "l": 0, "db": 0, "earned": defaultdict(float)})
    slot_lev = defaultdict(lambda: [0.0, 0])
    for mid, rows in matchups.items():
        rows.sort(key=lambda r: int(r["game_slot"]))
        a = b = 0
        for i, r in enumerate(rows):
            a_won = r["winner_side"] == "1"
            g = bygid.get(r["match_id"]) if r["completed_type"] == "5" else None
            if g is None:      # walkover / dead game: score moves, nobody banks
                a, b = (a + 1, b) if a_won else (a, b + 1)
                continue
            swing = (matchup_p(a + 1, b) if a_won else matchup_p(a, b + 1)) \
                - matchup_p(a, b)
            t1_won = int(g["t1_score"]) > int(g["t2_score"])
            winners = (g["t1_p1"], g["t1_p2"]) if t1_won else (g["t2_p1"], g["t2_p2"])
            facing_elim = (b == 2 and a_won) or (a == 2 and not a_won)
            a, b = (a + 1, b) if a_won else (a, b + 1)
            clinched = max(a, b) == 3
            slot_lev[i][0] += abs(swing)
            slot_lev[i][1] += 1
            for p in on_court(g):
                s = stats[p]
                won = p in winners
                s["wpa"] += abs(swing) if won else -abs(swing)
                s["n"] += 1
                s["lev"] += abs(swing)
                s["w"] += won
                if won and clinched:
                    s["clinch"] += 1
                if won and facing_elim:
                    s["save"] += 1
                tkey = rows[0]["team_one"] if (won == a_won) else rows[0]["team_two"]
                team[tkey]["earned"][p] += abs(swing) if won else -abs(swing)
        for side, wins, opp in ((rows[0]["team_one"], a, b), (rows[0]["team_two"], b, a)):
            team[side]["w" if wins >= 3 else "l" if opp >= 3 else "db"] += 1
    return stats, team, slot_lev, len(matchups)


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

    out = defaultdict(lambda: {"gwae": 0.0, "w": 0, "n": 0, "poe": 0.0})
    for g in games:
        month = g["date"][:7]
        t1 = (g["t1_p1"], g["t1_p2"])
        t2 = (g["t2_p1"], g["t2_p2"])
        v1 = [val_at(p, month) for p in t1]
        v2 = [val_at(p, month) for p in t2]
        s1 = sum(v1) + GAMMA * abs(v1[0] - v1[1])
        s2 = sum(v2) + GAMMA * abs(v2[0] - v2[1])
        p1 = p_game(sig(s1 - s2))
        em = race_dist(round(sig(s1 - s2), 4), 11)["exp_margin"]
        am = int(g["t1_score"]) - int(g["t2_score"])
        w1 = int(g["t1_score"]) > int(g["t2_score"])
        for p in t1:
            out[p]["gwae"] += (1 if w1 else 0) - p1
            out[p]["poe"] += am - em
            out[p]["w"] += w1
            out[p]["n"] += 1
        for p in t2:
            out[p]["gwae"] += (0 if w1 else 1) - (1 - p1)
            out[p]["poe"] += em - am
            out[p]["w"] += not w1
            out[p]["n"] += 1
    return out


def main():
    players, mlp, games, table = load()
    nm = lambda p: name(players, p)
    print(f"MLP {SEASON[:4]} doubles games (no DB/forfeit): {len(games)}\n")

    counts = mlp_game_counts(mlp)
    cohort = {p for p, n26 in counts["2026"].items()
              if n26 >= IMPROVED_MIN_MLP_GAMES
              and counts["2025"].get(p, 0) >= IMPROVED_MIN_MLP_GAMES}
    print(f"=== MOST IMPROVED, MLP-ONLY: full-timers (>= {IMPROVED_MIN_MLP_GAMES}"
          f" MLP games each season; cohort {len(cohort)}) ===")
    print("  (A) combined serve+return rally point share in MLP play")
    ra = improved_rally(cohort)
    for gender, label in (("F", "WOMEN"), ("M", "MEN")):
        sub = [r for r in ra if players.get(r[4], {}).get("gender") == gender]
        print(f"  -- {label} --")
        for d, s25, s26, se, pid in sub[:5]:
            print(f"  {nm(pid):30s} {s25:.3f} -> {s26:.3f}   delta {d:+.3f}"
                  f" (se {se:.3f})")
    rows_b, anchor = improved_fit(mlp, cohort)
    print(f"  (B) MLP-games-only ridge margin fit, cohort-anchored"
          f" (shift {anchor:+.2f} pts/game)")
    for gender, label in (("F", "WOMEN"), ("M", "MEN")):
        sub = [r for r in rows_b if players.get(r[3], {}).get("gender") == gender]
        print(f"  -- {label} --")
        for d, v25, v26, pid in sub[:5]:
            print(f"  {nm(pid):30s} {v25:+5.2f} -> {v26:+5.2f}   delta {d:+.2f}"
                  f"   MLP games {counts['2025'][pid]}->{counts['2026'][pid]}")

    stats, team, slot_lev, n_mu = wpa(games, table)
    ng = sum(s["n"] for s in stats.values()) // 4
    print(f"\n=== MVP: MATCHUP WPA ({n_mu} matchups from MLP records, "
          f"{ng}/{len(games)} games) ===")
    for pid, s in sorted(stats.items(), key=lambda kv: -kv[1]["wpa"])[:12]:
        print(f"  {nm(pid):30s} {s['wpa']:+5.2f}   {s['w']:2d}-{s['n']-s['w']:<2d}"
              f"   clinchers {s['clinch']:2d}   elim-saves {s['save']}"
              f"   avg leverage {s['lev']/s['n']:.3f}")
    mxc = max(s["clinch"] for s in stats.values())
    print(f"  clincher leaders ({mxc}): "
          + ", ".join(sorted(nm(p) for p, s in stats.items() if s["clinch"] == mxc)))
    mxs = max(s["save"] for s in stats.values())
    print(f"  elim-save leaders ({mxs}): "
          + ", ".join(sorted(nm(p) for p, s in stats.items() if s["save"] == mxs)))
    print("  avg |swing| by game slot: "
          + "  ".join(f"g{i+1} {v[0]/v[1]:.3f} (n={v[1]})"
                      for i, v in sorted(slot_lev.items())))
    regs = sorted((s["lev"] / s["n"], nm(p), s["n"]) for p, s in stats.items()
                  if s["n"] >= 30)
    print("  lowest avg leverage (>=30 games): "
          + ", ".join(f"{n} {v:.3f} ({c} gms)" for v, n, c in regs[:3]))
    print("  team records (decided W-L, +DB matchups) and roster identity:")
    for t, r in sorted(team.items(), key=lambda kv: -(kv[1]["w"] - kv[1]["l"]))[:3]:
        earned = sum(r["earned"].values())
        print(f"    {t:26s} {r['w']}-{r['l']} (+{r['db']} DB)   "
              f"sum within-team WPA {earned:+.2f}")
    nj = team.get("New Jersey 5s", {"earned": {}})
    print("  NJ 5s within-team WPA: "
          + ", ".join(f"{nm(p)} {v:+.2f}"
                      for p, v in sorted(nj["earned"].items(), key=lambda kv: -kv[1])))

    g = gwae(players, games)
    print(f"\n=== UNDER THE RADAR: GWAE (min {GWAE_MIN_GAMES} games) ===")
    for pid, s in sorted(g.items(), key=lambda kv: -kv[1]["gwae"])[:20]:
        if s["n"] < GWAE_MIN_GAMES:
            continue
        print(f"  {nm(pid):30s} {s['gwae']:+5.1f}   {s['w']:2d}-{s['n']-s['w']:<2d}"
              f"   (expected {s['w']-s['gwae']:.1f} wins)")

    print("\n=== ALTERNATIVES (considered + set aside; see the note) ===")
    pts = defaultdict(int)
    for gm in games:
        for p in (gm["t1_p1"], gm["t1_p2"]):
            pts[p] += int(gm["t1_score"])
        for p in (gm["t2_p1"], gm["t2_p2"]):
            pts[p] += int(gm["t2_score"])
    top = max(pts.items(), key=lambda kv: kv[1])
    print(f"  total points won: {nm(top[0])} {top[1]}")
    poe = max(g.items(), key=lambda kv: kv[1]["poe"])
    print(f"  points over expectation: {nm(poe[0])} {poe[1]['poe']:+.0f}")
    yv = {r["player_id"]: float(r["value"])
          for r in csv.DictReader(open(DATA / "yearly_values.csv")) if r["year"] == "2026"}
    active_counts = defaultdict(int)
    for gm in games:
        for p in on_court(gm):
            active_counts[p] += 1
    vals = sorted(yv[p] for p in active_counts if p in yv)
    repl = vals[int(0.2 * len(vals))]
    par = sorted(((yv[p] - repl) * n / 2, p) for p, n in active_counts.items()
                 if p in yv)
    print(f"  points above replacement (repl = 20th-pctile 2026 v1 value "
          f"{repl:+.2f}): {nm(par[-1][1])} {par[-1][0]:.0f}")


if __name__ == "__main__":
    main()
