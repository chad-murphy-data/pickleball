"""value_cap/dials_probe.py -- what, other than her price, moves the Waters
team relative to the Bright and Johns teams (user question, 2026-09-04 late).

    python value_cap/dials_probe.py        # ~2 min; prints every table in dials.md

One perfect-information snake draft on the mlp2026 board with the shipped
tag list gives the three teams. Then, holding the drafted field fixed:
DreamBreaker as a coin flip; Waters absent / rotated (and the measured
2026 playing time of every franchise's best players); a rival roster
best-responding to HER roster instead of the generic reference; split
gender caps ($500k + $500k, two $10M pools); and, in the second half,
season formats -> the favourite's title odds and how many teams can win.
Stdlib + the value_cap engine; seeds fixed."""
import sys, csv, random, statistics, math
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0]]
import draft_sim as D
from phase2_pricing import (DOUBLES, SINGLES, FLOOR, NAME, POOL, pid_named, prices, prices_tagged,
                            candidates, best_roster, win, cost, TEAM_CAP)
from fast_tie import FastTie, db_win
from sitelib.race import sigmoid

D.set_board("mlp2026")
waters, bright, johns = pid_named("Anna Leigh Waters"), pid_named("Anna Bright"), pid_named("Ben Johns")
price = prices_tagged(POOL, 1.0, waters, "joint")
price_b = {u: price.get(u, FLOOR) for u in D.BOARD}
E = D.TRUE_ENGINE
rng = random.Random(1)
rosters, spent, picks, left, owners = D.run_draft(price_b, "snake", 0.0, rng)
teams = {}
for t, r in enumerate(rosters):
    for s, nm in ((waters, "Waters"), (bright, "Bright"), (johns, "Johns")):
        if s in r: teams[nm] = (t, r)
def vs_field(r, others):
    return statistics.mean(E.tie(r, o) for o in others if o != r)
field = list(rosters)
print("== perfect-info snake draft, mlp2026 board, tag list ==")
for nm, (t, r) in teams.items():
    print(f"{nm} team (slot {t+1}): vs field {100*vs_field(r, field):.1f}%  vs reference {100*E.tie(r, D.REFERENCE):.1f}%  spend ${cost(r, price_b)/1e3:.0f}k")
    print("   ", ", ".join(f"{NAME.get(u, u[:6])} ${price_b[u]/1e3:.0f}k" for u in r))
W = teams["Waters"][1]
print("Waters team vs Bright team:", f"{100*E.tie(W, teams['Bright'][1]):.1f}%", " vs Johns team:", f"{100*E.tie(W, teams['Johns'][1]):.1f}%")

# per-game decomposition of the Waters team vs the Bright team and vs field
import fast_tie
def games2(a, b):
    wa, ma, xa1, xa2, dba = E.lineup(a); wb, mb, xb1, xb2, dbb = E.lineup(b)
    return (E.game(wa, wb, "WD"), E.game(ma, mb, "MD"), E.game(xa1, xb1, "MXD"), E.game(xa2, xb2, "MXD"),
            db_win(round(sigmoid(fast_tie.K_DB * (dba - dbb)), 4)))
g = [games2(W, o) for o in field if o != W]
print("Waters team per-game vs field (WD, MD, MXD1, MXD2, DB):", [f"{100*statistics.mean(x[i] for x in g):.0f}%" for i in range(5)])

# 1. DreamBreaker as a coin flip
def tie_db(a, b, pdb):
    wa, ma, xa1, xa2, _ = E.lineup(a); wb, mb, xb1, xb2, _ = E.lineup(b)
    ps = (E.game(wa, wb, "WD"), E.game(ma, mb, "MD"), E.game(xa1, xb1, "MXD"), E.game(xa2, xb2, "MXD"))
    d = [1.0, 0, 0, 0, 0]
    for p in ps:
        q = 1 - p
        d = [d[0]*q, d[1]*q + d[0]*p, d[2]*q + d[1]*p, d[3]*q + d[2]*p, d[4] + d[3]*p]
    return d[3] + d[4] + d[2] * pdb
print("\n== dial: DreamBreaker replaced by a coin flip (same rosters) ==")
for nm, (t, r) in teams.items():
    print(f"{nm} team vs field: {100*statistics.mean(tie_db(r, o, 0.5) for o in field if o != r):.1f}%")

# 2. Waters absent: replaced by the best woman left on the board
women_left = sorted((u for u in left if D.GENDER[u] == "F"), key=lambda u: -E.v[u])
sub = women_left[0]
W_out = tuple(sub if u == waters else u for u in W)
p_in = vs_field(W, field); p_out = vs_field(W_out, field)
print(f"\n== dial: playing time / availability ==\nWaters team vs field with her: {100*p_in:.1f}%; without her (sub {NAME.get(sub, sub[:6])}): {100*p_out:.1f}%")
for f in (1.0, 0.9, 0.8, 0.75, 0.67, 0.5):
    print(f"  she plays {f:.0%} of ties -> expected {100*(f*p_in + (1-f)*p_out):.1f}%")
B = teams["Bright"][1]; J = teams["Johns"][1]
B_out = tuple(sub if u == bright else u for u in B)
men_left = sorted((u for u in left if D.GENDER[u] == "M"), key=lambda u: -E.v[u])
J_out = tuple(men_left[0] if u == johns else u for u in J)
print(f"Bright team with/without her: {100*vs_field(B, field):.1f}% / {100*vs_field(B_out, field):.1f}%;  Johns team with/without him: {100*vs_field(J, field):.1f}% / {100*vs_field(J_out, field):.1f}%")

# 3. measured 2026 playing time of each franchise's best players
DATA = str(Path(__file__).resolve().parents[1] / "data")
mm = {r["match_id"]: r for r in csv.DictReader(open(f"{DATA}/mlp_matchups_2026.csv"))}
team_mus = defaultdict(set); player_mus = defaultdict(lambda: defaultdict(set))
for g in csv.DictReader(open(f"{DATA}/games.csv")):
    if g["tour"] != "MLP" or not g["date"].startswith("2026") or g["is_dreambreaker"] == "True":
        continue
    m = mm.get(g["match_id"])
    if not m or m["winner_side"] not in ("1", "2"):
        continue
    t1_won = int(g["t1_score"]) > int(g["t2_score"])
    t1_is_one = (m["winner_side"] == "1") == t1_won
    team_mus[m["team_one"]].add(m["matchup_id"]); team_mus[m["team_two"]].add(m["matchup_id"])
    for u in (g["t1_p1"], g["t1_p2"]):
        player_mus[u.lower()][m["team_one"] if t1_is_one else m["team_two"]].add(m["matchup_id"])
    for u in (g["t2_p1"], g["t2_p2"]):
        player_mus[u.lower()][m["team_two"] if t1_is_one else m["team_one"]].add(m["matchup_id"])
print("\n== measured 2026: share of the franchise's matchups its top players actually played ==")
rows = []
for fr, mus in team_mus.items():
    members = [(u, len(s[fr])) for u, s in player_mus.items() if fr in s]
    members.sort(key=lambda x: -DOUBLES.get(x[0], {"v": -9})["v"])
    top = members[:2]
    rows.append((fr, len(mus), [(NAME.get(u, u[:6]), n / len(mus)) for u, n in top], len(members)))
rows.sort(key=lambda r: -r[2][0][1])
for fr, n, tops, nm in rows:
    print(f"  {fr:22s} {n:3d} matchups, {nm} players used; " + "; ".join(f"{a} {100*f:.0f}%" for a, f in tops))
shares = [f for _, _, tops, _ in rows for _, f in tops[:1]]
print(f"  mean share for the #1 player: {100*statistics.mean(shares):.0f}%  (median {100*statistics.median(shares):.0f}%)")

# 4. a rival built to beat the Waters team specifically
print("\n== a rival roster built to beat the Waters team (best response vs HER roster, tag prices) ==")
p_r, R = best_roster(price, W, exclude=W)
print(f"rival: " + ", ".join(f"{NAME[u]} ${price[u]/1e3:.0f}k" for u in R) + f"  spend ${cost(R, price)/1e3:.0f}k")
print(f"  beats Waters team {100*p_r:.1f}%; vs reference {100*E.tie(R, D.REFERENCE):.1f}%; vs the drafted field {100*vs_field(R, field):.1f}%")
print(f"  per game vs Waters team (WD, MD, MXD1, MXD2, DB): " + ", ".join(f"{100*x:.0f}%" for x in games2(R, W)))
print(f"  compare: Bright's drafted team beats Waters team {100*E.tie(B, W):.1f}%, Johns' {100*E.tie(J, W):.1f}%")
p_g, G = best_roster(price, D.REFERENCE, exclude=W)
print(f"generic best-vs-reference roster (Waters excluded): beats Waters team {100*E.tie(G, W):.1f}%, vs field {100*vs_field(G, field):.1f}%")

# 5. split caps: 3 women for $500k, 3 men for $500k, each gender its own $10M pool
print("\n== dial: split caps ($500k women / $500k men, two $10M pools) ==")
ps = prices(POOL, 1.0, "split")
print(f"Waters' curve price in the women's $10M pool: ${ps[waters]/1e3:.0f}k (cap per gender $500k); Bright ${ps[bright]/1e3:.0f}k; Johns ${ps[johns]/1e3:.0f}k")
# tag within the women's pool: cap - 2 cheapest women, surplus redistributed over the other women
women = [u for u, _, _ in POOL["F"] if u != waters]
w = {u: ps[u] - FLOOR for u in women}; wt = sum(w.values())
for _ in range(5):
    cheapest2 = sorted(ps[u] for u in women)[:2]
    tag = 500_000 - sum(cheapest2)
    surplus = prices(POOL, 1.0, "split")[waters] - tag
    new = {u: FLOOR + w[u] + surplus * w[u] / wt for u in women}; new[waters] = tag
    ps.update(new)
print(f"Waters tagged at ${ps[waters]/1e3:.0f}k; Bright now ${ps[bright]/1e3:.0f}k; Todd ${ps[pid_named('Parris Todd')]/1e3:.0f}k")
cheap_w = tuple(sorted(women, key=lambda u: ps[u])[:2])
best = (-1, None)
for m in candidates("M", ps, 500_000, 60):
    r = m + (waters,) + cheap_w
    p = E.tie(r, D.REFERENCE)
    if p > best[0]: best = (p, r)
Wsplit = best[1]
print(f"Waters team under split caps: " + ", ".join(f"{NAME[u]} ${ps[u]/1e3:.0f}k" for u in Wsplit))
print(f"  vs reference {100*best[0]:.1f}% (joint-cap Waters team vs reference {100*E.tie(W, D.REFERENCE):.1f}%)")
over = [(NAME[u], ps[u]) for u in (bright, johns) if ps[u] > 500_000]
print("also over a $500k gender cap under the split (would need tags of their own): " +
      ", ".join(f"{n} ${p/1e3:.0f}k" for n, p in over))


# 6. season formats: same rosters, different seasons -> title odds and contenders
print("\n== dial: season format (same drafted rosters) ==")
n = 20
P = [[0.5 if i == j else E.tie(rosters[i], rosters[j]) for j in range(n)] for i in range(n)]
wslot = teams["Waters"][0]
def bracket(seeds, rng):
    """single elimination on a list of team indices (len = power of 2), 1v last seeding."""
    alive = list(seeds)
    while len(alive) > 1:
        nxt = []
        k = len(alive)
        for a in range(k // 2):
            i, j = alive[a], alive[k - 1 - a]
            nxt.append(i if rng.random() < P[i][j] else j)
        alive = nxt
    return alive[0]

def rr(legs, rng):
    wins = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            for _ in range(legs):
                if rng.random() < P[i][j]: wins[i] += 1
                else: wins[j] += 1
    return sorted(range(n), key=lambda t: (-wins[t], rng.random()))

def season(fmt, rng):
    if fmt == "double RR + top 4":            # the sim's current format (38 ties)
        order = rr(2, rng); return bracket(order[:4], rng)
    if fmt == "single RR + top 4":            # 19 ties
        order = rr(1, rng); return bracket(order[:4], rng)
    if fmt == "single RR + top 8":
        order = rr(1, rng); return bracket(order[:8], rng)
    if fmt == "double RR, table only":
        return rr(2, rng)[0]
    if fmt == "16-team bracket, random seeds":
        s = list(range(n)); rng.shuffle(s); return bracket(s[:16], rng)
    if fmt == "3 events: 4-tie pool + 8-team bracket, most event titles":
        # each event: random pools of 5 (4 ties), top 8 by pool wins + coin flip -> bracket; champion = most event wins, tie -> coin flip
        titles = [0] * n
        for _ in range(3):
            s = list(range(n)); rng.shuffle(s)
            wins = [0] * n
            for p in range(4):
                pool = s[p*5:(p+1)*5]
                for a in range(5):
                    for b in range(a + 1, 5):
                        i, j = pool[a], pool[b]
                        if rng.random() < P[i][j]: wins[i] += 1
                        else: wins[j] += 1
            order = sorted(range(n), key=lambda t: (-wins[t], rng.random()))
            titles[bracket(order[:8], rng)] += 1
        return max(range(n), key=lambda t: (titles[t], rng.random()))
    raise ValueError(fmt)

FMTS = ["double RR + top 4", "single RR + top 4", "single RR + top 8", "double RR, table only",
        "16-team bracket, random seeds", "3 events: 4-tie pool + 8-team bracket, most event titles"]
S = 4000
print(f"Waters team = slot {wslot+1}, tie win% vs field {100*statistics.mean(P[wslot][j] for j in range(n) if j != wslot):.1f}%")
print("| format | Waters team title | runner-up favourite | teams >= 10% | teams >= 5% | effective contenders |")
print("|---|---|---|---|---|---|")
for fmt in FMTS:
    rng = random.Random(7)
    cnt = [0] * n
    for _ in range(S):
        cnt[season(fmt, rng)] += 1
    p = sorted((c / S for c in cnt), reverse=True)
    eff = 1 / sum(x * x for x in p)
    print(f"| {fmt} | {100*cnt[wslot]/S:.0f}% | {100*p[1]:.0f}% | {sum(1 for x in p if x >= .10)} | {sum(1 for x in p if x >= .05)} | {eff:.1f} |")
