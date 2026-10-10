"""Pooled one-off partnerships: does playing together (familiarity) matter, and
does it matter differently when the new pairing was FORCED (usual partner absent
from the event) vs CHOSEN (usual partner is playing the event with someone else)?

Out-of-sample: games dated >= 2026-06-01 are scored against the FROZEN pre-June
v2 fit (data/v2_players_train.csv, gamma from v2_fit_summary_train.json), with no
dyad term. Residual = side's point share - sigmoid(eta). Familiarity = games the
pair played together BEFORE this game (counted from games.csv, forward-clean:
it only uses the past, so a stopping rule cannot bias it). Cluster-robust SEs by
event. Reproduce: python model/chem_oneoff.py
"""
import csv, math, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data"
import sys
FULL = "--full" in sys.argv     # full 2024-26 history vs month-of-game ratings (in-sample, tracking-biased; contrast forced-vs-chosen is the read)
SPLIT = "2024-01-01" if FULL else "2026-06-01"
gam = json.load(open(ROOT / ("model/v2_fit_summary%s.json" % ("" if FULL else "_train"))))["scalars"]["gamma"]["mean"]
val = {r["player_id"]: (float(r["value_now_mean"]), int(r["games"])) for r in csv.DictReader(open(D / ("v2_players.csv" if FULL else "v2_players_train.csv")))}
traj = {}
if FULL:
    for r in csv.DictReader(open(D / "v2_trajectories.csv")): traj[(r["player_id"], r["month"])] = float(r["value_mean"])
def v_at(p, date):
    return traj.get((p, date[:7]), val[p][0]) if FULL else val[p][0]
games = [g for g in csv.DictReader(open(D / "games.csv")) if g["is_forfeit"] == "False"
         and g["scoring_format"] in ("sideout_11", "sideout_15") and g["is_dreambreaker"] == "False"]
games.sort(key=lambda g: g["date"])
sig = lambda x: 1 / (1 + math.exp(-x))
def team(a, b, date): va, vb = v_at(a, date), v_at(b, date); return va + vb + gam * abs(va - vb)

together = collections.Counter()          # pair -> games so far
last_pair = collections.defaultdict(lambda: collections.defaultdict(list))  # player -> partner -> [dates]
ev_players = collections.defaultdict(set)  # (event, context) -> players who played
for g in games: ev_players[(g["event_id"], g["context"])].update([g["t1_p1"], g["t1_p2"], g["t2_p1"], g["t2_p2"]])

def usual_partner(p, ctx, date):
    """partner with most prior games in this context within 150 days (>=10 games together)."""
    best, n = None, 0
    for q, c in cnt[(p, ctx)].items():
        if c > n: best, n = q, c
    return (best, n) if n >= 10 else (None, 0)
cnt = collections.defaultdict(lambda: collections.Counter())   # (player, ctx) -> partner counts (all prior)

rows = []
for g in games:
    sides = [(g["t1_p1"], g["t1_p2"]), (g["t2_p1"], g["t2_p2"])]
    live = g["date"] >= SPLIT
    if live and all(p in val and val[p][1] >= 60 for s in sides for p in s):
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 > 0:
            eta = team(*sides[0], g["date"]) - team(*sides[1], g["date"])
            res = s1 / (s1 + s2) - sig(eta)          # t1-oriented residual
            for k, (a, b) in enumerate(sides):
                pair = tuple(sorted((a, b)))
                prior = together[pair]
                # forced vs chosen for first-time pairs: did either player's usual partner skip the event?
                kind = ""
                if prior == 0:
                    pl = ev_players[(g["event_id"], g["context"])]
                    for x, y in ((a, b), (b, a)):
                        u, n = usual_partner(x, g["context"], g["date"])
                        if u is not None and u != y:
                            kind = "forced" if u not in pl else "chosen"
                            break
                rows.append(dict(ev=g["event_id"], k=prior, side=1 if k == 0 else -1, res=res, kind=kind,
                                 opp=together[tuple(sorted(sides[1 - k]))]))
    for (a, b) in sides:
        pair = tuple(sorted((a, b))); together[pair] += 1
        cnt[(a, g["context"])][b] += 1; cnt[(b, g["context"])][a] += 1

def cluster_mean(sub):
    """mean side-oriented residual x 1e3 and cluster-robust se (by event)."""
    ev = collections.defaultdict(list)
    for r in sub: ev[r["ev"]].append(r["side"] * r["res"])
    n = sum(len(v) for v in ev.values())
    if n == 0: return None
    m = sum(sum(v) for v in ev.values()) / n
    se = math.sqrt(sum((sum(x - m for x in v)) ** 2 for v in ev.values())) / n
    return n, 1e3 * m, 1e3 * se
def show(label, sub):
    r = cluster_mean(sub)
    print(f"  {label:34s} " + (f"n={r[0]:5d}  {r[1]:+7.1f} ± {r[2]:5.1f}  (x1e-3 share)" if r else "n=0"))

# keep games where the OTHER side is settled (>=15 prior games), so the contrast is new-vs-settled
print(f"post-{SPLIT} sides, established players only; other side settled (>=15 games together)")
base = [r for r in rows if r["opp"] >= 15]
print("familiarity curve (prior games together), residual of that side vs frozen-rating forecast:")
for lab, lo, hi in (("0 (first-ever)", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6-14", 6, 14), ("15-39", 15, 39), ("40+", 40, 10**6)):
    show(lab, [r for r in base if lo <= r["k"] <= hi])
print("first-ever pairings by cause (usual partner >=10 games together exists):")
fo = [r for r in base if r["k"] == 0]
for kind in ("forced", "chosen", ""):
    show({"forced": "FORCED (usual partner absent)", "chosen": "CHOSEN (usual partner playing elsewhere)", "": "no usual partner / unclassified"}[kind],
         [r for r in fo if r["kind"] == kind])
fc = [r for r in fo if r["kind"] in ("forced", "chosen")]
show("forced+chosen", fc)
