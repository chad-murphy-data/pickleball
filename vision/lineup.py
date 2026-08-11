"""Who is standing where, at every serve — from the referee log alone.

This is the identity anchor the vision layer was missing.  The first
attribution attempt tried to name players from appearance and calibrated at
57% against a ~60% ceiling, i.e. nothing.  It did not need to: side-out
doubles is a state machine, and the log hands us its inputs.

THE STATE MACHINE
    Each player occupies one half of their own side of the court, called
    RIGHT or LEFT from that team's own perspective facing the net.  Two
    facts move the state and nothing else does:

      * a team's two players SWAP halves exactly when that team wins a
        rally while serving (that is what scoring a point means here), and
      * the receiver is always the opponent DIAGONALLY opposite the
        server, i.e. the opponent on the same designated half.

    So: initialise the four halves at the start of a game, then walk the
    log.  The server's identity is logged; the server's half is read out of
    the state; the receiver is whoever stands opposite.

WHY THIS IS WORTH ANYTHING
    The log gives server and receiver but says nothing about the other two
    players, and nothing at all about court position.  The state machine
    turns those two names into all four positions at every serve.  That is
    a per-rally identity anchor for the tracker: four blobs, four known
    labels, no appearance model, no hand labelling.

    It also self-validates.  The receiver is BOTH an input we could use and
    an output we can predict, so predicting it and comparing against the
    logged value scores the model on every rally of every match at zero
    cost.  Init uses rally 1's receiver only; rallies 2..n are honest
    predictions.

    Score parity is a redundant CHECK, not an input: the first server of a
    service turn stands on the right when their team's score is even.  It
    is reported, not relied on, because the convention at 0-0-2 (game's
    opening service turn is called "second server" though the server stands
    where a first server would) breaks a naive parity rule and does not
    touch the state machine at all.

WHAT STILL NEEDS THE VIDEO
    Two bits per game, and only two: which team is at the near end, and
    whether a team's "right" is image-right.  Everything else above is
    determined.  Both bits are massively over-identified by ~30 rallies of
    agreement, so the video resolves them by consistency rather than trust.

    python vision/lineup.py --match <uuid>
    python vision/lineup.py --validate-all      # every 2026 MLP match
Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "vision"

RIGHT, LEFT = "R", "L"


def other(h):
    return LEFT if h == RIGHT else RIGHT


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
CACHE = ROOT / "raw" / "rally_logs"          # gitignored; raw/ already is


def load_rallies(match_id):
    """Rally rows for a match, from the cached timeline or the live log."""
    p = OUT / f"rally_timeline_{match_id[:8]}.csv"
    if p.exists():
        rows = list(csv.DictReader(open(p)))
    else:
        import rally_timeline as rt                        # noqa: E402

        c = CACHE / f"{match_id}.json"
        if c.exists():
            log = json.loads(c.read_text())
        else:
            log = rt.fetch(match_id)
            CACHE.mkdir(parents=True, exist_ok=True)
            c.write_text(json.dumps(log))
            import time

            time.sleep(0.8)                                # be polite
        rows, _, _, _ = rt.build(log)
    if not rows:
        raise ValueError(f"no rallies for {match_id}")
    return rows


def player_names():
    f = DATA / "players.csv"
    if not f.exists():
        return {}
    return {r["player_id"]: r["full_name"] for r in csv.DictReader(open(f))}


# --------------------------------------------------------------------------
# the state machine
# --------------------------------------------------------------------------
def walk_game(rallies):
    """Replay one game.  Returns (per-rally records, diagnostics).

    State is halves[player] in {R, L}.  Teams are inferred from the log:
    server and receiver are always opponents, so the serve pairs partition
    the four players into two sides.
    """
    # --- work out the two teams from who serves to whom -------------------
    opp = {}
    for r in rallies:
        opp.setdefault(r["server_uuid"], set()).add(r["receiver_uuid"])
        opp.setdefault(r["receiver_uuid"], set()).add(r["server_uuid"])
    players = sorted(opp)
    if len(players) != 4:
        return [], {"error": f"{len(players)} players in log, expected 4"}
    a0 = players[0]
    team_a = [a0] + [p for p in players if p != a0 and p not in opp[a0]]
    team_b = [p for p in players if p not in team_a]
    if len(team_a) != 2 or len(team_b) != 2:
        return [], {"error": "could not split into two teams"}
    team_of = {p: "A" for p in team_a}
    team_of.update({p: "B" for p in team_b})
    mates = {team_a[0]: team_a[1], team_a[1]: team_a[0],
             team_b[0]: team_b[1], team_b[1]: team_b[0]}

    # --- initialise from rally 1 ------------------------------------------
    # The opening server of a game stands on the right (team score 0, even).
    # The receiver is diagonal, hence also on the right.  Their partners
    # take the left halves.  This is the ONLY place a logged receiver is
    # consumed; every later one is a prediction.
    r0 = rallies[0]
    halves = {r0["server_uuid"]: RIGHT, mates[r0["server_uuid"]]: LEFT,
              r0["receiver_uuid"]: RIGHT, mates[r0["receiver_uuid"]]: LEFT}

    # --- the independent check --------------------------------------------
    # The rule is NOT about the current server (both partners serve in a turn
    # and only one can be on the parity half).  It anchors on the player who
    # served FIRST in the game for that team: that player stands right when
    # their team's score is even, left when odd, whether serving or
    # receiving.  So each rally checks BOTH teams, and the check consumes the
    # SCORE — a column the state machine never reads.
    anchor = {}
    for r in rallies:
        anchor.setdefault(team_of[r["server_uuid"]], r["server_uuid"])

    recs, hits, tested, parity_ok, parity_tested = [], 0, 0, 0, 0
    for i, r in enumerate(rallies):
        srv, rcv = r["server_uuid"], r["receiver_uuid"]
        srv_half = halves[srv]
        # predicted receiver: the opponent standing on the same half
        pred = next(p for p in (team_b if team_of[srv] == "A" else team_a)
                    if halves[p] == srv_half)
        if i > 0:                                  # rally 1 initialised us
            tested += 1
            hits += pred == rcv

        # parity check (independent; never fed back into the state)
        score = r["start_score"].split("-")
        parity = None
        if len(score) == 3 and score[0].isdigit() and score[1].isdigit():
            srv_team = team_of[srv]
            by_team = {srv_team: int(score[0]),
                       ("B" if srv_team == "A" else "A"): int(score[1])}
            checks = [(RIGHT if s % 2 == 0 else LEFT) == halves[anchor[t]]
                      for t, s in by_team.items() if t in anchor]
            parity = all(checks)
            parity_tested += len(checks)
            parity_ok += sum(checks)

        recs.append({
            "rally": r["rally"], "game": r["game"],
            "t_start": r["t_start"], "t_end": r["t_end"],
            "outcome": r["outcome"], "start_score": r["start_score"],
            "server_uuid": srv, "receiver_uuid": rcv,
            "pred_receiver_uuid": pred,
            "receiver_ok": int(pred == rcv),
            "parity_ok": "" if parity is None else int(parity),
            "team_A_R": next(p for p in team_a if halves[p] == RIGHT),
            "team_A_L": next(p for p in team_a if halves[p] == LEFT),
            "team_B_R": next(p for p in team_b if halves[p] == RIGHT),
            "team_B_L": next(p for p in team_b if halves[p] == LEFT),
            "server_half": srv_half,
            "server_team": team_of[srv],
        })

        # --- transition: serving team swaps iff it won the rally ----------
        if r["outcome"] == "point":
            side = team_a if team_of[srv] == "A" else team_b
            for p in side:
                halves[p] = other(halves[p])

    diag = {"tested": tested, "hits": hits,
            "acc": hits / tested if tested else float("nan"),
            "parity_tested": parity_tested, "parity_ok": parity_ok,
            "team_A": team_a, "team_B": team_b}
    return recs, diag


def walk_match(rallies):
    """Games are independent — positions reset at each game start."""
    games, order = {}, []
    for r in rallies:
        if r["game"] not in games:
            games[r["game"]] = []
            order.append(r["game"])
        games[r["game"]].append(r)
    recs, tested, hits, ptested, pok = [], 0, 0, 0, 0
    for g in order:
        rr, d = walk_game(games[g])
        if "error" in d:
            print(f"  game {g}: {d['error']}", file=sys.stderr)
            continue
        recs.extend(rr)
        tested += d["tested"]
        hits += d["hits"]
        ptested += d["parity_tested"]
        pok += d["parity_ok"]
    return recs, {"tested": tested, "hits": hits,
                  "acc": hits / tested if tested else float("nan"),
                  "parity_tested": ptested, "parity_ok": pok}


# --------------------------------------------------------------------------
def run(match_id, quiet=False, write=True):
    rallies = load_rallies(match_id)
    recs, d = walk_match(rallies)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"lineup_{match_id[:8]}.csv"
    if recs and write:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
    if not quiet:
        nm = player_names()
        print(f"match {match_id}")
        print(f"  {len(recs)} rallies over {len({r['game'] for r in recs})} game(s)")
        print(f"  receiver prediction  {d['hits']}/{d['tested']} = "
              f"{100*d['acc']:.1f}%   (rally 1 of each game initialises, "
              f"the rest are predictions)")
        print(f"  parity cross-check   {d['parity_ok']}/{d['parity_tested']}")
        if recs:
            r = recs[0]
            f = lambda u: nm.get(u, u[:8])                       # noqa: E731
            print(f"  opening alignment    A: {f(r['team_A_R'])} (R) / "
                  f"{f(r['team_A_L'])} (L)   vs   B: {f(r['team_B_R'])} (R) / "
                  f"{f(r['team_B_L'])} (L)")
        print(f"  wrote {p.relative_to(ROOT)}")
    return recs, d


def validate_all(limit=None):
    """Score the state machine on every 2026 MLP match we have."""
    seen, rows = set(), []
    for r in csv.DictReader(open(DATA / "games.csv")):
        if r["tour"] != "MLP" or not r["date"].startswith("2026"):
            continue
        if r["is_dreambreaker"] == "1" or r["match_id"] in seen:
            continue
        seen.add(r["match_id"])
        rows.append(r)
    if limit:
        rows = rows[:limit]
    tot_t = tot_h = tot_pt = tot_p = 0
    perfect = bad = 0
    for i, r in enumerate(rows, 1):
        try:
            _, d = run(r["match_id"], quiet=True, write=False)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {r['match_id'][:8]}: {e}", file=sys.stderr)
            continue
        if not d["tested"]:
            continue
        tot_t += d["tested"]
        tot_h += d["hits"]
        tot_pt += d["parity_tested"]
        tot_p += d["parity_ok"]
        perfect += d["hits"] == d["tested"]
        if d["hits"] < d["tested"]:
            bad += 1
            print(f"  imperfect: {r['date']} {r['match_id'][:8]} "
                  f"{d['hits']}/{d['tested']}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(rows)} matches, running "
                  f"{100*tot_h/max(tot_t,1):.2f}%", flush=True)
    print(f"\nSTATE MACHINE vs {len(rows)} MLP 2026 matches")
    print(f"  receiver predicted correctly {tot_h}/{tot_t} = "
          f"{100*tot_h/max(tot_t,1):.2f}%")
    print(f"  matches with zero errors     {perfect} (imperfect: {bad})")
    print(f"  parity cross-check           {tot_p}/{tot_pt} = "
          f"{100*tot_p/max(tot_pt,1):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match")
    ap.add_argument("--validate-all", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a.validate_all:
        validate_all(a.limit)
    elif a.match:
        run(a.match)
    else:
        ap.error("need --match or --validate-all")


if __name__ == "__main__":
    main()
