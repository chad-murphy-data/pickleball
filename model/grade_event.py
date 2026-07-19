"""Stored accuracy scorecard: PICKLES model vs DUPR, per event.

Grades completed pro doubles on winner accuracy (+ Brier for the model)
against the model's last-refit v2 values and each player's latest synced
DUPR. Auto-detects the current event(s) from data/tournament_state.json
(the live MLP event + any PPA tournaments), so it can run unattended in
the nightly refresh. Writes data/accuracy.json (machine) and
model/accuracy.md (the just-open-it scorecard) so the numbers live in the
repo instead of being recomputed by hand.

    python model/grade_event.py

Honesty notes baked into the output:
  * Model values come from the most recent v2 fit (data/v2_players.csv),
    so the current event is genuinely out-of-sample.
  * DUPR = latest synced rating per player (data/platform_ratings.csv);
    a weekend barely moves a pro's number, but it is "latest", not strictly
    pre-match, so treat rank ties gently.
  * DUPR gets winner accuracy only: it is a rating, not a probability, so
    there is no honest Brier to quote for it.
  * Uses each match's ACTUAL lineups (a reproducible retrodiction),
    distinct from the frozen pre-match receipts in receipts.json.
  * Grades whatever detail is currently cached in raw/, so coverage grows
    as more matches finish; re-run to refresh.
"""
from __future__ import annotations
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import (calibrate, race_dist, set_calibration,  # noqa: E402
                          sigmoid, team_eta)

DATA = ROOT / "data"
CAL = json.loads((ROOT / "web" / "calibration.json").read_text())
set_calibration(CAL["a"], CAL["b"], CAL["eps"])
EPS = CAL["eps"]


def lc(u):
    return u.lower() if u else u


def load_values():
    v = {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        v[lc(r["player_id"])] = float(r["value_now_mean"])
    return v


def load_dupr():
    d = {}
    for r in csv.DictReader((DATA / "platform_ratings.csv").open()):
        try:
            d[lc(r["player_id"])] = float(r["platform_rating_latest"])
        except (ValueError, KeyError):
            pass
    return d


def model_game_p(a, b, vals, T=11):
    """Calibrated prob pair a beats pair b in one game to T. None if unrated."""
    try:
        va1, va2 = vals[lc(a[0])], vals[lc(a[1])]
        vb1, vb2 = vals[lc(b[0])], vals[lc(b[1])]
    except KeyError:
        return None
    eta = team_eta(va1, va2, vb1, vb2)
    return calibrate(race_dist(round(sigmoid(eta), 4), T)["p_win"])


def dupr_pick(a, b, dupr):
    """+1 if DUPR favors pair a, -1 pair b, 0 tie, None if under-covered."""
    ra = [dupr[lc(u)] for u in a if lc(u) in dupr]
    rb = [dupr[lc(u)] for u in b if lc(u) in dupr]
    if len(ra) < 2 or len(rb) < 2:
        return None
    sa, sb = sum(ra), sum(rb)
    return 0 if sa == sb else (1 if sa > sb else -1)


def bo3(p):
    """Best-of-3 match win prob from a per-game prob p."""
    return p * p * (3 - 2 * p)


def matchup_p(game_ps, p_db=0.5):
    """Model prob team-A wins an MLP tie: >=3 game wins, or 2-2 into a
    coin-flip DreamBreaker. game_ps = prob A wins each doubles game."""
    dist = [1.0]
    for p in game_ps:
        nxt = [0.0] * (len(dist) + 1)
        for k, w in enumerate(dist):
            nxt[k] += w * (1 - p)
            nxt[k + 1] += w * p
        dist = nxt
    n = len(game_ps)
    p_ge3 = sum(dist[k] for k in range(3, n + 1))
    p_22 = dist[2] if n >= 2 else 0.0
    return min(max(p_ge3 + p_22 * p_db, EPS / 2), 1 - EPS / 2)


# ---------------------------------------------------------------- MLP
def grade_mlp(event_title, vals, dupr):
    """Game- and matchup-level grade of every completed matchup of one MLP
    event we have cached detail for (found by matchupGroupTitle)."""
    g_pk = [0, 0]
    g_du = [0, 0]
    briers = []
    mrows = []
    dates = []
    pk_m = du_m = du_m_dec = 0
    for fp in sorted(glob.glob(str(ROOT / "raw" / "matchup_data" / "*.json"))):
        d = (json.loads(Path(fp).read_text()).get("data") or {})
        if d.get("matchupGroupTitle") != event_title:
            continue
        s1, s2 = d.get("teamOneScore"), d.get("teamTwoScore")
        if s1 is None or s2 is None or s1 == s2:
            continue
        a1, a2 = d["teamOneAbbreviation"], d["teamTwoAbbreviation"]
        win_a = s1 > s2
        game_ps, du_favs = [], []
        for g in d["matches"]:
            if g.get("isTieBreaker") or g.get("winner") not in (1, 2):
                continue
            a = (g["teamOnePlayerOneUuid"], g["teamOnePlayerTwoUuid"])
            b = (g["teamTwoPlayerOneUuid"], g["teamTwoPlayerTwoUuid"])
            w_a = g["winner"] == 1
            p = model_game_p(a, b, vals)
            if p is not None:
                game_ps.append(p)
                briers.append((p - (1 if w_a else 0)) ** 2)
                g_pk[0 if (p >= 0.5) == w_a else 1] += 1
            dp = dupr_pick(a, b, dupr)
            if dp is not None and dp != 0:
                du_favs.append(dp)
                g_du[0 if (dp > 0) == w_a else 1] += 1
        if not game_ps:
            continue
        if d.get("plannedStartDate"):
            dates.append(d["plannedStartDate"][:10])
        pm = matchup_p(game_ps)
        pk_ok = (pm >= 0.5) == win_a
        pk_m += pk_ok
        favA = sum(1 for x in du_favs if x > 0)
        favB = sum(1 for x in du_favs if x < 0)
        dpick = a1 if favA > favB else (a2 if favB > favA else "toss-up")
        du_ok = None if dpick == "toss-up" else (dpick == (a1 if win_a else a2))
        if du_ok is not None:
            du_m_dec += 1
            du_m += du_ok
        pfav = a1 if pm >= 0.5 else a2
        mrows.append({
            "actual": f"{a1} {s1}-{s2} {a2}",
            "pickles": f"{pfav} {round(max(pm, 1 - pm) * 100)}%",
            "pickles_ok": bool(pk_ok),
            "dupr": dpick,
            "dupr_ok": (None if du_ok is None else bool(du_ok)),
        })
    if not mrows:
        return None
    return {
        "event": event_title,
        "dates": [min(dates), max(dates)] if dates else None,
        "matchups": {"n": len(mrows), "pickles_correct": pk_m,
                     "dupr_correct": du_m, "dupr_decided": du_m_dec},
        "games": {"pickles": f"{g_pk[0]}/{g_pk[0] + g_pk[1]}",
                  "dupr": f"{g_du[0]}/{g_du[0] + g_du[1]}",
                  "pickles_correct": g_pk[0], "pickles_n": g_pk[0] + g_pk[1],
                  "dupr_correct": g_du[0], "dupr_n": g_du[0] + g_du[1]},
        "brier": round(sum(briers) / len(briers), 3) if briers else None,
        "rows": mrows,
    }


# ---------------------------------------------------------------- PPA
def grade_ppa(ev, vals, dupr):
    divs_out = []
    for div in ev["divisions"]:
        pk = [0, 0]
        du = [0, 0]
        briers = []
        unrated = 0
        for m in div["matches"]:
            if m.get("winner") not in (1, 2):
                continue
            a, b = m["p1"], m["p2"]
            w_a = m["winner"] == 1
            p1 = model_game_p(a, b, vals)
            if p1 is None:
                unrated += 1
            else:
                pm = bo3(p1)
                briers.append((pm - (1 if w_a else 0)) ** 2)
                pk[0 if (pm >= 0.5) == w_a else 1] += 1
            dp = dupr_pick(a, b, dupr)
            if dp is not None and dp != 0:
                du[0 if (dp > 0) == w_a else 1] += 1
        if pk[0] + pk[1] + du[0] + du[1] == 0:
            continue
        divs_out.append({
            "division": div["title"],
            "pickles": f"{pk[0]}/{pk[0] + pk[1]}",
            "pickles_correct": pk[0], "pickles_n": pk[0] + pk[1],
            "dupr": f"{du[0]}/{du[0] + du[1]}",
            "dupr_correct": du[0], "dupr_n": du[0] + du[1],
            "brier": round(sum(briers) / len(briers), 3) if briers else None,
            "unrated_by_model": unrated,
        })
    if not divs_out:
        return None
    return {"event": ev["tournament"], "divisions": divs_out}


def pct(c, n):
    return f"{100 * c / n:.0f}%" if n else "—"


def render_md(mlp, ppa_list):
    L = ["# Accuracy scorecard — PICKLES vs DUPR",
         "",
         "_Auto-generated by `python model/grade_event.py` (runs in the "
         "nightly refresh). Winner accuracy on completed pro doubles; model "
         "values are from the last v2 fit (out-of-sample for the current "
         "event), DUPR is each player's latest synced rating. DUPR shows "
         "winner accuracy only — it's a rating, not a probability. Uses each "
         "match's **actual** lineups (a reproducible retrodiction) — distinct "
         "from the frozen pre-match receipts in `receipts.json`. Brier is "
         "game/match level._", ""]
    if mlp:
        m, g = mlp["matchups"], mlp["games"]
        span = f" — {mlp['dates'][0]}→{mlp['dates'][1]}" if mlp.get("dates") else ""
        L += [f"## {mlp['event']}{span} (MLP, team format)", "",
              f"**Matchup winners — PICKLES {m['pickles_correct']}/{m['n']} "
              f"({pct(m['pickles_correct'], m['n'])})** · DUPR "
              f"{m['dupr_correct']}/{m['dupr_decided']} decided "
              f"({m['n'] - m['dupr_decided']} toss-ups)  ",
              f"**Doubles games — PICKLES {g['pickles']} "
              f"({pct(g['pickles_correct'], g['pickles_n'])}), Brier "
              f"{mlp['brier']} · DUPR {g['dupr']} "
              f"({pct(g['dupr_correct'], g['dupr_n'])})**", "",
              "| Actual | PICKLES | ✓ | DUPR | ✓ |",
              "|---|---|:--:|---|:--:|"]
        for r in mlp["rows"]:
            pk = "✓" if r["pickles_ok"] else "✗"
            du = "—" if r["dupr_ok"] is None else ("✓" if r["dupr_ok"] else "✗")
            L.append(f"| {r['actual']} | {r['pickles']} | {pk} | "
                     f"{r['dupr']} | {du} |")
        L.append("")
    for ppa in ppa_list:
        L += [f"## {ppa['event']} (PPA, seeded draws)", "",
              "| Division | PICKLES | DUPR | Brier | model-unrated |",
              "|---|---|---|:--:|:--:|"]
        tpc = tpn = tdc = tdn = 0
        for d in ppa["divisions"]:
            L.append(f"| {d['division']} | {d['pickles']} "
                     f"({pct(d['pickles_correct'], d['pickles_n'])}) | "
                     f"{d['dupr']} ({pct(d['dupr_correct'], d['dupr_n'])}) | "
                     f"{d['brier']} | {d['unrated_by_model']} |")
            tpc += d["pickles_correct"]; tpn += d["pickles_n"]
            tdc += d["dupr_correct"]; tdn += d["dupr_n"]
        L.append(f"| **all divisions** | **{tpc}/{tpn} ({pct(tpc, tpn)})** | "
                 f"**{tdc}/{tdn} ({pct(tdc, tdn)})** | | |")
        L.append("")
    return "\n".join(L)


def main():
    vals = load_values()
    dupr = load_dupr()
    st = json.loads((DATA / "tournament_state.json").read_text())
    mlp_event = (st.get("mlp") or {}).get("event")
    mlp = grade_mlp(mlp_event, vals, dupr) if mlp_event else None
    ppa_list = [g for g in (grade_ppa(ev, vals, dupr)
                            for ev in st.get("ppa", [])) if g]
    out = {"generated_from": st.get("generated"),
           "calibration": {"a": CAL["a"], "b": CAL["b"], "eps": EPS},
           "mlp": mlp, "ppa": ppa_list}
    (DATA / "accuracy.json").write_text(json.dumps(out, indent=1))
    (ROOT / "model" / "accuracy.md").write_text(render_md(mlp, ppa_list))
    print("wrote data/accuracy.json and model/accuracy.md")
    if mlp:
        print(f"  MLP {mlp['event']}: matchups "
              f"{mlp['matchups']['pickles_correct']}/{mlp['matchups']['n']}, "
              f"games PICKLES {mlp['games']['pickles']} / DUPR {mlp['games']['dupr']}")
    for ppa in ppa_list:
        for d in ppa["divisions"]:
            print(f"  {ppa['event']} · {d['division']}: "
                  f"PICKLES {d['pickles']} / DUPR {d['dupr']}")


if __name__ == "__main__":
    main()
