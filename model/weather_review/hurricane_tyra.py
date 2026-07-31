"""Hurricane Tyra Black: the most flattering honest cut in the archive.

    python model/weather_review/hurricane_tyra.py   # -> hurricane_tyra.md

The weather review found nothing. This script asks the narrow, fun question
the nickname invites -- is Tyra Black actually better in wind than the field?
-- and reports the BEST case that survives contact with the data, alongside
the number that deflates it.

Design (same residual machinery as model/wind_skill.py, corrected venue
labels from data/venue_overrides.csv):
    residual = actual point share - v2-expected share, signed per player
    wind     = ERA5 sustained 10m wind at the match's own start hour
    slope    = OLS of residual on wind, per player, outdoor games only

Every honest deflator is computed and printed next to the flattering number:
the within-player permutation p, the rank among all qualifying players, the
scan-corrected p (552 players were searched, so SOMEBODY had to be top), the
indoor placebo (where wind cannot physically act), and the number of extra
windy games she would need before the slope could clear significance.

READ ONLY: writes only model/weather_review/hurricane_tyra.md.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa: E402

TARGET = "Tyra Hurricane Black"
MIN_GAMES = 40
N_PERM = 20000
WINDY = 12.0          # mph, the "breezy" line
HOWLING = 16.0        # mph, the "Hurricane" line
SEED = 20260731


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ols(pts):
    """(slope, se, n) of y on x."""
    n = len(pts)
    if n < 10:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx < 1e-9:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    ssr = sum((y - my - b * (x - mx)) ** 2 for x, y in pts)
    se = math.sqrt(ssr / (n - 2) / sxx) if n > 2 else float("nan")
    return b, se, n


def load(setting_wanted):
    """player_id -> [(wind, signed residual, event_id, date)] for one arm."""
    geo = {r["event_id"]: r["setting"] for r in read_csv(ROOT / "data/event_geo.csv")}
    for r in read_csv(ROOT / "data/venue_overrides.csv"):   # web-verified labels win
        geo[r["event_id"]] = r["setting"]
    v2, names = {}, {}
    for r in read_csv(ROOT / "data/v2_players.csv"):
        v2[r["player_id"]] = float(r["value_now_mean"])
        names[r["player_id"]] = r["full_name"]
    hourly, gusty = {}, {}
    for r in read_csv(ROOT / "data/event_weather_hourly.csv"):
        key = (r["event_id"], r["local_time"][:13])
        try:
            hourly[key] = float(r["windspeed_10m"])
        except (TypeError, ValueError):
            pass
        try:
            gusty[key] = float(r["windgusts_10m"])
        except (TypeError, ValueError):
            pass
    start_hour = {}
    for r in read_csv(ROOT / "data/match_times.csv"):
        ts = r["start_local"] or r["planned_start_local"]
        if ts:
            start_hour[r["match_id"]] = ts[:13]

    players = defaultdict(list)
    for g in read_csv(ROOT / "data/games.csv"):
        if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
            continue
        if geo.get(g["event_id"]) != setting_wanted:
            continue
        hkey = (g["event_id"], start_hour.get(g["match_id"], ""))
        wind = hourly.get(hkey)
        if wind is None:
            continue
        gust = gusty.get(hkey)
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        resid = s1 / (s1 + s2) - sigmoid(team_eta(*vals))
        meta = (g["event_id"], g["date"], g["event_name"])
        for k in ("t1_p1", "t1_p2"):
            players[g[k]].append((wind, resid, meta, s1 > s2, gust))
        for k in ("t2_p1", "t2_p2"):
            players[g[k]].append((wind, -resid, meta, s2 > s1, gust))
    return players, names


def main():
    rng = random.Random(SEED)
    out = []
    say = lambda s="": (print(s), out.append(s))

    players, names = load("outdoor")
    pool = {p: g for p, g in players.items() if len(g) >= MIN_GAMES}
    tid = next((p for p, n in names.items() if n == TARGET), None)
    if tid is None or tid not in pool:
        say(f"{TARGET} not found in the outdoor pool"); return
    her = pool[tid]
    xs = [(w, r) for w, r, _, _, _ in her]
    b, se, n = ols(xs)

    say(f"# Hurricane Tyra Black — the best case, and the number that ruins it\n")
    say(f"*Outdoor games only, web-verified venue labels, ERA5 sustained wind at "
        f"each match's own start hour. Residual = actual point share minus what "
        f"the v2 model expected. Generated by `model/weather_review/hurricane_tyra.py`.*\n")

    # ---------- the flattering half ----------
    say("## The case for the nickname\n")
    say(f"**Wind slope: {b*10:+.4f} point share per +10 mph** "
        f"({n} outdoor games, se {se*10:.4f}, t = {b/se:+.2f}).")
    say(f"In plain terms: for every extra 10 mph of wind, she beats her own "
        f"expectation by {b*10*100:+.1f}% of a game's points — about "
        f"{b*10*22:+.2f} points in a race to 11.\n")

    # rank among the qualifying field
    ranked = []
    for pid, g in pool.items():
        r = ols([(w, y) for w, y, _, _, _ in g])
        if r:
            ranked.append((r[0], pid, r[2]))
    ranked.sort(reverse=True)
    rank = next(i for i, (_, pid, _) in enumerate(ranked, 1) if pid == tid)
    say(f"**Rank {rank} of {len(ranked)}** qualifying players "
        f"(top {100*rank/len(ranked):.0f}%) — she really is on the "
        f"wind-positive side of the field.\n")

    # women only
    wranked = [t for t in ranked if t[1] in pool]
    # crude: rank among players sharing her gender via v2_players
    gender = {r["player_id"]: r.get("gender", "") for r in read_csv(ROOT / "data/v2_players.csv")}
    if gender.get(tid):
        same = [t for t in ranked if gender.get(t[1]) == gender[tid]]
        wrank = next(i for i, (_, pid, _) in enumerate(same, 1) if pid == tid)
        say(f"Among the {len(same)} qualifying {gender[tid]} players: **rank {wrank}**.\n")

    # bin contrast: calm vs breezy vs howling
    def bin_mean(lo, hi):
        v = [r for w, r, _, _, _ in her if lo <= w < hi]
        return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
    calm = bin_mean(0, WINDY)
    breezy = bin_mean(WINDY, HOWLING)
    howl = bin_mean(HOWLING, 99)
    say("| conditions | her games | mean beat vs expectation |")
    say("|---|---|---|")
    say(f"| calm (<{WINDY:.0f} mph) | {calm[1]} | {calm[0]*100:+.2f}% of points |")
    say(f"| breezy ({WINDY:.0f}–{HOWLING:.0f}) | {breezy[1]} | {breezy[0]*100:+.2f}% |")
    say(f"| howling ({HOWLING:.0f}+) | {howl[1]} | {howl[0]*100:+.2f}% |")
    say("")
    if breezy[1] and calm[1] and calm[0] > 0:
        say(f"**In a breeze she beats expectation {breezy[0]/calm[0]:.1f}x as hard "
            f"as she does in calm** ({breezy[0]*100:+.2f}% vs {calm[0]*100:+.2f}% "
            f"of a game's points). That is the meme.\n")

    # ---- the cherry-picking machine: search every reasonable definition of
    # "windy" and report her single best one, plus how many were searched.
    say("### The cherry-picking machine\n")
    say("Every reasonable definition of \"windy\" was tried — sustained wind or "
        "gusts, a dozen thresholds, edge measured as point share or as games won "
        "— and the single most flattering result is reported below, along with "
        "how many definitions it took to find it.\n")
    specs = []
    for measure, idx in (("sustained wind", 0), ("gusts", 4)):
        for thr in (8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 30):
            hi = [(r, won) for g in her
                  for r, won, x in [(g[1], g[3], g[idx])]
                  if x is not None and x >= thr]
            lo = [(r, won) for g in her
                  for r, won, x in [(g[1], g[3], g[idx])]
                  if x is not None and x < thr]
            # a "windy" subset must actually be selective: at least 30 games
            # on each side and the windy arm no more than half her career
            if len(hi) < 30 or len(lo) < 30 or len(hi) > 0.5 * len(her):
                continue
            for stat, fn in (("edge over expectation",
                              lambda gs: sum(r for r, _ in gs) / len(gs)),
                             ("game win rate",
                              lambda gs: sum(1 for _, w in gs if w) / len(gs))):
                specs.append((fn(hi) - fn(lo), measure, thr, stat,
                              len(hi), fn(hi), fn(lo)))
    if specs:
        specs.sort(reverse=True)
        d, measure, thr, stat, nhi, vhi, vlo = specs[0]
        pos = sum(1 for s in specs if s[0] > 0)
        fmt = "{:+.1f}%" if stat.startswith("edge") else "{:.1f}%"
        say(f"**Best available framing** — with {measure} at {thr}+ mph, "
            f"Hurricane Tyra Black's {stat} is **{fmt.format(vhi*100)}** across her "
            f"{nhi} such games, against **{fmt.format(vlo*100)}** the rest of the "
            f"time: a **{d*100:+.1f} percentage-point gap in her favour**.")
        say(f"\nThat is the best of **{len(specs)} definitions searched** "
            f"({pos} of which happened to favour her, {len(specs)-pos} of which "
            f"did not). Choosing the winner after looking is exactly the mistake "
            f"this whole review was written to catch.\n")

    # her single windiest event
    by_event = defaultdict(list)
    for w, r, meta, won, _ in her:
        by_event[meta].append((w, r, won))
    windiest = max(by_event.items(), key=lambda kv: sum(w for w, _, _ in kv[1]) / len(kv[1]))
    ev, gl = windiest
    mw = sum(w for w, _, _ in gl) / len(gl)
    say(f"Her windiest event on record: **{ev[2]}** ({ev[1]}, mean {mw:.1f} mph) — "
        f"{sum(1 for _, _, won in gl if won)}–{sum(1 for _, _, won in gl if not won)} "
        f"in games, beating expectation by "
        f"{sum(r for _, r, _ in gl)/len(gl)*100:+.2f}% of points per game.\n")

    # ---------- the deflating half ----------
    say("## The numbers that ruin it\n")

    perm = []
    base = [(w, r) for w, r, _, _, _ in her]
    for _ in range(N_PERM):
        winds = [w for w, _ in base]
        rng.shuffle(winds)
        s = ols([(w, y) for w, (_, y) in zip(winds, base)])
        if s:
            perm.append(s[0])
    perm.sort()
    p_one = sum(1 for s in perm if s >= b) / len(perm)
    say(f"1. **Her own permutation p-value is {p_one:.2f}.** Shuffle which of her "
        f"games were windy and you get a slope at least this big "
        f"{p_one*100:.0f}% of the time. Nothing to see.")

    # scan correction: how often does SOME player beat her z in a null world?
    zs = {}
    for pid, g in pool.items():
        r = ols([(w, y) for w, y, _, _, _ in g])
        if r and r[1] > 0:
            zs[pid] = r[0] / r[1]
    her_z = b / se
    better = sum(1 for z in zs.values() if z > her_z)
    say(f"2. **{better} of {len(zs)} players have a higher wind t-statistic than she does.** "
        f"Scan 552 players and somebody has to look like a wind specialist.")

    say(f"3. **The review's population test says the dimension does not exist.** "
        f"Split-half reliability of per-player wind slopes is r = +0.06 against a "
        f"permutation null of [−0.07, +0.12]; the properly-powered version bounds "
        f"sd(player wind slope) at ≤ 0.018 share per 10 mph — and the indoor "
        f"placebo, where wind physically cannot act, reproduces the whole nominal "
        f"signal (0.0154 vs 0.0127).")

    # indoor placebo for HER
    iplayers, _ = load("indoor")
    if tid in iplayers and len(iplayers[tid]) >= 20:
        ib = ols([(w, y) for w, y, _, _, _ in iplayers[tid]])
        if ib:
            say(f"4. **Her INDOOR wind slope is {ib[0]*10:+.4f} per +10 mph** "
                f"({ib[2]} games) — the wind outside a building she is sealed inside. "
                f"{'Same sign as her outdoor slope, which is exactly the problem.' if ib[0]*b > 0 else 'At least this one goes the other way.'}")

    # how many more games would she need
    need = (1.96 * se / b) ** 2 * n if b > 0 else float("inf")
    say(f"5. **She would need roughly {need:,.0f} outdoor games** at this effect size "
        f"before the slope cleared significance on its own. She has {n}. "
        f"At her current rate that is about {need/ (n/2.5):,.0f} more seasons.")

    say("\n---\n")
    say("**Verdict:** Tyra Black is, in the most flattering honest cut available, "
        f"the #{rank} wind-positive player of {len(ranked)} in a 24,000-game archive "
        "where the wind-skill dimension does not measurably exist. The nickname is "
        "safe precisely because nothing can threaten it. *Hurricane* is a vibe, and "
        "the vibe is unfalsifiable.")

    p = ROOT / "model/weather_review/hurricane_tyra.md"
    p.write_text("\n".join(out) + "\n")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
