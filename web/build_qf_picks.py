"""Render the 2026 MLP quarterfinal bracket as an unlisted insights page.

    python web/build_qf_picks.py        # -> web/insights/qf-picks/index.html

Same card layout as the forecast page (build_site.build_forecast), but the
matchups are the projected selection-show outcome rather than a published
schedule, and the Columbus--Brooklyn card is doubled to show Brooklyn with
and without Rachel Rohrabacher in women's doubles.

Self-contained: the site stylesheet is inlined and nav links are absolute,
so the file renders correctly both under site/insights/ (copied verbatim by
build_site) and standalone.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_forecast as mf                                   # noqa: E402
from sitelib import style                                    # noqa: E402
from sitelib.charts import esc                               # noqa: E402
from sitelib.race import GAMMA                               # noqa: E402

OUT = ROOT / "web" / "insights" / "qf-picks" / "index.html"
SITE = "https://chad-murphy-data.github.io/pickleball/"
EPS = mf.CAL["eps"]
GENERATED = "2026-08-09"

vals, singles = mf.load_values(), mf.load_singles()
NID = {v[0]: k for k, v in vals.items()}

# --- the confirmed six-player rosters -----------------------------------
STARTERS: dict[str, list[str]] = {}
for r in csv.DictReader((ROOT / "data" / "mlp_rosters_2026.csv").open()):
    if r["role"] == "starter":
        STARTERS.setdefault(r["team"], []).append(r["player_id"])

SEED = {"New Jersey 5s": 1, "St. Louis Shock": 2, "Los Angeles Mad Drops": 3,
        "Columbus Sliders": 4, "Brooklyn Pickleball Team": 5, "Dallas Flash": 6,
        "Palm Beach Royals": 7, "Texas Ranchers": 8}
# the projected selection show: #1 takes Palm Beach, everyone after takes the
# lowest available seed
BRACKET = [("New Jersey 5s", "Palm Beach Royals"),
           ("St. Louis Shock", "Texas Ranchers"),
           ("Los Angeles Mad Drops", "Dallas Flash"),
           ("Columbus Sliders", "Brooklyn Pickleball Team")]

mixv = lambda p: (1 - EPS) * p + EPS / 2
last = lambda u: vals[u][0].split()[-1]


def pct(p):
    """Percent that never reads as 0% or 100% (house rule)."""
    if p < 0.005:
        return "&lt;1%"
    if p > 0.995:
        return "&gt;99%"
    return f"{round(p * 100)}%"


def pct1(p):
    return f"{min(max(p * 100, 0.5), 99.5):.1f}%"


def lineup_of(team, sub=None):
    """Best legal lineup from the team's four starters.

    sub = (slot, out, in) swaps ONE player in ONE game -- an MLP woman plays
    women's doubles plus one mixed, so resting her from women's doubles
    leaves her mixed game untouched, and she is still available for the
    DreamBreaker. Removing her from the pool instead would sit her twice."""
    lu = mf.best_lineup(set(STARTERS[team]), vals)
    if sub:
        slot, out, into = sub
        lu = dict(lu)
        lu[slot] = [into if u == out else u for u in lu[slot]]
    return lu


def price(la, lb, r1, r2):
    games = {s: mf.price_game(la[s], lb[s], vals) for s in mf.SLOTS}
    p_db = mf.db_win_prob(r1, r2, vals, singles)
    t = mf.matchup_tree([games[s]["p"] for s in mf.SLOTS], p_db)
    raw = t["p_40"] + t["p_31"] + t["p_db"] * p_db
    series = raw * raw * (3 - 2 * raw)
    return games, t, p_db, mixv(raw), mixv(series)


SLOT_LABEL = {"WD": "WD", "MD": "MD", "MXD1": "MX1", "MXD2": "MX2"}


def card(t1, t2, sub2=None, tag="", note=""):
    la, lb = lineup_of(t1), lineup_of(t2, sub2)
    r1 = set(STARTERS[t1])
    r2 = set(STARTERS[t2])          # DreamBreaker four: a rested starter is
    games, tree, p_db, p, s = price(la, lb, r1, r2)   # still on the team
    rows = "".join(
        f'<tr><td>{SLOT_LABEL[sl]}</td>'
        f'<td>{esc(" / ".join(last(u) for u in la[sl]))}</td>'
        f'<td>{esc(" / ".join(last(u) for u in lb[sl]))}</td>'
        f'<td class="num"><strong>{pct(games[sl]["p"])}</strong></td>'
        f'<td class="num">{games[sl]["modal"]}</td></tr>' for sl in mf.SLOTS)
    head = (
        f'<div class="big">{esc(t1)} {pct1(p)}'
        f'<span class="gray"> — {pct1(1 - p)} {esc(t2)}</span></div>'
        f'<div class="pmbar"><div class="a" style="width:{p * 100:.1f}%"></div>'
        f'<div class="b" style="flex:1"></div></div>'
        f'<p class="note"><strong>best-of-three series: {pct1(s)}</strong> · '
        f'single matchup {pct1(p)} · paths: 4–0 {pct(tree["p_40"])} · '
        f'3–1 {pct(tree["p_31"])} · DreamBreaker {pct(tree["p_db"])} '
        f'({esc(t1.split()[-1])} {pct(p_db)} if played) · '
        f'1–3 {pct(tree["p_13"])} · 0–4 {pct(tree["p_04"])}</p>')
    tbl = (f'<div class="tblwrap"><table><tr><th>game</th>'
           f'<th>#{SEED[t1]} {esc(t1)}</th><th>#{SEED[t2]} {esc(t2)}</th>'
           f'<th class="num">P({esc(t1.split()[-1])})</th>'
           f'<th class="num">modal</th></tr>{rows}</table></div>')
    return (f'<div class="card">{tag}{head}{tbl}'
            f'{f"<p class=note>{note}</p>" if note else ""}</div>'), p, s


ROHR, BLATT = NID["Rachel Rohrabacher"], NID["Hannah Blatt"]

cards = []
for t1, t2 in BRACKET:
    tag = (f'<p class="note" style="margin-top:0">quarterfinal · '
           f'#{SEED[t1]} picks #{SEED[t2]}</p>')
    html, p, s = card(t1, t2, tag=tag)
    cards.append(html)

# the doubled Brooklyn card
CBU, BKN = "Columbus Sliders", "Brooklyn Pickleball Team"
full_html, full_p, full_s = card(
    CBU, BKN,
    tag='<p class="note" style="margin-top:0"><strong>A — Rohrabacher plays '
        'women\'s doubles</strong> (Brooklyn at full strength)</p>',
    note="This is the lineup Brooklyn fielded on 8 August against SoCal.")
sub_html, sub_p, sub_s = card(
    CBU, BKN, sub2=("WD", ROHR, BLATT),
    tag='<p class="note" style="margin-top:0"><strong>B — Blatt plays women\'s '
        'doubles</strong> (Rohrabacher rested; Brooklyn\'s pattern in 12 of '
        'its last 13 matchups)</p>',
    note="Blatt is Brooklyn's only bench woman, so any game Rohrabacher sits, "
         "she must take. Rohrabacher keeps her mixed game in this scenario — "
         "only women's doubles changes.")

swing_p, swing_s = (sub_p - full_p) * 100, (sub_s - full_s) * 100

body = f"""
<h1>The quarterfinal bracket, priced</h1>
<p class="sub">MLP's selection show runs Newport Beach, 14–16 August: the
No.&nbsp;1 seed picks its opponent from the reseeded 5–8 pool, then No.&nbsp;2,
then No.&nbsp;3, and No.&nbsp;4 takes whoever is left. Each quarterfinal is a
<strong>best-of-three matchup series</strong>. This page prices the bracket
that follows if the seeds pick the way the model says they should —
New&nbsp;Jersey takes <strong>Palm Beach</strong> (the weakest men's pair in
the field, against the one soft slot New Jersey has), and everyone after takes
the lowest available seed. Lineups are each team's best legal four from its
confirmed six-player roster, as actually fielded in Dallas: Texas without
<strong>Layne Sleeth</strong> (shoulder, out since 20 June) and Palm Beach
without <strong>Tyson McGuffin</strong> (sat out the whole Dallas weekend).
Per-game probabilities use current v2 values, the weakest-link penalty and
display calibration; the DreamBreaker is rated by the singles model.
This page is a living view, not a commitment — nothing here is in the
<a href="{SITE}receipts.html">receipts ledger</a>.</p>

<h2>If the seeds pick to the model</h2>
{"".join(cards)}

<h2>If Brooklyn sits Rohrabacher</h2>
<p class="sub">Hannah Blatt has started women's doubles in 12 of Brooklyn's
last 13 matchups, Rohrabacher dropping to a single mixed game. It is a
0.93-logit hole in one of four games, and the record agrees with the model:
Brooklyn's 2026 MLP games with Blatt on court go 36.7% won at −2.10 average
margin (n=30); with Rohrabacher, 77.1% and +3.63 (n=35). Same matchup, both
ways — Columbus's numbers are on the left of each bar.</p>
{full_html}
{sub_html}
<div class="card"><div class="big">Swing {swing_p:+.1f} pts
<span class="gray"> single matchup · {swing_s:+.1f} pts series</span></div>
<p class="note">Sitting Rohrabacher for one game turns Columbus from a
<strong>{pct1(full_s)}</strong> underdog into a <strong>{pct1(sub_s)}</strong>
favourite over the series. Brooklyn is still the last team any of the four
top seeds would choose — but the No.&nbsp;4 seed being stuck with them is a
coin flip, not a death sentence.</p></div>

<h2>Reading it</h2>
<p class="note">Probabilities never read as 0% or 100%: about 1% of ≥99%
favourites lose (44 of 4,248 games on record), and the calibration layer
encodes that as a floor. The single-matchup numbers were checked against all
286 completed 2026 MLP matchups re-priced from actual lineups — the 95%+
bucket predicted 97.9% and observed 98.3% (n=60), with a recalibration slope
of 1.19 [0.92, 1.68]. Series numbers assume matchups within a series are
independent; if a team simply has a bad weekend that correlation makes
favourites less safe than shown, though it cannot change an ordering.
Full write-up and the pick-preference lists for all four seeds are in
<code>model/mlp_playoff_picks_2026.md</code>.</p>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex, nofollow">
<title>The quarterfinal bracket, priced — PICKLES</title>
{style.THEME_HEAD}
{style.FONTS_PRECONNECT}
<style>{style.CSS}</style>
<link rel="icon" href="{style.FAVICON}">
<header class="site"><div class="wrap">
  <span class="brand"><a href="{SITE}index.html">PICKLES</a></span>
  <nav>{"".join(f'<a href="{SITE}{h}">{t}</a>' for h, t in style.NAV)}</nav>
  <button class="themetog" type="button" title="toggle light/dark">◐</button>
</div></header>
<div class="wrap">
{body}
<footer class="site">Unofficial fan analytics based on public results data —
not affiliated with any tour. Every number that has an error bar shows it;
cross-gender rankings are never published as fact
(<a href="{SITE}methods.html">why</a>). Model: Bayesian, 36k games, validated
77.4% winner accuracy on 884 unseen games · generated {GENERATED}.</footer>
</div>
{style.THEME_TOGGLE_JS}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print(f"wrote {OUT.relative_to(ROOT)}  ({len(html):,} bytes)")
for (t1, t2) in BRACKET:
    _, p, s = card(t1, t2)
    print(f"  #{SEED[t1]} {t1:<24} v #{SEED[t2]} {t2:<26} "
          f"{pct1(p):>6} matchup  {pct1(s):>6} series")
print(f"  Brooklyn double: Columbus {pct1(full_s)} -> {pct1(sub_s)} series "
      f"({swing_s:+.1f} pts)")
