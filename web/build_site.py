"""Generate the static website from the model CSVs.

    python web/build_site.py          # writes site/
    python -m http.server -d site     # preview

Everything regenerates from data/*.csv + model/receipts.json; no backend,
no network.  Output (site/) is gitignored — it rebuilds in ~seconds on a
nightly pipeline run.  Pillars covered (ROADMAP Phase 2): power rankings,
player pages, matchup simulator, receipts ledger, plus methods / record
book / DUPR-vs-model pages.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib import charts, data as D, livepage, style
from sitelib.charts import esc
from sitelib.race import (GAMMA, calibrate, race_dist, set_calibration,
                          sigmoid, value_points)

CAL = json.loads((Path(__file__).resolve().parent / "calibration.json").read_text())
set_calibration(CAL["a"], CAL["b"], CAL["eps"])

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REPO = "https://github.com/chad-murphy-data/pickleball"


# ---------------------------------------------------------------- helpers

def write(path, html):
    p = SITE / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")


def plink(p, root=""):
    return f'<a href="{root}players/{p.pid}.html">{esc(p.name)}</a>' if p.dynamic \
        else esc(p.name)


def trend_arrow(p):
    if p.form_delta is None:
        return '<span class="flat" title="not enough history">·</span>'
    dpts = p.pts - value_points(p.value - p.form_delta)
    tip = f"6-month form change: {dpts:+.2f} pts"
    if dpts >= 0.7:
        return f'<span class="up" title="{tip}">▲▲</span>'
    if dpts >= 0.25:
        return f'<span class="up" title="{tip}">▲</span>'
    if dpts <= -0.7:
        return f'<span class="down" title="{tip}">▼▼</span>'
    if dpts <= -0.25:
        return f'<span class="down" title="{tip}">▼</span>'
    return f'<span class="flat" title="{tip}">—</span>'


def wl(w, l):
    return f'{w}–{l}'


def pct(x, dec=0):
    return f"{100 * x:.{dec}f}%"


def chem_lookup(chem, name_a, name_b):
    return chem.get(frozenset((name_a, name_b)))


def chem_pts(logit):
    return race_dist(round(sigmoid(logit), 4))["exp_margin"]


def dupr_cell(p):
    if p.dupr:
        asof = f' title="synced from the player&#39;s latest match ({p.dupr_asof})"' \
            if p.dupr_asof else ""
        return f'<span{asof}>{p.dupr:.2f}</span>'
    if p.dupr_glitch:
        return (f'<span class="gray" title="platform currently shows a ~3.5 reset '
                f'artifact; last credible value {p.dupr_glitch:.2f} — excluded">—⚠</span>')
    return '<span class="gray">—</span>'


# ---------------------------------------------------------------- rankings

def rank_probs(pool, n_draws=100_000, top=12):
    """P(true #1) and P(true top-3) by Monte Carlo over posterior marginals.
    Approximation: posterior correlations between players are ignored (the
    joint draws file is regenerated only at refit time); good enough for a
    who's-really-first panel, and labeled as such on the page."""
    import random
    rng = random.Random(20260713)          # deterministic build
    pool = sorted(pool, key=lambda p: -p.value)[:top]
    wins = [0] * len(pool)
    top3 = [0] * len(pool)
    for _ in range(n_draws):
        draws = [rng.gauss(p.value, p.sd) for p in pool]
        order = sorted(range(len(pool)), key=lambda i: -draws[i])
        wins[order[0]] += 1
        for i in order[:3]:
            top3[i] += 1
    return [(p, wins[i] / n_draws, top3[i] / n_draws) for i, p in enumerate(pool)]


NUMWORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def lead_growth(pool):
    """(gap_then, gap_now, since_year): the #1-vs-#2 gap in display pts at
    the first and last months both trajectories cover.  None when either
    trajectory is missing."""
    if len(pool) < 2 or not pool[0].traj or not pool[1].traj:
        return None
    one, two = pool[0], pool[1]

    def at(pl, m):
        for mo, v, _ in reversed(pl.traj):
            if mo <= m:
                return v
        return pl.traj[0][1]

    m0 = max(one.traj[0][0], two.traj[0][0])
    m1 = min(one.traj[-1][0], two.traj[-1][0])
    return (value_points(at(one, m0)) - value_points(at(two, m0)),
            value_points(at(one, m1)) - value_points(at(two, m1)), m0[:4])


def second_seat(pool):
    """(runner_up, P(truly better than #3), margin_pts) when the #2 seat is
    clearly claimed (≥85% by posterior marginals); else None."""
    if len(pool) < 3:
        return None
    two, three = pool[1], pool[2]
    z = (two.value - three.value) / math.hypot(two.sd, three.sd)
    p23 = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return (two, p23, two.pts - three.pts) if p23 >= 0.85 else None


def p1_verdict(probs, label, pool):
    """One-line verdict over the P(#1) rows, in the house voice.  probs is
    sorted by P(#1) descending; label is "men"/"women"; pool is the active
    list sorted by rank, for the career-high / runner-up receipts."""
    p, lead, _ = probs[0]
    if lead > 0.995:
        head = f"It's {esc(p.name)}. It is not close"
        career = p.traj and p.value >= max(v for _, v, _ in p.traj) - 1e-9
        gr = lead_growth(pool)
        if career and gr and gr[1] - gr[0] >= 0.25:
            head += (f" — a career-best rating, and the lead over #2 has grown "
                     f"from {gr[0]:+.1f} to {gr[1]:+.1f} pts since {gr[2]}")
        elif career:
            head += " — and the rating is a career best"
        head += "."
        sec = second_seat(pool)
        if sec:
            two, p23, margin = sec
            head += (f" Second is settled too: {esc(two.name)}, {margin:+.1f} pts "
                     f"clear of the chase pack ({pct_floor(p23)} sure). If the "
                     f"crown gets a rival, it starts with "
                     f"{esc(two.name.split()[-1])}.")
        return head
    if lead >= 0.5:
        return (f"{esc(p.name)} is the favorite at {pct_floor(lead)} — "
                f"with live challengers behind.")
    n = sum(1 for _, p1, _ in probs if p1 >= 0.05)
    word = NUMWORDS.get(n, str(n))
    return (f"Nobody clears {pct_floor(lead)}. The {label}'s crown is a "
            f"{word}-way pile-up.")


def p1_bar_width(p1):
    """Bar width that never renders empty or overfull (house rule cousin)."""
    w = 100 * p1
    return "99.5" if w > 99 else ("0.5" if w < 1 else f"{w:.0f}")


def p1_panel(probs, label, pool):
    head_note = ("posterior probability of being the true best — not the "
                 "point-estimate leader.")
    rows = []
    for p, p1, p3 in probs:
        rows.append(
            f'<div class="p1row"><span class="p1name">{plink(p)}</span>'
            f'<span class="p1bar"><span class="p1fill" style="width:{p1_bar_width(p1)}%">'
            f'</span></span><span class="p1pct">{pct_floor(p1)}</span>'
            f'<span class="p1meta">top-3 {pct_floor(p3)} · {p.games} g</span></div>')
    return (f'<div class="card p1card"><div class="p1head"><strong>Who is actually '
            f'#1?</strong> <span class="note">{head_note}</span></div>\n'
            f'<p class="p1kick">{p1_verdict(probs, label, pool)}</p>\n'
            + "\n".join(rows) +
            '\n<p class="note" style="margin:10px 0 2px">Monte Carlo over posterior '
            'marginals (correlations ignored); wide-interval players earn real '
            'probability — that is the point.</p></div>')


# Rankings gender tabs: the page ships with the women's panel visible
# (class "on"), so it is correct even before/without JS.  TABS_BOOT runs
# during parse and flags JS-capable browsers, which collapses the page to
# one panel at a time; without JS both panels stay stacked (women first).
# TABS_JS wires clicks and deep links (#men / #women) — replaceState so
# tab flips don't pile up in browser history.  Panel ids carry a "sec-"
# prefix so the tab hashes never name a real fragment target: the browser
# has nothing to anchor-scroll to, and deep links land at the top of the
# page with the right tab selected instead of jumped mid-page.
TABS_BOOT = '<script>document.documentElement.classList.add("tabbed")</script>'

TABS_JS = """
<script>
const tabs = [...document.querySelectorAll(".gtab")];
const secs = [...document.querySelectorAll(".gsec")];
const bar = document.querySelector(".gtabs");
function show(id, setHash) {
  if (!secs.some(s => s.id === "sec-" + id)) id = "women";
  for (const s of secs) s.classList.toggle("on", s.id === "sec-" + id);
  for (const t of tabs) {
    const cur = t.getAttribute("href") === "#" + id;
    t.classList.toggle("on", cur);
    t.setAttribute("aria-current", cur ? "true" : "false");
  }
  if (setHash) history.replaceState(null, "", "#" + id);
}
for (const t of tabs)
  t.addEventListener("click", e => {
    e.preventDefault();
    show(t.getAttribute("href").slice(1), true);
    // Fresh panel from the top: snap so the (sticky) tab bar sits at the
    // viewport top.  Measured off the panel, not the bar — a stuck bar's
    // offsetTop reports its pinned, not natural, position.
    const y = document.querySelector(".gsec.on").offsetTop - bar.offsetHeight - 14;
    if (scrollY > y) scrollTo({top: y});
  });
addEventListener("hashchange", () => show(location.hash.slice(1) || "women", false));
show(location.hash.slice(1) || "women", false);
</script>"""


def build_rankings(players, updated, n_games, val):
    def table(pool):
        if not pool:
            return "<p>none</p>"
        xmin = min(p.pts_lo for p in pool) - 0.3
        xmax = max(p.pts_hi for p in pool) + 0.3
        rows = []
        for p in pool:
            st = p.stats
            rec = wl(st.w, st.l) if st else "—"
            dupr = dupr_cell(p)
            last = p.last_date[:7] if p.last_date else "—"
            rows.append(
                f'<tr><td class="num">{p.rank or "—"}</td><td>{plink(p)}</td>'
                f'<td class="num"><strong>{p.pts:+.1f}</strong></td>'
                f'<td>{charts.interval_cell(p.pts_lo, p.pts, p.pts_hi, xmin, xmax)}</td>'
                f'<td>{trend_arrow(p)}</td>'
                f'<td class="num">{p.games}</td><td class="num">{rec}</td>'
                f'<td class="num">{dupr}</td><td class="num gray">{last}</td></tr>')
        return ('<div class="tblwrap"><table><tr><th class="num">#</th><th>player</th>'
                '<th class="num">pts</th><th>90% interval (pts vs avg pairing)</th>'
                '<th title="6-month form change">form</th><th class="num">games</th>'
                '<th class="num">W–L</th><th class="num">DUPR</th>'
                '<th class="num">last seen</th></tr>'
                + "".join(rows) + "</table></div>")

    n_dyn = sum(1 for p in players.values() if p.dynamic)
    tabs, panels = [], []
    for sec, gender, label, low in (("A", "F", "Women", "women"),
                                    ("B", "M", "Men", "men")):
        pool = [p for p in players.values() if p.dynamic and p.gender == gender]
        active = sorted((p for p in pool if D.is_active(p)),
                        key=lambda p: p.rank)[:75]
        inactive = sorted((p for p in pool if not D.is_active(p)),
                          key=lambda p: -p.value)
        probs = sorted(((p, p1, p3) for p, p1, p3 in rank_probs(active)
                        if p1 >= 0.01), key=lambda t: -t[1])
        inner = (f'<h2><span class="secno">SEC. {sec}</span>{label}</h2>'
                 + p1_panel(probs, low, active) + table(active))
        if inactive:
            inner += (
                f'<details><summary class="note">{len(inactive)} rated players '
                f'without a 2026 game (hidden, unranked)</summary>{table(inactive)}</details>')
        default = sec == "A"
        tabs.append(f'<a class="gtab{" on" if default else ""}" href="#{low}"'
                    + (' aria-current="true"' if default else "")
                    + f'>{label}</a>')
        panels.append(
            f'<section class="gsec{" on" if default else ""}" id="sec-{low}">{inner}</section>')

    dupr_acc = val["dupr_reference"]["accuracy"]
    body = f"""
<h1 class="runtitle">Power rankings</h1>
<div class="runmeta">RUN {updated} :: DYNAMIC BAYESIAN POSTERIOR :: MLP + PPA DOUBLES 2024–26</div>
<p class="sub">Every player with ≥60 pro doubles games since 2024, rated by a
dynamic Bayesian model of all {n_games:,} games. <strong>pts</strong> = expected
margin (with an average partner, vs an average pair, race to 11) — median
regular ≈ +2, star ≈ +5. The interval is the point, not fine print.</p>
<div class="syscheck">
 <div class="lrow"><span class="lk">GAMES IN THE MODEL (2024–26, MLP + PPA)</span><span class="ldot"></span><span class="lv">{n_games:,}</span></div>
 <div class="lrow"><span class="lk">WINNER ACCURACY · {val['n_games']} UNSEEN GAMES</span><span class="ldot"></span><span class="lv">{pct(val['accuracy'], 1)} <span class="lcmp">vs DUPR {pct(dupr_acc, 1)}</span></span></div>
 <div class="lrow"><span class="lk">PLAYERS WITH MONTHLY SKILL TRACKING</span><span class="ldot"></span><span class="lv">{n_dyn}</span></div>
 <div class="lrow"><span class="lk">DATA THROUGH</span><span class="ldot"></span><span class="lv">{updated}</span></div>
 <div class="lrow"><span class="lk">PREDICTIONS COMMITTED PRE-MATCH</span><span class="ldot"></span><span class="lv"><a href="receipts.html">[OK] → receipts</a></span></div>
</div>
<div class="houserule"><span class="hrtag">HOUSE RULE</span>Women's and men's lists are separate on purpose: the tours never
play cross-gender games, so no data links the two scales
(<a href="methods.html">methods</a>). Rankings use current-month posterior
values; ▲/▼ = 6-month form change beyond ±0.25 pts.</div>
{TABS_BOOT}
<nav class="gtabs" aria-label="Choose a rankings list">{''.join(tabs)}</nav>
{''.join(panels)}
""" + TABS_JS
    write("rankings.html", style.page("Power rankings — PICKLES",
                                      body, "rankings.html", "", updated))


# ---------------------------------------------------------------- players

def build_player_index(players, updated):
    pool = sorted((p for p in players.values() if p.dynamic),
                  key=lambda p: p.name.split()[-1])
    rows = "".join(
        f'<tr class="prow" data-name="{esc(p.name.lower())}">'
        f'<td>{plink(p, "../")}</td><td>{p.gender}</td>'
        f'<td class="num">{p.pts:+.1f}</td><td class="num">{f"#{p.rank}" if p.rank else "—"}</td>'
        f'<td class="num">{p.games}</td>'
        f'<td class="num gray">{(p.last_date or "—")[:7]}</td></tr>'
        for p in pool)
    body = f"""
<h1>Players</h1>
<p class="sub">All {len(pool)} players with monthly skill tracking (≥60 games
since 2024). Rank is within gender.</p>
<input class="searchbox" id="q" type="search" placeholder="Filter by name…"
 oninput="for(const r of document.querySelectorAll('.prow'))
 r.style.display=r.dataset.name.includes(this.value.toLowerCase())?'':'none'">
<div class="tblwrap"><table><tr><th>player</th><th>g</th><th class="num">pts</th>
<th class="num">rank</th><th class="num">games</th><th class="num">last seen</th></tr>
{rows}</table></div>
"""
    write("players/index.html",
          style.page("Players — PICKLES", body, "players/index.html",
                     "../", updated))


def build_player_page(p, players, chem, updated):
    st = p.stats
    root = "../"
    # enrich log entries with display names for chart tooltips
    for e in st.log:
        pr = players.get(e["partner"])
        e["partner_name"] = pr.name if pr else "?"
        e["opp_names"] = "/".join(
            (players[o].name.split()[-1] if o in players else "?") for o in e["opp"])

    share = st.pf / max(st.pf + st.pa, 1)
    if p.dupr:
        dupr_txt, dupr_k = f"{p.dupr:.2f}", \
            f"official DUPR (own scale{', as of ' + p.dupr_asof if p.dupr_asof else ''})"
    elif p.dupr_glitch:
        dupr_txt = "—"
        dupr_k = (f"official DUPR — the platform currently shows a ~3.5 reset "
                  f"artifact (last credible value {p.dupr_glitch:.2f}); excluded "
                  f"from comparisons")
    else:
        dupr_txt, dupr_k = "—", "official DUPR (none synced)"
    tiles = f"""
<div class="tiles">
 <div class="tile"><div class="v">{p.pts:+.1f} <span class="note">({p.pts_lo:+.1f}…{p.pts_hi:+.1f})</span></div>
   <div class="k">pts vs avg pairing (90% interval)</div></div>
 <div class="tile"><div class="v">{f"#{p.rank}" if p.rank else "—"}</div><div class="k">of {sum(1 for q in players.values() if q.dynamic and q.gender == p.gender and q.rank)} active {'men' if p.gender == 'M' else 'women'} (within gender only{'' if p.rank else '; no 2026 games'})</div></div>
 <div class="tile"><div class="v">{wl(st.w, st.l)}</div><div class="k">career record · {pct(st.w / max(st.w + st.l, 1))} wins, {pct(share)} of points</div></div>
 <div class="tile"><div class="v">{trend_arrow(p)}</div><div class="k">6-month form</div></div>
 <div class="tile"><div class="v">{dupr_txt}</div><div class="k">{dupr_k}</div></div>
</div>"""

    traj_html = charts.trajectory_chart(p.traj)
    spark = charts.dupr_spark(p.dupr_hist)
    glog = charts.gamelog_chart(st.log)

    years = sorted({k[0] for k in st.by_year_tour})
    split_rows = []
    for y in years:
        for tour in ("MLP", "PPA"):
            v = st.by_year_tour.get((y, tour))
            if not v:
                continue
            w_, l_, pf, pa = v
            split_rows.append(
                f'<tr><td>{y}</td><td>{tour}</td><td class="num">{wl(w_, l_)}</td>'
                f'<td class="num">{pct(w_ / max(w_ + l_, 1))}</td>'
                f'<td class="num">{pf / max(w_ + l_, 1):.1f}–{pa / max(w_ + l_, 1):.1f}</td></tr>')
    ctx_bits = []
    for c in ("mens", "womens", "mixed"):
        v = st.by_context.get(c)
        if v and sum(v):
            ctx_bits.append(f"{c} {wl(v[0], v[1])}")
    dec = st.deciding; ot = st.overtime
    clutch = f"""
<div class="tiles">
 <div class="tile"><div class="v">{wl(dec[0], dec[1]) if sum(dec) else '—'}</div><div class="k">deciding games (PPA winner-take-all)</div></div>
 <div class="tile"><div class="v">{wl(ot[0], ot[1]) if sum(ot) else '—'}</div><div class="k">overtime games (past the target)</div></div>
 <div class="tile"><div class="v">{st.blowout_w}</div><div class="k">blowout wins (opponents ≤5)</div></div>
 <div class="tile"><div class="v">{st.best_streak}</div><div class="k">longest win streak</div></div>
 <div class="tile"><div class="v">{st.cur_streak:+d}</div><div class="k">current streak (+W / −L)</div></div>
</div>"""

    top_partners = sorted(st.partners.items(), key=lambda kv: -kv[1][0])[:12]
    prows = []
    for pid2, (g, w_, pf, pa) in top_partners:
        q = players.get(pid2)
        nm = plink(q, root) if q else "?"
        ch = chem_lookup(chem, p.name, q.name) if q else None
        if ch is None:
            chtxt = '<span class="gray">—</span>'
        else:
            cp = chem_pts(ch[0])
            sd_pts = (chem_pts(ch[1]) - chem_pts(-ch[1])) / 2
            sig = "" if abs(ch[0]) > 2 * ch[1] else ' <span class="gray">(noise)</span>'
            chtxt = f'{cp:+.2f} ±{sd_pts:.2f}{sig}'
        prows.append(f'<tr><td>{nm}</td><td class="num">{g}</td>'
                     f'<td class="num">{wl(w_, g - w_)}</td>'
                     f'<td class="num">{pct(pf / max(pf + pa, 1))}</td>'
                     f'<td class="num">{chtxt}</td></tr>')

    recent = []
    for e in reversed(st.log[-12:]):
        d_ = f'{e["share"] - e["exp"]:+.0%}' if e["exp"] is not None else "—"
        recent.append(
            f'<tr><td class="num gray">{e["date"]}</td><td>{e["tour"]}'
            f'{" · " + e["context"] if e["context"] else ""}</td>'
            f'<td>{esc(e["partner_name"])}</td><td>{esc(e["opp_names"])}</td>'
            f'<td class="num">{"<strong>" if e["won"] else ""}{e["score"]}'
            f'{"</strong>" if e["won"] else ""}{" OT" if e["ot"] else ""}</td>'
            f'<td class="num">{d_}</td></tr>')

    body = f"""
<h1>{esc(p.name)} <span class="gray" style="font-size:16px">{'M' if p.gender == 'M' else 'W'}</span></h1>
<p class="sub">{ ' · '.join(ctx_bits) if ctx_bits else '' }</p>
{tiles}
<h2>Skill trajectory</h2>
<p class="note">Monthly posterior value (converted to points vs an average
pairing) with a 90% credible band. Flat months = little evidence, not zero
games.</p>
{traj_html}
{f'<p class="note">Official DUPR over the same period — separate chart because the scales are not comparable:</p>{spark}' if spark else ''}
<h2>Game log vs expectation</h2>
<p class="note">Dots = share of points actually won in each game; line = what
the model expected given both pairings that month (weakest-link included;
descriptive, not a frozen forecast).</p>
{glog}
<h2>Splits</h2>
<div class="cols"><div>
<div class="tblwrap"><table><tr><th>year</th><th>tour</th><th class="num">W–L</th>
<th class="num">win%</th><th class="num">avg score</th></tr>{''.join(split_rows)}</table></div>
</div><div>
{clutch}
</div></div>
<h2>Partners</h2>
<p class="note">Chemistry = pair effect beyond the two players' values, in
points per game; almost everything here is honest noise — no pair in the
sport clears statistical significance (<a href="{root}methods.html">why</a>).</p>
<div class="tblwrap"><table><tr><th>partner</th><th class="num">games</th>
<th class="num">W–L</th><th class="num">point share</th><th class="num">chem (pts)</th></tr>
{''.join(prows)}</table></div>
<h2>Recent games</h2>
<div class="tblwrap"><table><tr><th>date</th><th>tour</th><th>partner</th>
<th>opponents</th><th class="num">score</th><th class="num">vs exp</th></tr>
{''.join(recent)}</table></div>
<p class="note"><a href="{root}simulator.html?a={p.pid}">Put {esc(p.name.split()[0])} in the simulator →</a></p>
"""
    write(f"players/{p.pid}.html",
          style.page(f"{p.name} — PICKLES", body, "", root, updated))


# ---------------------------------------------------------------- simulator

def build_simulator(players, updated):
    pool = sorted((p for p in players.values() if p.dynamic),
                  key=lambda p: -p.value)
    seen, recs = {}, []
    for p in pool:
        label = p.name if p.name not in seen else f"{p.name} [{p.pid[:4]}]"
        seen[p.name] = True
        recs.append({"id": p.pid, "n": label, "g": p.gender,
                     "v": round(p.value, 4), "s": round(p.sd, 4),
                     "pts": round(p.pts, 1), "rk": p.rank})
    pdata = json.dumps(recs, separators=(",", ":"))

    body = f"""
<h1>Matchup simulator</h1>
<p class="sub">Any four tracked players, any format. Uses current-month
posterior values, the weakest-link penalty (γ = −0.18), and the exact
race-to-target score distribution. The interval reflects real uncertainty
about the players — that's the honest part.</p>
<div class="cols">
 <div class="card"><h3>Team A</h3>
  <input list="plist" id="a1" placeholder="Player 1…" autocomplete="off">
  <input list="plist" id="a2" placeholder="Player 2…" autocomplete="off">
  <label class="small"><input type="checkbox" id="newa"> new pairing (first ~6 games together)</label>
 </div>
 <div class="card"><h3>Team B</h3>
  <input list="plist" id="b1" placeholder="Player 1…" autocomplete="off">
  <input list="plist" id="b2" placeholder="Player 2…" autocomplete="off">
  <label class="small"><input type="checkbox" id="newb"> new pairing</label>
 </div>
</div>
<p>
 <select id="fmt">
  <option value="1-11">single game to 11</option>
  <option value="3-11">best of 3 (to 11)</option>
  <option value="5-11">best of 5 (to 11)</option>
  <option value="1-15">single game to 15</option>
 </select>
 <button id="swap" type="button">⇄ swap teams</button>
 <button id="share" type="button">copy link</button><span id="copied" class="note"></span>
</p>
<datalist id="plist"></datalist>
<div id="out"></div>
<p class="note">Fine print: pair chemistry is excluded on purpose — the fitted
effects are almost all within ±0.1 pts and none is statistically certifiable.
Values come from a joint Bayesian fit of ~36k games; the win-probability
interval is the 90% posterior range from player-value uncertainty.
Probabilities are calibrated against out-of-sample games and never reach
0% or 100% — across 36k games, favorites we rate above 99% still lost about
1 time in 100, so that's the ceiling (<a href="methods.html">methods</a>).
If the two sides' gender mix differs, the number rests on a modeling
convention that no game data can test.</p>
<script>
const P = {pdata};
const GAMMA = {GAMMA}, BETA_NEW = 0.088;
const CAL = {{ a: {CAL["a"]}, b: {CAL["b"]}, eps: {CAL["eps"]} }};  // web/calibration.json
function pCal(p) {{
  p = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  const l = Math.log(p / (1 - p));
  return (1 - CAL.eps) * sig(CAL.a + CAL.b * l) + CAL.eps / 2;
}}
const byName = Object.fromEntries(P.map(p => [p.n.toLowerCase(), p]));
const byId = Object.fromEntries(P.map(p => [p.id, p]));
const dl = document.getElementById('plist');
for (const p of P) {{
  const o = document.createElement('option');
  o.value = p.n; o.label = `${{p.g}} · ${{p.pts >= 0 ? '+' : ''}}${{p.pts}} pts (#${{p.rk}})`;
  dl.appendChild(o);
}}
const sig = x => 1 / (1 + Math.exp(-x));
// house rule: no probability ever displays as 0% or 100%
const fp = p => p < 0.005 ? '&lt;1' : p > 0.995 ? '&gt;99' : (100 * p).toFixed(0);
const fp1 = p => p < 0.0005 ? '&lt;0.1' : p > 0.9995 ? '&gt;99.9' : (100 * p).toFixed(1);
function comb(n, k) {{ let r = 1; for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }}
function raceDist(p, T) {{
  p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  const q = 1 - p, win = [], lose = [];
  for (let b = 0; b <= T - 2; b++) win.push([T, b, comb(T - 1 + b, b) * p ** T * q ** b]);
  for (let a = 0; a <= T - 2; a++) lose.push([a, T, comb(T - 1 + a, a) * q ** T * p ** a]);
  const deuce = comb(2 * T - 2, T - 1) * (p * q) ** (T - 1);
  const dwin = deuce * p * p / (p * p + q * q);
  const pw = win.reduce((s, x) => s + x[2], 0) + dwin;
  const margin = win.reduce((s, x) => s + (T - x[1]) * x[2], 0)
    - lose.reduce((s, x) => s + (T - x[0]) * x[2], 0) + 2 * dwin - 2 * (deuce - dwin);
  return {{ pw, win, lose, deuce, dwin, margin }};
}}
function gWin(eta, T) {{ return raceDist(sig(eta), T).pw; }}
function gWinAvg(mu, sd, T) {{           // integrate over value uncertainty
  if (sd <= 0) return gWin(mu, T);
  let tot = 0, ws = 0;
  for (let i = 0; i <= 40; i++) {{
    const z = -4 + i * 0.2, w = Math.exp(-0.5 * z * z);
    tot += w * gWin(mu + z * sd, T); ws += w;
  }}
  return tot / ws;
}}
function matchProb(g, bo) {{
  if (bo === 1) return {{ p: g, scores: null }};
  const q = 1 - g;
  if (bo === 3) return {{ p: g * g * (3 - 2 * g),
    scores: [['2–0', g * g], ['2–1', 2 * g * g * q], ['1–2', 2 * q * q * g], ['0–2', q * q]] }};
  const p3 = g ** 3, w = g ** 3 * (10 - 15 * g + 6 * g * g);
  return {{ p: w, scores: [['3–0', p3], ['3–1', 3 * p3 * q], ['3–2', 6 * p3 * q * q],
    ['2–3', 6 * (1 - g) ** 3 * g * g], ['1–3', 3 * (1 - g) ** 3 * g], ['0–3', (1 - g) ** 3]] }};
}}
const pick = id => byName[(document.getElementById(id).value || '').toLowerCase()] || null;
function teamEta(x, y) {{ return x.v + y.v + GAMMA * Math.abs(x.v - y.v); }}
function fmtPts(v) {{ return (v >= 0 ? '+' : '') + v.toFixed(1); }}
function teamCard(x, y, isNew) {{
  const sum = x.v + y.v, pen = GAMMA * Math.abs(x.v - y.v);
  const mNo = raceDist(sig(sum), 11).margin, mWith = raceDist(sig(sum + pen), 11).margin;
  return `<div class="card"><strong>${{x.n}}</strong> (${{fmtPts(x.pts)}}) + ` +
    `<strong>${{y.n}}</strong> (${{fmtPts(y.pts)}})<br>` +
    `<span class="note">team vs avg pair: ${{fmtPts(mWith)}} pts · ` +
    `weakest-link penalty ${{(Math.abs(mWith - mNo) < 0.05 ? 0 : mWith - mNo).toFixed(1)}} pts` +
    (isNew ? ' · new-pairing bump applied' : '') + `</span></div>`;
}}
function bars(dist, T, names) {{
  const rows = [];
  for (let i = dist.win.length - 1; i >= 0; i--) rows.push([`${{T}}–${{dist.win[i][1]}}`, dist.win[i][2], 'a']);
  rows.push([`OT win`, dist.dwin, 'a'], [`OT loss`, dist.deuce - dist.dwin, 'b']);
  for (const [a, , pr] of dist.lose) rows.push([`${{a}}–${{T}}`, pr, 'b']);
  const mx = Math.max(...rows.map(r => r[1]));
  return '<h3>Score distribution (' + names + ')</h3><table>' + rows.map(([s, pr, side]) =>
    `<tr><td class="num" style="width:70px">${{s}}</td><td><div style="height:12px;border-radius:3px;` +
    `width:${{Math.max(0.5, 100 * pr / mx)}}%;background:var(--${{side === 'a' ? 's1' : 'loss'}})"></div></td>` +
    `<td class="num" style="width:60px">${{fp1(pr)}}%</td></tr>`).join('') + '</table>';
}}
function update(push) {{
  const a1 = pick('a1'), a2 = pick('a2'), b1 = pick('b1'), b2 = pick('b2');
  const out = document.getElementById('out');
  if (!(a1 && a2 && b1 && b2)) {{ out.innerHTML = '<p class="note">Pick four players.</p>'; return; }}
  const [bo, T] = document.getElementById('fmt').value.split('-').map(Number);
  const newa = document.getElementById('newa').checked, newb = document.getElementById('newb').checked;
  let mu = teamEta(a1, a2) - teamEta(b1, b2) + (newa ? BETA_NEW : 0) - (newb ? BETA_NEW : 0);
  const sd = Math.sqrt(a1.s ** 2 + a2.s ** 2 + b1.s ** 2 + b2.s ** 2);
  const g = pCal(gWinAvg(mu, sd, T));
  const gLo = pCal(gWin(mu - 1.645 * sd, T)), gHi = pCal(gWin(mu + 1.645 * sd, T));
  const m = matchProb(g, bo), mLo = matchProb(gLo, bo).p, mHi = matchProb(gHi, bo).p;
  const dist = raceDist(sig(mu), T);
  const aN = `${{a1.n.split(' ').pop()}}/${{a2.n.split(' ').pop()}}`;
  const bN = `${{b1.n.split(' ').pop()}}/${{b2.n.split(' ').pop()}}`;
  const genderMix = [a1, a2].map(p => p.g).sort().join('') !== [b1, b2].map(p => p.g).sort().join('');
  let html = `<div class="card"><div class="big">${{aN}} ${{fp(m.p)}}%` +
    ` <span class="note" style="font-size:15px">(90% range ${{fp(mLo)}}–${{fp(mHi)}}%)</span></div>` +
    `<div class="pmbar"><div class="a" style="width:${{100 * m.p}}%"></div><div class="b" style="flex:1"></div></div>` +
    `<p class="note">${{aN}} vs ${{bN}} · per-point share ${{(100 * sig(mu)).toFixed(1)}}% · ` +
    `expected margin ${{fmtPts(dist.margin)}} per game (to ${{T}})</p>` +
    (m.scores ? '<p>' + m.scores.map(([s, pr]) => `${{s}}: <strong>${{fp(pr)}}%</strong>`).join(' · ') + '</p>' : '') +
    (genderMix ? '<p class="note">⚠ The two sides have different gender mixes — this number rests on a prior convention, not data (<a href="methods.html">methods</a>).</p>' : '') +
    `</div><div class="cols">${{teamCard(a1, a2, newa)}}${{teamCard(b1, b2, newb)}}</div>` +
    bars(dist, T, aN + ' perspective, single game');
  out.innerHTML = html;
  if (push) {{
    const q = new URLSearchParams({{ a: a1.id + ',' + a2.id, b: b1.id + ',' + b2.id,
      fmt: bo + '-' + T, ...(newa ? {{ na: 1 }} : {{}}), ...(newb ? {{ nb: 1 }} : {{}}) }});
    history.replaceState(null, '', '?' + q);
  }}
}}
for (const id of ['a1', 'a2', 'b1', 'b2', 'fmt', 'newa', 'newb'])
  document.getElementById(id).addEventListener(id === 'fmt' ? 'change' : 'input', () => update(true));
document.getElementById('swap').onclick = () => {{
  const g = i => document.getElementById(i);
  [g('a1').value, g('b1').value] = [g('b1').value, g('a1').value];
  [g('a2').value, g('b2').value] = [g('b2').value, g('a2').value];
  [g('newa').checked, g('newb').checked] = [g('newb').checked, g('newa').checked];
  update(true);
}};
document.getElementById('share').onclick = async () => {{
  await navigator.clipboard.writeText(location.href);
  document.getElementById('copied').textContent = ' copied!';
  setTimeout(() => document.getElementById('copied').textContent = '', 1500);
}};
(function init() {{
  const q = new URLSearchParams(location.search);
  const setp = (el, id) => {{ if (id && byId[id]) document.getElementById(el).value = byId[id].n; }};
  const [a, b] = [(q.get('a') || '').split(','), (q.get('b') || '').split(',')];
  setp('a1', a[0]); setp('a2', a[1]); setp('b1', b[0]); setp('b2', b[1]);
  if (q.get('fmt')) document.getElementById('fmt').value = q.get('fmt');
  document.getElementById('newa').checked = q.get('na') === '1';
  document.getElementById('newb').checked = q.get('nb') === '1';
  update(false);
}})();
</script>
"""
    write("simulator.html", style.page("Matchup simulator — PICKLES",
                                       body, "simulator.html", "", updated))


# ---------------------------------------------------------------- forecasts

def pct_floor(x):
    """Percent that never reads as 0% or 100% (house rule)."""
    if x < 0.005:
        return "&lt;1%"
    if x > 0.995:
        return "&gt;99%"
    return pct(x)


def build_forecast(players, updated):
    fj = D.DATA / "forecasts.json"
    if fj.exists():
        F = json.loads(fj.read_text())
        cards, cur_date = [], None
        for f in F["forecasts"]:
            if f["date"] != cur_date:
                cur_date = f["date"]
                cards.append(f"<h2>{cur_date}</h2>")
            t = f.get("tree")
            if t:
                head = (f'<div class="big">{esc(f["team1"])} {pct_floor(t["p_win"])}'
                        f'<span class="gray"> — {pct_floor(1 - t["p_win"])} {esc(f["team2"])}</span></div>'
                        f'<div class="pmbar"><div class="a" style="width:{100 * t["p_win"]}%"></div>'
                        f'<div class="b" style="flex:1"></div></div>'
                        f'<p class="note">paths: 4–0 {pct_floor(t["p_40"])} · 3–1 {pct_floor(t["p_31"])} · '
                        f'DreamBreaker {pct_floor(t["p_db"])}'
                        + (f' ({esc(f["team1"].split()[-1])} {pct_floor(t["p_db_win"])} if played)'
                           if t.get("p_db_win") is not None else ' (rated 50/50)')
                        + f' · 1–3 {pct_floor(t["p_13"])} · 0–4 {pct_floor(t["p_04"])}</p>')
            else:
                head = (f'<div class="big">{esc(f["team1"])} <span class="gray">vs</span> '
                        f'{esc(f["team2"])}</div><p class="note">not rateable yet '
                        f'(missing recent lineups or untracked players)</p>')
            rows = []
            for g in f["games"]:
                if not g:
                    continue
                rows.append(
                    f'<tr><td>{g["slot"]}</td>'
                    f'<td>{esc(" / ".join(n.split()[-1] for n in g["t1_pair"]))}</td>'
                    f'<td>{esc(" / ".join(n.split()[-1] for n in g["t2_pair"]))}</td>'
                    f'<td class="num"><strong>{pct_floor(g["p"])}</strong></td>'
                    f'<td class="num">{g["modal"]}</td></tr>')
            tbl = (f'<div class="tblwrap"><table><tr><th>game</th>'
                   f'<th>{esc(f["team1"])}</th><th>{esc(f["team2"])}</th>'
                   f'<th class="num">P({esc(f["team1"].split()[-1])})</th>'
                   f'<th class="num">modal</th></tr>{"".join(rows)}</table></div>'
                   if rows else "")
            src = f["lineups_from"]
            cards.append(f"""<div class="card">{head}{tbl}
<p class="note">projected lineups from each team's last completed matchup
({src["team1"] or "?"} / {src["team2"] or "?"}) — actual lineups are announced
close to match time and can differ.</p></div>""")
        gen = F["generated"]
        stale = ' <strong>(stale — regenerate with web/make_forecast.py)</strong>' \
            if gen < updated else ""
        body_mid = (f'<p class="note">generated {gen}{stale} · '
                    f'{len(F["forecasts"])} scheduled matchups rated</p>'
                    + "".join(cards))
    else:
        body_mid = ('<p class="note">No forecast snapshot yet — run '
                    '<code>python web/make_forecast.py</code> (network) to rate '
                    'the next week of scheduled MLP matchups.</p>')
    body = f"""
<h1>Upcoming matchup forecasts</h1>
<p class="sub">Every scheduled MLP matchup in the next week, rated before it
happens. Lineups are <strong>projected</strong> from each team's most recent
completed matchup; per-game probabilities use current player values, the
weakest-link penalty and display calibration; the DreamBreaker is rated by
a rally-level model fit on all 101 historical DreamBreakers — doubles skill
transfers to DB rallies at roughly half strength, so the stronger roster is
a mild (not heavy) DB favorite. To make a forecast part of the permanent
record, it must be frozen into the <a href="receipts.html">receipts ledger</a>
before first serve (<code>make_forecast.py --commit</code>) — this page alone
is a living view, not a commitment.</p>
{body_mid}
"""
    write("forecast.html", style.page("Forecasts — PICKLES",
                                      body, "forecast.html", "", updated))


# ---------------------------------------------------------------- results

def build_results(players, games, updated, days=14):
    mv = D.month_values(players)
    cutoff = (__import__("datetime").date.fromisoformat(updated)
              - __import__("datetime").timedelta(days=days)).isoformat()
    recent = [g for g in games if g["date"] >= cutoff]
    rows, cur = [], None
    n_upsets = 0
    for g in reversed(recent):
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 == s2:
            continue
        t1_won = s1 > s2
        exp = D.expected_share(players, mv, g)
        price = ""
        upset = ""
        if exp is not None:
            w_exp = exp if t1_won else 1 - exp
            T = 15 if g["scoring_format"].endswith("15") else 11
            pw = calibrate(race_dist(round(w_exp, 4), T)["p_win"])
            price = f"{100 * pw:.0f}%"
            if pw < 0.25:
                upset = ' <span class="chip miss">UPSET</span>'
                n_upsets += 1
        winners = (g["t1_p1"], g["t1_p2"]) if t1_won else (g["t2_p1"], g["t2_p2"])
        losers = (g["t2_p1"], g["t2_p2"]) if t1_won else (g["t1_p1"], g["t1_p2"])
        def pnames(pids):
            return " / ".join(
                plink(players[u]) if u in players else "?" for u in pids)
        if g["date"] != cur:
            cur = g["date"]
            rows.append(f'<tr><td colspan="6" style="padding-top:14px">'
                        f'<strong>{cur}</strong></td></tr>')
        ot = max(s1, s2) > (15 if g["scoring_format"].endswith("15") else 11)
        rows.append(
            f'<tr><td class="gray small">{esc(g["event_name"])[:34]}</td>'
            f'<td class="gray small">{g["tour"]}{" · " + (g["context"] or "") if g["context"] else ""}</td>'
            f'<td>{pnames(winners)}</td><td>{pnames(losers)}</td>'
            f'<td class="num"><strong>{max(s1, s2)}-{min(s1, s2)}</strong>'
            f'{" OT" if ot else ""}</td>'
            f'<td class="num">{price}{upset}</td></tr>')
    body = f"""
<h1>Recent results, rated</h1>
<p class="sub">Every pro doubles game of the last {days} days with the win
probability the model would have quoted for the eventual winners before the
game (current monthly values — a living retrospective, not a frozen
commitment; the <a href="receipts.html">receipts page</a> holds those).
Winners listed first. {n_upsets} upsets (winner rated under 25%).</p>
<div class="tblwrap"><table><tr><th>event</th><th></th><th>winners</th>
<th>over</th><th class="num">score</th><th class="num">winner was rated</th></tr>
{''.join(rows)}</table></div>
"""
    write("results.html", style.page("Results — PICKLES",
                                     body, "results.html", "", updated))


# ---------------------------------------------------------------- title race

ROUND_DEPTH = {"Finals": 0, "Semi-Finals": 1, "Quarter Finals": 2,
               "Round 16": 3, "Round 32": 4, "Round 64": 5}


def bracket_seed_order(n):
    """Seed at each slot of a standard 1-vs-N template, e.g. 8 ->
    [1, 8, 4, 5, 2, 7, 3, 6].  Verified against the 2026 Portland draw
    (semis = 1-quarter vs 4-quarter, 2-quarter vs 3-quarter)."""
    order = [1]
    while len(order) < n:
        m = 2 * len(order)
        order = [s for a in order for s in (a, m + 1 - a)]
    return order


def bo_win(p, best_of):
    """Match win prob from a single-game win prob."""
    if best_of == 3:
        return p * p * (3 - 2 * p)
    if best_of == 5:
        return p ** 3 * (10 - 15 * p + 6 * p * p)
    return p


def mlp_race_panel(state, F, rng):
    """Groups + playoff for the live MLP event.  Format (decoded from the
    completed MLP Dallas 2026 weekend): two round-robin groups, then
    cross-group "One and Done" placement matchups — the two GROUP WINNERS
    meet in the title matchup.  P(champion) = simulate the remaining round
    robin, then the title matchup between the simulated group winners."""
    from collections import defaultdict
    teams = {}

    def team(title):
        return teams.setdefault(title, {
            "title": title, "pool": None,
            "mw": 0, "ml": 0, "gw": 0, "gl": 0, "pd": 0})

    is_rr = lambda r: (r.get("bracket") or "RR") == "RR"
    rr_done = [r for r in state["completed"] if is_rr(r)]
    rr_left = [r for r in state["remaining"] if is_rr(r)]
    po_done = [r for r in state["completed"] if not is_rr(r)]
    po_left = [r for r in state["remaining"] if not is_rr(r)]

    for r in rr_done:
        t1, t2 = team(r["team1"]), team(r["team2"])
        t1["pool"] = r.get("pool") or t1["pool"]
        t2["pool"] = r.get("pool") or t2["pool"]
        g1, g2 = r["games1"] or 0, r["games2"] or 0
        t1["gw"] += g1; t1["gl"] += g2; t2["gw"] += g2; t2["gl"] += g1
        pd = (r["pts1"] or 0) - (r["pts2"] or 0)
        t1["pd"] += pd; t2["pd"] -= pd
        if r["winner"] == 1:
            t1["mw"] += 1; t2["ml"] += 1
        elif r["winner"] == 2:
            t2["mw"] += 1; t1["ml"] += 1
    for r in rr_left:
        for k in ("team1", "team2"):
            t = team(r[k])
            t["pool"] = r.get("pool") or t["pool"]

    pools = defaultdict(list)
    for v in teams.values():
        pools[v["pool"] or "?"].append(v["title"])
    pool_ids = sorted(pools, key=lambda p: (-len(pools[p]), p))
    pool_label = {p: chr(ord("A") + i) for i, p in enumerate(pool_ids)}

    matrix = state.get("matrix") or {}

    def matrix_tree(a, b):
        """(tree, flip): tree oriented to `a`.  Stored oriented to the
        alphabetically-first title."""
        first, flip = (a, False) if a <= b else (b, True)
        second = b if not flip else a
        t = matrix.get(f"{first}|{second}")
        return (t, flip) if t else (None, False)

    fmap = {}
    for f in (F or {}).get("forecasts", []):
        if f.get("tree"):
            fmap[(f["date"], f["team1"], f["team2"])] = f["tree"]

    slate = []
    for r in rr_left + po_left:
        tree = fmap.get((r["date"], r["team1"], r["team2"]))
        flip = False
        if tree is None:
            tree = fmap.get((r["date"], r["team2"], r["team1"]))
            flip = tree is not None
        if tree is None:
            tree, flip = matrix_tree(r["team1"], r["team2"])
        slate.append((r, tree, flip))

    def sample_score(tree, flip):
        """(team1_won, games1, games2) sampled from a matchup tree."""
        p40, p31 = tree["p_40"], tree["p_31"]
        pdb, p13, p04 = tree["p_db"], tree["p_13"], tree["p_04"]
        pdbw = tree.get("p_db_win", 0.5)
        if flip:
            p40, p31, p13, p04, pdbw = p04, p13, p31, p40, 1 - pdbw
        u = rng.random() * (p40 + p31 + pdb + p13 + p04)
        if u < p40:
            return True, 4, 0
        if u < p40 + p31:
            return True, 3, 1
        if u < p40 + p31 + pdb:
            return (True, 3, 2) if rng.random() < pdbw else (False, 2, 3)
        if u < p40 + p31 + pdb + p13:
            return False, 1, 3
        return False, 0, 4

    # completed playoff results pin the sim (once the title matchup is
    # played, P(champion) collapses to fact)
    po_result = {frozenset((r["team1"], r["team2"])):
                 (r["team1"] if r["winner"] == 1 else r["team2"])
                 for r in po_done if r.get("winner")}

    N = 20000
    n_unpriced = sum(1 for r, t, _ in slate if t is None and is_rr(r))
    tops, poolw = defaultdict(int), defaultdict(int)
    rr_slate = [(r, t, f) for r, t, f in slate if is_rr(r)]
    for _ in range(N):
        mw = {t: v["mw"] for t, v in teams.items()}
        gw = {t: v["gw"] for t, v in teams.items()}
        for r, tree, flip in rr_slate:
            a, b = r["team1"], r["team2"]
            if tree is None:
                won1, ga, gb = (rng.random() < 0.5), 3, 1
                if not won1:
                    ga, gb = gb, ga
            else:
                won1, ga, gb = sample_score(tree, flip)
            mw[a if won1 else b] += 1
            gw[a] += ga; gw[b] += gb
        winners = []
        for p in pool_ids:
            if p == "?":
                continue
            winners.append(max(pools[p],
                               key=lambda t: (mw[t], gw[t], teams[t]["pd"],
                                              rng.random())))
        for w in winners:
            poolw[w] += 1
        if len(winners) == 2:
            wa, wb = winners
            pinned = po_result.get(frozenset((wa, wb)))
            if pinned:
                champ = pinned
            else:
                tree, flip = matrix_tree(wa, wb)
                p = (1 - tree["p_win"] if flip else tree["p_win"]) if tree else 0.5
                champ = wa if rng.random() < p else wb
        elif winners:
            champ = winners[0]
        else:
            continue
        tops[champ] += 1

    bar_rows = []
    for v in sorted(teams.values(), key=lambda v: -tops[v["title"]]):
        p_c = tops[v["title"]] / N
        bar_rows.append(
            f'<div class="p1row"><span class="p1name">{esc(v["title"])}</span>'
            f'<span class="p1bar"><span class="p1fill" '
            f'style="width:{p1_bar_width(p_c)}%"></span></span>'
            f'<span class="p1pct">{pct_floor(p_c)}</span>'
            f'<span class="p1meta">group {pool_label.get(v["pool"] or "?", "?")} '
            f'· win group {pct_floor(poolw[v["title"]] / N)}</span></div>')

    group_tables = []
    for p in pool_ids:
        if p == "?" and len(pool_ids) > 1:
            continue
        rows = []
        for t in sorted(pools[p], key=lambda t: (-teams[t]["mw"],
                                                 -(teams[t]["gw"] - teams[t]["gl"]),
                                                 -teams[t]["pd"])):
            v = teams[t]
            rows.append(f'<tr><td>{esc(t)}</td>'
                        f'<td class="num">{v["mw"]}–{v["ml"]}</td>'
                        f'<td class="num">{v["gw"]}–{v["gl"]}</td>'
                        f'<td class="num">{v["pd"]:+d}</td></tr>')
        group_tables.append(
            f'<div><h3>Group {pool_label[p]}</h3>'
            f'<table><tr><th>team</th><th class="num">matchups</th>'
            f'<th class="num">games</th><th class="num">rally ±</th></tr>'
            + "".join(rows) + "</table></div>")

    srows = []
    for r, tree, flip in slate:
        if tree is None:
            price = '<span class="gray">not rated</span>'
        else:
            p1 = 1 - tree["p_win"] if flip else tree["p_win"]
            fav, pf = (r["team1"], p1) if p1 >= 0.5 else (r["team2"], 1 - p1)
            price = f"<strong>{team_short(fav)} {pct_floor(pf)}</strong>"
        tag = "" if is_rr(r) else ' <span class="chip">PLAYOFF</span>'
        srows.append(
            f'<tr><td class="num gray">{r["date"][5:]}</td>'
            f'<td class="num gray">{start_et(r.get("start") or "")}</td>'
            f'<td>{team_short(r["team1"])} v {team_short(r["team2"])}{tag}</td>'
            f'<td class="num">{price}</td></tr>')
    unpriced_note = (f" {n_unpriced} round-robin matchup"
                     f"{'s' if n_unpriced != 1 else ''} unrated — simulated "
                     f"as coin flips." if n_unpriced else "")
    return f"""
<h2><span class="secno">MLP</span>{esc(state["event"])}</h2>
<div class="card p1card">
<div class="p1head"><strong>Who wins the weekend?</strong> <span class="note">
two round-robin groups; the group winners meet in the One-and-Done title
matchup. {N:,} simulations from the actual standings.</span></div>
{''.join(bar_rows)}
<p class="note" style="margin:10px 0 2px">Group rank: matchup wins, then game
wins, then actual rally-point differential (simulated ties broken the same
way; rally points frozen at actuals). Round-robin matchups use the slate-page
forecasts; simulated title matchups use the same model on projected
lineups.{unpriced_note}</p>
</div>
<div class="cols">{''.join(group_tables)}</div>
<h3>Remaining slate</h3>
<div class="tblwrap"><table><tr><th class="num">date</th><th class="num">start</th>
<th>matchup</th><th class="num">favorite</th></tr>{''.join(srows)}</table></div>
"""


def ppa_pair_value(pair, players, floor_value):
    vs = []
    for u in pair:
        p = players.get(u)
        vs.append(p.value if p else floor_value)
    return vs


def ppa_bracket_panel(t, players, floor_value):
    """One PPA tournament: per-division seeded-bracket DP for P(title)."""
    from sitelib.race import race_dist as rd, sigmoid as sg, team_eta as te, \
        calibrate as cal
    panels = []
    for div in t["divisions"]:
        ms = div["matches"]
        entrants = {}                      # seed -> {"pair": [u,u], "names": []}
        best_of = 3
        for m in ms:
            for sd, pk, nk in ((m["seed1"], "p1", "n1"), (m["seed2"], "p2", "n2")):
                if sd and all(m[pk]):
                    entrants.setdefault(sd, {"pair": m[pk], "names": m[nk]})
            if m.get("best_of"):
                best_of = m["best_of"]
        if len(entrants) < 4:
            continue
        known = sum(1 for m in ms if m["round_text"] in ROUND_DEPTH)
        if known < len(ms) / 2:
            continue      # group-stage format (e.g. PPA Finals Top 8) — the
                          # knockout template does not apply; skip honestly
        size = 1
        while size < max(len(entrants), max(entrants)):
            size *= 2
        slots = bracket_seed_order(size)

        # Pin by each seed's KNOWN result at a depth (won -> advances,
        # lost -> out), not by exact opponent sets: real draws shuffle
        # low-seed slots after withdrawals (Portland's play-in was 32 v 29
        # where the template says 32 v 33), and per-seed fate is immune.
        won_at, lost_at = set(), set()
        n_done = 0
        latest_round = ""
        for m in ms:
            depth = ROUND_DEPTH.get(m["round_text"])
            if depth is None or not m["winner"] or m["completed_type"] != 5:
                continue
            if m["seed1"] and m["seed2"]:
                w, l = ((m["seed1"], m["seed2"]) if m["winner"] == 1
                        else (m["seed2"], m["seed1"]))
                won_at.add((depth, w))
                lost_at.add((depth, l))
                n_done += 1
                latest_round = latest_round or m["round_text"]

        def p_beat(sa, sb):
            va = ppa_pair_value(entrants[sa]["pair"], players, floor_value)
            vb = ppa_pair_value(entrants[sb]["pair"], players, floor_value)
            eta = te(va[0], va[1], vb[0], vb[1])
            T = 11 if best_of > 1 else 15
            return bo_win(cal(rd(round(sg(eta), 4), T)["p_win"]), best_of)

        max_depth = 0
        while 2 ** max_depth < size:
            max_depth += 1

        def dist_at(depth, idx):
            """{seed: prob} of who emerges from this node of the template."""
            if depth == max_depth:
                s = slots[idx]
                return {s: 1.0} if s in entrants else {}
            left = dist_at(depth + 1, 2 * idx)
            right = dist_at(depth + 1, 2 * idx + 1)
            if not left or not right:
                return left or right       # bye: free pass
            out = {}
            for sa, pa in left.items():
                for sb, pb in right.items():
                    a_won, b_won = (depth, sa) in won_at, (depth, sb) in won_at
                    a_lost, b_lost = (depth, sa) in lost_at, (depth, sb) in lost_at
                    if a_won and not b_won:
                        w = 1.0
                    elif b_won and not a_won:
                        w = 0.0
                    elif a_lost and not b_lost:
                        w = 0.0
                    elif b_lost and not a_lost:
                        w = 1.0
                    else:
                        w = p_beat(sa, sb)
                    out[sa] = out.get(sa, 0) + pa * pb * w
                    out[sb] = out.get(sb, 0) + pa * pb * (1 - w)
            return out

        title_p = dist_at(0, 0)
        board = sorted(title_p.items(), key=lambda kv: -kv[1])[:8]
        rows = []
        for sd, p in board:
            e = entrants[sd]
            names = " / ".join(esc(n) for n in e["names"] if n) or f"seed {sd}"
            unrated = any(u not in players for u in e["pair"])
            rows.append(
                f'<div class="p1row"><span class="p1name">{names}'
                f'{" *" if unrated else ""}</span>'
                f'<span class="p1bar"><span class="p1fill" '
                f'style="width:{p1_bar_width(p)}%"></span></span>'
                f'<span class="p1pct">{pct_floor(p)}</span>'
                f'<span class="p1meta">seed {sd}</span></div>')
        stage = (f"{n_done} main-draw matches played; latest round: "
                 f"{latest_round}." if n_done else "Draw set; play pending.")
        panels.append(
            f'<h3>{esc(div["title"])}</h3>'
            f'<div class="card p1card"><div class="p1head"><strong>P(title)'
            f'</strong> <span class="note">{stage} Seeded-bracket DP over the '
            f'actual draw; per-game probs from current model values.</span></div>'
            + "".join(rows) + '</div>')
    if not panels:
        return ""
    note = ("* pair includes a player without a model rating (fewer than 60 "
            "pro games) — filled with the field's 25th-percentile value.")
    return (f'<h2><span class="secno">PPA</span>{esc(t["tournament"])}</h2>'
            + "".join(panels) + f'<p class="note">{note}</p>')


def build_titlerace(players, updated):
    state = None
    p = D.DATA / "tournament_state.json"
    if p.exists():
        try:
            state = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            state = None
    import random
    rng = random.Random(20260717)
    F = load_forecasts()
    by_uuid = {p.pid: p for p in players.values()}
    rated = sorted(p.value for p in players.values() if p.dynamic)
    floor_value = rated[len(rated) // 4] if rated else 0.0

    sections = []
    if state and state.get("mlp"):
        sections.append(mlp_race_panel(state["mlp"], F, rng))
    for t in (state or {}).get("ppa") or []:
        sections.append(ppa_bracket_panel(t, by_uuid, floor_value))
    sections = [s for s in sections if s]

    if sections:
        body_main = "".join(sections)
    else:
        body_main = ('<div class="card"><p>No MLP event or PPA pro doubles '
                     'draw is live this week. This page wakes up with the '
                     'next event — it refreshes with every nightly data '
                     'build.</p></div>')
    gen = (state or {}).get("generated", updated)
    body = f"""
<h1 class="runtitle">Title race</h1>
<div class="runmeta">RUN {gen} :: WHO WINS THE WEEKEND :: UPDATES NIGHTLY WITH RESULTS</div>
<p class="sub">The live event, simulated to the end from the current state:
MLP standings use <strong>actual results and rally points</strong>, with the
rest of the round robin rated by the model; PPA uses the <strong>actual
seeded draw</strong>, with every remaining bracket path simulated from
current form values. Assumptions printed where they live.</p>
{body_main}
<p class="note">MLP matchup ratings come from the same projected-lineup
forecasts as the <a href="forecast.html">slate page</a> (lineups are
projections until posted). PPA games are modeled as rally races to 11
(best-of-3) from current values; single-game Challenger rounds are modeled
as races to 15 — side-out scoring makes that an approximation
(<a href="methods.html">methods</a>).</p>
"""
    write("titlerace.html", style.page("Title race — PICKLES",
                                       body, "titlerace.html", "", updated))


# ---------------------------------------------------------------- downloads

DOWNLOADS = [
    ("games.csv", "every game 2024–26: both pairs (player UUIDs), score, "
                  "date, tour, format, stage; the modeling unit"),
    ("players.csv", "canonical player registry: UUID, name, gender, name variants"),
    ("v2_players.csv", "current-form model values (per-point logit) ± sd per player"),
    ("v2_trajectories.csv", "monthly skill curve for every ≥60-game player"),
    ("v2_dyads.csv", "pair-chemistry posteriors (small and honest)"),
    ("yearly_values.csv", "season-by-season v1 values (points scale) + gender ranks"),
    ("platform_ratings.csv", "latest synced DUPR per player + snapshot count"),
    ("singles_games.csv", "every PPA pro singles game 2024–26 (26k games)"),
    ("singles_players.csv", "singles rating per player (MAP fit, recency-weighted)"),
]


def build_downloads(games, updated):
    import shutil
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for fname, desc in DOWNLOADS:
        src = D.DATA / fname
        if not src.exists():
            continue
        shutil.copy(src, SITE / "data" / fname)
        n = sum(1 for _ in src.open()) - 1
        mb = src.stat().st_size / 1e6
        rows.append(f'<tr><td><a href="data/{fname}" download>{fname}</a></td>'
                    f'<td class="num">{n:,}</td><td class="num">{mb:.1f} MB</td>'
                    f'<td>{desc}</td></tr>')
    shutil.copy(D.MODEL / "receipts.json", SITE / "data" / "receipts.json")
    rows.append('<tr><td><a href="data/receipts.json" download>receipts.json</a></td>'
                '<td class="num">—</td><td class="num">&lt;0.1 MB</td>'
                '<td>the public prediction ledger, machine-readable</td></tr>')
    body = f"""
<h1>Open data</h1>
<p class="sub">The CSVs behind every page, free to use with attribution
("based on public results data via PICKLES"). Player identity is
by UUID — names collide (there are three Kawamotos; two are twins).</p>
<div class="tblwrap"><table><tr><th>file</th><th class="num">rows</th>
<th class="num">size</th><th>contents</th></tr>{''.join(rows)}</table></div>
<p class="note">Model values are on a per-point logit scale; the site
displays them as expected margin vs an average pairing via the race DP
(<a href="methods.html">methods</a>). DreamBreakers and forfeits are
excluded from games.csv-derived stats. Refreshed with the nightly build;
data through {updated}.</p>
"""
    write("data.html", style.page("Open data — PICKLES",
                                  body, "data.html", "", updated))


# ---------------------------------------------------------------- receipts

def build_receipts(updated):
    R = D.load_receipts()
    val = R["validation"]
    cal = CAL["buckets_raw"]

    graded_items = [i for e in R["entries"] if e["status"] == "graded"
                    for i in e["items"] if i["result"] is not None]
    hits = sum(1 for i in graded_items if i["grade"] == "HIT")
    briers = [i["brier"] for i in graded_items if i["brier"] is not None]
    mean_brier = sum(briers) / len(briers) if briers else None

    cards = []
    for e in R["entries"]:
        chip = {"graded": "", "pending": '<span class="chip pending">PENDING</span>'}[e["status"]]
        rows = []
        for i in e["items"]:
            g = i["grade"]
            cls = {"HIT": "hit", "MISS": "miss", "VOID": "void", "PENDING": "pending"}[g]
            mark = {"HIT": "✓", "MISS": "✗", "VOID": "–", "PENDING": "●"}[g]
            prob = pct(i["prob"], 0) if i["prob"] is not None else "—"
            outc = esc(i["outcome"]) if i["outcome"] else '<span class="gray">awaiting games</span>'
            br = f'{i["brier"]:.3f}' if i["brier"] is not None else "—"
            rows.append(f'<tr><td>{esc(i["label"])}</td><td class="num">{prob}</td>'
                        f'<td>{outc}</td><td><span class="chip {cls}">{mark} {g}</span></td>'
                        f'<td class="num">{br}</td></tr>')
        meta_bits = [f'committed <strong>{e["committed"]}</strong>']
        if e.get("graded"):
            meta_bits.append(f'graded <strong>{e["graded"]}</strong>')
        if e.get("grade_after"):
            meta_bits.append(f'to be graded after {e["grade_after"]}')
        summary = f'<p>{esc(e["outcome_summary"])}</p>' if e.get("outcome_summary") else ""
        foot = f'<p class="note">{esc(e["footnote"])}</p>' if e.get("footnote") else ""
        cards.append(f"""
<div class="card"><h3 style="margin-top:0">{esc(e["title"])} {chip}</h3>
<p class="note">{' · '.join(meta_bits)} · {esc(e["model"])} ·
<a href="{REPO}/blob/main/{e["source"]}">frozen source ↗</a></p>
{summary}
<div class="tblwrap"><table><tr><th>call</th><th class="num">prob</th><th>outcome</th>
<th>grade</th><th class="num">Brier</th></tr>{''.join(rows)}</table></div>
{foot}</div>""")

    body = f"""
<h1>Receipts</h1>
<p class="sub">Every prediction is committed with a timestamp before the match
and graded in public — hit or miss. Brier score = (probability − outcome)²;
0 is clairvoyance, 0.25 is a coin flip, lower is better. The model being
publicly wrong is part of the product.</p>
<div class="tiles">
 <div class="tile"><div class="v">{pct(val['accuracy'], 1)}</div>
  <div class="k">winners called on {val['n_games']} unseen games (frozen 6 weeks)</div></div>
 <div class="tile"><div class="v">{pct(val['dupr_reference']['accuracy'], 1)}</div>
  <div class="k">DUPR on the same games — while its ratings kept updating</div></div>
 <div class="tile"><div class="v">{val['brier']:.3f}</div>
  <div class="k">holdout Brier (DUPR {val['dupr_reference']['brier']:.3f})</div></div>
 <div class="tile"><div class="v">{hits}/{len(graded_items)}</div>
  <div class="k">public registered calls hit so far{f" · mean Brier {mean_brier:.3f}" if mean_brier is not None else ""}</div></div>
</div>
{''.join(cards)}
<h2>Calibration</h2>
<p class="note">When the frozen model said a team should win X% of the time,
how often did they actually win? Dots on the diagonal = honest probabilities.
Curve measured out-of-sample: every post-June-2026 game rated by the frozen
pre-June fit ({CAL["fit_on"]["n_games"]} games). The fitted correction is
nearly the identity (slope {CAL["b"]:.2f}) — the earlier v1 model ran
underconfident, this one doesn't. What the data does insist on: across all
36k games, favorites rated above 99% still lost {CAL["tail"]["losses"]} of
{CAL["tail"]["n_extreme"]} games (~1%), so site probabilities are floored —
nothing is ever 0% or 100%. Frozen predictions in the ledger above are
graded exactly as committed, never re-calibrated after the fact.</p>
{charts.calibration_chart(cal)}
"""
    write("receipts.html", style.page("Receipts — PICKLES",
                                      body, "receipts.html", "", updated))


# ---------------------------------------------------------------- records

def build_records(players, agg, games, updated):
    def pair_names(k, root=""):
        return " & ".join(plink(players[u], root) if u in players else "?" for u in k)

    pair_best = sorted(agg["pair_best_streak"].items(), key=lambda kv: -kv[1])[:10]
    pair_rows = "".join(
        f'<tr><td>{pair_names(k)}</td><td class="num">{v}</td>'
        f'<td class="num">{agg["pair_games"][k][1]}–{agg["pair_games"][k][0] - agg["pair_games"][k][1]}</td></tr>'
        for k, v in pair_best)

    streaks = sorted(((p.stats.best_streak, p) for p in players.values()
                      if p.dynamic and p.stats), key=lambda t: -t[0])[:10]
    streak_rows = "".join(
        f'<tr><td>{plink(p)}</td><td class="num">{s}</td>'
        f'<td class="num">{wl(p.stats.w, p.stats.l)}</td></tr>' for s, p in streaks)

    upset_rows = []
    for pw, g in agg["upsets"][:12]:
        t1_won = int(g["t1_score"]) > int(g["t2_score"])
        winners = (g["t1_p1"], g["t1_p2"]) if t1_won else (g["t2_p1"], g["t2_p2"])
        losers = (g["t2_p1"], g["t2_p2"]) if t1_won else (g["t1_p1"], g["t1_p2"])
        sc = f'{max(int(g["t1_score"]), int(g["t2_score"]))}-{min(int(g["t1_score"]), int(g["t2_score"]))}'
        upset_rows.append(
            f'<tr><td class="num gray">{g["date"]}</td>'
            f'<td>{pair_names(winners)}</td><td>{pair_names(losers)}</td>'
            f'<td class="num">{sc}</td><td class="num"><strong>{100 * calibrate(pw):.1f}%</strong></td></tr>')

    mar_rows = []
    for total, g in agg["marathons"]:
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        hi, lo = max(s1, s2), min(s1, s2)
        winners = (g["t1_p1"], g["t1_p2"]) if s1 > s2 else (g["t2_p1"], g["t2_p2"])
        losers = (g["t2_p1"], g["t2_p2"]) if s1 > s2 else (g["t1_p1"], g["t1_p2"])
        mar_rows.append(f'<tr><td class="num gray">{g["date"]}</td>'
                        f'<td>{pair_names(winners)}</td><td>{pair_names(losers)}</td>'
                        f'<td class="num"><strong>{hi}-{lo}</strong></td>'
                        f'<td>{esc(g["event_name"])[:38]}</td></tr>')

    busiest = sorted(agg["pair_games"].items(), key=lambda kv: -kv[1][0])[:10]
    busy_rows = "".join(
        f'<tr><td>{pair_names(k)}</td><td class="num">{v[0]}</td>'
        f'<td class="num">{v[1]}–{v[0] - v[1]}</td>'
        f'<td class="num">{pct(v[1] / v[0])}</td></tr>' for k, v in busiest)

    grinders = sorted((p for p in players.values() if p.dynamic),
                      key=lambda p: -p.games)[:10]
    grind_rows = "".join(
        f'<tr><td>{plink(p)}</td><td class="num">{p.games}</td>'
        f'<td class="num">{wl(p.stats.w, p.stats.l)}</td></tr>' for p in grinders)

    blow = sorted((p for p in players.values() if p.dynamic and p.stats),
                  key=lambda p: -p.stats.blowout_w)[:10]
    blow_rows = "".join(
        f'<tr><td>{plink(p)}</td><td class="num">{p.stats.blowout_w}</td>'
        f'<td class="num">{pct(p.stats.blowout_w / max(p.stats.w, 1))}</td></tr>' for p in blow)

    body = f"""
<h1>Record book</h1>
<p class="sub">Mined from every game since January 2024 ({len(games):,} games,
DreamBreakers and forfeits excluded).</p>
<div class="cols">
<div><h2>Longest pair win streaks</h2>
<table><tr><th>pair</th><th class="num">streak</th><th class="num">career W–L</th></tr>{pair_rows}</table></div>
<div><h2>Longest individual win streaks</h2>
<table><tr><th>player</th><th class="num">streak</th><th class="num">career W–L</th></tr>{streak_rows}</table></div>
</div>
<h2>Biggest upsets</h2>
<p class="note">Games the model rates lowest for the eventual winners, using
each month's values (retrospective rating, all four players tracked;
calibrated probabilities — no probability ever reaches 0%, because across 36k
games about 1 in 100 "sure things" lost anyway). Games with recorded
mid-match player swaps and raw probabilities under 0.2% are excluded as probable
data quirks rather than miracles.</p>
<div class="tblwrap"><table><tr><th>date</th><th>winners</th><th>over</th>
<th class="num">score</th><th class="num">model gave them</th></tr>{''.join(upset_rows)}</table></div>
<h2>Overtime marathons</h2>
<div class="tblwrap"><table><tr><th>date</th><th>winners</th><th>over</th>
<th class="num">score</th><th>event</th></tr>{''.join(mar_rows)}</table></div>
<div class="cols">
<div><h2>Busiest pairs</h2>
<table><tr><th>pair</th><th class="num">games</th><th class="num">W–L</th>
<th class="num">win%</th></tr>{busy_rows}</table></div>
<div><h2>Most games played</h2>
<table><tr><th>player</th><th class="num">games</th><th class="num">W–L</th></tr>{grind_rows}</table></div>
</div>
<h2>Blowout leaders</h2>
<p class="note">Wins conceding 5 points or fewer (games to 11).</p>
<table><tr><th>player</th><th class="num">blowout wins</th>
<th class="num">share of all wins</th></tr>{blow_rows}</table>
"""
    write("records.html", style.page("Record book — PICKLES",
                                     body, "records.html", "", updated))


# ---------------------------------------------------------------- dupr page

def build_dupr(players, updated):
    panels, tables = [], []
    glitched = [p for p in players.values()
                if p.dynamic and p.dupr_glitch and not p.dupr]
    excluded_note = ""
    if glitched:
        names = ", ".join(
            f"{plink(p)} (platform shows a ~3.5 reset artifact; last credible "
            f"value {p.dupr_glitch:.2f})" for p in glitched)
        excluded_note = (f'<p class="note">Excluded from the comparison: '
                         f'{names}. A rating that collapses to DUPR\'s reset '
                         f'default while the player keeps winning pro games is '
                         f'a recording artifact, not a measurement.</p>')
    for gender, label in (("F", "Women"), ("M", "Men")):
        pool = [p for p in players.values()
                if p.dynamic and p.gender == gender and p.dupr
                and (p.last_date or "") >= "2026-01-01"]
        if len(pool) < 10:
            continue
        n = len(pool)
        mx = sum(p.dupr for p in pool) / n
        my = sum(p.pts for p in pool) / n
        sxy = sum((p.dupr - mx) * (p.pts - my) for p in pool)
        sxx = sum((p.dupr - mx) ** 2 for p in pool) or 1
        slope = sxy / sxx
        pts = [(p.name, f"players/{p.pid}.html", p.dupr, p.pts,
                p.pts - (my + slope * (p.dupr - mx))) for p in pool]
        panels.append(f"<div><h3>{label}</h3>{charts.dupr_scatter(pts)}</div>")

        ranked_dupr = sorted(pool, key=lambda p: -p.dupr)
        dr = {p.pid: i + 1 for i, p in enumerate(ranked_dupr)}
        ranked_model = sorted(pool, key=lambda p: -p.value)
        mr = {p.pid: i + 1 for i, p in enumerate(ranked_model)}
        div = sorted(pool, key=lambda p: dr[p.pid] - mr[p.pid])
        rows = []
        for p in div[-5:][::-1] + div[:5]:
            d = dr[p.pid] - mr[p.pid]
            cls = "up" if d > 0 else "down"
            rows.append(f'<tr><td>{plink(p)}</td>'
                        f'<td class="num">#{mr[p.pid]}</td><td class="num">#{dr[p.pid]}</td>'
                        f'<td class="num {cls}">{d:+d}</td></tr>')
        tables.append(f"""<div><h3>{label}: biggest disagreements</h3>
<table><tr><th>player</th><th class="num">model rank</th>
<th class="num">DUPR rank</th><th class="num">Δ</th></tr>{''.join(rows)}</table></div>""")

    body = f"""
<h1>DUPR × model</h1>
<p class="sub">DUPR is the sport's official rating. On 518 identical unseen
games the frozen model called 73.7% of winners; DUPR — whose ratings kept
updating all summer — called 64.7%. Below: where the two disagree, per
gender, among 2026-active tracked players.</p>
<div class="tiles">
 <div class="tile"><div class="v">77.4%</div><div class="k">model, 884 unseen games</div></div>
 <div class="tile"><div class="v">64.7%</div><div class="k">DUPR, same games where both rate all four players</div></div>
 <div class="tile"><div class="v">+Δ</div><div class="k">= model ranks the player higher than DUPR does</div></div>
</div>
<div class="cols">{''.join(panels)}</div>
<div class="cols">{''.join(tables)}</div>
{excluded_note}
<p class="note">Caveats we insist on: the DUPR scale compresses hard at the
top and its history contains recorded data glitches (a pro dropping 6.13 →
3.50 mid-season; a tour-wide ~0.5 overnight recalibration on 2026-05-22 —
both visible in our per-match snapshots), so treat rank gaps as
directional. Ratings shown are the value synced at each player's most
recent match, with the as-of date in the tooltip; DUPR's own site may
show a newer or different number. Correlation between the systems is
r ≈ 0.65 (men) / 0.53 (women) — they mostly agree; the divergences are
the story.</p>
"""
    write("dupr.html", style.page("DUPR × model — PICKLES",
                                  body, "dupr.html", "", updated))


# ---------------------------------------------------------------- methods

def md_to_html(md: str) -> str:
    """Tiny renderer sufficient for EXPLAINER.md (headers, bold, em, lists).
    Paragraphs and list items are buffered whole before inline formatting so
    **bold** spanning a wrapped line still renders."""
    out, items, para = [], None, []

    def flush_para():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def flush_list():
        nonlocal items
        if items is not None:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items = None

    for line in md.splitlines():
        if line.startswith("- "):
            flush_para()
            if items is None:
                items = []
            items.append(line[2:].strip())
        elif line.startswith("#"):
            flush_para(); flush_list()
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif not line.strip():
            flush_para(); flush_list()
        elif line.startswith((" ", "\t")) and items is not None:
            items[-1] += " " + line.strip()
        else:
            flush_list()
            para.append(line.strip())
    flush_para(); flush_list()
    return "\n".join(out)


def inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


def build_404(updated):
    """GitHub Pages serves this file for ANY missing path, so relative URLs
    would resolve against the requested directory — the page is fully
    self-contained (inline styles, no shell) and its one link is absolute
    under the project base path.  If a custom domain is ever pointed at the
    site root, change `base` to "/"."""
    base = "/" + REPO.rsplit("/", 1)[1] + "/"
    html = f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>404 — PICKLES</title>
<link rel="icon" href="{style.FAVICON}">
<style>
:root {{ --page:#f2f7e5; --ink:#16321e; --ink2:#46603a; --surface:#fbfdf3;
  --baseline:#a9bc8c; --border:rgba(22,50,30,0.18); --s1:#1e7a3c;
  --hl:#d9f154; --hl-ink:#16321e; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --page:#0e1410; --ink:#edf4e0; --ink2:#c2d3a8; --surface:#17211a;
    --baseline:#3a4a3c; --border:rgba(237,244,224,0.14); --s1:#cfe94f; }}
}}
body {{ margin:0; background:var(--page); color:var(--ink);
  font:15px/1.55 "Space Grotesk", system-ui, sans-serif; }}
.wrap {{ max-width:720px; margin:0 auto; padding:48px 24px; }}
.chip {{ font:700 17px ui-monospace,monospace; background:var(--hl);
  color:var(--hl-ink); padding:3px 10px; display:inline-block; }}
h1 {{ font:400 23px ui-monospace,monospace; margin:28px 0 4px; }}
h1::before {{ content:"> "; color:var(--s1); }}
p {{ color:var(--ink2); }}
.ledger {{ background:var(--surface); border:1.5px solid var(--border);
  padding:12px 16px; margin:16px 0; font-family:ui-monospace,monospace; }}
.lrow {{ display:flex; align-items:baseline; gap:10px; padding:3px 0; }}
.lk {{ font-size:12.5px; letter-spacing:0.04em; color:var(--ink2); }}
.ldot {{ flex:1; border-bottom:2px dotted var(--baseline);
  transform:translateY(-4px); min-width:24px; }}
.lv {{ font-size:14px; font-weight:700; white-space:nowrap; }}
a {{ color:var(--s1); text-decoration:none; }}
a:hover {{ text-decoration:underline 2px var(--hl); }}
</style>
<div class="wrap">
<span class="chip">PICKLES</span>
<h1>404 — not in the ledger</h1>
<p>No page at this address. If a player link brought you here, the player
may not have enough games for monthly tracking (≥60 since 2024).</p>
<div class="ledger">
 <div class="lrow"><span class="lk">REQUESTED PAGE</span><span class="ldot"></span><span class="lv">NOT FOUND</span></div>
 <div class="lrow"><span class="lk">PROBABILITY IT EXISTS</span><span class="ldot"></span><span class="lv">&lt;1%</span></div>
 <div class="lrow"><span class="lk">SUGGESTED ROUTE</span><span class="ldot"></span><span class="lv"><a href="{base}">[OK] → front page</a></span></div>
</div>
</div>
"""
    write("404.html", html)


def build_methods(updated):
    md = (ROOT / "EXPLAINER.md").read_text()
    body = f"""
{md_to_html(md)}
<div class="card"><h2 style="margin-top:0">The technical version, in brief</h2>
<ul>
<li><strong>Likelihood:</strong> every game is a per-point Binomial race —
points won ~ Binomial(total points, σ(η)) — so games to 11 and to 15 enter
one model natively.</li>
<li><strong>η</strong> = (team 1 skill − team 2 skill) + pair-chemistry
deviations + a per-match random effect; team skill = sum of player values
+ γ·|partner gap| (the weakest link, γ ≈ −0.18 on the per-point logit
scale).</li>
<li><strong>Dynamics:</strong> each ≥60-game player's value follows a monthly
Gaussian random walk (2024-01 → now); everyone else is static with
shrinkage.</li>
<li><strong>Validation:</strong> frozen on pre-June-2026 data, scored on the
next six weeks: 77.4% winners, Brier 0.165 (DUPR: 64.7% / 0.229 on the same
games).</li>
<li><strong>Display scale:</strong> the site converts per-point logits to
"expected margin vs an average pairing in a race to 11" — median regular
≈ +2 pts, superstar ≈ +7.</li>
<li><strong>Calibrated, floored probabilities:</strong> win probabilities are
passed through a map fitted on out-of-sample games (near-identity: slope
0.90) with an empirical tail floor — ~1% of ≥99% favorites lose, so no
probability is ever shown as 0% or 100%. There is always a chance.</li>
</ul>
<p class="note">Full writeups, code, data and diagnostics:
<a href="{REPO}">the repository</a> — see analysis.md for every table and
robustness check.</p></div>
<div class="card"><h2 style="margin-top:0">Things we refuse to publish</h2>
<ul>
<li><strong>Cross-gender rankings as fact.</strong> Every game has equal women
per side, so nothing in 36,000 games links the men's and women's scales.</li>
<li><strong>"Great partner" awards.</strong> A game score credits the team;
a player's own skill and their boost to a partner are mathematically
inseparable.</li>
<li><strong>Certified chemistry.</strong> Proving a typical pair effect takes
~1,000 games together; the record is 138. We publish the estimates with
their (wide) error bars instead.</li>
</ul></div>
"""
    write("methods.html", style.page("Methods — PICKLES",
                                     body, "methods.html", "", updated))


# ---------------------------------------------------------------- landing

# Threads/IG handle for the manifesto footer — set once the account exists;
# empty renders the archive line alone (no fake placeholder ships).
HANDLE = ""

# Editorial, hand-maintained (design mock DATA block) — not generated.
FIELD_NOTES = [
    ("F-01", "Chemistry is (mostly) a myth", "quality ≈ 5× pair fit"),
    ("F-02", "It's a weakest-link game", "gap costs ½ pt/game"),
    ("F-03", "Ben Johns never declined", "the field arrived"),
    ("F-04", "Waters' lead over #2", "= men's entire top 25"),
    ("F-05", "Mixed targeting is gender-blind", "attack the weaker player"),
    ("F-06", "Pros don't sandbag MLP", "same pts-per-skill rate"),
]

# MLP team display forms: full name -> (city line, ticker-style short code).
MLP_TEAMS = {
    "Atlanta Bouncers": ("Atlanta", "ATL"),
    "Bay Area Breakers": ("Bay Area", "BAY"),
    "Brooklyn Pickleball Team": ("Brooklyn", "BKN"),
    "California Black Bears": ("California", "CAL"),
    "Carolina Pickleball Club": ("Carolina", "CAR"),
    "Chicago Slice": ("Chicago", "CHI"),
    "Columbus Sliders": ("Columbus", "CLB"),
    "Dallas Flash": ("Dallas", "DAL"),
    "D.C. Pickleball Team": ("D.C.", "DC"),
    "Florida Smash": ("Florida", "FLA"),
    "Los Angeles Mad Drops": ("Los Angeles", "LA"),
    "Miami Pickleball Club": ("Miami", "MIA"),
    "New Jersey 5s": ("New Jersey", "NJ"),
    "New York Hustlers": ("NY", "NY"),
    "Orlando Squeeze": ("Orlando", "ORL"),
    "Phoenix Flames": ("Phoenix", "PHX"),
    "SoCal Hard Eights": ("SoCal", "SOC"),
    "St. Louis Shock": ("St. Louis", "STL"),
    "Texas Ranchers": ("Texas", "TEX"),
    "Utah Black Diamonds": ("Utah", "UTA"),
}


def team_city(name):
    if name in MLP_TEAMS:
        return MLP_TEAMS[name][0]
    ws = name.split()
    return " ".join(ws[:-1]) if len(ws) > 1 else name


def team_short(name):
    if name in MLP_TEAMS:
        return MLP_TEAMS[name][1]
    city = team_city(name)
    ws = city.split()
    return "".join(w[0] for w in ws).upper() if len(ws) > 1 else city[:3].upper()


SLOT_WORD = {"WD": "women's", "MD": "men's", "MXD1": "mixed", "MXD2": "mixed"}


def load_forecasts():
    fj = D.DATA / "forecasts.json"
    return json.loads(fj.read_text()) if fj.exists() else None


def pick_featured(F, today):
    """(featured forecast, tonight?) — the most contested fully-priced matchup
    of the nearest forecast day.  `tonight` is True only when that day falls
    inside the tonight-band window (build day .. +2), which is what gates the
    dark band; teasers may still use a recent-past matchup so the doorway
    cards never show placeholder data."""
    if not F:
        return None, False
    cands = [f for f in F.get("forecasts", []) if f.get("tree")]
    if not cands:
        return None, False
    from datetime import date
    upcoming = sorted({f["date"] for f in cands if f["date"] >= today})
    day = upcoming[0] if upcoming else max(f["date"] for f in cands)
    tonight = bool(upcoming) and \
        (date.fromisoformat(day) - date.fromisoformat(today)).days <= 2
    night = [f for f in cands if f["date"] == day]
    full = [f for f in night if len([g for g in f["games"] if g]) >= 4] or night
    feat = min(full, key=lambda f: abs(f["tree"]["p_win"] - 0.5))
    return feat, tonight


def featured_game_rows(feat):
    """(slot, FAV/PAIR, FAVSHORT, pct) per priced game, favored side."""
    rows = []
    t1s, t2s = team_short(feat["team1"]), team_short(feat["team2"])
    for g in feat["games"]:
        if not g:
            continue
        fav1 = g["p"] >= 0.5
        pair = g["t1_pair"] if fav1 else g["t2_pair"]
        pnm = "/".join(esc(n.split()[-1].upper()) for n in pair)
        rows.append((g["slot"].replace("MXD", "MX"), pnm,
                     t1s if fav1 else t2s,
                     pct_floor(max(g["p"], 1 - g["p"]))))
    return rows[:4]


def tonight_blurb(feat):
    """Formulaic one-liner in the mock's shape: favorite + up to two
    qualifiers (coin-flip slot, DreamBreaker exposure)."""
    tree = feat["tree"]
    fav1 = tree["p_win"] >= 0.5
    fav = team_city(feat["team1"] if fav1 else feat["team2"])
    favp = max(tree["p_win"], 1 - tree["p_win"])
    base = f"{esc(fav)} take the night in {pct_floor(favp)} of simulations"
    clauses = []
    games = [g for g in feat["games"] if g]
    if games:
        tight = min(games, key=lambda g: abs(g["p"] - 0.5))
        if abs(tight["p"] - 0.5) <= 0.05:
            clauses.append(f"the {SLOT_WORD.get(tight['slot'], tight['slot'])} "
                           f"slot is a genuine coin flip")
    if tree["p_db"] >= 0.45:
        clauses.append("almost half of all paths run through a DreamBreaker")
    elif tree["p_db"] >= 0.30:
        clauses.append(f"a DreamBreaker is live in {pct_floor(tree['p_db'])} of paths")
    if clauses:
        return base + " — but " + ", and ".join(clauses) + "."
    return base + "."


def start_et(iso):
    """'2026-07-16T17:00:00Z' -> '1:00P' Eastern (event-local) time."""
    from datetime import datetime, timedelta, timezone
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    try:
        from zoneinfo import ZoneInfo
        loc = dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:                       # no tzdata: EDT is close enough Mar-Nov
        loc = dt.astimezone(timezone(timedelta(hours=-4)))
    h = loc.hour % 12 or 12
    return f"{h}:{loc.minute:02d}{'A' if loc.hour < 12 else 'P'}"


def slate_day_label(day, today):
    from datetime import date, timedelta
    d, t = date.fromisoformat(day), date.fromisoformat(today)
    if d == t:
        return "TODAY'S SLATE"
    if d == t + timedelta(days=1):
        return "TOMORROW'S SLATE"
    return f"NEXT SLATE · {d.strftime('%b %d').upper()}"


def results_day_summary(players, games, day):
    """(n graded games, n upsets) on a date — same pricing as results.html."""
    mv = D.month_values(players)
    graded = upsets = 0
    for g in games:
        if g["date"] != day:
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 == s2:
            continue
        graded += 1
        exp = D.expected_share(players, mv, g)
        if exp is not None:
            w_exp = exp if s1 > s2 else 1 - exp
            T = 15 if g["scoring_format"].endswith("15") else 11
            if calibrate(race_dist(round(w_exp, 4), T)["p_win"]) < 0.25:
                upsets += 1
    return graded, upsets


def build_slate(F, players, games, updated, today):
    """Owner-requested top strip (post-handoff addition, same design
    vocabulary): the day's full card of priced matchups + a latest-results
    line, so the newest numbers are the first thing on the page.  Omitted
    entirely when there is neither an upcoming card nor a graded day."""
    rows, head = "", ""
    if F:
        cands = [f for f in F.get("forecasts", []) if f.get("date")]
        upcoming = sorted({f["date"] for f in cands if f["date"] >= today})
        if upcoming:
            day = upcoming[0]
            night = sorted((f for f in cands if f["date"] == day),
                           key=lambda f: f.get("start") or "")
            event = (night[0].get("event") or "MLP").upper()
            head = (f'<div class="slatehead"><span class="doortag">'
                    f'{slate_day_label(day, today)}</span>'
                    f'<span class="slateevent">{esc(event)} :: {len(night)} '
                    f'MATCHUP{"S" if len(night) != 1 else ""} RATED</span>'
                    f'<span class="fill"></span>'
                    f'<a class="slatelink" href="forecast.html">FULL FORECAST →</a></div>')
            bits = []
            for f in night:
                t = f.get("tree")
                if t:
                    fav1 = t["p_win"] >= 0.5
                    fav = team_short(f["team1"] if fav1 else f["team2"])
                    val = f'{fav} {pct_floor(max(t["p_win"], 1 - t["p_win"]))}'
                else:
                    val = "NOT RATED"
                bits.append(
                    f'<a class="srow" href="forecast.html">'
                    f'<span class="st">{start_et(f.get("start") or "")}</span>'
                    f'<span class="sm">{team_short(f["team1"])} v {team_short(f["team2"])}</span>'
                    f'<span class="lead"></span><span class="sp">{val}</span></a>')
            rows = f'<div class="slaterows">{"".join(bits)}</div>'
    graded, upsets = results_day_summary(players, games, updated)
    foot = ""
    if graded:
        foot = (f'<div class="slatefoot"><span>LATEST RESULTS :: {updated} — '
                f'{graded} GAME{"S" if graded != 1 else ""} GRADED · '
                f'{upsets} UPSET{"S" if upsets != 1 else ""}'
                f'</span><span><a href="results.html">→ RESULTS</a></span></div>')
    if not head and not foot:
        return ""
    return (f'<section class="lsec slate"><div class="slatebox">'
            f'{head}{rows}{foot}</div></section>')


def receipt_teaser_rows(R, n=3):
    """Most recent graded calls, compact: ("WD 88%", "MISS")."""
    rows = []
    for e in reversed(R["entries"]):
        if e["status"] != "graded":
            continue
        for i in e["items"]:
            if i["grade"] not in ("HIT", "MISS") or i["prob"] is None:
                continue
            pre = i["label"].split(":")[0] if ":" in i["label"] else i["label"]
            if pre == "Overall":
                pre = "MATCH"
            elif pre.startswith("Match reaches"):
                pre = "DB"
            rows.append((f'{esc(pre.upper())} {pct_floor(i["prob"])}', i["grade"]))
            if len(rows) >= n:
                return rows
    return rows


def build_landing(players, games, updated, n_games, R):
    from datetime import date
    val = R["validation"]
    acc = pct(val["accuracy"], 1)
    dupr = pct(val["dupr_reference"]["accuracy"], 1)
    gap = f'{100 * (val["accuracy"] - val["dupr_reference"]["accuracy"]):.1f}'
    holdout = val["n_games"]
    n_dyn = sum(1 for p in players.values() if p.dynamic)
    hero_games = f"{round(n_games / 1000) * 1000:,}"

    # -- doorway teasers, all from the same sources as their target pages
    def active_pool(gender):
        return sorted((p for p in players.values()
                       if p.dynamic and p.gender == gender and D.is_active(p)),
                      key=lambda p: p.rank)[:75]

    mprobs = sorted(rank_probs(active_pool("M")), key=lambda t: -t[1])
    wprobs = sorted(rank_probs(active_pool("F")), key=lambda t: -t[1])
    bars = []
    for p, p1, _ in wprobs[:5]:
        pv = min(max(round(100 * p1), 1), 99)
        bars.append(f'<div class="t-bar"><span class="nm">'
                    f'{esc(p.name.split()[-1].upper())}</span>'
                    f'<span class="track"><span class="fill" '
                    f'style="width:{min(round(pv * 2.6), 100)}%"></span></span>'
                    f'<span class="pv">{pv}%</span></div>')
    n_cont = sum(1 for _, p1, _ in mprobs if p1 >= 0.05)
    wsec = second_seat(active_pool("F"))
    if wprobs[0][1] > 0.995 and wsec:
        women_clause = (f"{esc(wprobs[0][0].name.split()[-1])} isn't being "
                        f"caught; {esc(wsec[0].name.split()[-1])} owns second.")
    elif wprobs[0][1] > 0.995:
        women_clause = "The women's #1 isn't close."
    else:
        women_clause = "The women's #1 is a live race."
    men_clause = (f"The men's is a {NUMWORDS.get(n_cont, str(n_cont))}-way "
                  f"statistical tie." if n_cont >= 2
                  else "The men's is settled for now.")
    rankings_blurb = f"{women_clause} {men_clause} Error bars included, always."

    today = date.today().isoformat()
    F = load_forecasts()
    slate = build_slate(F, players, games, updated, today)
    feat, tonight = pick_featured(F, today)
    frows = "".join(
        f'<div class="t-row"><span class="call"><strong>{slot} {pnm}</strong></span>'
        f'<span class="lead"></span><span class="res">{fav} {pv}</span></div>'
        for slot, pnm, fav, pv in (featured_game_rows(feat) if feat else []))
    if not frows:
        frows = ('<div class="t-row"><span class="call">NEXT EVENT</span>'
                 '<span class="lead"></span><span class="res">FORECAST SOON</span></div>')

    rrows = "".join(
        f'<div class="t-row"><span class="call">{call}</span><span class="lead"></span>'
        f'<span class="res {"hitv" if g == "HIT" else "missv"}">'
        f'{"✓ HIT" if g == "HIT" else "✗ MISS"}</span></div>'
        for call, g in receipt_teaser_rows(R))

    term = ("&gt; PICK ANY FOUR PROS<br>\n&gt; ANY PAIRING, ANY FORMAT<br>\n"
            "&gt; RUNNING 100,000 SIMS_<br>\n"
            '<span class="result">→ RATED, WITH ERROR BARS</span>')
    if feat:
        best = max((g for g in feat["games"] if g),
                   key=lambda g: max(g["p"], 1 - g["p"]), default=None)
        if best:
            fav1 = best["p"] >= 0.5
            favpair = "/".join(esc(n.split()[-1].upper()) for n in
                               (best["t1_pair"] if fav1 else best["t2_pair"]))
            dogpair = "/".join(esc(n.split()[-1].upper()) for n in
                               (best["t2_pair"] if fav1 else best["t1_pair"]))
            modal = best["modal"] if fav1 else "-".join(reversed(best["modal"].split("-")))
            term = (f"&gt; PICK ANY FOUR PROS<br>\n&gt; {favpair}<br>\n"
                    f"&nbsp;&nbsp;vs {dogpair}<br>\n&gt; RUNNING 100,000 SIMS_<br>\n"
                    f'<span class="result">→ {pct_floor(max(best["p"], 1 - best["p"]))}'
                    f" · LIKELY {modal}</span>")

    # -- tonight band (omitted entirely when no imminent priced matchup)
    band = ""
    if feat and tonight:
        tree = feat["tree"]
        fav1 = tree["p_win"] >= 0.5
        fav, dog = (feat["team1"], feat["team2"]) if fav1 else (feat["team2"], feat["team1"])
        favp = max(tree["p_win"], 1 - tree["p_win"])
        trows = "".join(
            f'<div class="trow"><span class="g">{slot} {pnm}</span>'
            f'<span class="lead"></span><span class="p">{favshort} {pv}</span></div>'
            for slot, pnm, favshort, pv in featured_game_rows(feat))
        band = f"""
<section class="tonight">
 <div class="inner">
  <div class="copy">
   <div class="kicker">FRESH OFF THE PRINTER</div>
   <h2>{esc(team_city(fav))} vs {esc(team_city(dog))}</h2>
   <p class="blurb">{tonight_blurb(feat)}</p>
   <div class="dbrow"><span class="k">DREAMBREAKER RISK</span><span class="lead"></span><span class="v">{pct_floor(tree["p_db"])}</span></div>
   <a class="cta bright" href="forecast.html">FULL FORECAST →</a>
  </div>
  <div class="ticketwrap">
   <div class="tractor"></div>
   <div class="ticket">
    <div class="th">PICKLES/2.1 :: PRE-MATCH</div>
    <div class="trun">RUN {esc(F["generated"])} :: {esc((feat.get("event") or "MLP").upper())}</div>
    <div class="tok">ALL 4 GAMES + DREAMBREAKER ..... [OK]</div>
    <div class="trule"></div>
    <div class="rows">{trows}</div>
    <div class="trule"></div>
    <div class="ttotal"><span class="k">TOTAL: {esc(team_city(fav).upper())}</span><span class="lead"></span><span class="v">{pct_floor(favp)}</span></div>
    <div class="tdog">({esc(team_city(dog).upper())}: {pct_floor(1 - favp)} — KEEP RECEIPT)</div>
   </div>
  </div>
 </div>
</section>"""

    fnotes = "".join(
        f'<div class="fnote"><span class="no">{no}</span>'
        f'<span class="claim">{esc(claim)}</span><span class="lead"></span>'
        f'<span class="ev">{esc(ev)}</span></div>'
        for no, claim, ev in FIELD_NOTES)

    handle_bit = f"{esc(HANDLE)} · " if HANDLE else ""
    html = f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>PICKLES — pro pickleball, probabilistically</title>
<meta name="description" content="Win probabilities, current-form power rankings and a matchup simulator for pro pickleball (MLP + PPA), from a Bayesian model of {hero_games} games — every forecast timestamped and graded in public.">
{style.FONTS_PRECONNECT}
<link rel="stylesheet" href="assets/style.css">
<link rel="icon" href="{style.FAVICON}">
<header class="landing"><div class="bar">
 <span class="brandchip">PICKLES</span>
 <span class="brandsub">Probabilistic Inference of Competitive Kitchen-Line Expected Scores</span>
 <nav><a href="rankings.html">RANKINGS</a><a href="forecast.html">FORECASTS</a><a href="receipts.html">RECEIPTS</a><a href="simulator.html">SIMULATOR</a><a href="methods.html">METHODS</a></nav>
</div></header>
{slate}
<section class="lsec hero">
 <div class="copy">
  <div class="kicker">PRO PICKLEBALL, PROBABILISTICALLY</div>
  <h1>Who wins?<br>Ask {hero_games} games.</h1>
  <p class="lede">Current-form power rankings, win probabilities for every MLP
night and PPA bracket, a simulator for any four pros — and the full archive
of past forecasts, hits and misses.</p>
  <div class="ctas">
   <a class="cta solid" href="forecast.html">TONIGHT'S SLATE →</a>
   <a class="cta outline" href="receipts.html">PAST FORECASTS</a>
  </div>
 </div>
 <div class="printout">
  <div class="plabel">&gt; P(WINNER CALLED CORRECTLY)</div>
  <div class="headline">
   <span class="hlnum"><span class="swipe"></span><span class="val">{acc}</span></span>
   <span class="whose">THIS MODEL</span>
  </div>
  <div class="cmprow"><span>DUPR, SAME {holdout} GAMES</span><span class="lead"></span><span class="v">{dupr}</span></div>
  <div class="prule"></div>
  <div class="pnote">Held-out games neither system saw. Our model was frozen;
DUPR kept updating. It still lost by {gap} points.</div>
 </div>
</section>

<section class="lsec inside">
 <h2 class="lh2">What the model keeps track of</h2>
 <div class="doorgrid">
  <a class="door" href="rankings.html">
   <span class="doortag">POWER RANKINGS</span>
   <div class="t-bars">{"".join(bars)}</div>
   <p class="doorblurb">{rankings_blurb}</p>
  </a>
  <a class="door" href="forecast.html">
   <span class="doortag">MATCH FORECASTS</span>
   <div class="t-rows">{frows}</div>
   <p class="doorblurb">Every MLP team night and PPA championship Sunday —
win probability, most-likely score, DreamBreaker chances.</p>
  </a>
  <a class="door" href="receipts.html">
   <span class="doortag">TRACK RECORD</span>
   <div class="t-rows wide">{rrows}</div>
   <p class="doorblurb">The full archive of past forecasts, scored: {acc} of
winners called on {holdout} games the model never saw. Misses kept too.</p>
  </a>
  <a class="door" href="simulator.html">
   <span class="doortag">SIMULATOR</span>
   <div class="t-term">{term}</div>
   <p class="doorblurb">Any hypothetical pairing, weakest-link penalty
included. Superstar + passenger ≠ two solids.</p>
  </a>
 </div>
</section>
{band}
<section class="lsec check">
 <div class="syscheck">
  <div class="lrow"><span class="lk">GAMES IN THE MODEL (2024–26, MLP + PPA)</span><span class="ldot"></span><span class="lv">{n_games:,}</span></div>
  <div class="lrow"><span class="lk">PLAYERS WITH MONTHLY SKILL TRACKING</span><span class="ldot"></span><span class="lv">{n_dyn}</span></div>
  <div class="lrow"><span class="lk">DATA THROUGH</span><span class="ldot"></span><span class="lv">{updated}</span></div>
  <div class="lrow"><span class="lk">TOURS COVERED</span><span class="ldot"></span><span class="lv">MLP + PPA</span></div>
 </div>
</section>

<section class="lsec notes">
 <h2 class="lh2">Field notes — what {hero_games} games actually say</h2>
 <div class="fnotes">{fnotes}</div>
</section>

<footer class="manifesto"><div class="inner">
 <div class="credo">
  <div class="slogan">Error bars are a <span class="hl">flex.</span></div>
  <div class="lines">Scored in public — hits and misses.<br>"We can't know
that" — said out loud, when true.</div>
 </div>
 <div class="baseline">
  <span>PICKLES · model output</span>
  <span>{handle_bit}full forecast archive on file</span>
 </div>
</div></footer>
"""
    write("index.html", html)


# ---------------------------------------------------------------- main

def main():
    print("loading model CSVs …")
    players = D.load_players()
    print("loading games …")
    games = D.load_games()
    updated = max(g["date"] for g in games)
    print(f"aggregating {len(games):,} games …")
    agg = D.aggregate(players, games)
    D.finalize_dupr(players)
    D.infer_missing_genders(players)
    D.rank_players(players)
    chem = D.load_dyads()

    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "style.css").write_text(style.CSS)
    (SITE / ".nojekyll").write_text("")

    R = D.load_receipts()
    print("pages: landing, rankings, forecasts, results, simulator, receipts, records, dupr, methods, data …")
    build_landing(players, games, updated, len(games), R)
    build_rankings(players, updated, len(games), R["validation"])
    build_player_index(players, updated)
    build_forecast(players, updated)
    build_results(players, games, updated)
    build_titlerace(players, updated)
    build_simulator(players, updated)
    n_live = livepage.build_live(players, CAL, updated, SITE, write)
    print(f"live page: {n_live} player values shipped")
    build_receipts(updated)
    build_records(players, agg, games, updated)
    build_dupr(players, updated)
    build_methods(updated)
    build_404(updated)
    build_downloads(games, updated)

    dyn = [p for p in players.values() if p.dynamic and p.stats]
    print(f"player pages: {len(dyn)} …")
    for p in dyn:
        build_player_page(p, players, chem, updated)
    n = sum(1 for _ in SITE.rglob("*.html"))
    print(f"done: {n} pages in {SITE}")


if __name__ == "__main__":
    main()
