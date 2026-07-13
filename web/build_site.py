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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib import charts, data as D, style
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


def build_rankings(players, updated, n_games):
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

    sections = []
    for gender, label in (("M", "Men"), ("F", "Women")):
        pool = [p for p in players.values() if p.dynamic and p.gender == gender]
        active = sorted((p for p in pool if D.is_active(p)),
                        key=lambda p: p.rank)[:75]
        inactive = sorted((p for p in pool if not D.is_active(p)),
                          key=lambda p: -p.value)
        probs = [(p, p1, p3) for p, p1, p3 in rank_probs(active) if p1 >= 0.01]
        prob_bits = " · ".join(
            f'{plink(p)} <strong>{pct_floor(p1)}</strong>'
            f'<span class="gray"> (top-3 {pct_floor(p3)}, {p.games} g)</span>'
            for p, p1, p3 in probs)
        panel = (f'<div class="card"><strong>Who is actually #1?</strong> '
                 f'Posterior probability each player is the true best, not just '
                 f'the current point-estimate leader: {prob_bits}. '
                 f'<span class="note">Monte Carlo over posterior marginals '
                 f'(correlations ignored); wide-interval players earn real '
                 f'probability — that is the point.</span></div>')
        sections.append(f"<h2>{label}</h2>" + panel + table(active))
        if inactive:
            sections.append(
                f'<details><summary class="note">{len(inactive)} rated players '
                f'without a 2026 game (hidden, unranked)</summary>{table(inactive)}</details>')

    body = f"""
<h1>Current-form power rankings</h1>
<p class="sub">Every player with ≥60 pro doubles games since 2024, rated by a
dynamic Bayesian model of all {n_games:,} games. <strong>pts</strong> = expected
margin (with an average partner, vs an average pair, race to 11) — median
regular ≈ +2, star ≈ +5. The interval is the point, not fine print.</p>
<div class="tiles">
 <div class="tile"><div class="v">{n_games:,}</div><div class="k">games in the model (2024–26, MLP + PPA)</div></div>
 <div class="tile"><div class="v">77.4%</div><div class="k">winner accuracy on 884 unseen games (DUPR: 64.7%)</div></div>
 <div class="tile"><div class="v">499</div><div class="k">players with monthly skill tracking</div></div>
 <div class="tile"><div class="v">{updated}</div><div class="k">data through</div></div>
</div>
<p class="note">Men's and women's lists are separate on purpose: the tours never
play cross-gender games, so no data links the two scales
(<a href="methods.html">methods</a>). Rankings use current-month posterior
values; ▲/▼ = 6-month form change beyond ±0.25 pts.</p>
{''.join(sections)}
"""
    write("index.html", style.page("Power rankings — Pickleball, Priced",
                                   body, "index.html", "", updated))


# ---------------------------------------------------------------- players

def build_player_index(players, updated):
    pool = sorted((p for p in players.values() if p.dynamic),
                  key=lambda p: p.name.split()[-1])
    rows = "".join(
        f'<tr class="prow" data-name="{esc(p.name.lower())}">'
        f'<td>{plink(p)}</td><td>{p.gender}</td>'
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
          style.page("Players — Pickleball, Priced", body, "players/index.html",
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
          style.page(f"{p.name} — Pickleball, Priced", body, "", root, updated))


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
0% or 100% — across 36k games, favorites we price above 99% still lost about
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
    `<td class="num" style="width:60px">${{(100 * pr).toFixed(1)}}%</td></tr>`).join('') + '</table>';
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
  let html = `<div class="card"><div class="big">${{aN}} ${{(100 * m.p).toFixed(0)}}%` +
    ` <span class="note" style="font-size:15px">(90% range ${{(100 * mLo).toFixed(0)}}–${{(100 * mHi).toFixed(0)}}%)</span></div>` +
    `<div class="pmbar"><div class="a" style="width:${{100 * m.p}}%"></div><div class="b" style="flex:1"></div></div>` +
    `<p class="note">${{aN}} vs ${{bN}} · per-point share ${{(100 * sig(mu)).toFixed(1)}}% · ` +
    `expected margin ${{fmtPts(dist.margin)}} per game (to ${{T}})</p>` +
    (m.scores ? '<p>' + m.scores.map(([s, pr]) => `${{s}}: <strong>${{(100 * pr).toFixed(0)}}%</strong>`).join(' · ') + '</p>' : '') +
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
    write("simulator.html", style.page("Matchup simulator — Pickleball, Priced",
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
                        f'DreamBreaker {pct_floor(t["p_db"])} (priced 50/50) · '
                        f'1–3 {pct_floor(t["p_13"])} · 0–4 {pct_floor(t["p_04"])}</p>')
            else:
                head = (f'<div class="big">{esc(f["team1"])} <span class="gray">vs</span> '
                        f'{esc(f["team2"])}</div><p class="note">not priceable yet '
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
                    f'{len(F["forecasts"])} scheduled matchups priced</p>'
                    + "".join(cards))
    else:
        body_mid = ('<p class="note">No forecast snapshot yet — run '
                    '<code>python web/make_forecast.py</code> (network) to price '
                    'the next week of scheduled MLP matchups.</p>')
    body = f"""
<h1>Upcoming matchup forecasts</h1>
<p class="sub">Every scheduled MLP matchup in the next week, priced before it
happens. Lineups are <strong>projected</strong> from each team's most recent
completed matchup; per-game probabilities use current player values, the
weakest-link penalty and display calibration; the DreamBreaker is treated as
a coin flip by stated convention. To make a forecast part of the permanent
record, it must be frozen into the <a href="receipts.html">receipts ledger</a>
before first serve (<code>make_forecast.py --commit</code>) — this page alone
is a living view, not a commitment.</p>
{body_mid}
"""
    write("forecast.html", style.page("Forecasts — Pickleball, Priced",
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
<h1>Recent results, priced</h1>
<p class="sub">Every pro doubles game of the last {days} days with the win
probability the model would have quoted for the eventual winners before the
game (current monthly values — a living retrospective, not a frozen
commitment; the <a href="receipts.html">receipts page</a> holds those).
Winners listed first. {n_upsets} upsets (winner priced under 25%).</p>
<div class="tblwrap"><table><tr><th>event</th><th></th><th>winners</th>
<th>over</th><th class="num">score</th><th class="num">winner was priced</th></tr>
{''.join(rows)}</table></div>
"""
    write("results.html", style.page("Results — Pickleball, Priced",
                                     body, "results.html", "", updated))


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
("based on public results data via Pickleball, Priced"). Player identity is
by UUID — names collide (there are three Kawamotos; two are twins).</p>
<div class="tblwrap"><table><tr><th>file</th><th class="num">rows</th>
<th class="num">size</th><th>contents</th></tr>{''.join(rows)}</table></div>
<p class="note">Model values are on a per-point logit scale; the site
displays them as expected margin vs an average pairing via the race DP
(<a href="methods.html">methods</a>). DreamBreakers and forfeits are
excluded from games.csv-derived stats. Refreshed with the nightly build;
data through {updated}.</p>
"""
    write("data.html", style.page("Open data — Pickleball, Priced",
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
Curve measured out-of-sample: every post-June-2026 game priced by the frozen
pre-June fit ({CAL["fit_on"]["n_games"]} games). The fitted correction is
nearly the identity (slope {CAL["b"]:.2f}) — the earlier v1 model ran
underconfident, this one doesn't. What the data does insist on: across all
36k games, favorites priced above 99% still lost {CAL["tail"]["losses"]} of
{CAL["tail"]["n_extreme"]} games (~1%), so site probabilities are floored —
nothing is ever 0% or 100%. Frozen predictions in the ledger above are
graded exactly as committed, never re-calibrated after the fact.</p>
{charts.calibration_chart(cal)}
"""
    write("receipts.html", style.page("Receipts — Pickleball, Priced",
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
<p class="note">Games the model prices lowest for the eventual winners, using
each month's values (retrospective pricing, all four players tracked;
calibrated probabilities — no price ever reaches 0%, because across 36k
games about 1 in 100 "sure things" lost anyway). Games with recorded
mid-match player swaps and raw prices under 0.2% are excluded as probable
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
    write("records.html", style.page("Record book — Pickleball, Priced",
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
    for gender, label in (("M", "Men"), ("F", "Women")):
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
    write("dupr.html", style.page("DUPR × model — Pickleball, Priced",
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
    write("methods.html", style.page("Methods — Pickleball, Priced",
                                     body, "methods.html", "", updated))


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

    print("pages: rankings, forecasts, results, simulator, receipts, records, dupr, methods, data …")
    build_rankings(players, updated, len(games))
    build_player_index(players, updated)
    build_forecast(players, updated)
    build_results(players, games, updated)
    build_simulator(players, updated)
    build_receipts(updated)
    build_records(players, agg, games, updated)
    build_dupr(players, updated)
    build_methods(updated)
    build_downloads(games, updated)

    dyn = [p for p in players.values() if p.dynamic and p.stats]
    print(f"player pages: {len(dyn)} …")
    for p in dyn:
        build_player_page(p, players, chem, updated)
    n = sum(1 for _ in SITE.rglob("*.html"))
    print(f"done: {n} pages in {SITE}")


if __name__ == "__main__":
    main()
