"""Page shell + the one stylesheet.

Palette follows the repo dataviz conventions: categorical blue (model/
primary) and aqua (secondary series), diverging blue<->red for win/loss
polarity, reserved status green/red for HIT/MISS (always paired with a
text label, never color alone).  Light and dark are both first-class.
"""
from __future__ import annotations

CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #1baf7a; --loss: #e34948;
  --good: #006300; --bad: #d03b3b; --warn: #a36b00;
  --band: rgba(42,120,214,0.14); --wash: rgba(11,11,11,0.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #199e70; --loss: #e66767;
    --good: #0ca30c; --bad: #e66767; --warn: #c98500;
    --band: rgba(57,135,229,0.20); --wash: rgba(255,255,255,0.05);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--s1); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1000px; margin: 0 auto; padding: 0 16px 48px; }
header.site { border-bottom: 1px solid var(--border); background: var(--surface); }
header.site .wrap { display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 4px 18px; padding: 12px 16px; }
header.site .brand { font-weight: 700; font-size: 17px; color: var(--ink); }
header.site nav { display: flex; flex-wrap: wrap; gap: 2px 14px; }
header.site nav a { color: var(--ink2); font-size: 14px; }
header.site nav a.here { color: var(--ink); font-weight: 600; }
h1 { font-size: 24px; margin: 26px 0 4px; }
h2 { font-size: 18px; margin: 30px 0 8px; }
h3 { font-size: 15px; margin: 20px 0 6px; }
p.sub, .sub { color: var(--ink2); margin-top: 2px; }
.note { color: var(--ink2); font-size: 13px; }
.small { font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 16px 0; }
.tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; }
.tile .v { font-size: 22px; font-weight: 700; }
.tile .k { font-size: 12px; color: var(--ink2); margin-top: 2px; }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
.tblwrap { overflow-x: auto; }
th { text-align: left; color: var(--ink2); font-weight: 600; font-size: 12.5px;
  border-bottom: 1px solid var(--baseline); padding: 6px 8px; white-space: nowrap; }
td { padding: 5px 8px; border-bottom: 1px solid var(--grid); vertical-align: middle; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.gray, .nowrap { white-space: nowrap; }
svg.chart.capped { max-width: 470px; }
tr:hover td { background: var(--wash); }
.chip { display: inline-block; border-radius: 999px; padding: 1px 9px;
  font-size: 12px; font-weight: 600; border: 1px solid var(--border); }
.chip.hit { color: var(--good); }
.chip.miss { color: var(--bad); }
.chip.pending { color: var(--muted); }
.chip.void { color: var(--muted); }
.up { color: var(--good); } .down { color: var(--bad); } .flat { color: var(--muted); }
.gray { color: var(--muted); }
svg.chart { width: 100%; height: auto; display: block; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; margin: 8px 0; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .zeroline { stroke: var(--baseline); stroke-width: 1.5; }
svg .axis { fill: var(--muted); font-size: 11px; font-family: inherit; }
svg .ptlabel { fill: var(--ink2); font-size: 11px; font-family: inherit; }
svg .legend text { fill: var(--ink2); font-size: 11px; font-family: inherit; }
svg .band { fill: var(--band); }
svg .s1line { stroke: var(--s1); stroke-width: 2; fill: none; }
svg .s2line { stroke: var(--s2); stroke-width: 2; fill: none; }
svg .s1dot { fill: var(--s1); }
svg .windot { fill: var(--s1); }
svg .lossdot { fill: var(--loss); }
svg .scdot { fill: var(--s1); fill-opacity: 0.55; }
svg .scdot:hover { fill-opacity: 1; }
svg .hoverdot { fill: transparent; }
svg .hoverdot:hover { fill: var(--s1); fill-opacity: 0.35; }
svg .hoverdot2 { fill: transparent; }
svg .hoverdot2:hover { fill: var(--s2); fill-opacity: 0.5; }
svg.ivl { display: inline-block; vertical-align: middle; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
footer.site { border-top: 1px solid var(--border); margin-top: 40px;
  padding: 14px 0; color: var(--ink2); font-size: 13px; }
input, select, button {
  font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--baseline); border-radius: 6px; padding: 6px 8px; }
button { cursor: pointer; font-weight: 600; }
.searchbox { width: 100%; max-width: 360px; margin: 8px 0; }
.big { font-size: 34px; font-weight: 800; }
.pmbar { height: 14px; border-radius: 7px; overflow: hidden; display: flex;
  border: 1px solid var(--border); }
.pmbar .a { background: var(--s1); } .pmbar .b { background: var(--loss); }
"""

NAV = [("index.html", "Rankings"), ("players/index.html", "Players"),
       ("forecast.html", "Forecasts"), ("results.html", "Results"),
       ("simulator.html", "Simulator"), ("receipts.html", "Receipts"),
       ("records.html", "Record book"), ("dupr.html", "DUPR × model"),
       ("methods.html", "Methods"), ("data.html", "Data")]


def page(title, body, here="", root="", updated=""):
    nav = "".join(
        '<a href="%s%s"%s>%s</a>' % (root, h, ' class="here"' if h == here else "", t)
        for h, t in NAV)
    foot_updated = f" · data through {updated}" if updated else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<link rel="stylesheet" href="{root}assets/style.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ccircle cx='8' cy='8' r='7' fill='%232a78d6'/%3E%3Ccircle cx='5.5' cy='6' r='1.3' fill='%23fcfcfb'/%3E%3Ccircle cx='10.5' cy='6' r='1.3' fill='%23fcfcfb'/%3E%3Ccircle cx='8' cy='10.5' r='1.3' fill='%23fcfcfb'/%3E%3C/svg%3E">
<header class="site"><div class="wrap">
  <span class="brand"><a href="{root}index.html" style="color:inherit">Pickleball, Priced</a></span>
  <nav>{nav}</nav>
</div></header>
<div class="wrap">
{body}
<footer class="site">Unofficial fan analytics based on public results data —
not affiliated with any tour. Every number that has an error bar shows it;
cross-gender rankings are never published as fact (<a href="{root}methods.html">why</a>).
Model: Bayesian, 36k games, validated 77.4% winner accuracy on 884 unseen
games{foot_updated}.</footer>
</div>
"""
