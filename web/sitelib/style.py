"""Page shell + the one stylesheet.

The CSS is the design-bundle port (PICKLES green-bar printout theme,
design_handoff_pickles_site/site/assets/style.css, ported verbatim per the
handoff) followed by a generator-side section for the landing page, which
existed only as an inline-styled mock.  Light and dark are both first-class
via the custom properties at the top; the "printout" artifacts (credential
card, ticket, tonight band) intentionally keep fixed paper/ink colors in
both themes — they read as physical objects, not chrome.
"""
from __future__ import annotations

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

/* ============================================================
   PICKLES — green-bar printout theme
   Drop-in replacement for the CSS string in web/sitelib/style.py.
   All original selectors kept; light + dark both defined.
   ============================================================ */

:root {
  --page: #f2f7e5; --surface: #fbfdf3; --ink: #16321e; --ink2: #46603a;
  --muted: #7c8f6d; --grid: #dbe7c2; --baseline: #a9bc8c;
  --border: rgba(22,50,30,0.18);
  --s1: #1e7a3c; --s2: #c05621; --loss: #e03a2f;
  --good: #1e7a3c; --bad: #d03b3b; --warn: #a36b00;
  --band: rgba(30,122,60,0.16); --wash: rgba(22,50,30,0.05);
  --stripe: rgba(30,122,60,0.055);
  --hl: #d9f154; --hl-ink: #16321e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0e1410; --surface: #17211a; --ink: #edf4e0; --ink2: #c2d3a8;
    --muted: #8aa07a; --grid: #26332a; --baseline: #3a4a3c;
    --border: rgba(237,244,224,0.14);
    --s1: #cfe94f; --s2: #e8935a; --loss: #e66767;
    --good: #7fc24a; --bad: #e66767; --warn: #d9a03f;
    --band: rgba(217,241,84,0.16); --wash: rgba(237,244,224,0.06);
    --stripe: rgba(217,241,84,0.05);
    --hl: #d9f154; --hl-ink: #16321e;
  }
}
* { box-sizing: border-box; }
::selection { background: var(--hl); color: var(--hl-ink); }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 "Space Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--s1); text-decoration: none; }
a:hover { text-decoration: underline; text-decoration-thickness: 2px;
  text-decoration-color: var(--hl); }
.wrap { max-width: 1000px; margin: 0 auto; padding: 0 16px 48px; }
header.site { border-bottom: 3px double var(--ink); background: var(--surface); }
header.site .wrap { display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 4px 18px; padding: 12px 16px; }
header.site .brand { font-family: "Space Mono", ui-monospace, monospace;
  font-weight: 700; font-size: 16px; background: var(--hl); color: var(--hl-ink);
  padding: 2px 9px; letter-spacing: 0.01em; }
header.site .brand a { color: inherit; }
header.site nav { display: flex; flex-wrap: wrap; gap: 2px 16px; }
header.site nav a { color: var(--ink2); font-family: "Space Mono", ui-monospace, monospace;
  font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.05em; }
header.site nav a:hover { text-decoration: none; color: var(--ink); }
header.site nav a.here { color: var(--ink); font-weight: 700;
  border-bottom: 3px solid var(--hl); }
h1 { font-family: "Space Mono", ui-monospace, monospace; font-size: 23px;
  letter-spacing: 0.01em; margin: 28px 0 4px; }
h1::before { content: "> "; color: var(--s1); }
h2 { font-family: "Space Mono", ui-monospace, monospace; font-size: 16px;
  text-transform: uppercase; letter-spacing: 0.09em; margin: 32px 0 8px;
  border-bottom: 3px double var(--baseline); padding-bottom: 5px; }
h3 { font-family: "Space Mono", ui-monospace, monospace; font-size: 13.5px;
  text-transform: uppercase; letter-spacing: 0.07em; margin: 20px 0 6px; }
p.sub, .sub { color: var(--ink2); margin-top: 2px; }
.note { color: var(--ink2); font-size: 13px; }
.small { font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 16px 0; }
.tile { background: var(--surface); border: 1.5px solid var(--border);
  border-radius: 0; padding: 10px 12px; }
.tile .v { font-family: "Space Mono", ui-monospace, monospace;
  font-size: 22px; font-weight: 700; }
.tile .k { font-family: "Space Mono", ui-monospace, monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--ink2); margin-top: 3px; }
.card { background: var(--surface); border: 1.5px dashed var(--baseline);
  border-radius: 0; padding: 14px 16px; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
.tblwrap { overflow-x: auto; }
th { text-align: left; color: var(--ink2); font-weight: 700;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.07em;
  border-bottom: 2px solid var(--ink); padding: 6px 8px; white-space: nowrap; }
td { padding: 5px 8px; border-bottom: 1px solid var(--grid); vertical-align: middle; }
tr:nth-child(even) td { background: var(--stripe); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.num { font-family: "Space Mono", ui-monospace, monospace; font-size: 13px; }
td.gray, .nowrap { white-space: nowrap; }
svg.chart.capped { max-width: 470px; }
tr:hover td { background: var(--wash); }
.chip { display: inline-block; border-radius: 0; padding: 1px 8px;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 11.5px;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
  border: 1.5px solid var(--border); }
.chip.hit { color: var(--good); border-color: var(--good); }
.chip.miss { color: var(--bad); border-color: var(--bad); }
.chip.pending { color: var(--muted); }
.chip.void { color: var(--muted); }
.up { color: var(--good); } .down { color: var(--bad); } .flat { color: var(--muted); }
.gray { color: var(--muted); }
svg.chart { width: 100%; height: auto; display: block; background: var(--surface);
  border: 1.5px solid var(--border); border-radius: 0; margin: 8px 0; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .zeroline { stroke: var(--baseline); stroke-width: 1.5; }
svg .axis { fill: var(--muted); font-size: 10px;
  font-family: "Space Mono", ui-monospace, monospace; }
svg .ptlabel { fill: var(--ink2); font-size: 10px;
  font-family: "Space Mono", ui-monospace, monospace; }
svg .legend text { fill: var(--ink2); font-size: 10px;
  font-family: "Space Mono", ui-monospace, monospace; }
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
footer.site { border-top: 3px double var(--baseline); margin-top: 40px;
  padding: 14px 0; color: var(--ink2);
  font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px; }
input, select, button {
  font: inherit; font-family: "Space Mono", ui-monospace, monospace; font-size: 14px;
  color: var(--ink); background: var(--surface);
  border: 1.5px solid var(--baseline); border-radius: 0; padding: 6px 8px; }
button { cursor: pointer; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; border: 2px solid var(--ink); }
button:hover { background: var(--hl); color: var(--hl-ink); border-color: var(--hl-ink); }
.searchbox { width: 100%; max-width: 360px; margin: 8px 0; }
.big { font-family: "Space Mono", ui-monospace, monospace; font-size: 34px;
  font-weight: 700; background: var(--hl); color: var(--hl-ink);
  padding: 0 8px; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.pmbar { height: 14px; border-radius: 0; overflow: hidden; display: flex;
  border: 1.5px solid var(--border); }
.pmbar .a { background: var(--s1); } .pmbar .b { background: var(--loss); }

/* ============================================================
   Structural mockup additions — rankings page
   (masthead, system-check ledger, house rule, section tags,
    "who is actually #1" probability panels)
   ============================================================ */
h1.runtitle { font-family: Anton, "Space Mono", sans-serif; font-weight: 400;
  font-size: 46px; text-transform: uppercase; letter-spacing: 0.02em;
  margin: 30px 0 2px; }
.runmeta { font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px;
  color: var(--muted); letter-spacing: 0.06em; margin: 0 0 8px; }
.syscheck { background: var(--surface); border: 1.5px solid var(--border);
  padding: 12px 16px; margin: 16px 0;
  font-family: "Space Mono", ui-monospace, monospace; }
.syscheck .lrow { display: flex; align-items: baseline; gap: 10px; padding: 3px 0; }
.syscheck .lk { font-size: 12.5px; letter-spacing: 0.04em; color: var(--ink2); }
.syscheck .ldot { flex: 1; border-bottom: 2px dotted var(--baseline);
  transform: translateY(-4px); min-width: 24px; }
.syscheck .lv { font-size: 14px; font-weight: 700; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
.syscheck .lcmp { color: var(--muted); font-weight: 400; font-size: 12px; }
.houserule { border: 1.5px solid var(--baseline); background: var(--wash);
  padding: 10px 14px; font-size: 13.5px; color: var(--ink2); margin: 14px 0; }
.hrtag { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 10.5px; background: var(--ink); color: var(--page);
  padding: 2px 8px; letter-spacing: 0.09em; margin-right: 9px;
  display: inline-block; }
.secno { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 10.5px; background: var(--hl); color: var(--hl-ink);
  padding: 3px 8px; letter-spacing: 0.09em; margin-right: 10px;
  display: inline-block; transform: translateY(-2px); }
.p1card .p1head { margin-bottom: 2px; }
.p1kick { font-size: 15px; font-weight: 600; margin: 4px 0 10px; }
.p1row { display: grid; grid-template-columns: 158px 1fr 52px minmax(140px, auto);
  gap: 12px; align-items: center; padding: 3px 0; font-size: 14px; }
.p1name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-weight: 600; }
.p1bar { display: block; height: 15px; background: var(--wash);
  border: 1px solid var(--border); }
.p1fill { display: block; height: 100%; background: var(--s1); }
.p1pct { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 14px; text-align: right; white-space: nowrap; }
.p1meta { font-family: "Space Mono", ui-monospace, monospace; font-size: 11.5px;
  color: var(--muted); white-space: nowrap; }
@media (max-width: 640px) {
  .p1row { grid-template-columns: 1fr 52px; }
  .p1bar { grid-column: 1 / -1; }
  .p1meta { grid-column: 1 / -1; padding-bottom: 4px; }
}

/* ============================================================
   Landing page (index.html) — generator-side additions.
   The landing mock is inline-styled; these classes recreate it.
   Only the CTA colors need theme-specific tokens; printout
   artifacts keep literal paper colors in both themes.
   ============================================================ */
:root { --cta-bg: #16321e; --cta-ink: #d9f154; --cta-hbg: #1e7a3c; --cta-hink: #f2f7e5; }
@media (prefers-color-scheme: dark) {
  :root { --cta-bg: #d9f154; --cta-ink: #16321e; --cta-hbg: #edf4e0; --cta-hink: #16321e; }
}
.lsec { max-width: 1080px; margin: 0 auto; padding: 0 24px; }
.lsec.inside { padding-top: 40px; padding-bottom: 56px; }
.lsec.check { padding-top: 48px; }
.lsec.notes { padding-top: 24px; padding-bottom: 56px; }
.lsec.slate { padding-top: 18px; }
.slatebox { background: var(--surface); border: 1.5px solid var(--border);
  padding: 12px 16px 10px; }
.slatehead { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  margin-bottom: 8px; }
.slateevent { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 12.5px; letter-spacing: 0.06em; color: var(--ink2); }
.slatehead .fill { flex: 1; }
.slatelink { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 12px; letter-spacing: 0.05em; white-space: nowrap; }
.slaterows { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 4px 28px; }
a.srow { display: flex; align-items: baseline; gap: 8px; padding: 2px 0;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px;
  color: var(--ink); }
a.srow .st { color: var(--muted); font-size: 11.5px; white-space: nowrap; }
a.srow .sm { font-weight: 700; white-space: nowrap; }
a.srow .lead { flex: 1; border-bottom: 2px dotted var(--baseline);
  transform: translateY(-3px); min-width: 12px; }
a.srow .sp { font-weight: 700; white-space: nowrap; }
.slatefoot { border-top: 1px solid var(--grid); margin-top: 8px; padding-top: 7px;
  display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 12px;
  letter-spacing: 0.04em; color: var(--ink2); }
header.landing { border-bottom: 3px double var(--ink); background: var(--surface); }
header.landing .bar { max-width: 1080px; margin: 0 auto; padding: 14px 24px;
  display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap; }
.brandchip { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 17px; background: var(--hl); color: var(--hl-ink); padding: 3px 10px; }
.brandsub { font-family: "Space Mono", ui-monospace, monospace; font-size: 12px;
  color: var(--muted); letter-spacing: 0.04em; }
header.landing nav { display: flex; gap: 18px; margin-left: auto; flex-wrap: wrap; }
header.landing nav a { font-family: "Space Mono", ui-monospace, monospace;
  font-size: 12.5px; letter-spacing: 0.05em; color: var(--ink2); }
.hero { display: grid; grid-template-columns: 1.25fr 1fr; gap: 56px;
  align-items: center; padding-top: 44px; padding-bottom: 40px; }
.hero .copy { display: flex; flex-direction: column; gap: 20px; align-items: flex-start; }
.kicker { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 13px; letter-spacing: 0.14em; color: var(--s1); }
.hero h1 { font-family: Anton, "Space Mono", sans-serif; font-weight: 400;
  font-size: 88px; line-height: 0.94; margin: 0; text-transform: uppercase;
  letter-spacing: 0.01em; }
.hero h1::before { content: none; }
.hero .lede { font-size: 19px; line-height: 1.55; margin: 0; max-width: 30em; }
.ctas { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 6px; }
a.cta { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 14px; letter-spacing: 0.06em; }
a.cta:hover { text-decoration: none; }
a.cta.solid { background: var(--cta-bg); color: var(--cta-ink); padding: 12px 20px; }
a.cta.solid:hover { background: var(--cta-hbg); color: var(--cta-hink); }
a.cta.outline { border: 2px solid var(--ink); color: var(--ink); padding: 10px 20px; }
a.cta.outline:hover { background: var(--hl); color: var(--hl-ink); }
a.cta.bright { background: #d9f154; color: #16321e; padding: 12px 20px; }
a.cta.bright:hover { background: #edf4e0; color: #16321e; }
.printout { background: repeating-linear-gradient(180deg, #fbfdf3 0 40px, #e9f2d4 40px 80px);
  border: 1.5px solid rgba(22,50,30,0.25); padding: 28px 30px; color: #16321e;
  font-family: "Space Mono", ui-monospace, monospace;
  box-shadow: 0 18px 34px rgba(22,50,30,0.14); transform: rotate(1.2deg); }
.printout .plabel { font-size: 13px; letter-spacing: 0.08em; }
.printout .headline { display: flex; align-items: baseline; gap: 10px; margin: 10px 0 4px; }
.hlnum { position: relative; display: inline-block; }
.hlnum .swipe { position: absolute; left: -10px; right: -10px; top: 20%; bottom: 6%;
  background: #d9f154; transform: skew(-6deg); }
.hlnum .val { position: relative; font-weight: 700; font-size: 84px; line-height: 1; }
.printout .whose { font-weight: 700; font-size: 20px; }
.printout .cmprow { display: flex; align-items: baseline; gap: 8px; font-size: 15px;
  margin-top: 14px; }
.printout .cmprow .lead { flex: 1; border-bottom: 2px dotted #a9bc8c;
  transform: translateY(-4px); }
.printout .cmprow .v { font-weight: 700; font-size: 20px; }
.printout .prule { border-top: 3px double #a9bc8c; margin: 16px 0 10px; }
.printout .pnote { font-size: 12.5px; line-height: 1.6; color: #46603a; }
.lh2 { font-size: 15px; margin: 0 0 20px; padding-bottom: 8px; }
.doorgrid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }
a.door { color: var(--ink); background: var(--surface); border: 1.5px dashed var(--baseline);
  padding: 20px 22px; display: flex; flex-direction: column; gap: 12px; }
a.door:hover { border-color: var(--ink); text-decoration: none; }
.doortag { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 12px; letter-spacing: 0.09em; background: var(--hl); color: var(--hl-ink);
  padding: 2px 8px; align-self: flex-start; }
.doorblurb { font-size: 14.5px; line-height: 1.5; margin: 0; color: var(--ink2); }
.t-bars { display: flex; flex-direction: column; gap: 5px; }
.t-bar { display: grid; grid-template-columns: 74px 1fr 38px; gap: 8px; align-items: center;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 11.5px; }
.t-bar .nm { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.t-bar .track { display: block; height: 11px; background: var(--wash);
  border: 1px solid var(--border); }
.t-bar .fill { display: block; height: 100%; background: var(--s1); }
.t-bar .pv { font-weight: 700; text-align: right; }
.t-rows { display: flex; flex-direction: column; gap: 6px;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px; }
.t-rows.wide { gap: 7px; }
.t-row { display: flex; align-items: baseline; gap: 8px; }
.t-row .lead { flex: 1; border-bottom: 2px dotted var(--baseline);
  transform: translateY(-3px); }
.t-row .call, .t-row .res { white-space: nowrap; }
.t-row .res { font-weight: 700; }
.t-row .res.hitv { color: var(--good); }
.t-row .res.missv { color: var(--loss); }
.t-term { font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px;
  line-height: 2; }
.t-term .result { background: var(--hl); color: var(--hl-ink); font-weight: 700;
  padding: 0 6px; }
section.tonight { background: #16321e; color: #edf4e0; }
.tonight .inner { max-width: 1080px; margin: 0 auto; padding: 64px 24px;
  display: grid; grid-template-columns: 1fr 1.1fr; gap: 56px; align-items: center; }
.tonight .copy { display: flex; flex-direction: column; gap: 18px; align-items: flex-start; }
.tonight .kicker { color: #d9f154; }
.tonight h2 { font-family: Anton, "Space Mono", sans-serif; font-weight: 400;
  font-size: 54px; line-height: 1; margin: 0; text-transform: uppercase;
  letter-spacing: normal; border-bottom: none; padding-bottom: 0; }
.tonight .blurb { font-size: 18px; line-height: 1.55; margin: 0; max-width: 26em;
  color: #c2d3a8; }
.tonight a { color: inherit; }
.dbrow { display: flex; align-items: baseline; gap: 12px; width: 100%; max-width: 380px;
  font-family: "Space Mono", ui-monospace, monospace; }
.dbrow .k { font-size: 13px; letter-spacing: 0.06em; color: #8aa07a; }
.dbrow .lead { flex: 1; border-bottom: 2px dotted #3a4a3c; transform: translateY(-4px); }
.dbrow .v { font-weight: 700; font-size: 18px; color: #d9f154; }
.ticketwrap { justify-self: center; transform: rotate(1.4deg); display: flex;
  filter: drop-shadow(0 22px 38px rgba(0,0,0,0.45)); max-width: 100%; }
.tractor { width: 34px; flex: none; background: #edf4e0;
  background-image: radial-gradient(circle at 17px 19px, #16321e 6px, transparent 7px);
  background-size: 34px 38px; }
.ticket { width: 430px; max-width: 100%;
  background: repeating-linear-gradient(180deg, #f7fbee 0 36px, #e2efcb 36px 72px);
  color: #16321e; padding: 24px 28px;
  font-family: "Space Mono", ui-monospace, monospace; }
.ticket .th { font-weight: 700; font-size: 16px; }
.ticket .trun { font-size: 12px; margin-top: 2px; }
.ticket .tok { font-size: 12px; font-weight: 700; margin-top: 2px; }
.ticket .trule { border-top: 3px double #16321e; margin: 12px 0; }
.ticket .rows { display: flex; flex-direction: column; gap: 9px; }
.ticket .trow { display: flex; align-items: baseline; gap: 8px; }
.ticket .trow .g { font-weight: 700; font-size: 14px; white-space: nowrap; }
.ticket .trow .lead { flex: 1; border-bottom: 2px dotted #16321e;
  transform: translateY(-3px); }
.ticket .trow .p { font-weight: 700; font-size: 16px; white-space: nowrap; }
.ticket .ttotal { display: flex; align-items: baseline; gap: 8px; }
.ticket .ttotal .k { font-weight: 700; font-size: 18px; }
.ticket .ttotal .lead { flex: 1; border-bottom: 3px dotted #16321e;
  transform: translateY(-4px); }
.ticket .ttotal .v { font-weight: 700; font-size: 24px; }
.ticket .tdog { font-size: 12px; margin-top: 4px; }
.fnotes { display: grid; grid-template-columns: 1fr 1fr; gap: 0 48px; }
.fnote { display: flex; align-items: baseline; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid var(--grid); }
.fnote .no { font-family: "Space Mono", ui-monospace, monospace; font-weight: 700;
  font-size: 12px; color: var(--s1); white-space: nowrap; }
.fnote .claim { font-size: 15.5px; font-weight: 600; }
.fnote .lead { flex: 1; border-bottom: 2px dotted var(--baseline);
  transform: translateY(-4px); min-width: 20px; }
.fnote .ev { font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px;
  color: var(--ink2); white-space: nowrap; }
footer.manifesto { border-top: 3px double var(--ink); background: var(--surface); }
footer.manifesto .inner { max-width: 1080px; margin: 0 auto; padding: 48px 24px 20px;
  display: flex; flex-direction: column; gap: 28px; }
.credo { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.credo .slogan { font-family: Anton, "Space Mono", sans-serif; font-size: 46px;
  text-transform: uppercase; line-height: 1; }
.credo .slogan .hl { background: var(--hl); color: var(--hl-ink); padding: 0 10px; }
.credo .lines { font-family: "Space Mono", ui-monospace, monospace; font-size: 13px;
  line-height: 1.8; color: var(--ink2); }
.manifesto .baseline { border-top: 1.5px solid var(--grid); padding-top: 14px;
  display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  font-family: "Space Mono", ui-monospace, monospace; font-size: 12.5px;
  color: var(--ink2); }
@media (max-width: 720px) {
  .hero { grid-template-columns: 1fr; gap: 40px; }
  .hero h1 { font-size: 52px; }
  .hlnum .val { font-size: 64px; }
  .doorgrid { grid-template-columns: 1fr; }
  .tonight .inner { grid-template-columns: 1fr; padding: 48px 24px; gap: 40px; }
  .tonight h2 { font-size: 40px; }
  .fnotes { grid-template-columns: 1fr; }
  .credo .slogan { font-size: 34px; }
  .ticket { width: auto; flex: 1; min-width: 0; }
}
"""

# Pickleball favicon in the PICKLES palette: chartreuse ball, ink holes.
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
           "viewBox='0 0 16 16'%3E%3Ccircle cx='8' cy='8' r='7' fill='%23d9f154'/%3E"
           "%3Ccircle cx='5.5' cy='6' r='1.3' fill='%2316321e'/%3E"
           "%3Ccircle cx='10.5' cy='6' r='1.3' fill='%2316321e'/%3E"
           "%3Ccircle cx='8' cy='10.5' r='1.3' fill='%2316321e'/%3E%3C/svg%3E")

FONTS_PRECONNECT = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>"""

NAV = [("rankings.html", "Rankings"), ("players/index.html", "Players"),
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
{FONTS_PRECONNECT}
<link rel="stylesheet" href="{root}assets/style.css">
<link rel="icon" href="{FAVICON}">
<header class="site"><div class="wrap">
  <span class="brand"><a href="{root}index.html">PICKLES</a></span>
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
