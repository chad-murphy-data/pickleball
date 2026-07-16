# Site snapshot — for design

A committed snapshot of the generated website (16 representative pages of
the 509 the generator produces). Open `index.html` directly in a browser —
everything is relative-linked and works from the filesystem, no server
needed. Regenerated snapshots land here occasionally; the live site (once
GitHub Pages is enabled) always has the current full build.

## How to work on these — IMPORTANT

These files are **generated** by `web/build_site.py`; direct HTML edits are
overwritten on every rebuild. The durable design surfaces are:

1. **`assets/style.css`** — the entire look lives in this ONE stylesheet
   (colors, layout, tables, cards, chart styling). Edit it freely against
   these pages; the master copy is the `CSS` string in
   `web/sitelib/style.py` and your edited file can be ported back verbatim.
   Light AND dark mode are both defined at the top via CSS custom
   properties — please keep both working (toggle with your OS/browser
   theme).
2. **Structural/markup changes** (different card layouts, new sections,
   reordered columns): mock them up on these files and hand them back —
   we port the structure into the generator.

## Constraints that are content decisions, not styling accidents

- Every number that has an error bar shows it: intervals, ± values and
  "(noise)" chips are the brand, not clutter — restyle them, don't drop
  them.
- Men's and women's rankings are separate sections on purpose and must
  never merge into one list.
- No probability is ever displayed as 0% or 100% (you'll see `<1%` and
  `>99%`).
- Charts are inline SVG styled by CSS classes in the same stylesheet
  (`svg .s1line`, `.band`, `.windot`, …) so they follow theme changes.

## What's in the snapshot

- `index.html` — power rankings (the "Who is actually #1?" panel + tables)
- `forecast.html` — upcoming-matchup cards (probability bars, path notes)
- `results.html` — recent games with prices + UPSET chips
- `simulator.html` — interactive matchup simulator (works offline)
- `receipts.html` — the prediction ledger + calibration chart
- `records.html`, `dupr.html`, `methods.html`, `data.html`
- `players/` — six representative player pages (Waters, Tardio, Johns,
  Bright, Patriquin, Jackie Kawamoto — the last one shows the DUPR
  data-glitch footnote state)

Note: links to player pages not included in the snapshot will 404 here;
they exist in the full build.
