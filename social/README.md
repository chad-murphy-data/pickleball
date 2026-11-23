# social/ — the evening forecast carousel

Every evening, price tomorrow's scheduled pro matches, render a slide
deck in the PICKLES look, and post it to r/PickleballStats as a gallery
with the numbers written out in the body.  Numbers are the live board's
pre-match numbers exactly (same code paths); no editorial voice anywhere.

```bash
python social/run.py                     # tomorrow: slate + slides + post.md (no posting)
python social/run.py --date 2026-09-06   # any date (today's or a past day works for testing)
python social/run.py --post --dry-run    # everything except the Reddit submit
python social/run.py --post              # the real thing (needs the REDDIT_* env below)
```

Outputs go to `social/out/<date>/` (gitignored):
`slate.json` (every match + probability), `slide-NN.png` (the deck),
`deck.json` (captions), `post.md` (title + body), `posted.json` after a
real post.

## Pieces

| file | job |
|---|---|
| `slate.py` | Tomorrow's matches from the open BFF, priced. PPA doubles = v2 + weakest link + race DP + calibration per game from the real score format + best-of tree; PPA singles = singles suite integrated over sd; MLP = `make_forecast` lineups/tree. Sides still TBD are priced against each possible opponent from the pending feeder match. |
| `render.py` | Slate → cover + one slide per bracket (Men's/Women's/Mixed Doubles, Men's/Women's Singles, MLP Matchups) + fixed methods slide. 1080×1350 PNG via headless Chromium, fonts vendored. |
| `text.py` | Title + markdown body + per-image captions. |
| `post_reddit.py` | Gallery submit with body text (PRAW 8), duplicate-safe per date. |
| `run.py` | The one-shot orchestrator the workflow calls. |
| `templates/slide.html` | **The template** — the Claude Design handoff (`design_handoff_pickles_carousel`) ported token for token: colors, Anton/Space Mono/Space Grotesk sizes, card chrome, the close-match rule (favorite under 65% → off-white number + dashed 50% mark), SCENARIO A/B cards, score-distribution block, methodology blocks. Restyle here. |

## Schedule

`.github/workflows/social.yml` runs at 03:30 UTC nightly (8:30 PM PDT).
Change the cron line to move the posting time.  Quiet days print
"nothing scheduled" and stop; a deck with zero priced matches (bracket
not published yet) is rendered but not posted.  Manual runs from the
Actions tab take a date and a post/dry-run toggle and always upload the
PNGs + post.md as an artifact for review.

Verified 2026-09-05 at 7:34 PM PT: the BFF already served Sunday's finals
with the decided sides filled in, so an 8:30 PM run sees tomorrow's
bracket; a side whose semi is still on the court is carried as TBD with
one number per possible opponent.

## Score formats

Best-of and points-per-game come from the API per bracket round (one
`getResultMatchInfos` lookup per (event, bracket side, round) group),
so the series tree and the "best-of-N" footer follow whatever
pickleball.com serves.  If the tour changes a rule before the API does,
`SOCIAL_PPA_BEST_OF=3` forces every PPA match to that length (as a repo
variable it flows into the workflow; drop it once the API catches up).

## Reddit credentials

1. reddit.com/prefs/apps → create app → type **script**; redirect URI
   can be `http://localhost:8080`.  Note the client id (under the app
   name) and secret.
2. Repo Settings → Secrets → Actions: `REDDIT_CLIENT_ID`,
   `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD` (the
   posting account; 2FA must be off for password auth on script apps).
3. Optional repo *variables*: `SOCIAL_SUBREDDIT` (default
   `PickleballStats`), `SOCIAL_FLAIR_ID` (a flair template id from the
   sub's mod tools).

Locally, export the same names and run `python social/run.py --post
--dry-run` first.

## Template contract

`render.py` replaces `{{FONT_CSS}}`, `{{KIND}}` (`cover` / `bracket` /
`methods`, set as a class on `.slide`), `{{HDR_RIGHT}}`, `{{BODY}}`,
`{{FTR_LEFT}}`, `{{FTR_CLASS}}`, `{{FTR_RIGHT}}` in `templates/slide.html`.
The body markup uses these classes, so a restyle is a CSS edit:

- cover: `.kicker`, `h1` (with `.hl` highlighted word), `.sub`, `.chips > .chip`
- bracket: `.kicker`, `h1`, `.cards[.compact|.list] > .card[.close]` with
  `.lbl`, `.top > .fav + .pct`, `.meter > .fill (+ .mid when close)`,
  `.bot > (vs .dog) + .meta`; a lone match also gets `.dist` (score
  distribution); a TBD opponent becomes one SCENARIO card per possibility
- methods: `.blocks > .block > .k + .t` (copy in `render.METHODS`; the
  handoff's DUPR comparison line was dropped per the house rule)

Density: ≤3 matches = big cards, 4–6 = `.compact`, 7–9 = `.list`,
more = paginated slides (9 per slide).  Fonts: Anton (display) + Space Mono, vendored
in `templates/fonts/` (OFL).  `PW_CHROMIUM=/path/to/chrome` overrides
the browser binary.
