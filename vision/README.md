# Vision POC — can we read shots off a broadcast?

Phase 6, Tier 0–1. The question this POC answers is narrow on purpose:
**do paddle contacts survive a YouTube broadcast well enough to measure
inter-contact intervals?** Everything else in the vision stack sits on top
of that, so it gets tested first and cheaply.

Attribution — *which* of the four players hit the ball — is deliberately
out of scope here. It is the highest-value field in the eventual schema and
the hardest, and proving it is pointless until contacts are reliable.

## Why this is cheap

Two things the project already owns do most of the work.

**The referee log is the ground truth.** `getListLogs` gives per-rally
start/end timestamps, server, receiver, score and outcome, with 100%
coverage of MLP 2026. So a contact detector can be scored **without hand
labelling**: contacts must fall inside rally windows, cluster in the latter
part of them, collapse to nothing during timeouts, and number 4–14 per
rally. Every one of those is a free validation.

**Audio does the timing.** A paddle strike is a sharp broadband transient.
Onset detection localises it to ~4 ms, where 30fps video gives ±33 ms —
16% of a 200 ms speed-up interval versus 2.5%. And audio needs no
homography, no calibration, no player detection, so it is testable in an
afternoon rather than a month.

Audio can never say *who* hit the ball. That is the division of labour:
**audio is a timing instrument, vision is an attribution instrument.**

## Where to run it

**On a laptop, not the droplet.** YouTube bot-checks datacenter IPs — the
agent sandbox gets `Sign in to confirm you're not a bot`, and a
DigitalOcean box will get the same. Residential internet does not.

```bash
git fetch origin
git checkout claude/pickleball-vision-match-analysis-dow5ds
pip install yt-dlp numpy imageio-ffmpeg     # static ffmpeg, no brew/apt

python vision/run_poc.py --vod chicago-0725
```

That's the whole thing. `--vod chicago-0725` is Chicago Slice v Utah Black
Diamonds, MLP Chicago 2026-07-25
([QOhu67FAeY4](https://www.youtube.com/watch?v=QOhu67FAeY4)) — all four
games start-marked, 193 rallies, **no DreamBreaker**, so every second of
competitive play in the VOD is logged.

## Full-matchup VODs are an upgrade, not a limitation

MLP only publishes whole matchups. That is better than a single game:

- **4× the data.** 193 rallies instead of ~74.
- **The offset is cross-validated four ways.** All games share one absolute
  clock, so a continuous VOD needs *one* offset for all of them. Fitting it
  per game gives four independent estimates that must agree — and if they
  don't, the disagreements locate the broadcast's edit points. A single
  game cannot check itself like this.
- **Better negative controls.** The changeovers between games are minutes of
  genuinely dead time, far stronger than the 1–4 s gaps between rallies.

Verified against the real 4-game timeline with planted contacts: per-game
offsets agreed to **0.22 s** against an estimator resolution of ~1.9 s,
97% of contacts landed inside rally windows, density ratio 13.8×, and the
interval modes came back at 206 ms / 554 ms.

That one command does all four steps and skips anything already done, so
re-running after a crash is cheap. Everything except the download works
anywhere, including here.

Individual steps, if you want them:

```bash
python vision/rally_timeline.py --pick midseason-womens      # the sync spine
python vision/audio_contacts.py --selftest                   # detector sanity
python vision/audio_contacts.py --audio match.m4a --out data/vision/contacts.csv
python vision/poc_report.py --timeline data/vision/rally_timeline_809fe252.csv \
                            --contacts data/vision/contacts.csv
```

## Vet a candidate video BEFORE downloading it

Referee style varies **by match, not by event** — two courts at the same
tournament log differently. So any VOD you find needs its matchup checked:

```bash
python vision/rally_timeline.py --teams "Florida Smash v Bay Area Breakers" \
                                --date 2026-07-08
```

It prints every game in that matchup with `USABLE` or
`degenerate windows`. A matchup VOD contains all four games, so the same
audio can be scored against four independent timelines — four checks from
one download.

The target matchup passes on all four:

| slot | division | rallies | style |
|---|---|---|---|
| 1 | women's | 74 | start-marked (96%, 20 s lead) |
| 2 | men's | 42 | start-marked (91%, 20 s lead) |
| 3 | mixed | 24 | start-marked (100%, 20 s lead) |
| 4 | mixed | 44 | start-marked (93%, 16 s lead) |

## The match

`--pick midseason-womens` → **2026-07-08, Edward Jones Mid-Season
Tournament, women's doubles, Frantova/Weil vs Yeh/Erokhina**, 73 rallies
over 33.5 min.

Chosen on four criteria:
- **women's** — the most dinking, so the interval histogram has the best
  chance of showing two modes
- **73 rallies** — near the densest in MLP 2026, i.e. most contacts per
  minute of video
- **referee style "start-marked"** — see below; this is the one that
  surprised us
- marquee event, so most likely to have a clean featured-court upload

### Referee logging style is bimodal, and it changes the design

Measured across nine MLP 2026 matches, one per event. `log_type 12` is
*supposed* to mark rally start, but its timing is a **workflow artifact**:

| style | rallies with a start marker >2s early | median lead |
|---|---|---|
| **start-marked** — Austin, Chicago, Mid-Season, New York, San Diego | 96–100% | 16–20 s |
| **batch-entered** — Columbus, St. Petersburg, Dallas, St. Louis | 0–5% | 0 s |

In batch-entered matches the referee logs the start and the outcome in the
same second, several rallies at a time. `rally_timeline.py` detects which
style a match uses and switches windowing accordingly — true
`[start, end]` windows when it can, `[previous end, this end]` tiling when
it cannot. **Prefer a start-marked match**: only those leave real dead time
between rallies, which is what lets the density check validate itself.

This is why the first pick (Columbus, the densest women's game at 79
rallies) was dropped — it is batch-entered, and its rally windows are
degenerate. Style also varies *within* an event: the Austin matchup with a
confirmed VOD (Florida Smash v Bay Area Breakers, 2026-06-13) is
batch-entered on all four courts, while a different Austin match the same
day is 96% start-marked. Always vet the specific matchup.

### Finding the video

MLP uploads per-matchup VODs titled `<Team A> v <Team B> at the MLP
<City>…`. Search their channel for the matchup, not the players.

Start-marked matchups worth looking for, best first:

| date | event | matchup | best game |
|---|---|---|---|
| 2026-07-08 | Mid-Season (Grand Rapids) | **Florida Smash v Bay Area Breakers** | women's, 74 rallies |
| 2026-07-25 | Chicago | Chicago Slice v Utah Black Diamonds | mixed, 75 rallies |
| 2026-06-13 | Austin | California Black Bears v SoCal Hard Eights | men's, 81 rallies |
| 2026-06-26 | New York | Florida Smash v Bay Area Breakers | men's, 73 rallies |
| 2026-07-17 | San Diego | Florida Smash v Phoenix Flames | mixed, 66 rallies |

Caution: some MLP uploads titled "MLP Grand Rapids presented by DoorDash"
are from an **earlier season** — they feature teams (FAU Pickleball Club,
D.C Pickleball Team) that do not exist in the 2026 data. Check the teams
against `data/mlp_matchups_2026.csv` before committing to a video.

## What the report tells you

`poc_report.py` prints four numbers plus two free controls:

1. **sync** — offset, and how sharply the alignment curve peaks (flat ≈ 1.0
   means the contacts are not tracking rally structure at all)
2. **density contrast** — contacts/s inside rally windows vs outside
3. **contacts per rally** — median should be 4–14; zeros and 40s are both
   detector failure
4. **interval modes** — the actual hypothesis: does the distribution
   separate a slow dink mode (~0.5 s) from a fast speed-up mode (~0.2 s)?

Free controls: contact density inside logged **timeouts** should collapse,
and contacts should cluster in the **latter part** of each rally window
(the referee marks the start before the serve, so uniform means noise).

## What the self-test does and does not prove

`--selftest` plants contacts at known times, buries them in a
crowd-plus-commentary bed, and sweeps both SNR and detection threshold. It
reports precision, recall and timing error at each point — the same
discipline as `--inject` in `model/gap_exploit.py`. A detector that cannot
recover a planted signal cannot be trusted to report a real one.

The threshold sweep is not decoration. The first version of this test ran a
single fixed `k=6` and reported a cliff from 0.71 recall to *exactly zero*
between 12 and 6 dB — which was the threshold being wrong, not the signal
being gone. A fixed threshold conflates the two.

It does **not** prove paddle pops survive a real broadcast mix. Synthetic
SNR is not calibrated to YouTube audio. Read the floor as *how much
headroom the method has*, not as a pass mark.

Measured 2026-08-09, planted contacts in a crowd + commentary bed. The
second row set is after the retune forced by the first real broadcast (see
"What the first real run found" below):

| SNR dB | v1 precision / recall | **v2 precision / recall** |
|---|---|---|
| 18 | 1.000 / 1.000 | **1.000 / 1.000** |
| 12 | 1.000 / 0.980 | **1.000 / 1.000** |
| 6 | 0.262 / 0.653 | **0.914 / 0.766** |
| 3 | 0.108 / 0.238 | **0.357 / 0.640** |

Operating floor moved from ≈12 dB to ≈**6–8 dB**, and precision at 6 dB went
0.26 → 0.91 — about 5 dB of extra headroom, which is what moving above the
commentary band buys. Timing error improved 3.7 → **2.1 ms**, well under the
33 ms that 30fps video floors out at.

## What the first real run found

Chicago 2026-07-25, 80.3 min of broadcast audio, first detector settings:
**5551 onsets where ~1550 real contacts should exist, and they were not
tracking play.** Two checks, neither needing a label:

- sliding game 1's 34 rally windows across the contact stream peaks at
  **1.12× chance**; a real detection would be several times chance
- only **3% of 10-second bins are quiet**, in a match that is ~32% dead time

So that run's interval histogram describes crowd and commentary, not
pickleball. In particular a median interval of 0.54 s looked seductively
like a dink mode and is not evidence of one — it survives no alignment test.
Recorded here because it is exactly the kind of number that gets quoted.

Causes, all addressed in the settings above: the 800 Hz band floor sat
inside the commentary bed; 22 kHz sampling discarded the high-frequency
content that most distinguishes a paddle strike; and a 45 ms refractory let
single strikes register twice, producing a 662-interval spike at 0.05–0.10 s.

Whether the retune clears a real mix is still open until it runs on the same
audio. **The first thing to read on any rerun is the alignment-vs-chance
number**, not the histogram.

### Sync and report, validated against the real timeline

Contacts planted inside the *real* 73 rally windows at a known offset of
137.4 s, plus 4% junk onsets, then run through `poc_report.py`:

- sync recovered **+137.40 s** (exact)
- 99% of contacts inside windows, alignment peak 3.45× the median
- density ratio **26×** inside vs outside
- 9 contacts/rally median, **0 empty rallies** of 73
- late-clustering 0.57 (correctly above the 0.5 uniform baseline)
- interval peaks named at **0.20 s and 0.55 s**

So the whole measurement chain — log → windows → sync → intervals → mode
detection — is verified. The single untested link is whether real broadcast
audio clears the 12 dB bar.

## Verification: smoke test early, validate later

Keep these separate — labelling while the detector is still being tuned
contaminates the labels, the same way picking top-K on the data you verify
on does.

- **Smoke test**, 2–3 rallies, watched live, purpose "is this working at
  all". Those rallies are then **excluded** from validation.
- **Blind validation**, later, on a random sample, labelled without seeing
  detector output. That is where the number comes from.

The first go/no-go needs no labels at all — the density and timeout
controls answer it straight from the referee log.

## The eventual schema (not built yet)

One row per shot, keyed `(match_id, game_number, rally_number, shot_index)`
so it joins straight onto `pb_rally`. Ordered by value ÷ cost:

| field | needs | why |
|---|---|---|
| `hitter_player_uuid` | player tracking | **the prize.** Breaks the actor/partner wall; per-player ball-share *is* the coverage dial `w`, measured instead of inferred through γ |
| `dt_prev` | audio | with contact height, gives the dink/drive/drop/speed-up 2×2 |
| `contact_height_net_units` | ball tracking, side camera | the net is a ruler in profile — no metric calibration needed |
| `hitter_x, hitter_y` + all four positions | homography | formation, kitchen arrival, court coverage |
| `landing_x, landing_y` | ball tracking | targeting and placement — where the freeze-out lives |
| `end_type` ∈ {winner, forced_error, unforced_error} | + rally end | |
| `winner_by`, `error_by` | attribution | unforced-error rates exist nowhere in this sport |
| `quality` | — | **mandatory.** Dinks are the hardest tracking case, so detector error correlates with the measurand. Without a per-row confidence you cannot condition on it or measure the bias |

Two standing principles:

**Store the physics, derive the taxonomy.** Persist intervals, heights and
positions — not `shot_type`. Taxonomies get revised; re-processing video is
expensive; measurements are stable. Shot type is a view.

**Every row carries a quality flag.** See above. This is the lesson from
the clutch work applied before the fact.
