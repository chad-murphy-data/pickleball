# Vision MVP: what tracks, what doesn't (2026-08-11)

Built a full video tracking system end to end — court geometry, player
identity, ball tracking, contact detection, attribution — and measured each
stage instead of assuming it. Two of the four stages are solid enough to
build on. One is the binding constraint, and it is not the one the earlier
sessions were worried about.

**Verdict: the court fit and the identity anchor are done. Ball detection
recovers about 1 shot per rally out of ~12. Everything downstream of the
ball is blocked on that, and no amount of cleverness in the linking or the
contact logic moves it — I tried, and the numbers below are what came back.**

---

## The matchup this was built for

`Waters/Johnson vs Bright/Fahey` is not a pair of matches. It is **four**,
and the shape is the reason it is worth filming:

| date | event | result | rallies | referee log |
|---|---|---|---|---|
| 2026-05-25 | MLP Dallas | Waters/Johnson **11-4** | 38 | start-marked, 100% |
| 2026-05-31 | MLP Columbus | Waters/Johnson **11-3** | 25 | start-marked, 100% |
| 2026-07-12 | **Mid-Season final** | Bright/Fahey **11-6** | 34 | start-marked, 94% |
| 2026-08-02 | MLP Orlando | Waters/Johnson **11-5** | 36 | start-marked, 100% |

Three routine beatdowns and one loss — in the final, the game the model
priced at 88% and missed (`model/receipts.json`). So the design writes
itself: the final against its own controls, same four players, same season.
Every one of the four has an informative referee log, which is not
guaranteed (four MLP events in 2026 log rallies in batches and are useless
for windowing).

The three routine wins are *the same matchup* as the loss. If the freeze-out
is real and visible, it is visible as a difference between rows 3 and
{1,2,4}, and nothing about ratings can see it — that comparison is an
interaction (my presence × their gap), which finding 11 established an
additive rating structurally cannot hold.

---

## Stage 1 — court geometry: SOLVED

`vision/court.py`. Median residual **0.06 ft** (under an inch), 81% of the
projected court model within 0.5 ft of a detected line pixel.

Three things made it work, each of which was a bug first:

1. **Confine the line search to the court.** The refinement is a direct
   search on four corners maximising how much of the projected court model
   lands on detected lines. Run over the whole frame it reliably scored a
   *skewed* quad higher than the correct one, by parking model lines on
   sponsor logos — this broadcast paints DOORDASH across the apron in
   letters taller than the kitchen line is thick. Masking the response to
   the court hull fixed it: score went 0.351 → 0.480, residual 0.80 → 0.06 ft.
2. **Distance-to-nearest-line, not mean brightness.** A soft objective lets
   impostor fits score well; requiring model points to land *on* a detected
   line does not, because sponsor bands do not reproduce the court's
   internal spacing.
3. **Convex hull of the colour blob.** A differently-coloured kitchen splits
   the playing surface into two components, so the hull of all large ones is
   the court whatever colour its middle is painted. An earlier version also
   accepted "any saturated colour" to catch the maroon kitchen and swallowed
   the entire apron with it.

**Independently validated.** The fit came from 30 sampled frames of a 60 s
360p clip. Projected onto a density map of all **691,883** ball candidates
from the full 80-minute match at 720p, every model line lands on the white
structure that density traced out — baselines, sidelines, both kitchen
lines, both centrelines. Different frames, different resolution, different
detector. It also confirms the clip and the full VOD share framing, so the
homography scales by 2 exactly.

Consequences worth stating: measurements are now in **feet and mph**, not
camera-specific pixels; the kitchen is a line rather than a guessed pixel
band; and 29% of raw ball candidates back-project off-court and can simply
be discarded.

---

## Stage 2 — player identity: SOLVED, and it was free

`vision/lineup.py`. The earlier attribution attempt tried to name players
from appearance and calibrated at 57% against a ~60% ceiling. It never
needed to. Side-out doubles is a state machine and the referee log hands us
its inputs:

* a team's two players **swap halves exactly when that team wins a rally
  while serving**, and
* the receiver is always the opponent **diagonally opposite** the server.

Initialise the four halves at the start of a game, walk the log, and every
rally yields all four players' court halves. The log only ever names two of
them.

**Measured on every 2026 MLP match — 1,218 matches, 45,689 rallies:**

| | |
|---|---|
| receiver predicted correctly | **45,348 / 45,689 = 99.25%** |
| matches with zero errors | 1,054 of 1,218 |
| independent parity cross-check | 91,630 / 93,792 = 97.69% |

Only rally 1 of each game consumes a logged receiver (to initialise); every
later one is a prediction. The parity check is genuinely independent — it
consumes the **score string**, which the state machine never reads. The rule
is not the obvious one: it anchors on the player who served *first in the
game* for that team, who stands right when their team's score is even. My
first version keyed it on the current server and scored 16/34, because both
partners serve in a turn and only one can be on the parity half.

This leaves exactly **one** unknown per game — which team is at the near end
— and even that is over-identified by ~30 rallies of agreement. The
left/right mapping is not a free parameter: the camera sits behind the near
team, so a near player's right is image-right and a far player's is
image-left, which is also what makes the serve come out diagonal.

---

## Stage 3 — player detection: adequate, with a hard framing limit

Detections in-rally average ~3.0 of 4. Looking at the frames rather than the
summary statistic: **where all four players are genuinely in frame, the
detector finds all four.** The shortfall is the broadcast, which frames the
court tightly enough that players standing behind either baseline — exactly
where servers and returners stand — are cropped out. That is not
recoverable by better segmentation.

Two real bugs found and fixed on the way:

* **The size prior used the wrong scale.** Apparent height follows the
  *lateral* ground scale (1/d), not the *depth* scale (1/d²). Using depth
  under-predicted far players badly enough to reject them as impossibly
  tall — 95% of surviving detections were near-side. The lateral scale
  predicts a near:far height ratio of 2.04 against 2.0 observed.
* **The search region clipped deep players in half.** It was a pixel
  dilation of the court hull; it is now a margin in *court feet* around the
  court, with a horizon guard (above the horizon the homogeneous coordinate
  flips sign and crowd rows back-project onto plausible court values).

---

## Stage 4 — ball tracking: THE BINDING CONSTRAINT

This is the finding. Measured on MLP Chicago women's doubles, 32 rallies,
both at 720p (previous extraction) and 360p (new tracker) — same verdict:

| | |
|---|---|
| ball candidates | 1.8–2.4 per frame at 720p, 11 per frame at 360p |
| tracks ≥6 frames, per rally | **~75** |
| track length | p50 **14 frames** (0.23 s) |
| tracks that cross the net, per rally | ~6.5 |
| contacts recovered per rally | **1.1** |
| shots per rally actually played | **~12** (0.58/s over 20 s rallies) |

The candidate stream is dominated by things that are not the ball. The
longest "track" in a sampled rally ran 410 frames over 7.15 s with a median
area of 44 px and never crossed the net — that is a player, not a ball.

**The label-free test that settled it.** Consecutive contacts must land on
opposite sides of the net, because the ball crosses it between them. Chance
is 50%. The detector returned **9%** — not noise, but structure: slow blobs
oscillating on one side, generating same-side repeats. Sweeping area,
strength, track speed and reversal strictness moved it nowhere:

```
 area<=25, sweeping min px/frame and reversal strictness
 px/frame  min_run  travel |  alternation   contacts/rally
     0        3       2.0  |     7.2%           8.6
     0        5       4.0  |     0.0%           0.7
    10        3       2.0  |     3.8%           5.5
    14        3       2.0  |     0.0%           2.1
    18        5       4.0  |     0.0%           0.0
```

Either the contacts are plentiful and meaningless, or they are meaningful
and almost absent. There is no setting where both hold.

**What did help**, and is now the design: identify the ball by the one
signature nothing else in the frame has — **it crosses the net**. Restricting
contact detection to net-crossing tracks lifts alternation 9% → **33–37%**,
and serve attribution to 63% on side (chance 50%) and 37% on player (chance
25%). Both are above chance and both are useless for analysis, because the
surviving sample is 1.1 contacts per rally out of ~12.

Two design improvements that are real and survive regardless, for whenever
detection is fixed:

* **Contacts are sign changes of d(court_y)/dt.** A paddle reverses the
  ball's direction across the court; a bounce does not. This replaces a
  22-pixel vertical-reversal threshold plus a hand-drawn "kitchen band" to
  suppress bounces — both guesses about one camera — with physics, and it
  dissolves the dink/bounce confusion that made the old detector inflate
  contact counts fourfold.
* **Attribution has two channels**, reported separately so their
  disagreement is visible: contact position + the lineup state machine
  (always available, wrong when players cross over), and nearest player blob
  in *image* space (exact when it fires, unavailable when the broadcast has
  cropped the player out). Image space matters — the ball is airborne, so
  its ground back-projection runs long while a player's does not.

### Detection or association? DETECTION. (`vision/ball_recall.py`)

"~1 shot in 12" cannot say which stage failed, and the two answers cost
very different amounts: better linking is an afternoon, a learned detector
is a GPU weekend plus labelling. Settled using net-crossing tracks as a
free label — nothing else in frame crosses the net — and asking what the
detector was doing in the frames *around* them.

| | |
|---|---|
| confident (net-crossing, ≥15 frame) tracks | 4.5 per rally |
| ...covering | **23.0%** of rally frames |
| in-track recall (ceiling, survivorship-biased up) | **89.1%** |
| one frame past a track end: nearest candidate | **median 239 px** |
| same, displaced-null control | median 243 px |
| within 20 px of where the ball should be | **3.6%** (null 0.0%) |

**The detector is bimodal.** While it holds the ball it holds it well —
89% of frames inside a confident track carry a detection. The moment a
track ends, there is *nothing anywhere near* where the ball must be: the
nearest candidate sits ~240 px away, which is indistinguishable from the
displaced null at ~243 px. The ball does not drift out of association. It
vanishes.

**The confound is closed.** Tracks often end *at a contact*, where forward
extrapolation is wrong by construction — so the same probe was rerun
holding the track's LAST OBSERVED position, which a ball that merely
changed direction would still be near. Same answer: median 238 px, 5.8%
within 20 px. Not a motion-model failure. The ball is not in the candidate
set.

Overall ball detection works out to roughly **20–25% of rally frames**,
clustered rather than spread.

**And the coverage is biased against the shots that matter.** Confident
ball detections sit in the middle of the court — 42% of them inside the
kitchen band (15 < y < 29) against a 14% base rate for all candidates,
median y 18.1 ft vs 22.6. The detector holds the ball during slow dinks
near the net and loses it on drives, serves and speed-ups from the
baseline. That is the anti-correlation with the events of interest that
this project cares about, now measured rather than suspected — and it is a
further reason to treat the "fast mode at 0.15–0.25 s" in
`interval_results.md` with suspicion, since the median ball track is
0.233 s long.

Association is a real but second-order effect: only 43% of consecutive
confident tracks join up under extrapolation (null 14%), on n = 14 pairs.

**Conclusion: perfect linking cannot beat ~25% of frames, because that is
all there is to link.** Trajectory-level association, three-frame
differencing, multi-hypothesis tracking — all of it is wasted effort here.
The learned detector is not one option among several; it is the only one
that addresses what is actually broken.

### Why the previous "50–65% recall" was optimistic

It was physics-referenced, never measured. The measurement says the usable
figure — shots recoverable as identified contacts — is closer to **10%**.
The gap is not detection of *some* ball pixels; it is that the track
fragments into 0.23 s pieces and cannot be told from ~75 other tracks per
rally.

---

## What this says about the GPU weekend

It converts it from a nice-to-have into the whole job. A TrackNet-class
learned ball detector is not an incremental precision gain on top of a
working pipeline — it is the one stage that is failing, and the three stages
around it are already built and measured:

* court geometry: **0.06 ft**
* player identity: **99.25%** over 45,689 rallies
* contact logic and attribution: built, physically motivated, currently
  starved of input

A learned detector outputs a single ball position per frame with a
confidence, which removes both failure modes at once: no 75-track
ambiguity, and no 0.23 s fragments. Everything downstream is written and
waiting. That is the cheapest possible position to be in before spending
money.

**Do not run the tale-of-two analysis on the current detector.** With 1.1
contacts per rally the touch-share difference between the final and its
controls would be dominated by which rallies happened to track, and it would
look like a result.

---

## Reproducing

```bash
# 1. fit the court once per broadcast (fixed camera)
python vision/court.py --video VOD.mp4 --out court.json --overlay check.png

# 2. one decode pass: ball + players in court feet, camera verified per frame
python vision/track_match.py --video VOD.mp4 --court court.json --out pfx

# 3. identity anchor from the referee log — no video needed
python vision/lineup.py --match <match-uuid>
python vision/lineup.py --validate-all        # the 99.25% number

# 4. contacts, attribution, and the honest scores
python vision/shots.py --ball pfx_ball.csv --players pfx_players.csv \
    --court court.json --rallies windows.csv --lineup data/vision/lineup_x.csv
```

Match UUIDs for the four meetings are in the table at the top; rally
windows in *video* time still come from the cheer/scorebug anchoring in
`vision/poc_report.py`, because the broadcast is condensed (80.3 min of
video for 107.2 min of wall clock, with cuts *inside* games) and no global
offset exists.

**Videos cannot be fetched from this environment**: `*.googlevideo.com` is
blocked by the egress policy (metadata resolves, media 403s), so VODs have
to be downloaded on a local machine. Note also that a datacenter IP gets
"Sign in to confirm you're not a bot" from every player client except
`android_vr`.
