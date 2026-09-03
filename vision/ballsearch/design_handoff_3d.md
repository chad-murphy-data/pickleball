# Design handoff — the 3D rally reconstruction

**Asset:** `court3d_r4.mp4` (2.2 MB) and `court3d_r4.html` (interactive).
**Subject:** one pickleball rally, reconstructed in 3D from broadcast video.
**Owner-facing question this answers:** what design can and cannot do with
the file, and what we should render instead.

---

## What it actually is

A 14.1-second rally from the MLP Chicago 2026-07-25 broadcast, rebuilt in
three dimensions from the video alone. The ball path is not animated by
hand and not a stylisation of a real path — it *is* the measurement:
piecewise ballistic arcs fitted through a single-camera projection solved
from 11 hand-clicked court landmarks. Player positions are floor points
from pose tracking (ankle midpoint through the ground-plane homography).

  16 flights · 17 attributed hits · 659 path samples · 4 players
  5 net crossings · median impact-point error 1.76 ft

Rally 4 was chosen because it is the best reconstruction on record — the
first to pass the project's hardest replication check.

## Current render specs

  container   MP4, H.264 (libx264), yuv420p, no audio
  quality     CRF 18, preset slow, +faststart
  frame       1280 x 720, 30 fps, 528 frames, 2.2 MB
  timing      1.0 s lead-in, 14.1 s rally at 1x, 2.5 s tail
  sampling    rendered at 2x supersample, downscaled INTER_AREA
  camera      orthographic; azimuth starts -2.4 rad, elevation 0.5 rad,
              zoom 13.0, orbits 0.55 rad across the rally

Universally editable — Premiere, After Effects, Final Cut, DaVinci,
CapCut and Canva all ingest it without transcode.

## Palette currently baked into the frame

  background      #0c0c10
  court lines     #3b6ea5
  net posts       #777777      net tape   #cccccc
  ball trail      #e8c44a at 25% over bg (past) / #ffd94a (live)
  near-side pair  #e05c5c      far-side pair  #5ca8e0
  captions        #aaaaaa, burned into the bottom two lines

## Court coordinate frame

Feet, `court.py` convention. Useful for reframing, overlays and any
graphic that has to line up with the render.

  x   0 = left sideline, 20 = right sideline
  y   0 = FAR baseline, 44 = NEAR baseline, net at 22
      kitchen (non-volley) lines at y = 15 and y = 29
      centre service line at x = 10
  z   up from the floor; net tape 34 in at the centre, 36 in at the posts

## Baked vs. parameterizable

Everything in the MP4 is flattened pixels. Design can trim, crop,
speed-ramp, overlay and score it. Design cannot recolour it, remove the
burned captions, change the camera angle, or composite it onto a
background — and 720p landscape will go soft if reframed to vertical.

The renderer is ours (`vision/ballsearch/render_court3d.py`, ~210 lines),
so all of the following are changes to our code, not to their file:

  - PNG or ProRes 4444 sequence with a real alpha channel
  - native vertical 1080 x 1920 framing (not a crop)
  - 1440p or 4K (the pipeline already supersamples 2x)
  - clean plate with no burned captions
  - PICKLES brand palette instead of the debug colours
  - custom camera move, speed ramps, a hold on the decisive shot
  - a different rally: r2, r3 and r5 are also built; r9 and r10 are
    graded seals and carry stricter rules

Regenerate at any time:

    cd vision/ballsearch
    python3 rally_3d.py 4          # -> court3d_r4.html
    python3 render_court3d.py 4    # -> court3d_r4.mp4

## Honesty rules — non-negotiable

The project's brand is receipts. These are not style preferences;
breaking one turns an honest measurement into a fabricated one.

1. **Dashed segments mean the ball was never seen.** Those frames exist
   only by extending the arcs on either side through an occlusion. 52 of
   659 samples in r4 are inferred. Never restyle a dash into a solid.
2. **Gaps mean the tracker lost the flight.** Never interpolate across a
   gap to produce a continuous trail.
3. **The ball vanishes while the track is lost.** Never draw a ball where
   the data has none.
4. **Depth is the weak axis.** In a one-camera fit, how far down the court
   something sits is the least trustworthy dimension. Do not build a
   graphic whose punchline depends on depth precision.
5. **One of the five net crossings passes under the 34-inch tape.** That
   is a visible fit error. Caption it rather than paint over it.
6. **This is a training rally, not a graded seal.** No caption may imply
   validated accuracy. The honest line is "reconstructed from broadcast
   video; median impact error 1.76 ft".
7. **No probability is ever displayed as 0% or 100%** (standing house
   rule) if any overlay carries one.

## Open questions for design

  - Which delivery format: alpha sequence, vertical master, or both?
  - Brand palette, or keep the debug colours as an honesty signal?
  - Does the caption stay in-frame, or move to their typography?
