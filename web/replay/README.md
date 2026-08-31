# PICKLES Replay — the 3D rally simulator

`pickles_replay.html` is the source of the published artifact
(https://claude.ai/code/artifact/8a2147c7-7038-48b6-ad24-5c734b80ba6c) —
a self-contained canvas 3D replay of rally 1, no build step, no
libraries. Open the file in a browser or republish it to the same
artifact URL to update the shared page.

Data pipeline: `vision/court3d.py --dump-show` writes
`data/vision/rally1_show.json` (path, players, impacts, bounces,
camera); the HTML embeds a copy of that JSON in its `const SHOW` line,
AUGMENTED with per-impact `mph` and `cat` fields computed by
`vision/shot_categories.py` (measured shot categories: dink /
speed-up / hand battle; the docstring there carries the frozen rules).
If the show JSON is regenerated, re-run the classifier and re-embed.

Feature notes (2026-08-31): camera presets incl. the true solved
broadcast pose, contact slow-mo, per-shot category chips + mph,
category-colored timeline ticks, speed-tinted trail, and the
"who's winning?" pressure bar — attack-heat only (fast >= 24 mph
shots, (mph-20) weight, 2 s decay), explicitly NOT a calibrated win
probability (receipt on the plate says so).

This page is the pre-Claude-Design source of record; design
iterations happen on the artifact.
