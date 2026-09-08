# Ball labeling v2

This is a new labeling path for a trainable temporal ball model. It leaves
the original audit pages and CSVs unchanged.

## Why it exists

The original pages densely label every nominal 30 fps frame from one match.
V2 makes three changes:

1. It probes the local source video with `ffprobe` and samples its native
   frame clock rather than assuming 30 fps.
2. It labels every native frame within 0.20 seconds of a known contact and
   every third frame elsewhere. This moves effort toward contacts and fast
   exchanges while retaining coverage of ordinary flight.
3. It seeks to the midpoint of each requested frame interval, verifies the
   frame actually decoded, and records both requested and displayed frame
   identities. Labels remain disabled while a seek is unresolved.

V2 also accepts an optional proposal CSV. Twenty percent of sampled frames
are deterministically blinded: the proposal is hidden until a human label is
present. The blind subset is for measuring proposal anchoring, not for a
pass/fail gate.

## Generate a page

Run this locally from the repository, where the VOD exists:

```bash
python vision/make_ball_audit_v2.py \
  --video full_match.mp4.webm \
  --rally 17
```

The output is `data/vision/ball_audit_v2_r17.html`. Open it in a browser and
load the same video. The page checks the filename and duration.

Useful overrides:

```bash
python vision/make_ball_audit_v2.py \
  --video full_match.mp4.webm --rally 17 \
  --flight-stride 3 --contact-radius 0.20 --blind-fraction 0.20
```

Run the self-test without a video:

```bash
python vision/make_ball_audit_v2.py --selftest
```

## Label controls

- Click: clean visible ball (`V`)
- `S`, then click: visible smear
- `I`, then click: inferred location while occluded
- `N`: position unknown
- `A`: accept the currently displayed proposal
- Arrow keys: previous/next sampled frame
- `,` / `.`: jump ten sampled frames
- `T`: toggle the recent human trail
- `P`: toggle proposals
- Backspace: clear the current answer

Export at the end of every session. Progress is cached in local storage, but
the exported CSV is the durable label artifact.

## Proposal format

The page accepts either of these headers:

```text
source_frame,x,y
```

or the current tracker-style form:

```text
frame,t_s,x,y
```

When both are present, `t_s` is used to map a clip-relative tracker frame to
the full video's source-frame clock. An explicit `source_frame` takes
precedence over both.

Proposal acceptance is recorded separately from a manual click. A proposal
is never visible before judgment on a blind frame.

## Export schema

Each answered sampled frame contains:

| field | meaning |
|---|---|
| `schema` | `ball-label-v2.1` |
| `video_name`, `video_duration_s` | source identity check |
| `fps`, `fps_rational` | probed native frame rate |
| `rally`, `sample_index`, `source_frame` | sample identity |
| `nominal_t_s` | requested source-frame timestamp |
| `displayed_t_s` | browser-reported displayed media time |
| `displayed_frame` | frame derived from the decoded media time |
| `seek_drift_frames` | displayed minus nominal time, in native frames |
| `x`, `y` | native-video pixel coordinate when available |
| `visibility` | `V`, `S`, `I`, or `N` |
| `sampling_reason` | `contact_dense` or `flight_sparse` |
| `blind` | proposal-blind assignment |
| `input_method` | manual click, key, import, or accepted proposal |
| `proposal_visible` | whether a proposal was visible before judgment |

These labels are intended for model training and ordinary evaluation. They
do not inherit the historical gate or seal semantics in the older vision
documents.
