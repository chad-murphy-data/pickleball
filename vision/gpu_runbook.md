# GPU runbook — the Gate C verdict hour (Colab path)

Written 2026-08-16 for the core-16 verdict run. You need a GPU for
exactly ONE step: ViTPose-plus-huge extraction (the pre-registered
verdict instrument). Everything before and after is CPU. Run this
AFTER the labeling evening — it consumes the exported
`contact_labels_chicago0725.csv`.

## Step 0 — the GPU: Google Colab, free tier

1. Go to colab.research.google.com, sign in with the Google account
   that has Drive space (~2 GB free needed for the video + models).
2. New notebook → menu **Runtime → Change runtime type → T4 GPU** →
   Save. That's the whole "getting a GPU" step: a free T4 is enough.
   (Paid fallbacks if free capacity is scarce: Colab Pro ~$10, or
   runpod.io / lambda.ai at well under $1/hour for this job.)

## Step 1 — one Drive folder with everything

In Google Drive, make a folder `pickleball-gate` containing:

    full_match.mp4.webm                       (the SAME file you labeled on
                                               — the run verifies this)
    contact_labels_chicago0725.csv            (your export from the tool)
    rally_windows_chicago0725_v4.csv
    pose_extract.py
    swing_probe.py
    serve_pin_windows.py
    contact_ceiling.py

The video upload is the slow part (~10-20 min once); everything else
is small. All outputs land back in this folder, so they survive Colab
session resets.

## Step 2 — notebook cells, in order

Cell 1 — mount Drive and enter the folder:

    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/pickleball-gate
    !ls

Cell 2 — installs (torch with CUDA is preinstalled on Colab):

    !pip -q install -U transformers scipy pillow imageio-ffmpeg

Cell 3 — sanity check (CPU, seconds; must end SELFTEST OK):

    !python pose_extract.py --selftest

Cell 4 — THE VERDICT EXTRACTION (leave the tab open; expect 1-3 hours
on a T4 — first run also downloads ~2-3 GB of model weights):

    !python pose_extract.py --video full_match.mp4.webm --device cuda \
        --labels contact_labels_chicago0725.csv \
        --windows rally_windows_chicago0725_v4.csv \
        --out-dir pose --debug-frames 2

Watch the first lines: the `pose_extract build:` banner, `same-file
check OK` (the labels' stamped duration vs this video — if it aborts
here, the video in Drive is not the file you labeled on), and `native
fps detected`. Per-rally progress prints with an ETA. If the session
dies mid-run, just re-run the cell — extraction resumes where it
stopped (completed rallies are skipped).

Cell 5 — THE GATE (CPU, seconds; prints the verdict):

    !python contact_ceiling.py --labels contact_labels_chicago0725.csv \
        --pose-dir pose --windows rally_windows_chicago0725_v4.csv \
        --report contact_ceiling_report.json

Cell 6 — OPTIONAL but recommended, the production-spine A/B (minutes):

    !pip -q install rtmlib onnxruntime-gpu
    !python pose_extract.py --video full_match.mp4.webm --device cuda \
        --backend rtmpose \
        --labels contact_labels_chicago0725.csv \
        --windows rally_windows_chicago0725_v4.csv \
        --out-dir pose_rtm
    !python contact_ceiling.py --labels contact_labels_chicago0725.csv \
        --pose-dir pose_rtm --windows rally_windows_chicago0725_v4.csv \
        --report contact_ceiling_report_rtm.json

## Step 3 — bring the verdict home

The report JSONs are already in Drive. Paste Cell 5's printed verdict
block into the Claude thread (and Cell 6's if run). PROCEED / KILL /
MIDDLE next steps are pre-registered in vision/contact_gate.md.

## No-GPU fallback (only if Colab and rentals are both out)

Do NOT run the ViT-huge extraction on CPU (days). Say so in the thread
BEFORE extracting: the pre-named fallback is a dated amendment making
rtmpose-balanced the verdict instrument (overnight on the Mac), at a
small cost to instrument strength and an asterisk on any narrow KILL.
