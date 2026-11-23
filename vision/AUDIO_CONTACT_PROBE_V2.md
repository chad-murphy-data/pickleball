# Audio diagnostic with human contact labels

Source: first 329.994 seconds of original audio, 48 kHz stereo FLAC.
No denoising, source separation, or learned audio classifier. Spectral flux in
four fixed bands (300–1200, 1200–3000, 3000–6000, 6000–12000 Hz) measures sudden
increases in spectral energy. Channel energies are averaged, avoiding phase
cancellation. Clip-level unlabeled normalization yields a percentile score.

For each test rally, choose an offset using only the other four rallies, on a
fixed -250 to +250 ms grid at 5 ms increments. Objective: contact-minus-control
mean peak score, averaged across training rallies. Score within +/-40 ms of
the aligned event. Do not equate this tolerance with timestamp accuracy.

Offsets for outer rallies 6,7,8,9,10: +125,+110,+105,+125,+115 ms.
Positive means the audio peak follows the labeled visual contact. This may
combine broadcast delay, extraction origin, sound propagation, and label error;
the diagnostic does not establish its physical cause or a universal offset.

| Comparison | Samples | Pooled AUC |
|---|---|---:|
| Contacts versus in-rally controls | 78 versus 311 | 0.908 |
| Contacts versus visual false detections | 78 versus 27 | 0.841 |
| Visually missed contacts versus visual false detections | 26 versus 27 | 0.816 |
| Visually missed contacts versus controls | 26 versus 311 | 0.897 |

AUC describes score ranking, not contact precision or recall. Ties count half.
Controls are a fixed 0.1-second grid at least 0.25 seconds from any labeled
contact; overlapping audio windows mean they are correlated. No significance
test or independent-sample confidence interval is claimed. AUC versus visual
extras is undefined for rally 6 (no extras); per-rally values for 7–10 are
0.844, 1.000 (only two extras), 0.899, and 0.718.

Conclusion: audio has promising complementary evidence, including at contacts
missed by vision. This supports an audio-plus-visual experiment; it does not
demonstrate reliable paddle sound classification or automatic contact detection.
Claps, bounces, footsteps, and other impacts may contribute to the score.
No listening-based sound identities are claimed by this numerical probe.

Next: estimate offset and any score thresholds strictly inside training folds,
then evaluate audio-supported contact proposals against the same visual baseline.
Do not use a pooled best offset or test-rally thresholds. Upstream ball training
and same-match feature-choice caveats still apply.

```bash
python3 vision/audio_contact_probe_v2.py --audio contact_audio_first330.flac \
  --hitter-report data/vision/ball_hitter_run1/report.json \
  --baseline-report data/vision/auto_contact_run1/report.json \
  --out data/vision/audio_contact_probe_v2
```

Requires NumPy, SciPy, ffmpeg and ffprobe. Three tests cover offset sign, AUC
ties, and stereo phase cancellation. Report records source hashes and individual
scores for follow-up. Original labels and detector defaults are unchanged.
