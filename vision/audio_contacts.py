"""Step 2 of the vision POC — find paddle contacts in the audio track.

    python vision/audio_contacts.py --selftest              # no video needed
    python vision/audio_contacts.py --audio match.m4a --out contacts.csv

WHY AUDIO.  A pickleball paddle strike is a sharp broadband transient — it
is the sound the noise lawsuits are about — so contact TIMES are recoverable
from the audio track alone, at ~5 ms, with no camera calibration, no
homography and no player detection.  Video at 30fps localises a contact to
+/-33 ms, which is 16% of a 200 ms speed-up interval; audio is ~2.5%.

WHAT AUDIO CANNOT DO: attribution.  It says WHEN, never WHO.  That needs
player tracking, and attribution is the highest-value field in the whole
schema, so this never replaces vision — it makes it cheaper and sharper,
and it is the one part of the pipeline testable in an afternoon.

METHOD.  Half-wave-rectified spectral flux in the paddle band, adaptive
threshold on a local median + MAD (robust to crowd swell and commentary,
which are broadband but not transient), peak-pick with a refractory gap.
Pure numpy; audio decoded through the static ffmpeg that ships with
imageio-ffmpeg, so there is no system dependency.

SELF-TEST.  The point of --selftest is that the detector gets scored against
KNOWN contact times before it ever sees a broadcast: synth a track with
contacts at chosen times, bury it in crowd noise at a range of SNRs, and
report precision / recall / timing error at each. That is the same
discipline as `--inject` in model/gap_exploit.py — a detector that cannot
recover a planted signal cannot be trusted to report a real one, and the
sweep says where it stops working. It does NOT prove paddle pops survive a
real broadcast mix; only the real video answers that.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 22050
N_FFT = 1024
HOP = 128                 # 5.8 ms — the resolution the whole idea rests on
BAND = (800.0, 8000.0)    # paddle transient energy; skips most speech/rumble
REFRACTORY_S = 0.045      # two contacts closer than this are one event
CHUNK_FRAMES = 200_000


# ----------------------------------------------------------------- decode ---

def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_audio(path, sr=SR):
    """Decode any container to mono float32 at sr via ffmpeg's PCM pipe."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path),
           "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{p.stderr.decode()[:500]}")
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------- detector ---

def spectral_flux(x, sr=SR, n_fft=N_FFT, hop=HOP, band=BAND):
    """Half-wave-rectified spectral flux over the paddle band, chunked so a
    35-minute match does not need a 400 MB frame matrix."""
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    flux = np.zeros(n_frames, dtype=np.float32)
    prev_tail = None
    for start in range(0, n_frames, CHUNK_FRAMES):
        stop = min(n_frames, start + CHUNK_FRAMES)
        idx = np.arange(start, stop)[:, None] * hop + np.arange(n_fft)[None, :]
        frames = x[idx] * win
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, keep]
        if prev_tail is not None:
            mag = np.vstack([prev_tail, mag])
        d = np.diff(mag, axis=0)
        np.maximum(d, 0.0, out=d)
        f = d.sum(axis=1)
        if prev_tail is None:            # first frame has no predecessor
            flux[start] = 0.0
            flux[start + 1:stop] = f
        else:
            flux[start:stop] = f
        prev_tail = mag[-1:][:]
    return flux


def _rolling_stat(v, w, fn):
    """Cheap rolling median/MAD via reshaped blocks + linear interpolation."""
    n = len(v)
    nb = max(1, n // w)
    edges = np.linspace(0, n, nb + 1).astype(int)
    centres, vals = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            centres.append((a + b) / 2.0)
            vals.append(fn(v[a:b]))
    if len(centres) == 1:
        return np.full(n, vals[0], dtype=np.float32)
    return np.interp(np.arange(n), np.array(centres),
                     np.array(vals)).astype(np.float32)


def detect(x, sr=SR, hop=HOP, k=6.0, refractory_s=REFRACTORY_S,
           adapt_s=3.0, return_flux=False):
    """Onset times in seconds. Convenience wrapper: flux + peak-picking."""
    flux = spectral_flux(x, sr=sr, hop=hop)
    times = peaks_from_flux(flux, sr=sr, hop=hop, k=k,
                            refractory_s=refractory_s, adapt_s=adapt_s)
    return (times, flux) if return_flux else times


def peaks_from_flux(flux, sr=SR, hop=HOP, k=6.0, refractory_s=REFRACTORY_S,
                    adapt_s=3.0):
    """Threshold + peak-pick a precomputed flux curve.

    Split out from detect() because the flux is the expensive part and does
    NOT depend on k — so a threshold sweep costs one STFT, not N.

    k is the threshold in robust sigmas above the LOCAL median. The local
    baseline is what makes this survive crowd swell and commentary: those
    raise the median, and a transient still has to beat it by k MADs.
    """
    if len(flux) < 8:
        return np.array([])
    w = max(16, int(adapt_s * sr / hop))
    med = _rolling_stat(flux, w, np.median)
    mad = _rolling_stat(flux, w, lambda a: np.median(np.abs(a - np.median(a))))
    thresh = med + k * (1.4826 * mad + 1e-9)

    over = flux > thresh
    # local maxima only
    peak = np.zeros_like(over)
    peak[1:-1] = over[1:-1] & (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
    cand = np.flatnonzero(peak)
    if len(cand) == 0:
        return np.array([])

    # refractory: keep the strongest peak in each cluster
    gap = int(refractory_s * sr / hop)
    kept, i = [], 0
    while i < len(cand):
        j = i
        while j + 1 < len(cand) and cand[j + 1] - cand[j] <= gap:
            j += 1
        kept.append(cand[i:j + 1][np.argmax(flux[cand[i:j + 1]])])
        i = j + 1
    return (np.array(kept) * hop + N_FFT / 2.0) / sr


# -------------------------------------------------------------- self-test ---

def synth_contact(sr, dur=0.06, f_lo=900.0, f_hi=7000.0, rng=None):
    """A paddle-like click: near-instant attack, fast decay, band-limited
    broadband content. Not a physical model — just the right envelope and
    spectrum for an onset detector to be honestly tested against."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[(f < f_lo) | (f > f_hi)] = 0
    click = np.fft.irfft(spec, n).astype(np.float32)
    env = np.exp(-t / 0.012).astype(np.float32)
    env[:int(0.0008 * sr)] *= np.linspace(0, 1, max(1, int(0.0008 * sr)))
    out = click * env
    return out / (np.max(np.abs(out)) + 1e-9)


def synth_track(seconds, contacts, snr_db, sr=SR, seed=0):
    """Contacts buried in pink-ish crowd noise plus a speech-band murmur
    (the commentary bed, which is the realistic nuisance)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    white = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / sr)
    spec /= np.sqrt(np.maximum(f, 1.0))                 # pink crowd
    crowd = np.fft.irfft(spec, n).astype(np.float32)
    crowd /= np.std(crowd) + 1e-9
    speech = rng.standard_normal(n).astype(np.float32)
    sp = np.fft.rfft(speech)
    sp[(f < 200) | (f > 3500)] = 0
    speech = np.fft.irfft(sp, n).astype(np.float32)
    speech *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * np.arange(n) / sr))
    speech /= np.std(speech) + 1e-9
    bed = 0.7 * crowd + 0.7 * speech
    bed /= np.std(bed) + 1e-9

    x = bed.copy()
    amp = 10 ** (snr_db / 20.0)
    for ct in contacts:
        c = synth_contact(sr, rng=rng) * amp
        i = int(ct * sr)
        j = min(n, i + len(c))
        if i < n:
            x[i:j] += c[:j - i]
    return x / (np.max(np.abs(x)) + 1e-9)


def score(truth, found, tol=0.030):
    """Greedy nearest match within tol seconds -> precision, recall, |error|."""
    truth = list(truth)
    used, errs, tp = set(), [], 0
    for f in found:
        best, bi = tol + 1, None
        for i, t in enumerate(truth):
            if i in used:
                continue
            d = abs(f - t)
            if d < best:
                best, bi = d, i
        if bi is not None and best <= tol:
            used.add(bi)
            errs.append(best)
            tp += 1
    prec = tp / len(found) if len(found) else 0.0
    rec = tp / len(truth) if truth else 0.0
    return prec, rec, (float(np.median(errs)) if errs else float("nan"))


def make_rallies(n_rallies=25, seed=3):
    """A plausible rally structure: dinks around 0.55 s, occasional speed-up
    bursts around 0.20 s, dead time between rallies."""
    rng = np.random.default_rng(seed)
    contacts, labels, t = [], [], 5.0
    for _ in range(n_rallies):
        n_shots = int(rng.integers(4, 14))
        fast = rng.random() < 0.45
        for s in range(n_shots):
            contacts.append(t)
            speedup = fast and s >= n_shots - 3
            labels.append("fast" if speedup else "slow")
            t += (rng.normal(0.20, 0.03) if speedup
                  else rng.normal(0.55, 0.09))
        t += float(rng.uniform(6.0, 14.0))
    return np.array(contacts), labels, t + 5.0


def selftest(args):
    truth, labels, dur = make_rallies(args.rallies)
    print(f"synthetic: {len(truth)} contacts over {dur/60:.1f} min "
          f"({sum(1 for l in labels if l=='fast')} in speed-up bursts)")
    print("sweeping the threshold at every SNR — a single fixed k conflates "
          "'signal is gone'\nwith 'threshold is wrong', and the first run of "
          "this test did exactly that.\n")
    print(f"{'SNR dB':>7} {'best k':>7} {'precision':>10} {'recall':>8} "
          f"{'F1':>6} {'median |err| ms':>16} {'n found':>8}")
    rows = []
    for snr in args.snrs:
        x = synth_track(dur, truth, snr, seed=11)
        flux = spectral_flux(x)                 # expensive; k-independent
        best = None
        for k in args.ks:
            found = peaks_from_flux(flux, k=k)
            p, r, e = score(truth, found)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if best is None or f1 > best[0]:
                best = (f1, k, p, r, e, len(found))
        f1, k, p, r, e, n = best
        rows.append((snr, k, p, r, f1, e, n))
        print(f"{snr:>7.0f} {k:>7.1f} {p:>10.3f} {r:>8.3f} {f1:>6.3f} "
              f"{(e*1000 if e==e else float('nan')):>16.1f} {n:>8}")

    good = [t for t in rows if t[2] >= 0.9 and t[3] >= 0.9]
    print()
    if good:
        print(f"  operating floor: precision & recall both >=0.90 down to "
              f"SNR {min(g[0] for g in good):.0f} dB "
              f"(at k={[g[1] for g in good if g[0]==min(x[0] for x in good)][0]:.1f})")
    else:
        print("  never reached 0.90/0.90 — detector needs work before video")
    print("  NB SNR here is contact peak vs a crowd+commentary bed. It is NOT "
          "calibrated to a\n  real broadcast mix, so read the floor as "
          "'how much headroom the method has',\n  not as a pass/fail for "
          "YouTube audio. Only the real video answers that.")

    # does the interval histogram separate, given good detection?
    best_row = max(rows, key=lambda t: t[4])
    x = synth_track(dur, truth, best_row[0], seed=11)
    found = detect(x, k=best_row[1])
    iv = np.diff(found)
    iv = iv[iv < 1.5]
    if len(iv) > 20:
        fast = iv[iv < 0.35]
        slow = iv[iv >= 0.35]
        print(f"\n  interval check at SNR {best_row[0]:.0f} dB: {len(fast)} fast "
              f"(median {np.median(fast)*1000:.0f} ms), {len(slow)} slow "
              f"(median {np.median(slow)*1000:.0f} ms)")
        print("  ^ recovering two modes here only proves the MEASUREMENT works; "
              "whether real\n    pickleball is bimodal is what the real video "
              "has to answer.")
    return rows


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audio", help="video or audio file to scan")
    ap.add_argument("--out", default="data/vision/contacts.csv")
    ap.add_argument("--k", type=float, default=6.0, help="threshold in MADs")
    ap.add_argument("--rallies", type=int, default=25)
    ap.add_argument("--snrs", type=float, nargs="*",
                    default=[18, 12, 6, 3, 0, -3])
    ap.add_argument("--ks", type=float, nargs="*",
                    default=[1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                    help="threshold grid swept at each SNR in --selftest")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return
    if not args.audio:
        ap.error("give --audio FILE or --selftest")

    x = load_audio(args.audio)
    print(f"decoded {len(x)/SR/60:.1f} min of audio")
    times = detect(x, k=args.k)
    print(f"found {len(times)} onsets "
          f"({len(times)/(len(x)/SR)*60:.1f} per minute)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_audio_s"])
        for t in times:
            w.writerow([f"{t:.4f}"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
