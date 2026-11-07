"""
F0 extraction.

Primary method (as in the paper, App. C.2, and the reference implementation):
PyWorldVocoder DIO + StoneMask, with frame period = hop_length / sr so F0 frames
line up 1:1 with mel frames.

Fallback (only if `pyworld` is not installed): a normalised-autocorrelation pitch
tracker in numpy. It is less accurate than WORLD and is reported loudly so it is
never used silently for a real training run.
"""
import warnings

import numpy as np

try:
    import pyworld as pw

    HAVE_PYWORLD = True
except Exception:  # pragma: no cover
    HAVE_PYWORLD = False

_warned = False


def extract_f0(wav, sr, hop_length, f0_min=60.0, f0_max=600.0, n_frames=None):
    """Return F0 in Hz per hop (0 for unvoiced). Length = n_frames if given."""
    global _warned
    wav = np.asarray(wav, dtype=np.float64)
    if HAVE_PYWORLD:
        f0, t = pw.dio(wav, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=hop_length / sr * 1000)
        f0 = pw.stonemask(wav, f0, t, sr)
    else:
        if not _warned:
            warnings.warn("pyworld not installed: using numpy autocorrelation F0 fallback "
                          "(install pyworld for real training runs).")
            _warned = True
        f0 = _autocorr_f0(wav, sr, hop_length, f0_min, f0_max)
    f0 = np.asarray(f0, dtype=np.float32)
    if n_frames is not None:
        if len(f0) >= n_frames:
            f0 = f0[:n_frames]
        else:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
    return f0


def _autocorr_f0(wav, sr, hop, f0_min, f0_max, frame_len=None, threshold=0.45):
    if frame_len is None:
        frame_len = int(2.5 * sr / f0_min)  # >= 2.5 periods of the lowest F0
    pad = frame_len // 2
    y = np.pad(wav, (pad, pad))
    n = 1 + len(wav) // hop
    lag_min = max(2, int(sr / f0_max))
    lag_max = min(frame_len - 1, int(sr / f0_min))
    global_rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    f0 = np.zeros(n, dtype=np.float64)
    win = np.hanning(frame_len)
    nfft = 1 << int(np.ceil(np.log2(2 * frame_len)))
    for i in range(n):
        seg = y[i * hop : i * hop + frame_len]
        if len(seg) < frame_len:
            break
        if np.sqrt(np.mean(seg ** 2)) < 0.05 * global_rms:
            continue
        seg = (seg - seg.mean()) * win
        spec = np.fft.rfft(seg, nfft)
        ac = np.fft.irfft(np.abs(spec) ** 2)[:frame_len]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        k = lag_min + int(np.argmax(ac[lag_min:lag_max]))
        if ac[k] < threshold:
            continue
        if 1 <= k < len(ac) - 1:  # parabolic interpolation
            a, b, c = ac[k - 1], ac[k], ac[k + 1]
            denom = a - 2 * b + c
            k = k + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
        f0[i] = sr / k
    return f0


def interpolate_unvoiced(f0):
    """Linear interpolation over unvoiced (0) frames; edges held constant (App. C.2, step 1)."""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    nz = np.nonzero(f0)[0]
    if len(nz) == 0:
        return f0
    f0[:] = np.interp(np.arange(len(f0)), nz, f0[nz])
    return f0
