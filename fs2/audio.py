"""
Audio front end, implemented with numpy/scipy only.

Re-implements the feature extraction of ming024/FastSpeech2 (audio/stft.py,
TacotronSTFT) so preprocessing has no torch/librosa dependency:
  * reflect-pad by n_fft//2, Hann window, magnitude STFT
  * Slaney-style mel filterbank with Slaney area normalisation
    (identical to librosa.filters.mel(htk=False, norm="slaney"))
  * log(clamp(mel, 1e-5))  -- the input expected by the HiFi-GAN universal vocoder
  * energy = L2 norm of each magnitude-STFT frame (FastSpeech 2, Sec. 2.3)
"""
from fractions import Fraction
import io

import numpy as np
from scipy.signal import get_window, resample_poly


# --------------------------------------------------------------------------- I/O
def load_audio_bytes_or_path(src):
    """Return (float32 mono waveform in [-1, 1], sample rate). `src` is a path or raw bytes."""
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src,
                           dtype="float32", always_2d=True)
        wav = data.mean(axis=1)
    except ImportError:
        from scipy.io import wavfile

        sr, data = wavfile.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        data = data.astype(np.float32)
        wav = data.mean(axis=1) if data.ndim == 2 else data
    return wav.astype(np.float32), int(sr)


def audio_duration_from_bytes(src):
    """Duration in seconds, read from the header when possible (no full decode)."""
    try:
        import soundfile as sf

        info = sf.info(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        return info.frames / float(info.samplerate)
    except Exception:
        wav, sr = load_audio_bytes_or_path(src)
        return len(wav) / float(sr)


def save_wav(path, wav, sr, max_wav_value=32768.0):
    from scipy.io import wavfile

    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    wavfile.write(path, sr, (wav * (max_wav_value - 1)).astype(np.int16))


def resample(wav, sr_in, sr_out):
    if sr_in == sr_out:
        return wav.astype(np.float32)
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(wav, frac.numerator, frac.denominator).astype(np.float32)


# --------------------------------------------------------------------------- cleaning
def frame_db(wav, frame_length=2048, hop_length=512):
    """Per-frame RMS in dB relative to the loudest frame (librosa.effects.trim semantics)."""
    if len(wav) < frame_length:
        wav = np.pad(wav, (0, frame_length - len(wav)))
    n = 1 + (len(wav) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(wav[idx] ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)


def trim_silence(wav, top_db=40.0, frame_length=2048, hop_length=512, keep_margin_s=0.05, sr=22050):
    """Trim leading/trailing silence quieter than `top_db` below the peak frame.
    A small margin is kept so word onsets/offsets are not clipped."""
    db = frame_db(wav, frame_length, hop_length)
    voiced = np.nonzero(db > -top_db)[0]
    if len(voiced) == 0:
        return wav
    start = max(0, voiced[0] * hop_length - int(keep_margin_s * sr))
    end = min(len(wav), (voiced[-1] + 1) * hop_length + frame_length // 2 + int(keep_margin_s * sr))
    return wav[start:end]


def peak_normalize(wav, peak=0.95):
    m = np.max(np.abs(wav)) if len(wav) else 0.0
    if m < 1e-6:
        return wav
    return (wav / m * peak).astype(np.float32)


# --------------------------------------------------------------------------- mel filterbank
def _hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr, n_fft, n_mels=80, fmin=0.0, fmax=None):
    """Equivalent to librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')."""
    if fmax is None:
        fmax = sr / 2.0
    n_freq = 1 + n_fft // 2
    fftfreqs = np.linspace(0, sr / 2.0, n_freq)
    mel_pts = np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2)
    mel_f = _mel_to_hz_slaney(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, n_freq))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


# --------------------------------------------------------------------------- STFT
class MelExtractor:
    def __init__(self, cfg):
        p = cfg["preprocessing"]
        self.sr = p["audio"]["sampling_rate"]
        self.n_fft = p["stft"]["filter_length"]
        self.hop = p["stft"]["hop_length"]
        self.win = p["stft"]["win_length"]
        self.n_mels = p["mel"]["n_mel_channels"]
        self.window = get_window("hann", self.win, fftbins=True).astype(np.float64)
        if self.win < self.n_fft:  # centre-pad window to n_fft
            lp = (self.n_fft - self.win) // 2
            self.window = np.pad(self.window, (lp, self.n_fft - self.win - lp))
        self.mel_basis = mel_filterbank(self.sr, self.n_fft, self.n_mels, p["mel"]["mel_fmin"], p["mel"]["mel_fmax"])
        self.mel_pinv = np.linalg.pinv(self.mel_basis)

    def stft_mag(self, wav):
        wav = np.clip(np.asarray(wav, dtype=np.float64), -1.0, 1.0)
        pad = self.n_fft // 2
        y = np.pad(wav, (pad, pad), mode="reflect")
        n_frames = 1 + (len(y) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        frames = y[idx] * self.window[None, :]
        spec = np.fft.rfft(frames, n=self.n_fft, axis=1)  # [T, F]
        return np.abs(spec).T.astype(np.float32)  # [F, T]

    def __call__(self, wav):
        """Return (log-mel [n_mels, T], energy [T])."""
        mag = self.stft_mag(wav)
        mel = self.mel_basis @ mag
        log_mel = np.log(np.clip(mel, 1e-5, None)).astype(np.float32)
        energy = np.linalg.norm(mag, axis=0).astype(np.float32)
        return log_mel, energy

    # ------------------------------------------------------------------ Griffin-Lim
    def griffin_lim(self, log_mel, n_iter=60, power=1.2):
        """Fallback vocoder (used only if HiFi-GAN weights are unavailable)."""
        mel = np.exp(log_mel.astype(np.float64))
        mag = np.maximum(self.mel_pinv @ mel, 1e-8) ** power  # [F, T]
        rng = np.random.default_rng(0)
        angles = np.exp(2j * np.pi * rng.random(mag.shape))
        y = self._istft(mag * angles)
        for _ in range(n_iter):
            spec = self._stft_complex(y)
            T = min(spec.shape[1], mag.shape[1])
            angles = np.exp(1j * np.angle(spec[:, :T]))
            y = self._istft(mag[:, :T] * angles)
        return (y / max(1e-6, np.max(np.abs(y))) * 0.95).astype(np.float32)

    def _stft_complex(self, wav):
        pad = self.n_fft // 2
        y = np.pad(wav, (pad, pad), mode="reflect")
        n_frames = 1 + (len(y) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        return np.fft.rfft(y[idx] * self.window[None, :], n=self.n_fft, axis=1).T

    def _istft(self, spec):
        frames = np.fft.irfft(spec.T, n=self.n_fft, axis=1) * self.window[None, :]
        n = frames.shape[0]
        length = self.n_fft + self.hop * (n - 1)
        y = np.zeros(length)
        wsum = np.zeros(length)
        for i in range(n):
            s = i * self.hop
            y[s : s + self.n_fft] += frames[i]
            wsum[s : s + self.n_fft] += self.window ** 2
        y /= np.maximum(wsum, 1e-8)
        pad = self.n_fft // 2
        return y[pad : length - pad]
