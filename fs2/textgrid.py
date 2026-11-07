"""
Minimal Praat TextGrid reader (long and short text formats, as written by the
Montreal Forced Aligner), plus the duration-extraction logic of the reference
implementation (ming024 preprocessor.get_alignment).

Replaces the `tgt` dependency of the reference implementation.
"""
import re

import numpy as np

_TOKEN = re.compile(r'"(?:[^"]|"")*"|<exists>|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

# MFA 1.x writes "sil"/"sp" for silence, MFA 2.x/3.x writes an empty label.
SILENCE = {"", "sil", "sp", "<eps>"}
UNKNOWN = {"spn"}


def read_textgrid(path):
    """Return {tier_name: [(xmin, xmax, text), ...]} for every IntervalTier."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    if "\x00" in raw:  # UTF-16 files written by Praat itself
        with open(path, "r", encoding="utf-16") as f:
            raw = f.read()
    raw = re.sub(r"\[\s*\d*\s*\]", " ", raw)  # drop "item [1]:" / "intervals [3]:" indices
    toks = [t.group(0) for t in _TOKEN.finditer(raw)]

    def val(t):
        if t.startswith('"'):
            return t[1:-1].replace('""', '"')
        return t

    toks = [val(t) for t in toks]
    i = 0
    assert toks[i] == "ooTextFile", f"{path}: not a Praat text file"
    i += 1
    assert toks[i] == "TextGrid", f"{path}: not a TextGrid"
    i += 1
    i += 2  # xmin xmax
    if toks[i] == "<exists>":
        i += 1
    n_tiers = int(float(toks[i]))
    i += 1
    tiers = {}
    for _ in range(n_tiers):
        cls, name = toks[i], toks[i + 1]
        i += 2
        i += 2  # tier xmin xmax
        n = int(float(toks[i]))
        i += 1
        if cls == "IntervalTier":
            ivs = []
            for _ in range(n):
                ivs.append((float(toks[i]), float(toks[i + 1]), toks[i + 2].strip()))
                i += 3
            tiers[name] = ivs
        else:  # TextTier: (time, mark)
            i += 2 * n
    return tiers


def phone_tier(tiers):
    for name in tiers:
        if name.strip().lower().endswith("phones"):
            return tiers[name]
    raise KeyError(f"no phones tier found (tiers: {list(tiers)})")


def get_alignment(intervals, sampling_rate, hop_length):
    """
    Convert MFA phone intervals to (phones, frame durations, start_s, end_s).

    As in the reference implementation: leading and trailing silences are
    trimmed (start/end are used to crop the waveform), internal silences become
    the pause token "sp", and durations are computed from rounded frame
    boundaries so that they sum exactly to the number of frames in [start, end].
    Raises ValueError if MFA marked a word as unknown ("spn").
    """
    phones, durations = [], []
    start_time, end_time, end_idx = 0.0, 0.0, 0
    for s, e, p in intervals:
        p = p.strip()
        if p in UNKNOWN:
            raise ValueError("alignment contains spn (out-of-dictionary word)")
        if phones == []:
            if p in SILENCE:
                continue
            start_time = s
        if p not in SILENCE:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append("sp")
        durations.append(
            int(np.round(e * sampling_rate / hop_length) - np.round(s * sampling_rate / hop_length))
        )
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    return phones, durations, start_time, end_time
