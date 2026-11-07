"""
Feature extraction (stage 3 of preprocess.py), adapted from ming024/FastSpeech2
preprocessor/preprocessor.py.

For every aligned utterance:
  phones + frame durations  <- MFA TextGrid (leading/trailing silence trimmed)
  mel (80 x T), energy      <- numpy STFT (fs2/audio.py)
  F0                        <- PyWorld DIO + StoneMask (fs2/pitch.py)
  pitch / energy are linearly interpolated over unvoiced frames and, for
  "phoneme_level" features, averaged over each phoneme's frames.

Changes vs. the reference implementation:
  * train/val membership comes from split.json written during data selection
    (hours-budgeted, no overlap) instead of a random val_size draw;
  * pitch/energy normalisation statistics are computed on the TRAIN split only;
  * utterances with an "spn" (unknown word) alignment are skipped, not silently
    turned into pauses; skip reasons are counted and reported;
  * the vocabulary is the language frontend's full phone inventory, so tokens
    that never occur in training are still representable at inference.
"""
import os
from collections import Counter

import numpy as np

from .audio import MelExtractor, load_audio_bytes_or_path
from .pitch import HAVE_PYWORLD, extract_f0, interpolate_unvoiced
from .textgrid import get_alignment, phone_tier, read_textgrid
from .utils import read_json, write_json


def remove_outlier(values):
    values = np.array(values)
    p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
    lower, upper = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
    return values[np.logical_and(values > lower, values < upper)]


def average_by_duration(x, durations):
    out = np.zeros(len(durations), dtype=np.float32)
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            out[i] = np.mean(x[pos : pos + d])
        pos += d
    return out


class Preprocessor:
    def __init__(self, cfg, frontend, log=print):
        self.cfg = cfg
        self.frontend = frontend
        self.log = log
        p = cfg["preprocessing"]
        self.sr = p["audio"]["sampling_rate"]
        self.hop = p["stft"]["hop_length"]
        self.pitch_level = p["pitch"]["feature"]
        self.energy_level = p["energy"]["feature"]
        self.f0_min = p["pitch"].get("f0_min", 60)
        self.f0_max = p["pitch"].get("f0_max", 600)
        self.max_frames = cfg["model"]["max_seq_len"]
        self.min_frames = p.get("min_frames", 20)
        self.mel = MelExtractor(cfg)
        self.raw = cfg["path"]["raw_path"]
        self.tg = cfg["path"]["textgrid_path"]
        self.out = cfg["path"]["preprocessed_path"]

    def _find_textgrid(self, speaker, uid):
        for cand in (os.path.join(self.tg, speaker, uid + ".TextGrid"), os.path.join(self.tg, uid + ".TextGrid")):
            if os.path.exists(cand):
                return cand
        return None

    def process_utterance(self, speaker, uid):
        tg_path = self._find_textgrid(speaker, uid)
        if tg_path is None:
            return None, "no_textgrid"
        try:
            phones, durations, start, end = get_alignment(phone_tier(read_textgrid(tg_path)), self.sr, self.hop)
        except ValueError:
            return None, "spn_in_alignment"
        if start >= end or len(phones) == 0:
            return None, "empty_alignment"
        unknown = [ph for ph in phones if ph not in self.frontend.vocab_index]
        if unknown:
            return None, "phone_not_in_inventory:" + unknown[0]

        wav, sr = load_audio_bytes_or_path(os.path.join(self.raw, speaker, uid + ".wav"))
        assert sr == self.sr, f"{uid}: expected {self.sr} Hz, got {sr}"
        wav = wav[int(self.sr * start) : int(self.sr * end)].astype(np.float32)
        n = int(sum(durations))
        if n < self.min_frames:
            return None, "too_short"
        if n > self.max_frames or len(phones) > self.max_frames:
            return None, "too_long"

        # F0
        pitch = extract_f0(wav, self.sr, self.hop, self.f0_min, self.f0_max, n_frames=n)
        if np.sum(pitch != 0) <= 1:
            return None, "no_voiced_frames"

        # mel + energy (T may exceed n by one frame; see textgrid.get_alignment)
        mel, energy = self.mel(wav)
        if mel.shape[1] < n:
            padw = n - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, padw)), mode="edge")
            energy = np.pad(energy, (0, padw), mode="edge")
        mel, energy = mel[:, :n], energy[:n]

        pitch = interpolate_unvoiced(pitch)
        if self.pitch_level == "phoneme_level":
            pitch = average_by_duration(pitch, durations)
        if self.energy_level == "phoneme_level":
            energy = average_by_duration(energy, durations)

        d = self.out
        np.save(os.path.join(d, "duration", f"{speaker}-duration-{uid}.npy"), np.array(durations, dtype=np.int64))
        np.save(os.path.join(d, "pitch", f"{speaker}-pitch-{uid}.npy"), pitch.astype(np.float32))
        np.save(os.path.join(d, "energy", f"{speaker}-energy-{uid}.npy"), energy.astype(np.float32))
        np.save(os.path.join(d, "mel", f"{speaker}-mel-{uid}.npy"), mel.T.astype(np.float32))
        return {
            "uid": uid,
            "speaker": speaker,
            "phones": phones,
            "frames": n,
            "pitch": remove_outlier(pitch) if len(pitch) > 4 else pitch,
            "energy": remove_outlier(energy) if len(energy) > 4 else energy,
        }, None

    def build_from_corpus(self):
        for sub in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        split = read_json(os.path.join(self.raw, "split.json"))
        uid2spk = {}
        for spk in sorted(os.listdir(self.raw)):
            sd = os.path.join(self.raw, spk)
            if os.path.isdir(sd):
                for fn in os.listdir(sd):
                    if fn.endswith(".wav"):
                        uid2spk[fn[:-4]] = spk
        uid2text = {}
        meta = os.path.join(self.raw, "metadata.tsv")
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 5:
                        uid2text[parts[0]] = parts[4]

        results = {"train": [], "val": []}
        skipped = Counter()
        frames = {"train": 0, "val": 0}
        for part in ("train", "val"):
            uids = split[part]
            for k, uid in enumerate(uids):
                if uid not in uid2spk:
                    skipped["missing_wav"] += 1
                    continue
                info, why = self.process_utterance(uid2spk[uid], uid)
                if info is None:
                    skipped[why] += 1
                    continue
                info["text"] = uid2text.get(uid, "")
                results[part].append(info)
                frames[part] += info["frames"]
                if (k + 1) % 500 == 0:
                    self.log(f"[features] {part}: {k + 1}/{len(uids)}")

        if not results["train"]:
            raise RuntimeError(f"no usable training utterances (skip reasons: {dict(skipped)})")

        # ---- normalisation statistics: TRAIN split only
        stats = {}
        for key in ("pitch", "energy"):
            vals = np.concatenate([r[key] for r in results["train"]]).astype(np.float64)
            mean, std = float(vals.mean()), float(vals.std())
            if std < 1e-8:
                std = 1.0
            vmin, vmax = np.inf, -np.inf
            for part in ("train", "val"):
                for r in results[part]:
                    fn = os.path.join(self.out, key, f"{r['speaker']}-{key}-{r['uid']}.npy")
                    v = (np.load(fn) - mean) / std
                    np.save(fn, v.astype(np.float32))
                    vmin, vmax = min(vmin, float(v.min())), max(vmax, float(v.max()))
            stats[key] = [vmin, vmax, mean, std]

        # ---- metadata files
        speakers = sorted({r["speaker"] for p in results.values() for r in p})
        spk_map = {s: i for i, s in enumerate(speakers)}
        spk_hours = Counter()
        for r in results["train"]:
            spk_hours[r["speaker"]] += r["frames"] * self.hop / self.sr / 3600.0
        for part, fname in (("train", "train.txt"), ("val", "val.txt")):
            with open(os.path.join(self.out, fname), "w", encoding="utf-8") as f:
                for r in results[part]:
                    text = r["text"].replace("|", " ").replace("\n", " ")
                    f.write(f"{r['uid']}|{r['speaker']}|{' '.join(r['phones'])}|{text}\n")
        write_json(stats, os.path.join(self.out, "stats.json"))
        write_json(spk_map, os.path.join(self.out, "speakers.json"))
        write_json({k: round(v, 4) for k, v in spk_hours.items()}, os.path.join(self.out, "speaker_train_hours.json"))
        write_json(self.frontend.vocab, os.path.join(self.out, "vocab.json"))

        phone_counts = Counter(ph for r in results["train"] for ph in r["phones"])
        summary = {
            "train_utterances": len(results["train"]),
            "val_utterances": len(results["val"]),
            "train_hours": round(frames["train"] * self.hop / self.sr / 3600, 3),
            "val_hours": round(frames["val"] * self.hop / self.sr / 3600, 3),
            "n_speakers": len(speakers),
            "skipped": dict(skipped),
            "stats": stats,
            "vocab_size": len(self.frontend.vocab),
            "phones_seen_in_train": len(phone_counts),
            "phones_never_seen_in_train": [p for p in self.frontend.phone_inventory if p not in phone_counts],
            "f0_method": "pyworld DIO + StoneMask" if HAVE_PYWORLD else "numpy autocorrelation (FALLBACK - install pyworld)",
            "pitch_feature": self.pitch_level,
            "energy_feature": self.energy_level,
        }
        write_json(summary, os.path.join(self.out, "preprocess_summary.json"))
        self.log(f"[features] train {summary['train_utterances']} utts / {summary['train_hours']} h, "
                 f"val {summary['val_utterances']} utts / {summary['val_hours']} h, skipped {dict(skipped)}")
        return summary
