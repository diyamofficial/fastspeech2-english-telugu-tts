"""
Dataset selection and MFA-corpus writing, shared by both languages.

Each language's preprocess.py turns its Hugging Face dataset into a list of
`Item`s (uid, speaker, duration, text, audio source) and hands it to:

  select_split(...)   -> deterministic, hours-budgeted, non-overlapping train/val
  write_corpus(...)   -> MFA corpus: <raw_path>/<speaker>/<uid>.wav + .lab,
                         lexicon.txt, split.json, metadata.tsv, selection_summary.json
"""
import csv
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .audio import load_audio_bytes_or_path, peak_normalize, resample, save_wav, trim_silence
from .utils import write_json


@dataclass
class Item:
    uid: str
    speaker: str
    duration: float  # seconds, of the ORIGINAL audio
    text: str
    audio: Any = None  # raw bytes or a path
    loader: Optional[Callable[[], Any]] = None  # lazily returns bytes/path
    meta: dict = field(default_factory=dict)

    def audio_source(self):
        return self.audio if self.audio is not None else self.loader()


def _hours(items):
    return sum(i.duration for i in items) / 3600.0


def select_split(items, cfg, log=print):
    """
    Hours-budgeted, speaker-stratified selection with no train/val overlap.

    1. Filter by duration range and non-empty normalised text (done by caller for text).
    2. Speaker policy:
         "all"    : keep every speaker (multi-speaker model, speaker embedding on)
         "single" : keep only the speaker with the most audio
         "auto"   : "single" if that speaker alone covers train+val hours, else "all"
         [names]  : explicit list of speaker labels
    3. For every kept speaker, shuffle its utterances with a fixed seed and take a
       share of the total budget proportional to the speaker's available audio.
    4. Within each speaker, the first ~val/(train+val) of the selected audio goes to
       validation and the rest to training. Every validation speaker is therefore
       also a training speaker (needed for the speaker embedding), but no utterance
       is ever in both splits.
    """
    d = cfg["dataset"]
    rng = random.Random(cfg.get("seed", 1234))
    tr_h, va_h = float(d["train_hours"]), float(d["val_hours"])
    over = float(d.get("oversample", 1.04))  # small margin for utterances dropped later (MFA/filters)
    target_s = (tr_h + va_h) * 3600.0 * over
    val_frac = va_h / (tr_h + va_h)

    lo, hi = float(d["min_duration"]), float(d["max_duration"])
    items = [i for i in items if lo <= i.duration <= hi and i.text.strip()]
    by_spk = OrderedDict()
    for it in sorted(items, key=lambda x: x.uid):
        by_spk.setdefault(it.speaker, []).append(it)
    spk_hours = {s: _hours(v) for s, v in by_spk.items()}
    log(f"[select] {len(items)} utterances / {_hours(items):.2f} h after duration filter "
        f"[{lo}, {hi}] s, {len(by_spk)} speakers")

    policy = d.get("speaker_policy", "all")
    if isinstance(policy, list):
        keep = [s for s in by_spk if s in set(map(str, policy))]
    elif policy in ("single", "auto"):
        best = max(spk_hours, key=spk_hours.get)
        if policy == "single" or spk_hours[best] * 3600.0 >= target_s:
            keep = [best]
        else:
            keep = list(by_spk)
    else:
        keep = list(by_spk)
    log(f"[select] speaker policy={policy!r} -> {len(keep)} speaker(s) kept")
    avail_s = sum(spk_hours[s] for s in keep) * 3600.0
    if avail_s < target_s:
        log(f"[select] WARNING: only {avail_s/3600:.2f} h available for a "
            f"{target_s/3600:.2f} h budget; using everything.")
    share = min(1.0, target_s / max(avail_s, 1e-9))

    train, val = [], []
    for s in keep:
        utts = list(by_spk[s])
        rng.shuffle(utts)
        budget = spk_hours[s] * 3600.0 * share
        chosen, acc = [], 0.0
        for u in utts:
            if acc >= budget:
                break
            chosen.append(u)
            acc += u.duration
        if len(chosen) < 2:  # a speaker needs >=1 train utterance; keep it train-only
            train += chosen
            continue
        v_budget, v_acc = acc * val_frac, 0.0
        for idx, u in enumerate(chosen):
            # the last utterance of every speaker always goes to train
            if v_acc < v_budget and idx < len(chosen) - 1:
                val.append(u)
                v_acc += u.duration
            else:
                train.append(u)
    assert not ({u.uid for u in train} & {u.uid for u in val}), "train/val overlap"
    log(f"[select] train: {len(train)} utts / {_hours(train):.2f} h | "
        f"val: {len(val)} utts / {_hours(val):.2f} h (before trimming/alignment)")
    return train, val, {"speaker_hours_available": spk_hours, "speakers_kept": keep, "policy": policy}


def write_corpus(train, val, frontend, cfg, extra_summary=None, log=print):
    """Decode, resample, trim, normalise and write the MFA corpus + lexicon."""
    raw = cfg["path"]["raw_path"]
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    top_db = cfg["preprocessing"]["audio"].get("trim_top_db", 40)
    os.makedirs(raw, exist_ok=True)

    words = set()
    rows, split = [], {"train": [], "val": []}
    dropped = defaultdict(int)
    total = len(train) + len(val)
    for name, part in (("train", train), ("val", val)):
        for it in part:
            norm_words = frontend.words_for_alignment(it.text)
            if not norm_words:
                dropped["empty_text_after_normalisation"] += 1
                continue
            try:
                wav, in_sr = load_audio_bytes_or_path(it.audio_source())
            except Exception as e:  # corrupt audio
                dropped[f"decode_error:{type(e).__name__}"] += 1
                continue
            wav = resample(wav, in_sr, sr)
            wav = trim_silence(wav, top_db=top_db, sr=sr)
            wav = peak_normalize(wav)
            if len(wav) < 0.3 * sr:
                dropped["too_short_after_trim"] += 1
                continue
            spk_dir = os.path.join(raw, it.speaker)
            os.makedirs(spk_dir, exist_ok=True)
            save_wav(os.path.join(spk_dir, it.uid + ".wav"), wav, sr)
            with open(os.path.join(spk_dir, it.uid + ".lab"), "w", encoding="utf-8") as f:
                f.write(" ".join(norm_words))
            words.update(norm_words)
            split[name].append(it.uid)
            rows.append([it.uid, it.speaker, name, f"{len(wav)/sr:.3f}", it.text, " ".join(norm_words)])
            if len(rows) % 500 == 0:
                log(f"[corpus] {len(rows)}/{total} written")

    lex_path = cfg["path"]["lexicon_path"]
    n_entries = 0
    with open(lex_path, "w", encoding="utf-8") as f:
        for w in sorted(words):
            prons = frontend.lexicon_entries(w)
            for p in prons:
                if p:
                    f.write(f"{w}\t{' '.join(p)}\n")
                    n_entries += 1
    write_json(split, os.path.join(raw, "split.json"))
    with open(os.path.join(raw, "metadata.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["uid", "speaker", "split", "seconds_after_trim", "original_text", "normalised_words"])
        w.writerows(rows)

    secs = defaultdict(float)
    for r in rows:
        secs[r[2]] += float(r[3])
    summary = {
        "train_utterances": len(split["train"]),
        "val_utterances": len(split["val"]),
        "train_hours_after_trim": round(secs["train"] / 3600, 3),
        "val_hours_after_trim": round(secs["val"] / 3600, 3),
        "speakers": sorted({r[1] for r in rows}),
        "lexicon_words": len(words),
        "lexicon_entries": n_entries,
        "dropped": dict(dropped),
    }
    if extra_summary:
        summary.update(extra_summary)
    write_json(summary, os.path.join(raw, "selection_summary.json"))
    log(f"[corpus] done: {summary['train_utterances']} train ({summary['train_hours_after_trim']} h), "
        f"{summary['val_utterances']} val ({summary['val_hours_after_trim']} h); dropped={dict(dropped)}")
    return summary
