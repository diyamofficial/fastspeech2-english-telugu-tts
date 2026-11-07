"""
Shared driver for the three preprocessing stages used by both languages:

  prepare   Hugging Face dataset -> hours-budgeted train/val selection ->
            MFA corpus (22.05 kHz mono, trimmed, peak-normalised wav + .lab) + lexicon
  align     Montreal Forced Aligner -> TextGrids (phone durations)
  features  TextGrids + audio -> mel, energy, F0, durations, stats, vocab, train/val lists
"""
import datetime
import hashlib
import os
import re

from .data_prep import Item, select_split, write_corpus
from .features import Preprocessor
from .mfa import run_mfa
from .utils import write_json


def make_logger(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}.log")

    def log(msg):
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def _safe(s):
    return re.sub(r"[^A-Za-z0-9]+", "", str(s))[:24] or "x"


def hf_items(cfg, log, uid_prefix, keep_text=lambda t: True, limit=None):
    """
    Load a Hugging Face audio dataset WITHOUT decoding audio (Audio(decode=False)),
    so selection only touches metadata; audio bytes are read lazily for the
    selected utterances. Returns a list of fs2.data_prep.Item.
    """
    from datasets import Audio, load_dataset

    d = cfg["dataset"]
    log(f"[prepare] loading {d['hf_name']} split={d['hf_split']}")
    try:
        ds = load_dataset(d["hf_name"], split=d["hf_split"], token=os.environ.get("HF_TOKEN") or None)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if any(k in msg for k in ("Gated", "gated", "401", "403", "authenticat", "Unauthorized", "token")):
            raise RuntimeError(
                f"Cannot download {d['hf_name']}: access denied.\n"
                f"  1. While logged in to huggingface.co, open https://huggingface.co/datasets/{d['hf_name']} "
                f"and accept the access conditions.\n"
                f"  2. Create a READ token at https://huggingface.co/settings/tokens and log in with it "
                f"(notebook cell 4: login()), or export HF_TOKEN=hf_...\n"
                f"  Original error: {msg}"
            ) from e
        raise
    acol, tcol = d["audio_column"], d["text_column"]
    ds = ds.cast_column(acol, Audio(decode=False))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    spk_cols = [c for c in d.get("speaker_columns", []) if c in ds.column_names]
    missing = [c for c in d.get("speaker_columns", []) if c not in ds.column_names]
    if missing:
        log(f"[prepare] WARNING speaker columns not in dataset: {missing}")
    class_names = {c: ds.features[c].names for c in spk_cols if hasattr(ds.features[c], "names")}
    has_duration = "duration" in ds.column_names

    def speaker_of(row):
        if "speaker_id" in row and row["speaker_id"] not in (None, ""):
            return "spk" + _safe(row["speaker_id"])
        vals = []
        for c in spk_cols:
            v = row[c]
            if c in class_names and isinstance(v, int):
                v = class_names[c][v]
            vals.append(str(v))
        if len(vals) == 1:
            return _safe(vals[0])
        return "spk" + hashlib.md5("|".join(vals).encode("utf-8")).hexdigest()[:8]

    def make_loader(idx):
        def _load():
            a = ds[idx][acol]
            return a["bytes"] if a.get("bytes") else a["path"]
        return _load

    from .audio import audio_duration_from_bytes

    items, dropped_text = [], 0
    meta_cols = [c for c in ds.column_names if c != acol]
    meta = ds.select_columns(meta_cols)
    for i, row in enumerate(meta):
        text = row[tcol] or ""
        if not keep_text(text):
            dropped_text += 1
            continue
        if has_duration and row.get("duration"):
            dur = float(row["duration"])
        else:
            dur = audio_duration_from_bytes(make_loader(i)())
        items.append(Item(uid=f"{uid_prefix}{i:05d}", speaker=speaker_of(row), duration=dur, text=text,
                          loader=make_loader(i)))
        if (i + 1) % 1000 == 0:
            log(f"[prepare] scanned {i + 1}/{len(meta)} rows")
    log(f"[prepare] {len(items)} candidate utterances, {dropped_text} dropped by the text filter")
    return items, {"rows_in_dataset": len(ds), "dropped_by_text_filter": dropped_text,
                   "speaker_columns_used": spk_cols}


def run_preprocess(cfg, frontend, stage, load_items, log):
    stages = ["prepare", "align", "features"] if stage == "all" else [stage]
    out = {}
    if "prepare" in stages:
        items, extra = load_items()
        # drop transcripts that normalise to nothing before selection
        items = [it for it in items if frontend.words_for_alignment(it.text)]
        train, val, sel_info = select_split(items, cfg, log)
        extra.update(sel_info)
        extra["dataset"] = cfg["dataset"]["hf_name"]
        extra["selection_policy"] = ("speaker-stratified random sampling (seed {}), hours budget "
                                     "{} h train / {} h val, no utterance in both splits"
                                     .format(cfg.get("seed", 1234), cfg["dataset"]["train_hours"],
                                             cfg["dataset"]["val_hours"]))
        out["prepare"] = write_corpus(train, val, frontend, cfg, extra, log)
    if "align" in stages:
        out["align"] = run_mfa(cfg, log)
    if "features" in stages:
        summary = Preprocessor(cfg, frontend, log).build_from_corpus()
        res = cfg["path"]["result_path"]
        sel = os.path.join(cfg["path"]["raw_path"], "selection_summary.json")
        os.makedirs(res, exist_ok=True)
        merged = {"features": summary}
        if os.path.exists(sel):
            from .utils import read_json

            merged["selection"] = read_json(sel)
        write_json(merged, os.path.join(res, "data_summary.json"))
        out["features"] = summary
    return out
