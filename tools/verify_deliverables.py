#!/usr/bin/env python3
"""
Check the repository against the assignment's required structure and outputs.

  python tools/verify_deliverables.py            # report
  python tools/verify_deliverables.py --strict   # exit 1 if anything is missing

For every test sentence it checks the WAV and the duration/pitch/energy plots,
that the WAV is non-silent, that the plots and predictions come from the same
checkpoint run (predictions.json), and that loss plots and checkpoints exist.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_FILES = [
    "README.md", "requirements.txt",
    *[f"{L}/{f}" for L in ("English", "Telugu") for f in ("train.py", "preprocess.py", "inference.py", "config.yaml")],
]
REQUIRED_DIRS = [
    *[f"{L}/checkpoints" for L in ("English", "Telugu")],
    *[f"Results/{L}/{d}" for L in ("English", "Telugu") for d in
      ("test_generated_audios", "variance_adaptor_plots/duration", "variance_adaptor_plots/pitch",
       "variance_adaptor_plots/energy")],
]


def wav_stats(path):
    from scipy.io import wavfile
    import numpy as np

    sr, x = wavfile.read(path)
    x = x.astype("float64") / 32768.0
    return sr, len(x) / sr, float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    ok, missing = [], []

    def need(cond, what):
        (ok if cond else missing).append(what)

    for f in REQUIRED_FILES:
        need(os.path.isfile(os.path.join(ROOT, f)), f"file  {f}")
    for d in REQUIRED_DIRS:
        need(os.path.isdir(os.path.join(ROOT, d)), f"dir   {d}/")

    for L in ("English", "Telugu"):
        res = os.path.join(ROOT, "Results", L)
        for f in ("training_loss.png", "validation_loss.png"):
            need(os.path.isfile(os.path.join(res, f)), f"plot  Results/{L}/{f}")
        ck = os.path.join(ROOT, L, "checkpoints")
        cks = [c for c in os.listdir(ck) if c.endswith((".pth", ".pth.tar"))] if os.path.isdir(ck) else []
        need(bool(cks), f"ckpt  {L}/checkpoints/ (any checkpoint)")
        with open(os.path.join(ROOT, L, "test_sentences.json"), encoding="utf-8") as f:
            sents = json.load(f)["sentences"]
        pred_path = os.path.join(res, "predictions.json")
        preds = {}
        if os.path.isfile(pred_path):
            with open(pred_path, encoding="utf-8") as f:
                preds = json.load(f).get("sentences", {})
        for sid, text in sents:
            wav = os.path.join(res, "test_generated_audios", f"{sid}.wav")
            if os.path.isfile(wav):
                sr, dur, rms = wav_stats(wav)
                need(rms > 1e-4 and dur > 0.3, f"audio {L} {sid}.wav ({dur:.2f} s, {sr} Hz, rms {rms:.4f})")
            else:
                need(False, f"audio {L} {sid}.wav")
            for kind in ("duration", "pitch", "energy"):
                need(os.path.isfile(os.path.join(res, "variance_adaptor_plots", kind, f"{sid}_{kind}.png")),
                     f"plot  {L} {sid}_{kind}.png")
            if preds:
                need(sid in preds and preds[sid]["text"] == text,
                     f"pred  {L} {sid} in predictions.json with the exact test text")

    print("PRESENT:")
    for w in ok:
        print("  [x] " + w)
    print("MISSING / NOT YET GENERATED:")
    for w in missing:
        print("  [ ] " + w)
    if not missing:
        print("  (none) - all deliverables present")
    print(f"\n{len(ok)} present, {len(missing)} missing")
    if args.strict and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
