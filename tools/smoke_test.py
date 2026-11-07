#!/usr/bin/env python3
"""
End-to-end smoke test on SYNTHETIC data (no downloads, no MFA, ~1-3 min).

It fabricates a tiny corpus whose phone alignments are known exactly (one tone or
noise burst per phone, written as MFA-style TextGrids), then runs the real code:

  1. feature extraction (fs2.features)  -> checks mel/pitch/energy/duration shapes
     and that the durations recovered from the TextGrids equal the generated ones
  2. mel filterbank vs librosa (if librosa is installed)
  3. if PyTorch is installed: a short training run (loss must stay finite, checkpoint
     + loss plots written, resume works), then inference on the real assignment test
     sentences, with audio + duration/pitch/energy plots

All outputs go to a temporary directory; nothing in Results/ is touched. The
synthetic "speech" is meaningless - this only proves the code paths run.

  python tools/smoke_test.py --language telugu
  python tools/smoke_test.py --language english   (needs g2p_en)
  python tools/smoke_test.py --language both --keep
"""
import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fs2.audio import MelExtractor, save_wav  # noqa: E402
from fs2.utils import load_config, read_json, write_json  # noqa: E402

VOWELS_EN = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
VOWELS_TE = {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"}

SAMPLE_TEXT = {
    "english": ["the cat sat on the mat", "a quick brown fox jumps over the lazy dog",
                "she sells sea shells by the shore", "we will meet at noon today",
                "good morning everyone and welcome", "this is only a test of the system"],
    "telugu": ["నేను ఇంటికి వెళ్తున్నాను", "ఈ రోజు వాతావరణం బాగుంది", "మీరు ఎలా ఉన్నారు",
               "పిల్లలు బడికి వెళ్లారు", "ఆమె పాట పాడింది", "మనం రేపు కలుద్దాం"],
}


def frontend_for(lang):
    sys.path.insert(0, os.path.join(ROOT, lang.capitalize()))
    if lang == "english":
        from english_frontend import EnglishFrontend

        return EnglishFrontend()
    from telugu_frontend import TeluguFrontend

    return TeluguFrontend()


def is_vowel(p):
    return p.rstrip("012") in VOWELS_EN or p in VOWELS_TE


def write_textgrid(path, words, phone_ivs, total):
    def tier(name, ivs):
        s = f'    item [{ {"words": 1, "phones": 2}[name]} ]:\n        class = "IntervalTier"\n        name = "{name}"\n'
        s += f"        xmin = 0\n        xmax = {total:.6f}\n        intervals: size = {len(ivs)}\n"
        for k, (a, b, t) in enumerate(ivs):
            s += f'        intervals [{k + 1}]:\n            xmin = {a:.6f}\n            xmax = {b:.6f}\n            text = "{t}"\n'
        return s

    with open(path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
        f.write(f"xmin = 0\nxmax = {total:.6f}\ntiers? <exists>\nsize = 2\nitem []:\n")
        f.write(tier("words", words))
        f.write(tier("phones", phone_ivs))


def make_corpus(cfg, fe, lang, n_utts, rng, log):
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    hop = cfg["preprocessing"]["stft"]["hop_length"]
    raw, tgd = cfg["path"]["raw_path"], cfg["path"]["textgrid_path"]
    spk = "spkA"
    os.makedirs(os.path.join(raw, spk), exist_ok=True)
    os.makedirs(os.path.join(tgd, spk), exist_ok=True)
    truth, split, rows = {}, {"train": [], "val": []}, []
    texts = SAMPLE_TEXT[lang]
    for u in range(n_utts):
        text = texts[u % len(texts)]
        words = fe.words_for_alignment(text)
        lex = [fe.lexicon_entries(w)[0] for w in words]
        # frames: leading silence, phones (with an internal pause after word 2), trailing silence
        segs = [("", int(rng.integers(8, 15)))]
        word_ivs_frames = []
        for wi, (w, pr) in enumerate(zip(words, lex)):
            start = sum(d for _, d in segs)
            for p in pr:
                segs.append((p, int(rng.integers(3, 12))))
            word_ivs_frames.append((start, sum(d for _, d in segs), w))
            if wi == 1:
                segs.append(("", int(rng.integers(5, 10))))
        segs.append(("", int(rng.integers(8, 15))))
        n_frames = sum(d for _, d in segs)
        wav = np.zeros(n_frames * hop, dtype=np.float64)
        t0 = 0
        f0_base = rng.uniform(110, 220)
        for k, (p, d) in enumerate(segs):
            n = d * hop
            t = np.arange(n) / sr
            if p == "":
                seg = 0.002 * rng.standard_normal(n)
            elif is_vowel(p):
                f0 = f0_base * (1 + 0.15 * np.sin(k))
                seg = sum(0.3 / h * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 6))
            else:
                seg = 0.15 * rng.standard_normal(n)
            wav[t0 : t0 + n] = seg
            t0 += n
        uid = f"syn{u:04d}"
        save_wav(os.path.join(raw, spk, uid + ".wav"), wav * 0.9 / np.max(np.abs(wav)), sr)
        with open(os.path.join(raw, spk, uid + ".lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(words))
        sec = lambda fr: fr * hop / sr  # noqa: E731
        phone_ivs, acc = [], 0
        for p, d in segs:
            phone_ivs.append((sec(acc), sec(acc + d), p))
            acc += d
        word_ivs = [(sec(a), sec(b), w) for a, b, w in word_ivs_frames]
        write_textgrid(os.path.join(tgd, spk, uid + ".TextGrid"), word_ivs, phone_ivs, sec(n_frames))
        inner = segs[1:-1]
        truth[uid] = ([p if p else "sp" for p, _ in inner], [d for _, d in inner])
        (split["val"] if u % 5 == 0 else split["train"]).append(uid)
        rows.append([uid, spk, "val" if u % 5 == 0 else "train", f"{n_frames * hop / sr:.3f}", text, " ".join(words)])
    write_json(split, os.path.join(raw, "split.json"))
    with open(os.path.join(raw, "metadata.tsv"), "w", encoding="utf-8") as f:
        f.write("uid\tspeaker\tsplit\tseconds_after_trim\toriginal_text\tnormalised_words\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    log(f"  synthetic corpus: {n_utts} utterances ({len(split['train'])} train / {len(split['val'])} val)")
    return truth


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK  {msg}")


def run(lang, args):
    print(f"\n=== smoke test: {lang} ===")
    cfg = load_config(os.path.join(ROOT, lang.capitalize(), "config.yaml"))
    work = tempfile.mkdtemp(prefix=f"fs2_smoke_{lang}_")
    cfg = copy.deepcopy(cfg)
    for k, sub in [("raw_path", "corpus"), ("textgrid_path", "TextGrid"), ("preprocessed_path", "preprocessed"),
                   ("checkpoint_path", "checkpoints"), ("log_path", "logs"), ("result_path", "Results"),
                   ("lexicon_path", "lexicon.txt")]:
        cfg["path"][k] = os.path.join(work, sub)
    # tiny model / schedule so CPU training is quick (architecture unchanged otherwise)
    cfg["train"].update({"batch_size": 4, "group_size": 2, "total_step": args.steps, "log_step": 5,
                         "val_step": max(10, args.steps // 2), "save_step": max(10, args.steps // 2),
                         "num_workers": 0})
    cfg["train"]["optimizer"]["warm_up_step"] = 10
    cfg["train"]["optimizer"]["anneal_steps"] = []
    print(f"  workdir: {work}")
    fe = frontend_for(lang)
    rng = np.random.default_rng(0)
    truth = make_corpus(cfg, fe, lang, args.utts, rng, print)

    # ---- features
    from fs2.features import Preprocessor

    summary = Preprocessor(cfg, fe, log=lambda m: print("  " + m)).build_from_corpus()
    pre = cfg["path"]["preprocessed_path"]
    check(summary["train_utterances"] + summary["val_utterances"] == args.utts,
          f"all {args.utts} synthetic utterances preprocessed (skipped: {summary['skipped']})")
    with open(os.path.join(pre, "train.txt"), encoding="utf-8") as f:
        lines = [l.rstrip("\n").split("|") for l in f]
    for uid, spk, phones, _ in lines:
        dur = np.load(os.path.join(pre, "duration", f"{spk}-duration-{uid}.npy"))
        mel = np.load(os.path.join(pre, "mel", f"{spk}-mel-{uid}.npy"))
        pitch = np.load(os.path.join(pre, "pitch", f"{spk}-pitch-{uid}.npy"))
        energy = np.load(os.path.join(pre, "energy", f"{spk}-energy-{uid}.npy"))
        tp, td = truth[uid]
        assert phones.split() == tp, (uid, phones, tp)
        assert dur.tolist() == td, (uid, dur.tolist(), td)
        assert mel.shape == (dur.sum(), 80), (mel.shape, dur.sum())
        assert pitch.shape == energy.shape == (len(tp),)
        assert np.all(np.isfinite(mel)) and np.all(np.isfinite(pitch)) and np.all(np.isfinite(energy))
    check(True, "TextGrid phones/durations recovered exactly; mel=[sum(dur),80]; pitch/energy per phoneme; all finite")
    stats = read_json(os.path.join(pre, "stats.json"))
    check(all(np.isfinite(v) for k in stats for v in stats[k]) and stats["pitch"][3] > 0, f"stats.json finite: {stats}")
    check(read_json(os.path.join(pre, "vocab.json")) == fe.vocab, f"vocab.json == frontend vocab ({len(fe.vocab)} symbols)")

    try:
        import librosa

        ours = MelExtractor(cfg).mel_basis
        ref = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
        check(np.allclose(ours, ref, atol=1e-6), "mel filterbank identical to librosa.filters.mel (slaney)")
    except ImportError:
        print("  --  librosa not installed: mel filterbank comparison skipped")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  --  PyTorch not installed: training / inference part skipped")
        return work

    # ---- training
    from fs2.trainer import train

    log = lambda m: print("  " + m)  # noqa: E731
    half = max(10, args.steps // 2)
    train(cfg, fe, max_steps=half, device=args.device, log=log)
    ck = cfg["path"]["checkpoint_path"]
    check(os.path.exists(os.path.join(ck, "latest.pth.tar")), "checkpoint saved")
    train(cfg, fe, restore="latest", max_steps=args.steps, device=args.device, log=log)
    import csv

    with open(os.path.join(cfg["path"]["log_path"], "train_log.csv")) as f:
        tl = list(csv.DictReader(f))
    check(len(tl) > 0 and all(np.isfinite(float(r["total"])) for r in tl),
          f"resume works and training losses are finite ({len(tl)} log rows, last total={tl[-1]['total']})")
    for f in ("training_loss.png", "validation_loss.png"):
        check(os.path.exists(os.path.join(cfg["path"]["result_path"], f)), f"{f} written")
    check(any(n.endswith("_fp16.pth") for n in os.listdir(ck)), "fp16 inference export written")

    # ---- inference on the real test sentences (vocoder: HiFi-GAN if present, else Griffin-Lim)
    from fs2.synthesizer import run_test_set

    with open(os.path.join(ROOT, lang.capitalize(), "test_sentences.json"), encoding="utf-8") as f:
        sents = [tuple(s) for s in json.load(f)["sentences"]]
    info = run_test_set(cfg, fe, sents, device=args.device, log=log)
    res = cfg["path"]["result_path"]
    for sid, _ in sents:
        check(os.path.exists(os.path.join(res, "test_generated_audios", f"{sid}.wav")), f"{sid}.wav")
        for kind in ("duration", "pitch", "energy"):
            assert os.path.exists(os.path.join(res, "variance_adaptor_plots", kind, f"{sid}_{kind}.png"))
    check(True, "duration/pitch/energy plots for every test sentence")
    check(set(info["sentences"]) == {s for s, _ in sents}, "predictions.json covers every test sentence")
    # the fp16 export must load and synthesise too
    run_test_set(cfg, fe, sents[:1], checkpoint=os.path.join(ck, f"{lang}_fastspeech2_fp16.pth"),
                 out_root=os.path.join(work, "fp16_check"), device=args.device, log=log)
    check(True, "fp16 export loads and synthesises")
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["english", "telugu", "both"], default="both")
    ap.add_argument("--utts", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temporary work directories")
    args = ap.parse_args()
    langs = ["english", "telugu"] if args.language == "both" else [args.language]
    for lang in langs:
        work = run(lang, args)
        if args.keep:
            print(f"  kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print("\nSMOKE TEST PASSED for: " + ", ".join(langs))


if __name__ == "__main__":
    main()
