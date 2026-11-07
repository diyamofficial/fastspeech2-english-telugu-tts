#!/usr/bin/env python3
"""
English FastSpeech 2 inference.

  # all assignment test sentences (E1, E2_a, E2_b, E3, E4) -> Results/English/
  python English/inference.py --test_set

  # any sentence
  python English/inference.py --text "Hello there." --out hello.wav

Outputs for --test_set:
  Results/English/test_generated_audios/<ID>.wav (+ <ID>_mel.npy)
  Results/English/variance_adaptor_plots/{duration,pitch,energy}/<ID>_<kind>.png
  Results/English/predictions.json   (phones + predicted durations / F0 / energy per sentence)
  Results/English/test_manifest.csv
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from english_frontend import EnglishFrontend  # noqa: E402
from fs2.pipeline import make_logger  # noqa: E402
from fs2.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--checkpoint", default=None, help="file or name in checkpoints/ (default: best, then latest)")
    ap.add_argument("--test_set", action="store_true", help="synthesise every assignment test sentence")
    ap.add_argument("--text", default=None)
    ap.add_argument("--out", default="output.wav")
    ap.add_argument("--speaker", default=None, help="speaker label (default: config inference.speaker)")
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = make_logger(cfg["path"]["log_path"], "inference")
    fe = EnglishFrontend()
    from fs2.synthesizer import Synthesizer, run_test_set
    from fs2.audio import save_wav

    if args.test_set:
        with open(os.path.join(HERE, "test_sentences.json"), encoding="utf-8") as f:
            sentences = [tuple(s) for s in json.load(f)["sentences"]]
        run_test_set(cfg, fe, sentences, checkpoint=args.checkpoint, speaker=args.speaker, device=args.device, log=log)
    elif args.text:
        syn = Synthesizer(cfg, fe, args.checkpoint, args.device, log)
        r = syn.synthesize(args.text, speaker=args.speaker or cfg["inference"]["speaker"],
                           p_control=args.pitch_control, e_control=args.energy_control,
                           d_control=args.duration_control)
        save_wav(args.out, r["_wav"], syn.sr)
        log(f"phones: {' '.join(r['phones'])}")
        log(f"wrote {args.out} ({r['seconds']:.2f} s)")
    else:
        ap.error("use --test_set or --text")


if __name__ == "__main__":
    main()
