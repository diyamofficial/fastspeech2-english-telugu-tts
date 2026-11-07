#!/usr/bin/env python3
"""
English (Svarah) preprocessing for FastSpeech 2.

  python English/preprocess.py --stage prepare    # download + select 5 h / 1 h, write MFA corpus + lexicon
  python English/preprocess.py --stage align      # Montreal Forced Aligner -> TextGrids
  python English/preprocess.py --stage features   # mel, F0, energy, durations, stats, vocab
  python English/preprocess.py --stage all

Svarah needs a Hugging Face token: `huggingface-cli login` or export HF_TOKEN=...
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from english_frontend import EnglishFrontend  # noqa: E402
from fs2.pipeline import hf_items, make_logger, run_preprocess  # noqa: E402
from fs2.utils import load_config  # noqa: E402

# Transcripts carrying annotation tags (e.g. "[noise]", "<unk>") are excluded: the
# tag would be either spoken by the model or silently mis-aligned.
_ANNOTATION = re.compile(r"[\[\]<>{}]")


def keep_text(t):
    return bool(t and t.strip()) and not _ANNOTATION.search(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--stage", choices=["prepare", "align", "features", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="debug: only scan the first N dataset rows")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = make_logger(cfg["path"]["log_path"], "preprocess")
    fe = EnglishFrontend()
    try:
        _run(cfg, fe, args, log)
    except Exception:
        import traceback

        log("[preprocess] FAILED:\n" + traceback.format_exc())
        raise


def _run(cfg, fe, args, log):
    run_preprocess(cfg, fe, args.stage,
                   lambda: hf_items(cfg, log, uid_prefix="sv", keep_text=keep_text, limit=args.limit), log)


if __name__ == "__main__":
    main()
