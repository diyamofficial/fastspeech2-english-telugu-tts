#!/usr/bin/env python3
"""
Telugu (IndicTTS Telugu) preprocessing for FastSpeech 2.

  python Telugu/preprocess.py --stage prepare    # download + select 5 h / 1 h, write MFA corpus + G2P lexicon
  python Telugu/preprocess.py --stage align      # train MFA on the corpus and align -> TextGrids
  python Telugu/preprocess.py --stage features   # mel, F0, energy, durations, stats, vocab
  python Telugu/preprocess.py --stage all

Telugu-specific steps (see telugu_frontend.py): NFC normalisation, removal of
ZWJ/ZWNJ, Telugu-digit and number expansion into Telugu words, rule-based G2P
with homorganic anusvara assimilation, and an MFA acoustic model trained on the
corpus itself with the G2P lexicon.
"""
import argparse
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fs2.pipeline import hf_items, make_logger, run_preprocess  # noqa: E402
from fs2.utils import load_config  # noqa: E402
from telugu_frontend import TeluguFrontend  # noqa: E402

_LATIN = re.compile(r"[A-Za-z]")
_TELUGU = re.compile(r"[ఀ-౿]")


def keep_text(t):
    """Training transcripts must be Telugu script. Utterances whose transcript contains
    Latin letters are excluded: the speaker says the English word, but a letter-by-letter
    reading is all the G2P can produce, which would corrupt the alignment."""
    if not t or not t.strip():
        return False
    t = unicodedata.normalize("NFC", t)
    return bool(_TELUGU.search(t)) and not _LATIN.search(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--stage", choices=["prepare", "align", "features", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="debug: only scan the first N dataset rows")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = make_logger(cfg["path"]["log_path"], "preprocess")
    fe = TeluguFrontend()
    try:
        _run(cfg, fe, args, log)
    except Exception:
        import traceback

        log("[preprocess] FAILED:\n" + traceback.format_exc())
        raise


def _run(cfg, fe, args, log):
    run_preprocess(cfg, fe, args.stage,
                   lambda: hf_items(cfg, log, uid_prefix="te", keep_text=keep_text, limit=args.limit), log)


if __name__ == "__main__":
    main()
