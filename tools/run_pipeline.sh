#!/usr/bin/env bash
# Run the whole pipeline for one language on a GPU machine (Colab, Kaggle or local).
#
#   bash tools/run_pipeline.sh English            # full run, 40k steps
#   bash tools/run_pipeline.sh Telugu 3.5         # stop training after 3.5 h (resumable)
#
# Requires: pip install -r requirements.txt, MFA reachable as `mfa` or via $MFA_CMD,
# HiFi-GAN weights at vocoder/generator_universal.pth.tar, HF login for Svarah.
set -euo pipefail
LANG_DIR="${1:?usage: run_pipeline.sh English|Telugu [max_hours]}"
MAX_HOURS="${2:-}"
cd "$(dirname "$0")/.."

python tools/smoke_test.py --language "$(echo "$LANG_DIR" | tr '[:upper:]' '[:lower:]')" --steps 20

if [ ! -f "$LANG_DIR/data/preprocessed/train.txt" ]; then
  [ -f "$LANG_DIR/data/corpus/split.json" ] || python "$LANG_DIR/preprocess.py" --stage prepare
  [ -n "$(find "$LANG_DIR/data/TextGrid" -name '*.TextGrid' 2>/dev/null | head -1)" ] || python "$LANG_DIR/preprocess.py" --stage align
  python "$LANG_DIR/preprocess.py" --stage features
fi

EXTRA=()
[ -n "$MAX_HOURS" ] && EXTRA+=(--max_hours "$MAX_HOURS")
python "$LANG_DIR/train.py" --restore latest "${EXTRA[@]}"
python "$LANG_DIR/inference.py" --test_set
python tools/verify_deliverables.py
