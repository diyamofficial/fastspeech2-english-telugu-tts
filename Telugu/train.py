#!/usr/bin/env python3
"""
Train the Telugu FastSpeech 2 model.

  python Telugu/train.py                       # from scratch
  python Telugu/train.py --restore latest      # resume after a disconnect
  python Telugu/train.py --max_hours 3.5       # stop cleanly (and save) after 3.5 h

Checkpoints -> Telugu/checkpoints/, logs -> Telugu/logs/,
loss plots -> Results/Telugu/{training_loss,validation_loss}.png
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fs2.pipeline import make_logger  # noqa: E402
from fs2.trainer import train  # noqa: E402
from fs2.utils import load_config  # noqa: E402
from telugu_frontend import TeluguFrontend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--restore", default=None, help="latest | best | <step> | path to checkpoint")
    ap.add_argument("--max_steps", type=int, default=None, help="override train.total_step")
    ap.add_argument("--max_hours", type=float, default=None, help="wall-clock limit; saves and exits")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = make_logger(cfg["path"]["log_path"], "train")
    train(cfg, TeluguFrontend(), restore=args.restore, max_steps=args.max_steps,
          max_hours=args.max_hours, device=args.device, log=log)


if __name__ == "__main__":
    main()
