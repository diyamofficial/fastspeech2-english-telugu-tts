"""
Training loop (adapted from ming024/FastSpeech2 train.py + evaluate.py).

Changes vs. reference:
  * one config.yaml per language; vocab / speakers / stats stored in every checkpoint
    so a checkpoint is self-contained for inference;
  * resumable (`--restore latest|best|<step>`), keeps the last N step checkpoints,
    tracks the best validation checkpoint;
  * full-validation-set loss every val_step, CSV logs for train and val, and
    training_loss.png / validation_loss.png regenerated after every validation;
  * non-finite losses are detected: the batch is skipped and counted (training
    aborts if they persist) instead of corrupting the weights;
  * optional wall-clock limit (`--max_hours`) that saves and exits cleanly,
    for Colab/Kaggle session limits;
  * fp16 inference-only export (fits GitHub's 100 MB file limit).
"""
import csv
import glob
import math
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BucketBatchSampler, FS2Dataset, collate
from .model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from .model.loss import LOSS_NAMES
from .plotting import plot_losses, plot_mel_pair
from .utils import read_json, set_seed, to_device

CSV_FIELDS = ["step", "lr"] + LOSS_NAMES + ["seconds"]


def build_model(cfg, n_symbols, n_speakers, stats, device):
    model = FastSpeech2(cfg["preprocessing"], cfg["model"], n_symbols, n_speakers, stats).to(device)
    return model


def _forward(model, batch, teacher_forced=True):
    return model(
        batch["speakers"],
        batch["texts"],
        batch["src_lens"],
        batch["max_src_len"],
        batch["mels"],
        batch["mel_lens"],
        batch["max_mel_len"],
        batch["pitches"] if teacher_forced else None,
        batch["energies"] if teacher_forced else None,
        batch["durations"] if teacher_forced else None,
    )


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _truncate_csv(path, max_step):
    """On resume, drop log rows written after the restored checkpoint."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(float(r["step"])) <= max_step]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, mel_plot_path=None, step=0):
    model.eval()
    sums = np.zeros(len(LOSS_NAMES))
    count = 0
    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        out = _forward(model, batch)
        losses = loss_fn(batch, out)
        n = batch["texts"].shape[0]
        sums += np.array([l.item() for l in losses]) * n
        count += n
        if i == 0 and mel_plot_path is not None:
            L = int(batch["mel_lens"][0])
            plot_mel_pair(batch["mels"][0, :L].cpu().numpy(), out[1][0, :L].cpu().numpy(),
                          mel_plot_path, f"val {batch['ids'][0]} @ step {step}")
    model.train()
    return dict(zip(LOSS_NAMES, (sums / max(count, 1)).tolist()))


def save_checkpoint(path, model, optim, step, cfg, frontend, meta, best_val):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "step": step,
            "best_val": best_val,
            "language": cfg["language"],
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "vocab": frontend.vocab,
            **meta,
        },
        path,
    )


def export_inference(src_ckpt, dst_path):
    """Strip the optimizer state and store weights in fp16 (~55 MB)."""
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    ck.pop("optimizer", None)
    ck["model"] = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()}
    ck["fp16_export"] = True
    torch.save(ck, dst_path)
    return dst_path


def resolve_checkpoint(ckpt_dir, which):
    if which in (None, "", "none"):
        return None
    if os.path.isfile(str(which)):
        return str(which)
    if which in ("latest", "best"):
        p = os.path.join(ckpt_dir, f"{which}.pth.tar")
        return p if os.path.exists(p) else None
    p = os.path.join(ckpt_dir, f"step_{int(which)}.pth.tar")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def train(cfg, frontend, restore=None, max_steps=None, max_hours=None, device=None, log=print):
    tc = cfg["train"]
    set_seed(cfg.get("seed", 1234))
    if tc.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pre = cfg["path"]["preprocessed_path"]
    ckpt_dir = cfg["path"]["checkpoint_path"]
    log_dir = cfg["path"]["log_path"]
    res_dir = cfg["path"]["result_path"]
    for d in (ckpt_dir, log_dir, res_dir):
        os.makedirs(d, exist_ok=True)

    # ---- data
    pre_vocab = read_json(os.path.join(pre, "vocab.json"))
    if pre_vocab != frontend.vocab:
        raise RuntimeError("vocab.json in preprocessed_path differs from the frontend vocabulary; "
                           "re-run `preprocess.py --stage features`.")
    stats = read_json(os.path.join(pre, "stats.json"))
    speakers = read_json(os.path.join(pre, "speakers.json"))
    spk_hours = read_json(os.path.join(pre, "speaker_train_hours.json"))
    train_set = FS2Dataset("train.txt", cfg, frontend.vocab)
    val_set = FS2Dataset("val.txt", cfg, frontend.vocab)
    bs = tc["batch_size"]
    if len(train_set) < bs:
        bs = max(1, len(train_set))
    g = torch.Generator()
    g.manual_seed(cfg.get("seed", 1234))
    sampler = BucketBatchSampler(train_set.lengths, bs, tc.get("group_size", 4), shuffle=True,
                                 drop_last=True, seed=cfg.get("seed", 1234))
    nw = tc.get("num_workers", 2)
    train_loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate, num_workers=nw,
                              pin_memory=device.type == "cuda", persistent_workers=nw > 0, generator=g)
    val_loader = DataLoader(val_set, batch_sampler=BucketBatchSampler(val_set.lengths, bs, 1, shuffle=False,
                                                                       drop_last=False),
                            collate_fn=collate, num_workers=0)
    log(f"[train] {cfg['language']}: {len(train_set)} train / {len(val_set)} val utterances, "
        f"{len(speakers)} speaker(s), vocab {len(frontend.vocab)}, device {device}")

    # ---- model
    model = build_model(cfg, len(frontend.vocab), len(speakers), stats, device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[train] FastSpeech2 parameters: {n_params/1e6:.2f} M")
    loss_fn = FastSpeech2Loss(cfg["preprocessing"], cfg["model"]).to(device)

    step, best_val = 0, math.inf
    ckpt_path = resolve_checkpoint(ckpt_dir, restore)
    optim_state = None
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optim_state = ck.get("optimizer")
        step = int(ck["step"])
        best_val = float(ck.get("best_val", math.inf))
        log(f"[train] resumed from {ckpt_path} at step {step}")
        _truncate_csv(os.path.join(log_dir, "train_log.csv"), step)
        _truncate_csv(os.path.join(log_dir, "val_log.csv"), step)
    elif restore not in (None, "", "none"):
        log(f"[train] no '{restore}' checkpoint in {ckpt_dir}; starting from scratch")
    optim = ScheduledOptim(model, tc, cfg["model"], step)
    if optim_state is not None:
        optim.load_state_dict(optim_state)

    meta = {"speakers": speakers, "speaker_train_hours": spk_hours, "stats": stats}
    total_step = int(max_steps or tc["total_step"])
    grad_acc = int(tc.get("grad_acc_step", 1))
    clip = float(tc.get("grad_clip_thresh", 1.0))
    log_step, val_step, save_step = tc["log_step"], tc["val_step"], tc["save_step"]
    keep_last = int(tc.get("keep_last_checkpoints", 3))
    deadline = time.time() + float(max_hours) * 3600 if max_hours else None
    nonfinite, nonfinite_streak = 0, 0
    window = np.zeros(len(LOSS_NAMES))
    window_n = 0
    t0 = time.time()
    epoch = 0
    stop_reason = "total_step reached"

    def checkpoint(tag_step):
        nonlocal best_val
        p = os.path.join(ckpt_dir, f"step_{tag_step}.pth.tar")
        save_checkpoint(p, model, optim, tag_step, cfg, frontend, meta, best_val)
        shutil.copyfile(p, os.path.join(ckpt_dir, "latest.pth.tar"))
        olds = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pth.tar")),
                      key=lambda x: int(os.path.basename(x)[5:-8]))
        for o in olds[:-keep_last]:
            os.remove(o)
        log(f"[train] saved {p}")

    def validate(at_step):
        nonlocal best_val
        v = evaluate(model, val_loader, loss_fn, device,
                     mel_plot_path=os.path.join(log_dir, f"val_mel_step{at_step}.png"), step=at_step)
        _append_csv(os.path.join(log_dir, "val_log.csv"),
                    {"step": at_step, "lr": f"{optim.current_lr():.3e}",
                     **{k: f"{v[k]:.5f}" for k in LOSS_NAMES}, "seconds": f"{time.time() - t0:.0f}"})
        msg = " ".join(f"{k}={v[k]:.4f}" for k in LOSS_NAMES)
        log(f"[val] step {at_step}: {msg}")
        if v["total"] < best_val:
            best_val = v["total"]
            save_checkpoint(os.path.join(ckpt_dir, "best.pth.tar"), model, optim, at_step, cfg, frontend, meta, best_val)
            log(f"[val] new best validation loss {best_val:.4f} -> best.pth.tar")
        plot_losses(log_dir, res_dir, cfg["language"].capitalize())

    model.train()
    optim.zero_grad()
    done = step >= total_step
    while not done:
        sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            out = _forward(model, batch)
            losses = loss_fn(batch, out)
            total = losses[0]
            if not torch.isfinite(total):
                nonfinite += 1
                nonfinite_streak += 1
                optim.zero_grad()
                log(f"[train] WARNING non-finite loss at step {step} ({batch['ids'][:3]}...), batch skipped")
                if nonfinite_streak >= 20:
                    raise RuntimeError("20 consecutive non-finite losses - check the data / lower the LR")
                continue
            nonfinite_streak = 0
            (total / grad_acc).backward()
            window += np.array([l.item() for l in losses])
            window_n += 1
            if (i + 1) % grad_acc != 0:
                continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not torch.isfinite(gn):
                optim.zero_grad()
                nonfinite += 1
                log(f"[train] WARNING non-finite gradient norm at step {step}, update skipped")
                continue
            optim.step_and_update_lr()
            optim.zero_grad()
            step += 1

            if step % log_step == 0:
                avg = window / max(window_n, 1)
                _append_csv(os.path.join(log_dir, "train_log.csv"),
                            {"step": step, "lr": f"{optim.current_lr():.3e}",
                             **{k: f"{a:.5f}" for k, a in zip(LOSS_NAMES, avg)},
                             "seconds": f"{time.time() - t0:.0f}"})
                log(f"[train] step {step}/{total_step} lr {optim.current_lr():.2e} "
                    + " ".join(f"{k}={a:.4f}" for k, a in zip(LOSS_NAMES, avg))
                    + f" | {time.time() - t0:.0f}s")
                window[:] = 0
                window_n = 0
            if step % val_step == 0:
                validate(step)
            if step % save_step == 0:
                checkpoint(step)
            if step >= total_step:
                done = True
                break
            if deadline and time.time() > deadline:
                stop_reason = f"--max_hours {max_hours} reached"
                done = True
                break
        epoch += 1

    # final validation + checkpoint (skip duplicates)
    if step % val_step != 0:
        validate(step)
    if step % save_step != 0 or not os.path.exists(os.path.join(ckpt_dir, f"step_{step}.pth.tar")):
        checkpoint(step)
    src = os.path.join(ckpt_dir, "best.pth.tar")
    if not os.path.exists(src):
        src = os.path.join(ckpt_dir, "latest.pth.tar")
    exp = export_inference(src, os.path.join(ckpt_dir, f"{cfg['language']}_fastspeech2_fp16.pth"))
    plot_losses(log_dir, res_dir, cfg["language"].capitalize())
    for f in ("train_log.csv", "val_log.csv"):
        if os.path.exists(os.path.join(log_dir, f)):
            os.makedirs(os.path.join(res_dir, "logs"), exist_ok=True)
            shutil.copyfile(os.path.join(log_dir, f), os.path.join(res_dir, "logs", f))
    log(f"[train] finished at step {step} ({stop_reason}); non-finite batches skipped: {nonfinite}; "
        f"best val {best_val:.4f}; inference export: {exp}")
    return step
