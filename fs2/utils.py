"""Small shared helpers (adapted from ming024/FastSpeech2 utils/tools.py)."""
import json
import os
import random

import numpy as np
import yaml


# --------------------------------------------------------------------------- config
def load_config(path):
    """Load a language config.yaml and resolve every entry of `path:` relative to the
    config file's own directory, so commands work from any working directory."""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(path)
    cfg["_config_path"] = path
    cfg["_base_dir"] = base
    for k, v in list(cfg.get("path", {}).items()):
        if isinstance(v, str) and not os.path.isabs(v):
            cfg["path"][k] = os.path.normpath(os.path.join(base, v))
    return cfg


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- torch helpers
def get_mask_from_lengths(lengths, max_len=None):
    """True at PADDED positions (same convention as the reference implementation)."""
    import torch

    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths).item())

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def pad(input_ele, mel_max_length=None):
    """Pad a list of [T, D] or [T] tensors to a batch."""
    import torch
    import torch.nn.functional as F

    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


# --------------------------------------------------------------------------- numpy padding
def pad_1D(inputs, PAD=0):
    max_len = max((len(x) for x in inputs))
    return np.stack([np.pad(x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD) for x in inputs])


def pad_2D(inputs, maxlen=None):
    max_len = maxlen if maxlen else max(np.shape(x)[0] for x in inputs)
    return np.stack(
        [np.pad(x, ((0, max_len - np.shape(x)[0]), (0, 0)), mode="constant", constant_values=0) for x in inputs]
    )


def to_device(batch, device):
    """Move a collated batch (dict) to device."""
    import torch

    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
