"""
Training dataset + length-bucketed batching (adapted from ming024/FastSpeech2 dataset.py).

The reference implementation loads batch_size * group_size utterances, sorts them
by length and splits them into batches inside collate_fn. The same idea is
implemented here as a BatchSampler, which keeps collate_fn simple and works with
num_workers > 0.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .utils import pad_1D, pad_2D, read_json


class FS2Dataset(Dataset):
    def __init__(self, filename, cfg, vocab):
        self.dir = cfg["path"]["preprocessed_path"]
        self.vocab = vocab
        self.speaker_map = read_json(os.path.join(self.dir, "speakers.json"))
        self.items = []
        with open(os.path.join(self.dir, filename), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                uid, spk, phones, text = line.split("|", 3)
                self.items.append((uid, spk, phones.split(), text))
        # mel lengths for bucketing (sum of durations == number of mel frames)
        self.lengths = []
        for uid, spk, _, _ in self.items:
            d = np.load(os.path.join(self.dir, "duration", f"{spk}-duration-{uid}.npy"))
            self.lengths.append(int(d.sum()))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        uid, spk, phones, text = self.items[idx]
        d = self.dir
        return {
            "id": uid,
            "raw_text": text,
            "speaker": self.speaker_map[spk],
            "text": np.array([self.vocab[p] for p in phones], dtype=np.int64),
            "mel": np.load(os.path.join(d, "mel", f"{spk}-mel-{uid}.npy")),
            "pitch": np.load(os.path.join(d, "pitch", f"{spk}-pitch-{uid}.npy")),
            "energy": np.load(os.path.join(d, "energy", f"{spk}-energy-{uid}.npy")),
            "duration": np.load(os.path.join(d, "duration", f"{spk}-duration-{uid}.npy")),
        }


def collate(data):
    text_lens = np.array([d["text"].shape[0] for d in data])
    mel_lens = np.array([d["mel"].shape[0] for d in data])
    return {
        "ids": [d["id"] for d in data],
        "raw_texts": [d["raw_text"] for d in data],
        "speakers": torch.from_numpy(np.array([d["speaker"] for d in data], dtype=np.int64)),
        "texts": torch.from_numpy(pad_1D([d["text"] for d in data])).long(),
        "src_lens": torch.from_numpy(text_lens).long(),
        "max_src_len": int(text_lens.max()),
        "mels": torch.from_numpy(pad_2D([d["mel"] for d in data])).float(),
        "mel_lens": torch.from_numpy(mel_lens).long(),
        "max_mel_len": int(mel_lens.max()),
        "pitches": torch.from_numpy(pad_1D([d["pitch"] for d in data])).float(),
        "energies": torch.from_numpy(pad_1D([d["energy"] for d in data])).float(),
        "durations": torch.from_numpy(pad_1D([d["duration"] for d in data])).long(),
    }


class BucketBatchSampler(Sampler):
    """Shuffle, take groups of batch_size*group_size, sort each group by length,
    cut into batches, shuffle the batches. Deterministic given the seed + epoch."""

    def __init__(self, lengths, batch_size, group_size=4, shuffle=True, drop_last=True, seed=1234):
        self.lengths = lengths
        self.batch_size = batch_size
        self.group = batch_size * group_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for g in range(0, len(idx), self.group):
            grp = sorted(idx[g : g + self.group], key=lambda i: -self.lengths[i])
            for b in range(0, len(grp), self.batch_size):
                batch = grp[b : b + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
