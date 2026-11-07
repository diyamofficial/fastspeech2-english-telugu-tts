# HiFi-GAN generator, from jik876/hifi-gan (MIT) via ming024/FastSpeech2 hifigan/models.py.
# Change vs. reference: works whether torch exposes the legacy weight_norm API
# (keys weight_g / weight_v) or only torch.nn.utils.parametrizations.weight_norm.
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d

try:  # legacy API: parameters are stored as weight_g / weight_v (matches checkpoints)
    from torch.nn.utils import remove_weight_norm, weight_norm

    _LEGACY_WN = True
except ImportError:  # pragma: no cover
    from torch.nn.utils.parametrizations import weight_norm
    from torch.nn.utils.parametrize import remove_parametrizations

    _LEGACY_WN = False

    def remove_weight_norm(m):
        remove_parametrizations(m, "weight")


LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i],
                                   padding=get_padding(kernel_size, dilation[i])))
                for i in range(3)
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                   padding=get_padding(kernel_size, 1)))
                for _ in range(3)
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def _remap_legacy_keys(state):
    """weight_g/weight_v -> parametrizations.weight.original0/1 (new weight_norm API)."""
    out = {}
    for k, v in state.items():
        if k.endswith(".weight_g"):
            out[k[: -len("weight_g")] + "parametrizations.weight.original0"] = v
        elif k.endswith(".weight_v"):
            out[k[: -len("weight_v")] + "parametrizations.weight.original1"] = v
        else:
            out[k] = v
    return out


def load_generator(ckpt_path, device, config_path=None):
    """Load the pretrained universal HiFi-GAN released with ming024/FastSpeech2."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_universal.json")
    with open(config_path) as f:
        h = AttrDict(json.load(f))
    vocoder = Generator(h)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["generator"] if "generator" in ckpt else ckpt
    if not _LEGACY_WN:
        state = _remap_legacy_keys(state)
    vocoder.load_state_dict(state)
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder
