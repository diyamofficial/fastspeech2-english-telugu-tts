# Adapted from ming024/FastSpeech2 (MIT) model/modules.py
#
# Changes vs. reference:
#   * pitch/energy bin edges come from the stats dict passed in (computed on the
#     TRAIN split only) instead of being read from a fixed preprocessed_path;
#   * LengthRegulator uses torch.repeat_interleave (same result, much faster than
#     the per-phoneme Python loop);
#   * at inference, an utterance whose predicted durations all round to zero is
#     given one frame per phoneme so the decoder never receives an empty sequence;
#   * pitch/energy control is applied in the physical domain (Hz / energy units)
#     and mapped back to the normalised domain, instead of scaling a z-score.
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from ..utils import get_mask_from_lengths, pad


class VarianceAdaptor(nn.Module):
    """Variance Adaptor: duration, pitch and energy predictors (FastSpeech 2, Sec. 2.3)."""

    def __init__(self, preprocess_config, model_config, stats):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["energy"]["feature"]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        # stats = {"pitch": [min, max, mean, std], "energy": [min, max, mean, std]}
        pitch_min, pitch_max = stats["pitch"][:2]
        energy_min, energy_max = stats["energy"][:2]
        self.pitch_mean, self.pitch_std = float(stats["pitch"][2]), float(stats["pitch"][3])
        self.energy_mean, self.energy_std = float(stats["energy"][2]), float(stats["energy"][3])

        if pitch_quantization == "log":
            # log-spaced bins need a positive range; normalised values can be negative,
            # so shift the range to start at 1 before taking the log.
            self.pitch_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(1.0), np.log(pitch_max - pitch_min + 1.0), n_bins - 1))
                + pitch_min - 1.0,
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(torch.linspace(np.log(1.0), np.log(energy_max - energy_min + 1.0), n_bins - 1))
                + energy_min - 1.0,
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        self.energy_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])

    # -- control helpers: scale in physical units, return to normalised domain
    def _control(self, pred, control, mean, std):
        if control == 1.0:
            return pred
        return ((pred * std + mean) * control - mean) / std

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = self._control(prediction, control, self.pitch_mean, self.pitch_std)
            embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = self._control(prediction, control, self.energy_mean, self.energy_std)
            embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            duration_rounded = duration_rounded.masked_fill(src_mask, 0)
            # guard: never hand an empty sequence to the decoder
            empty = duration_rounded.sum(dim=1) == 0
            if empty.any():
                ones = (~src_mask).float()
                duration_rounded = torch.where(empty.unsqueeze(1), ones, duration_rounded)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator (FastSpeech): repeat each phoneme hidden state d_i times."""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            if torch.is_floating_point(expand_target):
                expand_target = torch.round(expand_target)
            reps = torch.clamp(expand_target.long(), min=0)
            expanded = torch.repeat_interleave(batch, reps, dim=0)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor: 2x(Conv1D+ReLU+LN+Dropout) + Linear (Fig. 1c)."""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """Convolution module operating on [B, T, C] tensors."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
