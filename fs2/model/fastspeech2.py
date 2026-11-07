# Adapted from ming024/FastSpeech2 (MIT) model/fastspeech2.py
#
# Changes vs. reference: vocabulary size, speaker count and pitch/energy stats are
# constructor arguments (so English and Telugu models are fully independent); the
# speaker embedding is enabled by config instead of by reading speakers.json here.
import torch.nn as nn

from ..transformer import Decoder, Encoder, PostNet
from ..utils import get_mask_from_lengths
from .modules import VarianceAdaptor


class FastSpeech2(nn.Module):
    """FastSpeech 2: Encoder -> Variance Adaptor -> Mel Decoder (+ PostNet)."""

    def __init__(self, preprocess_config, model_config, n_symbols, n_speakers, stats):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config, n_symbols)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, stats)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet(n_mel_channels=preprocess_config["mel"]["n_mel_channels"])

        self.speaker_emb = None
        if model_config.get("multi_speaker", False) and n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, model_config["transformer"]["encoder_hidden"])

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
