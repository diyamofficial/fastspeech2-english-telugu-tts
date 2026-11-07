# Adapted from ming024/FastSpeech2 (MIT) transformer/Modules.py
import torch
import torch.nn as nn

# A large negative constant instead of -inf: keeps softmax finite when a whole
# row is masked and stays representable in fp16.
_NEG = -1e4


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, _NEG)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
