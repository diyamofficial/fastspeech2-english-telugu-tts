"""
fs2 - shared FastSpeech 2 core used by both the English and Telugu pipelines.

The model code (transformer/, model/) and the HiFi-GAN generator (hifigan/) are
adapted from ming024/FastSpeech2 (MIT License, https://github.com/ming024/FastSpeech2),
which in turn uses jik876/hifi-gan (MIT License). Modifications are documented in
the project README ("Modifications to the reference implementation").

Nothing language-specific lives here: each language folder supplies its own text
frontend, vocabulary, config, data and checkpoints.
"""

__version__ = "1.0.0"
