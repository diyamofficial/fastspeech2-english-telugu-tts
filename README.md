# FastSpeech 2 Text-to-Speech — English (Svarah) and Telugu (IndicTTS)

Two independent FastSpeech 2 acoustic models, one per language, each with its own
data, vocabulary, configuration and checkpoints, plus a pretrained HiFi-GAN vocoder.

| | English | Telugu |
|---|---|---|
| Dataset | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English, CC BY 4.0) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (IIT Madras Indic TTS licence) |
| Selection | ≈5 h train / ≈1 h validation | ≈5 h train / ≈1 h validation |
| Text → tokens | normalisation + g2p_en → ARPAbet (69 phones + pause) | normalisation + rule-based Telugu G2P (50 phones + pause) |
| Durations | MFA, pretrained `english_us_arpa` model | MFA acoustic model trained on the corpus |
| Speakers | multi-speaker (speaker embedding) | single speaker if one has ≥ 6 h, else 2 speakers |

---

## Execution status

All results in `Results/` come from real runs of this code on Google Colab (NVIDIA T4),
using `colab/FastSpeech2_Colab.ipynb`. Nothing is fabricated or placeholder.
`python tools/verify_deliverables.py` checks that every deliverable exists.

**Training length.** Because of the submission deadline, the models were trained for
**5,000 steps (English) and 3,000 steps (Telugu)** (batch 16), not the 40,000 configured in `config.yaml`. Training was
stopped early, and inference uses `best.pth.tar`, the checkpoint with the
lowest validation loss. Running `train.py --restore latest` continues training to the
configured schedule. Listening check: the Telugu outputs are understandable with good sound quality (slightly fast).
The English outputs are mostly too fast to follow; E4 is the exception and is almost clear.
See the technical report for the analysis.

| | English (Svarah) | Telugu (IndicTTS) |
|---|---|---|
| Train / val utterances (after alignment) | 3,683 / 835 | 2,724 / 558 |
| Train / val hours (after trimming + alignment) | 4.82 h / 1.06 h | 4.30 h / 0.86 h |
| Speakers | 115 (multi-speaker, speaker embedding) | 2 (male 6.21 h + female 1.89 h available; male alone fell just short of the 6.24 h budget, so `auto` kept both) |
| Utterances dropped in feature extraction | 3 (2 no voiced frames, 1 too short) | 24 (no TextGrid: MFA could not align them); 43 removed earlier by the Latin-script text filter |
| Vocabulary | 71 (69 ARPAbet + `_` + `sp`) — all 70 phones seen in training | 52 (50 phones + `_` + `sp`) — 4 rare phones (`dz`, `rr`, `ts`, `zh`) never occur in training; no test sentence uses them |
| Model parameters | 35.11 M | 35.08 M |
| Steps trained | 5,000 (≈ 38 min on a T4, ≈ 0.46 s/step incl. validation) | 3,000 (≈ 26 min on a T4; shortened further because Telugu MFA training took ≈ 2 h) |
| Non-finite (NaN/Inf) batches | 0 | 0 |
| Training loss (step 100 → last) | 13.29 → 1.83 (step 5,000) | 13.26 → 1.60 (step 3,000) |
| Best validation loss | 2.594 at step 5,000 (2.796 at step 1,000) | 2.117 at step 2,000 (2.653 at 1,000; 2.183 at 3,000) |
| Inference checkpoint / speaker | `best.pth.tar` (step 5,000) / `spkd270faea` | `best.pth.tar` (step 2,000) / `male` |
| Vocoder | HiFi-GAN universal | HiFi-GAN universal |

Validation loss components at step 5,000 (English): mel 0.756, postnet mel 0.755,
pitch 0.474, energy 0.391, duration 0.218. Per-step logs are in
`Results/<Lang>/logs/` (`train_log.csv`, `val_log.csv`, `train.log`).

Verified in the English outputs:
- the homograph in E2 is pronounced differently in each sentence: "conduct" is the
  noun `K AA1 N D AH0 K T` in E2_a and the verb `K AA0 N D AH1 K T` in E2_b;
- F0 re-extracted from the synthesised audio follows the model's predicted pitch,
  as shown in every pitch plot.

| Stage | Status |
|---|---|
| Text frontends, data selection, corpus writing, feature extraction | executed (unit-checked offline, then on the real datasets) |
| English: download → MFA → features → training → test-set synthesis | **executed on Colab (T4)** |
| Telugu: download → MFA → features → training → test-set synthesis | **executed on Colab (T4)** |

---

## Quick start (Google Colab, T4 GPU)

1. Upload this repository to GitHub (or zip the `project/` folder to
   `MyDrive/fastspeech2_assignment/fastspeech2_project.zip`).
2. Open `colab/FastSpeech2_Colab.ipynb` in Colab, select a **T4 GPU** runtime, set
   `LANG = "English"` (or `"Telugu"`) and `REPO_URL`, then **Run all**.
3. Run the notebook a second time in another session with the other language,
   in parallel if you can.

The notebook runs a 2-minute smoke test, installs MFA, preprocesses, trains
(resumable, with a wall-clock limit), synthesises all test sentences and verifies
the deliverables. Checkpoints, logs and `Results/<LANG>` are written to Google
Drive, so after a disconnect you re-run all cells and training continues.

---

## FastSpeech 2 architecture (reference)

Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, ICLR 2021
([arXiv:2006.04558](https://arxiv.org/abs/2006.04558)).

```
phonemes ─► Embedding + PosEnc ─► Encoder (4 × FFT block: self-attention + Conv1D 9/1)
          ─► [+ speaker embedding]
          ─► Variance Adaptor
               ├─ Duration predictor  (2×[Conv1D k3+ReLU+LN+Dropout 0.5] + Linear, log domain, MSE)
               ├─ Pitch predictor     (same structure; quantised to 256 bins → embedding, added)
               ├─ Energy predictor    (same structure; quantised to 256 bins → embedding, added)
               └─ Length regulator    (repeat each phoneme by its duration)
          ─► PosEnc ─► Mel decoder (6 × FFT block) ─► Linear → 80-bin mel ─► PostNet (5 Conv1D)
          ─► HiFi-GAN (universal, 22.05 kHz) ─► waveform
```

Training targets: ground-truth mel (L1, before and after PostNet), MFA durations,
WORLD F0 and STFT-frame energy (MSE). At inference the predicted duration, pitch and
energy are used. Optimiser: Adam (β = 0.9/0.98, ε = 1e-9) with Noam warm-up (4,000
steps) and step annealing, as in Appendix B of the paper.

## Reference implementation and modifications

Base: **[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)** (MIT), the
implementation the assignment points to. Its model code is reused in `fs2/transformer/`,
`fs2/model/` and `fs2/hifigan/` (HiFi-GAN from jik876/hifi-gan, MIT). Each modified
file carries a header listing its changes. Summary:

| Area | Reference | This project | Why |
|---|---|---|---|
| Languages | English (LJSpeech/LibriTTS), Mandarin | English (Svarah) + **Telugu**, fully separate folders/configs/vocabularies/checkpoints | assignment |
| Vocabulary | global English/pinyin symbol list | built from each language frontend's phone inventory; passed to the model | independent models; Telugu support |
| Telugu text | — | `Telugu/telugu_frontend.py`: NFC, ZWJ/ZWNJ removal, Telugu-digit & number expansion (ordinal `-వ`, times), rule-based G2P with homorganic anusvara | Telugu support |
| English text | g2p_en via a fixed LibriSpeech lexicon | g2p_en at sentence level (POS-based homographs); lexicon from the same g2p_en with **both** homograph pronunciations | train/inference consistency; E2 "conduct" |
| Config | 3 YAMLs per dataset | one `config.yaml` per language (assignment structure); paths relative to it | assignment |
| Train/val split | random `val_size` utterances | hours-budgeted, speaker-stratified, seeded, written to `split.json`; zero overlap asserted | 5 h / 1 h requirement |
| Normalisation stats | all data | **train split only** | no validation leakage |
| Unknown words (`spn`) | silently turned into a pause | utterance skipped and counted | avoids mis-aligned targets |
| Feature extraction | librosa + torch STFT, `tgt` | numpy/scipy re-implementation (identical Slaney mel filterbank, log clamp 1e-5, energy = L2 norm), own TextGrid reader (MFA long/short formats) | fewer dependencies; runs without torch |
| Length regulator | Python loop per phoneme | `torch.repeat_interleave` (same output) + empty-output guard at inference | speed, robustness |
| Pitch/energy control | scales the z-score | scales in Hz / energy units | semantically correct control |
| Training loop | fixed steps | resumable (`--restore latest`), wall-clock limit (`--max_hours`), best-val checkpoint, full-set validation, non-finite-loss guard, CSV logs, loss plots, fp16 export | Colab limits, reproducibility, NaN handling |
| Inference | single sentence | test-set mode: WAV + duration/pitch/energy plots + `predictions.json` from the model's own outputs | deliverables |

Not changed: FFT blocks, attention, variance predictors, quantised pitch/energy
embeddings, PostNet, losses, LR schedule, and model sizes (encoder 4 layers, decoder 6
layers, 256 hidden, 2 heads, Conv1D 1024/9). The decoder depth (6) and dropout (0.2) follow
the reference implementation's config; the paper's Table 7 uses 4 decoder layers.
Like the reference implementation, pitch is predicted directly (no CWT) at
phoneme level, which is more stable with only 5 h of data. The CWT pitch spectrogram of
Appendix C is not implemented.

---

## Datasets

### English — Svarah
* Hugging Face: `ai4bharat/Svarah` — one split `test`, 6,656 utterances, ≈9.6 h, 117 speakers
  from across India, columns `audio_filepath, duration, text, gender, age-group,
  primary_language, native_place_state, native_place_district, highest_qualification,
  job_category, occupation_domain`. Licence CC BY 4.0.
  Source repository: https://github.com/AI4Bharat/Svarah
* Requires a Hugging Face access token: `huggingface-cli login` or `export HF_TOKEN=hf_...`.

### Telugu — IndicTTS Telugu
* Hugging Face: `SPRINGLab/IndicTTS_Telugu` — split `train`, 8,576 utterances, 48 kHz,
  columns `audio, text, gender` (0 = female, 1 = male), ≈6.4 GB of parquet.
  Subject to the original Indic TTS licence (IIT Madras), linked from the dataset card.

Both are downloaded automatically by `preprocess.py --stage prepare` through the
`datasets` library. No audio is decoded until an utterance has been selected.

### Data selection and subsampling strategy
Implemented in `fs2/data_prep.py:select_split`, configured under `dataset:` in each `config.yaml`.

1. **Text filter.**
   * English drops transcripts with annotation brackets (`[...]`, `<...>`).
   * Telugu drops transcripts containing Latin letters. The speaker pronounces such
     words as English, but the G2P could only spell them letter by letter, which would
     corrupt the alignment.
   * Both drop transcripts that normalise to nothing.
2. **Duration filter:** 1–11 s. The upper bound keeps every utterance under the
   model's 1,000-frame limit.
3. **Speaker policy.**
   * *English:* `all`. No Svarah speaker has more than a few minutes of audio, so the
     model is multi-speaker with a learned speaker embedding. Svarah on Hugging Face has
     no speaker-ID column, so speakers are identified by their per-speaker demographic
     profile (`gender, age-group, primary_language, native place, qualification, job`),
     hashed to `spkXXXXXXXX`. Two speakers with identical profiles would be merged; the
     number of derived speakers is logged.
   * *Telugu:* `auto`. Use the speaker with the most audio alone if it covers the
     train + validation budget (cleanest single-voice model); otherwise use both
     speakers with a speaker embedding.
4. **Budget.** Each kept speaker contributes in proportion to its available audio,
   using a seeded random shuffle (seed 1234) until 5 h + 1 h (×1.04 margin for later
   losses) is reached.
5. **Split, inside each speaker.** The first ~1/6 of that speaker's selected audio goes
   to validation and the rest to training. Every validation speaker is also seen in
   training, which the speaker embedding needs. **No utterance is in both splits**
   (asserted in code).

The exact numbers are written to:
* `<Lang>/data/corpus/selection_summary.json`;
* `<Lang>/data/preprocessed/preprocess_summary.json` (after alignment filtering);
* `Results/<Lang>/data_summary.json`, combining both, for the report.

### Preprocessing pipeline (`preprocess.py --stage prepare | align | features | all`)

| Step | Details |
|---|---|
| Audio loading & validation | decode with soundfile; undecodable files and clips shorter than 0.3 s after trimming are dropped and counted |
| Resampling | to 22,050 Hz (polyphase), mono — the rate of the HiFi-GAN vocoder |
| Silence & loudness | leading/trailing silence below −40 dB of the peak trimmed (50 ms margin); peak-normalised to 0.95; internal pauses become `sp` from MFA |
| Text normalisation | English: times, dates, years, ordinals, money, %, decimals, cardinals, acronyms → words. Telugu: see below |
| Phonemes / tokens | English ARPAbet via g2p_en; Telugu rule-based G2P; punctuation → `sp`; MFA lexicon from the same frontend |
| Vocabulary | `_` (pad), `sp`, then the language's full phone inventory → `preprocessed/vocab.json`, stored in every checkpoint |
| Forced alignment | MFA → TextGrids. English aligns with pretrained `english_us_arpa` (beam 100/400). Telugu trains an MFA model on the corpus (`mfa train`) |
| Durations | phone intervals → frame counts from rounded boundaries (sum = mel length) |
| Mel | 1024-pt STFT, hop 256, Hann, 80 Slaney mel bins 0–8 kHz, log(clamp(·, 1e-5)) |
| Pitch | PyWorld DIO + StoneMask at hop resolution, unvoiced frames linearly interpolated, averaged per phoneme, z-normalised with train-split statistics |
| Energy | L2 norm of each magnitude-STFT frame, averaged per phoneme, z-normalised |
| Output | `preprocessed/{mel,pitch,energy,duration}/*.npy`, `train.txt`, `val.txt`, `stats.json`, `speakers.json`, `vocab.json` |

### Telugu-specific processing (`Telugu/telugu_frontend.py`)
* **Unicode:** NFC normalisation; ZWJ/ZWNJ/soft hyphen removed (they change glyph shaping, not sound); Telugu digits ౦–౯ → 0–9; danda → full stop.
* **Numbers → Telugu words:** e.g. 2026 → రెండు వేల ఇరవై ఆరు and 10:45 → పది నలభై ఐదు.
  The ordinal suffix attaches to a stemmed cardinal (12వ → పన్నెండవ, 21వ → ఇరవై ఒకటవ).
  Hundreds/thousands/lakhs/crores use their oblique forms (వందల, వేల, లక్షల).
* **G2P over akshara structure:**
  * consonant + virama → bare consonant; consonant + vowel sign → that vowel;
    a bare consonant keeps the inherent *a*, including word-finally (Telugu has no schwa deletion);
  * anusvara assimilates to the next consonant's place (బలంగా → `b a l a ng g aa`,
    ఉంచండి → `u ny ch a nn dd i`, నవంబర్ → `n a v a m b a r`), otherwise *m*;
  * visarga → *h*; vocalic ṛ → *r u*;
  * aspiration, retroflexion, vowel length and gemination stay distinct
    (పట్టు → `p a tt tt u`).
* **Word boundaries are preserved as written** — no silent spelling or spacing repair.

### Test sentences
Stored verbatim in `English/test_sentences.json` and `Telugu/test_sentences.json`
(IDs `E1, E2_a, E2_b, E3, E4` and `T1, T2_a, T2_b, T3, T4`). The assignment's
irregularities are preserved on purpose: the space inside "rhyth m" (E3), "andaroo"
(E4) and "అడ విలో" (T3).

The PDF's embedded text layer splits Telugu conjuncts (it yields e.g. `వచ్చ ింది`
for the rendered `వచ్చింది`). The Telugu sentences were therefore transcribed from the
rendered glyphs and checked programmatically: each sentence contains exactly the same
multiset of Telugu characters as the PDF text layer.

---

## Installation

```bash
git clone https://github.com/diyamofficial/fastspeech2-english-telugu-tts.git
cd fastspeech2-english-telugu-tts
pip install -r requirements.txt        # PyTorch: use your platform's CUDA build
```

**Vocoder** — the universal HiFi-GAN released with the reference implementation:
```bash
git clone --depth 1 https://github.com/ming024/FastSpeech2 /tmp/ming024
unzip /tmp/ming024/hifigan/generator_universal.pth.tar.zip -d vocoder/
# -> vocoder/generator_universal.pth.tar
```
Without it, inference falls back to Griffin-Lim and says so in `predictions.json`.

**Forced alignment** — Montreal Forced Aligner (conda-forge only):
```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
export MFA_CMD="conda run -n mfa mfa"          # or set mfa.command in config.yaml
# Colab (no restart): micromamba — see the notebook, section 5
```

## Usage

All commands run from the repository root; each script also accepts `--config`.

```bash
# 0. sanity check on synthetic data (~2 min)
python tools/smoke_test.py --language both

# 1. preprocessing (each stage can be run separately)
python English/preprocess.py --stage all      # prepare -> align -> features
python Telugu/preprocess.py  --stage all

# 2. training (resumable; --max_hours stops cleanly for Colab/Kaggle limits)
python English/train.py [--restore latest] [--max_hours 3.5] [--max_steps N]
python Telugu/train.py  [--restore latest] [--max_hours 3.5] [--max_steps N]

# 3. inference: all test sentences -> audio + variance-adaptor plots
python English/inference.py --test_set
python Telugu/inference.py  --test_set

#    any text (with optional prosody control)
python English/inference.py --text "Hello, how are you?" --out hello.wav --pitch_control 1.1
python Telugu/inference.py  --text "మీరు ఎలా ఉన్నారు?" --out te.wav

# 4. check all deliverables
python tools/verify_deliverables.py
```
`bash tools/run_pipeline.sh English 3.5` runs steps 0–4 for one language.

### Training outputs
* Checkpoints in `<Lang>/checkpoints/`:
  * `step_N.pth.tar` (last 3 kept), `latest.pth.tar` and `best.pth.tar` (lowest
    validation loss). Each holds model, optimizer, vocab, speakers, stats and config.
  * `<lang>_fastspeech2_fp16.pth`: inference-only fp16 export (~66 MB).
* Logs in `<Lang>/logs/`:
  * `train_log.csv`, logged every 100 steps: total, mel, postnet_mel, pitch, energy,
    duration, learning rate;
  * `val_log.csv`, the full validation set every 1,000 steps;
  * `val_mel_step*.png` and `train.log`.
* `Results/<Lang>/training_loss.png` and `validation_loss.png` are regenerated after
  every validation. The CSVs are copied to `Results/<Lang>/logs/`.
* Non-finite losses or gradients: the batch is skipped and logged. Training stops
  after 20 in a row.
* Reproducibility: seed 1234 for selection, batching and initialisation. Set
  `train.deterministic: true` for deterministic cuDNN, which is slower.

### Variance-adaptor plots
`inference.py --test_set` writes, for every test sentence, **from that sentence's
model predictions**:
* `variance_adaptor_plots/duration/<ID>_duration.png`: predicted frames per phoneme
  (bars, the length-regulator input), plus the un-rounded predictor output;
* `variance_adaptor_plots/pitch/<ID>_pitch.png`: predicted F0 in Hz (phoneme level,
  de-normalised, on the frame axis with phoneme boundaries). F0 re-extracted from the
  synthesised audio is shown for reference;
* `variance_adaptor_plots/energy/<ID>_energy.png`: predicted energy (de-normalised).

The raw numbers behind every plot (phones, durations, log-durations, pitch in z-score
and Hz, energy) are in `Results/<Lang>/predictions.json`, with the checkpoint step and
vocoder used. `test_manifest.csv` maps each ID to its exact text, audio and plots.
Output names follow the IDs: `E1.wav`, `E2_a.wav`, `E2_b.wav`, ….

---

## Project structure

```
project/
├── English/
│   ├── train.py  preprocess.py  inference.py  config.yaml   # required entry points
│   ├── english_frontend.py        # normalisation + g2p_en (ARPAbet)
│   ├── test_sentences.json        # E1, E2_a, E2_b, E3, E4 (verbatim)
│   ├── checkpoints/               # English checkpoints only
│   └── data/ logs/                # created by the pipeline (git-ignored)
├── Telugu/
│   ├── train.py  preprocess.py  inference.py  config.yaml
│   ├── telugu_frontend.py         # Telugu normalisation + rule-based G2P
│   ├── test_sentences.json        # T1, T2_a, T2_b, T3, T4 (verbatim)
│   ├── checkpoints/
│   └── data/ logs/
├── Results/
│   ├── English/  training_loss.png  validation_loss.png  test_generated_audios/
│   │             variance_adaptor_plots/{duration,pitch,energy}/
│   │             predictions.json  test_manifest.csv  data_summary.json  logs/
│   └── Telugu/   (same)
├── fs2/                           # shared FastSpeech 2 core adapted from ming024/FastSpeech2
│   ├── transformer/  model/  hifigan/
│   ├── audio.py  pitch.py  textgrid.py  features.py  data_prep.py  mfa.py  pipeline.py
│   ├── dataset.py  trainer.py  synthesizer.py  plotting.py  frontend_base.py  utils.py
├── tools/   smoke_test.py  verify_deliverables.py  run_pipeline.sh
├── colab/   FastSpeech2_Colab.ipynb
├── vocoder/                       # HiFi-GAN weights (downloaded)
├── Report/                        # Technical_Report.pdf (added separately)
├── README.md
└── requirements.txt
```
The two languages share only language-independent library code (`fs2/`). Each has
its own frontend, vocabulary, config, data, features, statistics and checkpoints;
neither pipeline reads anything from the other.

## Hardware and software requirements
* **Training:** one CUDA GPU with ≥ 12 GB (T4 16 GB is enough for batch 16), ~25 GB
  disk (the Telugu parquet is 6.4 GB, plus corpus and features), ≥ 12 GB RAM.
  Throughput depends on the GPU; read your real step time from `train.log` and set
  `--max_hours`/`--max_steps` accordingly. Training resumes after interruptions.
* **Preprocessing:** CPU only. MFA alignment of ~6 h of audio takes from tens of
  minutes to about an hour on a 2-core Colab VM; Telugu takes longer because it trains
  an acoustic model.
* **Inference:** CPU or GPU, seconds per sentence.
* **Software:** Python ≥ 3.9, PyTorch ≥ 2.0, Montreal Forced Aligner (conda-forge),
  and the packages in `requirements.txt`.

## Checkpoints on GitHub
* The fp16 exports (`<lang>_fastspeech2_fp16.pth`, ~66 MB) fit GitHub's 100 MB limit
  and are committed; `inference.py --checkpoint english_fastspeech2_fp16.pth` uses one.
* The full training checkpoints (~400 MB, with optimizer state) are git-ignored.
  Either track them with Git LFS (`git lfs track "*.pth.tar"`) or upload them (e.g. a
  public Google Drive folder) and put the link here.

**Full training checkpoints (Google Drive):**
- English: https://drive.google.com/drive/folders/140F1GGj9TiaXCMOJh7BPWksBI0S5c2Zx?usp=sharing
- Telugu: https://drive.google.com/drive/folders/1hOf6TIas7xcUZRbLdj-9n7b4RtvYA3f7?usp=sharing

Download `best.pth.tar` into `English/checkpoints/` or `Telugu/checkpoints/` and run
`inference.py --test_set`. Inference uses `best.pth.tar` first, then `latest.pth.tar`,
then the fp16 export.

## Known limitations
* **English:** Svarah is accented, multi-speaker, conversational read speech with
  only a few minutes per speaker and varied recording conditions. With 5 h this
  limits naturalness and speaker consistency compared with single-speaker corpora.
  Speaker identities are derived from demographic metadata.
* **Telugu:** the universal HiFi-GAN was trained on English. It generalises
  reasonably but is not tuned to Telugu.
* **Out-of-vocabulary words:** "antidisestablishmentarianism" (E1), the broken word
  "rhyth m" (E3) and "andaroo" (E4) are pronounced by g2p_en's neural model. Their
  pronunciation should be judged in the report.
* **Pitch:** predicted directly at phoneme level (as in the reference implementation)
  rather than with the paper's CWT pitch spectrogram.

## Licences and citations
* FastSpeech 2: Ren et al., ICLR 2021. Reference code: ming024/FastSpeech2 (MIT);
  HiFi-GAN: jik876/hifi-gan (MIT).
* Svarah: Javed et al., *Svarah: Evaluating English ASR Systems on Indian Accents*,
  Interspeech 2023 (CC BY 4.0).
* IndicTTS: IIT Madras Indic TTS consortium (licence on the dataset card).
* Montreal Forced Aligner: McAuliffe et al., Interspeech 2017.
