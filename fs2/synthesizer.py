"""
Inference: text -> phones -> FastSpeech 2 (predicted duration/pitch/energy) -> mel
-> HiFi-GAN (universal, 22.05 kHz) -> waveform, plus variance-adaptor plots.

Every number in the plots and in predictions.json is read from the model's own
outputs for that sentence (nothing is sampled or synthetic).
"""
import csv
import datetime
import json
import os
import warnings

import numpy as np
import torch

from .audio import MelExtractor, save_wav
from .pitch import extract_f0
from .plotting import plot_variance
from .trainer import build_model


def find_checkpoint(cfg, explicit=None):
    ckdir = cfg["path"]["checkpoint_path"]
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        cand = os.path.join(ckdir, explicit)
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(explicit)
    for name in ("best.pth.tar", "latest.pth.tar", f"{cfg['language']}_fastspeech2_fp16.pth"):
        p = os.path.join(ckdir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No checkpoint in {ckdir}. Train first (train.py) or download the released "
        f"checkpoint into that folder (see README)."
    )


class Synthesizer:
    def __init__(self, cfg, frontend, checkpoint=None, device=None, log=print):
        self.cfg = cfg
        self.frontend = frontend
        self.log = log
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt_path = find_checkpoint(cfg, checkpoint)
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if ck["vocab"] != frontend.vocab:
            raise RuntimeError("checkpoint vocabulary does not match this language's frontend")
        if ck.get("language", cfg["language"]) != cfg["language"]:
            raise RuntimeError(f"checkpoint is for {ck.get('language')}, config is {cfg['language']}")
        self.stats = ck["stats"]
        self.speakers = ck["speakers"]
        self.speaker_hours = ck.get("speaker_train_hours", {})
        self.step = int(ck.get("step", -1))
        self.model = build_model(cfg, len(frontend.vocab), len(self.speakers), self.stats, "cpu")
        self.model.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()})
        self.model.to(self.device).eval()
        self.mel = MelExtractor(cfg)
        self.sr = cfg["preprocessing"]["audio"]["sampling_rate"]
        self.hop = cfg["preprocessing"]["stft"]["hop_length"]
        self.vocoder, self.vocoder_name = self._load_vocoder()
        log(f"[infer] {cfg['language']}: checkpoint {self.ckpt_path} (step {self.step}), vocoder {self.vocoder_name}")

    def _load_vocoder(self):
        path = self.cfg["path"].get("hifigan_path")
        kind = self.cfg.get("inference", {}).get("vocoder", "hifigan")
        if kind == "hifigan" and path and os.path.isfile(path):
            from .hifigan import load_generator

            return load_generator(path, self.device), "HiFi-GAN (universal, ming024/FastSpeech2 release)"
        warnings.warn(f"HiFi-GAN weights not found at {path}; falling back to Griffin-Lim "
                      "(much lower audio quality). See README > Vocoder.")
        return None, "Griffin-Lim (fallback)"

    def speaker_id(self, name="auto"):
        if len(self.speakers) <= 1:
            return 0, next(iter(self.speakers), "single")
        if name in (None, "auto"):
            name = max(self.speaker_hours, key=self.speaker_hours.get) if self.speaker_hours else next(iter(self.speakers))
        if name not in self.speakers:
            raise KeyError(f"speaker {name!r} not in checkpoint ({len(self.speakers)} speakers)")
        return self.speakers[name], name

    @torch.no_grad()
    def synthesize(self, text, speaker="auto", p_control=1.0, e_control=1.0, d_control=1.0):
        phones = self.frontend.text_to_phones(text)
        if not phones:
            raise ValueError(f"no phones produced for {text!r}")
        ids = torch.tensor([self.frontend.phones_to_ids(phones)], dtype=torch.long, device=self.device)
        spk_id, spk_name = self.speaker_id(speaker)
        spk = torch.tensor([spk_id], dtype=torch.long, device=self.device)
        lens = torch.tensor([ids.shape[1]], dtype=torch.long, device=self.device)
        out = self.model(spk, ids, lens, ids.shape[1], p_control=p_control, e_control=e_control, d_control=d_control)
        mel_post = out[1][0].float()  # [T, 80]
        p_pred, e_pred = out[2][0].float().cpu().numpy(), out[3][0].float().cpu().numpy()
        log_d = out[4][0].float().cpu().numpy()
        d_round = out[5][0].float().cpu().numpy().astype(int)
        T = int(out[9][0])
        mel_np = mel_post[:T].cpu().numpy()

        if self.vocoder is not None:
            wav = self.vocoder(mel_post[:T].T.unsqueeze(0)).squeeze().float().cpu().numpy()
        else:
            wav = self.mel.griffin_lim(mel_np.T)
        wav = wav[: T * self.hop]

        pstat, estat = self.stats["pitch"], self.stats["energy"]
        p_level = self.cfg["preprocessing"]["pitch"]["feature"]
        e_level = self.cfg["preprocessing"]["energy"]["feature"]
        p_vals = p_pred[: len(phones)] if p_level == "phoneme_level" else p_pred[:T]
        e_vals = e_pred[: len(phones)] if e_level == "phoneme_level" else e_pred[:T]
        f0_audio = extract_f0(wav, self.sr, self.hop, n_frames=T)
        return {
            "text": text,
            "phones": phones,
            "speaker": spk_name,
            "durations": d_round[: len(phones)].tolist(),
            "log_duration": log_d[: len(phones)].tolist(),
            "pitch_norm": p_vals.tolist(),
            "pitch_hz": (p_vals * pstat[3] + pstat[2]).tolist(),
            "energy_norm": e_vals.tolist(),
            "energy": (e_vals * estat[3] + estat[2]).tolist(),
            "pitch_level": p_level,
            "energy_level": e_level,
            "f0_from_audio": f0_audio.tolist(),
            "n_frames": T,
            "seconds": len(wav) / self.sr,
            "_wav": wav,
            "_mel": mel_np,
        }


def run_test_set(cfg, frontend, sentences, checkpoint=None, speaker=None, out_root=None, device=None, log=print):
    """sentences: list of (sid, text). Writes audios, plots, predictions.json, manifest.csv."""
    out_root = out_root or cfg["path"]["result_path"]
    language = cfg["language"].capitalize()
    audio_dir = os.path.join(out_root, "test_generated_audios")
    plot_dirs = {k: os.path.join(out_root, "variance_adaptor_plots", k) for k in ("duration", "pitch", "energy")}
    for d in [audio_dir, *plot_dirs.values()]:
        os.makedirs(d, exist_ok=True)

    syn = Synthesizer(cfg, frontend, checkpoint, device, log)
    speaker = speaker or cfg.get("inference", {}).get("speaker", "auto")
    frame_ms = syn.hop / syn.sr * 1000.0
    preds, rows = {}, []
    for sid, text in sentences:
        r = syn.synthesize(text, speaker=speaker)
        wav_path = os.path.join(audio_dir, f"{sid}.wav")
        save_wav(wav_path, r["_wav"], syn.sr)
        np.save(os.path.join(audio_dir, f"{sid}_mel.npy"), r["_mel"])
        plots = plot_variance(r, plot_dirs, sid, text, language, frame_ms)
        log(f"[infer] {sid}: {len(r['phones'])} phones -> {r['seconds']:.2f} s  ({wav_path})")
        preds[sid] = {k: v for k, v in r.items() if not k.startswith("_")}
        rows.append([sid, text, " ".join(r["phones"]), r["speaker"], f"{r['seconds']:.2f}",
                     os.path.relpath(wav_path, out_root),
                     *[os.path.relpath(plots[k], out_root) for k in ("duration", "pitch", "energy")]])

    info = {
        "language": cfg["language"],
        "checkpoint": os.path.basename(syn.ckpt_path),
        "checkpoint_step": syn.step,
        "vocoder": syn.vocoder_name,
        "speaker": rows[0][3] if rows else None,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sentences": preds,
    }
    with open(os.path.join(out_root, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_root, "test_manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text_exact", "phones", "speaker", "seconds", "wav", "duration_plot", "pitch_plot", "energy_plot"])
        w.writerows(rows)
    log(f"[infer] wrote {len(rows)} sentences to {out_root}")
    return info
