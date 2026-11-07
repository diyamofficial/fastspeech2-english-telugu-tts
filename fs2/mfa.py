"""
Montreal Forced Aligner wrapper (stage 2 of preprocess.py).

FastSpeech 2 (Sec. 2.3) uses MFA phone alignments as duration targets. MFA is a
conda package, so it usually lives in its own environment; set `mfa.command`
in config.yaml (e.g. "micromamba run -n mfa mfa" on Colab) or the MFA_CMD
environment variable.

  mode "align": align with a pretrained MFA acoustic model + our lexicon
  mode "train": train an acoustic model on the corpus itself, then align
                (used for Telugu, for which we do not assume a pretrained model)
"""
import os
import shlex
import shutil
import subprocess


def mfa_command(cfg):
    cmd = os.environ.get("MFA_CMD") or cfg.get("mfa", {}).get("command", "mfa")
    return shlex.split(cmd)


def _run(args, log):
    log("[mfa] $ " + " ".join(shlex.quote(a) for a in args))
    p = subprocess.run(args, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"MFA command failed with exit code {p.returncode}")


def run_mfa(cfg, log=print):
    m = cfg.get("mfa", {})
    base = mfa_command(cfg)
    if shutil.which(base[0]) is None:
        raise RuntimeError(
            f"'{base[0]}' not found. Install MFA (see README > Forced alignment) or set "
            f"mfa.command / MFA_CMD."
        )
    corpus = cfg["path"]["raw_path"]
    lexicon = cfg["path"]["lexicon_path"]
    out = cfg["path"]["textgrid_path"]
    os.makedirs(out, exist_ok=True)
    jobs = str(m.get("jobs", os.cpu_count() or 2))
    extra = list(m.get("extra_args", []))
    mode = m.get("mode", "align")
    if mode == "align":
        model = m["acoustic_model"]
        if not os.path.exists(model):
            _run(base + ["model", "download", "acoustic", model], log)
        _run(base + ["align", corpus, lexicon, model, out, "--clean", "-j", jobs] + extra, log)
    elif mode == "train":
        model_out = cfg["path"].get("mfa_model_path", os.path.join(os.path.dirname(out), "mfa_acoustic_model.zip"))
        _run(base + ["train", corpus, lexicon, model_out, "--output_directory", out, "--clean", "-j", jobs]
             + extra, log)
    else:
        raise ValueError(f"unknown mfa.mode {mode!r}")
    n = sum(1 for _, _, fs in os.walk(out) for f in fs if f.endswith(".TextGrid"))
    log(f"[mfa] {n} TextGrids in {out}")
    if n == 0:
        raise RuntimeError("MFA produced no TextGrids")
    return n
