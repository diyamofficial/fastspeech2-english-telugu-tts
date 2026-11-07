"""Loss-curve and variance-adaptor plots (matplotlib, headless)."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMPONENTS = ["mel", "postnet_mel", "duration", "pitch", "energy"]


def _read_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _smooth(y, k=5):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def plot_losses(log_dir, result_dir, language):
    """Write training_loss.png and validation_loss.png from the CSV logs."""
    tr = _read_csv(os.path.join(log_dir, "train_log.csv"))
    va = _read_csv(os.path.join(log_dir, "val_log.csv"))
    os.makedirs(result_dir, exist_ok=True)
    written = []
    for name, d, title, fname in (
        ("train", tr, f"{language} FastSpeech 2 - training loss", "training_loss.png"),
        ("val", va, f"{language} FastSpeech 2 - validation loss", "validation_loss.png"),
    ):
        if not d or len(d["step"]) == 0:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        if name == "train":
            ax.plot(d["step"], d["total"], color="#9ab", lw=1, label="total (per log window)")
            ax.plot(d["step"], _smooth(d["total"]), color="#1f4e79", lw=2, label="total (smoothed)")
        else:
            ax.plot(d["step"], d["total"], "o-", color="#b5541c", lw=2, ms=3, label="validation total")
            if tr:
                ax.plot(tr["step"], _smooth(tr["total"]), "--", color="#1f4e79", lw=1.2, alpha=0.8,
                        label="training total (smoothed, reference)")
            best = int(np.argmin(d["total"]))
            ax.axvline(d["step"][best], color="grey", ls=":", lw=1)
            ax.annotate(f"best {d['total'][best]:.3f}\n@ step {int(d['step'][best])}",
                        (d["step"][best], d["total"][best]), textcoords="offset points", xytext=(8, 10), fontsize=8)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.set_title("Total loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        for c in COMPONENTS:
            if c in d:
                y = _smooth(d[c]) if name == "train" else d[c]
                ax.plot(d["step"], y, lw=1.5, label=c)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_title("Loss components (log scale)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        out = os.path.join(result_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written


def plot_mel_pair(target, pred, path, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, m, t in ((axes[0], target, "ground truth"), (axes[1], pred, "predicted (teacher-forced)")):
        ax.imshow(m.T, origin="lower", aspect="auto", interpolation="none")
        ax.set_title(t, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ variance adaptor
def _find_font(script_hint):
    """Return a font family that can render `script_hint` (e.g. Telugu), or None."""
    from matplotlib import font_manager

    keys = {"telugu": ["Telugu", "Gautami", "Pothana", "Vemana", "Lohit Telugu", "Ramabhadra", "Suranna"]}
    for f in font_manager.fontManager.ttflist:
        for k in keys.get(script_hint, []):
            if k.lower() in f.name.lower():
                return f.name
    return None


def _segments(durations):
    edges = np.concatenate([[0], np.cumsum(durations)])
    return edges[:-1], edges[1:]


def plot_variance(pred, out_dirs, sid, sentence, language, frame_ms):
    """
    pred: dict with phones, durations (frames, rounded, as used by the length
          regulator), log_duration (raw predictor output), pitch_hz / energy
          (denormalised predictions, phoneme- or frame-level), level flags, and
          optionally f0_from_audio (F0 re-extracted from the synthesised waveform).
    Writes <out_dirs[kind]>/<sid>_<kind>.png for kind in duration/pitch/energy.
    """
    phones = pred["phones"]
    dur = np.asarray(pred["durations"], dtype=float)
    starts, ends = _segments(dur)
    centers = (starts + ends) / 2.0
    n = len(phones)
    font = _find_font("telugu") if language.lower() == "telugu" else None
    title_text = sentence if (language.lower() != "telugu" or font) else ""
    wide = max(10, min(28, 0.32 * n + 4))
    paths = {}

    def _title(ax, what):
        t = f"{language} {sid} - predicted {what}"
        ax.set_title(t, fontsize=11, loc="left")
        if title_text:
            kw = {"fontfamily": font} if font else {}
            ax.text(0, 1.10, title_text, transform=ax.transAxes, fontsize=9, color="#444", **kw)

    # ---- duration: bar per phoneme (frames and ms)
    fig, ax = plt.subplots(figsize=(wide, 4.2))
    colors = ["#c9c9c9" if p == "sp" else "#2c7fb8" for p in phones]
    ax.bar(np.arange(n), dur, color=colors, edgecolor="none")
    raw = np.clip(np.exp(np.asarray(pred["log_duration"])) - 1, 0, None)
    ax.plot(np.arange(n), raw, "k.", ms=4, label="un-rounded predictor output exp(log d)-1")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(phones, rotation=90, fontsize=8)
    ax.set_ylabel("duration (mel frames)")
    sec = ax.secondary_yaxis("right", functions=(lambda f: f * frame_ms, lambda m: m / frame_ms))
    sec.set_ylabel("ms")
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    _title(ax, f"phoneme durations (total {int(dur.sum())} frames = {dur.sum() * frame_ms / 1000:.2f} s)")
    fig.tight_layout()
    paths["duration"] = os.path.join(out_dirs["duration"], f"{sid}_duration.png")
    fig.savefig(paths["duration"], dpi=150)
    plt.close(fig)

    # ---- pitch / energy on the frame axis
    for kind, key, unit, color in (("pitch", "pitch_hz", "F0 (Hz)", "#d95f0e"), ("energy", "energy", "energy (L2 norm of |STFT| frame)", "#31a354")):
        vals = np.asarray(pred[key], dtype=float)
        fig, ax = plt.subplots(figsize=(wide, 4.2))
        if pred[f"{kind}_level"] == "phoneme_level":
            for i in range(n):
                if dur[i] > 0:
                    ax.hlines(vals[i], starts[i], ends[i], color=color, lw=2.2)
            ax.plot(centers[dur > 0], vals[dur > 0], ".", color=color, ms=4, label=f"predicted {kind} (phoneme-level)")
        else:
            ax.plot(np.arange(len(vals)), vals, color=color, lw=1.8, label=f"predicted {kind} (frame-level)")
        if kind == "pitch" and pred.get("f0_from_audio") is not None:
            f0 = np.asarray(pred["f0_from_audio"], dtype=float)
            f0 = np.where(f0 > 0, f0, np.nan)
            ax.plot(np.arange(len(f0)), f0, color="#555", lw=0.9, alpha=0.7,
                    label="F0 re-extracted from the synthesised audio (for reference)")
        for s in starts[1:]:
            ax.axvline(s, color="#bbb", lw=0.5)
        finite = vals[np.isfinite(vals)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if kind == "pitch" and pred.get("f0_from_audio") is not None and np.any(np.isfinite(f0)):
            lo, hi = min(lo, float(np.nanmin(f0))), max(hi, float(np.nanmax(f0)))
        span = max(hi - lo, 1e-3)
        ax.set_ylim(lo - 0.22 * span, hi + 0.30 * span)  # room for phone labels below, legend above
        ymin, ymax = ax.get_ylim()
        for i in range(n):
            ax.text(centers[i], ymin + 0.02 * (ymax - ymin), phones[i], ha="center", va="bottom",
                    fontsize=7, rotation=90, color="#333")
        ax.set_xlim(0, max(1, dur.sum()))
        ax.set_xlabel(f"mel frame ({frame_ms:.1f} ms hop)")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)
        _title(ax, "pitch (F0)" if kind == "pitch" else "energy")
        fig.tight_layout()
        paths[kind] = os.path.join(out_dirs[kind], f"{sid}_{kind}.png")
        fig.savefig(paths[kind], dpi=150)
        plt.close(fig)
    return paths
