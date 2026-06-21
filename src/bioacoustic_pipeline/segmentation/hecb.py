"""HECB-Seg inference: PCEN+CRNN call detection + typed internal boundaries.

The runtime inference path for ``run_pipeline(segmentation="hecb")``. Stage 1
(PCEN+CRNN → call events) is the AUTHORITATIVE call-event detector for calls-in-noise;
``hecb_decode`` (in ``segmentation.hierarchical``) annotates typed internal sub-call
boundaries. Snapping is OFF by default — the CRNN's threshold-crossing onset/offset is
already precise; HECB annotates internal structure, it does not refine call edges.

This module is the canonical home for the HECB inference front-end (the training script
``scripts/train_hecb_seg.py`` trains the checkpoints consumed here). Torch is imported
LAZILY so the module imports without the ``ml`` extra; ``available``/``load_error`` mirror
the embeddings-bank convention. Checkpoints live under ``checkpoints/hecb/{crnn.pt,
typed.pkl}`` (gitignored) and are species-specific (currently meerkat @ 8 kHz).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.segmentation.hierarchical import (
    HECBDurationPriors,
    hecb_decode,
)
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals

# Match orchestration.CALIBRATOR_PROPOSAL (inlined to avoid a circular import).
_PROPOSAL = {"floor_energy": 0.04, "floor_flux": 0.02, "debounce_ms": 25.0}

try:  # lazy torch + torchaudio — module stays importable without the ml extra
    import torch
    import torch.nn as nn
    import torchaudio  # noqa: F401 — required by compute_pcen (mel filterbanks)

    _TORCH = True
    _TORCH_ERR = ""
except Exception as exc:  # noqa: BLE001
    torch = None
    nn = None
    torchaudio = None
    _TORCH = False
    _TORCH_ERR = repr(exc)

available = _TORCH
load_error = _TORCH_ERR


@dataclass
class HECBDSPConfig:
    """DSP + band params the CRNN checkpoint was trained with (species-specific)."""

    sr: int = 8000
    n_fft: int = 512
    hop: int = 160
    n_mels: int = 64
    call_band: tuple[int, int] = (1000, 4000)
    enter: float = 0.7
    exit_: float = 0.3

    @property
    def frame_ms(self) -> float:
        return self.hop / self.sr * 1000.0


# Per-species trained-checkpoint configs (extend as new species CRNNs are trained).
_SPECIES_CONFIG: dict[str, HECBDSPConfig] = {
    "meerkat": HECBDSPConfig(),
}


if _TORCH:

    class CRNNDetector(nn.Module):  # type: ignore[misc]
        """Conv2D → BiLSTM → per-frame P(call). Matches scripts/train_hecb_seg.py."""

        def __init__(self, n_mels: int = 64, hidden: int = 128):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(64, hidden, 3, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, None)),
            )
            self.lstm = nn.LSTM(hidden, hidden, 1, bidirectional=True, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
            )

        def forward(self, x):
            c = self.conv(x).squeeze(2).permute(0, 2, 1)
            return self.fc(self.lstm(c)[0]).squeeze(-1)


def compute_pcen(audio, sr: int, n_fft: int, hop: int, n_mels: int):
    """Manual PCEN: mel-STFT → IIR smoothing → adaptive gain → power compression."""
    import torchaudio  # noqa: PLC0415

    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    stft = torch.stft(
        x, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True
    )
    mag = stft.abs().pow(2).numpy()
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sr / 2, n_mels=n_mels, sample_rate=sr
    ).numpy()
    mel = fb.T @ mag
    b = 1.0 - np.exp(-hop / (0.08 * sr))
    sm = np.zeros_like(mel)
    for t in range(mel.shape[1]):
        sm[:, t] = (1 - b) * (sm[:, t - 1] if t > 0 else 0) + b * mel[:, t]
    return (np.maximum(0.0, mel / (1e-6 + sm) ** 0.98 + 2.0) ** 0.5).astype(np.float32)


def _crnn_proba(model, audio, dsp: HECBDSPConfig, device: str):
    model.eval()
    pcen = compute_pcen(audio, dsp.sr, dsp.n_fft, dsp.hop, dsp.n_mels)
    t = min(pcen.shape[1], 1000)
    x = torch.from_numpy(pcen[:, :t]).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()[0]


def _proba_to_events(proba, dsp: HECBDSPConfig, min_ms: float = 40.0):
    """Hysteresis → (onset_ms, offset_ms, confidence) call events."""
    frame_ms = dsp.frame_ms
    in_call = False
    events, start, conf_sum = [], 0, 0.0
    for i, p in enumerate(proba):
        if not in_call and p >= dsp.enter:
            in_call, start, conf_sum = True, i, p
        elif in_call:
            if p < dsp.exit_:
                in_call = False
                if (i - start) * frame_ms >= min_ms:
                    events.append((start * frame_ms, i * frame_ms, conf_sum / (i - start)))
            else:
                conf_sum += p
    if in_call and (len(proba) - start) * frame_ms >= min_ms:
        events.append((start * frame_ms, len(proba) * frame_ms, conf_sum / (len(proba) - start)))
    return events


def _win_spec(audio, sr, t0_ms, t1_ms):
    i, j = int(t0_ms / 1000 * sr), int(t1_ms / 1000 * sr)
    seg = audio[i : max(j, i + 8)]
    if len(seg) < 8:
        return np.ones(8) * 1e-12, np.zeros(8)
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return mag**2 + 1e-12, np.fft.rfftfreq(len(seg), 1 / sr)


def _band_ratio(psd, freqs, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(psd[m].sum() / (psd.sum() + 1e-12))


def _flatness(psd):
    return float(np.exp(np.log(psd + 1e-12).mean()) / (psd.mean() + 1e-12))


def _extract_typed_evidence(prop, audio, sr, crnn_proba, event, dsp, fp_fn):
    """Stage-3 typed evidence for one NBD proposal: NBD(13)+CRNN-ctx(9)+patch(8)+fp(2)."""
    t = prop.time_ms
    frame_ms = dsp.frame_ms
    fi = int(t / frame_ms)
    nbd = proposal_features(prop)
    p_at = float(crnn_proba[fi]) if fi < len(crnn_proba) else 0.0
    p_before = float(crnn_proba[max(0, fi - 5)]) if fi - 5 < len(crnn_proba) else 0.0
    p_after = (
        float(crnn_proba[min(len(crnn_proba) - 1, fi + 5)]) if fi + 5 < len(crnn_proba) else 0.0
    )
    rising, falling = max(0.0, p_at - p_before), max(0.0, p_before - p_at)
    if event is not None:
        on, off, conf = event
        d_on, d_off, inside, ec = abs(t - on), abs(t - off), (1.0 if on <= t <= off else 0.0), conf
    else:
        d_on = d_off = 9999.0
        inside = ec = 0.0
    crnn_ctx = np.array([p_at, p_before, p_after, rising, falling, inside, d_on, d_off, ec])
    psd_pre, fr_pre = _win_spec(audio, sr, t - 40, t)
    psd_post, fr_post = _win_spec(audio, sr, t, t + 40)
    psd_body, fr_body = _win_spec(audio, sr, t + 20, t + 60)
    body_rms = float(np.sqrt(psd_body.mean()))
    patch = np.array(
        [
            _band_ratio(psd_pre, fr_pre, dsp.call_band),
            _band_ratio(psd_post, fr_post, dsp.call_band),
            _flatness(psd_pre),
            _flatness(psd_post),
            _band_ratio(psd_body, fr_body, dsp.call_band),
            body_rms / (float(np.sqrt(psd_pre.mean())) + 1e-8),
            _flatness(psd_body),
            body_rms,
        ]
    )
    if fp_fn is not None:
        seg_pre = audio[max(0, int((t - 40) / 1000 * sr)) : int(t / 1000 * sr)]
        seg_post = audio[int(t / 1000 * sr) : int((t + 40) / 1000 * sr)]
        if len(seg_pre) > 8 and len(seg_post) > 8:
            delta = np.asarray(fp_fn(seg_post, sr), dtype=np.float64) - np.asarray(
                fp_fn(seg_pre, sr), dtype=np.float64
            )
            fp_feats = np.array([float(np.linalg.norm(delta)), float(np.mean(np.abs(delta)))])
        else:
            fp_feats = np.array([0.0, 0.0])
    else:
        fp_feats = np.array([0.0, 0.0])
    return np.concatenate([nbd, crnn_ctx, patch, fp_feats])


def _event_conditioned_proposals(audio, sr, crnn_proba, crnn_events, profile, dsp, fp_fn):
    """Stage 2+3: NBD inside each CRNN event → typed-evidence proposals (abs ms)."""
    pad_ms = 50.0
    out = []
    for on_ms, off_ms, conf in crnn_events:
        lo = max(0, int((on_ms - pad_ms) / 1000 * sr))
        hi = min(len(audio), int((off_ms + pad_ms) / 1000 * sr))
        seg = audio[lo:hi]
        if len(seg) < int(sr * 0.02):
            continue
        sig = compute_signals(seg, sr, hop_ms=profile.get("hop_ms", dsp.frame_ms))
        for prop in generate_proposals(sig, **_PROPOSAL):
            prop.time_ms += on_ms - pad_ms
            ev = _extract_typed_evidence(
                prop, audio, sr, crnn_proba, (on_ms, off_ms, conf), dsp, fp_fn
            )
            out.append((prop.time_ms, ev))
    return out


def load_hecb(species: str, device: str = "cpu", *, repo_root: Path | None = None):
    """Load the species HECB CRNN + typed classifier + DSP config.

    Returns ``(crnn, clf, dsp)`` or ``(None, None, None)`` if torch is unavailable or the
    checkpoints don't exist (callers fall back gracefully).
    """
    if not _TORCH:
        return None, None, None
    dsp = _SPECIES_CONFIG.get(species)
    if dsp is None:
        return None, None, None
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    ckpt_dir = root / "checkpoints" / "hecb"
    crnn_pt = ckpt_dir / "crnn.pt"
    typed_pkl = ckpt_dir / "typed.pkl"
    if not (crnn_pt.exists() and typed_pkl.exists()):
        return None, None, None
    import pickle  # noqa: PLC0415

    crnn = CRNNDetector(n_mels=dsp.n_mels).to(device)
    crnn.load_state_dict(torch.load(crnn_pt, map_location=device))
    crnn.eval()
    with open(typed_pkl, "rb") as fh:
        _x, _y, clf = pickle.load(fh)
    return crnn, clf, dsp


def hecb_segment_audio(
    audio,
    sr,
    crnn,
    clf,
    dsp: HECBDSPConfig,
    *,
    profile: dict,
    fp_fn=None,
    device: str = "cpu",
    priors: HECBDurationPriors | None = None,
):
    """Audio → HECB CallEvents (with typed internal segments).

    Stage 1 (CRNN call events) → Stage 2-3 (typed proposals inside events) →
    ``hecb_decode`` (conservative, snap off → CRNN timing preserved + internal annotation).
    """
    if not _TORCH:
        raise RuntimeError(f"torch unavailable: {load_error}")
    proba = _crnn_proba(crnn, audio, dsp, device)
    events = _proba_to_events(proba, dsp)
    props = _event_conditioned_proposals(audio, sr, proba, events, profile, dsp, fp_fn)
    typed = [(t, clf.predict_proba(ev.reshape(1, -1))[0]) for t, ev in props]
    return hecb_decode(events, typed, priors or HECBDurationPriors(), snap=False, file_id="hecb")
