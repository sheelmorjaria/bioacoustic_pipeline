#!/usr/bin/env python3
"""HECB-Seg: Hierarchical Event-Conditioned Boundary Segmentation (Stages 3-5).

Implements the full 5-stage HECB-Seg pipeline on MeerKAT:
  Stage 1: PCEN + CRNN → call event isolation
  Stage 2: NBD proposals inside CRNN call events only
  Stage 3: Typed boundary evidence (NBD scalars + CRNN context + PCEN patch + FP delta)
  Stage 4: Typed proposal classifier (onset / offset / internal / noise via GBT)
  Stage 5: Semi-Markov decoder (duration-constrained event structure)

The key innovation: NBD runs ONLY inside CRNN-detected call events, where its
acoustic-change proposals are meaningful (not firing on background noise). Each proposal
is typed (onset/offset/internal/noise) using CRNN event geometry as context, then
decoded with duration constraints into coherent call events with optional internal boundaries.

Honest evaluation framing: at a 100 ms onset tolerance, the CRNN onset is already within
tolerance for true positives, so onset-F1 barely moves with refinement. The REAL value of
Stages 3-5 is (a) boundary PRECISION (lower onset/offset MAE → better downstream call-type
recovery), (b) offset localization, (c) internal segmentation, (d) spurious-event rejection.
We report all of these, not just onset-F1.

Usage:
  python3 scripts/train_hecb_seg.py --n-train 2000 --n-test 500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import CALIBRATOR_PROPOSAL  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

DATA = Path("/home/sheel/birdsong_analysis/data/MeerKAT_10s_2024-06-12")
LBL, WAV = DATA / "lbl" / "08000Hz", DATA / "wav" / "08000Hz"
SR, N_FFT, HOP, N_MELS = 8000, 512, 160, 64
FRAME_MS = HOP / SR * 1000.0
CALL_BAND = (1000, 4000)
ONSET_TOL_MS = 100.0  # GT onset proximity for labeling
EVAL_TOL_MS = 100.0
PAD_MS = 50.0  # NBD context padding around each CRNN event
# Conservative snapping: the CRNN's threshold-crossing onset/offset is already well-localized,
# so only refine it when a typed proposal is high-confidence AND close. Otherwise keep the
# CRNN boundary verbatim. This preserves CRNN timing while retaining HECB's offset-recall
# gain and internal-boundary capability.
SNAP_CONF = 0.5  # min typed-probability to accept a snap
SNAP_DIST_MS = 60.0  # max |proposal - CRNN boundary| to accept a snap

# ─── Typed labels ─────────────────────────────────────────────────────────────
NOISE = 0
ONSET = 1
OFFSET = 2
INTERNAL = 3
LABEL_NAMES = {0: "noise", 1: "onset", 2: "offset", 3: "internal"}


# ─── PCEN + CRNN (Stage 1) ──────────────────────────────────────────────────


def compute_pcen(audio, sr=SR):
    import torchaudio  # noqa: PLC0415

    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    stft = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP, window=torch.hann_window(N_FFT), return_complex=True
    )
    mag = stft.abs().pow(2).numpy()
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1,
        f_min=0.0,
        f_max=sr / 2,
        n_mels=N_MELS,
        sample_rate=sr,
    ).numpy()
    mel = fb.T @ mag
    b = 1.0 - np.exp(-HOP / (0.08 * sr))
    sm = np.zeros_like(mel)
    for t in range(mel.shape[1]):
        sm[:, t] = (1 - b) * (sm[:, t - 1] if t > 0 else 0) + b * mel[:, t]
    return (np.maximum(0.0, mel / (1e-6 + sm) ** 0.98 + 2.0) ** 0.5).astype(np.float32)


class CRNNDetector(nn.Module):
    def __init__(self):
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
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.lstm = nn.LSTM(128, 128, 1, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        c = self.conv(x).squeeze(2).permute(0, 2, 1)
        return self.fc(self.lstm(c)[0]).squeeze(-1)


def train_crnn(train_clips, device, epochs=50):
    """Stage 1: train PCEN+CRNN frame detector."""
    train_data = []
    for stem, spans in train_clips:
        try:
            audio, _ = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
        except Exception:
            continue
        pcen = compute_pcen(audio)
        nf = pcen.shape[1]
        labels = np.zeros(nf, dtype=np.float32)
        for s, e in spans:
            labels[int(s * 1000 / FRAME_MS) : min(int(e * 1000 / FRAME_MS) + 1, nf)] = 1.0
        train_data.append((pcen, labels))
    pos_rate = float(np.mean([l.mean() for _, l in train_data]))
    print(f"  CRNN train: {len(train_data)} clips, pos_rate={pos_rate:.3f}", flush=True)

    model = CRNNDetector().to(device)
    pw = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    MF = 500
    for ep in range(epochs):
        model.train()
        random.Random(42 + ep).shuffle(train_data)
        tl, nb = 0.0, 0
        for i in range(0, len(train_data), 8):
            batch = train_data[i : i + 8]
            B = len(batch)
            X = np.zeros((B, 1, N_MELS, MF), dtype=np.float32)
            Y = np.zeros((B, MF), dtype=np.float32)
            for j, (pc, lb) in enumerate(batch):
                t = min(pc.shape[1], MF)
                X[j, 0, :, :t] = pc[:, :t]
                Y[j, :t] = lb[:t]
            Xt, Yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            opt.zero_grad()
            loss = crit(model(Xt), Yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            nb += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs} loss={tl / nb:.4f}", flush=True)
    return model


def crnn_inference(model, audio, device):
    """Run CRNN → P(call) per frame."""
    model.eval()
    pcen = compute_pcen(audio)
    t = min(pcen.shape[1], 500)
    X = torch.from_numpy(pcen[:, :t]).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(X)).cpu().numpy()[0]


def proba_to_events(proba, enter=0.7, exit_=0.3, min_ms=40.0):
    """Hysteresis → call event windows (onset_ms, offset_ms, confidence)."""
    in_call = False
    events = []
    start = conf_sum = 0
    for i, p in enumerate(proba):
        if not in_call and p >= enter:
            in_call = True
            start = i
            conf_sum = p
        elif in_call:
            if p < exit_:
                in_call = False
                if (i - start) * FRAME_MS >= min_ms:
                    events.append((start * FRAME_MS, i * FRAME_MS, conf_sum / (i - start)))
            else:
                conf_sum += p
    if in_call and (len(proba) - start) * FRAME_MS >= min_ms:
        events.append((start * FRAME_MS, len(proba) * FRAME_MS, conf_sum / (len(proba) - start)))
    return events


# ─── Stage 3: Typed boundary evidence ────────────────────────────────────────


def _band_ratio(psd, freqs):
    band = (freqs >= CALL_BAND[0]) & (freqs <= CALL_BAND[1])
    return float(psd[band].sum() / (psd.sum() + 1e-12))


def _flatness(psd):
    return float(np.exp(np.log(psd + 1e-12).mean()) / (psd.mean() + 1e-12))


def _win_spec(audio, sr, t0_ms, t1_ms):
    i, j = int(t0_ms / 1000 * sr), int(t1_ms / 1000 * sr)
    seg = audio[i : max(j, i + 8)]
    if len(seg) < 8:
        return np.ones(8) * 1e-12, np.zeros(8)
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return mag**2 + 1e-12, np.fft.rfftfreq(len(seg), 1 / sr)


def extract_typed_evidence(prop, audio, sr, crnn_proba, event, fp_fn, frame_ms=FRAME_MS):
    """Stage 3: extract 4 evidence groups for one NBD proposal.

    Returns a flat feature vector combining:
      A. NBD scalar features (13)
      B. CRNN event-context features (9)
      C. PCEN patch features (8)
      D. Fingerprint delta features (2)  [optional — pass fp_fn=None to zero]

    Args:
        fp_fn: callable(seg_audio, sr) -> 1-D vector, or None to skip (zeros).
    """
    t = prop.time_ms
    fi = int(t / frame_ms)

    # A. NBD scalars
    nbd_feats = proposal_features(prop)  # 13-D

    # B. CRNN event-context
    p_at = float(crnn_proba[fi]) if fi < len(crnn_proba) else 0.0
    p_before = float(crnn_proba[max(0, fi - 5)]) if fi - 5 < len(crnn_proba) else 0.0
    p_after = (
        float(crnn_proba[min(len(crnn_proba) - 1, fi + 5)]) if fi + 5 < len(crnn_proba) else 0.0
    )
    rising = max(0.0, p_at - p_before)
    falling = max(0.0, p_before - p_at)
    if event is not None:
        onset_ms, offset_ms, conf = event
        dist_onset = abs(t - onset_ms)
        dist_offset = abs(t - offset_ms)
        inside = 1.0 if onset_ms <= t <= offset_ms else 0.0
        event_conf = conf
    else:
        dist_onset = dist_offset = 9999.0
        event_conf = inside = 0.0
    crnn_feats = np.array(
        [p_at, p_before, p_after, rising, falling, inside, dist_onset, dist_offset, event_conf]
    )

    # C. PCEN patch features (pre/post/body windows)
    psd_pre, fr_pre = _win_spec(audio, sr, t - 40, t)
    psd_post, fr_post = _win_spec(audio, sr, t, t + 40)
    psd_body, fr_body = _win_spec(audio, sr, t + 20, t + 60)
    br_pre, br_post = _band_ratio(psd_pre, fr_pre), _band_ratio(psd_post, fr_post)
    br_body = _band_ratio(psd_body, fr_body)
    fl_pre, fl_post = _flatness(psd_pre), _flatness(psd_post)
    body_rms = float(np.sqrt(psd_body.mean()))
    sustain = body_rms / (float(np.sqrt(psd_pre.mean())) + 1e-8)
    patch_feats = np.array(
        [br_pre, br_post, fl_pre, fl_post, br_body, sustain, _flatness(psd_body), body_rms]
    )

    # D. Fingerprint delta (optional — expensive, 2 full extract() calls)
    if fp_fn is not None:
        seg_pre = audio[max(0, int((t - 40) / 1000 * sr)) : int(t / 1000 * sr)]
        seg_post = audio[int(t / 1000 * sr) : int((t + 40) / 1000 * sr)]
        if len(seg_pre) > 8 and len(seg_post) > 8:
            fp_pre = np.asarray(fp_fn(seg_pre, sr), dtype=np.float64)
            fp_post = np.asarray(fp_fn(seg_post, sr), dtype=np.float64)
            delta = fp_post - fp_pre
            fp_feats = np.array([float(np.linalg.norm(delta)), float(np.mean(np.abs(delta)))])
        else:
            fp_feats = np.array([0.0, 0.0])
    else:
        fp_feats = np.array([0.0, 0.0])

    return np.concatenate([nbd_feats, crnn_feats, patch_feats, fp_feats])


# ─── Stage 4: Typed proposal classifier ─────────────────────────────────────


def label_proposal(prop_time_ms, gt_spans, onset_tol=ONSET_TOL_MS):
    """Assign typed label: onset / offset / internal / noise."""
    for s, e in gt_spans:
        s_ms, e_ms = s * 1000, e * 1000
        if abs(prop_time_ms - s_ms) < onset_tol:
            return ONSET
        if abs(prop_time_ms - e_ms) < onset_tol:
            return OFFSET
        if s_ms <= prop_time_ms <= e_ms:
            return INTERNAL
    return NOISE


def train_typed_classifier(X, y):
    """Stage 4: train GBT typed proposal classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(X, y)
    return clf


# ─── Stage 5: Semi-Markov event decoder ─────────────────────────────────────


@dataclass
class DurationPriors:
    """Learned from GT call spans."""

    min_call_ms: float = 40.0
    max_call_ms: float = 2000.0
    min_gap_ms: float = 30.0
    max_internal_per_call: int = 5
    median_call_ms: float = 100.0


def fit_duration_priors(all_spans):
    """Learn call duration + gap distributions from GT."""
    durs = [(e - s) * 1000 for spans in all_spans for s, e in spans]
    gaps = []
    for spans in all_spans:
        ss = sorted(spans)
        for i in range(1, len(ss)):
            gaps.append((ss[i][0] - ss[i - 1][1]) * 1000)
    return DurationPriors(
        min_call_ms=max(30.0, float(np.percentile(durs, 5)) if durs else 40.0),
        max_call_ms=min(5000.0, float(np.percentile(durs, 99)) if durs else 2000.0),
        min_gap_ms=max(20.0, float(np.percentile(gaps, 5)) if gaps else 30.0),
        median_call_ms=float(np.median(durs)) if durs else 100.0,
    )


def _safe_snap(boundary_ms, ev_props, label_idx, conf, dist_ms):
    """Conservative onset/offset refinement.

    Only snap the CRNN boundary to a typed proposal when one is high-confidence (``conf``)
    AND close (``dist_ms``) AND role-matched (``label_idx``). Otherwise return the CRNN
    boundary unchanged — preserving the CRNN's good timing. Among qualifying proposals,
    pick the highest-confidence one (all are within ``dist_ms`` of the boundary, so the
    snap is guaranteed small).
    """
    cands = [
        (t, p[label_idx])
        for t, p in ev_props
        if p[label_idx] > conf and abs(t - boundary_ms) < dist_ms
    ]
    if not cands:
        return boundary_ms  # fallback: keep CRNN boundary verbatim
    return max(cands, key=lambda x: x[1])[0]


def decode_events(
    crnn_events,
    typed_proposals,
    priors,
    pad_ms=PAD_MS,
    snap_conf=SNAP_CONF,
    snap_dist_ms=SNAP_DIST_MS,
):
    """Stage 5: decode coherent call events with duration constraints.

    Greedy per-event decoder (interpretable + efficient): for each CRNN event, CONSERVATIVELY
    refine its onset/offset (only snap when a typed proposal is confident + close + role-matched,
    else keep the CRNN boundary), gate by duration priors, and extract internal boundaries
    for long calls. The CRNN is the authoritative call-event detector; this layer only
    annotates typed internal structure + tightens a boundary when it is clearly safe to.

    Args:
        crnn_events: list of (onset_ms, offset_ms, confidence).
        typed_proposals: list of (time_ms, proba_4vector) for proposals in all events.
        priors: DurationPriors.
        snap_conf / snap_dist_ms: conservative-snap gate (see _safe_snap).

    Returns: list of {call_id, onset_ms, offset_ms, crnn_confidence, internal_boundaries}.
    """
    decoded = []
    for ev_idx, (onset_ms, offset_ms, conf) in enumerate(crnn_events):
        ev_props = [
            (t, p) for t, p in typed_proposals if onset_ms - pad_ms <= t <= offset_ms + pad_ms
        ]

        # Conservative onset/offset refinement (preserves CRNN timing by default)
        refined_onset = _safe_snap(onset_ms, ev_props, ONSET, snap_conf, snap_dist_ms)
        refined_offset = _safe_snap(offset_ms, ev_props, OFFSET, snap_conf, snap_dist_ms)

        # Order safety (offset must follow onset) — revert offset on inversion
        if refined_offset <= refined_onset:
            refined_offset = offset_ms

        # Duration gate (Stage 5 constraint enforcement)
        call_dur = refined_offset - refined_onset
        if call_dur < priors.min_call_ms:
            continue  # too short — reject spurious event
        if call_dur > priors.max_call_ms:
            refined_offset = refined_onset + priors.max_call_ms

        # Internal boundaries (only for long calls)
        internal_boundaries = []
        if call_dur > 250:
            internal_props = [
                (t, p[INTERNAL])
                for t, p in ev_props
                if p[INTERNAL] > 0.3 and refined_onset + 50 < t < refined_offset - 50
            ]
            internal_props.sort(key=lambda x: -x[1])  # confidence desc
            placed = []
            for t, _ in internal_props:
                if len(placed) >= priors.max_internal_per_call:
                    break
                if all(abs(t - pt) > priors.min_gap_ms for pt in placed):
                    placed.append(t)
            internal_boundaries = sorted(placed)

        decoded.append(
            {
                "call_id": f"call_{ev_idx}",
                "onset_ms": refined_onset,
                "offset_ms": refined_offset,
                "crnn_confidence": conf,
                "internal_boundaries": internal_boundaries,
            }
        )
    return decoded


# ─── Pipeline glue: event-conditioned proposal generation ───────────────────


def event_conditioned_proposals(audio, sr, crnn_proba, crnn_events, profile, fp_fn):
    """Stage 2 + 3: run NBD inside each CRNN event, return typed-evidence proposals.

    Returns: list of (abs_time_ms, Proposal, evidence_vector).
    """
    out = []
    for onset_ms, offset_ms, conf in crnn_events:
        lo = max(0, int((onset_ms - PAD_MS) / 1000 * sr))
        hi = min(len(audio), int((offset_ms + PAD_MS) / 1000 * sr))
        event_audio = audio[lo:hi]
        if len(event_audio) < int(sr * 0.02):
            continue
        event_sig = compute_signals(event_audio, sr, hop_ms=profile["hop_ms"])
        event_props = generate_proposals(event_sig, **CALIBRATOR_PROPOSAL)
        ev_tuple = (onset_ms, offset_ms, conf)
        for prop in event_props:
            prop.time_ms += onset_ms - PAD_MS  # absolute time
            ev = extract_typed_evidence(prop, audio, sr, crnn_proba, ev_tuple, fp_fn)
            out.append((prop.time_ms, prop, ev))
    return out


# ─── Data + main ──────────────────────────────────────────────────────────────


def sample_clips(n, seed):
    rng = random.Random(seed)
    files = sorted(glob.glob(str(LBL / "*.h5")))
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        with h5py.File(f, "r") as h:
            st = np.asarray(h["start_time_lbl"][:]).ravel()
            et = np.asarray(h["end_time_lbl"][:]).ravel()
        st = st[np.isfinite(st)]
        stem = Path(f).stem
        if st.size and (WAV / f"{stem}.wav").exists():
            spans = [
                (float(s), float(e) if np.isfinite(e) else float(s) + 0.08) for s, e in zip(st, et)
            ]
            out.append((stem, spans))
    return out


def main():
    ap = argparse.ArgumentParser(description="HECB-Seg: full 5-stage pipeline")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--crnn-epochs", type=int, default=50)
    ap.add_argument("--enter", type=float, default=0.7)
    ap.add_argument("--exit", type=float, default=0.3)
    ap.add_argument("--no-fp", action="store_true", help="skip fingerprint-delta feature (faster)")
    ap.add_argument(
        "--retrain", action="store_true", help="force retrain CRNN + classifier (ignore cache)"
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="load cached CRNN + classifier + test fixture, re-run decode/eval only",
    )
    ap.add_argument(
        "--snap",
        action="store_true",
        help="ENABLE onset/offset snapping to typed proposals (default OFF — snapping "
        "empirically COARSENS boundaries; HECB's role is internal annotation only)",
    )
    ap.add_argument(
        "--snap-conf",
        type=float,
        default=SNAP_CONF,
        help="min typed-probability to snap a CRNN onset/offset (default %(default)s)",
    )
    ap.add_argument(
        "--snap-dist",
        type=float,
        default=SNAP_DIST_MS,
        help="max |proposal-CRNN boundary| ms to snap (default %(default)s)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # Snapping is OFF by default: the CRNN's threshold-crossing onset/offset (~21 ms MAE) is
    # already more precise than any 20 ms-quantized NBD proposal. HECB annotates internal
    # sub-call structure; it does not refine CRNN call-event boundaries.
    snap_conf = args.snap_conf if args.snap else float("inf")
    snap_dist = args.snap_dist if args.snap else float("inf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = load_species_profiles()["meerkat"]
    print(f"device: {device}", flush=True)

    if not args.no_fp:
        from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415

        fp_fn = lambda seg, sr: afp.extract(seg, sr, species="meerkat").combined  # noqa: E731
    else:
        fp_fn = None
    print(f"fingerprint delta: {'OFF' if args.no_fp else 'ON'}", flush=True)

    clips = sample_clips(args.n_train + args.n_test, args.seed)
    train_clips, test_clips = clips[: args.n_train], clips[args.n_train :]
    print(f"{len(train_clips)} train / {len(test_clips)} test", flush=True)

    # ─── Stage 1: train CRNN (cached so decoder iterations reuse the SAME model) ───
    cache = ROOT / "checkpoints" / "hecb"
    crnn_ckpt = cache / "crnn.pt"
    use_cache = not args.retrain and not args.eval_only
    if (crnn_ckpt.exists() and use_cache) or args.eval_only:
        print(f"Stage 1: loading cached CRNN ({crnn_ckpt.relative_to(ROOT)})...", flush=True)
        crnn = CRNNDetector().to(device)
        crnn.load_state_dict(torch.load(crnn_ckpt, map_location=device))
    else:
        print("Stage 1: training PCEN+CRNN...", flush=True)
        crnn = train_crnn(train_clips, device, args.crnn_epochs)
        cache.mkdir(parents=True, exist_ok=True)
        torch.save(crnn.state_dict(), crnn_ckpt)

    # ─── Fit duration priors (deterministic from train spans — always recompute) ───
    priors = fit_duration_priors([c[1] for c in train_clips])
    print(
        f"Duration priors: min_call={priors.min_call_ms:.0f}ms median={priors.median_call_ms:.0f}ms "
        f"max={priors.max_call_ms:.0f}ms min_gap={priors.min_gap_ms:.0f}ms",
        flush=True,
    )

    # ─── Stages 2-4: typed evidence + classifier (cached) ───
    import pickle  # noqa: PLC0415

    typed_ckpt = cache / "typed.pkl"
    if typed_ckpt.exists() and use_cache:
        print(
            f"Stages 2-4: loading cached typed classifier ({typed_ckpt.relative_to(ROOT)})...",
            flush=True,
        )
        with open(typed_ckpt, "rb") as fh:
            X_train, y_train, clf = pickle.load(fh)
    else:
        print("Stages 2-3: generating typed evidence (train)...", flush=True)
        X_train, y_train = [], []
        for ci, (stem, spans) in enumerate(train_clips):
            try:
                audio, _ = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
            except Exception:
                continue
            proba = crnn_inference(crnn, audio, device)
            events = proba_to_events(proba, args.enter, args.exit)
            props = event_conditioned_proposals(audio, SR, proba, events, profile, fp_fn)
            for _, prop, ev in props:
                X_train.append(ev)
                y_train.append(label_proposal(prop.time_ms, spans))
            if (ci + 1) % 500 == 0:
                print(f"  {ci + 1}/{len(train_clips)} clips, {len(X_train)} proposals", flush=True)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if len(X_train) == 0:
            print(
                "ERROR: no proposals generated — CRNN too weak (lower --enter or train longer).",
                flush=True,
            )
            sys.exit(1)

        print("Stage 4: training typed GBT classifier...", flush=True)
        clf = train_typed_classifier(X_train, y_train)
        cache.mkdir(parents=True, exist_ok=True)
        with open(typed_ckpt, "wb") as fh:
            pickle.dump([X_train, y_train, clf], fh)

    label_counts = Counter(y_train.tolist())
    print(
        f"Typed proposals: {len(X_train)} | "
        + " ".join(f"{LABEL_NAMES[k]}={v}" for k, v in sorted(label_counts.items())),
        flush=True,
    )
    from sklearn.metrics import classification_report  # noqa: PLC0415

    print(
        classification_report(
            y_train,
            clf.predict(X_train),
            target_names=[LABEL_NAMES[i] for i in range(4)],
            zero_division=0,
            digits=3,
        ),
        flush=True,
    )

    # ─── Phase A: extract test fixture (CRNN events + typed-proposal evidence) ───
    # Cached so decoder-threshold iterations (Phase B) are fast and reuse the SAME
    # CRNN + classifier — isolating the decoder change from model randomness.
    fixture_ckpt = cache / "test_fixture.pkl"
    if fixture_ckpt.exists() and not args.retrain:
        print(
            f"Phase A: loading cached test fixture ({fixture_ckpt.relative_to(ROOT)})...",
            flush=True,
        )
        with open(fixture_ckpt, "rb") as fh:
            fixture = pickle.load(fh)
    else:
        print("Phase A: extracting test fixture (CRNN events + typed evidence)...", flush=True)
        fixture = []  # per-clip: {gt_on, gt_off, crnn_events, proposals: [(t, evidence_vec)]}
        for ci, (stem, spans) in enumerate(test_clips):
            try:
                audio, _ = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
            except Exception:
                continue
            proba = crnn_inference(crnn, audio, device)
            crnn_events = proba_to_events(proba, args.enter, args.exit)
            props = event_conditioned_proposals(audio, SR, proba, crnn_events, profile, fp_fn)
            fixture.append(
                {
                    "gt_on": [s * 1000 for s, _ in spans],
                    "gt_off": [e * 1000 for _, e in spans],
                    "crnn_events": crnn_events,
                    "proposals": [(t, ev) for t, _, ev in props],  # (time_ms, evidence_vec)
                }
            )
            if (ci + 1) % 100 == 0:
                print(f"  {ci + 1}/{len(test_clips)} clips", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        with open(fixture_ckpt, "wb") as fh:
            pickle.dump(fixture, fh)

    # ─── Phase B: decode + score (fast — iterate snap thresholds here) ───
    snap_label = "OFF (preserve CRNN)" if not args.snap else f"conf={snap_conf} dist={snap_dist}ms"
    print(
        f"Phase B: decoding + scoring (onset/offset tol={EVAL_TOL_MS}ms, snap={snap_label})...",
        flush=True,
    )
    m = {"c": _aggr(), "h": _aggr()}
    # Per-clip matched (gt_idx -> pred) maps, to compute recall-unconfounded boundary
    # MAE on the INTERSECTION of GT calls both methods matched (same calls compared).
    on_match = {"c": [], "h": []}  # list of {gt_idx: error_ms}
    off_match = {"c": [], "h": []}
    n_internal_calls = 0
    n_internal_total = 0
    for fx in fixture:
        gt_on, gt_off = fx["gt_on"], fx["gt_off"]
        crnn_events = fx["crnn_events"]

        # CRNN-only baselines (onset + offset)
        crnn_on = [e[0] for e in crnn_events]
        crnn_off = [e[1] for e in crnn_events]
        _accumulate(m["c"], prf_one_to_one(crnn_on, gt_on, tol=EVAL_TOL_MS), "on")
        _accumulate(m["c"], prf_one_to_one(crnn_off, gt_off, tol=EVAL_TOL_MS), "off")
        on_match["c"].append(_match_map(crnn_on, gt_on, EVAL_TOL_MS))
        off_match["c"].append(_match_map(crnn_off, gt_off, EVAL_TOL_MS))

        # HECB-Seg full pipeline (typed classifier over cached evidence → conservative decoder)
        typed = [(t, clf.predict_proba(ev.reshape(1, -1))[0]) for t, ev in fx["proposals"]]
        decoded = decode_events(
            crnn_events, typed, priors, snap_conf=snap_conf, snap_dist_ms=snap_dist
        )
        hecb_on = [d["onset_ms"] for d in decoded]
        hecb_off = [d["offset_ms"] for d in decoded]
        _accumulate(m["h"], prf_one_to_one(hecb_on, gt_on, tol=EVAL_TOL_MS), "on")
        _accumulate(m["h"], prf_one_to_one(hecb_off, gt_off, tol=EVAL_TOL_MS), "off")
        on_match["h"].append(_match_map(hecb_on, gt_on, EVAL_TOL_MS))
        off_match["h"].append(_match_map(hecb_off, gt_off, EVAL_TOL_MS))
        for d in decoded:
            if d["internal_boundaries"]:
                n_internal_calls += 1
                n_internal_total += len(d["internal_boundaries"])

    # Intersection MAE: same GT calls compared across methods
    on_mae = {
        "c": _intersect_mae(on_match["c"], on_match["h"]),
        "h": _intersect_mae(on_match["h"], on_match["c"]),
    }
    off_mae = {
        "c": _intersect_mae(off_match["c"], off_match["h"]),
        "h": _intersect_mae(off_match["h"], off_match["c"]),
    }

    print(f"\n=== HECB-Seg vs CRNN-only (MeerKAT, {len(test_clips)} test, one-to-one) ===")
    _print_table("CRNN only", m["c"], on_mae["c"], off_mae["c"])
    _print_table("HECB-Seg (full)", m["h"], on_mae["h"], off_mae["h"])
    print("  (onset/offset MAE = mean |pred-gt| on GT calls matched by BOTH methods)")
    print(
        f"  internal segmentation: {n_internal_calls} calls with internal boundaries "
        f"({n_internal_total} boundaries total)"
    )
    print("  raw NBD ref F1~0.079 | v2 tabular ref ~0.26 | CRNN ref 0.703 (onset, 100ms)")
    print("HECB-SEG DONE")


def _aggr():
    return {
        "tp": {"on": 0, "off": 0},
        "fp": {"on": 0, "off": 0},
        "fn": {"on": 0, "off": 0},
        "n": {"on": 0, "off": 0},
    }


def _accumulate(a, prf, kind):
    a["tp"][kind] += prf.tp
    a["fp"][kind] += prf.fp
    a["fn"][kind] += prf.fn


def _match_map(pred, gt, tol):
    """Greedy one-to-one match → {gt_idx: abs(pred - gt) ms} for matched TPs."""
    used = set()
    mm = {}
    for p in pred:
        best, bd = None, tol
        for gi, g in enumerate(gt):
            if gi in used:
                continue
            d = abs(p - g)
            if d < bd:
                bd, best = d, gi
        if best is not None:
            used.add(best)
            mm[best] = bd
    return mm


def _intersect_mae(mine, other):
    """MAE of `mine` restricted to GT idxs matched by BOTH mine and other (per clip).

    Same calls → fair precision comparison, unconfounded by recall differences.
    """
    errs = []
    for my_mm, other_mm in zip(mine, other):
        for gi, err in my_mm.items():
            if gi in other_mm:
                errs.append(err)
    return float(np.mean(errs)) if errs else float("nan")


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-8), p, r


def _print_table(label, a, on_mae, off_mae):
    onf, onp, onr = _prf(a["tp"]["on"], a["fp"]["on"], a["fn"]["on"])
    off, ofp, ofr = _prf(a["tp"]["off"], a["fp"]["off"], a["fn"]["off"])
    print(
        f"  {label:20s} onset F1={onf:.3f} P={onp:.3f} R={onr:.3f} MAE={on_mae:.1f}ms | "
        f"offset F1={off:.3f} P={ofp:.3f} R={ofr:.3f} MAE={off_mae:.1f}ms"
    )


if __name__ == "__main__":
    main()
