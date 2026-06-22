#!/usr/bin/env python3
"""HECB vs HuBERT vs CPC vs fingerprint: unsupervised word-structure benchmark.

Same LibriSpeech words, four unit-discovery methods, same metrics. The decisive comparison:
is HECB's interpretable boundary-based approach competitive with SSL frame clustering?

  HECB:    oracle word → NBD proposals → typed classifier → internal segments → fp clusters
  HuBERT:  oracle word → HuBERT frames → k-means → collapse repeats
  CPC:     oracle word → CPC frames → k-means → collapse repeats
  AFP:     oracle word → NBD proposals → fingerprint per proposal → clusters

Metrics: same-word vs across-word edit distance + nearest-neighbor word accuracy.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.segmentation.hierarchical import hecb_decode, HECBDurationPriors  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402

import hecb_librispeech as L  # noqa: E402


def edit_dist_norm(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0 if m == n else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[n] / max(m, n)


def collapse_repeats(tokens):
    """Collapse consecutive duplicate tokens: [3,3,7,7,7,2] → [3,7,2]."""
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def kmeans_units(frames, k, seed=0):
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    scaler = StandardScaler().fit(frames)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(frames))
    sv = scaler.transform(frames)
    return np.argmin(((sv[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)


def same_word_test(word_seqs, min_freq=5):
    """Same-word consistency: within < across edit distance + NN accuracy."""
    groups = defaultdict(list)
    for word, seq in word_seqs:
        groups[word].append(seq)
    freq = {w: s for w, s in groups.items() if len(s) >= min_freq}
    within, across = [], []
    rng = random.Random(0)
    all_seqs = [s for _, s in word_seqs]
    for word, seqs in freq.items():
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                within.append(edit_dist_norm(seqs[i], seqs[j]))
            for _ in range(len(seqs)):
                across.append(edit_dist_norm(seqs[0], rng.choice(all_seqs)))
    # NN accuracy: for each word instance, is nearest neighbor the same word?
    nn_correct, nn_total = 0, 0
    for idx, (word, seq) in enumerate(word_seqs):
        if word not in freq:
            continue
        best_d, best_word = 2.0, "?"
        for jdx, (oword, oseq) in enumerate(word_seqs):
            if idx == jdx:
                continue
            d = edit_dist_norm(seq, oseq)
            if d < best_d:
                best_d, best_word = d, oword
        if best_word != "?":
            nn_total += 1
            nn_correct += (best_word == word)
    w = float(np.mean(within)) if within else 0
    a = float(np.mean(across)) if across else 0
    nn = nn_correct / max(nn_total, 1)
    return w, a, nn, len(within), len(freq)


def main():
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="HECB vs HuBERT vs CPC word-structure benchmark")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    print("loading wav2vec2 aligner...", flush=True)
    import torchaudio  # noqa: PLC0415
    import torch  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model = bundle.get_model()
    labels = bundle.get_labels()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading HuBERT...", flush=True)
    hubert_bundle = torchaudio.pipelines.HUBERT_BASE
    hubert_model = hubert_bundle.get_model().to(device)
    hubert_model.eval()

    print("loading CPC...", flush=True)
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
    cpc = CPCBackbone()
    cpc_available = cpc.available

    files = L.sample_files(args.n_test, 0)

    # ─── Pre-align + extract frames ───
    print(f"aligning {args.n_test} files + extracting HuBERT/CPC frames...", flush=True)
    hecb_seqs, hubert_seqs, cpc_seqs = [], [], []
    for fi, flac in enumerate(files):
        from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415
        audio, _ = load_audio(flac, target_sr=L.SR)
        if audio.size == 0:
            continue
        tr = L.load_librispeech_transcript(flac)
        if not tr:
            continue
        res = L.forced_align(tr, audio, L.SR, model=fa_model, labels=labels)
        if not res.word_spans:
            continue

        # HuBERT features (frame-level)
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            hubert_feats = hubert_model(wav)[0].squeeze(0).cpu().numpy()  # (T, 768)

        # CPC features (frame-level)
        if cpc_available:
            cpc_feats = cpc.embed_frames(audio, L.SR)  # (T_cpc, 256)
        else:
            cpc_feats = None

        for (ws, we, word) in res.word_spans:
            if we - ws < 80:
                continue
            # Frame indices for this word
            fi_start = int(ws / 1000 * 100)  # 10ms frames
            fi_end = int(we / 1000 * 100)

            # ── HECB sub-units (fingerprint clusters) ──
            props = L.word_internal_proposals(audio, L.SR, ws, we)
            if props:
                ev = np.asarray([L.ls_evidence(p, audio, L.SR, ws, we) for p in props])
                from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
                # Use raw proposal fingerprints as clusters (no typed clf needed for benchmark)
                seg_fps = []
                for p in props:
                    si = max(0, int(p.time_ms / 1000 * L.SR) - int(0.02 * L.SR))
                    sj = si + int(0.04 * L.SR)
                    seg = audio[si:sj]
                    if len(seg) >= int(0.012 * L.SR):
                        sfp = np.asarray(afp.extract(seg, L.SR, species="human").combined, dtype=np.float32)
                        if np.any(sfp):
                            seg_fps.append(sfp)
                if seg_fps:
                    hecb_seqs.append((word.upper(), seg_fps))

            # ── HuBERT frame units (k-means + collapse) ──
            h_start = max(0, min(fi_start, hubert_feats.shape[0] - 1))
            h_end = max(h_start + 1, min(fi_end, hubert_feats.shape[0]))
            word_hubert = hubert_feats[h_start:h_end]
            if len(word_hubert) >= 2:
                hubert_seqs.append((word.upper(), word_hubert))

            # ── CPC frame units ──
            if cpc_feats is not None:
                c_start = max(0, min(int(ws / 1000 * 100), cpc_feats.shape[0] - 1))
                c_end = max(c_start + 1, min(int(we / 1000 * 100), cpc_feats.shape[0]))
                word_cpc = cpc_feats[c_start:c_end]
                if len(word_cpc) >= 2:
                    cpc_seqs.append((word.upper(), word_cpc))

        if (fi + 1) % 50 == 0:
            print(f"  {fi+1}/{args.n_test} files | HECB={len(hecb_seqs)} HuBERT={len(hubert_seqs)} "
                  f"CPC={len(cpc_seqs)}", flush=True)

    print(f"\n  {len(hecb_seqs)} HECB words | {len(hubert_seqs)} HuBERT words | "
          f"{len(cpc_seqs)} CPC words", flush=True)

    # ─── Cluster + build unit sequences ───
    print(f"clustering (k={args.k})...", flush=True)

    # HECB: cluster all sub-call fingerprints
    hecb_all = [fp for _, fps in hecb_seqs for fp in fps]
    hecb_words_typed = []
    if len(hecb_all) >= args.k * 5:
        hecb_units = kmeans_units(np.asarray(hecb_all, dtype=np.float32), args.k)
        idx = 0
        for word, fps in hecb_seqs:
            n = len(fps)
            hecb_words_typed.append((word, hecb_units[idx:idx + n].tolist()))
            idx += n

    # HuBERT: cluster all frames, collapse per word
    hubert_all = np.concatenate([f for _, f in hubert_seqs]) if hubert_seqs else np.zeros((0, 768))
    hubert_words_typed = []
    if len(hubert_all) >= args.k * 5:
        hubert_units_global = kmeans_units(hubert_all, args.k)
        idx = 0
        for word, frames in hubert_seqs:
            n = len(frames)
            raw = hubert_units_global[idx:idx + n].tolist()
            hubert_words_typed.append((word, collapse_repeats(raw)))
            idx += n

    # CPC: same
    cpc_words_typed = []
    if cpc_seqs and len(cpc_seqs) >= args.k * 5:
        cpc_all = np.concatenate([f for _, f in cpc_seqs])
        cpc_units_global = kmeans_units(cpc_all, args.k)
        idx = 0
        for word, frames in cpc_seqs:
            n = len(frames)
            raw = cpc_units_global[idx:idx + n].tolist()
            cpc_words_typed.append((word, collapse_repeats(raw)))
            idx += n

    # ─── Run same-word consistency on each ───
    print(f"\n{'='*70}")
    print(f"UNSUPERVISED WORD-STRUCTURE BENCHMARK (k={args.k})")
    print(f"{'='*70}")
    print(f"\n{'method':16} {'within':>7} {'across':>7} {'ratio':>6} {'NN acc':>7} {'n_pairs':>7} {'n_types':>7}")
    print(f"{'-'*60}")
    for name, seqs in [("HECB", hecb_words_typed), ("HuBERT-km", hubert_words_typed),
                        ("CPC-km", cpc_words_typed)]:
        if len(seqs) < 50:
            print(f"  {name:14} insufficient data ({len(seqs)} words)")
            continue
        w, a, nn, npairs, ntypes = same_word_test(seqs, min_freq=5)
        print(f"  {name:14} {w:7.3f} {a:7.3f} {w/max(a,1e-6):6.3f} {nn:7.3f} {npairs:7d} {ntypes:7d}")

    print("\nUNIT COMPARISON DONE")


if __name__ == "__main__":
    main()
