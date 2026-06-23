#!/usr/bin/env python3
"""Stage D — ES-KMeans-style lexicon discovery on LibriSpeech (acoustic word-embedding clustering).

The classic unsupervised word-discovery core: represent each word span as a fixed-dim
acoustic embedding, cluster into discovered "word types", evaluate against GT word labels.
This isolates the *embedding/clustering* quality (oracle GT spans; the joint
segmentation+clustering of full ES-KMeans is deferred). Four embedding sources:

    fingerprint — whole-word acoustic fingerprint (~43-D, interpretable)
    CPC-mean    — mean-pooled in-house CPC frame embeddings over the word (~256-D)
    HuBERT-mean — mean-pooled HuBERT frame embeddings over the word (~768-D)
    HECB-pooled — mean of HECB sub-call fingerprints within the word (~43-D)

Metrics: V-measure, adjusted Rand index (ARI), cluster purity, token-F1 (many-to-one
cluster→word mapping). Complements Stage C (subsequence discovery) by testing whole-word
clustering — does a learned word lexicon emerge from each embedding?

Human-speech word GT → "word" honest here. Run:  python scripts/eskmeans_librispeech.py --n-files 200

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402

import hecb_librispeech as L  # noqa: E402


def cluster_eval(labels_true, labels_pred):
    from sklearn.metrics import (  # noqa: PLC0415
        v_measure_score,
        adjusted_rand_score,
    )

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    # purity: weighted mean of majority-class fraction per cluster
    clusters = {}
    for t, p in zip(labels_true, labels_pred):
        clusters.setdefault(p, []).append(t)
    purity = sum(max(Counter(v).values()) for v in clusters.values()) / max(n, 1)
    # token-F1 (many-to-one: each cluster → its majority word)
    clst_to_word = {p: Counter(v).most_common(1)[0][0] for p, v in clusters.items()}
    tp = fp = fn = 0  # per the many-to-one mapping, micro over tokens
    word_to_clst = {}
    for p, w in clst_to_word.items():
        word_to_clst.setdefault(w, set()).add(p)
    for w in set(labels_true.tolist()):
        assigned = {p for p in clusters if clst_to_word[p] == w}
        tp_w = sum(1 for t, p in zip(labels_true, labels_pred) if t == w and p in assigned)
        fp_w = sum(1 for t, p in zip(labels_true, labels_pred) if t != w and p in assigned)
        fn_w = sum(1 for t, p in zip(labels_true, labels_pred) if t == w and p not in assigned)
        tp += tp_w
        fp += fp_w
        fn += fn_w
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(
        V=v_measure_score(labels_true, labels_pred),
        ARI=adjusted_rand_score(labels_true, labels_pred),
        purity=purity,
        token_f1=f1,
        n_types=len(set(labels_true.tolist())),
    )


def main():
    ap = argparse.ArgumentParser(description="Stage D: ES-KMeans-style word-embedding clustering")
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner + HuBERT + CPC...", flush=True)
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model, labels = bundle.get_model(), bundle.get_labels()
    hubert = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415,E402

    cpc = CPCBackbone()
    cpc_ok = cpc.available
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="human").combined  # noqa: E731

    files = L.sample_files(args.n_files, args.seed)
    recs = []  # {word, fp(whole), hecb(mean sub-call), hubert(mean), cpc(mean)}
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
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            hubert_all = hubert(wav)[0].squeeze(0).cpu().numpy()
        cpc_all = cpc.embed_frames(audio, L.SR) if cpc_ok else None
        for ws, we, word in res.word_spans:
            if we - ws < 80:
                continue
            i, j = int(ws / 1000 * L.SR), int(we / 1000 * L.SR)
            seg = audio[i:j]
            if len(seg) < int(0.03 * L.SR):
                continue
            whole = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
            if not np.any(whole):
                continue
            hecb_fps = []
            for p in L.word_internal_proposals(audio, L.SR, ws, we):
                si = max(0, int(p.time_ms / 1000 * L.SR) - int(0.02 * L.SR))
                sj = si + int(0.04 * L.SR)
                s2 = audio[si:sj]
                if len(s2) >= int(0.012 * L.SR):
                    v = np.asarray(fp_fn(s2, L.SR), dtype=np.float32)
                    if np.any(v):
                        hecb_fps.append(v)
            hecb_vec = np.mean(hecb_fps, axis=0) if hecb_fps else np.zeros_like(whole)
            fs, fe = int(ws / 1000 * 100), int(we / 1000 * 100)
            hubert_vec = (
                hubert_all[fs:fe].mean(axis=0) if fe > fs + 1 and fe <= len(hubert_all) else None
            )
            cpc_vec = None
            if cpc_all is not None:
                cs = min(int(ws / 1000 * 100), len(cpc_all) - 1)
                ce = max(cs + 1, min(int(we / 1000 * 100), len(cpc_all)))
                if ce > cs + 1:
                    cpc_vec = cpc_all[cs:ce].mean(axis=0)
            if hubert_vec is not None:
                recs.append(
                    dict(word=word.upper(), fp=whole, hecb=hecb_vec, hubert=hubert_vec, cpc=cpc_vec)
                )
        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(files)} files → {len(recs)} words", flush=True)

    words = [r["word"] for r in recs]
    freq = {w for w, c in Counter(words).items() if c >= args.min_freq}
    keep = [i for i, r in enumerate(recs) if r["word"] in freq]
    random.Random(args.seed).shuffle(keep)
    keep = sorted(keep[: args.max_words])
    gt = [recs[i]["word"] for i in keep]
    k = min(len(set(gt)), 60)
    print(f"  {len(recs)} words; {len(freq)} freq types; kept {len(keep)}; k={k}", flush=True)

    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    def eval_source(name, key):
        X = np.asarray([recs[i][key] for i in keep], dtype=np.float32)
        if X.shape[1] == 0 or np.any(np.isnan(X)):
            print(f"  {name:16} (bad embedding)")
            return
        Xs = StandardScaler().fit_transform(X)
        pred = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
        ev = cluster_eval(gt, pred)
        print(
            f"  {name:16} V={ev['V']:.3f}  ARI={ev['ARI']:.3f}  purity={ev['purity']:.3f}  "
            f"tokenF1={ev['token_f1']:.3f}  (n_types={ev['n_types']})"
        )

    print(
        f"\n{'=' * 80}\nSTAGE D — WORD-EMBEDDING CLUSTERING (oracle spans, k={k} clusters)\n{'=' * 80}"
    )
    eval_source("fingerprint", "fp")
    eval_source("HECB-pooled", "hecb")
    eval_source("CPC-mean", "cpc")
    eval_source("HuBERT-mean", "hubert")
    print("\n  V = V-measure, ARI = adjusted Rand, purity = mean majority-class/cluster,")
    print("  tokenF1 = many-to-one cluster→word mapping F1 (chance ≈ 1/n_types).")
    print("\nSTAGE D DONE")


if __name__ == "__main__":
    main()
