#!/usr/bin/env python3
"""LibriSpeech word-level motif analysis: acoustic subunit sequences vs known word identity.

Controlled analogue of the animal sub-call motif analysis. Each oracle word span → HECB
internal boundaries → fingerprint-cluster sub-unit sequence → mine motifs against KNOWN
word identity. This validates that ordered acoustic subunit motifs correspond to known
phonetic/word structure before making analogous claims about animals.

Key test: same-word consistency — do instances of the same word produce similar internal
sub-unit sequences? (within-word similarity > across-word = word signal is in the sub-units)

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
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_features  # noqa: E402
from bioacoustic_pipeline.segmentation.hierarchical import hecb_decode, HECBDurationPriors, InternalSegment  # noqa: E402

import hecb_librispeech as L  # noqa: E402

ONSET_TOL = 50.0
PHON_TOL = 40.0
_PROPOSAL = {"floor_energy": 0.04, "floor_flux": 0.02, "debounce_ms": 25.0}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def edit_dist_normalized(a, b):
    """Levenshtein on token lists, normalized by max length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0 if m == n else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[n] / max(m, n)


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="LibriSpeech word-level motif analysis")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--k-clusters", type=int, default=15)
    ap.add_argument("--min-word-freq", type=int, default=5)
    args = ap.parse_args()

    print("loading wav2vec2 aligner (CPU)...", flush=True)
    import torchaudio  # noqa: PLC0415

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()
    labels = bundle.get_labels()

    files = L.sample_files(args.n_train + args.n_test, 0)
    train_files, test_files = files[: args.n_train], files[args.n_train :]

    def align(flac):
        from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

        audio, _ = load_audio(flac, target_sr=L.SR)
        if audio.size == 0:
            return None, None
        tr = L.load_librispeech_transcript(flac)
        if not tr:
            return None, None
        res = L.forced_align(tr, audio, L.SR, model=model, labels=labels)
        if not res.word_spans:
            return None, None
        return audio, res

    # ─── Pre-align once ───
    print("aligning train+test...", flush=True)
    train_aligned, test_aligned = [], []
    for flac in train_files:
        a = align(flac)
        if a[0] is not None:
            train_aligned.append(a)
    for flac in test_files:
        a = align(flac)
        if a[0] is not None:
            test_aligned.append(a)
    print(f"  {len(train_aligned)} train / {len(test_aligned)} test", flush=True)

    # ─── Train typed classifier ───
    print("training typed classifier...", flush=True)
    X_tr, y_tr = [], []
    for audio, res in train_aligned:
        for (ws, we, _w) in res.word_spans:
            if we - ws < 80:
                continue
            in_word_phon = [p for p in res.phoneme_starts if ws + ONSET_TOL < p < we - ONSET_TOL]
            for prop in L.word_internal_proposals(audio, L.SR, ws, we):
                X_tr.append(L.ls_evidence(prop, audio, L.SR, ws, we))
                y_tr.append(L.label_proposal(prop.time_ms, ws, we, in_word_phon, ONSET_TOL, PHON_TOL))
    X_tr, y_tr = np.asarray(X_tr), np.asarray(y_tr)
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    typed_clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=0)
    typed_clf.fit(X_tr, y_tr)
    print(f"  trained on {len(X_tr)} proposals", flush=True)

    def _proba4(ev):
        pred = typed_clf.predict_proba(ev.reshape(1, -1))[0]
        p4 = np.zeros(4)
        for ci, cls in enumerate(typed_clf.classes_):
            p4[int(cls)] = pred[ci]
        return p4

    # ─── Build per-word sub-unit sequences on test ───
    print("building per-word sub-unit sequences...", flush=True)
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="human").combined  # noqa: E731
    word_records = []  # {word, sub_fps, sub_tokens (later), n_phonemes}
    for fi, (audio, res) in enumerate(test_aligned):
        for (ws, we, word) in res.word_spans:
            if we - ws < 80:
                continue
            props = L.word_internal_proposals(audio, L.SR, ws, we)
            if not props:
                continue
            evidences = np.asarray([L.ls_evidence(p, audio, L.SR, ws, we) for p in props])
            typed = [(p.time_ms, _proba4(evidences[i])) for i, p in enumerate(props)]
            decoded = hecb_decode([(ws, we, 1.0)], typed, HECBDurationPriors(),
                                  snap=False, gate_durations=False, file_id="ls")
            if not decoded:
                continue
            call = decoded[0]
            # Extract sub-call fingerprints
            sub_fps = []
            for seg in call.internal_segments:
                si = int(seg.start_s * L.SR)
                sj = max(int(seg.end_s * L.SR), si + int(0.012 * L.SR))
                sseg = audio[si:sj]
                if len(sseg) < int(0.012 * L.SR):
                    continue
                sfp = np.asarray(fp_fn(sseg, L.SR), dtype=np.float32)
                if np.any(sfp):
                    sub_fps.append(sfp)
            if not sub_fps:
                continue
            gt_phon = [p for p in res.phoneme_starts if ws + ONSET_TOL < p < we - ONSET_TOL]
            word_records.append({
                "word": word.upper(), "sub_fps": sub_fps,
                "n_phonemes": len(gt_phon), "n_subs": len(sub_fps),
            })
        if (fi + 1) % 25 == 0:
            print(f"  {fi+1}/{len(test_aligned)} files → {len(word_records)} words", flush=True)
    print(f"  {len(word_records)} words with sub-units, "
          f"mean {np.mean([w['n_subs'] for w in word_records]):.1f} sub-units/word", flush=True)

    # ─── Tokenize sub-units ───
    print(f"tokenizing sub-units (k={args.k_clusters})...", flush=True)
    all_fps = [fp for w in word_records for fp in w["sub_fps"]]
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(np.asarray(all_fps, dtype=np.float32))
    km = KMeans(n_clusters=args.k_clusters, random_state=0, n_init=10).fit(
        scaler.transform(np.asarray(all_fps, dtype=np.float32)))
    cents = km.cluster_centers_

    def _tok(fps):
        sv = scaler.transform(np.asarray(fps, dtype=np.float32))
        return np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1).tolist()

    for w in word_records:
        w["tokens"] = _tok(w["sub_fps"])

    # ─── Same-word consistency test ───
    print(f"\n{'='*70}")
    print("SAME-WORD CONSISTENCY TEST")
    print(f"{'='*70}")
    word_groups = defaultdict(list)
    for w in word_records:
        word_groups[w["word"]].append(w["tokens"])
    # Filter to words with >= min_freq instances
    freq_words = {word: seqs for word, seqs in word_groups.items() if len(seqs) >= args.min_word_freq}
    print(f"  {len(freq_words)} word types with ≥{args.min_word_freq} instances")
    print(f"  top 10: {Counter({w: len(s) for w, s in freq_words.items()}).most_common(10)}")

    # Within-word vs across-word similarity (edit distance)
    within_dists, across_dists = [], []
    rng = random.Random(0)
    all_tokens = [w["tokens"] for w in word_records]
    for word, seqs in freq_words.items():
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                within_dists.append(edit_dist_normalized(seqs[i], seqs[j]))
        # Sample across-word pairs
        for _ in range(len(seqs)):
            other = rng.choice(all_tokens)
            across_dists.append(edit_dist_normalized(seqs[0], other))

    within_mean = np.mean(within_dists) if within_dists else 0
    across_mean = np.mean(across_dists) if across_dists else 0
    print(f"\n  within-word edit distance:  {within_mean:.3f} (n={len(within_dists)} pairs)")
    print(f"  across-word edit distance:  {across_mean:.3f} (n={len(across_dists)} pairs)")
    print(f"  ratio (within/across):     {within_mean/max(across_mean, 1e-6):.3f}")
    print(f"  → {'SAME-WORD IS MORE SIMILAR' if within_mean < across_mean else 'NO CONSISTENCY'} "
          f"({'word signal IS in sub-units' if within_mean < across_mean else 'no word signal'})")

    # ─── Motif enrichment (word identity) ───
    print(f"\n{'='*70}")
    print("WORD-IDENTITY MOTIF ENRICHMENT")
    print(f"{'='*70}")
    from scipy.stats import chi2_contingency  # noqa: PLC0415

    for n in [1, 2]:
        counts = Counter()
        for w in word_records:
            toks = w["tokens"]
            for i in range(len(toks) - n + 1):
                counts[tuple(toks[i:i+n])] += 1
        print(f"\n  {n}-grams: {len(counts)} distinct, top 5:")
        for motif, cnt in counts.most_common(5):
            # Enrichment: does this motif preferentially appear in certain words?
            has = [any(tuple(motif) == tuple(w["tokens"][i:i+n])
                   for i in range(len(w["tokens"]) - n + 1))
                   for w in word_records]
            words_with = [w["word"] for h, w in zip(has, word_records) if h]
            top_word = Counter(words_with).most_common(1)[0] if words_with else ("?", 0)
            print(f"    {str(motif):16} count={cnt:4d}  top_word={top_word[0]:12} ({top_word[1]})")

    print("\nLS WORD MOTIFS DONE")


if __name__ == "__main__":
    main()
