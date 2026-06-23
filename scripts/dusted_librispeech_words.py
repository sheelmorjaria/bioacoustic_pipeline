#!/usr/bin/env python3
"""Stage C — DUSTED word-term discovery on LibriSpeech across unit sources.

Stage B (§14) found HuBERT-km(50) FAILED same-word consistency under normalized
Levenshtein (ratio 1.008) while CPC / fingerprint / HECB succeeded. The zero-resource
literature uses HuBERT units with **DTW** (alignment), not collapse+Levenshtein. So this
stage runs the DUSTED module (DTW-with-indels + fixed-cosine + exact-n-gram, the user's
chosen cascade) over each unit source's per-word sequences, with `calltype=word`, and
scores discovered recurrent-term families by **word enrichment** (calltype Cramér's V).

The decisive question: does HuBERT recover word-associated recurrent terms under its
native DTW metric (closing the §14 "HuBERT fails" loop — was it the metric, not the unit)?

Each word instance = one CallUnits (ordered unit cluster sequence); calltype=word,
caller=speaker. 4 sources (HECB / HuBERT-km(50) / CPC-km(50) / fingerprint(32)).

This is a HUMAN-SPEECH benchmark (word GT) — "word" honest here. Firewall/ceiling
unchanged. Run:  python scripts/dusted_librispeech_words.py --n-files 200

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
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402

import hecb_librispeech as L  # noqa: E402

K_SSL = 50
K_FP = 32


def collapse_repeats(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def kmeans_units(frames, k, seed=0):
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(frames)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096).fit(
        scaler.transform(frames)
    )
    sv = scaler.transform(frames)
    return np.argmin(((sv[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)


def main():
    ap = argparse.ArgumentParser(description="Stage C: DUSTED word-term discovery (LibriSpeech)")
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--k-ssl", type=int, default=K_SSL)
    ap.add_argument("--k-fp", type=int, default=K_FP)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=600)
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=6)
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
    recs = []
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
        speaker = Path(flac).parent.parent.name
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            hubert_all = hubert(wav)[0].squeeze(0).cpu().numpy()
        cpc_all = cpc.embed_frames(audio, L.SR) if cpc_ok else None
        for ws, we, word in res.word_spans:
            if we - ws < 80:
                continue
            fs, fe = int(ws / 1000 * 100), int(we / 1000 * 100)
            hecb_fps, win_fps = [], []
            for p in L.word_internal_proposals(audio, L.SR, ws, we):
                si = max(0, int(p.time_ms / 1000 * L.SR) - int(0.02 * L.SR))
                sj = si + int(0.04 * L.SR)
                seg = audio[si:sj]
                if len(seg) >= int(0.012 * L.SR):
                    v = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
                    if np.any(v):
                        hecb_fps.append(v)
            hop, win = int(0.03 * L.SR), int(0.04 * L.SR)
            for s in range(int(ws / 1000 * L.SR), int(we / 1000 * L.SR) - win, hop):
                seg = audio[s : s + win]
                if len(seg) >= win:
                    v = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
                    if np.any(v):
                        win_fps.append(v)
            hf = hubert_all[fs:fe] if fe > fs + 1 and fe <= len(hubert_all) else None
            cf = None
            if cpc_all is not None:
                cs = min(int(ws / 1000 * 100), len(cpc_all) - 1)
                ce = max(cs + 1, min(int(we / 1000 * 100), len(cpc_all)))
                cf = cpc_all[cs:ce] if ce > cs + 1 else None
            if hecb_fps and win_fps and hf is not None and len(hf) >= 2:
                recs.append(
                    dict(
                        word=word.upper(),
                        speaker=speaker,
                        hecb=hecb_fps,
                        fp=win_fps,
                        hubert=hf,
                        cpc=cf,
                    )
                )
        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(files)} files → {len(recs)} words", flush=True)

    words = [r["word"] for r in recs]
    freq = {w for w, c in Counter(words).items() if c >= args.min_freq}
    keep = [i for i, r in enumerate(recs) if r["word"] in freq]
    random.Random(args.seed).shuffle(keep)
    keep = sorted(keep[: args.max_words])
    print(
        f"  {len(recs)} words; {len(freq)} freq types; kept {len(keep)} (cap {args.max_words})",
        flush=True,
    )

    # per-source unit cluster sequences over kept words
    def seqs_for(source_key, k):
        all_u = [v for i in keep for v in recs[i][source_key] if recs[i][source_key] is not None]
        if len(all_u) < k * 5:
            return None
        u = kmeans_units(np.asarray(all_u, dtype=np.float32), k)
        out, idx = [], 0
        for i in keep:
            vu = recs[i][source_key]
            if vu is None:
                out.append([])
                continue
            n = len(vu)
            out.append(collapse_repeats(u[idx : idx + n].tolist()))
            idx += n
        return out

    hecb_seq = seqs_for("hecb", args.k_fp)
    fp_seq = seqs_for("fp", args.k_fp)
    hubert_seq = seqs_for("hubert", args.k_ssl)
    cpc_seq = seqs_for("cpc", args.k_ssl) if cpc_ok else None

    def build_calls(seqs):
        calls = []
        for j, i in enumerate(keep):
            if not seqs or not seqs[j]:
                continue
            calls.append(
                sm.CallUnits(
                    fingerprints=[],
                    clusters=seqs[j],
                    calltype=recs[i]["word"],
                    caller=recs[i]["speaker"],
                    file_id=str(i),
                    call_idx=j,
                )
            )
        return calls

    common = dict(min_n=2, max_n=4, min_support=3, min_calls=2)

    def run_source(name, seqs):
        if seqs is None:
            print(f"  {name}: insufficient units")
            return
        calls = build_calls(seqs)
        if len(calls) < 20:
            print(f"  {name}: too few calls ({len(calls)})")
            return
        # discrete-unit DTW/cosine: unit "fingerprint" = one-hot cluster id (normalize=False
        # so cosine = 1 iff same id, 0 else → clean hard substitution metric; no z-norm).
        for c in calls:
            c.fingerprints = [
                np.eye(k_basis, dtype=np.float32)[i]
                if 0 <= i < k_basis
                else np.zeros(k_basis, dtype=np.float32)
                for i in c.clusters
            ]
        print(f"\n[{name}] {len(calls)} word instances")
        exact = sm.exact_ngram_clusters(calls, **common)
        fixed = sm.fixed_length_cosine_clusters(
            calls,
            **common,
            dist_threshold=args.dist_threshold,
            centroid_radius=args.centroid_radius,
            normalize=False,
        )
        dtw = sm.discover_recurrent_subsequences(
            calls,
            **common,
            dist_threshold=args.dist_threshold,
            centroid_radius=args.centroid_radius,
            normalize=False,
        )
        for mname, mm in [("exact", exact), ("fixed-cos", fixed), ("DTW", dtw)]:
            s = sm.summarize_discovery(mm, len(calls))
            print(
                f"  {mname:10} fams={s['n_motifs']:4d} cov={s['coverage']:.3f} "
                f"word_V={s['mean_calltype_V']:.3f} H={s['family_entropy_norm']:.2f} "
                f"top={s['top_family_support_fraction']:.2f} purity(w)={s['wmean_label_purity']:.2f}"
            )
        null = sm.discovery_null(
            calls[: args.n_null],
            dist_threshold=args.dist_threshold,
            centroid_radius=args.centroid_radius,
            min_calls=2,
            n_perm=args.n_perm,
            seed=args.seed,
            min_n=2,
            max_n=4,
            normalize=False,
        )
        jac = sm.bootstrap_stability(
            calls[: args.n_null],
            dist_threshold=args.dist_threshold,
            centroid_radius=args.centroid_radius,
            min_calls=2,
            n_boot=args.n_boot,
            top_k=10,
            seed=args.seed,
            min_n=2,
            max_n=4,
            normalize=False,
        )
        print(
            f"  DTW null: obs={null['observed_support']} null={null['null_mean']:.0f}"
            f"±{null['null_std']:.0f} ({null['observed_support'] / max(null['null_mean'], 1):.2f}x) "
            f"p={null['p']:.3f}  |  bootstrap Jaccard={jac:.3f}"
        )

    # k_basis must cover both k_fp and k_ssl → use the max
    k_basis = max(args.k_fp, args.k_ssl)
    print(
        f"\n{'=' * 90}\nSTAGE C — DUSTED WORD-TERM DISCOVERY (LibriSpeech, n={len(keep)} words)\n"
        f"{'=' * 90}\n(word_V = mean Cramér's V of discovered term vs word identity — HIGHER = "
        "more word-associated recurrent terms)"
    )
    run_source(f"HECB(k{args.k_fp})", hecb_seq)
    run_source(f"HuBERT-km(k{args.k_ssl})", hubert_seq)
    if cpc_seq is not None:
        run_source(f"CPC-km(k{args.k_ssl})", cpc_seq)
    run_source(f"fingerprint(k{args.k_fp})", fp_seq)
    print("\nSTAGE C DONE")


if __name__ == "__main__":
    main()
