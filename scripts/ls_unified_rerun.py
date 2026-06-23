#!/usr/bin/env python3
"""LibriSpeech UNIFIED rerun: one extraction pass, same words for §14 + Stage C + Stage D.

The three LibriSpeech word-structure scripts each independently call ``sample_files`` +
re-run wav2vec2 forced alignment + HuBERT/CPC forward passes, and each builds its own
freq-word ``keep`` subset with a different cap (§14 max_words=1200, Stage C=600, Stage
D=800). So their reported numbers are NOT on the same words — the cross-script comparison
in the paper is apples-to-apples only approximately.

This script does the extraction ONCE and runs all three metric sets on the EXACT same
subset:

  §14 (unsupervised_word_structure_benchmark) — per-source k-means cluster sequences →
      within/across normalized edit distance + ratio, NN accuracy (+ cross-speaker),
      motif(bigram)→word Cramér's V.
  Stage C (dusted_librispeech_words) — the SAME cluster sequences → DUSTED recurrent-term
      discovery (exact n-grams / fixed-cosine / DTW-with-indels) + word enrichment +
      order-shuffle null + bootstrap Jaccard.
  Stage D (eskmeans_librispeech) — per-word POOLED embeddings (whole-word fingerprint,
      HECB-pooled, HuBERT-mean, CPC-mean) → KMeans → V-measure / ARI / purity / token-F1.

Efficiency wins from unification: one forced-align + one HuBERT + one CPC pass (was 3x),
and one k-means per source shared by §14 and Stage C (was 2x). The expensive extraction is
also SAVED to a manifest (outputs/ls_unified/manifest.json + recs.pkl) so the three metric
sets can be re-scored (different k / thresholds / DTW params) without re-extraction.

Human-speech word GT → "word" honest here. wav2vec2 is the GT yardstick (human-only,
measurement instrument, not firewall-registered); never used as pipeline features.

Run:  python scripts/ls_unified_rerun.py --n-files 200 --max-words 600
      python scripts/ls_unified_rerun.py --load outputs/ls_unified   # re-score, no extract

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import pickle
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

K_SSL = 50  # HuBERT/CPC k-means (zero-resource standard)
K_FP = 32  # fingerprint/HECB fingerprint-cluster vocabulary
OUT = ROOT / "outputs" / "ls_unified"


# ── shared metric helpers (identical to the three standalone scripts) ──────────


def edit_dist_norm(a, b):
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


def cluster_eval(labels_true, labels_pred):
    from sklearn.metrics import adjusted_rand_score, v_measure_score  # noqa: PLC0415

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    clusters: dict = {}
    for t, p in zip(labels_true, labels_pred):
        clusters.setdefault(p, []).append(t)
    purity = sum(max(Counter(v).values()) for v in clusters.values()) / max(n, 1)
    clst_to_word = {p: Counter(v).most_common(1)[0][0] for p, v in clusters.items()}
    tp = fp = fn = 0
    for w in set(labels_true.tolist()):
        assigned = {p for p in clusters if clst_to_word[p] == w}
        tp += sum(1 for t, p in zip(labels_true, labels_pred) if t == w and p in assigned)
        fp += sum(1 for t, p in zip(labels_true, labels_pred) if t != w and p in assigned)
        fn += sum(1 for t, p in zip(labels_true, labels_pred) if t == w and p not in assigned)
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


# ── ONE extraction pass → unified per-word records ─────────────────────────────


def extract(n_files, seed):
    """One pass: forced-align + HuBERT + CPC per file, per-word units for all 3 stages.

    Each rec carries: word, speaker, dur, nphon, file_idx, ws, we,
      hecb  : list[43-D] NBD-internal-proposal fingerprints   (§14 + Stage C sequences)
      fp    : list[43-D] 30ms-window fingerprints              (§14 + Stage C sequences)
      fp_whole : 43-D whole-word fingerprint                   (Stage D)
      hubert : (Tw, 768) word frames                           (§14 + Stage C seq; Stage D mean)
      cpc    : (Tc, ~256) word frames or None                  (§14 + Stage C seq; Stage D mean)
    """
    print("loading wav2vec2 aligner + HuBERT + CPC (ONE pass for all three stages)...", flush=True)
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415,E402
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model, labels = bundle.get_model(), bundle.get_labels()
    hubert = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()
    cpc = CPCBackbone()
    cpc_ok = cpc.available
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="human").combined  # noqa: E731

    files = L.sample_files(n_files, seed)
    recs = []
    for fi, flac in enumerate(files):
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
            hubert_all = hubert(wav)[0].squeeze(0).cpu().numpy()  # (T,768) @100fps
        cpc_all = cpc.embed_frames(audio, L.SR) if cpc_ok else None
        for ws, we, word in res.word_spans:
            if we - ws < 80:  # skip tiny words (matches the 3 standalone scripts)
                continue
            # HECB: NBD-internal-proposal fingerprints
            hecb_fps = []
            for p in L.word_internal_proposals(audio, L.SR, ws, we):
                si = max(0, int(p.time_ms / 1000 * L.SR) - int(0.02 * L.SR))
                sj = si + int(0.04 * L.SR)
                seg = audio[si:sj]
                if len(seg) >= int(0.012 * L.SR):
                    v = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
                    if np.any(v):
                        hecb_fps.append(v)
            # fingerprint: uniform 30ms-hop / 40ms windows
            hop, win = int(0.03 * L.SR), int(0.04 * L.SR)
            win_fps = []
            for s in range(int(ws / 1000 * L.SR), int(we / 1000 * L.SR) - win, hop):
                seg = audio[s : s + win]
                if len(seg) >= win:
                    v = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
                    if np.any(v):
                        win_fps.append(v)
            # Stage D whole-word fingerprint (whole segment, NOT mean of windows)
            whole = np.asarray(
                fp_fn(audio[int(ws / 1000 * L.SR) : int(we / 1000 * L.SR)], L.SR), dtype=np.float32
            )
            # SSL frames
            fs, fe = int(ws / 1000 * 100), int(we / 1000 * 100)
            hf = (
                hubert_all[fs:fe]
                if fe > fs + 1 and fe <= len(hubert_all)
                else np.zeros((0, 768), np.float32)
            )
            cf = None
            if cpc_all is not None:
                cs = min(int(ws / 1000 * 100), len(cpc_all) - 1)
                ce = max(cs + 1, min(int(we / 1000 * 100), len(cpc_all)))
                if ce > cs + 1:
                    cf = cpc_all[cs:ce]
            # inclusion gate (matches §14 / Stage C): all four sources usable
            if not (hecb_fps and win_fps and hf is not None and len(hf) >= 2):
                continue
            nphon = len([p for p in res.phoneme_starts if ws + 50 < p < we - 50])
            recs.append(
                dict(
                    word=word.upper(),
                    speaker=speaker,
                    dur=we - ws,
                    nphon=nphon,
                    file_idx=fi,
                    ws=ws,
                    we=we,
                    hecb=hecb_fps,
                    fp=win_fps,
                    fp_whole=whole,
                    hubert=hf,
                    cpc=cf,
                )
            )
        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(files)} files → {len(recs)} words", flush=True)
    print(f"  {len(recs)} complete word instances (all 4 sources)", flush=True)
    return files, recs, cpc_ok


def save_manifest(out_dir, files, recs, config):
    """manifest.json (config + file_ids + word_spans, no vectors) + recs.pkl (vectors)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_ids = [str(Path(f).name) for f in files]
    word_spans = [
        [r["file_idx"], r["ws"], r["we"], r["word"], r["speaker"], r["dur"], r["nphon"]]
        for r in recs
    ]
    manifest = {
        "config": config,
        "n_files": len(files),
        "file_ids": file_ids,
        "n_words": len(recs),
        "word_spans": word_spans,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out_dir / "recs.pkl").open("wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"  manifest → {out_dir / 'manifest.json'}  ({len(recs)} words, {len(file_ids)} files)",
        flush=True,
    )


def load_manifest(out_dir):
    with (out_dir / "recs.pkl").open("rb") as f:
        recs = pickle.load(f)  # noqa: S301
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return manifest, recs


# ── unified keep subset (ONE freq-word list for all three stages) ──────────────


def keep_subset(recs, min_freq, max_words, seed):
    words = [r["word"] for r in recs]
    freq = {w for w, c in Counter(words).items() if c >= min_freq}
    keep = [i for i, r in enumerate(recs) if r["word"] in freq]
    random.Random(seed).shuffle(keep)
    keep = sorted(keep[:max_words])
    return keep, freq


def cluster_sequences(recs, keep, source_key, k):
    """Global k-means over kept words' units → per-word collapsed cluster-id sequence.

    Shared by §14 and Stage C (built ONCE here, reused by both). Returns None if too few units.
    """
    all_u = []
    for i in keep:
        vu = recs[i][source_key]
        if vu is None:
            continue
        all_u.extend(np.asarray(v, np.float32) for v in vu)
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


# ── §14 metrics (within/across edit-dist + NN + motif→word) ────────────────────


def metrics_s14(recs, keep, seqs_by_source):
    def within_across_nn(seqs_keep):
        """seqs_keep: keep-aligned list of cluster-id sequences (one per kept word)."""
        rng = random.Random(0)
        within, across, across_dur, across_phon = [], [], [], []
        nn_correct = nn_total = nn_xspk_correct = nn_xspk_total = 0
        for a_idx, i in enumerate(keep):
            wi = seqs_keep[a_idx]
            if not wi:
                continue
            best_d, best_w, same_spk = 2.0, "?", True
            for b_idx, j in enumerate(keep):
                wj = seqs_keep[b_idx]
                if a_idx == b_idx or not wj:
                    continue
                d = edit_dist_norm(wi, wj)
                if d < best_d:
                    best_d, best_w = d, recs[j]["word"]
                    same_spk = recs[j]["speaker"] == recs[i]["speaker"]
            if best_w != "?":
                nn_total += 1
                nn_correct += best_w == recs[i]["word"]
                if not same_spk:
                    nn_xspk_total += 1
                    nn_xspk_correct += best_w == recs[i]["word"]
            for b_idx, j in enumerate(keep):
                wj = seqs_keep[b_idx]
                if b_idx <= a_idx or recs[j]["word"] != recs[i]["word"] or not wj:
                    continue
                within.append(edit_dist_norm(wi, wj))
        for _ in range(len(keep)):
            ai = rng.randrange(len(keep))
            bi = rng.randrange(len(keep))
            sa, sb = seqs_keep[ai], seqs_keep[bi]
            if ai == bi or not sa or not sb:
                continue
            d = edit_dist_norm(sa, sb)
            across.append(d)
            ra, rb = recs[keep[ai]], recs[keep[bi]]
            if abs(ra["dur"] - rb["dur"]) < 0.25 * max(ra["dur"], rb["dur"]):
                across_dur.append(d)
            if abs(ra["nphon"] - rb["nphon"]) <= 1:
                across_phon.append(d)
        return (
            float(np.mean(within)) if within else 0.0,
            float(np.mean(across)) if across else 0.0,
            float(np.mean(across_dur)) if across_dur else 0.0,
            float(np.mean(across_phon)) if across_phon else 0.0,
            nn_correct / max(nn_total, 1),
            nn_xspk_correct / max(nn_xspk_total, 1),
        )

    def motif_enrich(seqs_keep):
        from scipy.stats import chi2_contingency  # noqa: PLC0415

        bg = Counter()
        for s in seqs_keep:
            if not s:
                continue
            for k in range(len(s) - 1):
                bg[(s[k], s[k + 1])] += 1
        top = [m for m, _ in bg.most_common(20)]
        words = [recs[i]["word"] for i in keep]
        uw = sorted(set(words))
        if len(uw) < 2:
            return 0.0
        vs = []
        for motif in top:
            has = [
                any(s[k] == motif[0] and s[k + 1] == motif[1] for k in range(len(s) - 1))
                if s
                else False
                for s in seqs_keep
            ]
            table = np.zeros((2, len(uw)))
            for h, w in zip(has, words):
                table[int(h), uw.index(w)] += 1
            if table.sum() < 10 or table.sum(axis=0).min() == 0:
                continue
            try:
                chi2, _, _, _ = chi2_contingency(table)
                vs.append(min(float(np.sqrt(chi2 / (table.sum() * (min(2, len(uw)) - 1)))), 1.0))
            except Exception:  # noqa: BLE001
                pass
        return float(np.mean(vs)) if vs else 0.0

    rows = []
    for name, seqs_keep in seqs_by_source.items():
        if seqs_keep is None:
            rows.append((name, None))
            continue
        w, a, ad, ap, nn, nnx = within_across_nn(seqs_keep)
        rows.append(
            (
                name,
                dict(
                    within=w,
                    across=a,
                    a_dur=ad,
                    a_phn=ap,
                    ratio=w / max(a, 1e-6),
                    nn=nn,
                    nn_xsp=nnx,
                    motifV=motif_enrich(seqs_keep),
                ),
            )
        )
    return rows


# ── Stage C metrics (DUSTED recurrent-term discovery + null + bootstrap) ───────


def metrics_stage_c(recs, keep, seqs_by_source, args, k_basis):
    common = dict(min_n=2, max_n=4, min_support=3, min_calls=2)
    out = {}
    for name, seqs in seqs_by_source.items():
        if seqs is None:
            out[name] = None
            continue
        calls = []
        for j, ki in enumerate(keep):
            if not seqs[j]:
                continue
            calls.append(
                sm.CallUnits(
                    fingerprints=[],
                    clusters=seqs[j],
                    calltype=recs[ki]["word"],
                    caller=recs[ki]["speaker"],
                    file_id=str(ki),
                    call_idx=j,
                )
            )
        if len(calls) < 20:
            out[name] = None
            continue
        # discrete-unit metric: one-hot cluster id (cosine = 1 iff same id)
        for c in calls:
            c.fingerprints = [
                np.eye(k_basis, dtype=np.float32)[i]
                if 0 <= i < k_basis
                else np.zeros(k_basis, dtype=np.float32)
                for i in c.clusters
            ]
        exact = sm.summarize_discovery(sm.exact_ngram_clusters(calls, **common), len(calls))
        fixed = sm.summarize_discovery(
            sm.fixed_length_cosine_clusters(
                calls,
                **common,
                dist_threshold=args.dist_threshold,
                centroid_radius=args.centroid_radius,
                normalize=False,
            ),
            len(calls),
        )
        dtw = sm.summarize_discovery(
            sm.discover_recurrent_subsequences(
                calls,
                **common,
                dist_threshold=args.dist_threshold,
                centroid_radius=args.centroid_radius,
                normalize=False,
            ),
            len(calls),
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
        out[name] = dict(exact=exact, fixed=fixed, dtw=dtw, null=null, jac=jac, n_calls=len(calls))
    return out


# ── Stage D metrics (whole-word embedding clustering) ──────────────────────────


def metrics_stage_d(recs, keep):
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    gt = [recs[i]["word"] for i in keep]
    k = min(len(set(gt)), 60)

    def vec_fp(i):
        return recs[i]["fp_whole"]

    def vec_hecb(i):
        h = recs[i]["hecb"]
        return np.mean(h, axis=0) if h else np.zeros_like(recs[i]["fp_whole"])

    def vec_hubert(i):
        return recs[i]["hubert"].mean(axis=0)

    def vec_cpc(i):
        c = recs[i]["cpc"]
        return c.mean(axis=0) if c is not None and len(c) else None

    rows = []
    for name, fn in [
        ("fingerprint", vec_fp),
        ("HECB-pooled", vec_hecb),
        ("HuBERT-mean", vec_hubert),
        ("CPC-mean", vec_cpc),
    ]:
        Xlist = []
        order = []
        for i in keep:
            v = fn(i)
            if v is None:
                continue
            Xlist.append(np.asarray(v, dtype=np.float32))
            order.append(i)
        if len(Xlist) < k:
            rows.append((name, None))
            continue
        X = np.asarray(Xlist, dtype=np.float32)
        if X.shape[1] == 0 or np.any(np.isnan(X)):
            rows.append((name, None))
            continue
        sub_gt = [recs[i]["word"] for i in order]
        Xs = StandardScaler().fit_transform(X)
        pred = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
        rows.append((name, cluster_eval(sub_gt, pred)))
    return rows, k


# ── report ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="LibriSpeech UNIFIED rerun (one pass, same words)")
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--k-ssl", type=int, default=K_SSL)
    ap.add_argument("--k-fp", type=int, default=K_FP)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument(
        "--max-words",
        type=int,
        default=600,
        help="ONE cap shared by all three stages → identical subset",
    )
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--load", type=str, default="", help="re-score an existing outputs/ls_unified dir"
    )
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.load:
        out_dir = Path(args.load)
        print(f"re-scoring from {out_dir} (no extraction)", flush=True)
        manifest, recs = load_manifest(out_dir)
        files = ["<loaded>"] * manifest["n_files"]
        cpc_ok = any(r["cpc"] is not None for r in recs)
    else:
        files, recs, cpc_ok = extract(args.n_files, args.seed)
        config = {
            "n_files": args.n_files,
            "seed": args.seed,
            "sr": L.SR,
            "k_ssl": args.k_ssl,
            "k_fp": args.k_fp,
            "min_freq": args.min_freq,
            "max_words": args.max_words,
        }
        save_manifest(out_dir, files, recs, config)

    keep, freq = keep_subset(recs, args.min_freq, args.max_words, args.seed)
    print(
        f"  kept {len(keep)} word instances across {len(freq)} freq types (cap {args.max_words})",
        flush=True,
    )
    if len(keep) < 20:
        print("ERROR: too few kept words; raise --n-files / lower --min-freq")
        sys.exit(1)

    # cluster sequences ONCE per source, shared by §14 and Stage C
    seqs_by_source = {
        f"HECB(k{args.k_fp})": cluster_sequences(recs, keep, "hecb", args.k_fp),
        f"HuBERT-km(k{args.k_ssl})": cluster_sequences(recs, keep, "hubert", args.k_ssl),
        f"fingerprint(k{args.k_fp})": cluster_sequences(recs, keep, "fp", args.k_fp),
    }
    if cpc_ok:
        seqs_by_source[f"CPC-km(k{args.k_ssl})"] = cluster_sequences(recs, keep, "cpc", args.k_ssl)
    k_basis = max(args.k_fp, args.k_ssl)

    print(f"\n{'=' * 92}")
    print(
        f"LIBRISPEECH UNIFIED RERUN — n={len(keep)} words × {len(freq)} types "
        f"(SAME subset for §14 + Stage C + Stage D)"
    )
    print(f"{'=' * 92}")

    # ── §14 ──
    print(f"\n§14  within/across edit-dist + NN + motif→word (n={len(keep)})")
    print(
        f"  {'source':20} {'within':>7} {'across':>7} {'a_dur':>6} {'a_phn':>6} "
        f"{'ratio':>6} {'NN':>6} {'NN_xsp':>7} {'motifV':>7}"
    )
    print("  " + "-" * 86)
    for name, r in metrics_s14(recs, keep, seqs_by_source):
        if r is None:
            print(f"  {name:20} (insufficient data)")
            continue
        print(
            f"  {name:20} {r['within']:7.3f} {r['across']:7.3f} {r['a_dur']:6.3f} "
            f"{r['a_phn']:6.3f} {r['ratio']:6.3f} {r['nn']:6.3f} {r['nn_xsp']:7.3f} "
            f"{r['motifV']:7.3f}"
        )

    # ── Stage C ──
    print(f"\nStage C  DUSTED recurrent-term discovery (same n={len(keep)} words)")
    sc = metrics_stage_c(recs, keep, seqs_by_source, args, k_basis)
    for name, r in sc.items():
        if r is None:
            print(f"\n  [{name}] insufficient units/calls")
            continue
        print(f"\n  [{name}] {r['n_calls']} word instances")
        for mname, key in [("exact", "exact"), ("fixed-cos", "fixed"), ("DTW", "dtw")]:
            s = r[key]
            print(
                f"    {mname:10} fams={s['n_motifs']:4d} cov={s['coverage']:.3f} "
                f"word_V={s['mean_calltype_V']:.3f} H={s['family_entropy_norm']:.2f} "
                f"top={s['top_family_support_fraction']:.2f} purity(w)={s['wmean_label_purity']:.2f}"
            )
        n = r["null"]
        print(
            f"    DTW null: obs={n['observed_support']} null={n['null_mean']:.0f}±"
            f"{n['null_std']:.0f} ({n['observed_support'] / max(n['null_mean'], 1):.2f}x) "
            f"p={n['p']:.3f}  |  bootstrap Jaccard={r['jac']:.3f}"
        )

    # ── Stage D ──
    print(f"\nStage D  whole-word embedding clustering (same n={len(keep)} words)")
    rows_d, kd = metrics_stage_d(recs, keep)
    print(f"  {'source':16} {'V':>6} {'ARI':>6} {'purity':>7} {'tokenF1':>8}   (k={kd})")
    print("  " + "-" * 50)
    for name, ev in rows_d:
        if ev is None:
            print(f"  {name:16} (bad/insufficient embedding)")
            continue
        print(
            f"  {name:16} {ev['V']:6.3f} {ev['ARI']:6.3f} {ev['purity']:7.3f} {ev['token_f1']:8.3f}"
        )

    print("\n  All three metric sets above run on the IDENTICAL kept subset.")
    print(
        "  Re-score without re-extraction:  python scripts/ls_unified_rerun.py "
        f"--load {out_dir} --max-words {args.max_words}"
    )
    print("LS UNIFIED RERUN DONE")


if __name__ == "__main__":
    main()
