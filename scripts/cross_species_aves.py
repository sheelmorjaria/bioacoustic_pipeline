#!/usr/bin/env python3
"""Cross-species comparison on a SHARED AVES tokenization (3rd representation).

Same idea as cross_species_fingerprint.py but the shared representation is AVES
(esp_aves2_eat_bio, multi-taxa self-supervised, loaded via avex): one learned model,
one 768-D embedding space, one shared k-means codebook for human and bat. Firewall-legal
(`aves` embedding allowed for all species); the tokenizer is our k-means, not the model.

CAVEAT (must travel with any bat-side AVES result): AVES is a 16 kHz model. Bat audio is
250 kHz; each bat segment is anti-alias resampled 250k -> 16k before AVES, so only the
<8 kHz portion of the (often ultrasonic) social calls reaches AVES. Human is native 16 kHz
(clean). So AVES-on-bat is a bandwidth-limited representation, not a clean bat view.

Output: outputs/cross_species_aves.json

Usage:
    python scripts/cross_species_aves.py --n-human 250 --n-bat 2000 --k 128

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIBRISPEECH = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
BAT_AUDIO_DIR = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
ANNOT_CSV = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")
OUT = ROOT / "outputs" / "cross_species_aves.json"
CALL_GRANULAR = {"sample_rate": 250000, "hop_ms": 10.0,
                 "energy_threshold": 0.05, "spectral_threshold": 0.05, "debounce_ms": 25.0}
AVES_SR = 16000


def _speaker_of(p: Path) -> str:
    return p.stem.split("-")[0]


def _load_bat_annotations() -> dict[str, dict]:
    out = {}
    with ANNOT_CSV.open() as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("File Name") or "").strip()
            if fn:
                out[fn] = {"context": int(row["Context"]), "emitter": row["Emitter"].strip()}
    return out


def _embed(aves_model, audio16k: np.ndarray) -> np.ndarray:
    """AVES forward -> mean-pool over frames -> 768-D numpy.

    Pads to >= 800 samples (50 ms) so the EAT fbank (400-sample / 25 ms window) has
    at least one full window; very short NBD segments would otherwise fail its assert.
    """
    import torch  # noqa: PLC0415
    a = np.asarray(audio16k, dtype=np.float32)
    if len(a) < 800:
        a = np.pad(a, (0, 800 - len(a)))
    x = torch.from_numpy(a).unsqueeze(0)
    with torch.no_grad():
        h = aves_model.forward(x)
    return h.mean(dim=1).squeeze(0).numpy().astype(np.float32)


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly  # noqa: PLC0415
    if sr == AVES_SR:
        return audio
    g = np.gcd(int(sr), AVES_SR)
    return resample_poly(audio, AVES_SR // g, int(sr) // g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-human", type=int, default=250)
    ap.add_argument("--n-bat", type=int, default=2000)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n-perm", type=int, default=150)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
        controlled_alias_tests, detect_functional_aliases,
    )
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
    from bioacoustic_pipeline.orchestration import _default_calibrator, segment_audio  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    fw = default_firewall()
    for sp in ("human", "bat"):
        fw.assert_allowed("aves", Use.EMBEDDING)
        fw.assert_cross_species("aves", sp)
    print("firewall: aves embedding legal for human + bat")

    import avex  # noqa: PLC0415
    print("loading AVES (esp_aves2_eat_bio)...")
    aves = avex.load_model("esp_aves2_eat_bio")
    aves.eval()

    profiles = load_species_profiles()
    hprof = profiles["human"]
    cal = _default_calibrator("human")
    hcpc = CPCBackbone() if (cal and getattr(cal, "_use_context", False)) else None
    bprof = {**profiles["bat"], **CALL_GRANULAR}
    annot = _load_bat_annotations()

    # gather human (native 16 kHz -> AVES)
    print(f"gathering human ({args.n_human} LibriSpeech)...")
    h_seqs, h_spk = [], []
    for i, f in enumerate(sorted(LIBRISPEECH.rglob("*.flac"))[: args.n_human]):
        try:
            a, sr = load_audio(str(f), target_sr=AVES_SR)
        except Exception:  # noqa: BLE001
            continue
        seq = [_embed(aves, seg.audio) for seg in segment_audio(a, sr, hprof, calibrator=cal, cpc=hcpc)]
        seq = [e for e in seq if np.any(e)]
        if seq:
            h_seqs.append(seq)
            h_spk.append(_speaker_of(f))
        if (i + 1) % 50 == 0:
            print(f"  human {i+1}/{args.n_human}, {sum(len(s) for s in h_seqs)} segs")

    # gather bat (segment at 250 kHz, resample each segment -> 16 kHz -> AVES)
    print(f"gathering bat ({args.n_bat} bouts; segments resampled 250k->16k, <8kHz only)...")
    b_seqs, b_emit, b_ctx = [], [], []
    for i, f in enumerate([x for x in sorted(BAT_AUDIO_DIR.rglob("*.wav")) if x.name in annot][: args.n_bat]):
        try:
            a, sr = load_audio(str(f), target_sr=bprof["sample_rate"])
        except Exception:  # noqa: BLE001
            continue
        segs = segment_audio(a, sr, bprof)
        seq = [_embed(aves, _resample_to_16k(seg.audio, seg.sr)) for seg in segs]
        seq = [e for e in seq if np.any(e)]
        if seq:
            b_seqs.append(seq)
            b_emit.append(annot[f.name]["emitter"])
            b_ctx.append(annot[f.name]["context"])
        if (i + 1) % 500 == 0:
            print(f"  bat {i+1}/{args.n_bat}, {sum(len(s) for s in b_seqs)} segs")

    h_emb = [e for s in h_seqs for e in s]
    b_emb = [e for s in b_seqs for e in s]
    print(f"human: {len(h_seqs)} utterances, {len(h_emb)} AVES-segments | "
          f"bat: {len(b_seqs)} bouts, {len(b_emb)} AVES-segments")

    # shared k-means codebook on pooled human+bat AVES embeddings (no Hz features to drop)
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    X = np.vstack(h_emb + b_emb)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(Xs)
    cents = km.cluster_centers_

    def _tok(v):
        xv = (scaler.transform(v[None, :])[0])[None, :]
        return int(np.argmin(((xv - cents) ** 2).sum(axis=1)))

    h_tok = [_tok(v) for v in h_emb]
    b_tok = [_tok(v) for v in b_emb]
    h_seqs_tok = []
    k = 0
    for s in h_seqs:
        h_seqs_tok.append(h_tok[k:k + len(s)])
        k += len(s)
    b_seqs_tok = []
    k = 0
    for s in b_seqs:
        b_seqs_tok.append(b_tok[k:k + len(s)])
        k += len(s)
    print(f"human codes used: {len(set(h_tok))}/{args.k} | bat codes used: {len(set(b_tok))}/{args.k}")

    h_suff = evaluate_sufficiency(h_seqs_tok)
    b_suff = evaluate_sufficiency(b_seqs_tok)
    print(f"\nhuman sufficiency: {h_suff.status} median={h_suff.median_tokens_per_bout:.1f} "
          f"bigrams={h_suff.total_bigrams} | bat: {b_suff.status} median={b_suff.median_tokens_per_bout:.1f} "
          f"bigrams={b_suff.total_bigrams}")

    def _alias(seqs, group):
        rep = detect_functional_aliases(seqs, similarity_threshold=0.80, min_instances=10)
        elig = len(rep.singleton_tokens) + rep.n_merged
        density = rep.n_merged / max(elig, 1)
        res = controlled_alias_tests(seqs, emitter=group, n_permutations=args.n_perm, seed=0)
        p = {kk: round(v.p_value, 4) for kk, v in res.items()}
        return {"density": round(density, 4), "n_groups": rep.n_groups,
                "per_null_p": p, "effective_p": max(p.values()) if p else None}

    def _downsample(seqs, groups, target):
        acc_s, acc_g, tot = [], [], 0
        for s, g in zip(seqs, groups):
            acc_s.append(s)
            acc_g.append(g)
            tot += len(s)
            if tot >= target:
                break
        return acc_s, acc_g

    matched = min(sum(len(s) for s in h_seqs_tok), sum(len(s) for s in b_seqs_tok))
    h_ms, h_mg = _downsample(h_seqs_tok, h_spk, matched)
    b_ms, b_mg = _downsample(b_seqs_tok, b_emit, matched)
    print(f"\n--- functional aliasing on SHARED AVES tokens (matched ~{matched} tokens) ---")
    h_alias = _alias(h_ms, h_mg) if evaluate_sufficiency(h_ms).evaluable else None
    b_alias = _alias(b_ms, b_mg) if evaluate_sufficiency(b_ms).evaluable else None
    for name, rep in (("human", h_alias), ("bat", b_alias)):
        if rep:
            print(f"  {name:6s} density={rep['density']} groups={rep['n_groups']} "
                  f"eff_p={rep['effective_p']} nulls={rep['per_null_p']}")
        else:
            print(f"  {name:6s} NOT_EVALUABLE at matched size")

    # bat context_binding on shared AVES tokens (within-Emitter)
    bf_t, bf_c, bf_e = [], [], []
    for s, e, c in zip(b_seqs_tok, b_emit, b_ctx):
        for t in s:
            bf_t.append(t)
            bf_c.append(c)
            bf_e.append(e)
    b_ctx_report = None
    if len(bf_t) > 50:
        r = association_test([[t] for t in bf_t], bf_c, statistic="nmi",
                             n_permutations=args.n_perm, seed=0,
                             group=bf_e, null_type="within_group").as_dict()
        b_ctx_report = {kk: r[kk] for kk in ("observed_nmi", "observed_chi2", "cramers_v", "p_value", "null_type")}
        print(f"\nbat context_binding (shared AVES tokens): nmi={r['observed_nmi']:.3f} "
              f"V={r['cramers_v']:.3f} p={r['p_value']} [{r['null_type']}]")

    report = {
        "representation": "shared AVES (esp_aves2_eat_bio) 768-D embeddings, k-means codebook, multi-taxa SSL, firewall all-species",
        "k": args.k,
        "bat_caveat": "AVES is 16kHz; bat segments anti-alias resampled 250k->16k => only <8kHz of social calls represented",
        "human": {"n_utterances": len(h_seqs), "n_tokens": len(h_tok), "sufficiency": h_suff.as_dict(),
                  "alias_matched": h_alias},
        "bat": {"n_bouts": len(b_seqs), "n_tokens": len(b_tok), "sufficiency": b_suff.as_dict(),
                "alias_matched": b_alias, "context_binding": b_ctx_report},
        "matched_token_count": matched,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
