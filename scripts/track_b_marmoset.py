#!/usr/bin/env python3
"""Track B — marmoset vocalization corpus (call-type ground truth).

Marmoset is the strongest-LABELED species in the project: 871k pre-extracted calls
(96 kHz, ~1 s), each with an ethological CALL TYPE (Phee / Twitter / Trill / Tsik /
Seep) and 66% timestamped within named recording sessions. This runs the falsifiable
framework on it:

  * whole-call acoustic fingerprint (all-species, firewall-legal) -> k-means tokens
    (FULL combined vector incl. spectral-Hz features: call types differ in spectrum,
    so Hz is informative WITHIN species, unlike the cross-species case)
  * HEADLINE: token<->call-type context binding (does the unsupervised token recover
    the biologist-defined call types?) + chi-square / Cramer's V / NMI / permutation p
  * semantic-ceiling analogue: supervised fingerprint -> call-type linear probe
    (does the representation carry call-type signal at all?)
  * sequence diagnostics on time-ordered within-session call streams: sufficiency gate,
    functional aliasing (within-session null), LZ, motif

No axis claims validated meaning (call types are acoustic categories, not semantics).

Output: outputs/track_b_marmoset.json
Usage:
    python scripts/track_b_marmoset.py --n-per-type 2500 --k 64

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VOC_DIR = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
ANNOT = VOC_DIR / "Annotations.tsv"
CALL_TYPES = ("Phee", "Twitter", "Trill", "Tsik", "Seep")
SR = 96000
AVES_SR = 16000  # AVES is a 16 kHz model; marmoset (96 kHz) is anti-alias resampled


def _resample_to_16k(audio, sr):
    from scipy.signal import resample_poly  # noqa: PLC0415
    if sr == AVES_SR:
        return audio
    g = np.gcd(int(sr), AVES_SR)
    return resample_poly(audio, AVES_SR // g, int(sr) // g).astype(np.float32)


def _aves_embed(aves_model, audio, sr):
    """AVES forward -> mean-pool over frames -> 768-D numpy (pads to >=800 samples)."""
    import torch  # noqa: PLC0415
    a = _resample_to_16k(np.asarray(audio, dtype=np.float32), sr)
    if len(a) < 800:
        a = np.pad(a, (0, 800 - len(a)))
    dev = next(aves_model.parameters()).device  # run on the model's device (GPU if loaded there)
    with torch.no_grad():
        h = aves_model.forward(torch.from_numpy(a).unsqueeze(0).to(dev))
    return h.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def _load_index():
    """file_name -> (label, session, t_seconds_or_None). t from h:m:s.ms if present."""
    idx = {}
    with ANNOT.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fn = (row.get("file_name") or "").strip()
            if not fn:
                continue
            h, mi, s, ms = row.get("hour"), row.get("minute"), row.get("second"), row.get("millisecond")
            t = None
            if h and mi and s:
                try:
                    t = int(h) * 3600 + int(mi) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)
                except ValueError:
                    t = None
            idx[fn] = (row.get("label", "").strip(), row.get("parent_name", "").strip(), t)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=2500)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-perm", type=int, default=150)
    ap.add_argument("--rep", choices=["fingerprint", "aves"], default="fingerprint")
    args = ap.parse_args()
    out_path = ROOT / "outputs" / f"track_b_marmoset_{args.rep}.json"

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.alias_detector import (  # noqa: PLC0415
        controlled_alias_tests, detect_functional_aliases,
    )
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics import motif_miner as mm  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.probes import linear_probe  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415
    from bioacoustic_pipeline.verdict_engine import is_valid  # noqa: PLC0415
    from bioacoustic_pipeline.verdict_engine.engine import report_from_profile  # noqa: PLC0415
    from bioacoustic_pipeline.verdict_engine.evidence_profile import (  # noqa: PLC0415
        AxisResult, compute_evidence_profile,
    )

    fw = default_firewall()
    aves = None
    if args.rep == "fingerprint":
        fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)
        fw.assert_cross_species("acoustic_fingerprint", "marmoset")
        print("firewall: acoustic_fingerprint tokenize legal for marmoset")
    else:
        fw.assert_allowed("aves", Use.EMBEDDING)
        fw.assert_cross_species("aves", "marmoset")
        import avex  # noqa: PLC0415
        import torch  # noqa: PLC0415
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"firewall: aves embedding legal for marmoset; loading AVES on {dev}...")
        aves = avex.load_model("esp_aves2_eat_bio", device=dev)
        aves.eval()
        print("  (caveat: AVES is 16 kHz; marmoset 96 kHz calls resampled -> <8 kHz)")

    idx = _load_index()
    # balanced sample by call type
    rng = np.random.RandomState(0)
    by_type = defaultdict(list)
    for fn, (lab, sess, t) in idx.items():
        if lab in CALL_TYPES:
            by_type[lab].append((fn, sess, t))
    chosen = []
    for ct in CALL_TYPES:
        pool = by_type[ct]
        rng.shuffle(pool)
        chosen.extend((ct, fn, sess, t) for fn, sess, t in pool[: args.n_per_type])
    rng.shuffle(chosen)
    print(f"sampled {len(chosen)} calls balanced over {CALL_TYPES}")

    vecs, labels, sessions, toks_meta = [], [], [], []
    for i, (ct, fn, sess, t) in enumerate(chosen):
        path = VOC_DIR / sess / fn
        if not path.exists():
            continue
        try:
            a, sr = load_audio(str(path), target_sr=SR)
        except Exception:  # noqa: BLE001
            continue
        fp = (afp.extract(a, sr, species="marmoset").combined
              if args.rep == "fingerprint" else _aves_embed(aves, a, sr))
        if not np.any(fp):
            continue
        vecs.append(fp)
        labels.append(ct)
        sessions.append(sess)
        toks_meta.append(t)
        if (i + 1) % 2000 == 0:
            print(f"  fingerprinted {i+1}/{len(chosen)}")
    vecs = np.asarray(vecs, dtype=np.float32)
    labels = np.asarray(labels)
    sessions = np.asarray(sessions)
    print(f"fingerprinted {len(vecs)} calls | feat dim {vecs.shape[1]}")

    # k-means tokens on FULL combined (Hz informative within-species)
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    scaler = StandardScaler().fit(vecs)
    km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(scaler.transform(vecs))
    cents = km.cluster_centers_
    toks = np.array([int(np.argmin(((scaler.transform(v[None, :])[0] - cents) ** 2).sum(axis=1)))
                     for v in vecs])

    # HEADLINE: token <-> call-type
    binding = association_test([[int(t)] for t in toks], labels.tolist(),
                               statistic="nmi", n_permutations=args.n_perm, seed=0).as_dict()
    # supervised ceiling: fingerprint -> call-type
    probe = linear_probe(vecs, labels, min_per_class=5).as_dict()

    # sequence diagnostics: time-ordered within-session token streams.
    # seq_sessions is aligned 1:1 with seqs (one session id per sequence) so it can
    # serve as the within-session null grouping for controlled_alias_tests.
    by_sess = defaultdict(list)
    for tk, sess, t in zip(toks, sessions, toks_meta):
        by_sess[sess].append((t if t is not None else float("inf"), int(tk)))
    seq_sessions = [sess for sess, v in by_sess.items() if len(v) >= 2]
    seqs = [[tk for _t, tk in sorted(by_sess[sess])] for sess in seq_sessions]
    suff = evaluate_sufficiency(seqs)
    print(f"\nsequences: {len(seqs)} sessions | sufficiency {suff.status} "
          f"median={suff.median_tokens_per_bout:.1f} bigrams={suff.total_bigrams}")

    alias = None
    if suff.evaluable:
        rep = detect_functional_aliases(seqs, similarity_threshold=0.80, min_instances=10)
        elig = len(rep.singleton_tokens) + rep.n_merged
        density = rep.n_merged / max(elig, 1)
        ctrl = controlled_alias_tests(seqs, emitter=seq_sessions, n_permutations=args.n_perm, seed=0)
        ctrl_p = {k: round(v.p_value, 4) for k, v in ctrl.items()}
        alias = {"density": round(density, 4), "n_groups": rep.n_groups,
                 "per_null_p": ctrl_p, "effective_p": max(ctrl_p.values()) if ctrl_p else None}
    lz = lempel_ziv_complexity([t for s in seqs for t in s][:5000]).normalized
    motif = None
    if suff.evaluable:
        raw = mm.mine_ngrams(seqs, min_n=2, max_n=5, min_support=3)
        motifs = mm.build_motifs(raw, seqs, motif_type="token")
        try:
            mp = round(float(mm.motif_support_null(seqs, n_permutations=args.n_perm, seed=0).p_value), 4)
        except Exception:  # noqa: BLE001
            mp = None
        motif = {"n_motifs": len(motifs), "motif_null_p": mp}

    # map context_binding result -> axis level
    sig = binding["p_value"] < 0.05
    cb_level = ("strong" if binding["observed_nmi"] >= 0.05 else "weak") if sig else "absent"
    cb = AxisResult("context_binding", cb_level, min(1.0, 1 - binding["p_value"]) if sig else 0.85,
                    f"token<->call-type nmi={binding['observed_nmi']:.3f} chi2={binding['observed_chi2']:.0f} "
                    f"V={binding['cramers_v']:.3f} p={binding['p_value']} [permutation null]")

    prof = compute_evidence_profile(
        "marmoset", lz_normalized=lz, sufficiency=suff,
        n_motifs=(motif or {}).get("n_motifs"), motif_null_p=(motif or {}).get("motif_null_p"),
        context_evaluable=True, turn_evaluable=False,
        turn_reason="no individual-ID labels; turn structure not defined",
    )
    prof.axes["context_binding"] = cb
    if suff.evaluable and alias:
        from bioacoustic_pipeline.verdict_engine.evidence_profile import score_functional_aliasing  # noqa: PLC0415
        prof.axes["functional_aliasing"] = score_functional_aliasing(alias["density"], alias["effective_p"])
    report = report_from_profile(prof)

    out = {
        "report": report,
        "trace": {
            "species": "marmoset", "representation": args.rep,
            "k": args.k, "n_calls": len(vecs),
            "call_type_counts": {ct: int((labels == ct).sum()) for ct in CALL_TYPES},
            "codes_used": int(len(set(toks.tolist()))), "n_sessions": len(seqs),
            "sufficiency": suff.as_dict(),
            "context_binding": binding,
            "call_type_probe": probe,
            "alias": alias, "motif": motif, "lz": round(lz, 4),
            "schema_valid": is_valid(report),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== MARMOSET VERDICT ===")
    for name, ax in report["axes"].items():
        print(f"  {name:24s} {ax['level']:16s} conf={ax['confidence']:.0%}")
    print(f"\ntoken<->call-type: nmi={binding['observed_nmi']:.3f} V={binding['cramers_v']:.3f} "
          f"p={binding['p_value']} | probe acc={probe.get('accuracy')} (chance {probe.get('chance_baseline')}, lift {probe.get('lift_over_chance')})")
    print(f"schema_valid={is_valid(report)} -> {out_path}")


if __name__ == "__main__":
    main()
