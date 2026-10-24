#!/usr/bin/env python3
"""§45 eval: caller-ID diagnostics for every arm vs the fingerprint baseline.

Arms: fingerprint (44-D) | bat_cpc_v2 (self-supervised) | bat_cpc_caller
(caller-conditioned, restricted-use — NEVER stop-rule/verdict evidence) |
fingerprint+bat_cpc_v2 (concat). Metrics per arm, all as MEASUREMENTS with
chance baselines, file-level permutation nulls and cluster-bootstrap CIs:
linear-probe macro-F1, EER (LFW-style leakage-safe pairs), few-shot kNN +
prototype on held-out callers, and noise robustness (ΔEER at 10 dB SNR).

Stop rule (cpc-policy.md, pre-registered in the JSON): the animal-CPC line
continues only if bat_cpc_v2 beats the fingerprint with CI separation
(EER cpc_ci_hi < fp_ci_lo, or macroF1 cpc_ci_lo > fp_ci_hi). Arm B is
excluded from adjudication (circularity).

Usage:
    python scripts/bat_caller_cpc_eval.py --data-dir data/egyptian_fruit_bats
    python scripts/bat_caller_cpc_eval.py --self-test

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

FRAMING = (
    "diagnostic caller-identity measurements with nulls/CIs; caller identity "
    "is acoustic metadata, NOT verdict evidence, NOT meaning."
)


def _load_eval_set(args, *, selftest: bool):
    """Returns (embeddings_by_arm, caller_ids, file_ids, heldout_callers)."""
    import numpy as np

    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        parse_annotations_csv,
    )

    td_ctx = tempfile.TemporaryDirectory() if not selftest else None
    if selftest:
        # synthetic corpus with a planted caller signal; "arms" are analytic
        # embeddings (fast, no training): cpc = clean caller signal, fp =
        # noisier projection, caller = strongest (supervised proxy). Noise
        # levels are tuned so the arms are SEPARABLE but not trivially zero.
        rng = np.random.RandomState(0)
        n_callers, files_per = 8, 40
        centers = rng.randn(n_callers, 24) * 2.0
        rows, callers, files = [], [], []
        for c in range(n_callers):
            for f in range(files_per):
                rows.append(centers[c] + 0.5 * rng.randn(24))
                callers.append(c)
                files.append(f"c{c:02d}_f{f:02d}.wav")
        emb = np.asarray(rows)
        callers_a = np.asarray(callers)
        files_a = np.asarray(files)
        arms = {
            "fingerprint": emb + 2.0 * np.random.RandomState(1).randn(*emb.shape),
            "bat_cpc_v2": emb + 0.9 * np.random.RandomState(2).randn(*emb.shape),
            "bat_cpc_caller": emb + 0.4 * np.random.RandomState(3).randn(*emb.shape),
        }
        try:
            from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation

            split = holdout_callers(
                [CallAnnotation(fn, int(c), 0, 0, 0, 0, 0, 0) for fn, c in zip(files_a, callers_a)],
                n_heldout=2,
                min_files=8,
                seed=0,
            )
            heldout = split.heldout_callers
            test_mask = np.isin(files_a, sorted(split.test_files))
            heldout_mask = np.isin(callers_a, heldout)
        except ValueError:
            heldout = []
            test_mask = np.ones(len(files_a), dtype=bool)
            heldout_mask = np.zeros(len(files_a), dtype=bool)
        return arms, callers_a, files_a, test_mask, heldout_mask, heldout, td_ctx
    # ---- real mode ----
    import numpy as np

    data_dir = Path(args.data_dir)
    annotations_csv = args.annotations or (data_dir / "annotations.csv")
    audio_dir = data_dir / "audio"
    if not annotations_csv.exists() or not audio_dir.exists():
        sys.exit(f"corpus not found under {data_dir}; use --self-test offline")
    annotations = parse_annotations_csv(annotations_csv)
    n_all = len(annotations)
    annotations = [a for a in annotations if a.emitter > 0]
    print(f"identified-caller bouts: {len(annotations)}/{n_all}", flush=True)
    split = holdout_callers(
        annotations, n_heldout=args.holdout_callers, min_files=8, seed=args.seed
    )
    heldout = split.heldout_callers
    test_files = sorted(split.test_files)
    heldout_files = sorted({a.file_name for a in annotations if a.emitter in set(heldout)})
    ann_by_file = {a.file_name: a for a in annotations}
    callers_a, files_a, audio_paths = [], [], []
    from bioacoustic_pipeline.utils.audio import load_audio

    for fn in test_files + heldout_files:
        p = audio_dir / fn
        if not p.exists():
            continue
        audio, sr = load_audio(str(p), target_sr=250000)
        callers_a.append(ann_by_file[fn].emitter)
        files_a.append(fn)
        audio_paths.append((audio, sr))
    callers_a = np.asarray(callers_a)
    files_a = np.asarray(files_a)
    test_mask = np.isin(files_a, test_files)
    heldout_mask = np.isin(callers_a, heldout)

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.embeddings.embedding_store import (
        cached_embeddings,
        checkpoint_version,
    )

    item_ids = [str(f) for f in files_a]
    arms = {
        "fingerprint": cached_embeddings(
            "fingerprint",
            item_ids,
            lambda i: afp.extract(*audio_paths[i], species="bat").combined,
            version_key="afp.combined|species=bat|v1",
        )
    }

    from bioacoustic_pipeline.embeddings.species_cpc_backbone import SpeciesCPCBackbone

    v2 = SpeciesCPCBackbone(name="bat_cpc_v2", provenance_name="bat_cpc_v2")
    if v2.available:
        arms["bat_cpc_v2"] = cached_embeddings(
            "bat_cpc_v2",
            item_ids,
            lambda i: v2.embed(*audio_paths[i]),
            version_key=f"ckpt:{checkpoint_version(ROOT / 'checkpoints' / 'bat_cpc_v2')}",
        )
    else:
        print(f"bat_cpc_v2 unavailable: {v2.load_error}", flush=True)

    from bioacoustic_pipeline.embeddings.caller_conditioned import (
        CallerConditionedBackbone,
    )
    from bioacoustic_pipeline.embeddings.firewall import Firewall, Use

    Firewall().assert_allowed("bat_cpc_caller", Use.PROBE)  # probe-only use
    cc = CallerConditionedBackbone(name="bat_cpc_caller", provenance_name="bat_cpc_caller")
    if cc.available:
        arms["bat_cpc_caller"] = cached_embeddings(
            "bat_cpc_caller",
            item_ids,
            lambda i: cc.embed(*audio_paths[i]),
            version_key=f"ckpt:{checkpoint_version(ROOT / 'checkpoints' / 'bat_cpc_caller')}",
        )
    else:
        print(f"bat_cpc_caller unavailable: {cc.load_error}", flush=True)

    def _z(x, mu, sd):
        return (x - mu) / np.maximum(sd, 1e-12)

    if "fingerprint" in arms and "bat_cpc_v2" in arms:
        fp, cpc = arms["fingerprint"], arms["bat_cpc_v2"]
        arms["fingerprint+bat_cpc_v2"] = np.concatenate(
            [_z(fp, fp.mean(0), fp.std(0)), _z(cpc, cpc.mean(0), cpc.std(0))], axis=1
        )
    if "fingerprint" in arms and "bat_cpc_caller" in arms:
        # SUPERVISED-DERIVED fusion: the CPC side comes from the
        # caller-conditioned arm (audio-only UNK path, probe/retrieval use) —
        # answers whether supervised caller-CPC adds variance the expert
        # fingerprint lacks; never citable as self-supervised evidence.
        fp, cc_emb = arms["fingerprint"], arms["bat_cpc_caller"]
        arms["fingerprint+bat_cpc_caller"] = np.concatenate(
            [_z(fp, fp.mean(0), fp.std(0)), _z(cc_emb, cc_emb.mean(0), cc_emb.std(0))],
            axis=1,
        )
    return arms, callers_a, files_a, test_mask, heldout_mask, heldout, None


def _probe_macro_f1(embeddings, callers, *, seed: int = 0) -> float:
    """Grouped-by-file LogReg macro-F1 (balanced weights)."""
    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(callers, return_counts=True)
    keep = np.isin(callers, classes[counts >= 5])
    if keep.sum() < 20 or len(classes[counts >= 5]) < 2:
        return float("nan")
    X, y = np.asarray(embeddings)[keep], callers[keep]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return float(f1_score(yte, clf.predict(sc.transform(Xte)), average="macro"))


def _eer_stat(embeddings, callers, files, *, n_pairs: int = 2000, seed: int = 0) -> float:
    from bioacoustic_pipeline.diagnostics.verification import (
        build_genuine_impostor_pairs,
        calculate_eer,
    )

    sample = build_genuine_impostor_pairs(embeddings, callers, files, n_pairs=n_pairs, seed=seed)
    return calculate_eer(sample.scores[sample.labels == 1], sample.scores[sample.labels == 0]).eer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "egyptian_fruit_bats")
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--holdout-callers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        args.n_permutations = 30
        args.n_bootstrap = 30

    import numpy as np

    from bioacoustic_pipeline.diagnostics.few_shot import knn_few_shot, prototype_accuracy
    from bioacoustic_pipeline.diagnostics.verification import (
        cluster_bootstrap_files,
        label_permutation_null,
    )

    arms, callers, files, test_mask, heldout_mask, heldout, td_ctx = _load_eval_set(
        args, selftest=args.self_test
    )
    try:
        results = {}
        # EER eligibility: genuine pairs need >=2 test files per caller
        classes, counts = np.unique(callers[test_mask], return_counts=True)
        eer_ok = np.isin(callers, classes[counts >= 2]) & test_mask
        for arm_name, emb in arms.items():
            e, c, f = np.asarray(emb)[test_mask], callers[test_mask], files[test_mask]
            ee, ce, fe = np.asarray(emb)[eer_ok], callers[eer_ok], files[eer_ok]
            cell: dict = {"n_bouts": int(len(e)), "n_bouts_eer": int(len(ce))}
            probe = _probe_macro_f1(e, c, seed=args.seed)
            cell["probe_macro_f1"] = probe
            if np.isfinite(probe):
                classes = np.unique(c)
                cell["probe_chance"] = 1.0 / len(classes)
                null = label_permutation_null(
                    lambda X, y, z: _probe_macro_f1(X, y, seed=args.seed),
                    e,
                    c,
                    f,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    direction="greater",
                )
                cell["probe_perm_p"] = null.p_value
            eer = _eer_stat(ee, ce, fe, n_pairs=args.n_pairs, seed=args.seed)
            cell["eer"] = eer
            boot = cluster_bootstrap_files(
                lambda X, y, z: _eer_stat(X, y, z, n_pairs=args.n_pairs, seed=args.seed),
                ee,
                ce,
                fe,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            cell["eer_ci"] = [boot.ci_lo, boot.ci_hi]
            null_eer = label_permutation_null(
                lambda X, y, z: _eer_stat(X, y, z, n_pairs=args.n_pairs, seed=args.seed),
                ee,
                ce,
                fe,
                n_permutations=args.n_permutations,
                seed=args.seed,
                direction="lower",
            )
            cell["eer_perm_p"] = null_eer.p_value
            cell["eer_chance"] = 0.5
            if heldout:
                eh = np.asarray(emb)[heldout_mask]
                ch, fh = callers[heldout_mask], files[heldout_mask]
                if len(eh) >= 10 and len(set(ch)) >= 2:
                    knn = knn_few_shot(eh, ch, fh, n_support=3, k=1, seed=args.seed)
                    proto = prototype_accuracy(eh, ch, fh, n_support=2, seed=args.seed)
                    cell["fewshot_knn_acc"] = knn.accuracy
                    cell["fewshot_knn_chance"] = knn.chance
                    cell["fewshot_proto_acc"] = proto.accuracy
                    cell["fewshot_n_callers"] = knn.n_callers
            results[arm_name] = cell
            print(f"{arm_name}: {json.dumps(cell, default=float)}", flush=True)

        adjudication = {"promote_this_configuration": None, "rule": "see framing"}
        if "fingerprint" in arms and "bat_cpc_v2" in arms:
            # Paired stop rule: identical recordings, identical bootstrap
            # resamples and pair seeds for both arms — CI on the DIFFERENCE
            # (cpc − fp), not two CIs to eyeball. Promote only if the whole
            # interval favors cpc (negative delta = lower EER).
            from bioacoustic_pipeline.diagnostics.verification import (
                paired_bootstrap_delta,
            )

            ee_fp = np.asarray(arms["fingerprint"])[eer_ok]
            ee_cpc = np.asarray(arms["bat_cpc_v2"])[eer_ok]

            def _eer_stat_factory(seed):
                def stat(X, y, z):
                    return _eer_stat(X, y, z, n_pairs=args.n_pairs, seed=seed)

                return stat

            paired = paired_bootstrap_delta(
                _eer_stat_factory(args.seed),
                _eer_stat_factory(args.seed),
                ee_fp,
                ee_cpc,
                ce,
                fe,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            adjudication["paired_eer_delta_cpc_minus_fp"] = {
                "observed": paired.observed,
                "ci": [paired.ci_lo, paired.ci_hi],
            }
            adjudication["promote_this_configuration"] = bool(paired.ci_hi < 0.0)
            # Robustness variant: recording-group proxy (numeric-id blocks) —
            # Robustness variant: recording-group proxy (numeric-id blocks) —
            # genuine/impostor pairs and permutations stay within-block where
            # consecutive ids plausibly share recording conditions. Reported,
            # never adjudicated on (it is a proxy, not a session column).
            from bioacoustic_pipeline.ingestion.labeled_corpus import filename_block_group

            fe_grouped = np.array([filename_block_group(f) for f in fe])
            try:
                eer_grouped_fp = _eer_stat(
                    ee_fp, ce, fe_grouped, n_pairs=args.n_pairs, seed=args.seed
                )
                eer_grouped_cpc = _eer_stat(
                    ee_cpc, ce, fe_grouped, n_pairs=args.n_pairs, seed=args.seed
                )
                adjudication["robustness_block_grouped"] = {
                    "eer_fingerprint": eer_grouped_fp,
                    "eer_bat_cpc_v2": eer_grouped_cpc,
                    "note": "filename-block proxy, reported not adjudicated",
                }
            except ValueError as exc:
                adjudication["robustness_block_grouped"] = {"skipped": str(exc)[:120]}
        payload = {
            "section": "45",
            "self_test": args.self_test,
            "arms": results,
            "heldout_callers": [int(h) for h in heldout],
            "stop_rule": adjudication,
            "config": {
                "n_pairs": args.n_pairs,
                "n_permutations": args.n_permutations,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
            },
            "framing": FRAMING,
        }
        out = (
            ROOT
            / "outputs"
            / ("bat_caller_cpc_selftest.json" if args.self_test else "bat_caller_cpc.json")
        )
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"wrote {out}", flush=True)
    finally:
        if td_ctx is not None:
            td_ctx.cleanup()


if __name__ == "__main__":
    main()
