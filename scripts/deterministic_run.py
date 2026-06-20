#!/usr/bin/env python3
"""One frozen deterministic DSP-core run -> canonical, hashable artifacts.

This is the single-run entry point the tri-run harness invokes three times. It proves
the pipeline **implementation** is deterministic (Claim A) under a fixed environment,
fixed seed, and fixed synthetic manifest — NOT that the biology is stable (Claim B).

Path (torch-free, offline, CPU):

  set_global_determinism(seed)
    -> generate a deterministic synthetic corpus (seeded; multi-burst .wav files)
    -> FIT PASS: fixed-threshold NBD segment each file -> extract acoustic fingerprints
       -> fit a seeded FingerprintTokenizer (k-means codebook) on the fingerprint matrix
    -> RUN PASS: run_pipeline(sorted paths, calibrator=False, tokenizer=fingerprint-adapter,
       fingerprint=True, seed=seed, intermediates=sink)  # real token sequences, no torch
    -> derive motifs + functional aliases from the token sequences (seeded nulls)
    -> materialize canonical artifacts (verdict / evidence / trace / segments / tokens /
       fingerprints / motifs / aliases / repro_manifest) into --out-dir

``calibrator=False`` keeps NBD on the pure-NumPy fixed-threshold path (no CPC -> no torch).
The FingerprintTokenizerAdapter produces token sequences from the deterministic acoustic
fingerprint, so LZ / alias / motif / coherence diagnostics are non-degenerate without
the human-only VQ-VAE.

Usage:
  python scripts/deterministic_run.py --seed 1337 --n-files 20 --out-dir repro_runs/run_1
  python scripts/deterministic_run.py --config configs/pipeline_deterministic.yaml

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.determinism import (  # noqa: E402
    build_repro_manifest,
    set_global_determinism,
)
from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (  # noqa: E402
    FingerprintTokenizer,
)
from bioacoustic_pipeline.diagnostics.alias_detector import detect_functional_aliases  # noqa: E402
from bioacoustic_pipeline.diagnostics import motif_miner as mm  # noqa: E402
from bioacoustic_pipeline.hashing import write_canonical_json, write_canonical_jsonl  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.orchestration import run_pipeline, segment_audio  # noqa: E402
from bioacoustic_pipeline.tokenization.fingerprint_tokenizer_adapter import (  # noqa: E402
    FingerprintTokenizerAdapter,
)
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

ARTIFACT_NAMES = [
    "segments.jsonl",
    "tokens.jsonl",
    "fingerprints.npy",
    "motifs.json",
    "aliases.json",
    "evidence_report.json",
    "verdict.json",
    "trace.json",
    "repro_manifest.json",
]


# --------------------------------------------------------------------------- #
# Deterministic synthetic corpus
# --------------------------------------------------------------------------- #


def _envelope(n: int, sr: int) -> np.ndarray:
    """Near-rectangular burst with a short attack + release so NBD fires a sharp onset."""
    env = np.ones(n, dtype=np.float32)
    attack = max(int(0.005 * sr), 1)
    release = max(int(0.020 * sr), 1)
    if attack < n:
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release < n:
        env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return env


def _one_call(rng: np.random.Generator, sr: int) -> np.ndarray:
    """One vocalization-like burst: a tone, a noise burst, or a chirp."""
    kind = int(rng.integers(0, 3))
    dur_ms = int(rng.integers(60, 200))
    n = max(int(sr * dur_ms / 1000), 1)
    t = np.arange(n, dtype=np.float64) / sr
    if kind == 0:  # tonal call
        f = float(rng.integers(300, 800))
        x = np.sin(2 * np.pi * f * t)
    elif kind == 1:  # broadband noise burst
        x = rng.standard_normal(n)
    else:  # linear chirp
        f0 = float(rng.integers(300, 600))
        f1 = float(rng.integers(600, 1200))
        x = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-9) * t * t))
    amp = float(rng.uniform(0.3, 0.7))
    return (amp * x * _envelope(n, sr)).astype(np.float32)


def generate_synthetic_corpus(out_dir: Path, n_files: int, sr: int, seed: int) -> list[str]:
    """Write ``n_files`` deterministic multi-burst .wav files; return sorted paths.

    Per-file RNG = ``default_rng((seed+1)*1_000_003 + i)`` so file content is independent
    of write order (the corpus is reproducible from seed + n_files + sr alone).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = int(sr * 0.250)  # 250 ms inter-call silence (> human 150 ms debounce)
    paths: list[str] = []
    for i in range(n_files):
        rng = np.random.default_rng((seed + 1) * 1_000_003 + i)
        n_calls = int(rng.integers(3, 6))
        parts = []
        for _ in range(n_calls):
            parts.append(_one_call(rng, sr))
            parts.append(np.zeros(gap, dtype=np.float32))
        audio = np.concatenate(parts).astype(np.float32)
        p = out_dir / f"synth_{i:04d}.wav"
        sf.write(str(p), audio, sr, subtype="FLOAT")
        paths.append(str(p))
    return sorted(paths)


# --------------------------------------------------------------------------- #
# Fit pass + run pass
# --------------------------------------------------------------------------- #


def fit_tokenizer(audio_paths, species: str, profile: dict, seed: int, k: int):
    """Segment each file (fixed-threshold NBD) + extract fingerprints -> fit the codebook."""
    sr = profile.get("sample_rate", 16000)
    matrix: list[np.ndarray] = []
    for path in audio_paths:
        audio, _ = load_audio(path, target_sr=sr)
        segs = segment_audio(audio, sr, profile, calibrator=None)  # threshold fallback, torch-free
        for seg in segs:
            fp = afp.extract(
                seg.audio,
                seg.sr,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                species=species,
                boundary_type=seg.boundary_type,
                boundary_confidence=seg.confidence,
            )
            if np.any(fp.combined):
                matrix.append(np.asarray(fp.combined, dtype=np.float64))
    fitted = FingerprintTokenizer.fit(np.stack(matrix), k=k, seed=seed)
    return fitted, len(matrix)


def _motifs_doc(sequences, seed: int) -> dict:
    """Token motifs + seeded support null (deterministic given sequences)."""
    total = sum(len(s) for s in sequences)
    doc: dict = {"n_sequences": len(sequences), "n_tokens": total}
    if total < 10:
        return doc
    raw = mm.mine_ngrams(sequences, min_n=2, max_n=5, min_support=3)
    motifs = mm.build_motifs(raw, sequences, motif_type="token")
    doc["n_motifs"] = len(motifs)
    doc["top_motifs"] = [m.as_dict() for m in motifs[:5]]
    try:
        doc["motif_support_null_p"] = float(
            mm.motif_support_null(sequences, n_permutations=100, seed=seed).p_value
        )
    except Exception:  # noqa: BLE001
        pass
    return doc


def _aliases_doc(sequences) -> dict:
    """Functional-alias groups (deterministic given sequences; controlled nulls elsewhere)."""
    if sum(len(s) for s in sequences) < 10:
        return {"n_groups": 0, "groups": []}
    rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
    return {
        "n_groups": len(rep.groups),
        "n_singleton_tokens": len(rep.singleton_tokens),
        "alias_density": round(
            len(rep.groups) / max(len(rep.singleton_tokens) + rep.n_merged, 1), 6
        ),
        "groups": [
            {
                "group_id": g.group_id,
                "member_tokens": list(g.member_tokens),
                "total_instances": g.total_instances,
            }
            for g in rep.groups
        ],
    }


def materialize(
    out_dir: Path, report: dict, trace: dict, sink: dict, sequences, seed: int, manifest: dict
) -> None:
    """Write the canonical artifact bundle into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(out_dir / "verdict.json", report)
    evidence = {k: report[k] for k in ("axes", "overall_confidence") if k in report}
    write_canonical_json(out_dir / "evidence_report.json", evidence)
    write_canonical_json(out_dir / "trace.json", trace)
    write_canonical_jsonl(out_dir / "segments.jsonl", sink["segments"])
    write_canonical_jsonl(
        out_dir / "tokens.jsonl",
        [{"file_idx": i, "n_tokens": len(s), "sequence": list(s)} for i, s in enumerate(sequences)],
    )
    fps = [np.asarray(f, dtype=np.float32) for f in sink["fingerprints"]]
    arr = np.stack(fps) if fps else np.zeros((0, 0), dtype=np.float32)
    np.save(out_dir / "fingerprints.npy", arr)
    write_canonical_json(out_dir / "motifs.json", _motifs_doc(sequences, seed))
    write_canonical_json(out_dir / "aliases.json", _aliases_doc(sequences))
    write_canonical_json(out_dir / "repro_manifest.json", manifest)


def run_deterministic(
    *, out_dir: str | Path, species: str, seed: int, n_files: int, k: int
) -> dict:
    """Execute one full frozen run; write artifacts; return the repro manifest."""
    out_dir = Path(out_dir)
    set_global_determinism(seed)

    profiles = load_species_profiles()
    if species not in profiles:
        raise ValueError(f"unknown species {species!r}; known: {sorted(profiles)}")
    profile = profiles[species]
    sr = profile.get("sample_rate", 16000)

    corpus_dir = out_dir / "corpus"
    audio_paths = generate_synthetic_corpus(corpus_dir, n_files, sr, seed)

    # FIT PASS: seed the fingerprint codebook on the corpus's own segments.
    fitted, n_fit = fit_tokenizer(audio_paths, species, profile, seed, k)
    tokenizer = FingerprintTokenizerAdapter(fitted, species=species)

    # RUN PASS: real token sequences via the fingerprint tokenizer, torch-free NBD.
    sink: dict = {}
    report, trace = run_pipeline(
        audio_paths,
        species,
        profiles=profiles,
        calibrator=False,
        tokenizer=tokenizer,
        fingerprint=True,
        with_null=True,
        seed=seed,
        intermediates=sink,
    )

    manifest = build_repro_manifest(
        species=species,
        seed=seed,
        audio_paths=audio_paths,
        profile=profile,
        tokenizer_info={
            "method": "kmeans_on_acoustic_fingerprint",
            "k": k,
            "seed": seed,
            "n_fit": n_fit,
        },
        calibrator_fitted=False,
        repo=ROOT,
    )
    materialize(out_dir, report, trace, sink, sink["sequences"], seed, manifest)
    return manifest


def _load_config(path: str | Path) -> dict:
    """Optional YAML config (lazy import; falls back to defaults if PyYAML absent)."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="One frozen deterministic DSP-core run")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline_deterministic.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-files", type=int, default=None)
    ap.add_argument("--species", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = _load_config(args.config) if Path(args.config).exists() else {}
    det = cfg.get("determinism", {}) if isinstance(cfg, dict) else {}
    fp = cfg.get("fingerprint_tokenizer", {}) if isinstance(cfg, dict) else {}

    seed = args.seed if args.seed is not None else int(det.get("seed", 1337))
    n_files = args.n_files if args.n_files is not None else int(cfg.get("n_files", 20))
    species = args.species or cfg.get("species", "human")
    k = args.k if args.k is not None else int(fp.get("k", 32))
    out_dir = args.out_dir or f"repro_runs/seed{seed}"

    manifest = run_deterministic(out_dir=out_dir, species=species, seed=seed, n_files=n_files, k=k)
    print(
        json.dumps(
            {
                "out_dir": out_dir,
                "seed": seed,
                "species": species,
                "n_files": n_files,
                "k": k,
                "dataset_manifest_hash": manifest["dataset_manifest_hash"],
                "git_commit": manifest["git_commit"],
                "device": manifest["device"]["device"],
                "n_artifacts": len(ARTIFACT_NAMES),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
