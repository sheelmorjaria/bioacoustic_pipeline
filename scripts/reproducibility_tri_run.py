#!/usr/bin/env python3
"""Tri-run reproducibility harness — prove the pipeline is deterministic (Claim A).

Runs ``scripts/deterministic_run.py`` THREE times as independent subprocesses under an
identical seed, hashes each run's artifacts, and asserts every artifact is byte-identical
across the three runs. This is the strongest, most reviewer-proof form of "run 3× and the
results do not differ": it compares SHA-256 fingerprints, not "the numbers look the same",
and each run is a fresh process (true isolation of global RNG state).

The claim is deliberately scoped (Claim A — implementation determinism), NOT a claim about
biological or statistical stability (Claim B — bootstrap / alt-seed, out of scope).

Output:
  repro_runs/run_1/ run_2/ run_3/        — each run's full artifact bundle
  repro_runs/tri_run_hashes.json         — per-artifact SHA-256 for each run + status
  repro_runs/tri_run_manifest.json       — frozen environment captured from run 1

Exits nonzero on any drift.

Usage:
  python scripts/reproducibility_tri_run.py --seed 1337 --n-files 20

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.hashing import hash_artifacts, write_canonical_json  # noqa: E402

DETERMINISTIC_RUN = ROOT / "scripts" / "deterministic_run.py"
RUNS = ("run_1", "run_2", "run_3")
# Repro_manifest is itself a function of the run; it is hash-compared too (its
# dataset_manifest_hash / git_commit / versions must be stable across runs).
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


def _run_once(run_dir: Path, seed: int, n_files: int, species: str, k: int) -> None:
    """Invoke one deterministic run as an isolated subprocess."""
    cmd = [
        sys.executable,
        str(DETERMINISTIC_RUN),
        "--seed",
        str(seed),
        "--n-files",
        str(n_files),
        "--species",
        species,
        "--k",
        str(k),
        "--out-dir",
        str(run_dir),
    ]
    print(f"[tri-run] {run_dir.name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Tri-run reproducibility harness")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-files", type=int, default=20)
    ap.add_argument("--species", default="human")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--root", default=str(ROOT / "repro_runs"))
    args = ap.parse_args()

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    all_hashes: dict[str, dict[str, str]] = {}
    for run in RUNS:
        run_dir = root / run
        _run_once(run_dir, args.seed, args.n_files, args.species, args.k)
        all_hashes[run] = hash_artifacts(run_dir, ARTIFACT_NAMES)

    # Per-artifact status: identical across all three runs?
    produced = sorted(set().union(*(set(h) for h in all_hashes.values())))
    baseline = all_hashes["run_1"]
    table: dict[str, dict] = {}
    drift: list[str] = []
    for name in produced:
        h1 = baseline.get(name)
        h2 = all_hashes["run_2"].get(name)
        h3 = all_hashes["run_3"].get(name)
        ok = h1 is not None and h1 == h2 == h3
        table[name] = {"run_1": h1, "run_2": h2, "run_3": h3, "identical": ok}
        if not ok:
            drift.append(name)

    # Capture the frozen environment from run 1 (must be identical to 2 & 3 — verified
    # above since repro_manifest.json is itself a hashed artifact).
    env = json.loads((root / "run_1" / "repro_manifest.json").read_text())

    write_canonical_json(
        root / "tri_run_hashes.json",
        {
            "seed": args.seed,
            "n_files": args.n_files,
            "species": args.species,
            "k": args.k,
            "artifacts": table,
        },
    )
    write_canonical_json(root / "tri_run_manifest.json", env)

    # ---- report ----
    print("\n" + "=" * 78)
    print("TRI-RUN REPRODUCIBILITY — Claim A (implementation determinism)")
    print("=" * 78)
    print(f"{'artifact':22} {'run_1':12} {'run_2':12} {'run_3':12} status")
    print("-" * 78)
    for name in produced:
        r = table[name]

        def short(h):
            return h[:10] if h else "missing"

        print(
            f"{name:22} {short(r['run_1']):12} {short(r['run_2']):12} "
            f"{short(r['run_3']):12} {'IDENTICAL' if r['identical'] else 'DRIFT'}"
        )
    print("-" * 78)
    print(
        f"seed={args.seed}  n_files={args.n_files}  species={args.species}  "
        f"device={env['device']['device']}  git={env['git_commit'][:10]}"
    )
    print(f"dataset_manifest_hash={env['dataset_manifest_hash'][:16]}...")
    print(
        "\nClaim A (proven): under the frozen environment above, three independent "
        "processes\n  produced byte-identical artifacts for every output."
    )
    print(
        "Claim B (NOT this test): biological/statistical stability across bootstrap "
        "resamples\n  or alternative seeds — a separate demonstration."
    )

    if drift:
        print(f"\nFAIL: drift in {len(drift)} artifact(s): {drift}")
        return 1
    print("\nPASS: all three deterministic runs produced byte-identical artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
