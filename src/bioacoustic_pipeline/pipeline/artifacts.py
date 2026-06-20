"""ArtifactWriter — one canonical output bundle for every run (factored from
``scripts/deterministic_run.materialize``).

Emits, under ``out_dir``: verdict / evidence_report / trace / domain_gate / config /
reproducibility_manifest JSON, segments / dual_stream_events / tokens JSONL, fingerprints
NPY, a per-diagnostic JSON, and a ``hashes.json`` (SHA-256 of each artifact — the tri-run
determinism contract). Canonical JSON via ``hashing.write_canonical_json``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioacoustic_pipeline.hashing import hash_artifacts, write_canonical_json, write_canonical_jsonl

# Artifact names written for every run (used by the tri-run harness + hashes.json).
ARTIFACT_NAMES = [
    "segments.jsonl",
    "dual_stream_events.jsonl",
    "tokens.jsonl",
    "fingerprints.npy",
    "diagnostics",
    "evidence_report.json",
    "verdict.json",
    "trace.json",
    "domain_gate.json",
    "config.json",
    "reproducibility_manifest.json",
]
# Flat (hashable) artifacts — diagnostics/ is a directory, hashed per-file separately.
_FLAT_ARTIFACTS = [n for n in ARTIFACT_NAMES if n != "diagnostics"]


class ArtifactWriter:
    """Write a :class:`PipelineResult` as the canonical artifact bundle."""

    @staticmethod
    def write(result, out_dir: str | Path) -> dict:
        out = Path(out_dir)
        (out / "diagnostics").mkdir(parents=True, exist_ok=True)

        write_canonical_json(out / "verdict.json", result.report)
        evidence = {
            k: result.report[k] for k in ("axes", "overall_confidence") if k in result.report
        }
        write_canonical_json(out / "evidence_report.json", evidence)
        write_canonical_json(out / "trace.json", result.trace)
        write_canonical_json(out / "domain_gate.json", result.domain_gate)
        write_canonical_json(out / "reproducibility_manifest.json", result.manifest)
        write_canonical_json(out / "config.json", result.config or {})

        write_canonical_jsonl(out / "segments.jsonl", result.segments)
        write_canonical_jsonl(out / "dual_stream_events.jsonl", result.dual_stream)
        write_canonical_jsonl(
            out / "tokens.jsonl",
            [
                {"file_idx": i, "n_tokens": len(s), "sequence": list(s)}
                for i, s in enumerate(result.sequences)
            ],
        )

        fps = []
        for e in result.dual_stream:
            fp = e.get("fingerprint")
            if fp is None:
                continue
            arr = np.asarray(fp, dtype=np.float32)
            if arr.size > 0:
                fps.append(arr)
        np.save(
            out / "fingerprints.npy", np.stack(fps) if fps else np.zeros((0, 0), dtype=np.float32)
        )

        for name, dr in result.diagnostics.items():
            write_canonical_json(out / "diagnostics" / f"{name}.json", dr.as_dict())

        hashes = hash_artifacts(out, _FLAT_ARTIFACTS)
        # include per-diagnostic hashes
        diag_hashes = {
            f"diagnostics/{p.name}": hash_artifacts(p.parent, [p.name])[p.name]
            for p in sorted((out / "diagnostics").glob("*.json"))
        }
        all_hashes = {**hashes, **diag_hashes}
        write_canonical_json(out / "hashes.json", all_hashes)

        return {"out_dir": str(out), "n_artifacts": len(all_hashes), "hashes": all_hashes}
