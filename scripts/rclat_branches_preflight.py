"""Read-only readiness audit for new ASR/communication development code."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bioacoustic_pipeline.diagnostics.rclat_communication import (
    canonicalize_responses,
    listener_targets,
    readiness,
)
from bioacoustic_pipeline.ingestion.request_protocol import wording_gate


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite an existing readiness report")
    base = Path("outputs/rclat_request_protocol")
    sources = Path("outputs/rclat_study_c_source_audit_stimuli.json")
    judgments = Path("outputs/rclat_study_c_source_audit_human_judgments.json")

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    original = json.loads((base / "build_manifest.json").read_text())
    for p, digest in {**original["source_files"], **original["code"]}.items():
        if sha(p) != digest:
            raise ValueError("Frozen request source changed: " + p)
    s = json.loads(sources.read_text())
    r = json.loads(judgments.read_text())
    if r.get("model_generated_labels") is not False:
        raise ValueError("Listener targets must be independently observed responses")
    aliases = {"temp": "temperature", "Spanish": "spanish"}
    normalized = canonicalize_responses(r["responses"], aliases)
    targets = listener_targets(s["stimuli"], normalized)
    crosswalk = json.loads((base / "custodian_only/wording_crosswalk.json").read_text())["stimuli"]
    reviews = [read_csv(base / name / "review.csv") for name in ["reviewer_A", "reviewer_B"]]
    metadata = [
        json.loads((base / name / "metadata.json").read_text())
        for name in ["reviewer_A", "reviewer_B"]
    ]
    wording = wording_gate(
        crosswalk, *reviews, *metadata, read_csv(base / "custodian_only/adjudication.csv")
    )
    status = readiness(wording=wording, source_use=False, methodology_frozen=False)
    inputs = [
        sources,
        judgments,
        base / "build_manifest.json",
        base / "custodian_only/wording_crosswalk.json",
        base / "custodian_only/adjudication.csv",
    ]
    inputs += [
        base / name / file
        for name in ["reviewer_A", "reviewer_B"]
        for file in ["review.csv", "metadata.json"]
    ]
    report = {
        "asr": {
            "implementation": "continuous encoder + character/phonological-feature readouts and observed-cell soft-reference loss",
            "baseline_character_path_tested": True,
            "independent_phonological_references_supplied": False,
            "scientific_ready": False,
            "blockers": [
                "independent phonological reference source and time-grid/support audit",
                "separate frozen auxiliary-supervision comparison; no modification of running dropout experiment",
            ],
            "scientific_models_fitted": 0,
        },
        "communication": {
            **status,
            "stimuli": len(targets),
            "explicit_text_id_aliases": aliases,
            "alias_response_counts": {
                key: sum(row.get("text_id") == key for row in r["responses"]) for key in aliases
            },
            "listener_responses": len(r["responses"]),
            "listeners": len({x["listener_id"] for x in r["responses"]}),
            "stimuli_with_disagreement": sum(
                0 < t["request_probability"] < 1 for t in targets.values()
            ),
            "wording_matched": wording["script_matched_stimuli"],
            "wording_unresolved": len(wording["unresolved_stimuli"]),
            "source_use_note": "Unresolved in existing study; this audit does not infer reuse permission.",
            "methodology_note": "Prototype and resampling kernels implemented; exact real-data features, scaler/fitting/selection policy and uncertainty use still require prospective freeze.",
            "scientific_models_fitted": 0,
        },
        "independent_reference_labels_created": False,
        "models_fitted": 0,
        "audio_processed": False,
        "running_experiment_modified": False,
        "input_hashes": {str(p): sha(p) for p in inputs},
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "listener_distributions.json").write_text(json.dumps(targets, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
