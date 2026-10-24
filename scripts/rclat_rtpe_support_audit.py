"""Metadata and source-code audit only; no real audio/cache reads or model fitting."""

import argparse
import hashlib
import json
from pathlib import Path

from bioacoustic_pipeline.diagnostics.rclat_rtpe_support import partition_metadata, timing_support
from bioacoustic_pipeline.representation.rclat_rtpe import validate_group_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path("outputs/rclat_p10/cohort.json")
    cohort = json.loads(source.read_text())
    # Use only the already exposed 10-hour training population; no dev/test selection.
    rows = partition_metadata(
        cohort["selected_training"], train_speakers=175, validation_speakers=38
    )
    validate_group_split(rows)
    parents = {}
    for r in rows:
        if (
            r["parent_recording_group"] in parents
            and parents[r["parent_recording_group"]] != r["split"]
        ):
            raise ValueError("Parent recording crosses partitions")
        parents[r["parent_recording_group"]] = r["split"]
    config = dict(
        history_frames=32,
        scales_frames=[1, 4, 16],
        horizons_frames=[4, 8, 16],
        primary_horizon_frames=8,
        frame_samples=640,
        hop_samples=160,
        sample_rate=16000,
    )
    inventory = []
    for row in rows:
        inventory.append(
            {
                **row,
                "source_file_present": Path(row["path"]).is_file(),
                "timing": timing_support(row["samples"], history=32, horizons=(4, 8, 16)),
            }
        )
    summary = {}
    for split in ("train", "validation", "test"):
        sub = [r for r in inventory if r["split"] == split]
        summary[split] = dict(
            speakers=len({r["individual"] for r in sub}),
            utterances=len(sub),
            parent_recording_groups=len({r["parent_recording_group"] for r in sub}),
            hours=sum(r["samples"] for r in sub) / 16000 / 3600,
            supported_recordings_upper_bound=sum(
                r["timing"]["common_origins_upper_bound"] > 0 for r in sub
            ),
            common_origins_upper_bound=sum(r["timing"]["common_origins_upper_bound"] for r in sub),
            missing_source_paths=sum(not r["source_file_present"] for r in sub),
        )
    design_columns = 32 * 64 * 2 + 1
    dense_bytes = summary["train"]["common_origins_upper_bound"] * design_columns * 8
    missing = sum(r["missing_source_paths"] for r in summary.values())
    report = dict(
        study="RTPE-1 metadata and causal-timing feasibility audit",
        completed=True,
        source="Existing P10 selected train-clean-100 subset only",
        source_metadata_sha256=sha(source),
        source_archive_receipt=cohort["archives"]["train-clean-100"],
        source_labels="Corpus speaker and speaker-chapter IDs; not independent identity re-adjudication",
        source_use_scope="Existing local research population; no new source rights certification",
        source_hashes="Recorded compressed-file and decoded-PCM hashes checked for exact metadata duplicates; audio bytes not rehashed",
        actual_waveform_and_cache_integrity_reverified=False,
        metadata_support_passed=missing == 0,
        split_summary=summary,
        timing_config=config,
        timing_convention="Frame j uses [160*j,160*j+640), latest consumed sample <=160*j+639; declared availability at exclusive end/16000.",
        static_mel_timing_audit_passed=True,
        synthetic_prefix_invariance_test_passed=True,
        feature_scope="64D mel only: direct trailing frame FFT, fixed Hann/filterbank/log; no per-recording normalisation. Physical features excluded.",
        global_scaler_reuse_prohibited=True,
        existing_cache_status="Not read or certified for RTPE. Need actual source/cache hash verification and new-partition train-only scaling.",
        observed_validity_support_verified=False,
        target_support="Targets use [origin_end+160*h-640, origin_end+160*h); h>=4 prevents waveform overlap. Common origins require all horizons.",
        dense_ridge_capacity=dict(
            design_columns=design_columns,
            full_training_design_float64_bytes=dense_bytes,
            full_training_design_GiB=dense_bytes / 2**30,
            note="Excludes raw windows, solver matrices and workspace; current HistoryRidge materialises a dense design and solves per target.",
        ),
        scientific_protocol_frozen=False,
        scientific_execution_ready=False,
        real_data_models_fitted=0,
        real_audio_read=False,
        feature_caches_read=False,
        final_test_access=False,
        previously_exposed_development_only=True,
        blockers=[
            "Actual waveform/cache integrity and validity re-verification",
            "Bounded common-origin sampling or streaming solver and resource preflight",
            "Final matched training/selection budgets and group-level uncertainty policy",
            "Source identity/parent provenance limitation and source-use scope documented before freeze",
        ],
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    draft = dict(
        status="DRAFT_NOT_FROZEN_NOT_EXECUTABLE",
        arms={"T0": "History-matched Ridge", "T1": "Causal multiscale GRU"},
        source_inventory="selected_metadata.json",
        exposure="All utterances previously used in ASR; held-out here means RTPE split-disjoint development, not untouched confirmation.",
        test_split_name="Internal RTPE evaluation only; not official LibriSpeech final test",
        input_features=64,
        timing=config,
        frame_history_extent_ms=350,
        forecast_offsets_from_availability_ms=[40, 80, 160],
        primary_horizon_ms=80,
        primary_target_window_start_after_origin_ms=40,
        scaling="One new training-only observed-cell scaler; no old P10 scaler",
        input_and_target_support="Identical ordered histories, masks and origin IDs; intersect eligibility across horizons before selecting any model",
        primary_loss="Per-recording mean squared error over observed standardised target cells at 80ms, then equal recording mean within speaker, then equal speaker mean",
        primary_contrast="Evaluation speaker-macro loss T0 minus T1; positive favours GRU",
        candidate_ridge_alpha=[0.1, 1.0, 10.0, 100.0],
        candidate_gru_hidden=64,
        selection="Validation speakers only; evaluation sealed until all choices frozen",
        uncertainty="Paired speaker-level descriptive analysis; exact method and replicate count unresolved. Do not use uncertified communication bootstrap.",
        training_budget=None,
        window_sampling_budget=None,
        auxiliary_order_active=False,
        behavior_active=False,
        linguistic_supervision_active=False,
        decision_rule="Unresolved; no scientific pass or representation/semantic claim permitted",
    )
    for name, value in [
        ("selected_metadata.json", inventory),
        ("audit.json", report),
        ("protocol_draft.json", draft),
    ]:
        (args.output / name).write_text(json.dumps(value, indent=2))
    lines = [
        "# RTPE-1 source and timing feasibility audit",
        "",
        "Metadata support and synthetic causal-prefix checks passed. Scientific execution remains blocked; this is a draft, not a frozen training protocol.",
        "",
        "| Draft partition | Speakers | Utterances | Hours | Common-origin capacity |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, r in summary.items():
        lines.append(
            f"| {split} | {r['speakers']} | {r['utterances']} | {r['hours']:.3f} | {r['common_origins_upper_bound']:,} |"
        )
    lines += [
        "",
        "The test label denotes a new internal development evaluation partition. All sources were exposed in previous ASR work; this is not an untouched confirmatory dataset. No dev-other or official final-test recordings were selected. Source labels support speaker-disjoint grouping and chapter-group consistency, not independently revalidated biological identities or deduplication beyond recorded exact hashes.",
        "",
        "Only metadata JSON and implementation source code were read; waveform paths were checked for existence. No real waveform bytes, features, predictions or checkpoints were read. Input file hashes were not reverified. Frame counts provide capacity upper bounds, not verified current validity support.",
        "",
        "Mel frame j consumes the half-open waveform interval [160j,160j+640) at 16kHz. Availability is its exclusive end. Horizons 4/8/16 correspond to 40/80/160ms; target windows never overlap observed waveform. The proposed primary 80ms target starts 40ms after the origin. A synthetic future-perturbation test confirmed prefix invariance of the existing mel extractor. This does not certify arbitrary cached or externally normalised features.",
        "",
        f"The unrestricted training Ridge design alone would require {dense_bytes / 2**30:.2f} GiB in float64, before windows, targets and solver workspace. A bounded shared-origin sampling rule or a tested streaming solver is required before scientific execution. No outcome-based sampling was performed.",
        "",
        "Draft primary endpoint: mean standardised squared target error at 80ms within recording, equal recording means within speaker, equal speaker means. Compare Ridge minus GRU on identical origins. Training budgets, bounded sampling, model selection and paired uncertainty still need prospective freeze. No temporal-order, linguistic or behavioural training is activated.",
        "",
        "Four new support tests and eight existing RTPE tests passed; lint passed. TDD began with the missing-module failure. Existing frozen scientific dependencies remain unchanged. Prior study outcomes and P100/P960/Stage 2 blocks remain unchanged.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "README.md").write_text(
        "See RESULTS.md, audit.json and the explicitly unfrozen protocol_draft.json. Metadata support is not scientific execution readiness.\n"
    )
    dependencies = [
        source,
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_support.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_ctc.py"),
        Path("src/bioacoustic_pipeline/diagnostics/macde_data.py"),
        Path("tests/test_rclat_rtpe_support.py"),
    ]
    files = dependencies + [p for p in args.output.iterdir() if p.is_file()]
    (args.output / "manifest.json").write_text(
        json.dumps({str(p): sha(p) for p in files}, indent=2)
    )
    print(
        json.dumps(
            {
                "split_summary": summary,
                "dense_design_GiB": dense_bytes / 2**30,
                "scientific_execution_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
