"""Local metadata-only transfer-support inventory; no audio or prediction access."""

from datetime import datetime, timezone
import json
from pathlib import Path
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_transfer_support_v1")


def main():
    ROOT.mkdir(exist_ok=False)
    inputs = [
        Path("data/LibriSpeech/SPEAKERS.TXT"),
        Path("outputs/rclat_rtpe_common_inventory_v1/selected_origins.json"),
        Path("outputs/rclat_scaling_preflight.json"),
        Path("outputs/rclat_study_b_source_audit.json"),
        Path("outputs/rclat_bilstm_p10_v1/protocol.json"),
    ]
    inputs = [p for p in inputs if p.is_file()]
    # Metadata receipts establish claimed archive identity, not fresh byte integrity.
    receipts = sorted(Path("data/rclat_scale/source").glob("*.verified.json"))
    inputs += receipts
    rows = json.loads(inputs[1].read_text())
    used = {str(r["speaker"]) for r in rows}
    groups = {}
    for line in inputs[0].read_text().splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [x.strip() for x in line.split("|")]
        groups.setdefault(fields[2], set()).add(fields[0])
    partition_info = []
    for partition in [
        "train-clean-100",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-360",
        "train-other-500",
    ]:
        speakers = groups.get(partition, set())
        partition_info.append(
            dict(
                partition=partition,
                corpus="LibriSpeech",
                official_metadata_speakers=len(speakers),
                overlap_with_RTPE_source_speakers=sorted(used & speakers),
                local_extracted_directory_present=any(
                    (base / partition).is_dir()
                    for base in [
                        Path("data/LibriSpeech"),
                        Path("data/rclat_scale/source/LibriSpeech"),
                    ]
                ),
                source_corpus_independent=False,
            )
        )
    receipt_info = [dict(metadata_path=str(p), **json.loads(p.read_text())) for p in receipts]
    result = dict(
        status="ABSTAIN_NO_INDEPENDENT_SOURCE_CORPUS_CERTIFIED",
        scope="Named local LibriSpeech speaker metadata,prior source/exposure audits,RTPE origin inventory and archive receipts only;not exhaustive system-wide dataset discovery",
        audit_utc=datetime.now(timezone.utc).isoformat(),
        RTPE_source_recordings=len(rows),
        RTPE_source_speakers=len(used),
        partitions=partition_info,
        source_receipts=receipt_info,
        exposure_assessment={
            "train-clean-100": "Exposed: completed RTPE and ASR development source",
            "dev-clean": "Exposed: historical RCLAT development",
            "dev-other": "Exposed: subsequent ASR validation and grounding studies;earlier scaling-preflight absence-of-reference note is superseded",
            "test-clean": "At least partial historical exposure documented in Study B",
            "test-other": "No untouched-data certification;absence of a usage reference is not proof of non-exposure",
        },
        source_corpus_confirmation_supported=False,
        new_speaker_same_corpus_transfer="Distinct prospective question;official metadata speaker disjointness is only preliminary,not an identity/source-integrity certificate",
        required_next=[
            "Identify an external speech corpus and obtain its official provenance/use documentation before acquisition",
            "Audit existing local/global exposure and independent speaker/recording identifiers",
            "Freeze support/selection rules before audio or predictive scoring",
            "Choose zero-shot transfer of frozen original models versus new-source refitting prospectively;these answer different questions",
            "Preserve original frontend,timing,history,Ridge comparator and group-level endpoint wherever technically compatible",
            "Certify source/feature support and protected evaluation access before freezing training/inference",
        ],
        audio_reads=0,
        archive_payload_reads=0,
        model_checkpoint_reads=0,
        prediction_reads=0,
        new_downloads=0,
        model_fitting=0,
        evaluation_scoring=0,
        source_rights_certified=False,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "audit.json", result)
    write(ROOT / "metadata_manifest.json", {str(p): sha(p) for p in inputs})
    (
        ROOT / "RESULTS.md"
    ).write_text("""# RTPE independent-source feasibility — metadata-only abstention

The inspected local records do not certify an independent speech-source corpus for the next RTPE study. The auditable source identities found in this bounded inspection belong to LibriSpeech. The 2,814-recording RTPE inventory contains 251 speakers from its train-clean-100 development population.

Different LibriSpeech partitions may offer distinct official speaker IDs, but remain the same source corpus. They must not be relabelled independent-source confirmation. Source-independent transfer and a new speaker sample within the same corpus are separate hypotheses. Official speaker-ID separation alone does not establish globally independent identities or historical non-exposure.

Prior exposure is documented for train-clean-100, dev-clean, dev-other and at least part of test-clean. The old scaling preflight's historical absence-of-reference note for dev-other is superseded by subsequent validation and grounding use. No untouched status is granted to test-other or any uninspected partition. Archive receipt hashes were read as metadata; waveform/archive bytes and feature validity were not reverified.

**Decision: ABSTAIN from certifying independent-source support.** This is a source/provenance limitation, not a negative predictive-utility result. The audit is bounded to the named local metadata and is not an exhaustive search of every mounted path or external resource.

Next: identify a candidate external corpus, verify official provenance/use conditions and exposure history, then freeze a support audit before audio processing. The protocol must distinguish zero-shot evaluation of frozen RTPE models from retraining on a new source. No seed/checkpoint selection using new evaluation outcomes is permitted.

No audio, checkpoint or prediction arrays were opened; no archive payload, download, model fitting or scoring occurred. Only JSON/text metadata and directory existence were inspected. No current source-rights certification or new protected split assignment is implied. Completed RTPE-1/R1 results remain unchanged. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
""")
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()})
    write(
        ROOT / "report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "manifest.json"): sha(ROOT / "manifest.json"),
        },
    )
    print(result["status"])
    print("RTPE source speakers:", len(used))
    print("No audio or predictive scoring.")


if __name__ == "__main__":
    main()
