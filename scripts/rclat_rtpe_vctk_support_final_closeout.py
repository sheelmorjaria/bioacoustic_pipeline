"""Independent host audit of completed VCTK source-support records;no inference."""

from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_vctk_support_execution_v2")


def main():
    spec = importlib.util.spec_from_file_location(
        "runner", "scripts/rclat_rtpe_vctk_support_corrected.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    protocol, execution = entry.bindings()
    entry.authorize()
    support = ROOT / "support"
    assert (ROOT / "source_support_complete.json").is_file() and not (
        support / "FAILURE.json"
    ).exists()
    summary = json.loads((support / "support_summary.json").read_text())
    assert summary["completed"]
    manifest = json.loads((ROOT / "source_artifact_manifest.json").read_text())
    assert all(sha(p) == h for p, h in manifest.items())
    metadata = json.loads((support / "metadata_inventory.json").read_text())
    assert metadata["speaker_roster_sha256"] == protocol["correction"]["roster_sha256"]
    events = {row["event"]: row["recordings"] for row in metadata["events"]}
    raw = [json.loads(s) for s in (support / "waveform_journal.jsonl").read_text().splitlines()]
    final = [json.loads(s) for s in (support / "recordings.jsonl").read_text().splitlines()]
    assert (
        len(raw) == len(final) == len(events) == 44455 and len({r["event"] for r in final}) == 44455
    )
    assert [r["event"] for r in raw] == [r["event"] for r in final] == sorted(events)
    groups = defaultdict(list)
    checked_origins = 0
    for row in raw:
        if row["reason"] == "mic2_absent":
            assert "mic2" not in events[row["event"]]
        else:
            assert row["member"] == events[row["event"]]["mic2"]["member"]
        if "decoded_pcm_sha256" in row:
            groups[row["decoded_pcm_sha256"]].append(row)
        if "mel_frames" in row:
            nout = (row["frames"] + 2) // 3
            frames = max(0, (nout - 640) // 160 + 1)
            assert row["mel_frames"] == frames
            # Native finite mono input and the frozen frontend imply every frame
            # after startup is observed. Independently reconstruct timing support.
            assert row["unsupported_mel_frames"] == 1
            origins = np.arange(32, max(32, frames - 16), dtype="<i8")
            import hashlib

            assert row["eligible_origin_count"] == len(origins)
            assert row["eligible_origin_sha256"] == hashlib.sha256(origins.tobytes()).hexdigest()
            checked_origins += 1
    duplicate_reason = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        if len({r["speaker"] for r in group}) > 1:
            duplicate_reason.update({r["event"]: "cross_speaker_duplicate" for r in group})
        else:
            duplicate_reason.update(
                {
                    r["event"]: "within_speaker_duplicate"
                    for r in sorted(group, key=lambda r: r["event"])[1:]
                }
            )
    for before, after in zip(raw, final):
        assert after["reason"] == duplicate_reason.get(before["event"], before["reason"])
        for key, value in before.items():
            if key != "reason":
                assert after[key] == value
    retained = [r for r in final if r["reason"] == "eligible"]
    independently = dict(
        recording_events=len(final),
        retained_recordings=len(retained),
        retained_speakers=len({r["speaker"] for r in retained}),
        eligible_origins=sum(r["eligible_origin_count"] for r in retained),
        exclusion_reasons=dict(
            sorted(Counter(r["reason"] for r in final if r["reason"] != "eligible").items())
        ),
    )
    for key, value in independently.items():
        assert summary["counts"][key] == value
    speakers = {}
    for row in retained:
        value = speakers.setdefault(row["speaker"], dict(recordings=0, eligible_origins=0))
        value["recordings"] += 1
        value["eligible_origins"] += row["eligible_origin_count"]
    assert summary["speaker_support"] == speakers
    source_receipt = json.loads((support / "archive_receipt.json").read_text())
    acquisition = json.loads(
        Path("outputs/rclat_rtpe_vctk_support_execution_v1/acquisition_receipt.json").read_text()
    )
    assert all(source_receipt[k] == acquisition[k] for k in ["bytes", "md5", "sha256"])
    assert source_receipt["md5"] == protocol["source"]["published_md5"]
    # Preserve predecessor failure and scientific evidence independently.
    for path in [
        "outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/manifest.json",
        "outputs/rclat_rtpe_multiseed_v2_execution/scientific_artifact_manifest.json",
    ]:
        assert all(sha(p) == h for p, h in json.loads(Path(path).read_text()).items())
    counts = np.array([r["eligible_origin_count"] for r in retained], dtype=float)
    audit = dict(
        status="SOURCE_ONLY_WAVEFORM_AND_FEATURE_SUPPORT_AUDIT_COMPLETED_VERIFIED",
        counts=independently,
        recording_rows_verified=len(final),
        timing_origin_arrays_independently_verified=checked_origins,
        duplicate_assignments_independently_verified=True,
        origin_count_per_retained_recording=dict(
            minimum=int(counts.min()) if len(counts) else 0,
            median=float(np.median(counts)) if len(counts) else 0,
            maximum=int(counts.max()) if len(counts) else 0,
        ),
        s5_support=speakers.get("s5"),
        archive_hashes=source_receipt,
        speaker_ids_scope="Corpus-local roster identities;global cross-corpus human identity unresolved",
        protected_origins_selected=False,
        source_support_threshold_applied=False,
        scientific_uncertainty_frozen=False,
        scientific_transfer_gate_frozen=False,
        forecasting_model_inference=False,
        predictive_scoring=False,
        independent_validation_scope="Saved-source-record and timing/duplicate recomputation only;no additional waveform decode",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "support_closeout.json", audit)
    lines = [
        "# VCTK real-source support audit v2 — COMPLETED AND VERIFIED",
        "\nThe separately authorised,roster-backed source-only audit reused the verified official archive. It completed archive/metadata/waveform/feature/support processing without forecasting-model access. The original failed v1attempt remains preserved.",
        "\n| Source-support measure | Verified result |",
        "|---|---:|",
        f"| Recording events inventoried | {independently['recording_events']:,} |",
        f"| Retained unique mic2 recordings | {independently['retained_recordings']:,} |",
        f"| Speakers with retained support | {independently['retained_speakers']:,} |",
        f"| Eligible origins across retained recordings | {independently['eligible_origins']:,} |",
        "\nExclusion reasons:",
    ]
    lines.extend(
        f"- {reason}: {count:,}" for reason, count in independently["exclusion_reasons"].items()
    )
    lines += [
        f"\nThe host closeout independently checked all{len(final):,}recording rows,duplicate assignments and{checked_origins:,}timing-derived origin-array hashes. Archived source receipts matched the acquired MD5/SHA256/byte count. Retained-record origin counts ranged from{audit['origin_count_per_retained_recording']['minimum']} to{audit['origin_count_per_retained_recording']['maximum']},with median{audit['origin_count_per_retained_recording']['median']:.1f}. Per-speaker support and all reason-coded records are retained under support/.",
        "\nEligibility uses the unchanged mic2-only/no-fallback rule,finite mono48kHz decoded data,certified193-tap causal FIR v2,explicit startup masks,complete32-frame histories and disjoint retained80/160ms targets. The40ms output remains outside scientific scoring. Exact decoded PCM duplicates are grouped deterministically;cross-speaker duplicates are withheld. Speaker identities are roster-local and do not prove globally independent human identities across corpora.",
        "\nThese are empirical source-support counts,not forecasting results or a protected origin sample. No scientific support threshold,uncertainty procedure or external-transfer gate is introduced by this audit. Next:prospectively freeze protected origin selection,group-level inference and the all-five zero-shot comparison,then prepare its inference runner. No GRU/Ridge/scaler fitting,checkpoint selection,model inference or predictive losses occurred. RTPE-1/R1 results and P100/P960/MACDE–CAVEM blocks remain unchanged.",
    ]
    with (ROOT / "SUPPORT_RESULTS.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    write(
        ROOT / "support_closeout_manifest.json",
        {
            str(p): sha(p)
            for p in [
                ROOT / "support_closeout.json",
                ROOT / "SUPPORT_RESULTS.md",
                Path(__file__),
                ROOT / "source_artifact_manifest.json",
            ]
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
