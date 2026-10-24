"""Corrected runner readiness closeout;synthetic PCM plus real metadata only."""

import importlib.util
import io
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import soundfile as sf
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion_v2 import audit_archive, write
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

ROOT = Path("outputs/rclat_rtpe_vctk_support_execution_v2")


def main():
    spec = importlib.util.spec_from_file_location(
        "runner", "scripts/rclat_rtpe_vctk_support_corrected.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    p, e = entry.bindings()
    metadata = json.loads((ROOT / "metadata_verification" / "result.json").read_text())
    assert metadata["passed"] and metadata["previously_rejected_recognized"] == 796
    assert not (ROOT / "SOURCE_AUTHORIZATION.json").exists() and not (ROOT / "support").exists()
    synthetic = ROOT / "synthetic_verification"
    synthetic.mkdir(exist_ok=False)
    roster = b"ID AGE GENDER\np225 23 F\ns5 22 F\n"
    roster_sha = hashlib.sha256(roster).hexdigest()
    archive = synthetic / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("speaker-info.txt", roster)
        for speaker, seed in [("p225", 19), ("s5", 20)]:
            samples = np.random.default_rng(seed).normal(0, 0.05, 48000)
            buf = io.BytesIO()
            sf.write(buf, samples, 48000, format="FLAC", subtype="PCM_16")
            z.writestr(f"fixture/{speaker}/{speaker}_001_mic2.flac", buf.getvalue())
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    a = audit_archive(
        archive,
        synthetic / "a",
        expected_md5=md5,
        expected_roster_sha256=roster_sha,
        coefficient_path=p["preprocessing"]["coefficient_artifact"],
    )
    b = audit_archive(
        archive,
        synthetic / "b",
        expected_md5=md5,
        expected_roster_sha256=roster_sha,
        coefficient_path=p["preprocessing"]["coefficient_artifact"],
    )
    assert (
        a == b
        and a["counts"]["eligible_origins"] == 98
        and set(a["speaker_support"]) == {"p225", "s5"}
    )
    assert (synthetic / "a" / "recordings.jsonl").read_bytes() == (
        synthetic / "b" / "recordings.jsonl"
    ).read_bytes()
    old = json.loads(entry.BASE.read_text())
    assert all(p[k] == v for k, v in old.items() if k != "status")
    for manifest in [
        "outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/manifest.json",
        "outputs/rclat_rtpe_multiseed_v2_execution/scientific_artifact_manifest.json",
    ]:
        assert all(sha(q) == h for q, h in json.loads(Path(manifest).read_text()).items())
    entry.bindings()
    audit = dict(
        status="CORRECTED_ROSTER_BACKED_SUPPORT_RUNNER_READY_NOT_AUTHORIZED",
        CPU_tests=120,
        lint_passed=True,
        real_metadata_FLAC_members=88328,
        real_metadata_rostered_speakers=110,
        previously_rejected_names_recognized=796,
        s5_event_groups=398,
        synthetic_source_events=2,
        synthetic_eligible_origins=98,
        synthetic_repeat_records_bit_identical=True,
        old_support_rules_preserved=True,
        frozen_failed_attempt_preserved=True,
        real_waveform_decodes=0,
        real_acoustic_eligibility="UNKNOWN",
        support_audit_authorized=False,
        forecasting_interface=False,
        protocol_sha256=sha(ROOT / "protocol.json"),
        execution_manifest_sha256=sha(ROOT / "execution_manifest.json"),
    )
    write(ROOT / "preparation_closeout.json", audit)
    report = """# VCTK roster-backed support successor v2 — readiness PASSED

A separately versioned source-identifier contract,ingestion module,source-only protocol and runner are pinned. The new parser requires the exact preserved official speaker-info.txt SHA256,recognises p-prefixed numeric IDs and explicit s5,and enforces roster membership plus directory/filename/utterance/microphone consistency. Unknown namespaces and unrostered IDs still fail. It does not accept arbitrary letter-number IDs.

**120 focused CPU tests and lint passed.** Regression tests accept p/s5 microphone pairs,reject unknown/malformed/mismatched identifiers and verify all796previously rejected names without declaring them acoustically eligible. Syntax-tree comparisons confirm the previous ingestion body is unchanged after accounting solely for the roster read,identifier-call binding and roster provenance fields. Waveform processing,FIR,checksum,duplicate,mask and support calculations are identical.

A metadata-only pass of the acquired archive recognised **88,328FLACmembers** across110rostered speakers and44,455event groups,with44,455mic1 and43,873mic2 files. All796s5members form398events. An event-inventory hash is saved. This is source metadata support,not acoustic eligibility or globally verified human identity. No real waveform was decoded.

A durable two-speaker synthetic integration—including s5—ran twice using the certified FIR. Both executions yielded identical source-support summaries and bit-identical recording records,with98synthetic eligible origins. These are synthetic counts only. All original exclusion and duplicate tests also pass against the corrected module.

The official archive is reused without download. The original failed support directory,its failure bytes and original code/manifests remain intact. The successor uses a fresh directory and requires a new authorisation bound to its exact protocol and execution hashes;prior v1permission does not authorise it. No successor authorisation record has been created and the real audit action has not run. It has no acquisition,forecasting or training action.

Next:explicitly authorise a fresh source-only support audit to establish actual waveform/feature/duplicate/eligible-origin counts. Protected sampling,uncertainty and external-transfer decisions remain unfrozen. Original RTPE-1/R1 positive findings and P100/P960/MACDE–CAVEM blocks remain unchanged.
"""
    with (ROOT / "RESULTS.md").open("x") as f:
        f.write(report)
    write(
        ROOT / "preparation_artifact_manifest.json",
        {str(q): sha(q) for q in ROOT.rglob("*") if q.is_file()},
    )
    write(
        ROOT / "preparation_report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "preparation_artifact_manifest.json"): sha(
                ROOT / "preparation_artifact_manifest.json"
            ),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
