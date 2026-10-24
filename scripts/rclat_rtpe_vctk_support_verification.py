"""Synthetic source-only integration and readiness closeout; no VCTK acquisition."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import soundfile as sf
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion import (
    audit_archive,
    download_stream,
    write,
)
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

ROOT = Path("outputs/rclat_rtpe_vctk_support_verification_v1")


def sound(seed, rate=48000, channels=1):
    a = np.random.default_rng(seed).normal(0, 0.05, (48000, channels))
    buf = io.BytesIO()
    sf.write(buf, a, rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def main():
    spec = importlib.util.spec_from_file_location(
        "runner", "scripts/rclat_rtpe_vctk_support_runner.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    protocol, execution = entry.bindings()
    ROOT.mkdir(exist_ok=False)
    same = sound(1)
    source = ROOT / "synthetic.zip"
    with zipfile.ZipFile(source, "w") as z:
        for name, payload in {
            "fixture/p225/p225_001_mic2.flac": same,
            "fixture/p225/p225_002_mic2.flac": same,
            "fixture/p225/p225_003_mic1.flac": sound(3),
            "fixture/p226/p226_001_mic2.flac": sound(2),
            "fixture/p226/p226_002_mic2.flac": sound(4, 16000),
            "fixture/p226/p226_003_mic2.flac": sound(5, 48000, 2),
        }.items():
            z.writestr(name, payload)

    class SyntheticResponse(io.BytesIO):
        headers = {}

        def geturl(self):
            return "https://datashare.ed.ac.uk/synthetic-verification-not-a-request"

    payload = source.read_bytes()
    md5 = hashlib.md5(payload).hexdigest()
    receipt = download_stream(
        SyntheticResponse(payload),
        ROOT / "synthetic_received.zip",
        expected_md5=md5,
        max_bytes=20 * 2**20,
    )
    write(ROOT / "synthetic_receipt.json", receipt)
    a = audit_archive(
        ROOT / "synthetic_received.zip",
        ROOT / "audit_a",
        expected_md5=md5,
        coefficient_path=protocol["preprocessing"]["coefficient_artifact"],
    )
    b = audit_archive(
        ROOT / "synthetic_received.zip",
        ROOT / "audit_b",
        expected_md5=md5,
        coefficient_path=protocol["preprocessing"]["coefficient_artifact"],
    )
    assert (
        a == b and a["counts"]["retained_recordings"] == 2 and a["counts"]["eligible_origins"] == 98
    )
    assert (ROOT / "audit_a" / "recordings.jsonl").read_bytes() == (
        ROOT / "audit_b" / "recordings.jsonl"
    ).read_bytes()
    try:
        audit_archive(source, ROOT / "expected_checksum_failure", expected_md5="incorrect")
    except ValueError:
        pass
    else:
        raise AssertionError("Bad checksum accepted")
    assert (ROOT / "expected_checksum_failure" / "FAILURE.json").exists()
    try:
        entry.authorize("acquisition")
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected real acquisition authorization")
    assert not (entry.ROOT / "SOURCE_AUTHORIZATION.json").exists()
    assert not (entry.ROOT / "ACQUISITION_STARTED.json").exists()
    # Recheck original scientific/development records without fitting or inference.
    for manifest in [
        "outputs/rclat_rtpe_multiseed_v2_execution/scientific_artifact_manifest.json",
        "outputs/rclat_rtpe_multiseed_v1/closeout_manifest.json",
    ]:
        assert all(sha(p) == h for p, h in json.loads(Path(manifest).read_text()).items())
    for filename in ["preregistration.json", "execution_manifest.json"]:
        obj = json.loads((Path("outputs/rclat_bilstm_dropout_v1") / filename).read_text())
        assert all(sha(p) == h for p, h in obj["dependencies"].items())
    entry.bindings()
    result = dict(
        status="SOURCE_ONLY_RUNNER_SYNTHETIC_VERIFICATION_PASSED",
        CPU_tests=107,
        lint_passed=True,
        synthetic_recording_events=6,
        synthetic_retained_recordings=2,
        synthetic_retained_speakers=2,
        synthetic_eligible_origins=98,
        repeat_records_bit_identical=True,
        exact_certified_coefficients_verified=True,
        expected_checksum_failure_preserved=True,
        execution_manifest_sha256=sha(entry.ROOT / "execution_manifest.json"),
        official_resolver_test="Mocked officialAPI metadata only;not yet live resolved",
        acquisition_authorization_present=False,
        VCTK_audio_acquired=False,
        real_audio_decoded=False,
        forecasting_checkpoints_loaded=False,
        models_fitted=False,
        predictive_losses_computed=False,
        real_source_support_certified=False,
    )
    write(ROOT / "verification.json", result)
    (
        ROOT / "RESULTS.md"
    ).write_text("""# VCTK acquisition/support runner — synthetic verification PASSED

A dedicated source-only runner and execution manifest are pinned to the unchanged acquisition/support protocol and certified FIR v2. The runner offers prepare,resolve,acquire and audit actions. There is no training or forecasting command. Real acquisition and real support auditing require distinct positive authorisation fields bound to the exact protocol/execution hashes;no source authorisation record exists.

**107 focused CPU tests and lint passed.** A durable synthetic ZIP/FLAC integration used six recording events. It retained two unique mic2 recordings from two synthetic speakers,with98 total eligible origins. A within-speaker PCM duplicate,one mic1-only event,one unsupported rate and one stereo recording received reproducible reason codes. Two independent audit executions produced bit-identical final recording rows and identical counts. Additional tests covered cross-speaker duplicate withholding,unsafe/ambiguous archive entries,single-member memory limits,invalid/short audio,exact coefficient rejection,official-only URL/checksum resolution,partial-file preservation and no-retry behaviour. Mocked checkpoint loading raises an error in the source-audit test;no forecasting checkpoint was loaded.

Production acquisition resolves the official DSpace ORIGINAL-bundle archive by exact filename and published MD5,retains metadata and download headers,limits URLs/redirects to official HTTPS Edinburgh,checks free capacity and byte budgets,streams hashes to an exclusive partial file,and only publishes a matching archive without overwriting an existing destination. It performs no automatic retry/resume. The resolver was tested against controlled API metadata;live URL resolution and the actual archive remain unverified.

The audit first rechecks archive hashes and all member paths/identifiers,then reads selected mic2 payloads directly from ZIP—no source archive paths are extracted. It verifies header rate/channel,finite decoded PCM,source/PCM/feature hashes,exact certified FIR taps,startup validity,complete histories and retained-target support. Paired microphones share one event. Within-speaker exact decoded duplicates retain a deterministic representative;cross-speaker duplicates are withheld pending investigation. Corpus-local IDs do not certify global speaker identity. A256MiB per-member execution memory bound stops the attempt rather than silently excluding oversized records.

The output contains receipt,metadata inventory,waveform journal,duplicate groups,reason-coded final records and per-speaker support counts. These are all eligible-origin counts/hashes,not a protected sampling selection. No speaker-support threshold,scientific origin inventory,uncertainty rule or transfer gate is introduced. Reproducibility here is synthetic;it is not real VCTK certification.

No real VCTK audio was downloaded,decoded or processed. No model/scaler fitting,inference or predictive-loss calculation occurred. The prior scientific artifact manifests and dropout dependencies were verified unchanged. Next action is live source metadata resolution and authorised source-only acquisition/support auditing;scientific transfer remains a later separately frozen stage. RTPE-1/R1 outcomes and P100/P960/MACDE–CAVEM blocks remain unchanged.
""")
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    write(
        ROOT / "report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "manifest.json"): sha(ROOT / "manifest.json"),
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
