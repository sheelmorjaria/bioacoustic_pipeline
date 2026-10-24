"""Roster-backed source-audit successor;no acquisition or scientific inference."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import traceback
import zipfile

import numpy as np
import scipy
import soundfile as sf
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion_v2 import audit_archive, hashes, write
from bioacoustic_pipeline.diagnostics.rclat_vctk_source_contract_v2 import (
    parse_speaker_roster,
    recording_identity,
    validate_archive_members,
)
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

ROOT = Path("outputs/rclat_rtpe_vctk_support_execution_v2")
BASE = Path("outputs/rclat_rtpe_vctk_acquisition_protocol_v1/protocol.json")
ROSTER = Path("outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/speaker-info.snapshot.txt")
RECEIPT = Path("outputs/rclat_rtpe_vctk_support_execution_v1/acquisition_receipt.json")


def environment():
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        soundfile=sf.__version__,
        libsndfile=sf.__libsndfile_version__,
    )


def prepare():
    base = json.loads(BASE.read_text())
    assert all(sha(p) == h for p, h in base["dependencies"].items())
    parse_speaker_roster(ROSTER.read_bytes(), sha(ROSTER))
    ROOT.mkdir(exist_ok=False)
    policy = copy.deepcopy(base)
    policy["status"] = "LOCALLY_FROZEN_ROSTER_BACKED_SUPPORT_SUCCESSOR_NOT_AUTHORIZED"
    policy["correction"] = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        predecessor_protocol=str(BASE),
        predecessor_protocol_sha256=sha(BASE),
        predecessor_failed_attempt="outputs/rclat_rtpe_vctk_support_execution_v1/support",
        scope="Only roster-backed identifier validation and roster evidence fields;all original support rules preserved",
        roster_member="speaker-info.txt",
        roster_snapshot=str(ROSTER),
        roster_sha256=sha(ROSTER),
        identifier_schema="p followed bydigits,plus explicit s5;mandatory pinned roster membership;strictdirectoryandfilenameagreement",
        archive_reuse="Alreadyacquiredverifiedarchive;no newdownload;freshauditdirectory;not continuation/resume",
        historical_preparation_fields="Inherited no_download/no_real_audio fields describe predecessor protocol freeze;archive is now acquired,no waveforms decoded yet",
    )
    write(ROOT / "protocol.json", policy)
    paths = [
        Path(__file__),
        BASE,
        ROSTER,
        RECEIPT,
        Path("outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/unrecognized_members.json"),
        Path("outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/metadata_summary.json"),
        Path("outputs/rclat_rtpe_vctk_support_execution_v1/support/FAILURE.json"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_source_contract_v2.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_source_contract.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_ingestion_v2.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_ingestion.py"),
        Path("tests/test_rclat_vctk_source_contract_v2.py"),
        Path("tests/test_rclat_vctk_ingestion_v2.py"),
        Path("tests/test_rclat_vctk_support_corrected.py"),
        Path(policy["preprocessing"]["coefficient_artifact"]),
    ]
    write(
        ROOT / "execution_manifest.json",
        dict(
            protocol_sha256=sha(ROOT / "protocol.json"),
            dependencies={str(p): sha(p) for p in paths},
            environment=environment(),
            maximum_decoded_member_bytes=256 * 2**20,
            no_automatic_retry=True,
            support_audit_authorized=False,
            forecasting_interface=False,
        ),
    )
    print("Roster-backed source-only successor pinned;no real support retry authorised.")


def bindings():
    p = json.loads((ROOT / "protocol.json").read_text())
    e = json.loads((ROOT / "execution_manifest.json").read_text())
    if (
        sha(ROOT / "protocol.json") != e["protocol_sha256"]
        or e["environment"] != environment()
        or not all(sha(q) == h for q, h in e["dependencies"].items())
        or not all(sha(q) == h for q, h in p["dependencies"].items())
    ):
        raise ValueError("Pinned successor dependency changed")
    return p, e


def authorize():
    path = ROOT / "SOURCE_AUTHORIZATION.json"
    if not path.is_file():
        raise ValueError("Fresh source-only support authorisation required")
    a = json.loads(path.read_text())
    if (
        a.get("support_audit_authorized") is not True
        or a.get("protocol_sha256") != sha(ROOT / "protocol.json")
        or a.get("execution_sha256") != sha(ROOT / "execution_manifest.json")
    ):
        raise ValueError("Fresh hash-bound support authorisation required")


def metadata():
    p, e = bindings()
    directory = ROOT / "metadata_verification"
    directory.mkdir(exist_ok=False)
    write(directory / "STARTED.json", dict(scope="Metadata only;zero waveform decodes"))
    with zipfile.ZipFile(p["acquisition"]["destination"]) as z:
        members = z.infolist()
        validate_archive_members(members)
        raw = z.read("speaker-info.txt")
        roster = parse_speaker_roster(raw, p["correction"]["roster_sha256"])
        events = defaultdict(dict)
        counts = Counter()
        names = []
        for member in sorted(members, key=lambda m: m.filename):
            if member.is_dir() or not member.filename.lower().endswith(".flac"):
                continue
            row = recording_identity(member.filename, roster)
            mic = row["microphone"]
            if mic in events[row["event"]]:
                raise ValueError("Ambiguous microphone event")
            events[row["event"]][mic] = member.filename
            counts[mic] += 1
            names.append(member.filename)
    bad = json.loads(
        Path(
            "outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/unrecognized_members.json"
        ).read_text()
    )
    assert all(recording_identity(row["member"], roster)["speaker"] == "s5" for row in bad)
    result = dict(
        passed=True,
        FLAC_members=len(names),
        rostered_speakers=len(roster),
        recording_events=len(events),
        mic_files=dict(counts),
        previously_rejected_recognized=len(bad),
        s5_events=sum(event.startswith("s5_") for event in events),
        waveform_decodes=0,
        acoustic_eligibility_certified=False,
        events_sha256=hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest(),
    )
    write(directory / "event_inventory.json", events)
    write(directory / "result.json", result)
    print(json.dumps(result, indent=2))


def audit():
    p, e = bindings()
    authorize()
    if shutil.disk_usage(".").free < p["acquisition"]["minimum_free_bytes"]:
        raise OSError("Frozen80GiB capacity requirement")
    if not json.loads((ROOT / "metadata_verification" / "result.json").read_text())["passed"]:
        raise ValueError("Verified corrected metadata required")
    archive = Path(p["acquisition"]["destination"])
    receipt = json.loads(RECEIPT.read_text())
    if hashes(archive)["sha256"] != receipt["sha256"]:
        raise ValueError("Verified acquired archive changed")
    result = audit_archive(
        archive,
        ROOT / "support",
        expected_md5=p["source"]["published_md5"],
        expected_roster_sha256=p["correction"]["roster_sha256"],
        coefficient_path=p["preprocessing"]["coefficient_artifact"],
        max_archive_bytes=p["acquisition"]["maximum_archive_bytes"],
        integrity_check=bindings,
        max_member_bytes=e["maximum_decoded_member_bytes"],
    )
    write(
        ROOT / "source_support_complete.json",
        dict(
            completed=True,
            counts=result["counts"],
            scientific_sampling_frozen=False,
            forecasting_unevaluated=True,
        ),
    )
    write(
        ROOT / "source_artifact_manifest.json",
        {str(q): sha(q) for q in ROOT.rglob("*") if q.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "metadata", "audit"])
    args = parser.parse_args()
    try:
        {"prepare": prepare, "metadata": metadata, "audit": audit}[args.action]()
    except Exception:
        if (
            args.action == "metadata"
            and ROOT.is_dir()
            and not (ROOT / "METADATA_FAILURE.json").exists()
        ):
            write(
                ROOT / "METADATA_FAILURE.json",
                dict(traceback=traceback.format_exc(), automatic_retry=False),
            )
        raise
