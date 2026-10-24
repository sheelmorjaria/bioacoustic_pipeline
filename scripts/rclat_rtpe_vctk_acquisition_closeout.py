"""Independent archive-byte checksum audit only; no ZIP/audio decoding."""

from datetime import datetime, timezone
import json
from pathlib import Path
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion import hashes, write
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

ROOT = Path("outputs/rclat_rtpe_vctk_support_execution_v1")
PROTOCOL = Path("outputs/rclat_rtpe_vctk_acquisition_protocol_v1/protocol.json")


def main():
    p = json.loads(PROTOCOL.read_text())
    e = json.loads((ROOT / "execution_manifest.json").read_text())
    assert sha(PROTOCOL) == e["protocol_sha256"]
    assert all(sha(q) == h for q, h in p["dependencies"].items()) and all(
        sha(q) == h for q, h in e["dependencies"].items()
    )
    auth = json.loads((ROOT / "SOURCE_AUTHORIZATION.json").read_text())
    assert auth["acquisition_authorized"] is True and auth["support_audit_authorized"] is False
    assert auth["protocol_sha256"] == sha(PROTOCOL) and auth["execution_sha256"] == sha(
        ROOT / "execution_manifest.json"
    )
    assert not (ROOT / "ACQUISITION_FAILURE.json").exists()
    receipt = json.loads((ROOT / "acquisition_receipt.json").read_text())
    source = json.loads((ROOT / "resolution" / "resolved_source.json").read_text())
    archive = Path(p["acquisition"]["destination"])
    actual = hashes(archive)
    assert actual["bytes"] == source["bytes"] == receipt["bytes"]
    assert actual["md5"] == p["source"]["published_md5"] == source["md5"] == receipt["md5"]
    assert actual["sha256"] == receipt["sha256"]
    assert receipt["protocol_sha256"] == sha(PROTOCOL) and receipt["execution_sha256"] == sha(
        ROOT / "execution_manifest.json"
    )
    assert not archive.with_name(archive.name + ".partial").exists()
    assert not (ROOT / "support").exists() and not (ROOT / "source_support_complete.json").exists()
    audit = dict(
        status="OFFICIAL_ARCHIVE_ACQUISITION_AND_INDEPENDENT_CHECKSUM_VERIFICATION_PASSED",
        verified_utc=datetime.now(timezone.utc).isoformat(),
        archive_path=str(archive),
        **actual,
        GiB=actual["bytes"] / 2**30,
        official_url=receipt["final_url"],
        independent_complete_byte_pass=True,
        archive_opened_for_member_inspection=False,
        waveforms_decoded=False,
        support_audit_authorized=False,
        model_inference=False,
        scientific_scoring=False,
        source_support_certified=False,
        scientific_transfer="UNEVALUATED",
    )
    write(ROOT / "acquisition_closeout.json", audit)
    report = f"""# VCTK archive acquisition — VERIFIED

The user-authorised source-only acquisition completed from the official Edinburgh repository. The downloaded file matched the publisher MD5;an independent complete-byte pass reproduced the download SHA256 and MD5.

- Archive: `{archive}`
- Size: **{actual["bytes"]:,} bytes** ({actual["bytes"] / 2**30:.4f} GiB)
- MD5: `{actual["md5"]}`
- SHA256: `{actual["sha256"]}`
- Final source: `{receipt["final_url"]}`

The pinned acquisition receipt retains response headers,source URL and protocol/execution hashes. Frozen dependencies and authorisation bindings were reverified. No existing archive was overwritten and no automatic retry/resume occurred. The verified final file is published;no partial file remains.

This is archive-byte integrity verification only. No ZIP-member inventory was read by this closeout,no waveform was decoded and no acoustic feature or forecasting score was computed. Source identity/speaker groups,duplicate events,mic2 coverage and real eligible-origin counts remain unknown. The separately required support-audit authorisation is still false. No scientific inference is authorised.

Next: separately authorised source-only archive/waveform/feature support auditing under the pinned runner. This must report actual source-support counts before a protected origin inventory and scientific transfer rules are frozen. Existing RTPE-1/R1 findings and P100/P960/MACDE–CAVEM blocks remain unchanged.
"""
    with (ROOT / "ACQUISITION_RESULTS.md").open("x") as f:
        f.write(report)
    files = [
        ROOT / "SOURCE_AUTHORIZATION.json",
        ROOT / "ACQUISITION_STARTED.json",
        ROOT / "acquisition_receipt.json",
        ROOT / "acquisition_closeout.json",
        ROOT / "ACQUISITION_RESULTS.md",
        ROOT / "acquisition_console.log",
        Path(__file__),
    ]
    write(ROOT / "acquisition_manifest.json", {str(q): sha(q) for q in files})
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
