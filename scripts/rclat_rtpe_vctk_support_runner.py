"""Pinned source-only VCTK runner:prepare,resolve,acquire,audit;no prediction action."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import traceback
from urllib.request import build_opener, HTTPRedirectHandler

import numpy as np
import scipy
import soundfile as sf
from bioacoustic_pipeline.diagnostics.rclat_vctk_ingestion import (
    audit_archive,
    download_stream,
    safe_url,
    write,
    hashes,
)
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha

ROOT = Path("outputs/rclat_rtpe_vctk_support_execution_v1")
PROTOCOL = Path("outputs/rclat_rtpe_vctk_acquisition_protocol_v1/protocol.json")
ITEM = "https://datashare.ed.ac.uk/server/api/core/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/bundles?size=100"


class OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def environment():
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        soundfile=sf.__version__,
        libsndfile=sf.__libsndfile_version__,
    )


def prepare():
    p = json.loads(PROTOCOL.read_text())
    assert all(sha(path) == h for path, h in p["dependencies"].items())
    ROOT.mkdir(exist_ok=False)
    paths = [
        Path(__file__),
        PROTOCOL,
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_ingestion.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_source_contract.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_causal_resample_v2.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_ctc.py"),
        Path("tests/test_rclat_vctk_ingestion.py"),
        Path("tests/test_rclat_vctk_support_runner.py"),
        Path(p["preprocessing"]["coefficient_artifact"]),
    ]
    write(
        ROOT / "execution_manifest.json",
        dict(
            status="PINNED_SOURCE_ONLY_RUNNER_NOT_AUTHORIZED",
            protocol_sha256=sha(PROTOCOL),
            dependencies={str(q): sha(q) for q in paths},
            environment=environment(),
            actions=["resolve", "acquire", "audit"],
            scientific_prediction_interface=False,
            max_metadata_bytes=2**21,
            network_timeout_seconds=30,
            no_automatic_retry=True,
            maximum_decoded_member_bytes=256 * 2**20,
        ),
    )
    print("Source-only runner pinned;no archive or real audio accessed.")


def bindings():
    e = json.loads((ROOT / "execution_manifest.json").read_text())
    p = json.loads(PROTOCOL.read_text())
    if (
        sha(PROTOCOL) != e["protocol_sha256"]
        or not all(sha(q) == h for q, h in e["dependencies"].items())
        or not all(sha(q) == h for q, h in p["dependencies"].items())
        or e["environment"] != environment()
    ):
        raise ValueError("Pinned source policy/dependency/environment changed")
    return p, e


def authorize(action):
    path = ROOT / "SOURCE_AUTHORIZATION.json"
    if not path.is_file():
        raise ValueError("Explicit source acquisition/support authorisation required")
    a = json.loads(path.read_text())
    if (
        a.get(action + "_authorized") is not True
        or a.get("protocol_sha256") != sha(PROTOCOL)
        or a.get("execution_sha256") != sha(ROOT / "execution_manifest.json")
    ):
        raise ValueError("Hash-bound source authorisation required")


def capacity(p):
    if shutil.disk_usage(".").free < p["acquisition"]["minimum_free_bytes"]:
        raise OSError("Frozen80GiB free-capacity requirement")


def resolve_archive(opener, p, root):
    def get(url, name):
        with opener.open(safe_url(url), timeout=30) as response:
            safe_url(response.geturl())
            payload = response.read(2**21 + 1)
        if len(payload) > 2**21:
            raise ValueError("Metadata size limit")
        data = json.loads(payload)
        if data.get("page", {}).get("totalPages", 1) > 1:
            raise ValueError("Uninspected metadata pagination")
        write(root / name, data)
        return data

    bundles = get(ITEM, "official_bundles.json")["_embedded"]["bundles"]
    candidates = []
    for index, bundle in enumerate(bundles):
        if bundle["name"] != "ORIGINAL":
            continue
        data = get(
            bundle["_links"]["bitstreams"]["href"], "official_bitstreams_" + str(index) + ".json"
        )
        candidates.extend(
            b for b in data["_embedded"]["bitstreams"] if b["name"] == p["source"]["archive_name"]
        )
    if len(candidates) != 1:
        raise ValueError("Unique official archive not resolved")
    candidate = candidates[0]
    checksum = candidate.get("checkSum", candidate.get("checksum", {}))
    if (
        checksum.get("checkSumAlgorithm", "").upper() != "MD5"
        or checksum.get("value") != p["source"]["published_md5"]
    ):
        raise ValueError("Official archive metadata checksum differs")
    url = safe_url(candidate["_links"]["content"]["href"])
    if candidate["sizeBytes"] > p["acquisition"]["maximum_archive_bytes"]:
        raise ValueError("Official archive exceeds byte budget")
    result = dict(
        official_archive_url=url,
        archive_name=candidate["name"],
        bytes=candidate["sizeBytes"],
        md5=checksum["value"],
    )
    write(root / "resolved_source.json", result)
    return result


def resolve():
    p, e = bindings()
    root = ROOT / "resolution"
    root.mkdir(exist_ok=False)
    write(
        root / "STARTED.json", dict(utc=datetime.now(timezone.utc).isoformat(), audio_access=False)
    )
    try:
        resolve_archive(build_opener(OfficialRedirect()), p, root)
    except Exception:
        write(root / "FAILURE.json", dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    print("Official archive metadata resolved;no archive downloaded.")


def acquire():
    p, e = bindings()
    authorize("acquisition")
    capacity(p)
    write(
        ROOT / "ACQUISITION_STARTED.json",
        dict(
            execution_sha256=sha(ROOT / "execution_manifest.json"),
            utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    try:
        source = json.loads((ROOT / "resolution" / "resolved_source.json").read_text())
        if source["md5"] != p["source"]["published_md5"]:
            raise ValueError("Resolved checksum changed")
        with build_opener(OfficialRedirect()).open(
            safe_url(source["official_archive_url"]), timeout=30
        ) as response:
            receipt = download_stream(
                response,
                p["acquisition"]["destination"],
                expected_md5=p["source"]["published_md5"],
                max_bytes=p["acquisition"]["maximum_archive_bytes"],
            )
        write(
            ROOT / "acquisition_receipt.json",
            dict(
                **receipt,
                protocol_sha256=sha(PROTOCOL),
                execution_sha256=sha(ROOT / "execution_manifest.json"),
            ),
        )
    except Exception:
        write(
            ROOT / "ACQUISITION_FAILURE.json",
            dict(traceback=traceback.format_exc(), automatic_retry=False),
        )
        raise


def audit():
    p, e = bindings()
    authorize("support_audit")
    capacity(p)
    receipt = json.loads((ROOT / "acquisition_receipt.json").read_text())
    if receipt["protocol_sha256"] != sha(PROTOCOL) or receipt["execution_sha256"] != sha(
        ROOT / "execution_manifest.json"
    ):
        raise ValueError("Receipt binding changed")
    archive = Path(p["acquisition"]["destination"])
    if hashes(archive)["sha256"] != receipt["sha256"]:
        raise ValueError("Acquired archive changed")
    # Audit directly from ZIP;do not extract paths or write source audio copies.
    result = audit_archive(
        archive,
        ROOT / "support",
        expected_md5=p["source"]["published_md5"],
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
    parser.add_argument("action", choices=["prepare", "resolve", "acquire", "audit"])
    args = parser.parse_args()
    {"prepare": prepare, "resolve": resolve, "acquire": acquire, "audit": audit}[args.action]()
