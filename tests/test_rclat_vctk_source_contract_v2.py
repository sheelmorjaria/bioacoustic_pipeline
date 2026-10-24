import hashlib
import json
from pathlib import Path
import pytest
from bioacoustic_pipeline.diagnostics.rclat_vctk_source_contract_v2 import (
    parse_speaker_roster,
    recording_identity,
)


def roster(raw=b"ID AGE GENDER\np225 23 F\ns5 22 F\n"):
    return parse_speaker_roster(raw, hashlib.sha256(raw).hexdigest())


def test_p_and_s5_share_strict_event_contract():
    ids = roster()
    assert ids == frozenset({"p225", "s5"})
    for speaker in ids:
        a = recording_identity(f"root/{speaker}/{speaker}_001_mic1.flac", ids)
        b = recording_identity(f"root/{speaker}/{speaker}_001_mic2.flac", ids)
        assert (
            a["event"] == b["event"] and b["selected_microphone"] and not a["selected_microphone"]
        )
    for name in [
        "p225/s5_001_mic2.flac",
        "s6/s6_001_mic2.flac",
        "p999/p999_001_mic2.flac",
        "s5/s5_bad_mic2.flac",
        "s5/s5_001_mic3.flac",
    ]:
        with pytest.raises(ValueError):
            recording_identity(name, ids)


def test_roster_integrity_duplicates_and_unexpected_schema_rejected():
    with pytest.raises(ValueError, match="hash"):
        parse_speaker_roster(b"ID\ns5", "wrong")
    for raw in [b"ID AGE\ns5 22\ns5 22\n", b"ID AGE\ns6 22\n", b"wrong header\ns5 22\n"]:
        with pytest.raises(ValueError):
            roster(raw)


def test_all_796_offending_names_are_roster_recognised_not_eligibility_certified():
    raw = Path(
        "outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/speaker-info.snapshot.txt"
    ).read_bytes()
    ids = roster(raw)
    rows = json.loads(
        Path(
            "outputs/rclat_rtpe_vctk_identifier_diagnostic_v1/unrecognized_members.json"
        ).read_text()
    )
    parsed = [recording_identity(row["member"], ids) for row in rows]
    assert len(parsed) == 796 and {x["speaker"] for x in parsed} == {"s5"}
    assert len({x["event"] for x in parsed}) == 398
    assert all("eligible_origin_count" not in x for x in parsed)
