"""Roster-backed correction;valid s5 explicit,unknown namespaces still blocked."""

import hashlib
from pathlib import PurePosixPath
import re

from .rclat_vctk_source_contract import validate_archive_members  # noqa:F401


def parse_speaker_roster(raw, expected_sha256):
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Official speaker roster hash differs")
    lines = [line.split() for line in raw.decode("utf8").splitlines() if line.strip()]
    if not lines or lines[0][0] != "ID":
        raise ValueError("Unexpected speaker roster header")
    identifiers = [fields[0] for fields in lines[1:]]
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(re.fullmatch(r"(?:p\d+|s5)", s) is None for s in identifiers)
    ):
        raise ValueError("Duplicate or unsupported speaker roster schema")
    return frozenset(identifiers)


def recording_identity(name, roster_ids):
    path = PurePosixPath(name)
    match = re.fullmatch(r"((?:p\d+|s5))_(\d+)_(mic[12])\.flac", path.name)
    if match is None:
        raise ValueError("Unexpected audio identifier: " + name)
    speaker, utterance, microphone = match.groups()
    if speaker not in roster_ids:
        raise ValueError("Unrostered speaker: " + speaker)
    if path.parent.name != speaker:
        raise ValueError("Speaker directory/name mismatch: " + name)
    return dict(
        speaker=speaker,
        utterance=utterance,
        event=f"{speaker}_{utterance}",
        microphone=microphone,
        selected_microphone=microphone == "mic2",
    )
