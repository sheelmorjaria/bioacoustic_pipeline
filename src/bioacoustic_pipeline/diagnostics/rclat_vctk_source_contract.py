"""Metadata contracts for future archive support auditing; no audio access."""

from pathlib import PurePosixPath
import re
import stat


def validate_archive_members(members):
    seen = set()
    total = 0
    for item in members:
        name = item.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or ":" in name
            or path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(item.external_attr >> 16)
        ):
            raise ValueError("Unsafe archive member")
        normalized = str(path)
        if normalized in seen:
            raise ValueError("Duplicate archive member")
        seen.add(normalized)
        total += item.file_size
        if total > 64 * 2**30:
            raise ValueError("Frozen expanded archive budget exceeded")
    return dict(members=len(seen), expanded_bytes=total)


def recording_identity(name):
    path = PurePosixPath(name)
    match = re.fullmatch(r"(p\d+)_(\d+)_(mic[12])\.flac", path.name)
    if match is None:
        raise ValueError("Unexpected audio identifier;requires independent audit")
    speaker, utterance, microphone = match.groups()
    if path.parent.name != speaker:
        raise ValueError("Speaker directory/name mismatch")
    return dict(
        speaker=speaker,
        utterance=utterance,
        event=f"{speaker}_{utterance}",
        microphone=microphone,
        selected_microphone=microphone == "mic2",
    )
