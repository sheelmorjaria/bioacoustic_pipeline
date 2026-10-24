"""Metadata-only RTPE source support and explicit waveform availability arithmetic."""

import hashlib
import re


def partition_metadata(rows, *, train_speakers, validation_speakers):
    """Draft partition; source speaker labels are used, not independently re-certified.

    No transcript, outcome, feature cache, audio, or model checkpoint is read.
    Reject exact metadata hash duplicates rather than choose exclusions after seeing scores.
    """
    keys = (
        "uid",
        "speaker",
        "path",
        "samples",
        "feature_frames",
        "source_sha256",
        "decoded_pcm16_sha256",
    )
    clean = [{k: row[k] for k in keys} for row in rows]
    for field in ("uid", "path", "source_sha256", "decoded_pcm16_sha256"):
        if len({r[field] for r in clean}) != len(clean):
            raise ValueError(f"Duplicate {field}")
    for row in clean:
        parts = row["uid"].split("-")
        if len(parts) != 3 or parts[0] != row["speaker"]:
            raise ValueError("Source speaker/UID identity conflict")
        if any(not re.fullmatch("[a-fA-F0-9]{64}", row[k]) for k in keys[-2:]):
            raise ValueError("Recorded source hashes required")
        if type(row["samples"]) is not int or row["samples"] < 640:
            raise ValueError("Native 16kHz sample-count support required")
        if row["feature_frames"] != 1 + (row["samples"] - 640) // 160:
            raise ValueError("Recorded feature count disagrees with timing contract")
    speakers = sorted(
        {r["speaker"] for r in clean},
        key=lambda s: hashlib.sha256(("RTPE-1-support-v1:speaker:" + s).encode()).hexdigest(),
    )
    if (
        type(train_speakers) is not int
        or type(validation_speakers) is not int
        or min(train_speakers, validation_speakers) <= 0
        or train_speakers + validation_speakers >= len(speakers)
    ):
        raise ValueError("Nonempty disjoint train, validation and evaluation groups required")
    split = {
        s: "train"
        if i < train_speakers
        else ("validation" if i < train_speakers + validation_speakers else "test")
        for i, s in enumerate(speakers)
    }
    return [
        {
            **r,
            "recording": r["uid"],
            "individual": r["speaker"],
            "parent_recording_group": "-".join(r["uid"].split("-")[:2]),
            "split": split[r["speaker"]],
        }
        for r in sorted(clean, key=lambda r: r["uid"])
    ]


def timing_support(samples, *, history, horizons):
    """16kHz, 640-sample trailing frame, 160-sample hop; exclusive sample ends.

    Each input frame j uses [160*j,160*j+640). Its conservative latest
    consumed sample is end-1, available at end/16000 seconds. Targets start
    no earlier than origin end. Counts are metadata capacity upper bounds;
    current waveform finiteness and cached validity are NOT verified here.
    """
    horizons = tuple(horizons)
    if (
        type(samples) is not int
        or samples < 0
        or type(history) is not int
        or history <= 0
        or not horizons
        or any(type(h) is not int or h <= 0 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError(
            "Nonnegative samples, positive history and unique positive horizons required"
        )
    gaps = [h * 160 - 640 for h in horizons]
    if min(gaps) < 0:
        raise ValueError("Future target waveform would overlap observed waveform")
    frames = max(0, 1 + (samples - 640) // 160)
    n = max(0, frames - history - max(horizons) + 1)
    first = (history - 1) * 160 + 640 if n else None
    last = (frames - 1 - max(horizons)) * 160 + 640 if n else None
    return dict(
        frames=frames,
        common_origins_upper_bound=n,
        first_origin_available_sample_exclusive=first,
        last_origin_available_sample_exclusive=last,
        target_start_minus_origin_end_samples=gaps,
        latest_target_end_sample_exclusive=last + max(horizons) * 160 if n else None,
    )
