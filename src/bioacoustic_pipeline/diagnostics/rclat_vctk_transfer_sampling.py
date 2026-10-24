"""Performance-blind protected metadata sampling;no feature or forecast access."""

from collections import defaultdict
import hashlib
import json
import numpy as np


FIELDS = (
    "speaker",
    "event",
    "member",
    "sample_rate",
    "channels",
    "frames",
    "source_bytes_sha256",
    "decoded_pcm_sha256",
    "mel_sha256",
    "valid_sha256",
    "eligible_origin_count",
    "eligible_origin_sha256",
    "first_eligible_origin",
    "last_eligible_origin",
)


def rank(salt, *parts):
    return hashlib.sha256(
        json.dumps([salt, *parts], separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def select_inventory(rows, *, salt, recordings, origins):
    if (
        not isinstance(salt, str)
        or not salt
        or type(recordings) is not int
        or type(origins) is not int
        or min(recordings, origins) <= 0
    ):
        raise ValueError("Explicit salt and positive integer sample sizes required")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row["event"] in seen:
            raise ValueError("Duplicate recording event")
        seen.add(row["event"])
        if row["reason"] != "eligible":
            continue
        supported = np.arange(
            row["first_eligible_origin"], row["last_eligible_origin"] + 1, dtype="<i8"
        )
        if (
            len(supported) != row["eligible_origin_count"]
            or hashlib.sha256(supported.tobytes()).hexdigest() != row["eligible_origin_sha256"]
        ):
            raise ValueError(
                "Verified eligible-origin hash changed or noncontiguous support unsupported"
            )
        groups[row["speaker"]].append(row)
    if not groups:
        raise ValueError("No eligible speaker support")
    selected = []
    for speaker, group in sorted(groups.items()):
        candidates = [r for r in group if r["eligible_origin_count"] >= origins]
        if len(candidates) < recordings:
            raise ValueError("Insufficient fixed per-speaker recording/origin support")
        ordered = sorted(
            candidates, key=lambda r: (rank(salt, "recording", speaker, r["event"]), r["event"])
        )[:recordings]
        for row in sorted(ordered, key=lambda r: r["event"]):
            supported = range(row["first_eligible_origin"], row["last_eligible_origin"] + 1)
            chosen = sorted(
                sorted(
                    supported, key=lambda o: (rank(salt, "origin", speaker, row["event"], o), o)
                )[:origins]
            )
            record = {key: row[key] for key in FIELDS if key in row}
            record.update(
                selected_origins=chosen,
                recording_weight=1 / (len(groups) * recordings),
                origin_weight=1 / (len(groups) * recordings * origins),
            )
            selected.append(record)
    return selected
