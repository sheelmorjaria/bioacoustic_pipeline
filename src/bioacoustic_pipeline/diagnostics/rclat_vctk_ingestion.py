"""Source-only VCTK ingestion. No model/scaler/checkpoint/scoring interfaces."""

import hashlib
import io
import json
import os
from pathlib import Path
import traceback
from urllib.parse import urlparse
import zipfile
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
from .rclat_vctk_source_contract import validate_archive_members, recording_identity
from ..representation.rclat_causal_resample_v2 import causal_decimate_v2, mel_features_v2


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def hashes(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest(), bytes=size)


def safe_url(url):
    p = urlparse(url)
    if (
        p.scheme != "https"
        or p.hostname != "datashare.ed.ac.uk"
        or p.username
        or p.password
        or p.port not in (None, 443)
    ):
        raise ValueError("Only official HTTPS Edinburgh source URLs allowed")
    return url


def download_stream(response, destination, *, expected_md5, max_bytes):
    dest = Path(destination)
    partial = dest.with_name(dest.name + ".partial")
    if dest.exists() or partial.exists():
        raise FileExistsError("Existing acquisition preserved;no retry/resume")
    safe_url(response.geturl())
    if int(response.headers.get("Content-Length", "0")) > max_bytes:
        raise ValueError("Archive size limit")
    dest.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    size = 0
    with partial.open("xb") as f:
        while True:
            chunk = response.read(2**20)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Archive size limit")
            md5.update(chunk)
            sha.update(chunk)
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    if md5.hexdigest() != expected_md5:
        raise ValueError("Archive checksum mismatch")
    os.link(partial, dest)
    partial.unlink()
    return dict(
        path=str(dest),
        bytes=size,
        md5=md5.hexdigest(),
        sha256=sha.hexdigest(),
        final_url=response.geturl(),
        headers=dict(response.headers),
    )


def waveform_support(payload):
    info = sf.info(io.BytesIO(payload))
    header = dict(
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        subtype=info.subtype,
        format=info.format,
    )
    if info.samplerate != 48000:
        return dict(**header, reason="unsupported_sample_rate")
    if info.channels != 1:
        return dict(**header, reason="unsupported_channels")
    audio, rate = sf.read(io.BytesIO(payload), dtype="float64", always_2d=False)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        return dict(**header, reason="invalid_waveform")
    if len(audio) != info.frames:
        raise ValueError("Decoded framecount differs from header")
    pcm = hashlib.sha256(
        json.dumps([rate, info.channels], separators=(",", ":")).encode()
        + np.asarray(audio, dtype="<f8").tobytes()
    ).hexdigest()
    base = dict(**header, decoded_pcm_sha256=pcm, duration_seconds=len(audio) / rate)
    if (len(audio) + 2) // 3 < 640:
        return dict(**base, reason="insufficient_duration")
    features = mel_features_v2(audio, rate)
    mel = features["mel"]
    valid = features["mel_valid"]
    if not np.isfinite(np.where(valid, mel, 0)).all():
        return dict(**base, reason="invalid_features")
    frame = valid.all(-1)
    candidates = np.arange(32, max(32, len(frame) - 16), dtype=np.int64)
    eligible = np.array(
        [
            o
            for o in candidates
            if frame[o - 31 : o + 1].all()
            and frame[o + 8]
            and frame[o + 16]
            and features["source_first_samples"][o + 8] > features["source_last_samples"][o]
            and features["source_first_samples"][o + 16] > features["source_last_samples"][o]
        ],
        dtype=np.int64,
    )
    return dict(
        **base,
        reason="eligible" if len(eligible) else "no_supported_origins",
        mel_frames=len(mel),
        unsupported_mel_frames=int((~frame).sum()),
        mel_sha256=hashlib.sha256(mel.tobytes()).hexdigest(),
        valid_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
        eligible_origin_count=len(eligible),
        eligible_origin_sha256=hashlib.sha256(eligible.astype("<i8").tobytes()).hexdigest(),
        first_eligible_origin=int(eligible[0]) if len(eligible) else None,
        last_eligible_origin=int(eligible[-1]) if len(eligible) else None,
    )


def audit_archive(
    archive,
    directory,
    *,
    expected_md5,
    coefficient_path=None,
    max_archive_bytes=20 * 2**30,
    integrity_check=None,
    max_member_bytes=256 * 2**20,
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "STARTED.json",
        dict(
            archive=str(archive),
            expected_md5=expected_md5,
            scope="Source support only;no scientific scoring",
        ),
    )
    try:
        receipt = hashes(archive)
        if receipt["bytes"] > max_archive_bytes:
            raise ValueError("Archive size limit")
        if receipt["md5"] != expected_md5:
            raise ValueError("Archive checksum mismatch")
        write(root / "archive_receipt.json", receipt)
        if coefficient_path:
            with np.load(coefficient_path, allow_pickle=False) as f:
                expected = f["taps"]
            assert np.array_equal(causal_decimate_v2(np.zeros(600), 48000)["taps"], expected), (
                "Certified FIR coefficients differ"
            )
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            metadata = validate_archive_members(members)
            files = {info.filename: info for info in members if not info.is_dir()}
            if any(info.file_size > max_member_bytes for info in files.values()):
                raise MemoryError(
                    "Frozen per-member memory budget exceeded;no silent source exclusion"
                )
            for name in files:
                parent = Path(name).parent
                while str(parent) not in (".", "/"):
                    if str(parent) in files:
                        raise ValueError("Archive file/directory collision")
                    parent = parent.parent
            events = defaultdict(dict)
            for name, info in sorted(files.items()):
                if not name.lower().endswith(".flac"):
                    continue
                identity = recording_identity(name)
                event = identity["event"]
                mic = identity["microphone"]
                if mic in events[event]:
                    raise ValueError("Ambiguous duplicate microphone/event")
                events[event][mic] = dict(member=name, **identity)
            write(
                root / "metadata_inventory.json",
                dict(
                    archive=metadata,
                    events=[
                        dict(event=event, recordings=recordings)
                        for event, recordings in sorted(events.items())
                    ],
                    microphone_suffix_verified=bool(events),
                    physical_mapping_scope="Documented upstream TFDS mapping;file suffix/layout validated;no acoustic microphone identification",
                ),
            )
            if integrity_check:
                integrity_check()
            rows = []
            with (root / "waveform_journal.jsonl").open("x", buffering=1) as journal:
                for event, recordings in sorted(events.items()):
                    if integrity_check:
                        integrity_check()
                    if "mic2" not in recordings:
                        row = dict(event=event, speaker=event.split("_")[0], reason="mic2_absent")
                    else:
                        row = dict(recordings["mic2"])
                        try:
                            payload = z.read(row["member"])
                            row["source_bytes_sha256"] = hashlib.sha256(payload).hexdigest()
                            row.update(waveform_support(payload))
                        except (sf.LibsndfileError, ValueError) as e:
                            row.update(reason="decode_or_support_error", error=str(e))
                    rows.append(row)
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
        duplicates = defaultdict(list)
        for row in rows:
            if "decoded_pcm_sha256" in row:
                duplicates[row["decoded_pcm_sha256"]].append(row)
        groups = []
        for digest, group in sorted(duplicates.items()):
            if len(group) < 2:
                continue
            cross = len({row["speaker"] for row in group}) > 1
            groups.append(
                dict(
                    decoded_pcm_sha256=digest,
                    events=[row["event"] for row in group],
                    cross_speaker=cross,
                )
            )
            for index, row in enumerate(sorted(group, key=lambda r: r["event"])):
                if cross or index > 0:
                    row["reason_before_duplicate"] = row["reason"]
                    row["reason"] = (
                        "cross_speaker_duplicate" if cross else "within_speaker_duplicate"
                    )
        with (root / "recordings.jsonl").open("x") as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        retained = [row for row in rows if row["reason"] == "eligible"]
        counts = dict(
            recording_events=len(rows),
            retained_recordings=len(retained),
            retained_speakers=len({row["speaker"] for row in retained}),
            eligible_origins=sum(row["eligible_origin_count"] for row in retained),
            exclusion_reasons=dict(
                sorted(
                    Counter(row["reason"] for row in rows if row["reason"] != "eligible").items()
                )
            ),
            sample_rate_counts=dict(
                Counter(str(row["sample_rate"]) for row in rows if "sample_rate" in row)
            ),
            channel_counts=dict(Counter(str(row["channels"]) for row in rows if "channels" in row)),
        )
        speaker_support = {}
        for row in retained:
            s = speaker_support.setdefault(row["speaker"], dict(recordings=0, eligible_origins=0))
            s["recordings"] += 1
            s["eligible_origins"] += row["eligible_origin_count"]
        write(root / "duplicate_groups.json", groups)
        result = dict(
            completed=True,
            counts=counts,
            speaker_support=speaker_support,
            source_support_threshold_applied=False,
            protected_origins_selected=False,
            scientific_sampling_frozen=False,
            model_inference=False,
            scientific_scoring=False,
            global_speaker_identity_certified=False,
        )
        if integrity_check:
            integrity_check()
        write(root / "support_summary.json", result)
        return result
    except Exception:
        write(
            root / "FAILURE.json",
            dict(traceback=traceback.format_exc(), automatic_retry=False, model_inference=False),
        )
        raise
