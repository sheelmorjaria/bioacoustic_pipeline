"""Prepare fixed diagnostic subsets; no checkpoint loading/inference/training."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import rclat_p10_train as original
from bioacoustic_pipeline.representation.rclat_ctc import encode_text

OUT = Path("outputs/rclat_ctc_learning_diagnostic_v1")
SALT = "RCLAT-CTC-DIAG-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def main():
    source = Path("outputs/rclat_p10/cohort.json")
    cohort = json.loads(source.read_text())
    selected = {}
    for name, n in [("selected_training", 16), ("selected_validation", 8)]:
        grouped = defaultdict(list)
        for row in cohort[name]:
            grouped[row["speaker"]].append(row)
        speakers = sorted(
            [s for s, rows in grouped.items() if len(rows) >= 4],
            key=lambda s: key(name + ":speaker:" + s),
        )[:n]
        rows = []
        for speaker in speakers:
            ordered = sorted(grouped[speaker], key=lambda r: (r["duration_seconds"], r["uid"]))
            for i, indices in enumerate(np.array_split(np.arange(len(ordered)), 4)):
                choices = [ordered[int(j)] for j in indices]
                row = min(choices, key=lambda r: key(name + f":quartile:{i}:" + r["uid"]))
                features, mask = original.sequence(row["uid"], "mel")
                target = encode_text(row["transcript"])
                minimum = len(target) + int((target[1:] == target[:-1]).sum())
                assert (
                    features.shape == mask.shape
                    and features.shape[1] == 64
                    and (len(features) + 3) // 4 >= minimum
                )
                rows.append(
                    {
                        **row,
                        "diagnostic_duration_quartile": i,
                        "cache_path": str(original.CACHE / f"{row['uid']}.npz"),
                        "cache_sha256": sha(original.CACHE / f"{row['uid']}.npz"),
                        "actual_feature_frames": len(features),
                        "actual_ctc_frames": (len(features) + 3) // 4,
                        "actual_target_length": len(target),
                        "actual_minimum_ctc_frames": minimum,
                    }
                )
        selected[name] = sorted(rows, key=lambda r: r["uid"])
    train = selected["selected_training"]
    val = selected["selected_validation"]
    assert len(train) == 64 and len(val) == 32
    assert not ({r["speaker"] for r in train} & {r["speaker"] for r in val})
    assert not (
        {r["decoded_pcm16_sha256"] for r in train} & {r["decoded_pcm16_sha256"] for r in val}
    )
    OUT.mkdir(exist_ok=True)
    result = dict(
        salt=SALT,
        source_cohort_sha256=sha(source),
        selection="Hash-select16eligibletraining/8validationspeakers;four durationquartilesperspeaker;hash-selectoneUIDperquartile;minimum4sources/speaker",
        training=train,
        validation=val,
        training_speakers=16,
        validation_speakers=8,
        training_hours=sum(r["duration_seconds"] for r in train) / 3600,
        validation_hours=sum(r["duration_seconds"] for r in val) / 3600,
        all_ctc_lengths_verified=True,
        speaker_disjoint=True,
        decoded_audio_overlap=False,
        inference_or_training_performed=False,
    )
    with (OUT / "cohort.json").open("x") as f:
        json.dump(result, f, indent=2)
    print(
        "Prepared64training/32validation utterances;no inference or training.",
        result["training_hours"],
        result["validation_hours"],
    )


if __name__ == "__main__":
    main()
