#!/usr/bin/env python3
"""Stage 2: frozen motif discovery on the bat C11-vs-C12 contrast (§46).

Pipeline: Stage-1 eligibility filter (28 callers @ 20 bouts/side) ->
recording-group split -> 44-D frame-fingerprint trajectories (§44 retune:
WIN 40 ms / HOP 10 ms; discovery subsample 40 bouts/caller/context, capped)
-> FrozenDUSTEDModel fit on TRAIN frames ONLY -> frozen vocabulary saved ->
assignment stats on the eval split against the FROZEN vocabulary.

Artifacts: cache/stage2/bat_c11c12_traj.npz (resumable extraction),
outputs/stage2_frozen_vocab.npz, outputs/stage2_frozen_motifs.json.
Framing: measurement of a fixed acoustic vocabulary — NOT meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WIN_MS, HOP_MS, MAX_FRAMES = 40.0, 10.0, 300
CAP_PER_CELL = 40
N_CLUSTERS = 24


def main() -> None:
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.ingestion.context_eligibility import (
        build_caller_context_support_matrix,
    )
    from bioacoustic_pipeline.ingestion.frozen_dusted import (
        FrozenDUSTEDModel,
        filter_eligible_intervals,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        parse_annotations_csv,
        split_recording_groups_by_caller,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    data_dir = ROOT / "data" / "egyptian_fruit_bats"
    annotations = [a for a in parse_annotations_csv(data_dir / "annotations.csv") if a.emitter > 0]
    matrix = build_caller_context_support_matrix(annotations)
    eligible = filter_eligible_intervals(
        annotations, contexts=(11, 12), min_bouts=20, eligibility_matrix=matrix
    )
    split = split_recording_groups_by_caller(eligible, seed=0)
    callers = sorted({a.emitter for a in eligible})
    print(f"eligible callers: {len(callers)} | bouts: {len(eligible)}", flush=True)

    # discovery subsample: cap bouts per caller x context (documented)
    import random

    rng = random.Random(0)
    per_cell: dict = {}
    train_bouts, eval_bouts = [], []
    for part, annos in (("train", split["train"]), ("eval", split["train"][:0] + split["test"])):
        for a in annos:
            key = (a.emitter, a.context)
            if part == "train":
                per_cell.setdefault(key, []).append(a)
                if len(per_cell[key]) <= CAP_PER_CELL:
                    train_bouts.append(a)
            else:
                eval_bouts.append(a)
    rng.shuffle(eval_bouts)
    eval_bouts = eval_bouts[: CAP_PER_CELL * len(callers)]  # same-scale eval sample
    print(f"discovery bouts: {len(train_bouts)} | eval bouts: {len(eval_bouts)}", flush=True)

    cache = ROOT / "cache" / "stage2" / "bat_c11c12_traj.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        frames, offsets, names, parts = (
            data["frames"],
            data["offsets"],
            list(data["names"]),
            data["parts"],
        )
        print(f"trajectories from cache: {len(names)} bouts", flush=True)
    else:
        frames, offsets, names, parts = [], [0], [], []
        for label, bouts in (("train", train_bouts), ("eval", eval_bouts)):
            for k, a in enumerate(bouts):
                audio, sr = load_audio(str(data_dir / "audio" / a.file_name), target_sr=250000)
                win, hop = int(sr * WIN_MS / 1000), int(sr * HOP_MS / 1000)
                rows = []
                i = 0
                while i + win <= len(audio) and len(rows) < MAX_FRAMES:
                    v = np.asarray(
                        afp.extract(audio[i : i + win], sr, species="bat").combined,
                        dtype=np.float32,
                    )
                    rows.append(v)
                    i += hop
                if not rows:
                    continue
                frames.extend(rows)
                offsets.append(offsets[-1] + len(rows))
                names.append(a.file_name)
                parts.append(0 if label == "train" else 1)
                if (k + 1) % 200 == 0:
                    print(f"  {label}: {k + 1}/{len(bouts)} bouts", flush=True)
        frames = np.asarray(frames, dtype=np.float32)
        offsets = np.asarray(offsets, dtype=np.int64)
        parts = np.asarray(parts, dtype=np.int8)
        np.savez_compressed(
            cache, frames=frames, offsets=offsets, names=np.asarray(names), parts=parts
        )
        print(f"extracted {frames.shape[0]} frames from {len(names)} bouts", flush=True)

    # parts is per-BOUT; expand to per-FRAME via the offsets
    frame_parts = np.repeat(parts, np.diff(offsets))
    is_train = frame_parts == 0
    train_frames = frames[is_train]
    eval_mask = frame_parts == 1
    print(
        f"train frames: {train_frames.shape[0]} | eval frames: {int(eval_mask.sum())}", flush=True
    )

    model = FrozenDUSTEDModel(n_clusters=N_CLUSTERS, random_state=42)
    model.fit(train_frames)
    vocab_path = ROOT / "outputs" / "stage2_frozen_vocab.npz"
    model.save(vocab_path)

    eval_labels = model.assign(frames[eval_mask])
    train_labels = model.assign(train_frames)
    usage = np.bincount(eval_labels, minlength=N_CLUSTERS) / max(eval_labels.sum(), 1)
    train_usage = np.bincount(train_labels, minlength=N_CLUSTERS) / max(train_labels.sum(), 1)
    payload = {
        "eligible_callers": len(callers),
        "contexts": [11, 12],
        "min_bouts": 20,
        "n_train_frames": int(train_frames.shape[0]),
        "n_eval_frames": int(eval_mask.sum()),
        "n_clusters": N_CLUSTERS,
        "eval_usage_histogram": [round(float(u), 5) for u in usage],
        "train_usage_histogram": [round(float(u), 5) for u in train_usage],
        "eval_vocab_entropy": float(-(usage[usage > 0] * np.log(usage[usage > 0])).sum()),
        "framing": "frozen acoustic vocabulary measurement — NOT meaning",
    }
    out = ROOT / "outputs" / "stage2_frozen_motifs.json"
    out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {k: payload[k] for k in ("n_train_frames", "n_eval_frames", "eval_vocab_entropy")}
        ),
        flush=True,
    )
    print(f"vocabulary -> {vocab_path}", flush=True)
    print("STAGE2 DONE", flush=True)


if __name__ == "__main__":
    main()
