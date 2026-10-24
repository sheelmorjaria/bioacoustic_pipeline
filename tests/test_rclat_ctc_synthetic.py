"""Synthetic fixtures and actual batch interface, independent of fitting."""

import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path("scripts").resolve()))
from rclat_ctc_synthetic_diagnostic import batcher  # noqa:E402
from bioacoustic_pipeline.representation.rclat_ctc import encode_text  # noqa:E402


def test_all_messages_unique_and_direct_code_reader_recovers_targets():
    c = json.loads(Path("outputs/rclat_ctc_synthetic_diagnostic_v1/cohort.json").read_text())
    assert len(c["records"]) == 96
    assert len({r["text"] for r in c["records"]}) == 96
    for r in c["records"]:
        with np.load(r["fixture_path"], allow_pickle=False) as f:
            target = "".join(
                "ABCD"[
                    int(
                        f["features"][a["start_frame"] : a["stop_frame_exclusive"], :4]
                        .mean(0)
                        .argmax()
                    )
                ]
                for a in r["intervals"]
            )
            assert target == r["text"]
            np.testing.assert_array_equal(f["target_ids"], encode_text(target).numpy())
            assert r["ctc_frames"] >= r["ctc_minimum_frames"]


def test_batch_only_supplies_features_masks_and_lengths_to_model():
    c = json.loads(Path("outputs/rclat_ctc_synthetic_diagnostic_v1/cohort.json").read_text())
    rows = [c["records"][0], c["records"][8]]
    inputs, targets = batcher("cpu")(rows)
    assert len(inputs) == 3
    values, masks, lengths = inputs
    assert lengths.tolist() == [96, 288]
    assert values.shape == masks.shape == (2, 288, 64)
    assert values.dtype == torch.float32 and masks.dtype == torch.bool
    assert not masks[0, 96:].any() and values[0, 96:].eq(0).all()
    assert [len(t) for t in targets] == [4, 12]
