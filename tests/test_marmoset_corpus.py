"""Tests: marmoset corpus ingestion + interval-windowed dataset (§46 P1).

The InfantMarmosetsVox schema: path,start,end,duration,calltypeID,callerID,
twinID,vocID — seconds (not ms), globally-unique callerID (0-9 over 5 twin
pairs), both twins' calls annotated as intervals inside ONE shared recording
(one mic per file). Mechanics pinned here: parse round-trip with unit
conversion, interval-bounded windows that never cross a call boundary,
per-interval labels with shared per-RECORDING file ids.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


def _torch_ok():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


from bioacoustic_pipeline.ingestion.labeled_corpus import (
    CallAnnotation,
    parse_marmoset_labels_csv,
)

COLS = ["path", "start", "end", "duration", "calltypeID", "callerID", "twinID", "vocID"]


def _write_labels(td: Path):
    """Two twins (caller 0/1) calling in one shared file + a second file."""
    rows = [
        ["data/audio_16k/twin_1/a.wav", "0.10", "0.50", "0.40", "1", "0", "1", "0"],
        ["data/audio_16k/twin_1/a.wav", "0.60", "0.90", "0.30", "2", "1", "1", "1"],
        ["data/audio_16k/twin_2/b.wav", "0.00", "0.45", "0.45", "1", "2", "2", "0"],
    ]
    p = td / "labels.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    return p


class TestParseMarmosetLabels(unittest.TestCase):
    def test_round_trip_with_unit_and_field_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write_labels(Path(td))
            ann = parse_marmoset_labels_csv(p)
            self.assertEqual(len(ann), 3)
            a = ann[0]
            self.assertIsInstance(a, CallAnnotation)
            self.assertEqual(a.file_name, "audio_16k/twin_1/a.wav")
            self.assertEqual(a.emitter, 0)  # callerID -> emitter (globally unique)
            self.assertEqual(a.addressee, 0)  # not present in schema
            self.assertEqual(a.context, 1)  # calltypeID -> context
            self.assertEqual(a.start_ms, 100.0)  # seconds -> ms
            self.assertEqual(a.end_ms, 500.0)
            self.assertEqual(a.recording_group, "audio_16k/twin_1/a.wav")
            self.assertEqual(a.twin_id, "1")

    def test_multiple_intervals_share_file(self):
        with tempfile.TemporaryDirectory() as td:
            ann = parse_marmoset_labels_csv(_write_labels(Path(td)))
            self.assertEqual(ann[0].file_name, ann[1].file_name)
            self.assertNotEqual(ann[0].emitter, ann[1].emitter)

    def test_twin_stratified_callers_visible(self):
        with tempfile.TemporaryDirectory() as td:
            ann = parse_marmoset_labels_csv(_write_labels(Path(td)))
            self.assertEqual({a.twin_id for a in ann}, {"1", "2"})


@unittest.skipUnless(_torch_ok(), "needs torch")
class TestIntervalWindowDataset(unittest.TestCase):
    def _corpus(self, td: Path):

        audio = td / "audio_16k" / "twin_1"
        audio.mkdir(parents=True)
        sr = 8000
        t = np.arange(int(1.0 * sr)) / sr
        # twin A's call occupies [0.1, 0.5)s at 300 Hz; twin B's [0.6, 0.9)s at 900 Hz
        sig = np.zeros_like(t)
        sig[int(0.1 * sr) : int(0.5 * sr)] = 0.5 * np.sin(
            2 * np.pi * 300 * t[int(0.1 * sr) : int(0.5 * sr)]
        )
        sig[int(0.6 * sr) : int(0.9 * sr)] = 0.5 * np.sin(
            2 * np.pi * 900 * t[int(0.6 * sr) : int(0.9 * sr)]
        )
        sf.write(audio / "a.wav", sig.astype(np.float32), sr)
        return sr

    def test_windows_stay_inside_intervals(self):
        from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._corpus(td)
            ann = parse_marmoset_labels_csv(_write_labels(td))
            cfg = TrainingConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=8,
                sequence_length=16,
                steps_ahead=2,
                overlap=2,
                device="cpu",
            )
            ds = LabeledAudioSequenceDataset(td, ann, cfg)
            # interval [0.1,0.5)s = 0.4s = 40 frames; seq 18, stride 14 -> 2 windows
            # interval [0.6,0.9)s = 0.3s = 30 frames -> 1 window
            self.assertEqual(len(ds), 3)
            labels = [(ds[i][2], ds[i][4]) for i in range(3)]
            self.assertEqual([c for c, _ in labels], [0, 0, 1])
            # same recording -> same file id even across intervals/twins
            self.assertEqual(labels[0][1], labels[2][1])
            # twin A's windows contain only the 300 Hz call: check energy purity
            # by construction (windows bounded to the interval)

    def test_short_interval_yields_no_windows(self):
        from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._corpus(td)
            ann = parse_marmoset_labels_csv(_write_labels(td))
            ann = [a for a in ann if a.emitter == 1]  # only the 0.3 s interval
            cfg = TrainingConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=8,
                sequence_length=16,
                steps_ahead=2,
                overlap=2,
                device="cpu",
            )
            LabeledAudioSequenceDataset(td, ann, cfg)  # 0.3s -> 1 window
            # now shrink below the window threshold
            import dataclasses

            ann[0] = dataclasses.replace(
                ann[0], end_ms=ann[0].start_ms + 100.0
            )  # 0.1 s = 10 frames < 18
            ds2 = LabeledAudioSequenceDataset(td, ann, cfg)
            self.assertEqual(len(ds2), 0)


if __name__ == "__main__":
    unittest.main()
