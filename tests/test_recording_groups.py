"""Tests: interval-aware contract + recording-group splits (§45 addendum).

The bat corpus is per-bout (no start/end columns — verified against the
archive.org release), so the interval contract collapses to whole-file bouts
by default and recording groups default to the file itself. These tests pin:
optional interval fields round-trip, interval-bounded window guards, and
group-level splits that keep recording groups out of two splits at once.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.ingestion.labeled_corpus import (
    CallAnnotation,
    interval_annotation,
    split_recording_groups_by_caller,
)


def _annos(groups_per_caller=2, callers=3, files_per_group=4):
    out = []
    for c in range(1, callers + 1):
        for g in range(groups_per_caller):
            for f in range(files_per_group):
                out.append(
                    CallAnnotation(
                        file_name=f"c{c}_g{g}_f{f}.wav",
                        emitter=c,
                        addressee=1,
                        context=1 + (f % 2),
                        emit_pre=0,
                        addr_pre=0,
                        emit_post=0,
                        addr_post=0,
                        start_ms=None,
                        end_ms=None,
                        recording_group=f"sess_{g}",
                    )
                )
    return out


class TestIntervalContract(unittest.TestCase):
    def test_default_annotation_is_whole_file(self):
        a = CallAnnotation("x.wav", 1, 2, 3, 0, 0, 0, 0)
        iv = interval_annotation(a, duration_ms=1500.0)
        self.assertEqual(iv.start_ms, 0.0)
        self.assertEqual(iv.end_ms, 1500.0)
        self.assertEqual(iv.recording_group, "x.wav")  # group defaults to file

    def test_explicit_interval_round_trips(self):
        a = CallAnnotation(
            "x.wav", 1, 2, 3, 0, 0, 0, 0, start_ms=100.0, end_ms=900.0, recording_group="sess_7"
        )
        iv = interval_annotation(a, duration_ms=1500.0)
        self.assertEqual((iv.start_ms, iv.end_ms), (100.0, 900.0))
        self.assertEqual(iv.recording_group, "sess_7")


class TestRecordingGroupSplits(unittest.TestCase):
    def test_groups_never_span_splits(self):
        annos = _annos(groups_per_caller=4, callers=5, files_per_group=4)
        split = split_recording_groups_by_caller(annos, seed=0)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            ga = {x.recording_group for x in split[a]}
            gb = {x.recording_group for x in split[b]}
            self.assertEqual(len(ga & gb), 0, f"{a}/{b} share recording groups")

    def test_every_annotation_in_exactly_one_split(self):
        annos = _annos(groups_per_caller=3, callers=4, files_per_group=3)
        split = split_recording_groups_by_caller(annos, seed=0)
        total = sum(len(split[k]) for k in ("train", "val", "test"))
        self.assertEqual(total, len(annos))

    def test_caller_stratification_when_groups_allow(self):
        # with several groups per caller, every caller reaches train
        annos = _annos(groups_per_caller=4, callers=5, files_per_group=3)
        split = split_recording_groups_by_caller(annos, seed=0)
        train_callers = {a.emitter for a in split["train"]}
        self.assertEqual(train_callers, set(range(1, 6)))

    def test_deterministic(self):
        annos = _annos()
        a = split_recording_groups_by_caller(annos, seed=3)
        b = split_recording_groups_by_caller(annos, seed=3)
        self.assertEqual(
            sorted(x.file_name for x in a["train"]), sorted(x.file_name for x in b["train"])
        )


class TestGroupProxy(unittest.TestCase):
    def test_filename_block_grouping(self):
        from bioacoustic_pipeline.ingestion.labeled_corpus import filename_block_group

        self.assertEqual(filename_block_group("0.wav", block=100), "block_0")
        self.assertEqual(filename_block_group("91079.wav", block=100), "block_910")
        self.assertEqual(filename_block_group("c3_g1_f2.wav", block=100), "c3_g1_f2.wav")


if __name__ == "__main__":
    unittest.main()
