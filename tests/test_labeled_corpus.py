"""Tests: labeled-corpus ingestion + caller-aware splits (ingestion.labeled_corpus).

Synthetic annotations CSVs via tempfile; the exact bat column schema. Mechanics
only: parse round-trip, split disjointness/coverage, held-out-caller exclusion,
manifest population. No torch, no network.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from bioacoustic_pipeline.ingestion.labeled_corpus import (
    CallAnnotation,
    annotation_metadata_fn,
    build_labeled_manifest,
    holdout_callers,
    parse_annotations_csv,
    split_files_by_caller,
    summarize_callers,
)

COLUMNS = [
    "File Name",
    "Emitter",
    "Addressee",
    "Context",
    "Emitter pre-vocalization action",
    "Addressee pre-vocalization action",
    "Emitter post-vocalization action",
    "Addressee post-vocalization action",
]


def _write_csv(rows: list[dict]) -> Path:
    fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    with fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return Path(fh.name)


def _fixture_rows(n_callers=10, files_per_caller=10):
    rows = []
    for c in range(1, n_callers + 1):
        for f in range(files_per_caller):
            rows.append(
                {
                    "File Name": f"{1000 + c * 100 + f}.wav",
                    "Emitter": str(c),
                    "Addressee": str((c % n_callers) + 1),
                    "Context": str(1 + (f % 5)),
                    "Emitter pre-vocalization action": "0",
                    "Addressee pre-vocalization action": "1",
                    "Emitter post-vocalization action": "2",
                    "Addressee post-vocalization action": "3",
                }
            )
    return rows


class TestParseAnnotationsCsv(unittest.TestCase):
    def test_round_trip(self):
        rows = _fixture_rows(n_callers=3, files_per_caller=2)
        path = _write_csv(rows)
        ann = parse_annotations_csv(path)
        self.assertEqual(len(ann), 6)
        first = ann[0]
        self.assertIsInstance(first, CallAnnotation)
        self.assertEqual(first.file_name, "1100.wav")
        self.assertEqual(first.emitter, 1)
        self.assertEqual(first.addressee, 2)
        self.assertEqual(first.context, 1)
        self.assertEqual(first.emit_pre, 0)
        self.assertEqual(first.addr_post, 3)

    def test_missing_column_raises_with_helpful_message(self):
        rows = _fixture_rows(1, 1)
        path = _write_csv(rows)
        text = path.read_text().replace("Addressee pre-vocalization action,", "")
        path.write_text(text)
        with self.assertRaises(ValueError) as ctx:
            parse_annotations_csv(path)
        self.assertIn("Addressee pre-vocalization action", str(ctx.exception))

    def test_bad_emitter_int_raises(self):
        rows = _fixture_rows(1, 1)
        rows[0]["Emitter"] = "unknown"
        path = _write_csv(rows)
        with self.assertRaises(ValueError):
            parse_annotations_csv(path)

    def test_empty_csv_raises(self):
        path = _write_csv([])
        with self.assertRaises(ValueError):
            parse_annotations_csv(path)


class TestSplits(unittest.TestCase):
    def _ann(self, n_callers=10, files=10):
        return parse_annotations_csv(_write_csv(_fixture_rows(n_callers, files)))

    def test_splits_disjoint_and_complete(self):
        ann = self._ann()
        split = split_files_by_caller(ann, seed=0)
        train, val, test = split["train"], split["val"], split["test"]
        self.assertEqual(len(train & val), 0)
        self.assertEqual(len(train & test), 0)
        self.assertEqual(len(val & test), 0)
        all_files = {a.file_name for a in ann}
        self.assertEqual(train | val | test, all_files)

    def test_split_proportions_approximate(self):
        ann = self._ann(n_callers=20, files=10)
        split = split_files_by_caller(ann, seed=0)
        n = len(split["train"]) + len(split["val"]) + len(split["test"])
        self.assertAlmostEqual(len(split["train"]) / n, 0.70, delta=0.08)
        self.assertAlmostEqual(len(split["val"]) / n, 0.15, delta=0.06)

    def test_caller_stratification(self):
        # every caller with enough files appears in train
        ann = self._ann(n_callers=8, files=10)
        split = split_files_by_caller(ann, seed=0)
        emitters_in_train = {a.emitter for a in ann if a.file_name in split["train"]}
        self.assertEqual(emitters_in_train, set(range(1, 9)))

    def test_deterministic(self):
        a = split_files_by_caller(self._ann(), seed=3)
        b = split_files_by_caller(self._ann(), seed=3)
        self.assertEqual(a, b)


class TestHoldoutCallers(unittest.TestCase):
    def test_heldout_callers_files_excluded_from_train(self):
        ann = parse_annotations_csv(_write_csv(_fixture_rows(12, 10)))
        split = holdout_callers(ann, n_heldout=3, min_files=5, seed=0)
        self.assertEqual(len(split.heldout_callers), 3)
        heldout_files = {a.file_name for a in ann if a.emitter in split.heldout_callers}
        for part in (split.train_files, split.val_files, split.test_files):
            self.assertEqual(len(part & heldout_files), 0)
        # remaining callers' files are fully distributed
        rest = {a.file_name for a in ann if a.emitter not in split.heldout_callers}
        self.assertEqual(split.train_files | split.val_files | split.test_files, rest)

    def test_holds_out_only_eligible_callers(self):
        # caller 99 has 30 files; callers 1..5 have 3 files each (below min_files)
        rows = _fixture_rows(5, 3)
        for big in (90, 91, 99):
            rows += [
                dict(r, **{"File Name": f"{big}_{i}.wav", "Emitter": str(big)})
                for i, r in enumerate(_fixture_rows(1, 30))
            ]
        ann = parse_annotations_csv(_write_csv(rows))
        split = holdout_callers(ann, n_heldout=1, min_files=20, seed=0)
        self.assertEqual(len(split.heldout_callers), 1)
        self.assertIn(split.heldout_callers[0], {90, 91, 99})
        # thin callers (1..5) must not be held out
        self.assertNotIn(1, split.heldout_callers)


class TestManifestAndMetadata(unittest.TestCase):
    def test_build_labeled_manifest(self):
        rows = _fixture_rows(2, 2)
        path = _write_csv(rows)
        with tempfile.TemporaryDirectory() as td:
            audio_dir = Path(td)
            for r in rows:
                (audio_dir / r["File Name"]).touch()
            manifest = build_labeled_manifest(audio_dir, path, species="bat")
        self.assertEqual(manifest.dataset, "egyptian_fruit_bats")
        self.assertEqual(manifest.dataset_role, "labeled_per_bout[bat]")
        self.assertEqual(len(manifest.audio_paths), 4)
        self.assertEqual(manifest.annotations, str(path))

    def test_annotation_metadata_fn(self):
        ann = parse_annotations_csv(_write_csv(_fixture_rows(2, 2)))
        fn = annotation_metadata_fn(ann)
        meta = fn("1100.wav")
        self.assertEqual(meta, {"emitter": 1, "addressee": 2, "context": 1})
        self.assertIsNone(fn("not_a_file.wav"))

    def test_summarize_callers(self):
        ann = parse_annotations_csv(_write_csv(_fixture_rows(3, 4)))
        s = summarize_callers(ann)
        self.assertEqual(s["n_callers"], 3)
        self.assertEqual(s["files_per_caller"]["min"], 4)
        self.assertEqual(s["files_per_caller"]["max"], 4)
        self.assertEqual(s["n_contexts"], 4)


if __name__ == "__main__":
    unittest.main()
