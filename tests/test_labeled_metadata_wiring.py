"""Tests: caller-metadata wiring through pipeline.run (orchestrator seam).

The metadata seam attaches per-file caller annotations (emitter/addressee/
context) to dual-stream events and segment records WITHOUT influencing any
verdict axis — proven here by running the pipeline twice, with and without
metadata_fn, and asserting the reports are identical. Ceiling-bound: caller
identity is acoustic/behavioral metadata, never meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.ingestion.labeled_corpus import annotation_metadata_fn
from bioacoustic_pipeline.pipeline.orchestrator import run


def _corpus(td: Path):
    """Two tiny bat-rate-ish files with tone bursts + a two-row annotations CSV."""
    sr = 48000
    t = np.arange(int(1.0 * sr)) / sr
    paths = []
    for i, f0 in enumerate((800, 1200)):
        bursts = 0.5 * np.sin(2 * np.pi * f0 * t) * ((np.arange(len(t)) // 8000) % 2)
        p = td / f"bout{i}.wav"
        sf.write(p, bursts.astype(np.float32), sr)
        paths.append(p)
    csv_path = td / "annotations.csv"
    cols = [
        "File Name",
        "Emitter",
        "Addressee",
        "Context",
        "Emitter pre-vocalization action",
        "Addressee pre-vocalization action",
        "Emitter post-vocalization action",
        "Addressee post-vocalization action",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerow(dict(zip(cols, ["bout0.wav", 3, 7, 1, 0, 0, 0, 0])))
        w.writerow(dict(zip(cols, ["bout1.wav", 5, 3, 2, 0, 0, 0, 0])))
    return paths, csv_path


class _NoTok:
    """Minimal unavailable tokenizer — run() falls back to boundary tokens."""

    name = "none"
    provenance_name = "none"
    available = False
    load_error = "not used"


class TestMetadataWiring(unittest.TestCase):
    def _run(self, td, metadata_fn=None):
        paths, csv_path = _corpus(td)
        return run(
            paths,
            species="bat",
            annotations=str(csv_path),
            tokenizer=_NoTok(),
            fingerprint=False,
            metadata_fn=metadata_fn,
        )

    def test_events_and_records_carry_metadata(self):
        from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths, csv_path = _corpus(td)
            base_fn = annotation_metadata_fn(parse_annotations_csv(csv_path))
            meta_fn = lambda path, seg: base_fn(Path(path).name)  # noqa: E731
            result = self._run(td, metadata_fn=meta_fn)
            events = result.dual_stream
            self.assertGreater(len(events), 0)
            self.assertEqual({e["emitter"] for e in events}, {3, 5})
            records = result.segments
            with_meta = [r for r in records if "emitter" in r]
            self.assertEqual(len(with_meta), len(records))
            self.assertEqual({r["emitter"] for r in with_meta}, {3, 5})
            self.assertTrue(all({"addressee", "context"} <= set(r) for r in with_meta))

    def test_records_have_no_metadata_keys_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            result = self._run(Path(td))
            self.assertTrue(all("emitter" not in r for r in result.segments))

    def test_report_identical_with_and_without_metadata(self):
        with tempfile.TemporaryDirectory() as td0, tempfile.TemporaryDirectory() as td1:
            from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

            td, td2 = Path(td0), Path(td1)
            _, csv_path = _corpus(td)
            base_fn = annotation_metadata_fn(parse_annotations_csv(csv_path))
            meta_fn = lambda path, seg: base_fn(Path(path).name)  # noqa: E731
            base = self._run(td)
            wired = self._run(td2, metadata_fn=meta_fn)
            self.assertEqual(base.report, wired.report)


if __name__ == "__main__":
    unittest.main()
