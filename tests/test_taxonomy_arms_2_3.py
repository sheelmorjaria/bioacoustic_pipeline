"""Tests: taxonomy Arms 2-3 — event-level extraction (annotated / automatic).

Arm 2 extracts from an ANNOTATED interval; Arm 3 from a LABEL-BLIND
energy-gated event (the detector sees samples only — the taxon label never
enters its processing path). Both return the 44-D vector from the event,
not the blind first-500ms crop.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


def _recording(td: Path):
    """1 s silence, 1 s tone @1000-2000 ms, 1 s silence at 16 kHz."""
    sr = 16000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.concatenate([np.zeros(sr), tone, np.zeros(sr)]).astype(np.float32)
    p = td / "r.wav"
    sf.write(p, audio, sr)
    return p, audio, sr


class TestArm2AnnotatedEvent(unittest.TestCase):
    def test_extracts_from_annotation_interval(self):
        from bioacoustic_pipeline.diagnostics.taxonomy_arms import (
            run_arm2_annotated_event,
        )

        with tempfile.TemporaryDirectory() as td:
            p, audio, sr = _recording(Path(td))
            res = run_arm2_annotated_event(str(p), {"start_ms": 1000, "end_ms": 2000}, sr=16000)
            self.assertEqual(res["fingerprint"].shape, (44,))
            self.assertEqual(res["source"], "annotated_interval")

    def test_missing_annotation_raises(self):
        from bioacoustic_pipeline.diagnostics.taxonomy_arms import (
            run_arm2_annotated_event,
        )

        with self.assertRaises(ValueError):
            run_arm2_annotated_event("x.wav", None)


class TestArm3AutomaticEvent(unittest.TestCase):
    def test_detector_finds_tone_between_silence(self):
        from bioacoustic_pipeline.diagnostics.taxonomy_arms import (
            run_arm3_automatic_event,
        )

        with tempfile.TemporaryDirectory() as td:
            _p, audio, sr = _recording(Path(td))
            res = run_arm3_automatic_event(audio, sr)
            self.assertTrue(res["event_found"])
            self.assertEqual(res["fingerprint"].shape, (44,))
            # the detected window must sit inside the tone (1.0-2.0 s)
            self.assertGreaterEqual(res["start_ms"], 900.0)
            self.assertLessEqual(res["end_ms"], 2100.0)

    def test_signature_is_label_blind(self):
        """The detector accepts (audio, sr) only — no label parameter exists."""
        import inspect

        from bioacoustic_pipeline.diagnostics import taxonomy_arms

        sig = inspect.signature(taxonomy_arms.run_arm3_automatic_event)
        self.assertNotIn("label", sig.parameters)
        self.assertNotIn("taxon", sig.parameters)
        self.assertNotIn("y", sig.parameters)

    def test_all_silence_reports_no_event(self):
        from bioacoustic_pipeline.diagnostics.taxonomy_arms import (
            run_arm3_automatic_event,
        )

        res = run_arm3_automatic_event(np.zeros(16000, dtype=np.float32), 16000)
        self.assertFalse(res["event_found"])
        self.assertNotIn("fingerprint", res)


if __name__ == "__main__":
    unittest.main()
