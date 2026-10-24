"""Tests: silence-aware acoustic-activity segmentation pilot (Phase 1).

Auditable ACTIVITY units with preserved gaps — not assumed biological atoms.
Three groups: state-machine mechanics (known log-RMS fixtures), waveform and
numerical behaviour, and provenance/contamination-regression invariants.

All tests operate on mechanics with known answers. No test asserts that
arbitrary noise is always rejected, that synthetic chirps are calls, or that
real-data metrics must improve.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.segmentation.silence_units import (
    AcousticActivityUnit,
    IntervalStatus,
    decode_activity,
    extract_activity_units,
    frame_log_rms,
    HysteresisConfig,
)


def _cfg(**kw):
    base = dict(
        enter_db=-30.0,
        exit_db=-36.0,
        min_on_frames=2,
        min_off_frames=3,
        min_unit_frames=2,
        frame_ms=10,
    )
    base.update(kw)
    return HysteresisConfig(**base)


class TestActivityHysteresis(unittest.TestCase):
    """Group A: state-machine tests on direct log-RMS levels."""

    def test_short_dip_retained_long_gap_splits(self):
        levels = np.array(
            [-60, -60, -25, -25, -32, -40, -25, -40, -40, -40, -25, -25, -40, -40, -40],
            dtype=np.float64,
        )
        result = decode_activity(
            levels, frame_valid=np.ones(levels.size, dtype=bool), config=_cfg()
        )
        observed = [(s.start_frame, s.stop_frame) for s in result.spans]
        self.assertEqual(observed, [(2, 7), (10, 12)])

    def test_all_below_threshold_no_units(self):
        levels = np.full(20, -60.0)
        result = decode_activity(levels, frame_valid=np.ones(20, dtype=bool), config=_cfg())
        self.assertEqual(result.spans, [])
        # observed low-activity interval retained in the timeline
        self.assertTrue(any(iv.status == IntervalStatus.BELOW_ACTIVITY for iv in result.gaps))

    def test_isolated_burst(self):
        levels = np.array([-60] * 5 + [-20] * 5 + [-60] * 5, dtype=np.float64)
        result = decode_activity(levels, frame_valid=np.ones(15, dtype=bool), config=_cfg())
        self.assertEqual([(s.start_frame, s.stop_frame) for s in result.spans], [(5, 10)])

    def test_insufficient_onset_support(self):
        # single above-entry frame, then below — needs min_on_frames=2
        levels = np.array([-60, -60, -25, -60, -60], dtype=np.float64)
        result = decode_activity(levels, frame_valid=np.ones(5, dtype=bool), config=_cfg())
        self.assertEqual(result.spans, [])

    def test_hysteresis_dead_band(self):
        # value between exit and entry cannot START a unit but sustains one
        levels = np.array([-60, -25, -25, -33, -33, -60], dtype=np.float64)
        result = decode_activity(levels, frame_valid=np.ones(6, dtype=bool), config=_cfg())
        # -33 >= exit(-36) keeps active; unit runs frames 1..5
        self.assertEqual([(s.start_frame, s.stop_frame) for s in result.spans], [(1, 5)])

    def test_dead_band_alone_cannot_start(self):
        levels = np.array([-60, -33, -33, -60], dtype=np.float64)
        result = decode_activity(levels, frame_valid=np.ones(4, dtype=bool), config=_cfg())
        self.assertEqual(result.spans, [])

    def test_entry_at_equality_exits_at_equality_excluded(self):
        # documented convention: entry uses >=, exit uses < (strictly below exits)
        levels = np.array([-60, -30, -30, -60], dtype=np.float64)
        result = decode_activity(levels, frame_valid=np.ones(4, dtype=bool), config=_cfg())
        self.assertEqual([(s.start_frame, s.stop_frame) for s in result.spans], [(1, 3)])

    def test_invalid_interruption_hard_breaks(self):
        levels = np.array([-25, -25, np.nan, -25, -25], dtype=np.float64)
        valid = np.array([True, True, False, True, True])
        result = decode_activity(levels, frame_valid=valid, config=_cfg())
        # hard break regardless of min_off_frames: two units (if each qualifies)
        spans = [(s.start_frame, s.stop_frame) for s in result.spans]
        self.assertIn((0, 2), spans)
        self.assertIn((3, 5), spans)

    def test_trailing_active_event_censored(self):
        levels = np.array([-25] * 6, dtype=np.float64)
        result = decode_activity(levels, frame_valid=np.ones(6, dtype=bool), config=_cfg())
        self.assertEqual(len(result.spans), 1)
        self.assertTrue(result.spans[0].censored_at_end)

    def test_unit_below_min_duration_dropped_with_reason(self):
        # qualifying onset+gap but only 1 frame long (min_unit_frames=2)
        levels = np.array([-60, -25, -25, -60], dtype=np.float64)
        cfg = _cfg(min_unit_frames=3)
        result = decode_activity(levels, frame_valid=np.ones(4, dtype=bool), config=cfg)
        self.assertEqual(result.spans, [])
        self.assertTrue(any(r["reason"] == "below_min_duration" for r in result.rejections))


class TestWaveformAndNumerics(unittest.TestCase):
    """Group B: RMS, dtypes, edge behaviour, config validation."""

    def test_frame_log_rms_matches_direct(self):
        sr, fms = 1000, 10
        t = np.arange(sr) / sr
        x = 0.5 * np.sin(2 * np.pi * 50 * t)
        levels = frame_log_rms(x, sr=sr, frame_ms=fms, ref_dbfs=0.0)
        frame = x[: sr * fms // 1000]
        direct = 20 * np.log10(max(np.sqrt((frame.astype(np.float64) ** 2).mean()), 1e-12))
        self.assertAlmostEqual(float(levels[0]), direct, places=6)

    def test_float32_float64_equivalent_decisions(self):
        t = np.arange(4000) / 8000
        chirp = 0.4 * np.sin(2 * np.pi * (100 + 500 * t) * t)
        x = np.concatenate([np.zeros(1000), chirp, np.zeros(1000)])
        r32 = extract_activity_units(x.astype(np.float32), sr=8000, config=_cfg(frame_ms=10))
        r64 = extract_activity_units(x.astype(np.float64), sr=8000, config=_cfg(frame_ms=10))
        self.assertEqual(
            [(s.start_frame, s.stop_frame) for s in r32.spans],
            [(s.start_frame, s.stop_frame) for s in r64.spans],
        )

    def test_zero_recording_clean_empty(self):
        result = extract_activity_units(np.zeros(5000), sr=8000, config=_cfg())
        self.assertEqual(result.spans, [])
        self.assertTrue(all(np.isfinite(iv.start_frame) for iv in result.gaps))

    def test_incomplete_analysis_frame_excluded(self):
        # length not a multiple of the frame: no synthetic samples admitted
        result = extract_activity_units(np.ones(4005) * 0.5, sr=8000, config=_cfg(frame_ms=10))
        self.assertEqual(result.unscored_tail_samples, 4005 % 80)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            _cfg(enter_db=-40.0, exit_db=-30.0)  # enter must exceed exit
        with self.assertRaises(ValueError):
            _cfg(min_on_frames=0)
        with self.assertRaises(ValueError):
            HysteresisConfig(
                enter_db=-30.0,
                exit_db=-36.0,
                min_on_frames=2,
                min_off_frames=3,
                min_unit_frames=2,
                frame_ms=0,
            )

    def test_nonfinite_input_raises(self):
        x = np.zeros(100)
        x[50] = np.nan
        with self.assertRaises(ValueError):
            extract_activity_units(x, sr=8000, config=_cfg(), allow_nonfinite=False)

    def test_reference_scaling_preserves_decisions(self):
        # gain + consistent reference scaling: floor-inactive regime preserves
        t = np.arange(6000) / 8000
        sig = np.concatenate(
            [np.zeros(2000), 0.3 * np.sin(2 * np.pi * 200 * t[:2000]), np.zeros(2000)]
        )
        a = extract_activity_units(sig, sr=8000, config=_cfg(frame_ms=10), ref_amplitude=1.0)
        b = extract_activity_units(sig * 2, sr=8000, config=_cfg(frame_ms=10), ref_amplitude=2.0)
        self.assertEqual(
            [(s.start_frame, s.stop_frame) for s in a.spans],
            [(s.start_frame, s.stop_frame) for s in b.spans],
        )


class TestProvenanceAndContamination(unittest.TestCase):
    """Group C: the §46 regression invariants, at the unit level."""

    def _burst_audio(self, sr=8000):
        t = np.arange(sr) / sr
        tone = 0.5 * np.sin(2 * np.pi * 300 * t)
        return np.concatenate([np.zeros(sr), tone, np.zeros(sr // 2), tone, np.zeros(sr)])

    def test_artificial_tail_padding_inert(self):
        x = self._burst_audio()
        a = extract_activity_units(x, sr=8000, config=_cfg(frame_ms=10))
        padded = extract_activity_units(
            np.concatenate([x, np.zeros(8000)]), sr=8000, config=_cfg(frame_ms=10)
        )
        self.assertEqual(
            [(s.start_frame, s.stop_frame) for s in a.spans],
            [(s.start_frame, s.stop_frame) for s in padded.spans],
        )

    def test_invalid_internal_span_hard_break(self):
        x = self._burst_audio()
        x[8000:12000] = np.nan  # invalidate the middle of the first burst
        valid = ~np.isnan(x)
        result = extract_activity_units(
            np.nan_to_num(x), sr=8000, config=_cfg(frame_ms=10), frame_valid_override=valid
        )
        for s in result.spans:
            self.assertTrue(all(valid[s.start_frame * 80 : (s.stop_frame) * 80]))

    def test_long_leading_silence_discoverable(self):
        x = np.concatenate([np.zeros(8000 * 60), self._burst_audio()])
        result = extract_activity_units(x, sr=8000, config=_cfg(frame_ms=10))
        self.assertGreaterEqual(len(result.spans), 1)

    def test_gap_conservation_no_time_compression(self):
        x = self._burst_audio()
        result = extract_activity_units(x, sr=8000, config=_cfg(frame_ms=10))
        units = sorted([(s.start_frame, s.stop_frame) for s in result.spans])
        for (s1, e1), (s2, e2) in zip(units, units[1:]):
            self.assertLessEqual(e1, s2)
        # total covered <= available frames
        self.assertLessEqual(sum(e - s for s, e in units), len(x) // 80)

    def test_caller_ambiguity_not_resolved_from_energy(self):
        units = [
            AcousticActivityUnit(
                start_frame=0,
                stop_frame=100,
                start_sample=0,
                end_sample=8000,
                sr=8000,
                caller_candidates=[1, 2],  # conflicting annotations
            )
        ]
        self.assertEqual(units[0].attribution_status, "ambiguous")
        self.assertIsNone(units[0].resolved_caller)

    def test_no_mutation(self):
        x = self._burst_audio().copy()
        before = x.copy()
        extract_activity_units(x, sr=8000, config=_cfg(frame_ms=10))
        np.testing.assert_array_equal(x, before)

    def test_save_load_round_trip(self):
        import json

        x = self._burst_audio()
        result = extract_activity_units(x, sr=8000, config=_cfg(frame_ms=10))
        d = result.to_dict()
        json.dumps(d)  # serializable
        self.assertIn("spans", d)
        self.assertIn("gaps", d)
        self.assertIn("rejections", d)
        self.assertIn("config", d)


if __name__ == "__main__":
    unittest.main()
