"""Tests for the canonical typed pipeline (Phase 1): contracts, registry, domain mask,
artifact writer, and an end-to-end ``pipeline.run`` on synthetic audio.

The legacy ``orchestration.run_pipeline`` is intentionally NOT touched in Phase 1 (it
carries features the registry doesn't yet have — bank warming, with_null, the in-pipeline
fingerprint-coherence channel); ``pipeline.run`` is the additive typed path. Delegation is
a Phase-2 task once the registry reaches feature parity. These tests cover the new package.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.diagnostics.domain_gate import gate_from_manifest
from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity
from bioacoustic_pipeline.pipeline import (
    ArtifactWriter,
    ConfigResolver,
    DiagnosticRegistry,
    default_registry,
)
from bioacoustic_pipeline.pipeline.contracts import (
    STATUS_COMPUTED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_EVALUABLE,
    DiagnosticResult,
    PipelineResult,
)
from bioacoustic_pipeline.verdict_engine import is_valid


class TestContracts(unittest.TestCase):
    def test_diagnostic_result_status_validated(self):
        DiagnosticResult("x", STATUS_COMPUTED)
        with self.assertRaises(ValueError):
            DiagnosticResult("x", "bogus")

    def test_pipeline_result_as_tuple(self):
        r = PipelineResult(
            report={"a": 1},
            trace={"b": 2},
            manifest={},
            domain_gate={},
            segments=[],
            dual_stream=[],
            sequences=[[]],
            diagnostics={},
        )
        self.assertEqual(r.as_tuple(), ({"a": 1}, {"b": 2}))

    def test_lz_as_dict(self):
        d = lempel_ziv_complexity([1, 2, 3, 1, 2]).as_dict()
        self.assertEqual({"c_n", "normalized", "structure_class"}, set(d))


class TestRegistry(unittest.TestCase):
    def test_default_registry_order(self):
        self.assertEqual(
            default_registry().names(), ["sequence_sufficiency", "lempel_ziv", "motif"]
        )

    def test_domain_gate_skips_axis(self):
        gate = gate_from_manifest("esc-50")  # ethological axes disabled
        reg = DiagnosticRegistry()
        reg.register(
            "motif", lambda e, c: {"ok": 1}, axis="motif_structure", requires_sufficiency=False
        )
        out = reg.run([], {"domain_gate": gate, "sequences": [], "sufficiency": None})
        self.assertEqual(out["motif"].status, STATUS_NOT_APPLICABLE)

    def test_sufficiency_gate_abstains(self):
        gate = gate_from_manifest("egyptian_fruit_bat")  # candidate: nothing disabled
        gate.has_context = True
        reg = DiagnosticRegistry()
        reg.register("sequence_sufficiency", lambda e, c: _set_suff(c, False), first=True)
        reg.register(
            "lempel_ziv",
            lambda e, c: {"normalized": 0.5},
            axis="sequence_recombination",
            requires_sufficiency=True,
        )
        ctx = {"domain_gate": gate, "sequences": [], "sufficiency": None}
        out = reg.run([], ctx)
        self.assertEqual(out["sequence_sufficiency"].status, STATUS_COMPUTED)
        self.assertEqual(out["lempel_ziv"].status, STATUS_NOT_EVALUABLE)

    def test_requires_context_skips(self):
        gate = gate_from_manifest("egyptian_fruit_bat")  # has_context defaults False
        reg = DiagnosticRegistry()
        reg.register(
            "context_binding", lambda e, c: {}, axis="context_binding", requires_context=True
        )
        out = reg.run([], {"domain_gate": gate, "sequences": [], "sufficiency": None})
        self.assertEqual(out["context_binding"].status, "skipped")


def _set_suff(ctx, evaluable):
    class _S:
        evaluable = False

    s = _S()
    s.evaluable = evaluable
    ctx["sufficiency"] = s
    return {"evaluable": evaluable}


class TestDomainMask(unittest.TestCase):
    def test_esc50_disables_ethological(self):
        g = gate_from_manifest("esc-50")
        self.assertTrue(set(g.disabled_axes) >= {"functional_aliasing", "context_binding"})
        self.assertFalse(g.enabled_axes & g.disabled_axes)

    def test_candidate_carries_has_flags(self):
        g = gate_from_manifest("bat", has_context=True, has_emitter=True, annotations="x.csv")
        self.assertTrue(g.has_context and g.has_emitter)
        self.assertFalse(g.disabled_axes)  # nothing refused for a candidate


class TestConfigResolver(unittest.TestCase):
    def test_resolves_profile_and_role(self):
        cfg = ConfigResolver.resolve(species="bat", dataset="esc-50", seed=7)
        self.assertEqual(cfg.species, "bat")
        self.assertEqual(cfg.dataset_role, "negative_control_environmental_sound")
        self.assertEqual(cfg.seed, 7)
        self.assertIn("sample_rate", cfg.profile)


class TestArtifactWriter(unittest.TestCase):
    def test_writes_bundle_and_hashes(self):
        report = {"axes": {}, "overall_confidence": 0.5}
        diag = {
            "sequence_sufficiency": DiagnosticResult(
                "sequence_sufficiency", STATUS_COMPUTED, {"x": 1}
            )
        }
        res = PipelineResult(
            report=report,
            trace={"t": 1},
            manifest={"m": 1},
            domain_gate={"dataset": "esc-50"},
            segments=[{"segment_id": "s0"}],
            dual_stream=[{"token": 0, "fingerprint": np.ones(4)}],
            sequences=[[0]],
            diagnostics=diag,
            config={"species": "human"},
        )
        with tempfile.TemporaryDirectory() as d:
            info = ArtifactWriter.write(res, d)
            out = Path(d)
            self.assertTrue((out / "verdict.json").exists())
            self.assertTrue((out / "fingerprints.npy").exists())
            self.assertTrue((out / "diagnostics" / "sequence_sufficiency.json").exists())
            self.assertGreater(info["n_artifacts"], 5)
            self.assertIn("verdict.json", info["hashes"])


class TestPipelineRun(unittest.TestCase):
    """End-to-end pipeline.run on synthetic audio (torch-free fingerprint tokenizer)."""

    def _corpus_and_tokenizer(self, d):
        import soundfile as sf  # noqa: PLC0415

        from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
        from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (  # noqa: PLC0415
            FingerprintTokenizer,
        )
        from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: PLC0415
        from bioacoustic_pipeline.orchestration import segment_audio  # noqa: PLC0415
        from bioacoustic_pipeline.tokenization.fingerprint_tokenizer_adapter import (  # noqa: PLC0415
            FingerprintTokenizerAdapter,
        )
        from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

        sr = 16000
        profile = load_species_profiles()["human"]
        paths = []
        for i in range(8):
            rng = np.random.default_rng(100 + i)
            parts = []
            for _ in range(int(rng.integers(3, 6))):
                f = float(rng.uniform(300, 800))
                n = int(sr * float(rng.uniform(0.06, 0.18)))
                x = 0.5 * np.sin(2 * np.pi * f * np.arange(n) / sr)
                parts.append(x.astype(np.float32))
                parts.append(np.zeros(int(sr * 0.25), dtype=np.float32))
            p = Path(d) / f"s{i}.wav"
            sf.write(str(p), np.concatenate(parts), sr, subtype="FLOAT")
            paths.append(str(p))
        matrix = []
        for p in sorted(paths):
            a, _ = load_audio(p, target_sr=sr)
            for seg in segment_audio(a, sr, profile, calibrator=None):
                fp = afp.extract(seg.audio, seg.sr, species="human")
                if np.any(fp.combined):
                    matrix.append(np.asarray(fp.combined, dtype=np.float64))
        tok = FingerprintTokenizerAdapter(
            FingerprintTokenizer.fit(np.stack(matrix), k=8, seed=1337), species="human"
        )
        return sorted(paths), tok

    def test_run_returns_schema_valid_result(self):
        from bioacoustic_pipeline.pipeline import run

        with tempfile.TemporaryDirectory() as d:
            paths, tok = self._corpus_and_tokenizer(d)
            res = run(paths, species="human", tokenizer=tok, out_dir=Path(d) / "out", seed=1337)
        self.assertTrue(is_valid(res.report))
        self.assertEqual(res.as_tuple()[0], res.report)
        self.assertIn("sequence_sufficiency", res.diagnostics)
        self.assertEqual(res.diagnostics["sequence_sufficiency"].status, STATUS_COMPUTED)
        self.assertGreater(len(res.dual_stream), 0)
        self.assertIsNotNone(res.artifacts)


if __name__ == "__main__":
    unittest.main()
