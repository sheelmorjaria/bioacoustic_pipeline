"""Determinism tests — prove the pipeline implementation is reproducible (Claim A).

All torch-free: they exercise the DSP-core path (NBD + acoustic fingerprint + seeded
k-means tokenizer + diagnostics) and the hashing/canonical-serialization utilities. The
capstone test runs the full deterministic pipeline twice in-process and asserts every
artifact hashes identically — the in-process analog of ``scripts/reproducibility_tri_run.py``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_deterministic_run():
    """Load scripts/deterministic_run.py as a module (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / "deterministic_run.py"
    spec = importlib.util.spec_from_file_location("_det_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestHashing(unittest.TestCase):
    def test_canonical_json_key_order_independent(self):
        from bioacoustic_pipeline.hashing import canonical_json

        self.assertEqual(canonical_json({"b": 1, "a": 2}), canonical_json({"a": 2, "b": 1}))

    def test_canonical_json_numpy_coercion(self):
        from bioacoustic_pipeline.hashing import canonical_json

        obj = {"x": np.float64(0.5), "i": np.int64(3), "v": np.arange(3)}
        s = canonical_json(obj)  # must not raise
        self.assertIn("0.5", s)

    def test_sha256_file_stable(self):
        from bioacoustic_pipeline.hashing import sha256_file

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"deterministic")
            h1 = sha256_file(p)
            p2 = Path(d) / "g.bin"
            p2.write_bytes(b"deterministic")
            self.assertEqual(h1, sha256_file(p2))


class TestGlobalDeterminism(unittest.TestCase):
    def test_set_global_determinism_seeds_numpy(self):
        from bioacoustic_pipeline.determinism import set_global_determinism

        set_global_determinism(7)
        a = np.random.rand(5)
        set_global_determinism(7)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_pythonhashseed_set(self):
        import os
        from bioacoustic_pipeline.determinism import set_global_determinism

        set_global_determinism(4242)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "4242")


class TestComponentDeterminism(unittest.TestCase):
    """Each deterministic component, called twice on identical input, gives identical output."""

    def _burst(self, sr=16000):
        t = np.arange(sr) / sr
        x = np.zeros(sr)
        x[int(0.1 * sr) : int(0.4 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * t[: int(0.3 * sr)])
        return x.astype(np.float32), sr

    def test_nbd_segmentation_deterministic(self):
        from bioacoustic_pipeline.ingestion.manifest import load_species_profiles
        from bioacoustic_pipeline.orchestration import segment_audio

        audio, sr = self._burst()
        profile = load_species_profiles()["human"]
        s1 = segment_audio(audio, sr, profile, calibrator=None)
        s2 = segment_audio(audio, sr, profile, calibrator=None)
        t1 = [(s.start_ms, s.end_ms, s.boundary_type, round(s.confidence, 6)) for s in s1]
        t2 = [(s.start_ms, s.end_ms, s.boundary_type, round(s.confidence, 6)) for s in s2]
        self.assertEqual(t1, t2)

    def test_fingerprint_extract_bit_identical(self):
        from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp

        audio, sr = self._burst()
        a = afp.extract(audio, sr, species="human").combined
        b = afp.extract(audio, sr, species="human").combined
        np.testing.assert_array_equal(a, b)

    def test_fingerprint_tokenizer_seeded_reproducible(self):
        from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (
            FingerprintTokenizer,
        )

        rng = np.random.RandomState(0)
        X = rng.randn(80, 43)
        t1 = FingerprintTokenizer.fit(X, k=8, seed=123)
        t2 = FingerprintTokenizer.fit(X, k=8, seed=123)
        np.testing.assert_array_equal(t1.centroids, t2.centroids)
        self.assertEqual(t1.tokenize(X[0]), t2.tokenize(X[0]))

    def test_motif_miner_stable_order(self):
        from bioacoustic_pipeline.diagnostics import motif_miner as mm

        seqs = [[1, 2, 3, 1, 2, 4], [1, 2, 3, 4, 1, 2], [3, 1, 2, 4, 3, 1]]
        m1 = [m.as_dict() for m in mm.build_motifs(mm.mine_ngrams(seqs, 2, 3, 2), seqs, "token")]
        m2 = [m.as_dict() for m in mm.build_motifs(mm.mine_ngrams(seqs, 2, 3, 2), seqs, "token")]
        self.assertEqual(m1, m2)

    def test_alias_detector_stable_components(self):
        from bioacoustic_pipeline.diagnostics.alias_detector import detect_functional_aliases

        rng = np.random.RandomState(1)
        seqs = [[int(rng.randint(0, 10)) for _ in range(40)] for _ in range(20)]
        r1 = detect_functional_aliases(seqs, similarity_threshold=0.5, min_instances=3)
        r2 = detect_functional_aliases(seqs, similarity_threshold=0.5, min_instances=3)
        self.assertEqual(
            [sorted(g.member_tokens) for g in r1.groups],
            [sorted(g.member_tokens) for g in r2.groups],
        )


class TestReproManifest(unittest.TestCase):
    def test_build_repro_manifest_has_required_fields(self):
        from bioacoustic_pipeline.determinism import build_repro_manifest

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            p.write_bytes(b"x" * 100)
            m = build_repro_manifest(
                species="human",
                seed=1337,
                audio_paths=[str(p)],
                profile={"sample_rate": 16000},
                tokenizer_info={"k": 32, "seed": 1337},
                calibrator_fitted=False,
                repo=REPO_ROOT,
            )
        for key in (
            "pipeline_version",
            "git_commit",
            "seed",
            "device",
            "species",
            "dataset_manifest_hash",
            "species_profile",
            "nbd_config",
            "fingerprint_tokenizer",
            "claim_scope",
            "claim",
        ):
            self.assertIn(key, m, f"missing {key}")
        self.assertEqual(m["device"]["device"], "cpu")
        self.assertEqual(m["seed"], 1337)


class TestFullDeterministicRun(unittest.TestCase):
    """Capstone: two full frozen runs produce byte-identical artifacts (Claim A)."""

    def test_two_runs_hash_identical(self):
        from bioacoustic_pipeline.hashing import hash_artifacts

        det = _load_deterministic_run()
        names = det.ARTIFACT_NAMES
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "run_a", Path(d) / "run_b"
            det.run_deterministic(out_dir=a, species="human", seed=1337, n_files=6, k=8)
            det.run_deterministic(out_dir=b, species="human", seed=1337, n_files=6, k=8)
            ha = hash_artifacts(a, names)
            hb = hash_artifacts(b, names)
        self.assertEqual(set(ha), set(hb), "different artifact sets produced")
        for name in ha:
            self.assertEqual(ha[name], hb[name], f"artifact {name} drifted between runs")


if __name__ == "__main__":
    unittest.main()
