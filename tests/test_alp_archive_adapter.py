"""Tests: ALP Tierstimmenarchiv adapter + stratified subsampling (cross-taxa).

The archive manifest (tierstimmenarchiv/v0.1.0/raw/all_v2.csv) carries
taxonomic fields; the adapter normalizes them into pipeline-consumable
records and subsamples per stratum with fixed seed. License: CC-BY-NC-SA
(non-commercial) — analysis-only use, recorded in every artifact.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.ingestion.alp_archive_adapter import (
    parse_alp_manifest,
    stratified_subsample,
)


class TestALPArchiveAdapter(unittest.TestCase):
    def setUp(self):
        self.manifest = [
            {
                "file_path": "audio/bird1.wav",
                "class": "Aves",
                "order": "Passeriformes",
                "family": "Paridae",
                "species": "Parus_major",
            },
            {
                "file_path": "audio/bird2.wav",
                "class": "Aves",
                "order": "Passeriformes",
                "family": "Paridae",
                "species": "Parus_major",
            },
            {
                "file_path": "audio/frog1.wav",
                "class": "Amphibia",
                "order": "Anura",
                "family": "Ranidae",
                "species": "Rana_temporaria",
            },
            {
                "file_path": "audio/insect1.wav",
                "class": "Insecta",
                "order": "Orthoptera",
                "family": "Tettigoniidae",
                "species": "Tettigonia_viridissima",
            },
        ]

    def test_parse_preserves_taxonomy_and_defaults(self):
        records = parse_alp_manifest(self.manifest)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["species"], "Parus_major")
        self.assertEqual(records[0]["class"], "Aves")
        missing = parse_alp_manifest([{"file_path": "x.wav"}])[0]
        self.assertEqual(missing["class"], "Unknown")

    def test_stratified_subsample_fixed_per_stratum(self):
        pool = self.manifest * 50  # 200 items, 3 classes
        sub = stratified_subsample(pool, strata_key="class", n_per_strata=10, seed=42)
        counts: dict = {}
        for r in sub:
            counts[r["class"]] = counts.get(r["class"], 0) + 1
        self.assertEqual(len(sub), 30)
        self.assertEqual(sorted(counts.values()), [10, 10, 10])

    def test_small_strata_taken_wholesale(self):
        sub = stratified_subsample(self.manifest, strata_key="class", n_per_strata=10)
        self.assertEqual(len(sub), 4)  # every stratum smaller than 10

    def test_seed_reproducible(self):
        pool = self.manifest * 50
        a = stratified_subsample(pool, strata_key="class", n_per_strata=5, seed=1)
        b = stratified_subsample(pool, strata_key="class", n_per_strata=5, seed=1)
        self.assertEqual([r["file_path"] for r in a], [r["file_path"] for r in b])


if __name__ == "__main__":
    unittest.main()
