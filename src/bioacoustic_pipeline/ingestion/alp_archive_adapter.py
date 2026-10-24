"""ALP Tierstimmenarchiv adapter — taxonomy-preserving manifest ingestion.

The animal-sound-archive (Tierstimmenarchiv) manifest carries Class/Order/
Family/Species per recording. This adapter normalizes rows into pipeline
records (missing fields -> "Unknown", never guessed) and subsamples per
stratum with a fixed seed for the cross-taxa fingerprint sweep.

License: CC-BY-NC-SA (non-commercial) — analysis-only; recorded in every
artifact this module feeds. NOT meaning: taxonomy is metadata for
stratification, and silhouette separation is acoustic-geometry, not
phylogenetic evidence.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import random

_TAXON_FIELDS = ("class", "order", "family", "species")


def parse_alp_manifest(manifest_rows) -> list[dict]:
    """Normalize manifest rows into taxonomy records (Unknown for missing)."""
    records = []
    for item in manifest_rows:
        rec = {"file_path": item.get("file_path", "")}
        for f in _TAXON_FIELDS:
            rec[f] = item.get(f, "Unknown") or "Unknown"
        records.append(rec)
    return records


def stratified_subsample(
    records: list[dict], *, strata_key: str, n_per_strata: int, seed: int = 42
) -> list[dict]:
    """Fixed count per stratum; strata smaller than n are taken wholesale."""
    rng = random.Random(seed)
    strata: dict = {}
    for r in records:
        strata.setdefault(r[strata_key], []).append(r)
    out = []
    for _name, items in sorted(strata.items(), key=lambda kv: str(kv[0])):
        out.extend(items if len(items) <= n_per_strata else rng.sample(items, n_per_strata))
    return out
