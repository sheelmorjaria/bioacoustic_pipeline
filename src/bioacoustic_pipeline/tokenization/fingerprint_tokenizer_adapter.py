"""FingerprintTokenizerAdapter — torch-free, firewall-legal drop-in tokenizer.

``orchestration.run_pipeline`` accepts any ``tokenizer`` exposing ``.available``,
``.provenance_name`` (firewall-gated), and ``.tokenize(audio, sr) -> int``. The default
is the human VQ-VAE on CPC — torch-backed and firewall-restricted to humans.

This adapter wraps a **pre-fit** :class:`acoustic_fingerprint.tokenizer.FingerprintTokenizer`
(a seeded k-means codebook over the deterministic ~43-D acoustic fingerprint) so the
DSP-core path can emit real token sequences **without torch** and **for any species**
(the fingerprint is registered ``applies_to_species: []`` → cross-species allowed).

Determinism: ``extract`` is pure NumPy, and ``FingerprintTokenizer.tokenize`` uses
``np.argmin`` (first-index tie-break). Same audio + same fitted codebook → same token.
The codebook is fit once (seeded) and frozen; this adapter never mutates it.

This is an acoustic-similarity tokenization, NOT meaning (Semantic Ceiling still holds).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp


class FingerprintTokenizerAdapter:
    """Adapt a fitted ``FingerprintTokenizer`` to the run_pipeline tokenizer interface."""

    available: bool = True
    provenance_name: str = "acoustic_fingerprint"  # firewall: all-species, tokenize allowed
    input_source: str = "acoustic_fingerprint"  # tokens come from the fingerprint space

    def __init__(self, fitted, species: str = ""):
        self._tok = fitted
        self.species = species

    def tokenize(self, audio: np.ndarray, sr: int) -> int:
        """Extract the segment fingerprint, then assign the nearest codebook token.

        Returns -1 for a zeroed (silent / too-short) fingerprint, matching the VQ-VAE
        convention that run_pipeline skips.
        """
        audio = np.asarray(audio, dtype=np.float32)
        fp = afp.extract(
            audio, sr, start_ms=0.0, end_ms=len(audio) / sr * 1000.0, species=self.species
        )
        return self._tok.tokenize(np.asarray(fp.combined, dtype=np.float64))
