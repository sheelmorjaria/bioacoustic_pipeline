"""H0-CMC (Causal Multimodal Convergence) — the evidence-tier spine.

Architecture v1.0 Steps 5–7. The T0→T5 evidence ladder connects every pipeline module to the
verdict engine's ``semantic_validation`` axis. See ``evidence_tiers`` for the tier definitions +
the ``H0CMCEvidence`` dataclass. The explicit-implicit test + social axis are companion modules.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.diagnostics.h0_cmc.evidence_tiers import (
    EvidenceTier,
    H0CMCEvidence,
    classify_tier,
)

__all__ = ["EvidenceTier", "H0CMCEvidence", "classify_tier"]
