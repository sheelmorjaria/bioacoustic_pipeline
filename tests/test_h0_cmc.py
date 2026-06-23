"""Unit tests for the H0-CMC evidence-tier engine (Architecture v1.0 Step 5)."""

from bioacoustic_pipeline.diagnostics.h0_cmc import EvidenceTier, H0CMCEvidence, classify_tier


def test_autonomous_tiers_are_t0_t1():
    # No context score → T0 (acoustic only)
    assert classify_tier(acoustic_score=0.5, context_score=0.0) == EvidenceTier.T0_ACOUSTIC
    # Context score > 0.15 → T1
    assert classify_tier(acoustic_score=0.5, context_score=0.2) == EvidenceTier.T1_CONTEXT


def test_wet_lab_tiers_require_biological_data():
    # Without behavioral/causal/field data, never above T1
    assert classify_tier(acoustic_score=0.9, context_score=0.9) == EvidenceTier.T1_CONTEXT
    # With behavioral response → T3
    assert (
        classify_tier(acoustic_score=0.5, context_score=0.2, behavioral_response=0.5)
        == EvidenceTier.T3_CONVERGENCE
    )
    # With causal specificity → T4
    assert (
        classify_tier(acoustic_score=0.5, context_score=0.2, causal_specificity=0.7)
        == EvidenceTier.T4_COUNTERFACTUAL
    )
    # With field fitness → T5
    assert (
        classify_tier(acoustic_score=0.5, context_score=0.2, field_fitness=0.5)
        == EvidenceTier.T5_FIELD_REPLICATION
    )


def test_semantic_validation_mapping():
    e = H0CMCEvidence(corpus="bat", tier=EvidenceTier.T0_ACOUSTIC, acoustic_score=0.5)
    assert e.semantic_validation == "none"
    e.tier = EvidenceTier.T3_CONVERGENCE
    assert e.semantic_validation == "candidate_functional_interpretation"
    e.tier = EvidenceTier.T5_FIELD_REPLICATION
    assert e.semantic_validation == "ecologically_validated_function"


def test_autonomous_vs_wet_lab():
    e = H0CMCEvidence(corpus="bat", tier=EvidenceTier.T1_CONTEXT)
    assert e.is_autonomous is True
    assert e.requires_wet_lab is False
    e.tier = EvidenceTier.T2_PLAYBACK_1M
    assert e.is_autonomous is False
    assert e.requires_wet_lab is True


def test_evidence_score_increases_with_wet_lab():
    e0 = H0CMCEvidence(corpus="x", acoustic_score=0.5, context_score=0.3)
    s0 = e0.evidence_score()
    e0.behavioral_response = 0.6
    e0.causal_specificity = 0.7
    s1 = e0.evidence_score()
    assert s1 > s0


def test_as_dict_roundtrip():
    e = H0CMCEvidence(
        corpus="bat",
        tier=EvidenceTier.T1_CONTEXT,
        acoustic_score=0.5,
        context_score=0.2,
        social_lift=1.74,
    )
    d = e.as_dict()
    assert d["corpus"] == "bat"
    assert d["tier"] == "T1"
    assert d["semantic_validation"] == "none"
    assert d["social_lift"] == 1.74


def test_explicit_implicit_explicit_acoustic():
    """When acoustics alone predict context well → EXPLICIT_ACOUSTIC."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.h0_cmc import (
        CommunicationMode,
        explicit_vs_implicit_test,
    )

    rng = np.random.RandomState(0)
    # 3 context classes, acoustic features well-separated (offset = 5 std)
    n = 60
    acoustic = np.vstack([rng.randn(20, 5) + c * 5 for c in range(3)])
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    social = rng.randn(n, 3)  # social features are noise
    result = explicit_vs_implicit_test(acoustic, social, labels, full_features=acoustic, cv_folds=3)
    assert result["mode"] == CommunicationMode.EXPLICIT_ACOUSTIC


def test_explicit_implicit_socially_grounded():
    """When social features carry the signal → SOCIALLY_GROUNDED_IMPLICIT."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.h0_cmc import (
        explicit_vs_implicit_test,
    )

    rng = np.random.RandomState(0)
    n = 60
    acoustic = rng.randn(n, 5)  # acoustic = noise (A low)
    # social features encode the context
    social = np.vstack([rng.randn(20, 3) + [c] * 3 for c in range(3)])
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    result = explicit_vs_implicit_test(acoustic, social, labels, cv_folds=3)
    assert result["accuracy_acoustic_social"] > result["accuracy_acoustic"]
    assert result["social_lift"] > 1.15


def test_split_context_routes_correctly():
    from bioacoustic_pipeline.diagnostics.h0_cmc import split_context

    social, generic = split_context(
        {
            "emitter": "bat_1",
            "addressee": "bat_2",
            "behavior": "feeding",
            "context": "outdoor",
            "relationship": "kin",
        }
    )
    assert social.emitter_id == "bat_1"
    assert social.addressee_id == "bat_2"
    assert social.relationship == "kin"
    assert generic.behavioral_state == "feeding"
    assert generic.environmental == "outdoor"
