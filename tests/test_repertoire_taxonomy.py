"""Unit tests for the repertoire taxonomy classifier (Architecture v1.0 Step 3).

Pure-stdlib classifier; tests pin each reference corpus to its validated type and cover the
None-field / secondary-flag paths. Numbers mirror BENCHMARK §22 + binding.
"""

from bioacoustic_pipeline.diagnostics.repertoire_taxonomy import (
    RepertoireProfile,
    RepertoireType,
    classify_repertoire,
    route_by_continuous,
)


def test_bengalese_discrete_syllable():
    p = RepertoireProfile(
        corpus="bengalese", coverage=0.480, null_ratio=2.67, bootstrap_stability=0.509
    )
    assert p.repertoire_type == RepertoireType.DISCRETE_SYLLABLE
    assert p.secondary_flags == []


def test_bat_graded_with_social_and_identity_flags():
    p = RepertoireProfile(
        corpus="bat",
        coverage=0.153,
        null_ratio=1.36,
        bootstrap_stability=0.394,
        emitter_v=0.446,
        context_v=0.167,
        social_lift=1.74,
    )
    assert p.repertoire_type == RepertoireType.GRADED
    assert "identity_dominant" in p.secondary_flags
    assert "socially_grounded" in p.secondary_flags


def test_marmoset_sparse_repetitive():
    p = RepertoireProfile(
        corpus="marmoset",
        coverage=0.028,
        null_ratio=1.55,
        bootstrap_stability=0.229,
        context_v=0.599,
    )
    assert p.repertoire_type == RepertoireType.SPARSE_REPETITIVE
    assert p.secondary_flags == []  # no emitter_v, no social_lift


def test_rook_individually_stereotyped():
    p = RepertoireProfile(corpus="rook", emitter_v=0.31, context_v=0.04)
    assert p.repertoire_type == RepertoireType.INDIVIDUALLY_STEREOTYPED
    assert "identity_dominant" in p.secondary_flags


def test_human_anchor():
    p = RepertoireProfile(corpus="librispeech", is_human=True)
    assert p.repertoire_type == RepertoireType.HUMAN_LEXICAL


def test_empty_profile_unknown():
    assert classify_repertoire(RepertoireProfile(corpus="x")) == RepertoireType.UNKNOWN


def test_identity_dominant_requires_context_v():
    # emitter_v alone (no context_v) is not identity_dominant — avoids a false flag
    p = RepertoireProfile(corpus="x", emitter_v=0.9)
    assert not p.identity_dominant
    assert "identity_dominant" not in p.secondary_flags


def test_description_nonempty_for_known_types():
    for kwargs, expected in [
        (
            dict(coverage=0.5, null_ratio=3.0, bootstrap_stability=0.6),
            RepertoireType.DISCRETE_SYLLABLE,
        ),
        (dict(coverage=0.2, null_ratio=1.0), RepertoireType.GRADED),
        (dict(is_human=True), RepertoireType.HUMAN_LEXICAL),
    ]:
        p = RepertoireProfile(corpus="x", **kwargs)
        assert p.repertoire_type == expected
        assert p.description


def test_route_by_continuous_matches_validated_regimes():
    # marmoset §33: MP 0.788, LAM 0.493 → sparse + skip DUSTED
    t, r = route_by_continuous(0.788, 0.493)
    assert t == RepertoireType.SPARSE_REPETITIVE
    assert r == "skip_dusted_use_mp"
    # bat §33: MP 0.872, LAM 0.169 → graded + rqa-then-dusted
    t, r = route_by_continuous(0.872, 0.169)
    assert t == RepertoireType.GRADED
    assert r == "use_rqa_for_dynamics_then_dusted_for_motifs"
    # discrete default (e.g. Bengalese-style moderate signals)
    t, r = route_by_continuous(0.50, 0.25)
    assert t == RepertoireType.DISCRETE_SYLLABLE
    assert r == "run_full_dusted"


def test_route_by_continuous_missing_signals_defaults_to_full_dusted():
    t, r = route_by_continuous(None, None)
    assert t == RepertoireType.DISCRETE_SYLLABLE
    assert r == "run_full_dusted"
