import pytest
from bioacoustic_pipeline.ingestion.gombe_alignment_pilot import sampled_windows, validate_review


def test_sampling_is_disjoint_deterministic_and_not_boundary_dependent():
    windows = sampled_windows(969, 24, 20)
    assert len(windows) == 24
    assert windows == sampled_windows(969, 24, 20)
    assert all(w['end_seconds']-w['start_seconds'] == 20 for w in windows)
    assert all(a['end_seconds'] < b['start_seconds'] for a,b in zip(windows, windows[1:]))


def test_independent_flags_do_not_promote_boundary_to_catalogue_or_session():
    review = {'status': 'reviewed', 'boundary_verified': True, 'catalogue_id_verified': False,
              'session_independence_verified': False, 'evidence': 'Spoken cut announcement',
              'catalogue_id': None, 'session_evidence': None}
    validate_review(review)
    review['catalogue_id_verified'] = True
    with pytest.raises(ValueError):
        validate_review(review)
    review.update(catalogue_id='196101', session_independence_verified=True)
    with pytest.raises(ValueError):
        validate_review(review)


def test_unreviewed_rows_cannot_be_certified_and_acoustic_change_is_insufficient():
    with pytest.raises(ValueError):
        validate_review({'status': 'not_reviewed', 'boundary_verified': True})
    with pytest.raises(ValueError):
        validate_review({'status': 'reviewed', 'boundary_verified': True, 'evidence': ''})
