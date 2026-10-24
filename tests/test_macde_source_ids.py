import pytest
from bioacoustic_pipeline.ingestion.macde_source_ids import zhao_reference, certification_level


def test_zhao_original_audio_date_is_distinct_from_annotation_save_date():
    r = zhao_reference(r'D:\wave\pairisolation_cj181_cj931_20230524_120408.wav')
    assert r['filename_timestamp'] == '2023-05-24T12:04:08'
    assert r['subject_tokens'] == ['cj181', 'cj931']
    assert r['independence_status'] == 'unresolved'
    with pytest.raises(ValueError):
        zhao_reference('annotation 2025-10-15 4_01 PM.mat')


def test_provenance_stages_cannot_be_skipped():
    assert certification_level(integrity=True, acoustic=True, provenance=False, independent=False) == 'S1'
    with pytest.raises(ValueError):
        certification_level(integrity=True, acoustic=True, provenance=False, independent=True)
    with pytest.raises(ValueError):
        certification_level(integrity=False, acoustic=True, provenance=False, independent=False)
