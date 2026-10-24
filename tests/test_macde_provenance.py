import pytest
from bioacoustic_pipeline.ingestion.macde_provenance import fact, validate_manifest


def test_unknown_relations_cannot_certify_independent_groups():
    item = {'id': 'a', 'session': fact(None, 'unresolved', 'No session crosswalk'),
            'independence_status': 'unresolved', 'partition': None}
    manifest = {'records': [item], 'certified_independent_groups': 0}
    validate_manifest(manifest)
    manifest['certified_independent_groups'] = 1
    with pytest.raises(ValueError, match='certif'):
        validate_manifest(manifest)


def test_unresolved_value_cannot_be_presented_as_verified():
    with pytest.raises(ValueError):
        fact(None, 'verified', 'No evidence')
    assert fact('archive folder', 'verified', 'Metadata table')['status'] == 'verified'


def test_candidate_records_cannot_receive_partitions_or_duplicate_ids():
    item = {'id': 'a', 'independence_status': 'unresolved', 'partition': 'heldout'}
    with pytest.raises(ValueError):
        validate_manifest({'records': [item], 'certified_independent_groups': 0})
    item['partition'] = None
    with pytest.raises(ValueError):
        validate_manifest({'records': [item, item], 'certified_independent_groups': 0})
