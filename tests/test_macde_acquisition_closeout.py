import pytest
from bioacoustic_pipeline.ingestion.macde_closeout import validate_decision


def test_certified_requires_all_gates_and_explicit_contract():
    row = {'verdict': 'Certified', 'source_lineage': True, 'acoustic_support': True,
           'independence': False, 'contract_reference': None, 'reopen_when': 'Evidence arrives'}
    with pytest.raises(ValueError):
        validate_decision(row)
    row.update(independence=True, contract_reference='frozen-successor-spec')
    validate_decision(row)


def test_resolvable_requires_an_identified_artifact_not_a_hypothetical_crosswalk():
    row = {'verdict': 'Resolvable', 'authoritative_artifact': None, 'reopen_when': 'New evidence'}
    with pytest.raises(ValueError):
        validate_decision(row)
    row['authoritative_artifact'] = 'https://repository.example/record/file'
    validate_decision(row)


def test_blocked_requires_reopening_evidence():
    with pytest.raises(ValueError):
        validate_decision({'verdict': 'Blocked', 'reopen_when': ''})
