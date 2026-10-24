"""Validate evidence-qualified acquisition closeout verdicts."""


def validate_decision(row):
    if row['verdict'] not in ('Certified', 'Resolvable', 'Blocked'):
        raise ValueError('Unknown acquisition verdict')
    if not row.get('reopen_when'):
        raise ValueError('Every verdict needs an explicit evidence trigger')
    if row['verdict'] == 'Certified' and not all(row.get(k) for k in
            ('source_lineage', 'acoustic_support', 'independence', 'contract_reference')):
        raise ValueError('Certification requires all evidence gates and a declared contract')
    if row['verdict'] == 'Resolvable' and not row.get('authoritative_artifact'):
        raise ValueError('Resolvable requires an identified authoritative artifact')
