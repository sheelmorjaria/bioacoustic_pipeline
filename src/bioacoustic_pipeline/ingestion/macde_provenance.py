"""Evidence states for candidate sources; unresolved independence blocks assignment."""


def fact(value, status, evidence):
    if status not in ('verified', 'unresolved', 'candidate'):
        raise ValueError('Unknown evidence status')
    if status == 'verified' and value is None:
        raise ValueError('Verified facts require a value')
    return {'value': value, 'status': status, 'evidence': evidence}


def validate_manifest(manifest):
    records = manifest['records']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate source IDs')
    for row in records:
        if row['independence_status'] != 'verified' and row['partition'] is not None:
            raise ValueError('Unresolved source independence blocks partition assignment')
    verified = {r['session']['value'] for r in records
                if r['independence_status'] == 'verified' and r['session']['status'] == 'verified'}
    if manifest['certified_independent_groups'] != len(verified):
        raise ValueError('Independent-group certification requires explicit session evidence')
