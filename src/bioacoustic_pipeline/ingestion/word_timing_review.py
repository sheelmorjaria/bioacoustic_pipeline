"""Fail-closed human word-interval reference checks; never infer missing labels."""
TIMING_FIELDS = ('onset_sample','offset_sample','onset_uncertainty_samples','offset_uncertainty_samples')


def integer(value):
    if isinstance(value,bool) or str(value).strip() != str(int(value)):
        raise ValueError('Timing values must be native integer samples')
    return int(value)


def validate_row(row, samples):
    status = row.get('status','').strip()
    if status in ('','unreviewed'):
        if any(str(row.get(k,'')).strip() for k in (*TIMING_FIELDS,'evidence')):
            raise ValueError('Unreviewed rows cannot carry timing evidence')
        return {'supported':False,'status':'unreviewed'}
    if status in ('uncertain','unusable'):
        return {'supported':False,'status':status}
    if status != 'supported':
        raise ValueError('Unknown review status')
    try:
        values = {k:integer(row.get(k,'')) for k in TIMING_FIELDS}
    except (ValueError,TypeError) as exc:
        raise ValueError('Supported interval requires explicit integer samples and uncertainty') from exc
    if not 0 <= values['onset_sample'] < values['offset_sample'] <= samples:
        raise ValueError('Word interval outside recording or reversed')
    if not all(0 <= values[k] <= samples for k in TIMING_FIELDS[2:]):
        raise ValueError('Invalid uncertainty')
    if not str(row.get('evidence','')).strip():
        raise ValueError('Supported interval requires independent evidence')
    return {'supported':True,'status':status,**values}


def review_gate(rows_a, rows_b, meta_a, meta_b, adjudications, lengths):
    ids = [m.get('annotator_id','').strip() for m in (meta_a,meta_b)]
    if all(ids) and ids[0] == ids[1]:
        raise ValueError('Independent reviews require distinct annotator IDs')
    attested = all(ids) and all(m.get('independent_review') is True and
        m.get('model_boundaries_hidden') is True for m in (meta_a,meta_b))
    reviewed = []
    for rows in (rows_a,rows_b):
        mapping = {}
        for row in rows:
            target = row.get('target_id')
            if target not in lengths or target in mapping:
                raise ValueError('Unknown or duplicated review target')
            mapping[target] = validate_row(row,lengths[target])
        reviewed.append(mapping)
    if set(adjudications)-set(lengths):
        raise ValueError('Unknown adjudication target')
    approved,agreements = [],{}
    for target,samples in lengths.items():
        a,b = (r.get(target,{'supported':False}) for r in reviewed)
        if a['supported'] and b['supported']:
            agreements[target] = {k:abs(a[k]-b[k]) for k in TIMING_FIELDS[:2]}
        decision = adjudications.get(target,{})
        if decision.get('status') == 'approved':
            if not decision.get('adjudicator_id','').strip():
                raise ValueError('Adjudication needs an identified human reviewer')
            validated = validate_row({**decision,'status':'supported'},samples)
            if a['supported'] and b['supported'] and validated['supported']:
                approved.append(target)
    return {'ready':bool(lengths) and bool(attested) and len(approved) == len(lengths),
            'independence_attested':bool(attested),
            'supported_A':sum(r['supported'] for r in reviewed[0].values()),
            'supported_B':sum(r['supported'] for r in reviewed[1].values()),
            'manually_adjudicated_supported_targets':len(approved),
            'total_targets':len(lengths),'absolute_boundary_disagreements_samples':agreements,
            'scope':'Validated annotation intervals are not exact linguistic-information availability times.'}
