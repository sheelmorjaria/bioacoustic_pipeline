"""Parse explicit source references without promoting filenames to session evidence."""
from datetime import datetime
import re


def zhao_reference(filename):
    match = re.search(r'(pairisolation_cj\d+_cj\d+)_(\d{8})_(\d{6})\.wav$', filename.replace('\\', '/'))
    if not match:
        raise ValueError('Unresolved original Zhao audio reference')
    stamp = datetime.strptime(match[2] + match[3], '%Y%m%d%H%M%S')
    return {'recording_reference': match[0], 'subject_tokens': match[1].split('_')[1:],
            'filename_timestamp': stamp.isoformat(), 'timestamp_status': 'filename encoded; not independently certified session',
            'independence_status': 'unresolved'}


def certification_level(*, integrity, acoustic, provenance, independent):
    if independent and not provenance:
        raise ValueError('Independent groups require resolved provenance')
    if provenance and not acoustic:
        raise ValueError('This acoustic-study hierarchy requires waveform eligibility before promotion')
    if acoustic and not integrity:
        raise ValueError('Acoustic eligibility requires verified source integrity')
    return 'S3' if independent else 'S2' if provenance else 'S1' if acoustic else 'S0' if integrity else 'unverified'
