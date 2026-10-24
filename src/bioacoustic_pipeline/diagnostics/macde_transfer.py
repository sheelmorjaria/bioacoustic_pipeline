"""Frozen grouping and matched-history controls for MACDE transfer."""
import hashlib
import re
from pathlib import Path

import numpy as np


def recording_group(domain, name):
    name = Path(name).name
    expressions = {'pipit': r'(pipit\d{4}fg_(?:train|test)_\d+)_\d+\.wav',
                   'chiffchaff': r'cutted_day(\d+)_PC\d+_\d+\.wav',
                   'littleowl': r'(littleowl\d{4}fg_(?:train|test)_[^_]+)_\d+\.wav',
                   'marmoset': r'(\d{8})_.+\.wav'}
    match = re.fullmatch(expressions[domain], name)
    if match is None:
        raise ValueError(f'Unresolved recording group: {name}')
    return ('day' if domain == 'chiffchaff' else '') + re.sub(r'_(train|test)_', '_', match.group(1))


def partition(domain, group):
    value = int(hashlib.sha256(f'macde-transfer-v1:{domain}:{group}'.encode()).hexdigest(), 16) % 100
    return 'fit' if value < 60 else ('validation' if value < 80 else 'heldout')


def history_order(seed, *, disrupted):
    if not disrupted:
        return np.arange(52)
    return np.r_[np.random.default_rng(seed).permutation(50), 50, 51]
