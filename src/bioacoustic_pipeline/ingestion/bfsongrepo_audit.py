"""File-local BFSongRepo source/continuity checks;annotations never select frames."""
from datetime import datetime
import re
import numpy as np
import torch
from bioacoustic_pipeline.embeddings.macde import eligible_mask


def rec_metadata(text):
    values = {}
    for field, key in [('ADFREQ', 'sample_rate'), ('Chans', 'channels'), ('Samples', 'samples')]:
        match = re.search(r'^'+field+r'\s*=\s*([0-9.Ee+\-]+)', text, re.M)
        if match is None:
            raise ValueError('Missing REC field: '+field)
        values[key] = int(float(match[1]))
    match = re.search(r'^File created:\s*(.+)$', text, re.M)
    values['file_created'] = datetime.strptime(match[1].strip(), '%a, %b %d, %Y, %H:%M:%S').isoformat() if match else None
    return values


def decode_cbin_channel(payload, metadata):
    if len(payload) % (2*metadata['channels']):
        raise ValueError('Incomplete interleaved CBIN frame')
    audio = np.frombuffer(payload, dtype='>i2')[::metadata['channels']]
    if len(audio) != metadata['samples']:
        raise ValueError('CBIN sample count differs from REC')
    return audio


def continuity_summary(valid, sr):
    valid = np.asarray(valid, bool)
    diff = np.diff(np.r_[False, valid, False].astype(int))
    starts, ends = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    lengths = ends-starts
    strict = np.zeros_like(valid)
    for a, b in zip(starts, ends):
        if b-a >= 128:
            strict[a:b] = True
    longest = int(lengths.max()) if len(lengths) else 0
    eligible = int(eligible_mask(torch.from_numpy(valid)[None])[0].sum()) if len(valid) >= 128 else 0
    strict_n = int(eligible_mask(torch.from_numpy(strict)[None])[0].sum()) if len(valid) >= 128 else 0
    return {'frames': len(valid), 'valid_frames': int(valid.sum()),
            'longest_valid_run_frames': longest,
            'longest_valid_run_seconds': (longest+3)*round(sr*.01)/sr if longest else 0.,
            'runs_at_least128_frames': int((lengths >= 128).sum()),
            'technical_forecast_positions': eligible,
            'positions_in_runs_at_least128_frames': strict_n}
