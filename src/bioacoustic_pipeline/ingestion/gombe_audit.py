"""Separate Gombe catalogue entries and waveform support from session evidence."""
import numpy as np
import torch
from bioacoustic_pipeline.embeddings.macde import eligible_mask


def catalogue_rows(rows):
    return [r for r in rows if r and isinstance(r[0], (int, float)) and np.isfinite(r[0]) and r[0] > 0]


def support_summary(valid, sr):
    valid = np.asarray(valid, bool)
    changes = np.diff(np.r_[False, valid, False].astype(int))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    longest = int(lengths.max()) if len(lengths) else 0
    hop = round(sr*.01)
    count = int(eligible_mask(torch.from_numpy(valid)[None])[0].sum()) if len(valid) else 0
    return {'frames': len(valid), 'valid_frames': int(valid.sum()), 'valid_fraction': float(valid.mean()) if len(valid) else None,
            'longest_valid_run_frames': longest,
            'longest_valid_run_waveform_seconds': ((longest+3)*hop/sr) if longest else 0.,
            'runs_at_least128_frames': int((lengths >= 128).sum()),
            'technical_eligible_positions': count, 'certified_cut_local_positions': 0,
            'certified_independent_sessions': 0,
            'limitation': 'Technical support only; cut edits and vocal-source identity are not masked, so eligible positions cannot be certified as within one cut or chimpanzee vocalization'}
