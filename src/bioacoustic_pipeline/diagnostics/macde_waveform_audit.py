"""Apply the closed transfer study's source-window rule without fitting."""
import numpy as np
import torch
from .macde_data import frame_observations
from bioacoustic_pipeline.embeddings.macde import eligible_mask


def window_decision(valid):
    valid = np.asarray(valid, bool)
    if len(valid) < 128:
        return {'passes_length': False, 'passes_acoustic_support': False, 'selected_start_frame': None, 'eligible_positions': 0}
    positions = np.flatnonzero(eligible_mask(torch.from_numpy(valid)[None])[0].numpy())
    if not len(positions):
        return {'passes_length': True, 'passes_acoustic_support': False, 'selected_start_frame': None, 'eligible_positions': 0}
    start = min(max(0, int(positions[0])-51), len(valid)-128)
    count = int(eligible_mask(torch.from_numpy(valid[start:start+128])[None])[0].sum())
    return {'passes_length': True, 'passes_acoustic_support': count > 0,
            'selected_start_frame': start, 'eligible_positions': count}


def audit_audio(audio, sr):
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(1)
    hop = round(sr*.01)
    audio = audio[:hop*259]
    obs = frame_observations(audio, sr, max_frames=256)
    return window_decision(obs['time_valid']) | {'sample_rate': sr, 'audited_frames': len(obs['time_valid'])}
