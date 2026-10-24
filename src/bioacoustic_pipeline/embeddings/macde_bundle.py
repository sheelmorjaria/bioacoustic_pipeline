"""Frozen continuous-audio inference for MACDE stage-1 checkpoints.

Exports unavailable states as NaN with explicit support masks. Prediction
residuals appear only once their future waveform has actually been observed.
This API does not discover boundaries or authorize CAVEM receiver experiments.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .macde import MACDE, MaskedStandardizer, eligible_mask
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations, BAND_EDGES
from bioacoustic_pipeline.diagnostics.acoustic_derivatives import supported_features


def load_macde(path, *, expected_sha256=None):
    path=Path(path)
    if expected_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('MACDE checkpoint checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    return FrozenMACDE(payload)


class FrozenMACDE:
    def __init__(self,payload):
        if 'physical_feature_names' in payload and tuple(payload['physical_feature_names'])!=supported_features():
            raise ValueError('MACDE physical feature schema differs from frozen checkpoint')
        if 'frequency_band_edges_hz' in payload and not np.array_equal(payload['frequency_band_edges_hz'],BAND_EDGES):
            raise ValueError('MACDE spectral feature grid differs from frozen checkpoint')
        self.model=MACDE(use_physical=payload['use_physical'],width=payload['width']).eval()
        self.model.load_state_dict(payload['state_dict'])
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.scalers={}
        for kind in ('spectral','physical'):
            scale=MaskedStandardizer()
            scale.mean=np.asarray(payload['scalers'][kind]['mean'])
            scale.scale=np.asarray(payload['scalers'][kind]['scale'])
            scale.support=np.asarray(payload['scalers'][kind]['support'],bool)
            self.scalers[kind]=scale
        self.stage2_ready=bool(payload.get('stage2_ready',False))

    def encode_audio(self,audio,sample_rate,*,sample_valid=None):
        obs=frame_observations(audio,sample_rate,sample_valid=sample_valid)
        data={}
        for kind in ('spectral','physical'):
            x,mask=self.scalers[kind].transform(obs[kind],obs[kind+'_valid'] & obs['time_valid'][:,None])
            data[kind]=torch.from_numpy(x)[None]
            data[kind+'_valid']=torch.from_numpy(mask)[None]
        valid=torch.from_numpy(obs['time_valid'])[None]
        if not len(obs['time_valid']):
            raise ValueError('Audio shorter than one complete observation window')
        with torch.no_grad():
            result=self.model(data['spectral'],data['physical'],valid,data['spectral_valid'],data['physical_valid'])
        state_valid=eligible_mask(valid,history=32,horizon=0)[0].numpy()
        change_valid=np.stack([eligible_mask(valid,history=32+lag,horizon=0)[0].numpy() for lag in (2,5,10,20)],1)
        state=result['state'][0].numpy().copy()
        state[~state_valid]=np.nan
        changes=result['changes'][0].numpy().copy()
        changes[~change_valid]=np.nan
        physical_scale=self.scalers['physical']
        forecast=result['forecast'][0].numpy()*physical_scale.scale+physical_scale.mean
        forecast[~state_valid]=np.nan
        residual=np.full_like(forecast,np.nan)
        residual_valid=np.zeros(len(forecast),bool)
        if len(forecast)>4:
            source=eligible_mask(valid,history=32,horizon=4)[0].numpy()[:-4]
            residual_valid[4:]=source & obs['physical_valid'][4:].all(1)
            delta=obs['physical'][4:]-forecast[:-4]
            residual[4:]=np.where(residual_valid[4:,None],delta,np.nan)
        return {'state':state,'state_valid':state_valid,'changes':changes,'change_valid':change_valid,
                'forecast_physical':forecast,'forecast_residual_physical':residual,
                'residual_valid':residual_valid,'frame_end_seconds':obs['frame_end_samples']/sample_rate,
                'source_valid':obs['source_valid'],'detected_activity':obs['detected_activity'],
                'stage2_ready':self.stage2_ready,
                'residual_interpretation':'model-relative physical acoustic prediction error; no biological surprise or boundary claim'}
