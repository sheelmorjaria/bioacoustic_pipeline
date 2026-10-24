"""Reusable MACDE export must retain masks, clocks and stage separation."""
import numpy as np
import pytest
import torch

from bioacoustic_pipeline.embeddings.macde import MACDE,MaskedStandardizer
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.embeddings.macde_bundle import load_macde


def bundle(tmp_path):
    sr=16000
    t=np.arange(24000)/sr
    audio=.2*np.sin(2*np.pi*1200*t)*(1+.2*np.sin(2*np.pi*9*t))
    obs=frame_observations(audio,sr)
    scalers={k:MaskedStandardizer().fit(obs[k],obs[k+'_valid']).state() for k in ('spectral','physical')}
    torch.manual_seed(0)
    path=tmp_path/'encoder.pt'
    torch.save({'state_dict':MACDE().state_dict(),'scalers':scalers,'width':64,
                'use_physical':True,'stage2_ready':False},path)
    return path,audio,sr


def test_bundle_exports_continuous_states_with_complete_warmup_masks(tmp_path):
    path,audio,sr=bundle(tmp_path)
    out=load_macde(path).encode_audio(audio,sr)
    assert out['state'].shape[1]==64
    assert not out['state_valid'][:31].any()
    assert out['state_valid'][31:].all()
    assert np.isnan(out['state'][:31]).all()
    assert not out['change_valid'][:51,-1].any()
    assert out['change_valid'][51:,-1].all()
    assert not out['stage2_ready']


def test_residual_is_aligned_to_observed_future_not_current_state(tmp_path):
    path,audio,sr=bundle(tmp_path)
    out=load_macde(path).encode_audio(audio,sr)
    obs=frame_observations(audio,sr)
    keep=out['residual_valid'][4:]
    expected=obs['physical'][4:]-out['forecast_physical'][:-4]
    np.testing.assert_allclose(out['forecast_residual_physical'][4:][keep],expected[keep],rtol=1e-5,atol=1e-5)
    assert not out['residual_valid'][:35].any()


def test_bundle_hash_mismatch_fails_before_loading(tmp_path):
    path,_,_=bundle(tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        load_macde(path,expected_sha256='0'*64)


def test_bundle_rejects_feature_schema_drift(tmp_path):
    path,_,_=bundle(tmp_path)
    payload=torch.load(path,weights_only=True)
    payload['physical_feature_names']=['wrong']*12
    torch.save(payload,path)
    with pytest.raises(ValueError,match='feature'):
        load_macde(path)
