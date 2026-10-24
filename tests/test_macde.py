"""MACDE mechanics: causality, support, independent targets and framing."""
import numpy as np
import pytest
import torch

from bioacoustic_pipeline.embeddings.macde import (
    MACDE, MaskedStandardizer, eligible_mask, multiscale_changes, forecast_objective,
)
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations


def inputs(length=96):
    torch.manual_seed(4)
    return torch.randn(2,length,64),torch.randn(2,length,12),torch.ones(2,length,dtype=torch.bool)


def test_future_changes_cannot_change_current_or_past_states():
    model=MACDE().eval()
    spectral,physical,valid=inputs()
    first=model(spectral,physical,valid)['state']
    spectral[:,50:]+=100
    physical[:,50:]-=100
    second=model(spectral,physical,valid)['state']
    torch.testing.assert_close(first[:,:50],second[:,:50])


def test_invalid_gaps_reset_context_and_nan_values_are_inert():
    model=MACDE().eval()
    spectral,physical,valid=inputs()
    valid[:,30:35]=False
    before=model(spectral,physical,valid)['state']
    spectral[:,:30]+=100
    spectral[:,30:35]=torch.nan
    physical[:,:30]-=100
    physical[:,30:35]=torch.nan
    after=model(spectral,physical,valid)['state']
    torch.testing.assert_close(before[:,35:],after[:,35:])
    assert torch.isfinite(after).all()
    assert torch.count_nonzero(after[:,30:35])==0


def test_appended_padding_leaves_states_changes_and_eligibility_unchanged():
    model=MACDE().eval()
    a,b,mask=inputs()
    original=model(a,b,mask)
    padded=model(torch.cat([a,torch.full((2,10,64),torch.nan)],1),
                 torch.cat([b,torch.full((2,10,12),torch.nan)],1),
                 torch.cat([mask,torch.zeros(2,10,dtype=torch.bool)],1))
    torch.testing.assert_close(original['state'],padded['state'][:,:96])
    torch.testing.assert_close(original['changes'],padded['changes'][:,:96])
    torch.testing.assert_close(eligible_mask(mask),eligible_mask(torch.cat([mask,torch.zeros(2,10,dtype=torch.bool)],1))[:,:96])


def test_multiscale_differences_do_not_cross_gaps():
    state=torch.arange(12.).view(1,12,1)
    mask=torch.ones(1,12,dtype=torch.bool)
    mask[:,5]=False
    delta,support=multiscale_changes(state,mask,lags=(2,5))
    assert delta[0,4,0,0]==2
    assert not support[0,6,0]
    assert not support[0,9,1]
    assert support[0,11,1]


def test_scaler_ignores_unsupported_values_and_never_refits_during_transform():
    x=np.array([[1.,np.nan],[3.,1000.],[1e9,2000.]])
    mask=np.array([[True,False],[True,False],[False,False]])
    scale=MaskedStandardizer().fit(x,mask)
    before=scale.mean.copy()
    z,support=scale.transform(x,mask)
    np.testing.assert_array_equal(scale.mean,before)
    np.testing.assert_array_equal(z[:,1],0)
    assert not support[:,1].any()
    np.testing.assert_allclose(z[:2,0],[-1,1])


def test_targets_are_frozen_physical_measurements_and_empty_support_abstains():
    model=MACDE()
    a,b,valid=inputs()
    output=model(a,b,valid)
    target=b.clone().requires_grad_(True)
    loss=forecast_objective(output,target,valid,aux_weight=.1)
    loss.backward()
    assert target.grad is None
    assert model.forecast_head.weight.grad is not None
    assert torch.isfinite(loss)
    with pytest.raises(ValueError,match='support'):
        forecast_objective(output,target,torch.zeros_like(valid))
    with pytest.raises(ValueError,match='overlap'):
        forecast_objective(output,target,valid,horizon=2)


def test_frontend_has_no_centered_or_future_audio_and_retains_frequency_support():
    sr=16000
    time=np.arange(sr//2)/sr
    audio=.2*np.sin(2*np.pi*1000*time)
    first=frame_observations(audio,sr)
    changed=audio.copy()
    changed[4000:]=5
    second=frame_observations(changed,sr)
    keep=first['frame_end_samples']<=4000
    np.testing.assert_array_equal(first['spectral'][keep],second['spectral'][keep])
    assert first['spectral'].shape[1]==64
    assert first['physical'].shape[1]==12
    assert not first['spectral_valid'][:,-1].any()  # bands above Nyquist are unavailable
    assert first['source_valid'].all()
    assert first['time_valid'].any()


def test_physical_reference_exactly_matches_existing_supported_fingerprint():
    from bioacoustic_pipeline.diagnostics.acoustic_fingerprint import extract
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import _NAME_TO_IDX, supported_features
    sr=16000
    time=np.arange(640)/sr
    waveform=.15*np.sin(2*np.pi*900*time)*(1+.4*np.sin(2*np.pi*10*time))
    actual=frame_observations(waveform,sr)['physical'][0]
    expected=extract(waveform,sr).combined[[_NAME_TO_IDX[n] for n in supported_features()]]
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


def test_longest_multiscale_view_requires_full_contiguous_receptive_support():
    mask=torch.ones(1,128,dtype=torch.bool)
    expected=eligible_mask(mask)
    assert torch.where(expected[0])[0][0]==51
    assert torch.where(expected[0])[0][-1]==107
    mask[:,60]=False
    gated=eligible_mask(mask)
    assert not gated.any()  # neither side contains 52-history + 20-future frames


def test_fractional_sample_rate_never_introduces_a_one_sample_forecast_overlap():
    sr=24414.0625
    time=np.arange(10000)/sr
    observed=frame_observations(.2*np.sin(2*np.pi*1000*time),sr)
    end=observed['frame_end_samples']
    first_target_start=end[4]-end[0]
    assert end[0]<=first_target_start


def test_r2_r3_have_identical_architectures_and_auxiliary_uses_all_scales():
    first=MACDE()
    second=MACDE()
    assert [(n,tuple(p.shape)) for n,p in first.named_parameters()]==[(n,tuple(p.shape)) for n,p in second.named_parameters()]
    assert first.change_head.in_features==4*64


def test_recorded_silence_is_source_valid_but_not_supported_acoustic_evidence():
    observed=frame_observations(np.zeros(8000),16000)
    assert observed['source_valid'].all()
    assert not observed['detected_activity'].any()
    assert not observed['physical_valid'].any()
    assert not observed['time_valid'].any()
