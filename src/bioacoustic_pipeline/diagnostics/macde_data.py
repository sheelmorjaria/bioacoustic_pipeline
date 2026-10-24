"""Native-rate, file-local observations for MACDE; no centered audio windows."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from bioacoustic_pipeline.diagnostics.acoustic_derivatives import _NAME_TO_IDX, supported_features
from bioacoustic_pipeline.diagnostics.acoustic_fingerprint import _dsp, physics

BAND_EDGES=np.geomspace(25.,125000.,65)


def frame_observations(audio,sr,*,max_frames=None,sample_valid=None):
    audio=np.asarray(audio,float)
    source=np.isfinite(audio) if sample_valid is None else np.asarray(sample_valid,bool) & np.isfinite(audio)
    hop=round(sr*.01)
    win=4*hop  # fractional native rates: h=4 must never overlap by one sample
    length=max(0,1+(len(audio)-win)//hop)
    if max_frames is not None:
        length=min(length,max_frames)
    spectral=np.full((length,64),np.nan,np.float32)
    physical=np.full((length,12),np.nan,np.float32)
    sv=np.zeros((length,64),bool)
    pv=np.zeros((length,12),bool)
    source_valid=np.zeros(length,bool)
    active=np.zeros(length,bool)
    freqs=np.fft.rfftfreq(win,1/sr)
    frequency_bins=[np.where((freqs>=a)&(freqs<b))[0] for a,b in zip(BAND_EDGES[:-1],BAND_EDGES[1:])]
    for i in range(length):
        start=i*hop
        x=audio[start:start+win]
        source_valid[i]=source[start:start+win].all()
        if not source_valid[i]:
            continue
        rms=np.sqrt(np.mean(x*x))
        active[i]=rms>=1e-3  # same absolute gate as the frozen fingerprint, never peak-normalize
        if not active[i]:
            continue
        spectrum=np.abs(np.fft.rfft(x*np.hanning(win)))**2/(win*sr)
        for band,idx in enumerate(frequency_bins):
            if idx.size and BAND_EDGES[band+1]<=sr/2:
                spectral[i,band]=np.log(np.mean(spectrum[idx])+1e-20)
                sv[i,band]=True
        vec,quality=physics.physics(x,sr)
        env=_dsp.envelope(x,sr)
        dyn_cv=float(env.std()/(env.mean()+_dsp.EPS)) if env.size>1 else 0.
        physical[i]=[vec[_NAME_TO_IDX[n]] if _NAME_TO_IDX[n]<15 else dyn_cv for n in supported_features()]
        pv[i]=np.isfinite(physical[i]) & not_quality(quality)
    valid=source_valid & active & pv.all(1) & sv.any(1)
    return {'spectral':spectral,'spectral_valid':sv,'physical':physical,'physical_valid':pv,
            'source_valid':source_valid,'detected_activity':active,'time_valid':valid,
            'frame_end_samples':np.arange(length)*hop+win,'sample_rate':sr}


def not_quality(quality):
    return not (quality['too_short'] or quality['low_energy'])


def waveform_detail_reference(audio,sr):
    """Independent full-waveform log-centroid, log-bandwidth, envelope CV."""
    audio=np.asarray(audio,float)
    power=np.abs(np.fft.rfft(audio*np.hanning(len(audio))))**2
    f=np.fft.rfftfreq(len(audio),1/sr)
    power=power/max(power.sum(),1e-20)
    centroid=float((f*power).sum())
    spread=float(np.sqrt(((f-centroid)**2*power).sum()))
    env=np.abs(hilbert(audio))
    return np.array([np.log1p(centroid),np.log1p(spread),env.std()/max(env.mean(),1e-20)])
