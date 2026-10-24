"""Multiscale Acoustic Change and Detail Encoder: causal, masked, 64-D.

No boundaries, unit labels or social annotations. Forecast targets are frozen
physical measurements, never a jointly learned latent state. Surprise is only
model-relative residual error. Unsupported values never enter a convolution.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHANGE_LAGS=(2,5,10,20)  # 10 ms hop; proposed acoustic scales, not biological units
CONTEXT_FRAMES=52       # 32-frame TCN plus oldest 20-frame change lag
HORIZONS=(4,8,20)       # 40 ms windows: all target windows prospectively nonoverlapping


class MaskedStandardizer:
    def fit(self, values, supported):
        values=np.asarray(values,float)
        ok=np.asarray(supported,bool) & np.isfinite(values)
        flat=values.reshape(-1,values.shape[-1])
        mask=ok.reshape(flat.shape)
        count=mask.sum(0)
        safe=np.where(mask,flat,0.)
        self.mean=safe.sum(0)/np.maximum(count,1)
        variance=np.where(mask,(flat-self.mean)**2,0.).sum(0)/np.maximum(count,1)
        self.scale=np.where(variance>1e-10,np.sqrt(variance),1.)
        self.support=count>0
        return self

    def transform(self, values, supported):
        values=np.asarray(values,float)
        ok=np.asarray(supported,bool) & np.isfinite(values) & self.support
        safe=np.where(ok,values,self.mean)
        return np.where(ok,(safe-self.mean)/self.scale,0.).astype(np.float32),ok

    def state(self):
        return {'mean':self.mean.tolist(),'scale':self.scale.tolist(),'support':self.support.tolist()}


def lag_support(valid, lag):
    if lag>=valid.shape[1]:
        return torch.zeros_like(valid)
    runs=(~valid).long().cumsum(1)
    keep=valid[:,lag:] & valid[:,:-lag] & (runs[:,lag:]==runs[:,:-lag])
    return F.pad(keep,(lag,0),value=False)


def multiscale_changes(state, valid, lags=CHANGE_LAGS):
    deltas,masks=[],[]
    for lag in lags:
        keep=lag_support(valid,lag)
        delta=torch.zeros_like(state) if lag>=state.shape[1] else F.pad(state[:,lag:]-state[:,:-lag],(0,0,lag,0))
        deltas.append(torch.where(keep[...,None],delta,0.))
        masks.append(keep)
    return torch.stack(deltas,2),torch.stack(masks,2)


def eligible_mask(valid, history=CONTEXT_FRAMES, horizon=max(HORIZONS)):
    result=torch.zeros_like(valid)
    for t in range(history-1,valid.shape[1]-horizon):
        result[:,t]=valid[:,t-history+1:t+horizon+1].all(1)
    return result


class CausalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.current=nn.Linear(width,width)
        self.previous=nn.Linear(width,width,bias=False)
        self.norm=nn.LayerNorm(width)
        self.dilation=dilation

    def forward(self,state,valid):
        lag=self.dilation
        past=torch.zeros_like(state) if lag>=state.shape[1] else F.pad(state[:,:-lag],(0,0,lag,0))
        past=torch.where(lag_support(valid,lag)[...,None],past,0.)
        output=self.norm(state+F.gelu(self.current(state)+self.previous(past)))
        return torch.where(valid[...,None],output,0.)


class MACDE(nn.Module):
    def __init__(self, *, use_physical=True, bands=64, physical_dim=12, width=64):
        super().__init__()
        self.use_physical=use_physical
        self.bands=bands
        self.width=width
        self.spectral_encoder=nn.Sequential(nn.Conv1d(2,8,5,padding=2),nn.GELU(),
            nn.Conv1d(8,16,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(4),nn.Flatten(),nn.Linear(64,32))
        self.physical_encoder=nn.Sequential(nn.Linear(physical_dim*2,32),nn.GELU(),nn.Linear(32,32))
        self.fusion=nn.Linear(64,width)
        self.context=nn.ModuleList([CausalBlock(width,d) for d in (1,2,4,8,16)])
        self.forecast_head=nn.Linear(width,physical_dim)
        self.change_head=nn.Linear(width*len(CHANGE_LAGS),physical_dim)

    def forward(self,spectral,physical,valid,spectral_valid=None,physical_valid=None):
        sm=(torch.isfinite(spectral) if spectral_valid is None else spectral_valid & torch.isfinite(spectral)) & valid[...,None]
        pm=(torch.isfinite(physical) if physical_valid is None else physical_valid & torch.isfinite(physical)) & valid[...,None]
        spectral=torch.where(sm,spectral,0.)
        physical=torch.where(pm,physical,0.)
        batch,length,_=spectral.shape
        spec=torch.stack([spectral,sm.to(spectral.dtype)],2).reshape(batch*length,2,self.bands)
        spec=self.spectral_encoder(spec).reshape(batch,length,32)
        phys=self.physical_encoder(torch.cat([physical,pm.to(physical.dtype)],-1)) if self.use_physical else torch.zeros_like(spec)
        state=torch.where(valid[...,None],self.fusion(torch.cat([spec,phys],-1)),0.)
        for block in self.context:
            state=block(state,valid)
        changes,change_valid=multiscale_changes(state,valid)
        return {'state':state,'changes':changes,'change_valid':change_valid,
                'forecast':torch.where(valid[...,None],self.forecast_head(state),0.),
                'change_forecast':torch.where(valid[...,None],self.change_head(changes.flatten(2)),0.)}


def forecast_objective(output,physical,valid,*,horizon=4,aux_weight=0.):
    if horizon<4:
        raise ValueError('Forecast target windows overlap input audio')
    keep=eligible_mask(valid)[:,:-horizon]
    if not keep.any():
        raise ValueError('No contiguous support for the declared objective')
    target=physical[:,horizon:].detach()
    prediction=output['forecast'][:,:-horizon]
    loss=F.mse_loss(prediction[keep],target[keep])
    if aux_weight:
        change=(target-physical[:,:-horizon].detach())
        loss=loss+aux_weight*F.mse_loss(output['change_forecast'][:,:-horizon][keep],change[keep])
    return loss
