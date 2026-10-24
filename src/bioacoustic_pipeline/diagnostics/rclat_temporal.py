"""Two-scale temporal controls with common sample-exact tapered-cell inventory."""
import numpy as np
from .rclat import block_order


def tapered_cells(audio, sr):
    x = np.asarray(audio)
    if x.ndim != 1 or x.dtype != np.dtype('int16') or sr != 16000:
        raise ValueError('Native mono PCM16 at16kHz required')
    width,fade = round(.02*sr),round(.001*sr)
    count = len(x)//width
    window = np.ones(width)
    window[:fade] = np.sin(np.linspace(0,np.pi/2,fade))**2
    window[-fade:] = window[:fade][::-1]
    y = x.copy()
    y[:count*width] = np.rint(x[:count*width].reshape(count,width)*window).astype(np.int16).reshape(-1)
    return y


def rearrange(sham, sr, policy, seed):
    width,macro = round(.02*sr),25
    count = len(sham)//(width*macro)
    if count < 2:
        raise ValueError('At least two500ms groups required')
    body = sham[:count*width*macro].reshape(count,macro,width)
    if policy == 'global':
        result = body[block_order(count,seed)].reshape(-1)
    elif policy == 'local':
        rng = np.random.default_rng(seed)
        result = np.stack([group[block_order(macro,int(rng.integers(0,2**63)))] for group in body]).reshape(-1)
    else:
        raise ValueError('Unknown temporal policy')
    return np.r_[result,sham[count*width*macro:]].astype(np.int16)


def rmse(a,b):
    return float(np.sqrt(np.mean(((a.astype(float)-b.astype(float))/32768.)**2)))


def match_distortion(sham, sr, *, seed, candidates=32):
    if candidates < 1:
        raise ValueError('Positive candidate count required')
    global_rmse = np.array([rmse(rearrange(sham,sr,'global',seed+i),sham) for i in range(candidates)])
    local_rmse = np.array([rmse(rearrange(sham,sr,'local',seed+i),sham) for i in range(candidates)])
    differences = np.abs(global_rmse[:,None]-local_rmse[None,:])/np.maximum(
        np.maximum(global_rmse[:,None],local_rmse[None,:]),1e-20)
    i,j = np.unravel_index(np.argmin(differences),differences.shape)
    info = {'global_seed':int(seed+i),'local_seed':int(seed+j),
            'global_rmse_vs_sham':float(global_rmse[i]),'local_rmse_vs_sham':float(local_rmse[j]),
            'relative_rmse_mismatch':float(differences[i,j]),'candidates_per_policy':candidates}
    return rearrange(sham,sr,'global',seed+i),rearrange(sham,sr,'local',seed+j),info
