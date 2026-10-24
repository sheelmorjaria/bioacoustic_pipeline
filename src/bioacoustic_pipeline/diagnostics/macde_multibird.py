"""Bird-isolated splitting,preprocessing and training-side comparator selection."""
import numpy as np
import torch
from bioacoustic_pipeline.embeddings.macde import MaskedStandardizer


def bird_folds(row_birds, heldout):
    birds = np.asarray(row_birds)
    unique = sorted(set(birds))
    if len(unique) != 4 or heldout not in unique:
        raise ValueError('Exactly four identified birds required')
    train = np.flatnonzero(birds != heldout)
    evaluation = np.flatnonzero(birds == heldout)
    inner = []
    for val in unique:
        if val != heldout:
            inner.append((np.flatnonzero((birds != heldout) & (birds != val)),
                          np.flatnonzero(birds == val)))
    return train,evaluation,inner


def fit_preprocessing(arrays, train):
    data,states = {'valid':torch.from_numpy(arrays['time_valid'])},{}
    for key in ('spectral','physical'):
        mask = arrays[key+'_valid'] & arrays['time_valid'][...,None]
        scale = MaskedStandardizer().fit(arrays[key][train],mask[train])
        values,valid = scale.transform(arrays[key],mask)
        data[key],data[key+'_valid'] = torch.from_numpy(values),torch.from_numpy(valid)
        states[key] = scale.state()
    target_sd = np.maximum(arrays['target'][train].std(0),1e-6)
    return data,states,target_sd


def select_ridge(scores):
    alphas = {family:min(values,key=lambda alpha:(float(np.mean(values[alpha])),alpha))
              for family,values in scores.items()}
    family = min(scores,key=lambda name:(float(np.mean(scores[name][alphas[name]])),
                                         name != 'R0',alphas[name]))
    return {'alphas':alphas,'primary_R0_alpha':alphas['R0'],'selected_family':family,
            'selected_family_alpha':alphas[family],
            'inner_macro_nrmse':{name:{str(a):float(np.mean(v)) for a,v in values.items()}
                                for name,values in scores.items()}}
