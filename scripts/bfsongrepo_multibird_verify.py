#!/usr/bin/env python3
"""Verify completed predictions and export safe inference checkpoint copies.

Original locally generated training artifacts are checksum-verified before
loading. Copies normalize NumPy string metadata to builtin strings without
changing any parameter; original frozen artifacts remain untouched.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.macde_transfer import history_order
from bioacoustic_pipeline.embeddings.macde_bundle import FrozenMACDE

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {canonical(k):canonical(v) for k,v in value.items()}
    if isinstance(value,list):
        return [canonical(v) for v in value]
    if isinstance(value,tuple):
        return tuple(canonical(v) for v in value)
    return value


def main():
    torch.set_num_threads(2)
    output = ROOT/'outputs/bfsongrepo_multibird_checkpoint_verification.json'
    if output.exists():
        raise RuntimeError('Verification already exists')
    result_path = ROOT/'outputs/bfsongrepo_multibird_results.json'
    spec_path = ROOT/'outputs/bfsongrepo_multibird_preregistration.json'
    result,spec = json.loads(result_path.read_text()),json.loads(spec_path.read_text())
    if result['preregistration_sha256'] != sha(spec_path):
        raise ValueError('Specification mismatch')
    for path,digest in spec['dependencies'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed')
    if sha(ROOT/result['residual_artifact']['path']) != result['residual_artifact']['sha256']:
        raise ValueError('Residual mismatch')
    with np.load(ROOT/'cache/bfsongrepo_frozen_transfer.npz') as z:
        arrays = {k:z[k] for k in ('physical','spectral','physical_valid','spectral_valid','target')}
    checks = []
    with np.load(ROOT/result['residual_artifact']['path']) as residuals:
        for bird,models in result['checkpoints'].items():
            row = spec['folds'][bird]['evaluation_rows'][0]
            report_sd = np.array(result['evaluations'][bird]['reporting_target_sd'])
            for arm in ('R2_multi','R2_multi_shuffled'):
                entry = models[arm]
                original = ROOT/entry['path']
                if sha(original) != entry['sha256']:
                    raise ValueError('Training checkpoint mismatch')
                # Trusted local output of the just-verified runner,not an external pickle.
                payload = torch.load(original,map_location='cpu',weights_only=False)
                inference = original.with_suffix('.inference.pt')
                if inference.exists():
                    raise RuntimeError('Inference copy already exists')
                torch.save(canonical(payload),inference)
                safe = torch.load(inference,map_location='cpu',weights_only=True)
                assert payload['state_dict'].keys() == safe['state_dict'].keys()
                assert all(torch.equal(value,safe['state_dict'][key]) for key,value in payload['state_dict'].items())
                assert bird not in safe['training_birds'] and safe['heldout_bird'] == bird
                encoder = FrozenMACDE(safe)
                disrupted = arm == 'R2_multi_shuffled'
                seed = int.from_bytes(hashlib.sha256(f'{row}:51'.encode()).digest()[:8],'little')
                order = history_order(seed,disrupted=disrupted)
                data = {}
                for key in ('spectral','physical'):
                    value,mask = frozen_scale(arrays[key][row],arrays[key+'_valid'][row],safe['scalers'][key])
                    data[key] = torch.from_numpy(value[order])[None]
                    data[key+'_valid'] = torch.from_numpy(mask[order])[None]
                with torch.no_grad():
                    prediction = encoder.model(data['spectral'],data['physical'],torch.ones((1,52),dtype=torch.bool),data['spectral_valid'],data['physical_valid'])['forecast'][0,-1].numpy()
                mu,scale = np.array(safe['scalers']['physical']['mean']),np.array(safe['scalers']['physical']['scale'])
                residual = (prediction*scale+mu-arrays['target'][row])/report_sd
                expected = residuals[bird+'__'+arm][0]
                np.testing.assert_allclose(residual,expected,atol=1e-5,rtol=0)
                checks.append({'heldout_bird':bird,'arm':arm,'original_path':entry['path'],
                    'original_sha256':entry['sha256'],'inference_path':str(inference.relative_to(ROOT)),
                    'inference_sha256':sha(inference),'state_tensors_exact':True,'weights_only_reload_passed':True,
                    'heldout_bird_excluded':True,'saved_residual_match_max_abs':float(np.max(np.abs(residual-expected)))})
    report = {'result_sha256':sha(result_path),'preregistration_sha256':sha(spec_path),
        'checkpoint_checks':checks,'all_eight_checkpoints_verified':len(checks)==8,
        'metadata_only_conversion':'NumPy string scalars converted to builtin strings in additive inference copies;original training checkpoints and all tensors unchanged.',
        'stage2_ready':False,'prior_dependencies_unchanged':True}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Verified 8 checkpoints, safe reloads and saved predictions; prior results unchanged.')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
