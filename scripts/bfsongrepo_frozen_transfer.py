#!/usr/bin/env python3
"""One frozen end-to-end transfer evaluation; no target-bird adaptation."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import macde_stage1 as base
import macde_transfer as neural
from bioacoustic_pipeline.contextual.bfsongrepo_development import selected_positions
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale, frozen_ridge
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import rec_metadata, decode_cbin_channel
from bioacoustic_pipeline.embeddings.macde import MACDE
from macde_source_resolution import pcm_hash

ROOT = base.ROOT
PREFIX = ROOT/'outputs/bfsongrepo_frozen_transfer'
SPEC = Path(str(PREFIX)+'_preregistration.json')
MANIFEST = Path(str(PREFIX)+'_population.json')
RESULT = Path(str(PREFIX)+'_results.json')
CACHE = ROOT/'cache/bfsongrepo_frozen_transfer.npz'
OLDMAN = ROOT/'outputs/bfsongrepo_development_population.json'
OLDCACHE = ROOT/'cache/bfsongrepo_development.npz'
OLDRESULT = ROOT/'outputs/bfsongrepo_development_results.json'
NEW = ROOT/'outputs/bfsongrepo_cross_individual_support_audit.json'


def dependencies():
    paths = [OLDMAN, OLDCACHE, OLDRESULT, NEW,
             ROOT/'outputs/bfsongrepo_development_residuals.npz',
             ROOT/'outputs/bfsongrepo_development_preregistration.json',
             ROOT/'checkpoints/bfsongrepo_development_R2.pt',
             ROOT/'checkpoints/bfsongrepo_development_R2_shuffled.pt',
             ROOT/'scripts/bfsongrepo_development.py', Path(__file__).resolve(),
             ROOT/'scripts/macde_transfer.py',ROOT/'scripts/acoustic_dynamics_report.py',
             ROOT/'scripts/macde_source_resolution.py',
             ROOT/'src/bioacoustic_pipeline/contextual/bfsongrepo_development.py',
             ROOT/'src/bioacoustic_pipeline/diagnostics/frozen_acoustic_transfer.py',
             ROOT/'src/bioacoustic_pipeline/ingestion/bfsongrepo_audit.py']
    paths += [ROOT/p for p in base.code_hashes()]
    return {str(p.relative_to(ROOT)):base.sha(p) for p in paths}


def freeze():
    if SPEC.exists() or RESULT.exists():
        raise RuntimeError('Study already frozen')
    audit = json.loads(NEW.read_text())
    old = json.loads(OLDRESULT.read_text())
    for checkpoint in old['checkpoint_integrity'].values():
        if base.sha(ROOT/checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Original checkpoint identity mismatch')
    base.freeze(SPEC,{
        'study':'BFSongRepo frozen end-to-end cross-individual transfer v1',
        'dependencies':dependencies(),'archives':audit['archive_integrity'],
        'population':'552 audited recordings,six bird-days,four birds. Scored populations: original heldout or60yw70 day,original gy6or6 cross-bird day,two newly audited bl26lb16/gr41rd51 days. Fit/validation or60 recordings excluded from evaluation.',
        'sampling':'Exactly16 uniformly spaced eligible indices per recording,unchanged128-frame gate plus52past/current and20future supported frames;40ms12-physical-feature target;no denoising,resampling,labels or concatenation. Stop if a new source has fewer16.',
        'transfer_type':'Frozen end-to-end encoder AND physical prediction head;no newly fitted target-bird head,scaler,normalization or adaptation.',
        'ridge_recovery':'Original Ridge coefficients were not persisted. Reconstruct ONCE using only original or60 fit+validation cached rows and saved alpha(R0=10,R0_long=300);persist coefficients before new-bird scoring. No alpha selection or target-bird fitting.',
        'reproduction_gate':'All four arms must reproduce original heldout-day and gy6or6 normalized residuals within absolute1e-5 BEFORE new-bird scoring. Stop on failure.',
        'normalization':'Fixed per-feature target standard deviations from original fit+validation,saved in closed results. Never estimate target-bird scales.',
        'arms':['R0','R0_long','R2','R2_shuffled'],
        'order_control':'Saved independently trained disrupted checkpoint with original window-wise deterministic permutation of older50frames,newest2 anchored. Uses original cache row indices for reproduction and appended row indices for new sources. Ordered11epochs vs disrupted6epochs;procedure comparison,not isolated architectural causality.',
        'primary':'Separate per-unseen-bird differences nRMSE(R0_long)-nRMSE(R2);positive favors R2. New unscored birds bl26lb16 and gr41rd51 reported separately from previously evaluated gy6or6.',
        'secondary':'R0-R2 and R2_shuffled-R2,raw per-feature RMSE and all arm errors per bird. Familiar bird only heldout_day as replication reference.',
        'uncertainty':'2000paired recording bootstrap draws seed0 conditional on each bird/day and fixed predictors. No frame bootstrap,no pooled primary effect,no population-wide bird confidence interval. Report all3 unseen-bird signs without treating recordings as biological replicates.',
        'exposure':'Bengalese corpus explored previously;gy6or6 already scored negatively. Newly audited birds have not been scored by this evaluation;not a wholly untouched species-level confirmation.',
        'stop':'One run;abort on provenance,reproduction,eligibility or missing-source failure;no retuning,multi-bird training or checkpoint replacement.',
        'stage2_ready':False,'tier_b_certified':False,
        'interpretation':'Cross-individual acoustic forecasting in selected triggered bouts at shared lab;not independent sessions,populations,communicative function or semantics.'})
    print('Frozen specification',base.sha(SPEC),flush=True)


def check():
    plan = json.loads(SPEC.read_text())
    for path, digest in plan['dependencies'].items():
        if base.sha(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed: '+path)
    for archive in plan['archives']:
        if base.sha(ROOT/archive['path']) != archive['sha256']:
            raise ValueError('Source archive changed')
    return plan


def prepare():
    check()
    if CACHE.exists() or MANIFEST.exists():
        raise RuntimeError('Transfer population already prepared')
    old = json.loads(OLDMAN.read_text())
    if base.sha(OLDCACHE) != old['cache_sha256']:
        raise ValueError('Original population cache mismatch')
    with np.load(OLDCACHE) as z:
        arrays = {k:z[k] for k in z.files}
    positions = arrays.pop('positions').tolist()
    items = old['items'].copy()
    chunks = {k:[v] for k,v in arrays.items()}
    additions = {k:[] for k in arrays}
    audit = json.loads(NEW.read_text())
    records = {r['recording_id']:r for r in audit['records']}
    seen = set()
    for source in audit['archive_integrity']:
        path = ROOT/source['path']
        metadata = {}
        with tarfile.open(path) as archive:
            for m in archive:
                if m.isfile() and m.name.endswith('.rec'):
                    metadata[m.name] = rec_metadata(archive.extractfile(m).read().decode())
        with tarfile.open(path) as archive:
            for m in archive:
                if not m.isfile() or not m.name.endswith('.cbin'):
                    continue
                recording = str(Path(m.name).with_suffix(''))
                if recording not in records or recording in seen:
                    raise ValueError('Unexpected/repeated source')
                seen.add(recording)
                rec = metadata[recording+'.rec']
                audio = decode_cbin_channel(archive.extractfile(m).read(),rec).astype(float)/32768.
                if pcm_hash(audio,rec['sample_rate']) != records[recording]['decoded_pcm_sha256']:
                    raise ValueError('Audited source changed')
                obs = base.frame_observations(audio,rec['sample_rate'])
                chosen = selected_positions(obs['time_valid'])
                if len(chosen) != 16:
                    raise ValueError('Frozen support gate failed')
                index = len(items)
                items.append({'recording_id':recording,'split':'new_unseen_bird',
                              'sample_rate':rec['sample_rate'],'archive':source['path'],'archive_member':m.name})
                for t in chosen:
                    for key in ('spectral','spectral_valid','physical','physical_valid'):
                        additions[key].append(obs[key][t-51:t+1])
                    additions['time_valid'].append(obs['time_valid'][t-51:t+1])
                    additions['target'].append(obs['physical'][t+4])
                    positions.append([index,int(t)])
                if len(seen)%40 == 0:
                    print('Prepared new recordings',len(seen),flush=True)
    if seen != set(records):
        raise ValueError('Missing frozen source recordings')
    for key in chunks:
        chunks[key].append(np.stack(additions[key]))
    np.savez_compressed(CACHE,**{k:np.concatenate(v) for k,v in chunks.items()},positions=np.array(positions))
    base.freeze(MANIFEST,{'preregistration_sha256':base.sha(SPEC),'cache_sha256':base.sha(CACHE),
                         'items':items,'targets':len(positions),'new_recordings':len(seen),'new_targets':len(seen)*16})
    print('Prepared',len(items),'recordings',len(positions),'targets',flush=True)


def run():
    check()
    marker = Path(str(PREFIX)+'_RUN_STARTED.json')
    if marker.exists() or RESULT.exists():
        raise RuntimeError('One-run stop rule')
    manifest = json.loads(MANIFEST.read_text())
    if manifest['preregistration_sha256'] != base.sha(SPEC) or manifest['cache_sha256'] != base.sha(CACHE):
        raise ValueError('Transfer population changed')
    base.freeze(marker,{'preregistration_sha256':base.sha(SPEC)})
    old = json.loads(OLDRESULT.read_text())
    with np.load(CACHE) as z:
        arrays = {k:z[k] for k in z.files}
    positions, items = arrays['positions'],manifest['items']
    tags = np.array([items[i]['split'] for i,_ in positions])
    fit = np.flatnonzero(np.isin(tags,['fit','validation']))
    x = {'R0':np.concatenate([arrays['physical'][:,-1],arrays['physical'][:,-2]],1),
         'R0_long':arrays['physical'].reshape(len(positions),-1)}
    ridges, weights = {}, {}
    for arm in x:
        scale,mu,sd,w,b = base.fit_ridge(x[arm][fit],arrays['target'][fit],old['ridge_selection'][arm]['alpha'])
        state = {'input_mean':scale.mean_,'input_scale':scale.scale_,'output_mean':mu,
                 'output_scale':sd,'weight':w,'intercept':b}
        ridges[arm] = state
        weights.update({arm+'__'+k:v for k,v in state.items()})
    recovered = ROOT/'checkpoints/bfsongrepo_frozen_transfer_recovered_ridge.npz'
    if recovered.exists():
        raise RuntimeError('Recovery artifact already exists')
    np.savez_compressed(recovered,**weights)
    models, scales = {}, {}
    for arm in ('R2','R2_shuffled'):
        payload = torch.load(ROOT/old['checkpoint_integrity'][arm]['path'],map_location='cpu',weights_only=True)
        model = MACDE(use_physical=True,width=payload['width']).eval()
        model.load_state_dict(payload['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[arm],scales[arm] = model,payload['scalers']
    if scales['R2'] != scales['R2_shuffled']:
        raise ValueError('Original neural scalers differ')
    data = {'valid':torch.from_numpy(arrays['time_valid'])}
    for key in ('physical','spectral'):
        value,mask = frozen_scale(arrays[key],arrays[key+'_valid'] & arrays['time_valid'][...,None],scales['R2'][key])
        data[key],data[key+'_valid'] = torch.from_numpy(value),torch.from_numpy(mask)
    report_sd = np.array(old['reporting_target_sd'])
    mu,sd = np.array(scales['R2']['physical']['mean']),np.array(scales['R2']['physical']['scale'])
    model_positions = np.c_[np.arange(len(positions)),np.full(len(positions),51)]
    def residuals(indices):
        predictions = {k:frozen_ridge(ridges[k],x[k][indices]) for k in ridges}
        for arm,policy in [('R2','combined'),('R2_shuffled','combined_disrupted')]:
            predictions[arm] = neural.predict(models[arm],data,model_positions[indices],policy)*sd+mu
        return {k:(p-arrays['target'][indices])/report_sd for k,p in predictions.items()}
    references, differences = {}, {}
    with np.load(ROOT/'outputs/bfsongrepo_development_residuals.npz') as reference:
        for tag in ('heldout_day','cross_bird'):
            indices = np.flatnonzero(tags == tag)
            res = residuals(indices)
            references[tag] = res
            for arm,value in res.items():
                expected = reference[tag+'__'+arm]
                np.testing.assert_allclose(value,expected,atol=1e-5,rtol=0)
                differences[tag+'__'+arm] = float(np.max(np.abs(value-expected)))
    base.freeze(Path(str(PREFIX)+'_recovery_verification.json'),{
        'all_original_residuals_reproduced':True,'max_absolute_residual_differences':differences,
        'recovered_ridge_sha256':base.sha(recovered),'reconstruction_rows':len(fit),
        'source_bird':'or60yw70','target_domain_fitting':False,
        'encoder_or_prediction_head_training':False,'stage2_ready':False})
    print('Original residual reproduction gate passed',flush=True)
    evaluations,residual_artifact = {},{}
    for bird in ('or60yw70','gy6or6','bl26lb16','gr41rd51'):
        indices = np.array([j for j,(i,t) in enumerate(positions) if items[i]['recording_id'].split('/')[0] == bird and tags[j] not in ('fit','validation')])
        res = references['heldout_day'] if bird == 'or60yw70' else references['cross_bird'] if bird == 'gy6or6' else residuals(indices)
        groups = [items[positions[j,0]]['recording_id'] for j in indices]
        evaluations[bird] = {'positions':len(indices),'recordings':len(set(groups)),
            'exposure':'familiar bird,previous heldout day' if bird == 'or60yw70' else 'previously evaluated unseen bird' if bird == 'gy6or6' else 'newly scored unseen bird',
            'nrmse':{k:base.loss(v) for k,v in res.items()},
            'raw_rmse_per_feature':{k:np.sqrt(((v*report_sd)**2).mean(0)).tolist() for k,v in res.items()},
            'R0long_minus_R2':base.boot(res['R0_long'],res['R2'],groups),
            'R0_minus_R2':base.boot(res['R0'],res['R2'],groups),
            'shuffled_minus_ordered':base.boot(res['R2_shuffled'],res['R2'],groups),
            'uncertainty_scope':'Conditional recording bootstrap for this bird/day,not across-bird uncertainty.'}
        residual_artifact.update({bird+'__'+k:v for k,v in res.items()})
        residual_artifact[bird+'__source_frame'] = positions[indices]
        print(bird,evaluations[bird]['nrmse'],flush=True)
    residual_path = Path(str(PREFIX)+'_residuals.npz')
    np.savez_compressed(residual_path,**residual_artifact)
    check()
    base.freeze(RESULT,{'study':'BFSongRepo frozen end-to-end cross-individual transfer v1',
        'preregistration_sha256':base.sha(SPEC),'population_sha256':base.sha(MANIFEST),
        'evaluations':evaluations,'residual_artifact':{'path':str(residual_path.relative_to(ROOT)),'sha256':base.sha(residual_path)},
        'all_original_residuals_reproduced':True,'target_bird_fitting':False,'neural_training':False,
        'ridge_reconstructed_from_original_training_only':True,'stage2_ready':False,'tier_b_certified':False,
        'unseen_bird_effects':{b:evaluations[b]['R0long_minus_R2']['point'] for b in ('gy6or6','bl26lb16','gr41rd51')},
        'no_pooled_primary_effect':True,'scope':json.loads(SPEC.read_text())['interpretation']})
    print('Completed frozen transfer;Stage2 remains blocked',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['freeze','prepare','run'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        {'freeze':freeze,'prepare':prepare,'run':run}[args.action]()
