#!/usr/bin/env python3
"""Prospective four-bird training evaluation; closed studies are read-only inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import macde_stage1 as base
import macde_transfer as neural
from bioacoustic_pipeline.diagnostics.macde_multibird import bird_folds, fit_preprocessing, select_ridge
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.embeddings.macde import MACDE

ROOT = base.ROOT
PREFIX = ROOT/'outputs/bfsongrepo_multibird'
SPEC = Path(str(PREFIX)+'_preregistration.json')
RESULT = Path(str(PREFIX)+'_results.json')
MANIFEST = ROOT/'outputs/bfsongrepo_frozen_transfer_population.json'
CACHE = ROOT/'cache/bfsongrepo_frozen_transfer.npz'
ARMS = {'R2_multi':'combined','R2_multi_shuffled':'combined_disrupted'}


def load():
    manifest = json.loads(MANIFEST.read_text())
    if base.sha(CACHE) != manifest['cache_sha256']:
        raise ValueError('Audited target population changed')
    with np.load(CACHE) as saved:
        arrays = {k:saved[k] for k in saved.files}
    positions = arrays['positions']
    items = manifest['items']
    birds = np.array([items[i]['recording_id'].split('/')[0] for i,t in positions])
    if len(items) != 552 or len(positions) != 8832 or len(set(birds)) != 4:
        raise ValueError('Unexpected audited population')
    if not arrays['time_valid'].all() or not arrays['physical_valid'].all() or not np.isfinite(arrays['target']).all():
        raise ValueError('Required fully supported physical inputs/targets absent')
    return manifest,arrays,birds


def provenance():
    paths = [Path(__file__).resolve(),MANIFEST,CACHE,
        ROOT/'outputs/bfsongrepo_multibird_design_draft.json',
        ROOT/'outputs/bfsongrepo_frozen_transfer_closeout.json',
        ROOT/'outputs/bfsongrepo_frozen_transfer_results.json',
        ROOT/'outputs/bfsongrepo_development_results.json',
        ROOT/'checkpoints/bfsongrepo_development_R2.pt',
        ROOT/'scripts/macde_transfer.py',ROOT/'scripts/acoustic_dynamics_report.py',
        ROOT/'src/bioacoustic_pipeline/diagnostics/macde_multibird.py',
        ROOT/'src/bioacoustic_pipeline/diagnostics/frozen_acoustic_transfer.py',
        ROOT/'src/bioacoustic_pipeline/diagnostics/macde_transfer.py',
        ROOT/'tests/test_macde_multibird.py',ROOT/'tests/test_macde_transfer.py']
    paths += [ROOT/p for p in base.code_hashes()]
    return {str(p.relative_to(ROOT)):base.sha(p) for p in paths}


def freeze():
    if SPEC.exists() or RESULT.exists():
        raise RuntimeError('Study already frozen')
    manifest,arrays,birds = load()
    folds = {}
    for heldout in sorted(set(birds)):
        fit,evaluation,inner = bird_folds(birds,heldout)
        def ids(indices):
            return sorted({manifest['items'][arrays['positions'][j,0]]['recording_id'] for j in indices})
        folds[heldout] = {'training_birds':sorted(set(birds[fit])),
            'training_rows':fit.tolist(),'evaluation_rows':evaluation.tolist(),
            'training_recordings':ids(fit),'evaluation_recordings':ids(evaluation),
            'inner_folds':[{'fit_birds':sorted(set(birds[a])),'validation_bird':str(birds[b[0]]),
                            'fit_rows':a.tolist(),'validation_rows':b.tolist()} for a,b in inner]}
    base.freeze(SPEC,{
        'study':'BFSongRepo multi-bird prospective acoustic forecasting v1',
        'dependencies':provenance(),'folds':folds,
        'population':'552 triggered song recordings,six bird-days,four identified males;8832frozen target positions16per recording. Previously explored corpus,not untouched confirmation.',
        'question':'Does multi-individual training improve forecasting for a bird excluded from all training and model selection,relative to short-history physical Ridge?',
        'source_and_target':'Unchanged frozen128-frame eligibility,52-frame causal history,40ms-ahead12physical targets,native-rate audio,no denoising,syllable labels,context or cross-recording concatenation.',
        'primary_comparator':'R0=current+previous physical-feature Ridge;alpha selected by bird-disjoint inner validation within three training birds only.',
        'ridge_selection':{'alphas':list(base.ALPHAS),'inner_folds':'Each of three training birds is validation once;fit on other two.',
            'metric':'Equal mean of three validation-bird mean per-feature normalized RMSE,normalization using two fit birds future-target SD only.',
            'selections':'Select alpha per family. Also select best family R0/R0_long by inner macro score,ties prefer R0 then smaller alpha. Freeze before outer evaluation.',
            'refit':'Both Ridge families refit on all three training birds with their selected alpha;no held-out selection.'},
        'neural_training':{'arms':ARMS,'width':64,'epochs':12,'seed':0,'batch_size':32,
            'optimizer':'Adam lr0.001 weight_decay0.0001 clip5',
            'selection':'Fixed12epochs for both arms,using previously declared maximum budget;no outer or inner outcome-dependent epoch selection,warm start or architecture tuning.',
            'data':'Every training row once per epoch,shuffled deterministically. Recording-count imbalance retained;not bird-balanced training.',
            'objective':'Direct future physical-feature MSE normalized by training-input physical scalers. No extra objective.',
            'shuffle':'Separately trained and evaluated oldest50frames jointly permuted with masks,newest2 fixed;original deterministic row-index policy.'},
        'preprocessing':'Input scalers,neural target transforms,Ridge scalers and common reporting future-target SD fit exclusively on current fold training rows. No target-bird adaptation.',
        'R2_single':'Unchanged existing or60yw70 single-bird checkpoint including forecast head,using its original fixed input scalers. Score only gy6or6,bl26lb16,gr41rd51;ineligible as unseen-bird reference for or60yw70. No new head fit.',
        'reference_limits':'Single-vs-multi contrasts confound training diversity,data amount,training target scale and optimization exposure;not an isolated causal test of diversity.',
        'primary':'For each of four held-out birds,nRMSE(R0)-nRMSE(R2_multi),positive favors multi. R0_long,training-selected Ridge family,R2_single where eligible,and ordered-vs-disrupted are secondary.',
        'reporting':'All arms use same rows,truth and per-feature SD estimated from three training birds. Report each bird separately,all arm errors,raw feature RMSE and paired effects;no pooled-frame primary.',
        'uncertainty':'2000paired recording-bootstrap draws seed0 conditional on each observed bird and its dates and fixed models. Not across-bird population inference. Outer training sets overlap,so four folds are not independent experiments.',
        'screening_rule':'All four per-bird R0-minus-multi conditional95%lower bounds positive AND all four training-selected-Ridge-minus-multi lower bounds positive. This is a stringent conditional development screen,not population-level evidence or Stage2 authorization.',
        'stop_rule':'One four-fold run;abort on source,dependency,split,nonfinite or checkpoint failure. No reseeding,fold replacement,hyperparameter changes,extra epochs or post-hoc retraining.',
        'interpretation':'Acoustic forecasting across identified birds in shared laboratory collection;not independently certified sessions,populations,communicative semantics or receiver outcomes.',
        'stage2_ready':False,'tier_b_certified':False})
    print('Preregistered',base.sha(SPEC),flush=True)


def check():
    spec = json.loads(SPEC.read_text())
    for path,digest in spec['dependencies'].items():
        if base.sha(ROOT/path) != digest:
            raise ValueError('Frozen dependency changed: '+path)
    return spec


def single_reference(arrays):
    original = json.loads((ROOT/'outputs/bfsongrepo_development_results.json').read_text())
    checkpoint = original['checkpoint_integrity']['R2']
    if base.sha(ROOT/checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Single-bird checkpoint mismatch')
    payload = torch.load(ROOT/checkpoint['path'],map_location='cpu',weights_only=True)
    model = MACDE(use_physical=True,width=64).eval()
    model.load_state_dict(payload['state_dict'])
    for p in model.parameters():
        p.requires_grad_(False)
    data = {'valid':torch.from_numpy(arrays['time_valid'])}
    for key in ('spectral','physical'):
        value,valid = frozen_scale(arrays[key],arrays[key+'_valid'] & arrays['time_valid'][...,None],payload['scalers'][key])
        data[key],data[key+'_valid'] = torch.from_numpy(value),torch.from_numpy(valid)
    return model,data,payload['scalers']['physical']


def run():
    spec = check()
    marker = Path(str(PREFIX)+'_RUN_STARTED.json')
    if marker.exists() or RESULT.exists():
        raise RuntimeError('One-run stop rule')
    base.freeze(marker,{'preregistration_sha256':base.sha(SPEC)})
    manifest,arrays,birds = load()
    positions = arrays['positions']
    model_positions = np.c_[np.arange(len(positions)),np.full(len(positions),51)]
    x = {'R0':np.concatenate([arrays['physical'][:,-1],arrays['physical'][:,-2]],1),
         'R0_long':arrays['physical'].reshape(len(positions),-1)}
    truth = arrays['target']
    single_model,single_data,single_scale = single_reference(arrays)
    evaluations,artifact,checkpoints = {},{},{}
    for heldout in sorted(set(birds)):
        fit,evaluation,inner = bird_folds(birds,heldout)
        frozen = spec['folds'][heldout]
        if fit.tolist() != frozen['training_rows'] or evaluation.tolist() != frozen['evaluation_rows']:
            raise ValueError('Frozen fold assignment changed')
        print(heldout,'training-only Ridge selection',flush=True)
        scores = {family:{alpha:[] for alpha in base.ALPHAS} for family in x}
        for inner_fit,val in inner:
            report_sd = np.maximum(truth[inner_fit].std(0),1e-6)
            for family,values in x.items():
                for alpha in base.ALPHAS:
                    ridge = base.fit_ridge(values[inner_fit],truth[inner_fit],alpha)
                    res = (neural.ridge_raw(ridge,values[val])-truth[val])/report_sd
                    scores[family][alpha].append(base.loss(res))
        selection = select_ridge(scores)
        selection_path = Path(str(PREFIX)+f'_{heldout}_selection.json')
        base.freeze(selection_path,{'heldout_bird':heldout,'selection':selection,'evaluation_used':False})
        data,scales,report_sd = fit_preprocessing(arrays,fit)
        mu,sd = np.array(scales['physical']['mean']),np.array(scales['physical']['scale'])
        target = ((truth-mu)/sd).astype(np.float32)
        predictions,ridge_weights = {},{}
        for family,values in x.items():
            ridge = base.fit_ridge(values[fit],truth[fit],selection['alphas'][family])
            predictions[family] = neural.ridge_raw(ridge,values[evaluation])
            scaler,ymu,ysd,w,b = ridge
            for key,value in {'input_mean':scaler.mean_,'input_scale':scaler.scale_,
                              'output_mean':ymu,'output_scale':ysd,'weight':w,'intercept':b}.items():
                ridge_weights[family+'__'+key] = value
        ridge_path = ROOT/f'checkpoints/bfsongrepo_multibird_{heldout}_ridge.npz'
        if ridge_path.exists():
            raise RuntimeError('Existing ridge checkpoint')
        np.savez_compressed(ridge_path,**ridge_weights)
        checkpoints[heldout] = {'ridge':{'path':str(ridge_path.relative_to(ROOT)),'sha256':base.sha(ridge_path)}}
        for arm,policy in ARMS.items():
            print(heldout,arm,'training12epochs on',sorted(set(birds[fit])),len(fit),'targets',flush=True)
            model,info = neural.train(data,model_positions[fit],target[fit],policy,epochs=12)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite learned parameters')
            path = ROOT/f'checkpoints/bfsongrepo_multibird_{heldout}_{arm}.pt'
            if path.exists():
                raise RuntimeError('Existing neural checkpoint')
            torch.save({'state_dict':model.state_dict(),'scalers':scales,'arm_policy':policy,
                        'width':64,'use_physical':True,'stage2_ready':False,
                        'heldout_bird':heldout,'training_birds':sorted(set(birds[fit])),
                        'preregistration_sha256':base.sha(SPEC),'training':info},path)
            checkpoints[heldout][arm] = {'path':str(path.relative_to(ROOT)),'sha256':base.sha(path),
                                        'parameters':sum(p.numel() for p in model.parameters())}
            predictions[arm] = neural.predict(model,data,model_positions[evaluation],policy)*sd+mu
        if heldout != 'or60yw70':
            predictions['R2_single'] = neural.predict(single_model,single_data,model_positions[evaluation],'combined')*np.array(single_scale['scale'])+np.array(single_scale['mean'])
        residuals = {k:(v-truth[evaluation])/report_sd for k,v in predictions.items()}
        if not all(np.isfinite(v).all() for v in residuals.values()):
            raise ValueError('Nonfinite evaluation')
        groups = [manifest['items'][positions[j,0]]['recording_id'] for j in evaluation]
        family = selection['selected_family']
        comparisons = {'R0_minus_multi':base.boot(residuals['R0'],residuals['R2_multi'],groups),
            'R0long_minus_multi':base.boot(residuals['R0_long'],residuals['R2_multi'],groups),
            'selected_Ridge_minus_multi':base.boot(residuals[family],residuals['R2_multi'],groups),
            'shuffled_minus_ordered':base.boot(residuals['R2_multi_shuffled'],residuals['R2_multi'],groups)}
        if 'R2_single' in residuals:
            comparisons['single_minus_multi'] = base.boot(residuals['R2_single'],residuals['R2_multi'],groups)
        evaluation_result = {'positions':len(evaluation),'recordings':len(set(groups)),
            'bird_days':sorted({'/'.join(g.split('/')[:2]) for g in groups}),
            'training_birds':sorted(set(birds[fit])),'training_positions':len(fit),
            'ridge_selection':selection,'reporting_target_sd':report_sd.tolist(),
            'nrmse':{k:base.loss(v) for k,v in residuals.items()},
            'raw_rmse_per_feature':{k:np.sqrt(((v-truth[evaluation])**2).mean(0)).tolist() for k,v in predictions.items()},
            'comparisons':comparisons,
            'R2_single_eligible':heldout != 'or60yw70',
            'single_reference_note':'Ineligible for source bird:existing checkpoint trained on this bird' if heldout == 'or60yw70' else 'Unchanged single-bird end-to-end reference,no target adaptation',
            'uncertainty_scope':'Conditional recording bootstrap for this bird and observed dates;not across-bird population confidence.'}
        evaluations[heldout] = evaluation_result
        base.freeze(Path(str(PREFIX)+f'_{heldout}_results.json'),evaluation_result)
        artifact.update({heldout+'__'+k:v for k,v in residuals.items()})
        artifact[heldout+'__source_frame'] = positions[evaluation]
        print(heldout,'completed',evaluation_result['nrmse'],flush=True)
    path = Path(str(PREFIX)+'_residuals.npz')
    np.savez_compressed(path,**artifact)
    check()
    passed = all(v['comparisons']['R0_minus_multi']['ci95'][0] > 0 and
                 v['comparisons']['selected_Ridge_minus_multi']['ci95'][0] > 0 for v in evaluations.values())
    base.freeze(RESULT,{'study':spec['study'],'preregistration_sha256':base.sha(SPEC),
        'population_sha256':base.sha(MANIFEST),'evaluations':evaluations,'checkpoints':checkpoints,
        'residual_artifact':{'path':str(path.relative_to(ROOT)),'sha256':base.sha(path)},
        'conditional_screening_rule_passed':passed,'stage2_ready':False,'tier_b_certified':False,
        'closed_studies_unchanged':True,'no_pooled_primary_effect':True,
        'scope':spec['interpretation']})
    print('Completed four-fold study;conditional screen:',passed,'Stage2 blocked',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['freeze','run'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        {'freeze':freeze,'run':run}[args.action]()
