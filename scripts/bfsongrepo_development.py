#!/usr/bin/env python3
"""New preregistered BFSongRepo TierA study;four audited dates,no denoising."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits

import macde_stage1 as base
import macde_transfer as neural
from bioacoustic_pipeline.contextual.bfsongrepo_development import partition_sources, selected_positions

ROOT = base.ROOT
SPEC = ROOT/'outputs/bfsongrepo_development_preregistration.json'
MANIFEST = ROOT/'outputs/bfsongrepo_development_population.json'
CACHE = ROOT/'cache/bfsongrepo_development.npz'
RESULT = ROOT/'outputs/bfsongrepo_development_results.json'
MARKER = ROOT/'outputs/bfsongrepo_development_RUN_STARTED.json'
EVIDENCE = [ROOT/'outputs'/n for n in ('bfsongrepo_tier_a_audit.json','bfsongrepo_multiday_audit.json','bfsongrepo_fourth_day_audit.json')]
ARMS = {'R2': 'combined', 'R2_shuffled': 'combined_disrupted'}


def code_hashes():
    paths = [Path(__file__), ROOT/'scripts/macde_transfer.py',
             ROOT/'src/bioacoustic_pipeline/contextual/bfsongrepo_development.py',
             ROOT/'src/bioacoustic_pipeline/diagnostics/macde_transfer.py']
    return base.code_hashes() | {str(p.relative_to(ROOT)): base.sha(p) for p in paths}


def protocol():
    return {'study': 'BFSongRepo TierA continuous-acoustic development v1',
            'source_evidence': {str(p.relative_to(ROOT)): base.sha(p) for p in EVIDENCE},
            'code_provenance': code_hashes(),
            'population': 'Four already audited bird-days;all eligible sources,exact duplicate decoded waveforms cause a stop rather than adaptive reassignment;previous Bengalese programme exposure acknowledged',
            'partition': {'fit': 'or60yw70:28Sep2012', 'validation': 'or60yw70:30Sep2012',
                          'heldout_day': 'or60yw70:1Oct2012', 'cross_bird': 'gy6or6:26Mar2012,secondary descriptive only'},
            'partition_claim': 'Whole-day separation for fit/validation/heldout;not certified independent sessions or unseen-animal primary evaluation. Cross-bird check has only one target individual.',
            'eligibility': 'Unchanged native40ms/10ms frontend;no denoising or gain normalization;require128consecutive supported frames and52past/current+20future support within one source file;never concatenate recordings/gaps',
            'target_selection': 'Exactly16 eligible frame indices per recording,uniformly spaced in the eligible-index list using integer linspace;same rows and targets for all arms;stop if any source has fewer16 or any partition fewer20 recordings',
            'target': 'Same frozen12 physical features40ms ahead for all arms;source and target waveform windows do not overlap;no syllable labels or social variables',
            'arms': {'R0': 'Ridge,current+previous12 physical states',
                     'R0_long': 'Ridge,all52 past/current physical states',
                     'R2': 'MACDE combined spectral+physical64-D causal encoder with direct physical forecast head',
                     'R2_shuffled': 'Same neural architecture/parameter count/initial seed/budget;independently per target permute oldest50 input frames jointly with masks;anchor newest2;train and evaluate disrupted'},
            'architecture_limits': '52frames available to each history arm;existing MACDE forecast head has32-frame effective causal receptive field. Comparison is between complete pipelines,not an isolated test of architecture or continuity.',
            'training': {'epochs_budget': 12, 'batch': 32, 'optimizer': 'Adam,lr0.001,weight_decay0.0001,clip5',
                         'seed': 0, 'selection': 'Best validation-day physical-target MSE epoch independently for each neural arm;Ridge alpha from unchanged grid on same validation day',
                         'ridge_alphas': list(base.ALPHAS),
                         'scalers': 'Fit on sampled fit-day input histories for selection;final refit on fit+validation sampled histories;never heldout or cross-bird adaptation',
                         'final_refit': 'After epoch/alpha selection,refit on fit+validation from same initial seed for selected epochs'},
            'primary': 'heldout_day mean normalized per-feature RMSE(R0_long)-RMSE(R2),positive favors R2',
            'required_reference': 'Also report R0-R2;do not redefine original Stage1 primary endpoint or claim it is repaired',
            'secondary': ['R2_shuffled-R2 on same heldout rows','All arms on cross_bird descriptive check'],
            'reporting_scale': 'Same per-feature std from fit+validation future physical targets for all arms;report raw per-feature RMSE too',
            'uncertainty': '2000paired recording-cluster bootstrap draws;conditional on one heldout bird-day and fixed models. Not between-day,between-animal or independently certified session uncertainty.',
            'developmental_criterion': 'Primary and R0 reference both have positive lower95%conditional recording-bootstrap bounds;this is only a developmental screening criterion,not TierB certification',
            'stop_rule': 'One bounded run;no retuning,replacement,denoising or model expansion after outcomes. Stage2 remains blocked regardless of results.',
            'protected_results': 'Earlier MACDE Stage1 and closed transfer/acquisition studies remain unchanged',
            'scientific_scope': 'Prospective acoustic forecasting conditional on selected triggered song bouts and observed birds/days;no silence/nonresponse,semantic,word,unit-absence or biological causality claims'}


def sources():
    a, b, c = [json.loads(p.read_text()) for p in EVIDENCE]
    records = a['records'] + b['new_records'] + c['records']
    if len({r['decoded_pcm_sha256'] for r in records}) != len(records):
        raise RuntimeError('Source decoded duplicate detected;study stops')
    split = partition_sources(records)
    for name in ('fit','validation','heldout_day','cross_bird'):
        if sum(split[r['recording_id']] == name for r in records) < 20:
            raise RuntimeError('Insufficient whole-recording population')
    return records, split


def check_protocol():
    if not SPEC.exists() or SPEC.read_text() != base.text(protocol()):
        raise RuntimeError('Missing or changed prospective specification')


def prepare():
    check_protocol()
    if MANIFEST.exists() or CACHE.exists():
        raise RuntimeError('Prepared study population exists')
    records, split = sources()
    by_member = {r['wav_member']: r for r in records}
    cache = {k: [] for k in ('spectral','spectral_valid','physical','physical_valid','time_valid','target')}
    positions, items = [], []
    archives = [ROOT/'data/bfsongrepo'/n for n in ('wav_csv.tar.gz','day0_wav.tar.gz','day1_wav.tar.gz','fourth_wav.tar.gz')]
    expected = {}
    for p in EVIDENCE:
        j = json.loads(p.read_text())
        for f in j['archive_integrity']:
            local = f.get('local') or f.get('local_path')
            expected[local] = f['sha256']
    for archive_path in archives:
        if base.sha(archive_path) != expected[str(archive_path.relative_to(ROOT))]:
            raise ValueError('Source archive integrity changed')
        with tarfile.open(archive_path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.wav'):
                    continue
                record = by_member[member.name]
                raw = archive.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() != record['wav_sha256']:
                    raise ValueError('Source waveform identity changed')
                audio, sr = sf.read(io.BytesIO(raw))
                obs = base.frame_observations(audio, sr)
                chosen = selected_positions(obs['time_valid'])
                if len(chosen) != 16:
                    raise RuntimeError('Predeclared per-source support gate failed')
                source_index = len(items)
                items.append({'recording_id': record['recording_id'], 'split': split[record['recording_id']],
                              'sha256': record['wav_sha256'], 'archive': str(archive_path.relative_to(ROOT)),
                              'archive_member': member.name, 'sample_rate': sr})
                for t in chosen:
                    for k in ('spectral','spectral_valid','physical','physical_valid'):
                        cache[k].append(obs[k][t-51:t+1])
                    cache['time_valid'].append(obs['time_valid'][t-51:t+1])
                    cache['target'].append(obs['physical'][t+4])
                    positions.append([source_index, int(t)])
                if len(items) % 40 == 0:
                    print('Prepared recordings', len(items), flush=True)
    if len(items) != len(records):
        raise RuntimeError('Source population mismatch')
    np.savez_compressed(CACHE, **{k: np.stack(v) for k,v in cache.items()}, positions=np.array(positions))
    counts = {p: sum(i['split'] == p for i in items) for p in ('fit','validation','heldout_day','cross_bird')}
    base.freeze(MANIFEST, {'preregistration_sha256': base.sha(SPEC), 'cache_sha256': base.sha(CACHE),
                           'items': items, 'recordings_per_partition': counts,
                           'targets_per_partition': {p: n*16 for p,n in counts.items()},
                           'total_targets': len(positions), 'models_fitted': False})
    print(base.text({'recordings': counts, 'targets': len(positions)}), flush=True)


def normalized_targets(raw, scalers):
    mu = np.array(scalers['physical']['mean'])
    sd = np.array(scalers['physical']['scale'])
    return ((raw-mu)/sd).astype(np.float32)


def run():
    check_protocol()
    manifest = json.loads(MANIFEST.read_text())
    if manifest['preregistration_sha256'] != base.sha(SPEC) or base.sha(CACHE) != manifest['cache_sha256']:
        raise ValueError('Study population provenance mismatch')
    if MARKER.exists() or RESULT.exists():
        raise RuntimeError('One-run stop rule')
    base.freeze(MARKER, {'preregistration_sha256': base.sha(SPEC)})
    with np.load(CACHE) as saved:
        arrays = {k: saved[k] for k in saved.files}
    positions = arrays['positions']
    items = manifest['items']
    tags = np.array([items[i]['split'] for i,_ in positions])
    partitions = {p: np.flatnonzero(tags == p) for p in ('fit','validation','heldout_day','cross_bird')}
    fit, val = partitions['fit'], partitions['validation']
    allfit = np.sort(np.r_[fit,val])
    model_positions = np.c_[np.arange(len(positions)), np.full(len(positions),51)]
    data, scales = base.scaled(arrays, fit)
    target = arrays['target']
    y = normalized_targets(target, scales)
    selection = {}
    for name, arm in ARMS.items():
        print(name, 'fit-only selection', flush=True)
        _, info = neural.train(data, model_positions[fit], y[fit], arm,
                               (model_positions[val],y[val]), epochs=12)
        selection[name] = info
    x = {'R0': np.concatenate([arrays['physical'][:,-1],arrays['physical'][:,-2]],1),
         'R0_long': arrays['physical'].reshape(len(positions),-1)}
    ridge_selection = {k: base.choose_ridge(v[fit], target[fit], v[val], target[val]) for k,v in x.items()}
    final_data, final_scales = base.scaled(arrays, allfit)
    final_target = normalized_targets(target, final_scales)
    models, checkpoints = {}, {}
    for name, arm in ARMS.items():
        epochs = selection[name]['selected_epochs']
        print(name, 'refit for', epochs, 'epochs', flush=True)
        model, _ = neural.train(final_data, model_positions[allfit], final_target[allfit], arm, epochs=epochs)
        models[name] = model
        path = ROOT/f'checkpoints/bfsongrepo_development_{name}.pt'
        torch.save({'state_dict': model.state_dict(), 'scalers': final_scales,
                    'arm_policy': arm, 'width': 64, 'use_physical': True, 'stage2_ready': False,
                    'preregistration_sha256': base.sha(SPEC),
                    'inference_note': 'Use study window/history policy;shuffled checkpoint is not interchangeable with ordered continuous inference'}, path)
        checkpoints[name] = {'path': str(path.relative_to(ROOT)), 'sha256': base.sha(path),
                             'parameters': sum(p.numel() for p in model.parameters())}
    ridges = {k: base.fit_ridge(v[allfit], target[allfit], ridge_selection[k]['alpha']) for k,v in x.items()}
    report_sd = np.maximum(target[allfit].std(0), 1e-6)
    mu = np.array(final_scales['physical']['mean'])
    sd = np.array(final_scales['physical']['scale'])
    evaluations, residuals = {}, {}
    for population in ('heldout_day','cross_bird'):
        idx = partitions[population]
        truth = target[idx]
        predictions = {k: neural.ridge_raw(model, x[k][idx]) for k,model in ridges.items()}
        for name, model in models.items():
            predictions[name] = neural.predict(model, final_data, model_positions[idx], ARMS[name])*sd+mu
        res = {k: (p-truth)/report_sd for k,p in predictions.items()}
        groups = [items[positions[j,0]]['recording_id'] for j in idx]
        evaluations[population] = {'positions': len(idx), 'recordings': len(set(groups)),
                                  'nrmse': {k: base.loss(v) for k,v in res.items()},
                                  'raw_rmse_per_feature': {k: np.sqrt(((v-truth)**2).mean(0)).tolist() for k,v in predictions.items()},
                                  'R0long_minus_R2': base.boot(res['R0_long'], res['R2'], groups),
                                  'R0_minus_R2': base.boot(res['R0'], res['R2'], groups),
                                  'shuffled_minus_ordered': base.boot(res['R2_shuffled'], res['R2'], groups),
                                  'uncertainty_scope': 'Conditional recording-bootstrap within one observed day/individual;not between-day or independent-session uncertainty'}
        residuals.update({population+'__'+k: v for k,v in res.items()})
        residuals[population+'__source_frame'] = positions[idx]
    primary = evaluations['heldout_day']
    passed = primary['R0long_minus_R2']['ci95'][0] > 0 and primary['R0_minus_R2']['ci95'][0] > 0
    residual_path = ROOT/'outputs/bfsongrepo_development_residuals.npz'
    np.savez_compressed(residual_path, **residuals)
    base.freeze(RESULT, {'study': protocol()['study'], 'preregistration_sha256': base.sha(SPEC),
                         'population_sha256': base.sha(MANIFEST), 'training_selection': selection,
                         'ridge_selection': ridge_selection, 'evaluations': evaluations,
                         'checkpoint_integrity': checkpoints, 'reporting_target_sd': report_sd.tolist(),
                         'residual_artifact': {'path': str(residual_path.relative_to(ROOT)), 'sha256': base.sha(residual_path)},
                         'conditional_developmental_criterion_passed': passed,
                         'stage2_ready': False, 'tier_b_certified': False,
                         'limits': [protocol()['partition_claim'],protocol()['uncertainty'],protocol()['architecture_limits'],protocol()['scientific_scope']]})
    print(base.text({'evaluations': evaluations,'conditional_developmental_criterion_passed':passed,'stage2_ready':False}),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--preregister',action='store_true')
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.preregister:
            sources()
            base.freeze(SPEC,protocol())
            print('Preregistered',base.sha(SPEC),flush=True)
        elif args.prepare:
            prepare()
        else:
            run()
