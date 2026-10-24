#!/usr/bin/env python3
"""MACDE frozen multi-domain acoustic transfer and matched neural order study."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits

import macde_stage1 as pilot
from bioacoustic_pipeline.diagnostics.macde_transfer import recording_group, partition, history_order
from bioacoustic_pipeline.embeddings.macde import MACDE, eligible_mask

ROOT = pilot.ROOT
PREFIX = ROOT / 'outputs/macde_transfer'
SPEC = Path(str(PREFIX) + '_preregistration.json')
AUDIT = Path(str(PREFIX) + '_source_audit.json')
CACHE = ROOT / 'cache/macde_transfer.npz'
DOMAINS = {
    'marmoset': 'data/alp/infant_marmosets_vox/data/audio_44k/**/*.wav',
    'pipit': 'data/alp/pipit_id/**/pipit-fg/*.wav',
    'littleowl': 'data/alp/littleowl_id/**/littleowl-fg/*.wav',
}
ARMS = ('physical', 'spectral', 'combined', 'combined_disrupted')
EPOCHS = 12


def code_hashes():
    paths = [Path(__file__), ROOT / 'src/bioacoustic_pipeline/diagnostics/macde_transfer.py']
    return pilot.code_hashes() | {str(p.relative_to(ROOT)): pilot.sha(p) for p in paths}


def protocol():
    return {
        'study': 'MACDE transfer v1; new frozen analysis of previously used corpora',
        'domains': DOMAINS, 'code_hashes': code_hashes(),
        'source_rule': 'hash-sort paths; inspect at most256 files/domain; native mono first256 frames; take first128 frames containing a full52-history+20future valid position; at most48 accepted files/domain and2/group; deduplicate full-file SHA globally',
        'partition': 'sha256(macde-transfer-v1:domain:group) modulo100; fit<60, validation<80, heldout otherwise',
        'grouping': 'marmoset date across twins; pipit and littleowl filename year/identity proxy merging original train/test release labels; proxies do not verify recording independence',
        'eligibility': 'all 52 past/current and20 future frames supported; same native 40ms/10ms physical schema, spectral grid and absolute activity gate as pilot; no relaxed gates',
        'support_gate': 'each of three domains >=4 fit groups, >=2 validation groups, >=3 heldout groups; >=40 positions per partition; otherwise audit-only terminal outcome, no fitting',
        'arms': ARMS, 'architecture': 'same MACDE width64; physical arm zeros spectral values AND masks; spectral arm zeros physical stream; same trainable parameter count',
        'disruption': 'per source/frame seeded permutation of oldest50 of52 frames; newest2 fixed; spectral/physical values AND masks jointly permuted; frozen mapping in training and evaluation',
        'target': 'frozen 12 physical features40ms ahead, nonoverlapping waveform; target unmodified by history disruption',
        'training': {'epochs': EPOCHS, 'batch': 32, 'lr': .001, 'weight_decay': .0001, 'seed': 0,
                     'selection': 'best validation normalized physical forecast MSE epoch; all arms same budget; final refit fit+validation for selected epochs',
                     'scalers': 'source fit-only for selection, source fit+validation for final; never target-domain adaptation'},
        'baselines': 'source-fitted recent-two-state Ridge and52-state Ridge, same alpha grid and validation policy as pilot',
        'scoring': 'mean per-feature RMSE normalized by TARGET DOMAIN fit+validation physical target std for every compared predictor; target normalization used only for reporting, never source prediction/training; report raw per-feature RMSE too',
        'primary': 'each source->other domain combined versus source physical Ridge, on identical target heldout rows; report all six directions individually, no selective pooling',
        'secondary': ['within-domain versions of every arm', 'combined versus physical neural', 'combined versus spectral neural',
                      'combined_disrupted versus combined (matched model family)', 'full-history Ridge comparison'],
        'uncertainty': '2000 paired target-recording-group bootstrap, fixed trained models; intervals descriptive, no multiplicity-adjusted generalization claim',
        'success': 'all six zero-shot combined-minus-recent-Ridge error contrasts must have upper95%CI<0; matched-order contrast descriptive; no Stage2 unlock',
        'scope': 'new independently frozen protocol, not unseen corpora; recording proxies do not imply unseen individuals; no receiver, semantic, unit or biological causality claims',
        'stop_rule': 'one bounded run; no replacements or hyperparameter changes based on results; Stage2 remains blocked',
    }


def prepare():
    if SPEC.read_text() != pilot.text(protocol()):
        raise RuntimeError('Specification missing or changed')
    if AUDIT.exists() or CACHE.exists():
        raise RuntimeError('Prepared source population already exists')
    items, observations, excluded = [], [], Counter()
    seen = set()
    for domain, pattern in DOMAINS.items():
        counts = Counter()
        accepted = 0
        paths = sorted(ROOT.glob(pattern), key=lambda p: hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest())
        for path in paths[:256]:
            if accepted >= 48:
                break
            try:
                group = recording_group(domain, path.name)
            except ValueError:
                excluded[domain + ':unresolved_group'] += 1
                continue
            if counts[group] >= 2:
                continue
            info = sf.info(path)
            hop = round(info.samplerate * .01)
            if info.frames < hop * 131:
                excluded[domain + ':short'] += 1
                continue
            digest = pilot.sha(path)
            if digest in seen:
                excluded[domain + ':duplicate'] += 1
                continue
            audio, sr = sf.read(path, frames=hop * 259)
            if audio.ndim > 1:
                audio = audio.mean(1)
            obs = pilot.frame_observations(audio, sr, max_frames=256)
            keep = eligible_mask(torch.from_numpy(obs['time_valid'])[None])[0].numpy()
            positions = np.flatnonzero(keep)
            if not len(positions):
                excluded[domain + ':no_contiguous_support'] += 1
                continue
            start = min(max(0, int(positions[0]) - 51), len(keep) - 128)
            obs = {k: v[start:start+128] for k, v in obs.items() if isinstance(v, np.ndarray)}
            n = int(eligible_mask(torch.from_numpy(obs['time_valid'])[None])[0].sum())
            if n == 0:
                raise RuntimeError('Window-selection eligibility invariant failed')
            items.append({'domain': domain, 'group': group, 'split': partition(domain, group),
                          'path': str(path.relative_to(ROOT)), 'sha256': digest,
                          'start_frame': start, 'sample_rate': sr, 'positions': n})
            observations.append(obs)
            counts[group] += 1
            accepted += 1
            seen.add(digest)
            if accepted % 12 == 0:
                print(domain, 'accepted', accepted, flush=True)
        print(domain, 'accepted total', accepted, flush=True)
    summary = {}
    adequate = True
    for domain in DOMAINS:
        summary[domain] = {}
        for split, minimum in (('fit', 4), ('validation', 2), ('heldout', 3)):
            subset = [i for i in items if i['domain'] == domain and i['split'] == split]
            n, groups = sum(i['positions'] for i in subset), len({i['group'] for i in subset})
            summary[domain][split] = {'files': len(subset), 'groups': groups, 'positions': n}
            adequate &= groups >= minimum and n >= 40
    if observations:
        keys = ('spectral', 'spectral_valid', 'physical', 'physical_valid', 'time_valid', 'source_valid', 'detected_activity')
        np.savez_compressed(CACHE, **{k: np.stack([o[k] for o in observations]) for k in keys})
    audit = {'preregistration_sha256': pilot.sha(SPEC), 'cache_sha256': pilot.sha(CACHE) if observations else None,
             'items': items, 'support': summary, 'exclusions': dict(excluded), 'support_gate_passed': bool(adequate),
             'status': 'ready_for_frozen_fitting' if adequate else 'terminal_insufficient_support_no_models_fitted'}
    pilot.freeze(AUDIT, audit)
    print(pilot.text({k: audit[k] for k in ('support', 'support_gate_passed', 'status')}), flush=True)


def rows(arrays, items, domain, split):
    result = []
    for i, item in enumerate(items):
        if item['domain'] != domain or item['split'] not in split:
            continue
        keep = eligible_mask(torch.from_numpy(arrays['time_valid'][i])[None])[0].numpy()
        result.extend((i, int(t)) for t in np.flatnonzero(keep))
    return np.asarray(result, int).reshape(-1, 2)


def batches(data, positions, arm, indices):
    values = {k: [] for k in ('spectral', 'physical', 'spectral_valid', 'physical_valid')}
    for j in indices:
        i, t = positions[j]
        seed = int.from_bytes(hashlib.sha256(f'{i}:{t}'.encode()).digest()[:8], 'little')
        order = history_order(seed, disrupted=arm == 'combined_disrupted')
        for k in values:
            values[k].append(data[k][i, t-51:t+1].numpy()[order])
    output = {k: torch.from_numpy(np.stack(v)) for k, v in values.items()}
    if arm == 'physical':
        output['spectral'].zero_()
        output['spectral_valid'].zero_()
    elif arm == 'spectral':
        output['physical'].zero_()
        output['physical_valid'].zero_()
    return output


def predict(model, data, positions, arm):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(positions), 32):
            b = batches(data, positions, arm, np.arange(start, min(start+32, len(positions))))
            output = model(b['spectral'], b['physical'], torch.ones((len(b['physical']), 52), dtype=torch.bool), b['spectral_valid'], b['physical_valid'])
            predictions.append(output['forecast'][:, -1].numpy())
    return np.concatenate(predictions)


def train(data, positions, target, arm, validation=None, epochs=EPOCHS):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = MACDE(use_physical=True)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_epoch, best_state, trace = float('inf'), 0, None, []
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(positions))
        for start in range(0, len(order), 32):
            idx = order[start:start+32]
            b = batches(data, positions, arm, idx)
            opt.zero_grad()
            out = model(b['spectral'], b['physical'], torch.ones((len(idx), 52), dtype=torch.bool), b['spectral_valid'], b['physical_valid'])
            loss = (out['forecast'][:, -1] - torch.from_numpy(target[idx])).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        if validation is not None:
            vp, vy = validation
            value = float(np.mean((predict(model, data, vp, arm)-vy)**2))
            trace.append(value)
            if value < best:
                best, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
    if validation is not None:
        model.load_state_dict(best_state)
    return model, {'selected_epochs': best_epoch if validation is not None else epochs, 'validation_mse': best if validation is not None else None, 'trace': trace}


def targets(arrays, positions):
    return np.array([arrays['physical'][i, t+4] for i, t in positions])


def ridge_x(arrays, positions):
    p = arrays['physical']
    return {'Ridge_recent': np.array([np.r_[p[i, t], p[i, t-1]] for i, t in positions]),
            'Ridge_history': np.array([p[i, t-51:t+1].ravel() for i, t in positions])}


def ridge_raw(fit, x):
    scale, mu, sd, w, b = fit
    return pilot.ridge_predict(w, b, scale.transform(x)) * sd + mu


def run():
    if SPEC.read_text() != pilot.text(protocol()):
        raise RuntimeError('Frozen specification mismatch')
    audit = json.loads(AUDIT.read_text())
    if audit['preregistration_sha256'] != pilot.sha(SPEC) or audit['cache_sha256'] != pilot.sha(CACHE):
        raise RuntimeError('Source provenance mismatch')
    if not audit['support_gate_passed']:
        raise RuntimeError('Terminal insufficient support: model fitting is blocked')
    pilot.freeze(Path(str(PREFIX) + '_RUN_STARTED.json'), {'preregistration_sha256': pilot.sha(SPEC)})
    # freeze() is idempotent for identical content; enforce one-run separately.
    items = audit['items']
    with np.load(CACHE) as stored:
        arrays = {k: stored[k] for k in stored.files}
    outputs, checkpoints, selections = {}, {}, {}
    # Target reporting scales are common across source predictors and never enter source training.
    target_scales = {d: np.maximum(targets(arrays, rows(arrays, items, d, ('fit', 'validation'))).std(0), 1e-6) for d in DOMAINS}
    residuals = {}
    for source in DOMAINS:
        print(source, 'selection fits', flush=True)
        fit = rows(arrays, items, source, ('fit',))
        val = rows(arrays, items, source, ('validation',))
        final = rows(arrays, items, source, ('fit', 'validation'))
        source_indices = np.array([i for i, item in enumerate(items) if item['domain'] == source and item['split'] == 'fit'])
        selection_data, _ = pilot.scaled(arrays, source_indices)
        yt = selection_data['physical'][fit[:, 0], fit[:, 1]+4].numpy()
        yv = selection_data['physical'][val[:, 0], val[:, 1]+4].numpy()
        selected = {}
        for arm in ARMS:
            _, info = train(selection_data, fit, yt, arm, (val, yv))
            selected[arm] = info
            print(source, arm, 'selected epoch', info['selected_epochs'], flush=True)
        xt, xv = ridge_x(arrays, fit), ridge_x(arrays, val)
        ridge_choices = {k: pilot.choose_ridge(xt[k], targets(arrays, fit), xv[k], targets(arrays, val)) for k in xt}
        indices = np.array([i for i, item in enumerate(items) if item['domain'] == source and item['split'] in ('fit', 'validation')])
        data, scalers = pilot.scaled(arrays, indices)
        yf = data['physical'][final[:, 0], final[:, 1]+4].numpy()
        models = {}
        for arm in ARMS:
            model, _ = train(data, final, yf, arm, epochs=selected[arm]['selected_epochs'])
            models[arm] = model
            path = ROOT / f'checkpoints/macde_transfer_{source}_{arm}.pt'
            torch.save({'state_dict': model.state_dict(), 'scalers': scalers, 'arm': arm, 'width': 64,
                        'use_physical': True, 'stage2_ready': False, 'preregistration_sha256': pilot.sha(SPEC)}, path)
            checkpoints[source + ':' + arm] = {'path': str(path.relative_to(ROOT)), 'sha256': pilot.sha(path),
                                               'parameters': sum(p.numel() for p in model.parameters())}
        xfinal = ridge_x(arrays, final)
        ridges = {k: pilot.fit_ridge(xfinal[k], targets(arrays, final), ridge_choices[k]['alpha']) for k in xfinal}
        selections[source] = {'neural': selected, 'ridge': ridge_choices}
        outputs[source] = {}
        for target_domain in DOMAINS:
            positions = rows(arrays, items, target_domain, ('heldout',))
            truth = targets(arrays, positions)
            groups = [items[i]['group'] for i, _ in positions]
            mu, sd = np.array(scalers['physical']['mean']), np.array(scalers['physical']['scale'])
            preds = {arm: predict(model, data, positions, arm) * sd + mu for arm, model in models.items()}
            xx = ridge_x(arrays, positions)
            preds.update({k: ridge_raw(model, xx[k]) for k, model in ridges.items()})
            res = {k: (p-truth) / target_scales[target_domain] for k, p in preds.items()}
            outputs[source][target_domain] = {
                'mode': 'within_domain' if source == target_domain else 'zero_shot_transfer',
                'positions': len(positions), 'groups': len(set(groups)),
                'nrmse': {k: pilot.loss(v) for k, v in res.items()},
                'raw_rmse_per_feature': {k: np.sqrt(((p-truth)**2).mean(0)).tolist() for k, p in preds.items()},
                'combined_minus_recent_Ridge': pilot.boot(res['combined'], res['Ridge_recent'], groups),
                'combined_minus_history_Ridge': pilot.boot(res['combined'], res['Ridge_history'], groups),
                'disrupted_minus_ordered': pilot.boot(res['combined_disrupted'], res['combined'], groups),
                'physical_neural_minus_combined': pilot.boot(res['physical'], res['combined'], groups),
                'spectral_neural_minus_combined': pilot.boot(res['spectral'], res['combined'], groups),
            }
            for arm, value in res.items():
                residuals[f'{source}__{target_domain}__{arm}'] = value
            residuals[f'{source}__{target_domain}__positions'] = positions
    success = all(outputs[s][t]['combined_minus_recent_Ridge']['ci95'][1] < 0 for s in DOMAINS for t in DOMAINS if s != t)
    residual_path = Path(str(PREFIX) + '_residuals.npz')
    np.savez_compressed(residual_path, **residuals)
    result = {'preregistration_sha256': pilot.sha(SPEC), 'source_audit_sha256': pilot.sha(AUDIT),
              'selection': selections, 'evaluations': outputs, 'checkpoints': checkpoints,
              'all_six_transfer_directions_passed': success, 'stage2_ready': False,
              'reporting_target_scales': {k: v.tolist() for k, v in target_scales.items()},
              'residual_artifact': {'path': str(residual_path.relative_to(ROOT)), 'sha256': pilot.sha(residual_path)},
              'limits': protocol()['scope']}
    pilot.freeze(Path(str(PREFIX) + '_results.json'), result)
    print(pilot.text({'all_six_transfer_directions_passed': success, 'stage2_ready': False}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument('--preregister', action='store_true')
    choice.add_argument('--prepare', action='store_true')
    choice.add_argument('--run', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if args.preregister:
            pilot.freeze(SPEC, protocol())
        elif args.prepare:
            prepare()
        else:
            if Path(str(PREFIX) + '_RUN_STARTED.json').exists():
                raise RuntimeError('One-run stop rule: run already started')
            run()
