#!/usr/bin/env python3
"""Preregister, then run one next-annotated-caller study (no nonresponse labels).

Usage: --preregister writes an immutable specification without fitting;
--run requires the specification and audited inputs to match exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/grijseels_dataset_audit.json'
DATA = ROOT / 'data/grijseels_turntaking/vocal_turn_taking_data'
PREFIX = ROOT / 'outputs/grijseels_next_caller'
SEED = 0
HISTORY = 32
EPOCHS = 200
PATIENCE = 20
BOOTSTRAPS = 2000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def freeze(path, value):
    payload = canonical(value)
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f'Refusing to overwrite differing frozen artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def event_windows(events, max_history=HISTORY):
    """Only singleton onset groups can be ordered. A tie resets past context.

    No target onset/offset is included in inputs. Overlaps with unequal onsets
    retain their onset order; target overlap is an evaluation stratum only.
    """
    by_onset = defaultdict(list)
    for event in events:
        by_onset[event[0]].append(event)
    groups = [by_onset[t] for t in sorted(by_onset)]
    history = []
    output = []
    counts = Counter()
    for i, group in enumerate(groups):
        if len(group) != 1:
            counts['tied_groups'] += 1
            counts['events_in_tied_groups'] += len(group)
            history = []
            continue
        current = group[0]
        history.append(current)
        history = history[-max_history:]
        if i + 1 == len(groups):
            counts['terminal_singletons'] += 1
            continue
        if len(groups[i + 1]) != 1:
            counts['singleton_before_tie'] += 1
            continue
        nxt = groups[i + 1][0]
        output.append({'history': list(history), 'target': nxt[2],
                       'current': current[2], 'current_onset': current[0],
                       'target_onset': nxt[0], 'overlap': nxt[0] < current[1]})
    return output, dict(counts)


def prepare():
    audit = json.loads(AUDIT.read_text())
    # Verify the frozen extracted inputs, not just the original ZIP.
    for row in audit['inventory']:
        if digest(DATA / row['path']) != row['sha256']:
            raise RuntimeError(f'Audited input changed: {row["path"]}')
    names = sorted({n for r in audit['recordings'] if r['selected']
                    for n in r['callers_in_cell_order']})
    ids = {n: i for i, n in enumerate(names)}
    selected = [r for r in audit['recordings'] if r['selected']
                and not r['recording_id'].startswith('one_monkey/') and not r['issues']]
    configs = sorted({tuple(sorted(ids[n] for n in r['callers_in_cell_order'])) for r in selected})
    cfg_ids = {c: i for i, c in enumerate(configs)}
    fps = audit['timestamps']['frames_per_second']
    samples = []
    exclusions = Counter()
    records = []
    for rec in selected:
        if rec['candidate_partition'] == 'heldout':
            split = 'heldout'
        else:
            v = int(hashlib.sha256(('grijseels-validation-v1:' + rec['partition_group']).encode()).hexdigest(), 16)
            split = 'validation' if v % 100 < 25 else 'train'
        candidates = tuple(sorted(ids[n] for n in rec['callers_in_cell_order']))
        events = []
        for name, cell in zip(rec['callers_in_cell_order'], loadmat(DATA / rec['path'])['call_data'].flat):
            if not cell.size:
                continue
            for start, end in cell:
                if not (np.isfinite([start, end]).all() and 0 <= start < end):
                    raise RuntimeError(f'Unflagged invalid event in {rec["path"]}')
                events.append((float(start) / fps, float(end) / fps, ids[name]))
        windows, dropped = event_windows(events)
        exclusions.update(dropped)
        records.append({'recording_id': rec['recording_id'], 'partition_group': rec['partition_group'],
                        'split': split, 'events': len(events), 'targets': len(windows),
                        'candidates': [names[c] for c in candidates]})
        for window in windows:
            window.update(recording=rec['recording_id'], group=rec['partition_group'],
                          split=split, candidates=candidates, configuration=cfg_ids[candidates])
            samples.append(window)
    manifest = [{'recording': s['recording'], 'onset_s': s['current_onset'],
                 'target_onset_s': s['target_onset'], 'current': names[s['current']],
                 'target': names[s['target']], 'history_onsets_s': [e[0] for e in s['history']],
                 'split': s['split'], 'overlap': s['overlap']} for s in samples]
    plan = {
        'schema_version': 1, 'seed': SEED,
        'endpoint': 'identity of next annotated caller at the next singleton onset, conditional on the available annotation sequence',
        'claims': 'history predictability only; no intended recipient, nonresponse, semantic, intentional or comprehension endpoint',
        'prediction_time': 'current-event onset; call offsets/durations and future intervals excluded from model inputs',
        'population': 'paired and trio author-selected versions; solo excluded; any recording flagged by audit excluded in full; empty recordings yield no targets',
        'scope_limit': 'complete monitoring not verified; no claim that next annotated event is necessarily next natural vocalization',
        'history': {'max_events_including_current': HISTORY,
                    'tie_policy': 'exclude current/target tied groups; reset history at each tie; never order simultaneous callers',
                    'overlap_policy': 'retain different-onset overlaps in primary onset-order task; report overlapping/nonoverlapping target strata; no offset features',
                    'session_policy': 'never cross recording boundary or pool variants; no maximum gap exclusion (annotation-order endpoint only)'},
        'candidate_space': 'mask to all named animals in the recording, including animals with zero observed calls; never infer availability from targets',
        'models': {
            'M0': 'training-target marginal caller counts +1 Laplace smoothing; renormalize over available callers',
            'M1': 'masked first-order softmax: trainable current-caller-to-next-caller table plus available-animal-configuration logits; no history or timing inputs',
            'M2': 'M1 plus linear softmax logits from history summaries: caller proportions and log-gap mean/std/known fraction over last 4/16/32 events; per-caller log recency and presence; history length and current log gap',
            'M3': 'M1 plus GRU(hidden=32) over same raw history (one-hot caller, log1p preceding onset gap, gap-known flag); linear output logits',
            'information_parity': 'M2 and M3 derive inputs from exactly the same <=32 onset/caller tokens; summaries compress sequence; no acoustics, responses, per-session fitted intercepts, or unverified demographics',
        },
        'training': {'device': 'CPU', 'torch_threads': 2, 'optimizer': 'Adam', 'learning_rate': 0.01,
                     'weight_decay': 0.001, 'batch_size': 256, 'gradient_clip_norm': 5,
                     'max_epochs': EPOCHS, 'early_stopping_patience': PATIENCE,
                     'validation_min_improvement': 1e-5, 'gru_hidden': 32,
                     'selection': 'one fixed seed and architecture; select epoch by validation log loss; refit from same seed on train+validation for selected epochs',
                     'scaling': 'M2 summaries and M3 gap values standardized with train-only statistics for selection and all-fit-only statistics for refit; one-hot inputs unscaled'},
        'partitions': {'outer': 'use frozen audit fit/heldout date groups unchanged',
                       'inner': 'fit groups with SHA256(grijseels-validation-v1:group) modulo100 <25 => validation',
                       'no_tuning_on_heldout': True,
                       'sample_counts': dict(Counter(s['split'] for s in samples)),
                       'records': records},
        'eligibility_exclusions': {'flagged_recordings': audit['event_quality']['records_requiring_review'],
                                   'tie_and_terminal_counts': dict(exclusions)},
        'primary': {'metric': 'event-weighted categorical log loss (natural logarithm)',
                    'contrast': 'M1 logloss - M2 logloss',
                    'positive_rule': 'paired session-clustered bootstrap percentile 95% interval entirely >0',
                    'bootstrap_draws': BOOTSTRAPS,
                    'uncertainty': 'resample whole recording sessions with replacement; same weights across arms'},
        'secondary': ['M2 logloss - M3 logloss (recurrent contrast; exploratory, no multiplicity-adjusted claim)',
                      'accuracy', 'macro recall over target callers present in heldout (balanced accuracy)',
                      'session-macro logloss', 'overlapping and nonoverlapping target strata',
                      'paired date-group-clustered bootstrap',
                      'paired current-caller-clustered sensitivity (not independent-animal generalization)'],
        'provenance': {'audit_sha256': digest(AUDIT), 'audit_script_sha256': digest(ROOT / 'scripts/grijseels_dataset_audit.py'),
                       'study_script_sha256': digest(Path(__file__)),
                       'event_manifest_sha256': hashlib.sha256(canonical(manifest).encode()).hexdigest(),
                       'names': names, 'available_configurations': [[names[i] for i in c] for c in configs]},
        'stop_rule': 'one heldout evaluation after preregistration; no adaptive model/search expansion',
    }
    return samples, names, configs, manifest, plan


def input_arrays(samples, n_callers):
    tokens = np.zeros((len(samples), HISTORY, n_callers + 2), np.float32)
    lengths = np.zeros(len(samples), np.int64)
    summaries = []
    for i, sample in enumerate(samples):
        history = sample['history']
        lengths[i] = len(history)
        for j, (onset, _, caller) in enumerate(history):
            tokens[i, j, caller] = 1
            if j:
                tokens[i, j, -2] = math.log1p(onset - history[j - 1][0])
                tokens[i, j, -1] = 1
        pieces = []
        for window in (4, 16, 32):
            t = tokens[i, max(0, len(history) - window):len(history)]
            gaps = t[t[:, -1] > 0, -2]
            pieces.extend(t[:, :n_callers].mean(0))
            pieces.extend([float(gaps.mean()) if gaps.size else 0,
                           float(gaps.std()) if gaps.size else 0, float(t[:, -1].mean())])
        recency = np.full(n_callers, math.log1p(1800), np.float32)
        present = np.zeros(n_callers, np.float32)
        for onset, _, caller in history:
            recency[caller] = math.log1p(min(1800, history[-1][0] - onset))
            present[caller] = 1
        pieces.extend(recency)
        pieces.extend(present)
        pieces.extend([len(history) / HISTORY, tokens[i, len(history) - 1, -2]])
        summaries.append(pieces)
    mask = np.zeros((len(samples), n_callers), bool)
    for i, sample in enumerate(samples):
        mask[i, list(sample['candidates'])] = True
    return {'tokens': tokens, 'lengths': lengths, 'summaries': np.asarray(summaries, np.float32),
            'current': np.array([s['current'] for s in samples]),
            'configuration': np.array([s['configuration'] for s in samples]),
            'mask': mask, 'target': np.array([s['target'] for s in samples])}


def normalized_inputs(arrays, fit_indices):
    result = {k: v.copy() for k, v in arrays.items()}
    features = arrays['summaries'][fit_indices]
    mean, std = features.mean(0), features.std(0)
    result['summaries'] = (arrays['summaries'] - mean) / np.where(std < 1e-5, 1.0, std)
    known = arrays['tokens'][fit_indices, :, -1] > 0
    gaps = arrays['tokens'][fit_indices, :, -2][known]
    gmean, gstd = (float(gaps.mean()), max(float(gaps.std()), 1e-6)) if gaps.size else (0., 1.)
    all_known = result['tokens'][:, :, -1] > 0
    result['tokens'][:, :, -2][all_known] = (result['tokens'][:, :, -2][all_known] - gmean) / gstd
    return result


def build_model(kind, n_callers, n_configs, n_summary):
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transitions = nn.Embedding(n_callers, n_callers)
            self.configurations = nn.Embedding(n_configs, n_callers)
            nn.init.zeros_(self.transitions.weight)
            nn.init.zeros_(self.configurations.weight)
            if kind == 'M2':
                self.history_head = nn.Linear(n_summary, n_callers, bias=False)
                nn.init.zeros_(self.history_head.weight)
            if kind == 'M3':
                self.gru = nn.GRU(n_callers + 2, 32, batch_first=True)
                self.history_head = nn.Linear(32, n_callers, bias=False)

        def forward(self, data, idx):
            logits = self.transitions(data['current'][idx]) + self.configurations(data['configuration'][idx])
            if kind == 'M2':
                logits = logits + self.history_head(data['summaries'][idx])
            if kind == 'M3':
                packed = pack_padded_sequence(data['tokens'][idx], data['lengths'][idx].cpu(),
                                              batch_first=True, enforce_sorted=False)
                _, hidden = self.gru(packed)
                logits = logits + self.history_head(hidden[-1])
            return logits.masked_fill(~data['mask'][idx], -torch.inf)

    return Model()


def fit_model(kind, arrays, n_configs, fit_idx, val_idx=None, epochs=None):
    import torch
    from torch.nn import functional as F
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = {k: torch.from_numpy(v) for k, v in normalized_inputs(arrays, fit_idx).items()}
    model = build_model(kind, arrays['mask'].shape[1], n_configs, arrays['summaries'].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    best, best_epoch, stale = float('inf'), 0, 0
    trace = []
    for epoch in range(1, (epochs or EPOCHS) + 1):
        model.train()
        for batch in np.array_split(rng.permutation(fit_idx), max(1, math.ceil(len(fit_idx) / 256))):
            opt.zero_grad()
            loss = F.cross_entropy(model(data, batch), data['target'][batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                value = float(F.cross_entropy(model(data, val_idx), data['target'][val_idx]))
            trace.append(value)
            if value < best - 1e-5:
                best, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    return model, data, {'selected_epochs': best_epoch if val_idx is not None else epochs,
                         'validation_logloss': best if val_idx is not None else None,
                         'validation_trace': trace}


def probabilities(model, data, indices):
    import torch
    model.eval()
    with torch.no_grad():
        # float64 scoring prevents underflow; unavailable classes stay exactly zero.
        return model(data, indices).double().softmax(dim=1).numpy()


def paired_bootstrap(difference, clusters, seed=SEED):
    labels = sorted(set(clusters))
    sums = np.array([difference[np.array(clusters) == c].sum() for c in labels])
    sizes = np.array([sum(x == c for x in clusters) for c in labels])
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(labels), (BOOTSTRAPS, len(labels)))
    draws = sums[ids].sum(1) / sizes[ids].sum(1)
    return {'point': float(difference.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'clusters': len(labels), 'draws': BOOTSTRAPS}


def metrics(prob, targets, recordings):
    pred = prob.argmax(1)
    loss = -np.log(np.maximum(prob[np.arange(len(targets)), targets], np.finfo(float).tiny))
    return {'logloss': float(loss.mean()), 'accuracy': float((pred == targets).mean()),
            'balanced_accuracy': float(np.mean([(pred[targets == c] == c).mean() for c in np.unique(targets)])),
            'session_macro_logloss': float(np.mean([loss[np.array(recordings) == s].mean() for s in sorted(set(recordings))]))}, loss


def run(samples, names, configs, manifest, plan):
    import sklearn
    import scipy
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arrays = input_arrays(samples, len(names))
    indices = {s: np.array([i for i, row in enumerate(samples) if row['split'] == s], np.int64)
               for s in ('train', 'validation', 'heldout')}
    if not all(len(v) for v in indices.values()):
        raise RuntimeError('Empty partition; do not change split after looking at outcomes')
    fit = np.sort(np.concatenate([indices['train'], indices['validation']]))
    evaluation = indices['heldout']
    # Do not create heldout probabilities until every model has selected and refitted.
    models, traces = {}, {}
    for kind in ('M1', 'M2', 'M3'):
        print(f'{kind}: selecting epoch on fit-side validation', flush=True)
        _, _, traces[kind] = fit_model(kind, arrays, len(configs), indices['train'], indices['validation'])
        epochs = traces[kind]['selected_epochs']
        print(f'{kind}: refitting all fit sessions for {epochs} epochs', flush=True)
        model, data, _ = fit_model(kind, arrays, len(configs), fit, epochs=epochs)
        models[kind] = (model, data)
    counts = np.bincount(arrays['target'][fit], minlength=len(names)) + 1
    m0 = arrays['mask'][evaluation] * counts[None, :]
    preds = {'M0': m0 / m0.sum(1, keepdims=True)}
    preds.update({k: probabilities(model, data, evaluation) for k, (model, data) in models.items()})
    rows = [samples[i] for i in evaluation]
    target = arrays['target'][evaluation]
    recordings = [s['recording'] for s in rows]
    losses, scores = {}, {}
    for kind, prob in preds.items():
        assert np.all(prob[~arrays['mask'][evaluation]] == 0)
        assert np.allclose(prob.sum(1), 1)
        scores[kind], losses[kind] = metrics(prob, target, recordings)
    history = losses['M1'] - losses['M2']
    recurrent = losses['M2'] - losses['M3']
    clusters = {'session': recordings, 'date_group': [s['group'] for s in rows],
                'current_caller': [names[s['current']] for s in rows]}
    contrasts = {name: {key: paired_bootstrap(delta, cluster) for key, cluster in clusters.items()}
                 for name, delta in [('history_M1_minus_M2', history), ('recurrent_M2_minus_M3', recurrent)]}
    strata = {}
    for overlap in (False, True):
        keep = np.array([s['overlap'] == overlap for s in rows])
        strata['overlapping' if overlap else 'nonoverlapping'] = {
            'events': int(keep.sum()),
            'logloss': {k: float(v[keep].mean()) for k, v in losses.items()} if keep.any() else None,
        }
    predictions = []
    for j, s in enumerate(rows):
        candidates = s['candidates']
        predictions.append({'recording': s['recording'], 'group': s['group'],
                            'onset_s': s['current_onset'], 'target_onset_s': s['target_onset'],
                            'current_caller': names[s['current']], 'next_annotated_caller': names[s['target']],
                            'overlap': s['overlap'], 'candidates': [names[c] for c in candidates],
                            'probabilities': {k: p[j, list(candidates)].tolist() for k, p in preds.items()}})
    freeze(Path(str(PREFIX) + '_events.json'), manifest)
    freeze(Path(str(PREFIX) + '_predictions.json'), predictions)
    out = {'endpoint': plan['endpoint'], 'preregistration_sha256': digest(Path(str(PREFIX) + '_preregistration.json')),
           'audit_sha256': digest(AUDIT), 'study_script_sha256': digest(Path(__file__)),
           'software': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__},
           'sample_counts': plan['partitions']['sample_counts'], 'heldout_sessions': len(set(recordings)),
           'heldout_date_groups': len(set(clusters['date_group'])),
           'scores': scores, 'contrasts': contrasts, 'overlap_strata': strata,
           'training_selection': traces,
           'history_positive_by_preregistered_session_rule': contrasts['history_M1_minus_M2']['session']['ci95'][0] > 0,
           'limits': ['next annotated caller only; annotation completeness and recording coverage unverified',
                      'no nonresponse or intended-recipient endpoint',
                      'animals recur across partitions; no unseen-animal generalization',
                      'session bootstrap assumes session independence; date-group sensitivity is reported',
                      'current-caller bootstrap is a sensitivity, not a full multi-membership animal model',
                      'overlap-stratified scores are descriptive, not separately tuned models'],
           'artifact_sha256': {suffix: digest(Path(str(PREFIX) + suffix))
                               for suffix in ('_events.json', '_predictions.json')}}
    freeze(Path(str(PREFIX) + '_results.json'), out)
    print(canonical({k: out[k] for k in ('sample_counts', 'scores', 'contrasts', 'history_positive_by_preregistered_session_rule')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preregister', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    samples, names, configs, manifest, plan = prepare()
    prereg = Path(str(PREFIX) + '_preregistration.json')
    if args.preregister:
        freeze(prereg, plan)
        print(canonical({'preregistration': str(prereg), 'sha256': digest(prereg),
                         'sample_counts': plan['partitions']['sample_counts']}))
    else:
        if not prereg.exists() or prereg.read_text() != canonical(plan):
            raise RuntimeError('Missing or changed preregistration; no fitting performed')
        if Path(str(PREFIX) + '_results.json').exists() or Path(str(PREFIX) + '_RUN_STARTED.json').exists():
            raise RuntimeError('Study already started; one-run stop rule prevents rerunning')
        freeze(Path(str(PREFIX) + '_RUN_STARTED.json'), {'preregistration_sha256': digest(prereg)})
        run(samples, names, configs, manifest, plan)


if __name__ == '__main__':
    main()
