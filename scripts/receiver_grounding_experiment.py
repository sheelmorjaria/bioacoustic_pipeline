#!/usr/bin/env python3
"""Audit receiver-grounding datasets, freeze a protocol, then run gated feasibility.

BirdPark: temporal candidates only until independent video behaviour is validated.
Elephants: controlled-condition response feasibility; acoustic-response prediction
is unavailable without a verified playback-stimulus-to-acoustic-call crosswalk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.receiver_grounding import (
    audit_birdpark_pair, endpoint_gate, independent_groups, load_elephant_playback,
)
from bioacoustic_pipeline.contextual.receiver_response import evaluate_response_models

OUT = ROOT / 'outputs'
ELEPHANTS = ROOT / 'data/elephant_playback'


def sha(path):
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def serialize(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def freeze(path, value):
    payload = serialize(value)
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f'Existing artifact differs; use a new version: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def code_provenance():
    files = [Path(__file__), ROOT/'src/bioacoustic_pipeline/ingestion/receiver_grounding.py',
             ROOT/'src/bioacoustic_pipeline/contextual/receiver_response.py',
             ROOT/'scripts/download_receiver_datasets.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def verified_inputs():
    manifest = json.loads((OUT/'receiver_dataset_downloads.json').read_text())
    for row in manifest:
        if row['status'] in ('verified', 'downloaded', 'available'):
            if sha(ROOT/row['path']) != row['sha256']:
                raise RuntimeError(f'Downloaded input changed: {row["path"]}')
    return manifest


def audit():
    verified_inputs()
    records, candidates = [], []
    for h5 in sorted((ROOT/'data/birdpark_pairs/Data').glob('*.h5')):
        annotation = ROOT/'data/birdpark_pairs/Annotations'/f'{h5.stem}_annotations.csv'
        record, windows = audit_birdpark_pair(h5, annotation)
        record['extracted_sha256'] = {p.name: sha(p) for p in (h5, annotation, h5.with_suffix('.mp4'))}
        records.append(record)
        candidates.extend(windows)
    group_path = ROOT/'data/birdpark_groups/segments.h5'
    group_summary = {}
    if group_path.exists():
        with h5py.File(group_path) as f:
            episodes = f['episodes'][:]
            segments = f['segments'][:]
            names = [r['filename'].decode() for r in episodes]
            group_summary = {'recording_episodes': len(episodes), 'annotation_rows_all_layers': len(segments),
                             'experiment_ids': sorted({r['expInstanceId'].decode() for r in episodes}),
                             'annotation_windows': [{'file': r['filename'].decode(), 'onset_sample_1based': int(r['onset']),
                                                     'offset_sample_inclusive': int(r['offset'])} for r in episodes],
                             'pair_recordings_also_in_group_release': sorted(set(Path(n).name for n in names) & {r['file'] for r in records}),
                             'rule': 'do not pool duplicate pair recordings; group annotation coverage is limited to /episodes, not all 112 minutes',
                             'raw_group_data': '12.3 GB archive deferred; availability and sensor coverage outside pair files unverified'}
    try:
        trials = load_elephant_playback(ELEPHANTS)
    except FileNotFoundError:
        trials = []
    elephant_files = {p.name: sha(p) for p in ELEPHANTS.iterdir() if p.is_file()}
    acoustic = list(csv.DictReader((ELEPHANTS/'20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv').open())) if trials else []
    groups = independent_groups([{'subject': t['subject'], 'stimulus_id': 'caller:' + t['caller']} for t in trials])
    elephant = {
        'source': '10.5061/dryad.hmgqnk9nj; version 2025-09-27',
        'sha256': elephant_files, 'corrected_playback_trials': len(trials),
        'subjects': len({t['subject'] for t in trials}), 'callers': len({t['caller'] for t in trials}),
        'treatment_counts': dict(Counter(t['raw']['Treatment'] for t in trials)),
        'subject_and_caller_connected_groups': len(set(groups)),
        'response_windows': {'vigilance': '60 seconds before and after stimulus',
                             'vocal_and_approach_latency': '600 seconds; event flags distinguish nonresponse from missingness'},
        'vocal_response_600s_counts': dict(Counter(str(t['vocal_response_600s']) for t in trials)),
        'missing_vigilance_post': sum(t['vigilance_post_s'] is None for t in trials),
        'stimulus_acoustic_crosswalk': {'status': 'unavailable', 'exactly_linked_trials': 0,
                                        'observational_acoustic_rows': len(acoustic),
                                        'reason': 'No stimulus call identifier in playback table. Playback is elapsed time in video, as defined in the XLSX dictionary. Subject/caller/treatment do not identify a unique call.'},
        'original_addressee': 'Treatment indicates originally addressed to subject versus another; control original receiver identity is not a separate column',
        'available_endpoint': 'independently coded vigilance duration and vocal/approach event flags per playback subject',
        'acoustic_response_experiment': {'ready': False, 'blocked_by': ['exact_stimulus_call_crosswalk']},
        'condition_response_feasibility': {'ready': bool(trials) and len(set(groups)) >= 3,
                                           'scope': 'context-only versus context plus manipulated addressed condition; no acoustic model'},
        'limits': ['no original audio in this deposit', 'behaviour labels accepted from published trial coding; additional blinded reliability audit not supplied',
                   'small trial count; only seven subject/caller components; no semantic/comprehension claim'],
    }
    result = {
        'schema_version': 1, 'code_provenance': code_provenance(),
        'download_manifest_sha256': sha(OUT/'receiver_dataset_downloads.json'),
        'birdpark': {'source_pairs': '10.5281/zenodo.7105196', 'source_groups': '10.5281/zenodo.13144875',
                     'records': records, 'group_annotation_release': group_summary,
                     'prediction_time': 'vocal offset; temporal candidates require 1-second pre and 5-second post windows',
                     'temporal_candidates': len(candidates), 'validated_behaviour_examples': 0,
                     'endpoint_gate': endpoint_gate(coverage=False, attributable=bool(records), independent=False,
                                                    validated=False, groups=len(records), valid_events=0),
                     'sensor_semantics': 'back-mounted accelerometer functions as a 15 Hz high-pass-filtered contact microphone; not independently validated locomotor behaviour',
                     'clock': 'shared clock and rational H5 rates; frame index is recording-global, annotation samples file-local',
                     'video_limits': 'documentation reports possible view offsets up to eight frames in an experiment; video presence is not proof of camera-level synchronization or visible bird identity',
                     'required_before_modelling': ['independent nonvocal video coding', 'bird identity and visibility per pre/post interval',
                                                  'camera alignment validation and inter-rater reliability', 'additional independent recording groups']},
        'elephants': elephant,
        'other_candidates': {'wild_marmosets': 'not included in this two-source experiment; Figshare landing page returned HTTP 403; selected exchange completeness remains unaudited'},
    }
    freeze(OUT/'birdpark_receiver_annotation_candidates.json', candidates)
    freeze(OUT/'receiver_grounding_dataset_audit.json', result)
    return result


def examples():
    trials = load_elephant_playback(ELEPHANTS)
    groups = independent_groups([{'subject': t['subject'], 'stimulus_id': 'caller:' + t['caller']} for t in trials])
    output, excluded = [], []
    for t, group in zip(trials, groups):
        raw = t['raw']
        values = [t['vigilance_pre_s'], math.log1p(float(raw['Distance'])), float(raw['Order60']),
                  float(raw['OtherAdults'])]
        if t['vigilance_post_s'] is None or not all(v is not None and math.isfinite(v) for v in values):
            excluded.append(t['event_id'])
            continue
        output.append({'event_id': t['event_id'], 'subject': t['subject'], 'caller': t['caller'],
                       'group': str(group), 'outcome': t['vigilance_post_s'], 'metadata': values,
                       # Common evaluator's additional-feature slot; kind=treatment prevents any acoustic claim.
                       'acoustics': [float(t['addressed_condition'])]})
    return output, excluded


def protocol():
    verified_inputs()
    rows, excluded = examples()
    return {
        'schema_version': 1, 'study': 'elephant independently measured receiver-response feasibility',
        'registration_status': 'local prospective specification; not an external registry',
        'code_provenance': code_provenance(), 'audit_sha256': sha(OUT/'receiver_grounding_dataset_audit.json'),
        'data_file': '20250704_playback_data_for_upload_corrected.csv',
        'data_sha256': sha(ELEPHANTS/'20250704_playback_data_for_upload_corrected.csv'),
        'endpoint': 'DurVigPost: seconds vigilant in 60 seconds after playback, per published trial coding',
        'prediction_time': 'stimulus onset; only pre-stimulus or stimulus-design inputs',
        'baseline_features': ['DurVigPre', 'log1p(Distance)', 'Order60', 'OtherAdults'],
        'added_feature': 'Treatment T=1/C=0, whether call originally addressed to playback subject',
        'models': 'context/pre-response baseline and same baseline plus treatment; StandardScaler then Ridge(alpha=1), no tuning',
        'identity_role': 'subjects/callers define grouped evaluation; no memorized individual intercept for unseen subjects',
        'partition': 'leave one connected component out; connect repeated subjects and all trials sharing a caller (conservative proxy for unknown stimulus reuse, not a stimulus ID)',
        'groups': len({r['group'] for r in rows}), 'eligible_trials': len(rows), 'excluded_trials': excluded,
        'metric': 'event-weighted squared error in seconds squared',
        'contrast': 'MSE(context) - MSE(context+treatment); positive favours added treatment information',
        'uncertainty': '2000 paired component bootstrap draws, seed=0; descriptive because OOF training folds overlap and only seven components',
        'preprocessing': 'fit scaler and ridge exclusively on training components, identical rows across arms; no imputation or feature selection',
        'stop_rule': 'one evaluation; no adaptive expansion',
        'blocked_comparisons': {'acoustic_response': 'no verified stimulus-call crosswalk',
                                'birdpark_behaviour': 'no independently validated behavioural outcomes'},
        'limits': ['small feasibility study; not a confirmatory test', 'treatment information is not an acoustic representation',
                   'predictive condition contrast does not establish semantics, intended comprehension or causality',
                   'NoiseBursts excluded because some noise is post-stimulus; no response-derived covariates used'],
        'cohort_sha256': hashlib.sha256(serialize(rows).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--audit', action='store_true')
    modes.add_argument('--preregister', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prereg = OUT/'elephant_receiver_condition_preregistration.json'
    if args.audit:
        a = audit()
        print(serialize({'audit': str(OUT/'receiver_grounding_dataset_audit.json'),
                         'birdpark_temporal_candidates': a['birdpark']['temporal_candidates'],
                         'birdpark_behaviour_gate': a['birdpark']['endpoint_gate'],
                         'elephant_trials': a['elephants']['corrected_playback_trials'],
                         'elephant_groups': a['elephants']['subject_and_caller_connected_groups']}))
    elif args.preregister:
        plan = protocol()
        freeze(prereg, plan)
        print(serialize({'preregistration': str(prereg), 'sha256': sha(prereg),
                         'trials': plan['eligible_trials'], 'groups': plan['groups']}))
    else:
        plan = protocol()
        if not prereg.exists() or prereg.read_text() != serialize(plan):
            raise RuntimeError('Missing or changed preregistration; no fitting')
        marker = OUT/'elephant_receiver_condition_RUN_STARTED.json'
        if marker.exists():
            raise RuntimeError('One-run stop rule prevents rerunning this study')
        freeze(marker, {'preregistration_sha256': sha(prereg)})
        rows, _ = examples()
        result = evaluate_response_models(rows, added_feature_kind='treatment')
        result['endpoint'] = plan['endpoint']
        result['preregistration_sha256'] = sha(prereg)
        result['code_provenance'] = code_provenance()
        result['scope'] = 'controlled-condition feasibility; acoustic comparison and BirdPark behavioural modelling remain blocked'
        freeze(OUT/'elephant_receiver_condition_results.json', result)
        print(serialize({k: result[k] for k in ('endpoint', 'mse', 'delta_metadata_minus_added', 'scope')}))


if __name__ == '__main__':
    main()
