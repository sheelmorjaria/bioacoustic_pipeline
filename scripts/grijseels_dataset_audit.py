#!/usr/bin/env python3
"""Source-specific, deterministic Grijseels audit; never fits a model.

Uses the annotation versions selected by the authors' analysis code.
The output freezes inventory, descriptive counts and candidate partitions,
not an assertion of complete observation or a training-ready dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/grijseels_turntaking/vocal_turn_taking_data'
CODE = ROOT / 'data/grijseels_analysis_code/code'
VERSIONS = {'one_monkey': 'v3', 'two_monkey': 'v2', 'three_monkey': 'v2cleaned'}
EXPECTED_SHA256 = 'cee6f11d0a01ad041945635a0acdec18b8d56f66d1d63b3dcb6febbbfeb88fe9'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def recording_identity(relative):
    path = Path(relative)
    context = path.parts[0]
    stem = path.parent.name if context == 'two_monkey' else path.stem.split('_call_data')[0]
    match = re.fullmatch(r'(.+?)_(\d{8})(.*)', stem)
    if not match:
        raise ValueError(f'Unparseable recording: {relative}')
    names = match[1].lower().split('_')
    expected = {'one_monkey': 1, 'two_monkey': 2, 'three_monkey': 3}[context]
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError(f'Caller order unresolved: {relative}')
    return f'{context}/{stem.lower()}', match[2], names


def timeline_stats(events, fps, horizon=5.0):
    """Events are (onset_frame, offset_frame, caller); ties have no chosen order."""
    events = sorted(events)
    groups = defaultdict(list)
    for event in events:
        groups[event[0]].append(event)
    starts = sorted(groups)
    tied_groups = sum(len(v) > 1 for v in groups.values())
    unique_pairs = [(groups[a][0], groups[b][0]) for a, b in zip(starts, starts[1:])
                    if len(groups[a]) == len(groups[b]) == 1]
    short = [(a, b) for a, b in unique_pairs if 0 < (b[0] - a[0]) / fps <= horizon]
    # Runs are descriptive annotation chains, not verified conversations.
    runs = []
    for start in starts:
        if not runs or (start - runs[-1][-1]) / fps > horizon:
            runs.append([])
        runs[-1].append(start)
    run_callers = [len({e[2] for start in run for e in groups[start]}) for run in runs]
    return {
        'events': len(events), 'unique_onsets': len(starts),
        'tied_onset_groups': tied_groups,
        'events_in_tied_onset_groups': sum(len(v) for v in groups.values() if len(v) > 1),
        'ordered_adjacent_onset_group_pairs': max(0, len(starts) - 1),
        'adjacent_pairs_with_singleton_onsets': len(unique_pairs),
        'singleton_pairs_within_5s': len(short),
        'different_caller_singleton_pairs_within_5s': sum(a[2] != b[2] for a, b in short),
        'next_call_overlaps_current_call': sum(b[0] < a[1] for a, b in unique_pairs),
        'chains_with_exactly_two_callers': run_callers.count(2),
        'chains_with_exactly_three_callers': run_callers.count(3),
        'first_onset_s': events[0][0] / fps if events else None,
        'last_offset_s': max(e[1] for e in events) / fps if events else None,
        'events_ending_after_1800s': sum(e[1] / fps > 1800 for e in events),
    }


def partition_for_group(group):
    value = int(hashlib.sha256(('grijseels-audit-v1:' + group).encode()).hexdigest(), 16)
    return 'fit' if value % 100 < 70 else 'heldout'


def build_audit():
    archive = ROOT / 'data/grijseels_turntaking.zip'
    if sha256(archive) != EXPECTED_SHA256:
        raise ValueError('Dataset archive does not match the verified Dryad digest')
    with zipfile.ZipFile(archive) as zipped:
        archived_files = {n for n in zipped.namelist() if not n.endswith('/')}
        local_files = {p.relative_to(DATA.parent).as_posix()
                       for p in DATA.rglob('*') if p.is_file()}
        if archived_files != local_files:
            raise ValueError('Extracted file inventory differs from verified archive')
        for name in sorted(archived_files):
            if hashlib.sha256(zipped.read(name)).hexdigest() != sha256(DATA.parent / name):
                raise ValueError(f'Extracted file differs from archive: {name}')
    fps = float(loadmat(DATA / 'metadata/fps.mat')['fps'].item())
    monkey_rows = csv_rows(DATA / 'metadata/monkey_list.csv')
    name_ids = {r['Monkey name'].strip("'").lower(): int(r['Monkey number']) for r in monkey_rows}
    inventory = []
    records = []
    signatures = defaultdict(list)
    event_keys = {}
    schema_counts = Counter()
    for path in sorted(DATA.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(DATA).as_posix()
        inventory.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
        if path.parts[-2] == 'metadata' or path.suffix != '.mat':
            continue
        rid, date, names = recording_identity(rel)
        version = path.stem.split('_call_data_')[-1] if '_call_data_' in path.stem else path.stem.removeprefix('call_data_')
        selected = version == VERSIONS[Path(rel).parts[0]]
        d = loadmat(path)
        schema_counts.update(k for k in d if not k.startswith('__'))
        cells = d['call_data']
        issues = Counter()
        events = []
        per_caller = []
        if cells.shape != (1, len(names)):
            issues['unexpected_cell_array_shape'] += 1
        for i, cell in enumerate(cells.flat):
            a = np.asarray(cell)
            if a.size == 0:
                per_caller.append(0)
                continue
            if a.ndim != 2 or a.shape[1] != 2 or i >= len(names):
                issues['invalid_event_array_shape'] += 1
                continue
            per_caller.append(len(a))
            finite = np.isfinite(a).all(axis=1)
            issues['nonfinite_events'] += int((~finite).sum())
            clean = a[finite].astype(float)
            issues['negative_timestamps'] += int((clean < 0).any(axis=1).sum())
            issues['nonpositive_duration'] += int((clean[:, 1] <= clean[:, 0]).sum())
            issues['noninteger_frame_values'] += int((clean != np.round(clean)).any(axis=1).sum())
            issues['onset_order_violations'] += int((np.diff(clean[:, 0]) < 0).sum())
            issues['duplicate_rows_within_caller'] += len(clean) - len(np.unique(clean, axis=0))
            events.extend((float(a), float(b), names[i]) for a, b in clean)
        row = {'recording_id': rid, 'date': date, 'path': rel, 'version': version,
               'selected': selected, 'callers_in_cell_order': names,
               'global_monkey_ids': [name_ids.get(n) for n in names],
               'events_by_caller': per_caller,
               'mat_variables': sorted(k for k in d if not k.startswith('__')),
               'issues': {k: v for k, v in issues.items() if v}}
        if selected:
            row['timeline'] = timeline_stats(events, fps)
            # Fingerprint includes caller names, ignoring MAT encoding and row order.
            signature = hashlib.sha256(json.dumps(sorted(events)).encode()).hexdigest()
            row['timeline_sha256'] = signature
            if events:
                signatures[signature].append(rid)
            if rid.startswith('two_monkey/'):
                keys = sorted((name_ids[n], round(a), round(b)) for a, b, n in events)
                event_keys[rid] = keys
        records.append(row)
    chosen = [r for r in records if r['selected']]
    selected_issues = Counter()
    for r in chosen:
        selected_issues.update(r['issues'])
    if len({r['recording_id'] for r in chosen}) != len(chosen):
        raise ValueError('Selected recording IDs are not unique')
    # Keep dates and identical timelines together. Caller overlap is reported separately.
    parent = {r['date']: r['date'] for r in chosen}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    by_id = {r['recording_id']: r for r in chosen}
    duplicate_timelines = [ids for ids in signatures.values() if len(ids) > 1]
    for ids in duplicate_timelines:
        dates = sorted({by_id[rid]['date'] for rid in ids})
        for date in dates[1:]:
            parent[find(date)] = find(dates[0])
    for r in chosen:
        r['partition_group'] = find(r['date'])
        r['candidate_partition'] = partition_for_group(r['partition_group'])
    # Crosswalk Fig 2 numeric session IDs by all caller/onset/offset keys,
    # rather than trusting directory enumeration or CSV row order.
    feature_rows = csv_rows(DATA / 'metadata/Fig_2_features.csv')
    by_session = defaultdict(list)
    missing_features = Counter()
    for r in feature_rows:
        by_session[int(r['Session'])].append(r)
        for key, value in r.items():
            if value.lower() == 'nan' or not value:
                missing_features[key] += 1
    crosswalk = []
    for session, rows in sorted(by_session.items()):
        keys = sorted((int(r['Monkey 2']), round(float(r['Start time']) * fps),
                       round((float(r['Start time']) + float(r['Call length'])) * fps))
                      for r in rows)
        matches = [rid for rid, candidate in event_keys.items() if candidate == keys]
        crosswalk.append({'feature_session': session, 'events': len(rows), 'matching_recording_ids': matches})
    summaries = {}
    for context in VERSIONS:
        selected = [r for r in chosen if r['recording_id'].startswith(context + '/')]
        all_records = [r for r in records if r['recording_id'].startswith(context + '/')]
        numeric_keys = [k for k in selected[0]['timeline'] if k not in ('first_onset_s', 'last_offset_s')]
        summaries[context] = {
            'all_annotation_files': len(all_records), 'selected_recordings': len(selected),
            'recording_groups_without_selected_version': sorted({r['recording_id'] for r in all_records} - {r['recording_id'] for r in selected}),
            'callers': sorted({n for r in selected for n in r['callers_in_cell_order']}),
            'counts': {k: sum(r['timeline'][k] for r in selected) for k in numeric_keys},
        }
    caller_sets = {p: {n for r in chosen if r['candidate_partition'] == p for n in r['callers_in_cell_order']}
                   for p in ('fit', 'heldout')}
    evidence_paths = ['utils/get_monkey_call_times.m', 'utils/load_names.m', 'utils/load_names_3m.m',
                      'utils/findFolders.m', 'fig3_linear_and_random_forest_models.m',
                      'fig5_one_and_three_monkey_paradigms.m']
    return {
        'schema_version': 1, 'audit_kind': 'descriptive source contract; no training or model selection',
        'source': {'doi': '10.5061/dryad.9ghx3ffpx', 'code_doi': '10.5281/zenodo.8415531',
                   'archive_sha256': EXPECTED_SHA256, 'archive_sha256_verified': True,
                   'all_extracted_files_verified_against_archive': True,
                   'readme_sha256': sha256(ROOT / 'data/grijseels_README.md'),
                   'script_sha256': sha256(Path(__file__)),
                   'code_evidence': {p: sha256(CODE / p) for p in evidence_paths}},
        'recording_identity': {
            'definition': 'context plus filename/folder recording stem; versions share recording ID',
            'selected_versions': VERSIONS, 'selection_evidence': evidence_paths[-1],
            'all_annotation_files': len(records), 'selected_recordings': len(chosen),
            'unique_dates': len(parent), 'duplicate_selected_timelines': duplicate_timelines,
            'boundary_status': 'file/folder is a recording group; independence between same-day recordings is not established',
        },
        'timestamps': {
            'stored_units': 'downsampled frame indices', 'frames_per_second': fps,
            'text_fps_rounded': float((DATA / 'metadata/fps.txt').read_text()),
            'conversion': 'seconds = stored frame index / fps.mat fps; matches get_monkey_call_times.m',
            'reference': 'shared recording-local coordinate across caller cells, as used by author code',
            'absolute_clock_time': 'not provided per event; date is available in names',
            'frame_origin': 'zero versus one based not explicitly documented; no offset subtracted, matching author code',
            'versions': 'conversion applies to selected versions only; discarded versions are not pooled',
        },
        'caller_identity': {
            'attribution': 'cell index maps to ordered names in recording filename/folder, as in author code',
            'global_metadata_registry_callers': len(name_ids),
            'selected_names_absent_from_registry': sorted({n for r in chosen for n in r['callers_in_cell_order']} - name_ids.keys()),
            'aliases': 'no aliases inferred; e.g. leo and leonardo remain distinct',
        },
        'recipient_identity': {'status': 'unavailable in inspected event arrays',
                               'rule': 'other participant and next caller are not annotated intended recipients',
                               'observed_mat_variables': dict(schema_counts)},
        'event_quality': {
            'selected_issue_totals': dict(selected_issues),
            'records_requiring_review': [r['recording_id'] for r in chosen if r['issues']],
            'empty_selected_recordings': [r['recording_id'] for r in chosen if not r['timeline']['events']],
            'count_policy': 'descriptive counts retain all finite rows, including flagged events; not a modelling risk set',
            'modelling_gate': 'exclude flagged recordings until a separate correction/exclusion policy is frozen; never repair end-before-start silently',
        },
        'temporal_continuity': {
            'event_spacing': 'uncompressed onset/offset indices retain inter-call distances',
            'coverage_intervals': 'not supplied as explicit recording start/end and gap intervals',
            'silence': 'absence of listed calls does not independently verify complete monitoring or annotation',
            'boundaries': 'first/last calls are observed event limits, not recording bounds',
            'nominal_1800s': 'author code uses 1800s in some analyses; events beyond 1800s prevent adopting this as a universal bound',
            'solo_context_caveat': 'five supplemental txt logs mention antiphonal/playback calls; single caller cell is not proof of acoustic isolation',
        },
        'exchange_structure': {
            'definition': 'descriptive annotation chains split when consecutive distinct onsets differ by >5 seconds; count distinct callers in each chain',
            'pair_definition': 'adjacent singleton onset groups with onset-to-onset interval in (0,5] seconds; different caller count reported separately',
            'ties': 'no arbitrary caller ordering; exclude pairs touching tied onset groups',
            'overlap': 'retained and counted; onset order does not imply the preceding call ended',
            'claim_limit': 'chains and pairs are operational counts, not intended exchanges or recipient responses',
            'by_context': summaries,
        },
        'negative_examples': {
            'verified_nonresponse_windows': None,
            'status': 'blocked pending independent coverage and annotation-completeness evidence',
            'rule': 'do not code terminal windows, unverified gaps or missing observations as negatives',
            'future_rule': 'for a frozen horizon H require [prediction time, prediction time + H] inside verified coverage for candidate j',
        },
        'metadata': {
            'feature_rows': len(feature_rows), 'feature_sessions': len(by_session),
            'missing_values_by_feature': dict(missing_features),
            'feature_crosswalk_method': 'exact multiset of caller Monkey 2, rounded start frame and rounded end frame; author fig3 labels Monkey 2 as caller and Monkey 1 as response candidate',
            'feature_session_crosswalk': crosswalk,
            'uniquely_matched_feature_sessions': sum(len(r['matching_recording_ids']) == 1 for r in crosswalk),
            'feature_crosswalk_complete_for_nonempty_pair_recordings': (
                {r['recording_id'] for r in chosen
                 if r['recording_id'].startswith('two_monkey/') and r['timeline']['events']}
                == {r['matching_recording_ids'][0] for r in crosswalk
                    if len(r['matching_recording_ids']) == 1}),
            'per_session_demographics_rows': len(csv_rows(DATA / 'metadata/monkey_data_per_session_v2.csv')),
            'demographics_join': 'unkeyed CSV rows: not automatically joined by row number; author enumeration needs separate verification',
            'future_leakage': 'published Response and ICIs require feature provenance review; Response is not recipient annotation; do not use target labels as input',
            'audio': 'no raw audio in this archive; spectral feature table covers paired context only; pulse counts also occur in MAT files',
        },
        'data_independence': {
            'candidate_split_only': True, 'algorithm': 'SHA256(grijseels-audit-v1:group) modulo 100 <70 => fit; otherwise heldout',
            'grouping': 'same date across contexts plus union of dates containing identical nonempty selected timelines',
            'partition_counts': dict(Counter(r['candidate_partition'] for r in chosen)),
            'partition_group_counts': dict(Counter(partition_for_group(g) for g in {r['partition_group'] for r in chosen})),
            'callers_shared_across_partitions': sorted(caller_sets['fit'] & caller_sets['heldout']),
            'claim_limit': 'recording/date holdout is not unseen-individual or independent-social-group generalization',
            'training_rule': 'identical eligible sessions/events across arms; validation and preprocessing within fit partition only; cluster uncertainty by partition group',
        },
        'task_contracts': {
            'A_next_caller': {
                'status': 'supports observed-next-annotated-caller ordering; natural next-caller claim conditional on coverage/completeness',
                'target': 'global caller of next annotated onset in same selected recording',
                'eligibility': 'two/three callers; quality-cleared recordings; current and next onset groups singleton; never link across recording boundaries; exclude final event',
                'prediction_time': 'must be frozen before modelling; onset uses onset-available features only; offset-time prediction must exclude successors already started',
            },
            'B_within_H': {'status': 'blocked for reliable binary nonresponse endpoint',
                           'target': 'named candidate j calls in (t,t+H], without intended-recipient claim',
                           'requirements': 'freeze H, prediction time, candidate set, coverage and censoring policy before fitting'},
            'shared_interface': ['source_id', 'recording_id', 'partition_group', 'caller_id',
                                 'onset_s', 'offset_s', 'recipient_id=null', 'coverage_status', 'provenance'],
            'model_ladder': 'metadata / +current event / +past events on identical eligible heldout material; no fits performed',
            'cross_species': 'compare incremental predictive value on source-specific endpoints; do not equate marmoset next-caller with bat annotated-recipient response',
        },
        'inventory': inventory, 'recordings': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/grijseels_dataset_audit.json')
    args = parser.parse_args()
    out = build_audit()
    payload = json.dumps(out, indent=2, allow_nan=False, sort_keys=True) + '\n'
    if args.output.exists() and args.output.read_text() != payload:
        raise SystemExit('Existing frozen audit differs; choose a new --output path instead of overwriting')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({'output': str(args.output), 'sha256': sha256(args.output),
                      'selected_recordings': out['recording_identity']['selected_recordings'],
                      'contexts': out['exchange_structure']['by_context'],
                      'feature_crosswalk_matches': out['metadata']['uniquely_matched_feature_sessions'],
                      'partition_counts': out['data_independence']['partition_counts']}, indent=2))


if __name__ == '__main__':
    main()
