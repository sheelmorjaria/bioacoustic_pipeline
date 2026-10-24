"""Source-independent contracts for attributable, observed receiver outcomes.

Available partners are candidate receivers, never inferred addressees. Coverage,
identity certainty and independent behavioural validation are separate gates.
"""
from __future__ import annotations

import hashlib
import csv
import io
import math
import stat
import zipfile
from collections import Counter
from pathlib import Path


def parse_birdpark_events(rows, sampling_rate, available):
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError('Positive sampling rate required')
    events, excluded = [], Counter()
    for row in rows:
        if row.get('segmentType', row.get('segType')) != 'consolidated':
            excluded['other_annotation_layer'] += 1
            continue
        caller, transmitter = str(row.get('bird', '')), str(row.get('transmitter', ''))
        if caller not in available:
            excluded['unavailable_or_unknown_caller'] += 1
            continue
        if transmitter != caller:
            excluded['inferred_or_inconsistent_identity'] += 1
            continue
        try:
            onset, offset = float(row['onset']), float(row['offset'])
        except (ValueError, TypeError, KeyError):
            excluded['invalid_timestamps'] += 1
            continue
        if not (math.isfinite(onset) and math.isfinite(offset)
                and onset.is_integer() and offset.is_integer() and 1 <= onset <= offset):
            excluded['invalid_timestamps'] += 1
            continue
        events.append({'onset_s': (onset - 1) / sampling_rate,
                       'offset_s': offset / sampling_rate, 'caller': caller, 'recipient': None,
                       'overlapping_annotation': float(row.get('nOverlapSyl', 0)) > 0})
    return sorted(events, key=lambda e: e['onset_s']), dict(excluded)


def covers_window(start, end, intervals):
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        return False
    cursor = start
    for left, right in sorted(intervals):
        if not (math.isfinite(left) and math.isfinite(right)) or right < left:
            raise ValueError('Invalid coverage interval')
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = right
        if cursor >= end:
            return True
    return False


def response_label(latency, event_observed, coverage_seconds, horizon):
    """Explicit event/censor flag required. Missing data never become silence."""
    if horizon <= 0:
        raise ValueError('Positive horizon required')
    if latency is not None and math.isfinite(latency) and latency < 0:
        raise ValueError('Negative response latency')
    if event_observed is None or coverage_seconds is None:
        return None
    if not math.isfinite(coverage_seconds) or coverage_seconds < 0:
        raise ValueError('Invalid observation coverage')
    if event_observed:
        if latency is None or not math.isfinite(latency) or latency > coverage_seconds:
            return None
        if latency <= horizon:
            return 1
    return 0 if coverage_seconds >= horizon else None


def join_stimulus_features(trials, features):
    lookup = {}
    for row in features:
        key = row['call_id']
        if key in lookup:
            raise ValueError(f'Duplicate acoustic call ID: {key}')
        lookup[key] = row
    output = []
    for trial in trials:
        key = trial['stimulus_id']
        if key not in lookup:
            raise ValueError(f'Missing exact stimulus acoustic match: {key}')
        output.append({**trial, 'stimulus_features': lookup[key]})
    return output


def independent_groups(rows):
    """Connected components: repeated subjects OR stimuli stay in one group."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for field in ('subject', 'stimulus_id'):
            value = row.get(field)
            if value is None or str(value).strip() in ('', 'NA', 'nan'):
                raise ValueError(f'Missing independence key: {field}')
            key = (field, value)
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    stable = {i: min(members) for members in components.values() for i in members}
    return [stable[i] for i in range(len(rows))]


def verified_extract(path, destination, checksum):
    path, destination = Path(path), Path(destination).resolve()
    algorithm, expected = checksum.split(':', 1)
    hasher = hashlib.new(algorithm)
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            hasher.update(chunk)
    if hasher.hexdigest() != expected:
        raise ValueError(f'Download checksum mismatch: {path}')
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        if archive.testzip() is not None:
            raise ValueError('Corrupt ZIP archive')
        archive.extractall(destination)
    return sorted(str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file())


def endpoint_gate(*, coverage, attributable, independent, validated, groups, valid_events):
    checks = {'coverage': coverage, 'attributable_identity': attributable,
              'independent_outcome': independent, 'validated_behaviour': validated,
              'independent_groups': groups >= 3, 'valid_events': valid_events > 0}
    return {'ready': all(checks.values()),
            'blocked_by': [key for key, passed in checks.items() if not passed]}


def load_elephant_playback(folder):
    path = Path(folder) / '20250704_playback_data_for_upload_corrected.csv'
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252')
    trials = []
    for row in csv.DictReader(io.StringIO(text)):
        def number(key):
            value = row.get(key, '')
            if value in ('', 'NA', 'NaN'):
                return None
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            return numeric

        if row['Treatment'] not in ('T', 'C'):
            raise ValueError('Unknown playback treatment')
        pre, post, delta = (number(k) for k in ('DurVigPre', 'DurVigPost', 'DurVigDiff'))
        if any(v is not None and not 0 <= v <= 60 for v in (pre, post)):
            raise ValueError('Vigilance duration outside documented 60-second window')
        if all(v is not None for v in (pre, post, delta)) and not math.isclose(post-pre, delta, abs_tol=.01):
            raise ValueError('Inconsistent vigilance difference')
        flag = number('VocCen')
        if flag not in (None, 0, 1):
            raise ValueError('Unknown vocal censor/event flag')
        trials.append({
            'event_id': ':'.join(row[k] for k in ('Date', 'Time', 'Subject')),
            'subject': row['Subject'], 'caller': row['Caller'], 'stimulus_id': None,
            'video_playback_offset_s': number('Playback'),
            'addressed_condition': int(row['Treatment'] == 'T'),
            'vigilance_pre_s': pre, 'vigilance_post_s': post,
            'vigilance_difference_s': delta,
            'vocal_response_600s': response_label(number('LatVoc'), None if flag is None else bool(flag), 600, 600),
            'raw': row,
        })
    if len({t['event_id'] for t in trials}) != len(trials):
        raise ValueError('Duplicate playback trial identifier')
    return trials


def audit_birdpark_pair(h5_path, csv_path, *, pre_seconds=1., horizon_seconds=5.):
    """Return temporal candidate windows, never imputed behavioural labels."""
    import h5py
    import numpy as np
    with h5py.File(h5_path) as f:
        rec, info = f['recInfo'].attrs, f['fileInfo'].attrs
        fs = float(rec['sampleRate_Hz'])
        fps = float(rec['frameRate_Hz'])
        samples, frames = int(info['nSamplesInFile']), int(info['nFramesInFile'])
        birds = [b.decode() if isinstance(b, bytes) else str(b) for b in rec['radioChNames']]
        if len(birds) != len(set(birds)):
            raise ValueError('Ambiguous radio/bird identity mapping')
        if not math.isclose(fs, fps * int(rec['nSamplesPerFrame'])):
            raise ValueError('Inconsistent H5 clock rates')
        if samples != frames * int(rec['nSamplesPerFrame']):
            raise ValueError('Inconsistent H5 time coverage')
        duration = samples / fs
        signal_quality = {}
        for key in ('radioSignals', 'daqSignals'):
            data = f[key]
            if data.shape[1] != samples:
                raise ValueError('Signal length differs from declared coverage')
            if key == 'radioSignals' and data.shape[0] != len(birds):
                raise ValueError('Radio identity count differs from channels')
            invalid = 0
            for start in range(0, samples, 1_000_000):
                invalid += int((~np.isfinite(data[:, start:start+1_000_000])).sum())
            signal_quality[key] = {'shape': list(data.shape), 'nonfinite_samples': invalid}
        radio_quality = None
        if 'radioFrameSignals/powRM' in f and 'radioFrameSignals/powNM' in f:
            margin = f['radioFrameSignals/powRM'][:] - f['radioFrameSignals/powNM'][:]
            radio_quality = {'power_minus_noise_quantiles': np.nanquantile(margin, [0, .01, .5, 1]).tolist(),
                             'nonfinite_frames': int((~np.isfinite(margin)).sum()),
                             'quality_gate': 'diagnostic only; no behavioural validity inferred from radio power'}
        recording_id = str(rec['recId'])
        video_declared = bool(info.get('hasVideo', False))
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as fh:
        events, excluded = parse_birdpark_events(list(csv.DictReader(fh)), fs, birds)
    candidates, outside = [], 0
    for i, event in enumerate(events):
        t = event['offset_s']
        if event['offset_s'] > duration or not covers_window(t-pre_seconds, t+horizon_seconds, [(0., duration)]):
            outside += 1
            continue
        for receiver in birds:
            if receiver == event['caller']:
                continue
            candidates.append({'event_id': f'{Path(h5_path).stem}:{i}:{receiver}',
                               'recording_id': recording_id, 'file': Path(h5_path).name,
                               'caller': event['caller'], 'candidate_receiver': receiver,
                               'recipient': None, 'prediction_time_s': t,
                               'pre_window_s': [t-pre_seconds, t], 'post_window_s': [t, t+horizon_seconds],
                               'overlapping_annotation': event['overlapping_annotation'],
                               'behaviour_validated': False, 'behaviour_pre': None, 'behaviour_post': None,
                               'required_validation': 'individual identity, visible video intervals and cross-camera timing; independent nonvocal coding and reliability before fitting'})
    return {'file': Path(h5_path).name, 'recording_id': recording_id, 'sample_rate_hz': fs,
            'video_frame_rate_hz': fps, 'duration_s': duration, 'available_birds': birds,
            'attributable_consolidated_events': len(events), 'exclusions': excluded,
            'outside_full_pre_post_window': outside,
            'temporally_bounded_candidate_windows': len(candidates),
            'validated_behaviour_windows': 0, 'video_declared': video_declared,
            'video_file_present': Path(h5_path).with_suffix('.mp4').exists(),
            'video_clock_identity_and_behaviour_validated': False,
            'signal_quality': signal_quality, 'radio_quality': radio_quality}, candidates
