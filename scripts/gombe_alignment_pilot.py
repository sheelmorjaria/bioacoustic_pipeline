#!/usr/bin/env python3
"""Prepare a bounded two-reviewer alignment pilot; no automatic annotations."""
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import sys

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.gombe_alignment_pilot import sampled_windows


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    folder = ROOT / 'outputs/gombe_alignment_pilot'
    if folder.exists():
        raise RuntimeError('Pilot namespace already exists;no regeneration after review')
    source = ROOT / 'data/gombe/Adults_3.wav'
    prior_path = ROOT / 'outputs/gombe_candidate_audit.json'
    prior = json.loads(prior_path.read_text())
    expected = next(f['sha256'] for f in prior['files'] if f['original_name'] == 'Adults 3.wav')
    if sha(source) != expected:
        raise ValueError('Source waveform integrity mismatch')
    info = sf.info(source)
    windows = sampled_windows(info.duration)
    protocol = {
        'study': 'Gombe recording-cut and catalogue-alignment feasibility pilot v1',
        'status': 'prospective annotation protocol;not a MACDE modelling study',
        'source': {'path': str(source.relative_to(ROOT)), 'sha256': expected,
                   'sample_rate': info.samplerate, 'subtype': info.subtype},
        'sampling': {'windows': 24, 'seconds_per_window': 20,
                     'rule': 'One fixed20-second window centered in each of24 equal-duration tape strata;start rounded down to whole second;no acoustic-boundary or catalogue-based selection',
                     'scope': 'These are sampled inspection windows,not24 verified/candidate recording cuts;multiple or zero cut transitions may occur in a window',
                     'coverage': 'One acquired tape only;no generalization to all seven tapes'},
        'review_design': {'reviewers': 2, 'independence': 'Distinct humans complete initial sheets separately without seeing each other;record reviewer identity and attest independent initial review',
                          'phase1': 'Review cut transitions before consulting catalogue;record evidence such as spoken cut number or unambiguous recording/edit cue. Acoustic change alone is insufficient.',
                          'phase2': 'After phase1 is saved and checksummed,consult catalogue reference;record spoken number verbatim and evidence for a specific ML catalogue link separately',
                          'source_population': 'Mixed recording audio;do not count a chimp call onset as a source-cut boundary',
                          'uncertainty': 'Mark uncertain or no_transition explicitly;preserve uncertainty interval and alternative cut/catalogue interpretations'},
        'coordinates': {'review_timestamp': 'Seconds relative to the extracted20-second clip,not field-clock time',
                        'parent_sample_conversion': 'parent_sample = window_start_sample + round(clip_seconds *96000);source96kHz/sample offsets pinned per manifest',
                        'sample_rounding': 'Nearest integer sample;no conversion to a field session or catalogue identity without separate evidence'},
        'analysis_plan': {'matching': 'One-to-one nearest same-window boundary matches within0.25seconds;maximize match count then minimize timing difference;report unmatched events for each reviewer',
                          'report': ['window-level transition/no-transition/uncertain agreement including negatives',
                                     'paired-boundary precision/recall/F1 (reviewer-relative,not ground-truth accuracy)',
                                     'absolute timestamp difference median and95th percentile',
                                     'catalogue agreement only among pairs where both independently supplied IDs;report abstention denominator',
                                     'disagreements and original uncertainty preserved before adjudication'],
                          'feasibility_rule': 'At least20 independently paired boundary events;boundary F1>=0.8;median timing difference<=0.25seconds;catalogue agreement>=0.9 among at least15 independently assigned pairs;otherwise insufficient/inconclusive and no automatic expansion',
                          'adjudication': 'After initial reviews are locked,resolve disagreements with explicit source evidence;agreement alone is not authoritative correctness',
                          'session_gate': 'Neither boundary agreement nor catalogue agreement certifies a session;parent/event relationships and prospective group contract remain mandatory'},
        'stop_rule': 'Fixed24 windows;no adaptive replacement,extra listening outside sampled windows or threshold changes based on outcomes;all unreviewed fields remain missing',
        'limits': 'No human reviews yet;no verified boundaries or identities;no independent groups;no partitions,model fitting or semantic conclusions',
        'code_sha256': sha(Path(__file__)),
        'sampling_contract_sha256': sha(ROOT / 'src/bioacoustic_pipeline/ingestion/gombe_alignment_pilot.py'),
        'prior_catalogue_audit_sha256': sha(prior_path),
    }
    folder.mkdir(parents=True)
    dump(folder / 'protocol.json', protocol)
    clips = folder / 'clips'
    clips.mkdir()
    for window in windows:
        start = window['start_seconds'] * info.samplerate
        stop = window['end_seconds'] * info.samplerate
        audio, sr = sf.read(source, start=start, stop=stop, dtype='int32')
        path = clips / (window['window_id']+'.wav')
        sf.write(path, audio, sr, subtype=info.subtype)
        decoded, decoded_sr = sf.read(path, dtype='int32')
        if decoded_sr != sr or decoded.shape != audio.shape or not (decoded == audio).all():
            raise RuntimeError('Extracted native PCM does not match original samples')
        window.update({'start_sample': start, 'end_sample_exclusive': stop,
                       'path': str(path.relative_to(folder)), 'sha256': sha(path), 'frames': len(audio)})
    dump(folder / 'window_manifest.json', {'source_sha256': expected, 'windows': windows})
    fields = ['window_id', 'status', 'event_index', 'boundary_clip_seconds',
              'uncertainty_start_clip_seconds', 'uncertainty_end_clip_seconds',
              'boundary_verified', 'evidence', 'spoken_cut_number_verbatim',
              'catalogue_id', 'catalogue_id_verified', 'catalogue_evidence',
              'session_independence_verified', 'session_evidence', 'notes']
    for reviewer, seed in [('A', 4701), ('B', 4702)]:
        order = list(windows)
        random.Random(seed).shuffle(order)
        with (folder / f'reviewer_{reviewer}.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows({'window_id': r['window_id'], 'status': 'not_reviewed'} for r in order)
        dump(folder / f'reviewer_{reviewer}_attestation.json', {'reviewer_id': None, 'independent_initial_review': None,
                                                             'phase1_sheet_sha256': None, 'phase2_sheet_sha256': None,
                                                             'note': 'Complete only after actual review;do not self-attest for an automated model'})
    with (folder / 'catalogue_reference.csv').open('w', newline='') as target:
        columns = ['catalogue_id', 'Recordist record number', 'parsed_date', 'Individuals Vocalizing', 'Individual(s) with Sound/call type', 'Context of vocalizations']
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in columns} for r in prior['catalogue']['records'])
    cards = ''.join(f'<section><h2>{html.escape(w["window_id"])}</h2><audio controls preload="none" src="{html.escape(w["path"])}"></audio></section>' for w in windows)
    (folder / 'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Gombe alignment pilot</title>'
        '<style>body{font:18px system-ui;max-width:850px;margin:3rem auto}section{padding:1rem;border-bottom:1px solid #bbb}</style>'
        '<h1>Gombe alignment review</h1><p>Review your assigned CSV order independently. These are sampled windows,not verified cuts. '
        'Timestamps are relative to each clip. Save and checksum phase1 before consulting catalogue_reference.csv. '
        'Use reviewed/no_transition/uncertain status as appropriate;copy a row for each additional event in the same window. '
        'A vocal onset is not a recording-cut boundary. Keep boundary,catalogue and session evidence separate.</p>'+cards+'</html>')
    dump(folder / 'package_integrity.json', {'protocol_sha256': sha(folder / 'protocol.json'),
                                           'files': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
                                           'human_reviews_received': 0, 'verified_intervals': 0,
                                           'certified_independent_groups': 0, 'stage2_ready': False})
    print('Prepared', folder, '24 nativePCM windows;zero human reviews', flush=True)


if __name__ == '__main__':
    main()
