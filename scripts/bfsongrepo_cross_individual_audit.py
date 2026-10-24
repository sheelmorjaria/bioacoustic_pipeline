#!/usr/bin/env python3
"""Bounded source-support audit, not a new forecasting evaluation or partition."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf
import torch

import bfsongrepo_tier_a_audit as shared
from macde_source_resolution import pcm_hash
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import (
    rec_metadata, decode_cbin_channel, continuity_summary,
)

ROOT = shared.ROOT


def audit_recording(name, payload, rec_bytes, expected_bird, expected_day):
    bird, day = Path(name).parts[-3:-1]
    if (bird, day) != (expected_bird, expected_day):
        raise ValueError('Unexpected bird/day archive member')
    metadata = rec_metadata(rec_bytes.decode())
    native = decode_cbin_channel(payload, metadata)
    audio = native.astype(np.float64) / 32768.
    sr = metadata['sample_rate']
    obs = frame_observations(audio, sr)
    valid = obs['time_valid']
    support = continuity_summary(valid, sr)
    date = datetime.strptime(day, '%m%d%y').date().isoformat()
    created = metadata['file_created']
    return {
        'recording_id': str(Path(name).with_suffix('')), 'bird_id': bird,
        'archive_day_code': day, 'documented_date': date,
        'file_created_from_rec': created,
        'rec_date_agrees_with_archive': created is not None and created[:10] == date,
        'cbin_member': name, 'cbin_sha256': hashlib.sha256(payload).hexdigest(),
        'rec_sha256': hashlib.sha256(rec_bytes).hexdigest(),
        'decoded_pcm_sha256': pcm_hash(audio, sr), 'sample_rate': sr,
        'duration_seconds': len(audio)/sr, 'full_recording_support': support,
        'session_id': None, 'partition': None,
        'decoded_audio_policy': 'Original channel0 signed big-endian int16 /32768; no gain normalization',
    }, valid


def main():
    torch.set_num_threads(2)
    output = ROOT/'outputs/bfsongrepo_cross_individual_support_audit.json'
    mask_path = ROOT/'outputs/bfsongrepo_cross_individual_support_masks.npz'
    if output.exists() or mask_path.exists():
        raise RuntimeError('Refusing to overwrite support audit')
    plan_path = ROOT/'data/bfsongrepo/cross_individual_support_plan.json'
    plan = json.loads(plan_path.read_text())
    old_pcm = set()
    for path in sorted((ROOT/'data/bengalifinch').rglob('*.wav')):
        audio, sr = sf.read(path)
        old_pcm.add(pcm_hash(audio, sr))
    previous_pcm = set()
    prior = []
    for filename in ['bfsongrepo_tier_a_audit.json', 'bfsongrepo_multiday_audit.json',
                     'bfsongrepo_fourth_day_audit.json']:
        path = ROOT/'outputs'/filename
        document = json.loads(path.read_text())
        prior.append({'path': str(path.relative_to(ROOT)), 'sha256': shared.sha(path)})
        # Preserve nesting in older multi-day audit without assuming its schema.
        def collect(value):
            if isinstance(value, dict):
                if 'decoded_pcm_sha256' in value:
                    previous_pcm.add(value['decoded_pcm_sha256'])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(document)
    rows, masks, integrity = [], {}, []
    seen = {}
    for selection in plan['selected']:
        path, info = ROOT/selection['local_path'], selection['file']
        if path.stat().st_size != info['size'] or shared.sha(path, 'md5') != info['computed_md5']:
            raise ValueError('Archive integrity failure')
        integrity.append({'path': selection['local_path'], 'file_id': info['id'],
                          'md5': info['computed_md5'], 'sha256': shared.sha(path)})
        day = info['name'].split('.')[3]
        recs, member_names = {}, set()
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name in member_names:
                    raise ValueError('Duplicate archive members')
                member_names.add(member.name)
                if member.name.endswith('.rec'):
                    recs[member.name] = archive.extractfile(member).read()
        # Sequential decompression avoids repeated seeks through the gzip stream.
        with tarfile.open(path) as archive:
            for member in archive:
                name = member.name
                if not member.isfile() or not name.endswith('.cbin'):
                    continue
                stem = str(Path(name).with_suffix(''))
                row, valid = audit_recording(name, archive.extractfile(member).read(),
                    recs[stem+'.rec'], selection['bird_id'], day)
                digest = row['decoded_pcm_sha256']
                row['existing_bengalese_exact_pcm_overlap'] = digest in old_pcm
                row['prior_audit_exact_pcm_overlap'] = digest in previous_pcm
                row['duplicate_of_new_recording'] = seen.get(digest)
                row['partial_transformed_overlap'] = 'unresolved'
                seen.setdefault(digest, row['recording_id'])
                rows.append(row)
                masks[row['recording_id']] = valid
                if len(rows) % 20 == 0:
                    print('Audited', len(rows), flush=True)
    summaries = []
    for bird in [s['bird_id'] for s in plan['selected']]:
        subset = [r for r in rows if r['bird_id'] == bird]
        summaries.append({'bird_id': bird, 'recordings': len(subset),
            'documented_dates': sorted({r['documented_date'] for r in subset}),
            'date_disagreements': sum(not r['rec_date_agrees_with_archive'] for r in subset),
            'recordings_with128_valid_frame_run': sum(r['full_recording_support']['runs_at_least128_frames'] > 0 for r in subset),
            'technical_forecast_positions': sum(r['full_recording_support']['technical_forecast_positions'] for r in subset),
            'positions_in_runs_at_least128_frames': sum(r['full_recording_support']['positions_in_runs_at_least128_frames'] for r in subset),
            'existing_pcm_matches': sum(r['existing_bengalese_exact_pcm_overlap'] for r in subset),
            'prior_audit_pcm_matches': sum(r['prior_audit_exact_pcm_overlap'] for r in subset),
            'new_duplicate_recordings': sum(r['duplicate_of_new_recording'] is not None for r in subset)})
    np.savez_compressed(mask_path, **masks)
    result = {'study': 'BFSongRepo cross-individual support and evaluation-design audit',
        'plan_sha256': shared.sha(plan_path), 'code_sha256': shared.sha(Path(__file__)),
        'archive_integrity': integrity, 'prior_audits': prior, 'records': rows, 'summary': summaries,
        'mask_path': str(mask_path.relative_to(ROOT)), 'mask_sha256': shared.sha(mask_path),
        'models_fitted': False, 'partitions_assigned': False, 'stage2_ready': False,
        'certified_independent_sessions': 0,
        'prior_population_correction': 'Completed Tier A used four bird-days,274 recordings,two birds,not three days.',
        'evaluation_design': {
            'status': 'Design assessment only; no successor preregistration or allocation enacted',
            'frozen_encoder_endpoint': 'Evaluate unchanged saved encoder and matching frozen Ridge comparators on all eligible target birds; frozen checkpoint trained on or60yw70,so familiar-bird and unseen-bird results must be separate.',
            'successor_representation_endpoint': 'Any multi-bird representation training requires prospective bird-disjoint outer evaluation and inner model selection that excludes the target bird.',
            'outer_unit': 'Identified bird; recording and date are nested units,not additional animals.',
            'uncertainty': 'Report each bird paired contrast; recording bootstrap conditional on bird/day. At most three unseen birds for current encoder,insufficient for precise population-wide inference.',
            'adaptation': 'Separate future arm only. One audited day per new bird cannot establish day-disjoint adaptation/evaluation; acquire additional documented dates before such an arm.',
            'controls': 'Same targets,horizons,eligibility across Ridge and MACDE; no denoising or new objectives; masks reset per recording. No tuning against existing negative cross-bird result.',
            'exposure': 'Prior Bengalese work and existing gy6or6 evaluation prohibit calling the entire corpus untouched. Newly audited birds have not been scored in this audit.',
            'biological_claims': 'No interaction,receiver-response,semantics or independent-session claims.'}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
