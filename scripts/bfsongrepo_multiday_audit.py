#!/usr/bin/env python3
"""Bounded, prospectively selected BFSongRepo multi-day acoustic/provenance audit."""
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

import bfsongrepo_tier_a_audit as single
from macde_source_resolution import pcm_hash
ROOT = single.ROOT
sys.path.insert(0, str(ROOT/'src'))
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import rec_metadata, decode_cbin_channel, continuity_summary


def main():
    torch.set_num_threads(2)
    folder = ROOT/'data/bfsongrepo'
    plan_path = folder/'multiday_acquisition_plan.json'
    plan = json.loads(plan_path.read_text())
    prior_path = ROOT/'outputs/bfsongrepo_tier_a_audit.json'
    prior = json.loads(prior_path.read_text())
    frozen = json.loads((ROOT/'outputs/macde_stage1_source_audit.json').read_text())
    dependencies = {p: single.sha(ROOT/p) for p in frozen['code_provenance'] if p.startswith('src/')}
    if any(dependencies[p] != frozen['code_provenance'][p] for p in dependencies):
        raise ValueError('Frozen MACDE dependencies changed')
    old_bytes, old_pcm = {}, {}
    for path in sorted((ROOT/'data/bengalifinch').rglob('*')):
        if path.suffix.lower() == '.wav':
            old_bytes.setdefault(single.sha(path), []).append(str(path.relative_to(ROOT)))
            audio, sr = sf.read(path)
            old_pcm.setdefault(pcm_hash(audio, sr), []).append(str(path.relative_to(ROOT)))
    rows = []
    integrity = []
    masks = {}
    for index, pair in enumerate(plan['selected']):
        for key, suffix in [('wav','wav'),('original','original')]:
            info = pair[key]
            path = folder/f'day{index}_{suffix}.tar.gz'
            if path.stat().st_size != info['size'] or single.sha(path, 'md5') != info['computed_md5']:
                raise ValueError('Archive integrity failure: '+str(path))
            integrity.append({'original_name': info['name'], 'local_path': str(path.relative_to(ROOT)),
                              'file_id': info['id'], 'md5': info['computed_md5'], 'sha256': single.sha(path)})
        waves = single.members(folder/f'day{index}_wav.tar.gz')
        originals = single.members(folder/f'day{index}_original.tar.gz')
        count = 0
        for name, raw in sorted(waves.items()):
            if not name.endswith('.wav'):
                continue
            bird, day = Path(name).parts[-3:-1]
            stem = str(Path(name).with_suffix(''))
            rec = rec_metadata(originals[stem+'.rec'].decode())
            original = decode_cbin_channel(originals[stem+'.cbin'], rec)
            integer, sr = sf.read(io.BytesIO(raw), dtype='int16')
            identical = integer.ndim == 1 and sr == rec['sample_rate'] and np.array_equal(integer, original)
            if not identical:
                raise ValueError('WAV-original alias failure: '+name)
            audio, _ = sf.read(io.BytesIO(raw))
            obs = frame_observations(audio, sr)
            summary = continuity_summary(obs['time_valid'], sr)
            ann = name+'.csv'
            notes = list(csv.DictReader(io.StringIO(waves[ann].decode()))) if ann in waves else []
            byte_hash = hashlib.sha256(raw).hexdigest()
            canonical = pcm_hash(audio, sr)
            archive_date = datetime.strptime(day, '%m%d%y').date().isoformat()
            rows.append({'recording_id': stem, 'bird_id': bird, 'archive_day_code': day,
                         'archive_date': archive_date, 'file_created_from_rec': rec['file_created'],
                         'rec_date_agrees_with_archive': rec['file_created'][:10] == archive_date if rec['file_created'] else None,
                         'wav_member': name, 'wav_sha256': byte_hash, 'decoded_pcm_sha256': canonical,
                         'cbin_member': stem+'.cbin', 'original_cbin_sha256': hashlib.sha256(originals[stem+'.cbin']).hexdigest(),
                         'rec_sha256': hashlib.sha256(originals[stem+'.rec']).hexdigest(),
                         'wav_equals_original_cbin_channel0': identical,
                         'duration_seconds': len(audio)/sr, 'sample_rate': sr, 'samples': len(audio),
                         'annotation_available': ann in waves, 'annotation_rows': len(notes),
                         'annotation_bounds_valid': all(0 <= float(n['onset_s']) < float(n['offset_s']) <= len(audio)/sr for n in notes),
                         'full_recording_support': summary, 'first256_frame_support': continuity_summary(obs['time_valid'][:256], sr),
                         'existing_bengalese_exact_byte_matches': old_bytes.get(byte_hash, []),
                         'existing_bengalese_decoded_pcm_matches': old_pcm.get(canonical, []),
                         'partial_transformed_overlap': 'unresolved',
                         'session_id': None, 'session_evidence': 'REC records file creation/acquisition parameters,not verified session boundaries',
                         'independence_group': None, 'partition': None})
            masks[Path(name).name] = obs['time_valid']
            count += 1
            if count % 10 == 0:
                print(bird, day, 'audited', count, flush=True)
    combined = prior['records'] + rows
    by_day = {}
    for row in combined:
        by_day.setdefault(row['bird_id']+':'+row['archive_day_code'], []).append(row)
    day_summary = {}
    for key, values in by_day.items():
        day_summary[key] = {'recordings': len(values),
                            'files_with128_valid_frame_run': sum(v['full_recording_support']['runs_at_least128_frames'] > 0 for v in values),
                            'forecast_positions': sum(v['full_recording_support']['technical_forecast_positions'] for v in values),
                            'positions_in128plus_runs': sum(v['full_recording_support']['positions_in_runs_at_least128_frames'] for v in values),
                            'aliases_verified': sum(v['wav_equals_original_cbin_channel0'] for v in values)}
    canonical_groups = {}
    for row in combined:
        canonical_groups.setdefault(row['decoded_pcm_sha256'], []).append(row['recording_id'])
    repository = json.loads((folder/'record.json').read_text())
    inventory = []
    for info in repository['files']:
        if info['name'].endswith('.wav.csv.tar.gz'):
            tokens = info['name'].split('.')
            bird, day = tokens[2:4]
            inventory.append({'bird_id': bird, 'archive_day_code': day,
                              'archive_date': datetime.strptime(day, '%m%d%y').date().isoformat(),
                              'file_id': info['id'], 'archive': info['name'], 'bytes': info['size'],
                              'audited': bird+':'+day in day_summary,
                              'certified_session_id': None})
    mask_path = ROOT/'outputs/bfsongrepo_multiday_support.npz'
    out = ROOT/'outputs/bfsongrepo_multiday_audit.json'
    if out.exists() or mask_path.exists():
        raise RuntimeError('Refusing to overwrite multi-day audit')
    np.savez_compressed(mask_path, **masks)
    result = {'study': 'BFSongRepo bounded multi-day source-independence audit;no model experiment',
              'code_sha256': single.sha(Path(__file__)), 'plan_sha256': single.sha(plan_path),
              'prior_audit_sha256': single.sha(prior_path), 'frozen_dependency_hashes': dependencies,
              'summary': {'audited_bird_days': len(day_summary), 'birds': len({r['bird_id'] for r in combined}),
                          'total_recordings': len(combined), 'new_recordings': len(rows),
                          'files_with128_valid_frame_run': sum(r['full_recording_support']['runs_at_least128_frames'] > 0 for r in combined),
                          'forecast_positions': sum(r['full_recording_support']['technical_forecast_positions'] for r in combined),
                          'new_rec_date_disagreements': sum(r['rec_date_agrees_with_archive'] is False for r in rows),
                          'new_files_with_exact_or_pcm_overlap': sum(bool(r['existing_bengalese_exact_byte_matches'] or r['existing_bengalese_decoded_pcm_matches']) for r in rows),
                          'source_inventory_bird_days': len(inventory), 'source_inventory_birds': len({r['bird_id'] for r in inventory})},
              'by_bird_day': day_summary, 'source_inventory': inventory, 'archive_integrity': integrity,
              'new_records': rows, 'decoded_duplicate_groups_across_audited_days': [v for v in canonical_groups.values() if len(v) > 1],
              'support_mask': {'path': str(mask_path.relative_to(ROOT)), 'sha256': single.sha(mask_path)},
              'prior_exposure': 'Programme previously used Bengalese corpora and syllable references;negative exact overlap does not establish untouched confirmatory material',
              'scope_limits': ['Triggered/sorted bouts;recording histories reset at every file and unsupported span',
                               'Bird-day identity and REC file creation time are verified provenance attributes,not certified session identifiers',
                               'Three size-selected containers from two birds are not representative of all18 bird-days or four birds',
                               'No source-specific syllable label influenced feature extraction,validity or inclusion',
                               'Whole-recording audit is separate from prior closed fixed-window model populations'],
              'tier_a': 'Support counts reported per file/day;no universal pass assumed',
              'tier_b': 'Unresolved:explicit session/group contract still required;no independent-session certification',
              'certified_independent_sessions': 0, 'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summary': result['summary'], 'by_day': day_summary}, indent=2), flush=True)


if __name__ == '__main__':
    main()
