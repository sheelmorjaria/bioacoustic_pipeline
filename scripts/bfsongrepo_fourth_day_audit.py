#!/usr/bin/env python3
"""Verify the fourth BFSongRepo day before assigning study partitions."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import bfsongrepo_tier_a_audit as shared
from macde_source_resolution import pcm_hash
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import rec_metadata, decode_cbin_channel, continuity_summary

ROOT = shared.ROOT


def main():
    torch.set_num_threads(2)
    folder = ROOT/'data/bfsongrepo'
    plan_path = folder/'fourth_day_acquisition_plan.json'
    plan = json.loads(plan_path.read_text())
    integrity = []
    for key, local in [('wav','fourth_wav.tar.gz'), ('original','fourth_original.tar.gz')]:
        path, info = folder/local, plan[key]
        if path.stat().st_size != info['size'] or shared.sha(path, 'md5') != info['computed_md5']:
            raise ValueError('Fourth-day source integrity failure')
        integrity.append({'local_path': str(path.relative_to(ROOT)), 'original_name': info['name'],
                          'sha256': shared.sha(path), 'md5': info['computed_md5']})
    waves = shared.members(folder/'fourth_wav.tar.gz')
    originals = shared.members(folder/'fourth_original.tar.gz')
    old_hashes = set()
    for path in (ROOT/'data/bengalifinch').rglob('*'):
        if path.suffix.lower() == '.wav':
            audio, sr = sf.read(path)
            old_hashes.add(pcm_hash(audio, sr))
    rows, masks = [], {}
    for name, raw in sorted(waves.items()):
        if not name.endswith('.wav'):
            continue
        stem = str(Path(name).with_suffix(''))
        bird, day = Path(name).parts[-3:-1]
        rec = rec_metadata(originals[stem+'.rec'].decode())
        original = decode_cbin_channel(originals[stem+'.cbin'], rec)
        integers, sr = sf.read(io.BytesIO(raw), dtype='int16')
        if sr != rec['sample_rate'] or integers.ndim != 1 or not np.array_equal(integers, original):
            raise ValueError('WAV-CBIN alias differs')
        audio, _ = sf.read(io.BytesIO(raw))
        valid = frame_observations(audio, sr)['time_valid']
        annotation = name+'.csv'
        notes = list(csv.DictReader(io.StringIO(waves[annotation].decode()))) if annotation in waves else []
        canonical = pcm_hash(audio, sr)
        rows.append({'recording_id': stem, 'bird_id': bird, 'archive_day_code': day,
                     'file_created_from_rec': rec['file_created'], 'sample_rate': sr,
                     'wav_member': name, 'wav_sha256': hashlib.sha256(raw).hexdigest(),
                     'decoded_pcm_sha256': canonical, 'duration_seconds': len(audio)/sr,
                     'wav_equals_original_cbin_channel0': True,
                     'rec_created_date_agrees_with_archive': rec['file_created'][:10] == '2012-09-30',
                     'full_recording_support': continuity_summary(valid, sr),
                     'annotation_available': annotation in waves,
                     'annotation_bounds_valid': all(0 <= float(n['onset_s']) < float(n['offset_s']) <= len(audio)/sr for n in notes),
                     'existing_bengalese_decoded_pcm_overlap': canonical in old_hashes,
                     'partial_transformed_overlap': 'unresolved', 'session_id': None,
                     'partition': None})
        masks[Path(name).name] = valid
        if len(rows) % 20 == 0:
            print('Fourth day audited', len(rows), flush=True)
    out = ROOT/'outputs/bfsongrepo_fourth_day_audit.json'
    mask_path = ROOT/'outputs/bfsongrepo_fourth_day_support.npz'
    if out.exists() or mask_path.exists():
        raise RuntimeError('Refusing to overwrite fourth-day audit')
    np.savez_compressed(mask_path, **masks)
    summary = {'recordings': len(rows), 'files_with128_valid_frame_run': sum(r['full_recording_support']['runs_at_least128_frames'] > 0 for r in rows),
               'forecast_positions': sum(r['full_recording_support']['technical_forecast_positions'] for r in rows),
               'date_disagreements': sum(not r['rec_created_date_agrees_with_archive'] for r in rows),
               'decoded_overlap_files': sum(r['existing_bengalese_decoded_pcm_overlap'] for r in rows)}
    result = {'study': 'Fourth bird-day source audit,prior population unchanged',
              'plan_sha256': shared.sha(plan_path), 'code_sha256': shared.sha(Path(__file__)),
              'archive_integrity': integrity, 'records': rows, 'summary': summary,
              'support_mask': {'path': str(mask_path.relative_to(ROOT)), 'sha256': shared.sha(mask_path)},
              'certified_independent_sessions': 0, 'partitions_assigned': False, 'models_fitted': False,
              'scope': 'Documented or60yw70 30Sep2012 acquisition files;additional date does not certify session independence'}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
