#!/usr/bin/env python3
"""Single prespecified BFSongRepo bird-day TierA audit,no partitions or fitting."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.ingestion.bfsongrepo_audit import rec_metadata, decode_cbin_channel, continuity_summary
from macde_source_resolution import pcm_hash


def sha(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def members(path):
    with tarfile.open(path) as archive:
        result = {}
        for member in archive:
            if member.isfile():
                if member.name in result:
                    raise ValueError('Duplicate archive member')
                result[member.name] = archive.extractfile(member).read()
        return result


def main():
    torch.set_num_threads(2)
    folder = ROOT / 'data/bfsongrepo'
    plan_path = folder / 'single_day_acquisition_plan.json'
    plan = json.loads(plan_path.read_text())
    integrity = []
    for key, local in [('wav', 'wav_csv.tar.gz'), ('original', 'cbin_notmat.tar.gz')]:
        info = plan[key]
        path = folder / local
        if path.stat().st_size != info['size'] or sha(path, 'md5') != info['computed_md5']:
            raise ValueError('Figshare archive integrity mismatch: '+local)
        integrity.append({'local': str(path.relative_to(ROOT)), 'original_name': info['name'],
                          'file_id': info['id'], 'md5': info['computed_md5'], 'sha256': sha(path)})
    wavs, originals = members(folder / 'wav_csv.tar.gz'), members(folder / 'cbin_notmat.tar.gz')
    old_bytes, old_pcm = {}, {}
    old_count = 0
    for path in sorted((ROOT / 'data/bengalifinch').rglob('*')):
        if path.suffix.lower() != '.wav':
            continue
        old_count += 1
        old_bytes.setdefault(sha(path), []).append(str(path.relative_to(ROOT)))
        audio, sr = sf.read(path)
        old_pcm.setdefault(pcm_hash(audio, sr), []).append(str(path.relative_to(ROOT)))
    rows, masks, archive_duplicates = [], {}, {}
    for name, raw in sorted(wavs.items()):
        if not name.endswith('.wav'):
            continue
        parts = Path(name).parts
        bird, day = parts[-3:-1]
        stem = str(Path(name).with_suffix(''))
        rec = rec_metadata(originals[stem+'.rec'].decode())
        cbin = decode_cbin_channel(originals[stem+'.cbin'], rec)
        integer, sr = sf.read(io.BytesIO(raw), dtype='int16')
        alias_verified = sr == rec['sample_rate'] and integer.ndim == 1 and np.array_equal(integer, cbin)
        if not alias_verified:
            raise ValueError('Original CBIN channel0 differs from WAV: '+name)
        audio, _ = sf.read(io.BytesIO(raw))
        digest = hashlib.sha256(raw).hexdigest()
        canonical = pcm_hash(audio, sr)
        annotation = name+'.csv'
        notes = list(csv.DictReader(io.StringIO(wavs[annotation].decode()))) if annotation in wavs else []
        annotations_valid = all(0 <= float(n['onset_s']) < float(n['offset_s']) <= len(audio)/sr for n in notes)
        # Labels and intervals never enter frame extraction,validity or recording selection.
        obs = frame_observations(audio, sr)
        summary = continuity_summary(obs['time_valid'], sr)
        prefix = continuity_summary(obs['time_valid'][:256], sr)
        archive_duplicates.setdefault(canonical, []).append(name)
        rows.append({'recording_id': stem, 'bird_id': bird, 'archive_day_code': day,
                     'file_created_from_rec': rec['file_created'], 'rec_created_time_is_session_id': False,
                     'wav_member': name, 'cbin_member': stem+'.cbin', 'rec_member': stem+'.rec',
                     'wav_sha256': digest, 'decoded_pcm_sha256': canonical,
                     'rec_sha256': hashlib.sha256(originals[stem+'.rec']).hexdigest(),
                     'original_cbin_sha256': hashlib.sha256(originals[stem+'.cbin']).hexdigest(),
                     'wav_equals_original_cbin_channel0': alias_verified, 'unique_source_representations': 1,
                     'sample_rate': sr, 'samples': len(audio), 'duration_seconds': len(audio)/sr,
                     'original_channels': rec['channels'], 'sample_count_matches_rec': len(audio) == rec['samples'],
                     'annotation_available': annotation in wavs, 'annotation_rows': len(notes),
                     'annotation_times_in_waveform': annotations_valid,
                     'full_recording_support': summary, 'first256_frame_support': prefix,
                     'existing_bengalese_exact_byte_matches': old_bytes.get(digest, []),
                     'existing_bengalese_decoded_pcm_matches': old_pcm.get(canonical, []),
                     'partial_transformed_source_overlap': 'unresolved',
                     'sequence_boundary': 'Start/end of this amplitude-triggered file;never concatenate bouts',
                     'session_id': None, 'session_status': 'unresolved;bird-day and file-created timestamps do not certify sessions',
                     'independence_group': None, 'partition': None})
        masks[Path(name).name] = obs['time_valid']
        print('Audited', len(rows), name, '128run:', summary['runs_at_least128_frames'] > 0, flush=True)
    output = ROOT / 'outputs/bfsongrepo_tier_a_audit.json'
    mask_path = ROOT / 'outputs/bfsongrepo_single_day_support.npz'
    if output.exists() or mask_path.exists():
        raise RuntimeError('Refusing to overwrite single-day audit')
    np.savez_compressed(mask_path, **masks)
    durations = np.array([r['duration_seconds'] for r in rows])
    result = {'study': 'BFSongRepo single bird-day TierA audit;independent candidate namespace',
              'repository_doi': '10.6084/m9.figshare.4805749.v'+str(plan['repository_version']),
              'license': 'CC BY4.0', 'plan_sha256': sha(plan_path), 'code_sha256': sha(Path(__file__)),
              'helper_sha256': sha(ROOT / 'src/bioacoustic_pipeline/ingestion/bfsongrepo_audit.py'),
              'source_reader_reference': {'repository': 'NickleDave/evfuncs',
                                         'commit': json.loads((folder / 'evfuncs_tree.json').read_text())['sha'],
                                         'source_sha256': sha(folder / 'evfuncs_source.py'),
                                         'decode_rule': 'Inspected load_cbin:big-endian int16,interleaved channels,channel0;not executed as external code'},
              'archive_integrity': integrity, 'records': rows,
              'summary': {'unique_wav_recordings': len(rows),
                          'cbin_recordings': sum(n.endswith('.cbin') for n in originals),
                          'verified_wav_cbin_aliases': sum(r['wav_equals_original_cbin_channel0'] for r in rows),
                          'annotated_recordings': sum(r['annotation_available'] for r in rows),
                          'annotation_bounds_valid_recordings': sum(r['annotation_times_in_waveform'] for r in rows),
                          'duration_quantiles_seconds': np.quantile(durations, [0,.25,.5,.75,1]).tolist(),
                          'sample_rate_counts': dict(Counter(str(r['sample_rate']) for r in rows)),
                          'files_with128_valid_frame_run': sum(r['full_recording_support']['runs_at_least128_frames'] > 0 for r in rows),
                          'forecast_eligible_files': sum(r['full_recording_support']['technical_forecast_positions'] > 0 for r in rows),
                          'technical_forecast_positions': sum(r['full_recording_support']['technical_forecast_positions'] for r in rows),
                          'positions_in128plus_frame_runs': sum(r['full_recording_support']['positions_in_runs_at_least128_frames'] for r in rows),
                          'longest_valid_run_seconds': max(r['full_recording_support']['longest_valid_run_seconds'] for r in rows),
                          'existing_bengalese_files_compared': old_count,
                          'files_with_exact_or_pcm_overlap': sum(bool(r['existing_bengalese_exact_byte_matches'] or r['existing_bengalese_decoded_pcm_matches']) for r in rows),
                          'within_day_decoded_duplicates': [v for v in archive_duplicates.values() if len(v) > 1]},
              'support_mask': {'path': str(mask_path.relative_to(ROOT)), 'sha256': sha(mask_path)},
              'limits': ['Amplitude-triggered and sorted song-bout population;not continuous day observation',
                         '128consecutive supported frames reported separately from52past+20future forecasting eligibility;no gate relaxed',
                         'Annotations inventoried only;no unit labels used for feature extraction or support',
                         'REC file creation date/time is source metadata,not an inferred session',
                         'REC output-sound/playback configuration fields do not prove actual playback or spontaneous response rates',
                         'No exact overlap is not proof of an untouched corpus;prior Bengalese research exposure remains acknowledged',
                         'One bird-day cannot test between-day,unseen-individual or acquisition-environment generalization'],
              'certified_independent_sessions': 0, 'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
