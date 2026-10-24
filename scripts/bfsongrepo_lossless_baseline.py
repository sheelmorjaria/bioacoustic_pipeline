#!/usr/bin/env python3
"""Freeze/run a separate sample-exact mono PCM16 coding benchmark."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.ingestion.bfsongrepo_audit import rec_metadata, decode_cbin_channel
from bioacoustic_pipeline.representation.lossless_acoustic import encode, decode
from macde_source_resolution import pcm_hash

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'outputs/bfsongrepo_lossless_baseline_preregistration.json'
RESULT = ROOT/'outputs/bfsongrepo_lossless_baseline_results.json'
CODE = ROOT/'src/bioacoustic_pipeline/representation/lossless_acoustic.py'
ARCHIVES = {
    'gy6or6/032612': 'cbin_notmat.tar.gz',
    'or60yw70/092812': 'day1_original.tar.gz',
    'or60yw70/093012': 'fourth_original.tar.gz',
    'or60yw70/100112': 'day0_original.tar.gz',
    'bl26lb16/042112': 'bl26lb16_support_original.tar.gz',
    'gr41rd51/062312': 'gr41rd51_support_original.tar.gz',
}


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def freeze():
    if PLAN.exists() or RESULT.exists():
        raise RuntimeError('Refusing to overwrite frozen study')
    records, audits = {}, []
    for name, key in [('bfsongrepo_tier_a_audit','records'),
                      ('bfsongrepo_multiday_audit','new_records'),
                      ('bfsongrepo_fourth_day_audit','records'),
                      ('bfsongrepo_cross_individual_support_audit','records')]:
        path = ROOT/'outputs'/f'{name}.json'
        audits.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        for row in json.loads(path.read_text())[key]:
            recording = row['recording_id']
            if recording in records:
                raise ValueError('Repeated source recording in audit population')
            if row['full_recording_support']['runs_at_least128_frames'] < 1:
                raise ValueError('Previously audited support gate failed')
            records[recording] = {'recording_id': recording, 'decoded_pcm_sha256': row['decoded_pcm_sha256']}
    if len(records) != 552:
        raise ValueError('Unexpected audited population')
    archives = []
    for group, local in ARCHIVES.items():
        path = ROOT/'data/bfsongrepo'/local
        archives.append({'bird_day': group, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
    payload = {
        'study': 'BFSongRepo exact acoustic coding baseline v1',
        'scope': 'Information preservation only; not learned representation,forecasting,behaviour or semantics.',
        'selection': 'All552 recordings from six completed acoustic source audits,four birds,six days. No codec outcomes used for selection.',
        'source_audio': 'Native channel0 mono signed PCM16 at REC sample rate,entire recording including silence and unsupported MACDE frames. No resampling,gain change,denoising or concatenation.',
        'codecs': ['flac','delta_zlib'],
        'codec_parameters': {'flac': 'libsndfile FLAC PCM16 compression_level=1.0',
                             'delta_zlib': 'v1 modular signed16 first differences,zigzag uint16,zlib level9,48-byte header with rate,count,PCM SHA256'},
        'primary_technical_gate': 'Every decoded sample and sample rate must exactly equal native source; mismatches abort study.',
        'rate_metrics': 'Complete encoded bytes including headers / raw channel0 PCM bytes; bits/sample and bits/second. No original stereo CBIN or TAR size in denominator.',
        'aggregation': 'Pooled byte-weighted totals plus bird-day totals; paired per-record sizes retained. No population inference or model selection.',
        'records': [records[k] for k in sorted(records)], 'audits': audits, 'archives': archives,
        'dependencies': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in [
            Path(__file__).resolve(), CODE, ROOT/'src/bioacoustic_pipeline/ingestion/bfsongrepo_audit.py',
            ROOT/'scripts/macde_source_resolution.py']],
        'software': {'numpy': np.__version__, 'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__},
        'stage2_ready': False, 'models_fitted': False,
        'previous_results': {'path':'outputs/bfsongrepo_development_results.json',
                             'sha256':sha(ROOT/'outputs/bfsongrepo_development_results.json')},
    }
    PLAN.write_text(json.dumps(payload,indent=2)+'\n')
    print('Frozen',len(records),'recordings;',len(archives),'bird-days',flush=True)


def run():
    if RESULT.exists():
        raise RuntimeError('Refusing to overwrite codec results')
    plan = json.loads(PLAN.read_text())
    for item in plan['dependencies']+plan['audits']+plan['archives']+[plan['previous_results']]:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen dependency/source changed: '+item['path'])
    if plan['software'] != {'numpy':np.__version__,'soundfile':sf.__version__,'libsndfile':sf.__libsndfile_version__}:
        raise ValueError('Software changed since freeze')
    expected = {r['recording_id']:r for r in plan['records']}
    rows, seen = [], set()
    for source in plan['archives']:
        path = ROOT/source['path']
        metadata, names = {}, set()
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name in names:
                    raise ValueError('Duplicate archive member')
                names.add(member.name)
                if member.name.endswith('.rec'):
                    metadata[member.name] = rec_metadata(archive.extractfile(member).read().decode())
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.cbin'):
                    continue
                recording = str(Path(member.name).with_suffix(''))
                if recording not in expected or recording in seen:
                    raise ValueError('Unexpected or repeated source recording')
                seen.add(recording)
                rec = metadata[recording+'.rec']
                samples = decode_cbin_channel(archive.extractfile(member).read(),rec).astype('<i2')
                rate = rec['sample_rate']
                if pcm_hash(samples.astype(float)/32768.,rate) != expected[recording]['decoded_pcm_sha256']:
                    raise ValueError('Source PCM differs from completed audit')
                pcm = samples.tobytes()
                row = {'recording_id':recording,'bird_day':source['bird_day'],
                       'sample_rate':rate,'samples':len(samples),'duration_seconds':len(samples)/rate,
                       'raw_pcm_bytes':len(pcm),'native_pcm_sha256':hashlib.sha256(pcm).hexdigest(),'codecs':{}}
                for codec in plan['codecs']:
                    blob = encode(samples,rate,codec)
                    restored, restored_rate = decode(blob,codec)
                    if restored_rate != rate or restored.astype('<i2').tobytes() != pcm:
                        raise ValueError('Exact reconstruction gate failed')
                    row['codecs'][codec] = {'encoded_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
                        'sample_exact':True,'rate_exact':True,'encoded_over_raw':len(blob)/len(pcm),
                        'bits_per_sample':8*len(blob)/len(samples),'bits_per_second':8*len(blob)/(len(samples)/rate)}
                rows.append(row)
                if len(rows)%50 == 0:
                    print('Exact roundtrips verified',len(rows),flush=True)
    if seen != set(expected):
        raise ValueError('Missing frozen source recordings')
    groups = defaultdict(list)
    for row in rows:
        groups[row['bird_day']].append(row)
    def aggregate(group):
        raw = sum(r['raw_pcm_bytes'] for r in group)
        samples = sum(r['samples'] for r in group)
        seconds = sum(r['duration_seconds'] for r in group)
        summary = {'recordings':len(group),'raw_pcm_bytes':raw,'duration_seconds':seconds,'codecs':{}}
        for codec in plan['codecs']:
            count = sum(r['codecs'][codec]['encoded_bytes'] for r in group)
            summary['codecs'][codec] = {'encoded_bytes':count,'encoded_over_raw':count/raw,
                'savings_fraction':1-count/raw,'bits_per_sample':count*8/samples,'bits_per_second':count*8/seconds,
                'exact_roundtrips':sum(r['codecs'][codec]['sample_exact'] for r in group)}
        return summary
    for item in [plan['previous_results']]:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Closed study changed during benchmark')
    result = {'study':plan['study'],'preregistration_sha256':sha(PLAN),'summary':aggregate(rows),
              'by_bird_day':{k:aggregate(v) for k,v in sorted(groups.items())},
              'records':sorted(rows,key=lambda r:r['recording_id']),
              'all_exact':True,'stage2_ready':False,'models_fitted':False,
              'interpretation':'Sample-exact mono PCM preservation and storage rates only. Not learned acoustic detail,semantic understanding,behavioural prediction or transfer.',
              'original_closed_results_unchanged':True}
    RESULT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['freeze','run'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
