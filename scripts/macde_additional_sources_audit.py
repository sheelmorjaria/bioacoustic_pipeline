#!/usr/bin/env python3
"""Audit already acquired alternative source metadata; no model fitting."""
import base64
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import h5py
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.macde_source_ids import zhao_reference
from bioacoustic_pipeline.diagnostics.macde_waveform_audit import audit_audio
from macde_source_resolution import sha, pcm_hash


def verify(folder, mapping):
    repo = json.loads((folder / 'record.json').read_text())
    result = []
    for original, local in mapping.items():
        info = next(f for f in repo['files'] if f['key'] == original)
        path = folder / local
        algorithm, expected = info['checksum'].split(':')
        if path.stat().st_size != info['size'] or sha(path, algorithm) != expected:
            raise ValueError('Repository integrity mismatch: ' + str(path))
        result.append({'local_path': str(path.relative_to(ROOT)), 'original_name': original,
                       'repository_checksum': info['checksum'], 'sha256': sha(path)})
    return result


def petruskova(xls_reader):
    sys.path.insert(0, xls_reader)
    import xlrd
    folder = ROOT / 'data/petruskova_tree_pipit'
    integrity = verify(folder, {'Tree Pipit male ID - test package.zip': 'test_package.zip',
                               'recording and male repertoires, coded songs.xls': 'repertoires.xls',
                               'Tree Pipit male ID - test solution.pdf': 'test_solution.pdf'})
    solution = {}
    for line in (folder / 'test_solution.txt').read_text().splitlines():
        cells = re.split(r'\s{2,}', line.strip())
        if len(cells) > 1 and re.fullmatch(r'\d+[a-z]?\.\d+', cells[0]):
            for number in re.findall(r'\d+', cells[1]):
                if int(number) in solution:
                    raise ValueError('Repeated blind test ID')
                solution[int(number)] = cells[0]
    if set(solution) != set(range(1, 31)):
        raise ValueError('Incomplete solution mapping')
    workbook = xlrd.open_workbook(folder / 'repertoires.xls')
    sheet = workbook.sheet_by_name('recording repertoires')
    recording_rows = []
    for i in range(1, sheet.nrows):
        row = sheet.row_values(i)
        if not row[1]:
            continue
        decoded = xlrd.xldate_as_datetime(row[3], workbook.datemode).isoformat() if isinstance(row[3], (int, float)) else str(row[3])
        recording_rows.append({'male_raw': row[0], 'recording_code': row[1], 'year_column': row[2],
                               'date_cell_raw': row[3], 'date_cell_type': sheet.cell_type(i, 3),
                               'date_excel_decoded': decoded,
                               'date_year_conflict': isinstance(row[2], (int, float)) and decoded[:4] != str(int(row[2])),
                               'ring_observation': row[5]})
    old_bytes, old_pcm = {}, {}
    for path in sorted((ROOT / 'data/alp/pipit_id').rglob('*.wav')):
        old_bytes[sha(path)] = str(path.relative_to(ROOT))
        audio, sr = sf.read(path)
        old_pcm[pcm_hash(audio, sr)] = str(path.relative_to(ROOT))
    candidates = []
    with zipfile.ZipFile(folder / 'test_package.zip') as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith('.wav'):
                continue
            number = int(Path(member).stem)
            raw = archive.read(member)
            audio, sr = sf.read(io.BytesIO(raw))
            digest = hashlib.sha256(raw).hexdigest()
            candidates.append({'test_id': number, 'archive_member': member, 'male_reference_from_solution': solution[number],
                               'sha256': digest, 'decoded_pcm_sha256': pcm_hash(audio, sr),
                               'duration_seconds': len(audio)/sr, 'acoustic_audit': audit_audio(audio, sr),
                               'exact_byte_overlap_with_stowell': old_bytes.get(digest),
                               'exact_decoded_pcm_overlap_with_stowell': old_pcm.get(pcm_hash(audio, sr)),
                               'partial_transformed_overlap': 'unresolved',
                               'original_recording_code': None, 'session': None, 'date': None,
                               'crosswalk_limit': 'Solution maps test number to male, not to a spreadsheet recording code; do not infer date from male ID',
                               'independence_status': 'unresolved', 'partition': None})
    return {'integrity': integrity, 'xls_reader_version': xlrd.__version__,
            'solution_text_sha256': sha(folder / 'test_solution.txt'),
            'records': candidates, 'original_recording_metadata': recording_rows,
            'summary': {'wav_files': len(candidates), 'male_references': len(set(solution.values())),
                        'passes_acoustic_gate': sum(r['acoustic_audit']['passes_acoustic_support'] for r in candidates),
                        'exact_overlap_files': sum(bool(r['exact_byte_overlap_with_stowell'] or r['exact_decoded_pcm_overlap_with_stowell']) for r in candidates),
                        'spreadsheet_recording_rows': len(recording_rows),
                        'date_year_conflicts': sum(r['date_year_conflict'] for r in recording_rows),
                        'certified_independent_groups': 0},
            'limitation': 'Study collected more recordings than the30 public test WAVs. Individual identities are not source-session evidence. Prior Stowell overlap remains unresolved beyond exact checks.'}


def mat_text(h, key):
    return ''.join(chr(int(c)) for c in h[key][...].flatten())


def zhao():
    folder = ROOT / 'data/zhao_marmoset'
    integrity = verify(folder, {'annotations.zip': 'annotations.zip'})
    rows = []
    with zipfile.ZipFile(folder / 'annotations.zip') as archive:
        for member in archive.namelist():
            if not member.endswith('.mat'):
                continue
            raw = archive.read(member)
            with h5py.File(io.BytesIO(raw)) as h:
                path = mat_text(h, 'audiodata/Filename')
                reference = zhao_reference(path)
                duration = float(h['audiodata/Duration'][0, 0])
                sr = float(h['audiodata/SampleRate'][0, 0])
                samples = int(h['audiodata/TotalSamples'][0, 0])
                rows.append({'annotation_member': member, 'annotation_sha256': hashlib.sha256(raw).hexdigest(),
                             'source_audio_path_from_metadata': path, **reference,
                             'source_subset': member.split('/')[0], 'reported_duration_seconds': duration,
                             'reported_sample_rate': sr, 'reported_samples': samples,
                             'reported_channels': int(h['audiodata/NumChannels'][0, 0]),
                             'duration_consistent_with_samples': abs(duration-samples/sr) < 1/sr,
                             'waveform_integrity': 'not acquired', 'acoustic_support': 'not audited',
                             'calls_structure': 'MATLAB MCOS object; not decoded by treating opaque IDs as labels or call counts',
                             'source_session_verified': False, 'partition': None})
    return {'integrity': integrity, 'records': rows, 'summary': {'annotation_files': len(rows),
             'unique_audio_references': len({r['recording_reference'] for r in rows}),
             'filename_dates': len({r['filename_timestamp'][:10] for r in rows}),
             'subject_tokens': len({s for r in rows for s in r['subject_tokens']}),
             'subsets': dict(Counter(r['source_subset'] for r in rows)), 'certified_independent_groups': 0},
            'paper': 'https://doi.org/10.3389/fncir.2026.1821810',
            'source_context': 'Paper describes parent-infant pair recordings; annotated training material also includes gene-knockdown animals. Subject roles and conditions require authoritative crosswalks, not inference from cj IDs.',
            'next_evidence': 'Match metadata source references to wave archive members, verify audio and calendar/session crosswalk, and preserve original dataset QC/training exposure; no11GB download performed.'}


def infant():
    folder = ROOT / 'data/alp/infant_marmosets_vox'
    source_manifest = json.loads((folder / 'manifest.json').read_text())
    checksum_lookup = {}
    for item in source_manifest['audio_hashes']:
        suffix = item['path'].split('/infant_marmosets_vox/')[-1]
        if suffix in checksum_lookup and checksum_lookup[suffix] != item['hash']:
            raise ValueError('Conflicting original source checksums')
        checksum_lookup[suffix] = item['hash']
    with (folder / 'labels.csv').open() as source:
        labels = list(csv.DictReader(source))
    labelled_paths = {r['path'] for r in labels}
    by_path = {}
    for r in labels:
        by_path.setdefault(r['path'], set()).add(r['callerID'])
    records = []
    for path in sorted((folder / 'data/audio_44k').rglob('*.wav')):
        relative = str(path.relative_to(folder))
        expected = checksum_lookup[relative]
        if not expected.startswith('md5Hash:'):
            raise ValueError('Unsupported source integrity scheme')
        verified = base64.b64encode(bytes.fromhex(sha(path, 'md5'))).decode() == expected.split(':', 1)[1]
        if not verified:
            raise ValueError('Infant source checksum mismatch: ' + relative)
        info = sf.info(path)
        match = re.fullmatch(r'(\d{8})_Twin(\d+)_marmoset(\d+)\.wav', path.name)
        if not match:
            raise ValueError('Unresolved infant parent filename')
        records.append({'source_recording': relative, 'repository_checksum': expected, 'verified_integrity': True,
                        'duration_seconds': info.duration, 'sample_rate': info.samplerate, 'channels': info.channels,
                        'filename_date': datetime.strptime(match[1], '%Y%m%d').date().isoformat(),
                        'filename_twin': match[2], 'filename_subject': match[3],
                        'annotation_caller_ids': sorted(by_path.get(relative, set())),
                        'resampled_16k_alias_exists': (folder / relative.replace('audio_44k', 'audio_16k')).exists(),
                        'acoustic_support': 'Not newly audited across full source; v1 checked a bounded prefix only',
                        'session_independence': 'unresolved beyond original filename/date and individual annotation',
                        'partition': None})
    return {'source_manifest_sha256': sha(folder / 'manifest.json'), 'labels_sha256': sha(folder / 'labels.csv'),
            'records': records, 'summary': {'native_parent_waveforms': len(records), 'label_rows': len(labels),
             'label_referenced_paths': len(labelled_paths), 'missing_label_referenced_waveforms': sorted(p for p in labelled_paths if not (folder / p).exists()),
             'caller_ids': len({r['callerID'] for r in labels}),
             'filename_dates': len({r['filename_date'] for r in records}),
             'duration_min_max_seconds': [min(r['duration_seconds'] for r in records), max(r['duration_seconds'] for r in records)]},
            'novelty': 'Already used in MACDE transfer v1; not an independent new corpus. Native and16k resampled copies are the same source, not extra sessions.',
            'new_protocol_requirement': 'Any full-recording window search must be specified prospectively; do not expand the closed first256-frame selection rule post hoc.'}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xls-reader', default='/tmp/macde_xls_reader')
    args = ap.parse_args()
    torch.set_num_threads(2)
    result = {'study': 'Additional MACDE source provenance audit; not a model experiment',
              'code_sha256': sha(Path(__file__)), 'petruskova': petruskova(args.xls_reader),
              'zhao': zhao(), 'infant_marmosets': infant(),
              'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False}
    path = ROOT / 'outputs/macde_additional_sources_audit.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite audit')
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k]['summary'] for k in ('petruskova', 'zhao', 'infant_marmosets')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
