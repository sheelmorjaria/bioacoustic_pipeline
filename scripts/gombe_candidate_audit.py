#!/usr/bin/env python3
"""Source-level Gombe audit, separate from all closed MACDE experiments."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.ingestion.gombe_audit import catalogue_rows, support_summary


def sha(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xls-reader', default='/tmp/macde_xls_reader')
    args = ap.parse_args()
    sys.path.insert(0, args.xls_reader)
    import xlrd
    torch.set_num_threads(2)
    folder = ROOT / 'data/gombe'
    repository = json.loads((folder / 'mirror_record.json').read_text())
    mapping = {'AdultDirSounds11Dec14Final.xls': 'catalogue.xls',
               'Dryad Readme file adults.docx': 'README.docx', 'Adults 3.wav': 'Adults_3.wav'}
    files = []
    for original, local in mapping.items():
        info = next(f for f in repository['files'] if f['key'] == original)
        path = folder / local
        algorithm, expected = info['checksum'].split(':')
        if path.stat().st_size != info['size'] or sha(path, algorithm) != expected:
            raise ValueError('Repository checksum/size mismatch: ' + local)
        files.append({'path': str(path.relative_to(ROOT)), 'original_name': original,
                      'repository_checksum': info['checksum'], 'sha256': sha(path), 'bytes': info['size']})
    book = xlrd.open_workbook(folder / 'catalogue.xls')
    sheet = book.sheet_by_name('Sheet1')
    raw_rows = catalogue_rows([sheet.row_values(i) for i in range(1, sheet.nrows)])
    headers = sheet.row_values(0)
    rows = []
    calendar_dates, invalid_dates = set(), []
    for raw in raw_rows:
        item = dict(zip(headers, raw))
        item['catalogue_id'] = str(int(raw[0]))
        try:
            stamp = date(int(raw[8]), int(raw[6]), int(raw[7])).isoformat()
            calendar_dates.add(stamp)
        except (ValueError, TypeError):
            stamp = None
            invalid_dates.append(item['catalogue_id'])
        item['parsed_date'] = stamp
        item['parent_field_recording_id'] = None
        item['verified_session_id'] = None
        item['compilation_sample_start'] = None
        item['compilation_sample_end'] = None
        item['partition'] = None
        rows.append(item)
    markers = []
    for row in rows:
        text = ' '.join(str(row.get(key, '')) for key in ('Individual(s) with Sound/call type', 'Context of vocalizations'))
        if re.search(r'adult\s+tape\s+[IVX\d]+\s+starts', text, re.I):
            markers.append({'catalogue_id': row['catalogue_id'], 'recordist_record_number': row['Recordist record number'], 'text': text})
    with zipfile.ZipFile(folder / 'README.docx') as archive:
        doc = ET.fromstring(archive.read('word/document.xml'))
        readme = ' '.join(doc.itertext())
    path = folder / 'Adults_3.wav'
    audio_info = sf.info(path)
    sr = audio_info.samplerate
    hop = round(sr*.01)
    total_frames = max(0, 1+(audio_info.frames-4*hop)//hop)
    validity = []
    for start in range(0, total_frames, 2048):
        count = min(2048, total_frames-start)
        audio, _ = sf.read(path, start=start*hop, frames=(count+3)*hop)
        if audio.ndim > 1:
            audio = audio.mean(1)
        obs = frame_observations(audio, sr, max_frames=count)
        if len(obs['time_valid']) != count:
            raise RuntimeError('Chunk framing invariant failed')
        validity.append(obs['time_valid'])
        if start % 16384 == 0:
            print('Audited frames', start+count, '/', total_frames, flush=True)
    valid = np.concatenate(validity)
    mask_path = ROOT / 'outputs/gombe_Adults3_technical_support.npz'
    if mask_path.exists():
        raise RuntimeError('Refusing to replace support artifact')
    np.savez_compressed(mask_path, technical_valid=valid)
    counts = {field: dict(Counter(str(row.get(field, '')) for row in rows)) for field in
              ('Recording Device', 'Focal indiv', 'Microphone', 'Recorder', 'Tape Speed')}
    ids = [r['catalogue_id'] for r in rows]
    result = {'study': 'Gombe-specific candidate source and acoustic-support audit; no experiment reopening',
              'code_sha256': sha(Path(__file__)),
              'dependency_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                                   [ROOT / 'src/bioacoustic_pipeline/ingestion/gombe_audit.py',
                                    ROOT / 'src/bioacoustic_pipeline/diagnostics/macde_data.py',
                                    ROOT / 'src/bioacoustic_pipeline/embeddings/macde.py']},
              'files': files, 'repository': 'https://zenodo.org/records/4979944',
              'catalogue': {'active_sheet': sheet.name, 'rows': len(rows), 'unique_catalogue_ids': len(set(ids)),
                            'duplicate_catalogue_ids': [k for k, n in Counter(ids).items() if n > 1],
                            'published_cut_count': 605, 'count_discrepancy_unresolved': len(rows) != 605,
                            'distinct_valid_dates': len(calendar_dates), 'invalid_date_catalogue_ids': invalid_dates,
                            'field_value_counts': counts, 'explicit_per_cut_recordist_column': False,
                            'raw_vocalizer_field_values': sorted({str(r['Individuals Vocalizing']) for r in rows}),
                            'named_focal_individual_count': None,
                            'focal_caveat': 'Focal indiv is predominantly code A, not a verified named focal individual; vocalizer strings are separate and may contain aliases/uncertainty/non-chimp sources',
                            'removed_sheet_rows': book.sheet_by_name('removed 7Oct2014').nrows-1,
                            'removed_sheet_policy': 'Not merged with active catalogue',
                            'compilation_start_markers': markers, 'records': rows},
              'waveform': {'path': str(path.relative_to(ROOT)), 'sample_rate': sr, 'channels': audio_info.channels,
                           'duration_seconds': audio_info.duration, 'subtype': audio_info.subtype,
                           'audit_population': 'All frames of one acquired unparsed compilation; exploratory technical support, not a frozen model population',
                           'support': support_summary(valid, sr),
                           'support_mask': {'path': str(mask_path.relative_to(ROOT)), 'sha256': sha(mask_path)},
                           'human_announcement_mask': 'unavailable', 'chimpanzee_vocal_source_mask': 'unavailable',
                           'cut_boundary_mask': 'unavailable'},
              'source_lineage': {'documented_unparsed_compilations': 7,
                                 'macaulay_ids_are': 'catalogued audio specimens, not verified original field-session IDs',
                                 'cut_to_compilation_sample_coordinates': 'not recovered',
                                 'parent_field_recording_count': None,
                                 'eligible_parent_recording_count': None,
                                 'readme_text_sha256': hashlib.sha256(readme.encode()).hexdigest()},
              'decision': 'Resolvable candidate: catalogue cut-to-audio-coordinate and original-session mapping required before grouping; no session certification',
              'reopening_requirements': ['Resolve606-versus605 catalogue discrepancy',
                                        'Map each acquired cut to compilation/parent and exact sample offsets; mask edit boundaries',
                                        'Obtain documented session relationships; retain repeated subjects within intended grouping claim',
                                        'Validate chimpanzee-source mask or explicitly preregister an acoustically mixed population'],
              'certified_independent_groups': 0, 'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False}
    out = ROOT / 'outputs/gombe_candidate_audit.json'
    if out.exists():
        raise RuntimeError('Refusing to replace candidate audit')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'catalogue_rows': len(rows), 'distinct_dates': len(calendar_dates),
                      'waveform_support': result['waveform']['support'], 'certified_groups': 0}, indent=2), flush=True)


if __name__ == '__main__':
    main()
