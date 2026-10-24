#!/usr/bin/env python3
"""Metadata-first MACDE acquisition audit; never fits or modifies frozen studies."""
import csv
from collections import Counter, defaultdict
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.embeddings.macde import eligible_mask


def digest(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def verify(folder):
    record = json.loads((folder / 'record.json').read_text())
    files = []
    for item in record['files']:
        path = folder / item['key']
        if not path.exists():
            continue
        algorithm, expected = item['checksum'].split(':')
        if path.stat().st_size != item['size'] or digest(path, algorithm) != expected:
            raise ValueError(f'Checksum/size mismatch: {path}')
        files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                      'repository_checksum': item['checksum'], 'sha256': digest(path),
                      'repository_manifest_sha256': digest(folder / 'record.json'),
                      'source_url': item['links']['self'], 'license': record['metadata'].get('license')})
    return files


def main():
    torch.set_num_threads(2)
    files = verify(ROOT / 'data/marmaudio') + verify(ROOT / 'data/ceb')
    examples = {}
    with zipfile.ZipFile(ROOT / 'data/marmaudio/Audio_Examples.zip') as archive:
        for name in archive.namelist():
            if '/audios/' not in name or not name.endswith('.wav'):
                continue
            examples[Path(name).stem] = {'archive_member': name, 'metadata_matches': []}
    counts = Counter()
    dates, folders = Counter(), set()
    sufficient_by_date = Counter()
    with (ROOT / 'data/marmaudio/Annotations.tsv').open() as source:
        reader = csv.DictReader(source, delimiter='\t')
        columns = reader.fieldnames
        for row in reader:
            counts['rows'] += 1
            folders.add(row['parent_name'])
            try:
                group = date(int(row['year']), int(row['month']), int(row['day'])).isoformat()
            except (ValueError, TypeError):
                counts['invalid_calendar_date'] += 1
                group = None
            if group:
                dates[group] += 1
            duration = float(row['duration'])
            if duration >= 1.31:
                counts['duration_at_least_128_frames'] += 1
                if group:
                    sufficient_by_date[group] += 1
            if duration >= .75:
                counts['duration_at_least_one_eligible_span_75_frames'] += 1
            stem = Path(row['file_name']).stem
            if stem in examples:
                examples[stem]['metadata_matches'].append({
                    'parent_name': row['parent_name'], 'file_name': row['file_name'],
                    'date': group, 'duration': duration})
    old_hashes = set()
    for name in ('macde_stage1_source_audit.json', 'macde_transfer_source_audit.json'):
        manifest = json.loads((ROOT / 'outputs' / name).read_text())
        old_hashes.update(v['sha256'] for v in manifest['items'])
    sample_counts = Counter()
    with zipfile.ZipFile(ROOT / 'data/marmaudio/Audio_Examples.zip') as archive:
        for stem, item in examples.items():
            raw = archive.read(item['archive_member'])
            item['sha256'] = hashlib.sha256(raw).hexdigest()
            item['exact_byte_overlap_with_macde_audits'] = item['sha256'] in old_hashes
            audio, sr = sf.read(io.BytesIO(raw))
            if audio.ndim > 1:
                audio = audio.mean(1)
            obs = frame_observations(audio, sr, max_frames=256)
            keep = eligible_mask(torch.from_numpy(obs['time_valid'])[None])[0].numpy()
            item.update({'sample_rate': sr, 'frames': len(keep), 'eligible_positions': int(keep.sum()),
                         'metadata_link_unique': len(item['metadata_matches']) == 1})
            sample_counts['files'] += 1
            if keep.any():
                sample_counts['files_with_acoustic_support'] += 1
                if item['metadata_link_unique'] and item['metadata_matches'][0]['date']:
                    sample_counts['supported_and_uniquely_date_linked_files'] += 1
    tree = defaultdict(list)
    with (ROOT / 'data/ceb/train_xenocanto.csv').open() as source:
        for row in csv.DictReader(source):
            if 'Anthus trivialis' in row['scientific_name'].split('|'):
                tree[row['filepath']].append(row)
    candidates = []
    for path, rows in sorted(tree.items()):
        row = rows[0]
        candidates.append({'filepath': path, 'xc_id': row['xc_id'], 'recordist': row['xc_recordist'],
                           'source_url': row['xc_url'], 'license': row['license'],
                           'annotation_rows': len(rows), 'date': None, 'parent_session': None,
                           'coordinates': [row['lat'] or None, row['long'] or None],
                           'group_status': 'unresolved; XC ID alone does not establish session independence',
                           'deduplication_status': 'XC ID retained; original pipit corpus has no XC crosswalk; audio not downloaded'})
    names = json.loads((ROOT / 'data/ceb/scientific_name_counts.json').read_text())
    result = {
        'purpose': 'acquisition audit only; neither patches frozen v1 nor assigns candidates to its deficits',
        'code_sha256': digest(Path(__file__)), 'downloads': files,
        'pipit_taxon': {'common_name': 'Tree pipit', 'scientific_name': 'Anthus trivialis',
                        'source': 'https://zenodo.org/records/1413495',
                        'local_readme_sha256': digest(ROOT / 'data/alp/pipit_id/source_README.txt')},
        'marmaudio': {'record': '15017207', 'columns': columns, 'counts': dict(counts),
                     'calendar_dates': len(dates), 'archive_folders': len(folders),
                     'dates_with_128_frame_duration_candidates': len(sufficient_by_date),
                     'date_counts': dict(sorted(dates.items())), 'example_summary': dict(sample_counts),
                     'examples': examples,
                     'supported_uniquely_linked_example_dates': sorted({v['metadata_matches'][0]['date'] for v in examples.values() if v['eligible_positions'] and v['metadata_link_unique'] and v['metadata_matches'][0]['date']}),
                     'caller_identity': 'unavailable in annotation table',
                     'parent_session_identity': 'unavailable; parent_name is an archive folder, not a verified session',
                     'limits': 'Duration counts are metadata candidates, not acoustic eligibility. Examples are selected illustrations. No session-independence or transcoding-resistant overlap certification.'},
        'ceb_tree_pipit': {'species_counts': names['scientific_name_counts'].get('Anthus trivialis'),
                           'focal_files': len(candidates), 'annotation_rows': sum(v['annotation_rows'] for v in candidates),
                           'unique_xc_ids': len({v['xc_id'] for v in candidates}),
                           'recordists': len({v['recordist'] for v in candidates}), 'candidates': candidates},
        'xeno_canto': {'species_page': 'https://xeno-canto.org/species/Anthus-trivialis',
                      'direct_access': 'Access Denied via browser tool; CEB provenance used as independent acquisition index, not verified session data'},
        'certified_additional_validation_groups': 0, 'certified_additional_heldout_groups': 0,
        'models_fitted': False, 'stage2_ready': False,
        'next_requirement': 'New prospective source/grouping plan; session/date provenance and waveform eligibility required before any fitting',
    }
    output = ROOT / 'outputs/macde_acquisition_audit.json'
    if output.exists():
        raise RuntimeError('Refusing to replace acquisition audit')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'marmaudio_counts': dict(counts),
                      'marmaudio_dates': len(dates), 'example_support': dict(sample_counts),
                      'CEB_tree_pipit_files': len(candidates)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
