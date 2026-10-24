#!/usr/bin/env python3
"""Resolve fixed candidate source evidence and audit waveforms without fitting."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.diagnostics.macde_waveform_audit import audit_audio


def sha(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def pcm_hash(audio, sr):
    # Lossless decoded equivalence only, not robust to compression, trimming or resampling.
    payload = np.ascontiguousarray(audio, dtype='<f4')
    header = json.dumps({'sample_rate': sr, 'shape': list(payload.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + payload.tobytes()).hexdigest()


def marm_audio():
    manifest_path = ROOT / 'outputs/macde_marmaudio_provenance_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    publication = ROOT / 'data/marmaudio/publication.xml'
    root = ET.parse(publication).getroot()
    paragraphs = [''.join(p.itertext()) for p in root.iter('p')]
    equipment_evidence = next(p for p in paragraphs if 'C-100' in p and 'Fireface' in p)
    code_path = ROOT / 'data/marmaudio/source_segment_filtered_predictions.py'
    code = code_path.read_text()
    fields = ('raw_recording_id', 'raw_recording_onset', 'raw_recording_offset')
    if not all(f"'{key}':" in code for key in fields):
        raise ValueError('Expected raw-recording sidecar fields absent from pinned source')
    acquisition = json.loads((ROOT / 'outputs/macde_acquisition_audit.json').read_text())
    released = acquisition['marmaudio']['columns']
    records = []
    with zipfile.ZipFile(ROOT / 'data/marmaudio/Audio_Examples.zip') as archive:
        for item in manifest['records']:
            if not item['acoustic_support']['meets_v1_128_frame_file_length']:
                continue
            raw = archive.read(item['audio_location']['value'])
            if hashlib.sha256(raw).hexdigest() != item['audio_sha256']['value']:
                raise ValueError('Example bytes changed')
            audio, sr = sf.read(io.BytesIO(raw))
            records.append({'id': item['id'], 'metadata_identifier': item['metadata_identifier']['value'],
                            'date': item['date']['value'], 'archive_folder': item['archive_folder']['value'],
                            'waveform_sha256': item['audio_sha256']['value'], 'acoustic_audit': audit_audio(audio, sr),
                            'parent_recording': None, 'session': None, 'independence_status': 'unresolved',
                            'equipment': 'Collection-level setup documented; per-file equipment continuity unverified',
                            'partition': None})
    return {
        'input_manifest_sha256': sha(manifest_path), 'records': records,
        'dates': len({r['date'] for r in records}), 'certified_independent_groups': 0,
        'publication_evidence': {'path': str(publication.relative_to(ROOT)), 'sha256': sha(publication),
                                 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12075697/',
                                 'equipment_paragraph_sha256': hashlib.sha256(equipment_evidence.encode()).hexdigest(),
                                 'collection_summary': 'Sony C-100 microphone, RME Fireface UFX II interface, Adobe Audition; 96kHz,32-bit raw WAV; same room containing three families; Dec2019-Apr2023. Collection documentation does not prove unchanged per-file acquisition or independent sessions.'},
        'pipeline_evidence': {'commit': json.loads((ROOT / 'data/marmaudio/code_tree.json').read_text())['sha'],
                              'path': str(code_path.relative_to(ROOT)), 'sha256': sha(code_path),
                              'upstream_sidecar_fields': list(fields),
                              'missing_from_released_annotation_table': [f for f in fields if f not in released],
                              'interpretation': 'Original code writes parent IDs and offsets; acquired public annotation table omits them. This does not establish that a matching sidecar exists in the release.'},
        'required_evidence': 'File-level crosswalk to original recording filename/ID and start/end sample or second offsets; session boundaries and equipment/preprocessing history. Do not infer IDs from example numeric suffixes.',
    }


def ceb_audio():
    manifest_path = ROOT / 'outputs/macde_ceb_tree_pipit_provenance_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    selected = [r for r in manifest['records'] if r['original_focal_species_matches_target']]
    repository = json.loads((ROOT / 'data/ceb/record.json').read_text())
    info = next(f for f in repository['files'] if f['key'] == 'train_xenocanto.tar.gz')
    archive_path = ROOT / 'data/ceb/train_xenocanto.tar.gz'
    partial = archive_path.with_suffix('.gz.part')
    if not archive_path.exists():
        if not partial.exists() or partial.stat().st_size != info['size']:
            raise RuntimeError('CEB archive incomplete; no waveform certification')
        algorithm, expected = info['checksum'].split(':')
        if sha(partial, algorithm) != expected:
            raise ValueError('CEB archive repository checksum mismatch')
        partial.rename(archive_path)
    algorithm, expected = info['checksum'].split(':')
    if archive_path.stat().st_size != info['size'] or sha(archive_path, algorithm) != expected:
        raise ValueError('CEB archive integrity failed')
    print('CEB archive verified; extracting fixed38 sources', flush=True)
    targets = {r['derivative_path']['value']: r for r in selected}
    folder = ROOT / 'data/ceb/tree_pipit_focal'
    folder.mkdir(parents=True, exist_ok=True)
    extracted = {}
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            suffix = '/'.join(Path(member.name).parts[-2:])
            if not member.isfile() or suffix not in targets:
                continue
            item = targets[suffix]
            identifier = item['original_identifier']['value']
            if not identifier.isdigit() or identifier in extracted:
                raise ValueError('Invalid or repeated original recording ID')
            stream = archive.extractfile(member)
            raw = stream.read()
            path = folder / f'XC{identifier}.flac'
            if path.exists() and path.read_bytes() != raw:
                raise ValueError('Existing waveform differs')
            path.write_bytes(raw)
            extracted[identifier] = (item, path, member.name)
    if len(extracted) != len(selected):
        raise RuntimeError(f'Missing selected recordings: extracted {len(extracted)}/{len(selected)}')
    old_bytes, old_pcm = {}, {}
    for path in sorted((ROOT / 'data/alp/pipit_id').rglob('*.wav')):
        old_bytes.setdefault(sha(path), []).append(str(path.relative_to(ROOT)))
        audio, sr = sf.read(path)
        old_pcm.setdefault(pcm_hash(audio, sr), []).append(str(path.relative_to(ROOT)))
    records = []
    for identifier, (item, path, member) in sorted(extracted.items()):
        audio, sr = sf.read(path)
        byte_digest, pcm_digest = sha(path), pcm_hash(audio, sr)
        record = {'id': item['id'], 'taxon': 'Anthus trivialis', 'xc_id': identifier,
                  'archive_member': member, 'local_path': str(path.relative_to(ROOT)),
                  'source_reference': item['source_reference']['value'], 'recordist': item['recordist']['value'],
                  'license': item['license']['value'], 'waveform_sha256': byte_digest,
                  'decoded_pcm_sha256': pcm_digest, 'acoustic_audit': audit_audio(audio, sr),
                  'duration_seconds': len(audio)/sr,
                  'overlap_with_local_pipit': {'exact_container_bytes': old_bytes.get(byte_digest, []),
                                              'exact_decoded_pcm': old_pcm.get(pcm_digest, []),
                                              'partial_lossy_or_resampled_overlap': 'unresolved',
                                              'source_session_crosswalk': 'unavailable'},
                  'date': None, 'parent_session': None, 'independence_status': 'unresolved', 'partition': None}
        records.append(record)
        print('CEB audited', identifier, record['acoustic_audit']['passes_acoustic_support'], flush=True)
    return {'input_manifest_sha256': sha(manifest_path), 'archive': {'path': str(archive_path.relative_to(ROOT)),
             'sha256': sha(archive_path), 'repository_checksum': info['checksum'], 'size': info['size']},
            'records': records, 'summary': {'files': len(records),
             'passes_length_and_acoustic_support': sum(r['acoustic_audit']['passes_acoustic_support'] for r in records),
             'byte_or_decoded_pcm_overlap_files': sum(bool(r['overlap_with_local_pipit']['exact_container_bytes'] or r['overlap_with_local_pipit']['exact_decoded_pcm']) for r in records),
             'certified_independent_groups': 0},
            'original_metadata_access': {'endpoint': 'https://xeno-canto.org/api/3/recordings?query=nr:133693',
                                         'http_status': 401, 'dates_and_sessions_resolved': False}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--marmaudio-only', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    result = {'code_sha256': sha(Path(__file__)), 'scope': 'Source resolution and acoustic eligibility only; no group assignment or model fits',
              'marmaudio': marm_audio(), 'stage2_ready': False, 'models_fitted': False}
    if not args.marmaudio_only:
        result['ceb_tree_pipit'] = ceb_audio()
    path = ROOT / ('outputs/macde_marmaudio_six_source_resolution.json' if args.marmaudio_only else 'outputs/macde_source_resolution.json')
    if path.exists():
        raise RuntimeError('Refusing to overwrite source-resolution result')
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Saved', path, flush=True)


if __name__ == '__main__':
    main()
