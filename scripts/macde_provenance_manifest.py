#!/usr/bin/env python3
"""Build evidence-qualified candidate manifests; never assign training partitions."""
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.macde_provenance import fact, validate_manifest


AUDIT = ROOT / 'outputs/macde_acquisition_audit.json'


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def base(collection, audit):
    return {'schema_version': 1, 'source_collection': collection,
            'acquisition_audit_sha256': sha(AUDIT), 'builder_sha256': sha(Path(__file__)),
            'evidence_contract_sha256': sha(ROOT / 'src/bioacoustic_pipeline/ingestion/macde_provenance.py'),
            'scope': 'Provenance inventory of fixed acquired candidates; not a successor preregistration or confirmation of session independence',
            'original_transfer_study': 'closed, insufficient support; no modifications',
            'certified_independent_groups': 0, 'models_fitted': False, 'stage2_ready': False,
            'overlap_limits': 'Byte hashes cannot exclude transcoded/trimmed duplicates. Source-ID and parent/session crosswalks remain required.',
            'source_integrity': audit['downloads'], 'records': []}


def main():
    audit = json.loads(AUDIT.read_text())
    # Recheck the downloaded evidence on which this manifest relies.
    for item in audit['downloads']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Evidence checksum changed: ' + item['path'])
    marm = base('MarmAudio Zenodo 15017207, selected example archive', audit)
    with zipfile.ZipFile(ROOT / 'data/marmaudio/Audio_Examples.zip') as archive:
        for stem, item in sorted(audit['marmaudio']['examples'].items()):
            if not (item['metadata_link_unique'] and item['eligible_positions']):
                continue
            digest = hashlib.sha256(archive.read(item['archive_member'])).hexdigest()
            if digest != item['sha256']:
                raise ValueError('Example waveform hash changed')
            link = item['metadata_matches'][0]
            marm['records'].append({
                'id': 'zenodo:15017207:' + item['archive_member'],
                'taxon': fact('Callithrix jacchus', 'verified', 'MarmAudio collection identity; not per-file caller attribution'),
                'source_reference': fact('https://zenodo.org/records/15017207', 'verified', 'Repository metadata'),
                'audio_sha256': fact(digest, 'verified', 'Recomputed archive member bytes'),
                'audio_location': fact(item['archive_member'], 'verified', 'Verified Audio_Examples.zip'),
                'metadata_identifier': fact(link['parent_name'] + '/' + link['file_name'], 'verified', 'One annotation-table filename-stem match'),
                'waveform_equivalence_to_metadata_flac': fact(None, 'unresolved', 'Example WAV and annotation FLAC share a unique stem; original FLAC bytes not acquired'),
                'archive_folder': fact(link['parent_name'], 'verified', 'Metadata.pdf defines parent_name as archive folder, not session'),
                'parent_recording': fact(None, 'unresolved', 'No parent-recording crosswalk supplied'),
                'session': fact(None, 'unresolved', 'Calendar date does not certify an independent session'),
                'date': fact(link['date'], 'verified', 'Annotation calendar fields after unique filename linkage; original waveform time reference unverified'),
                'candidate_group': fact('date:' + link['date'], 'candidate', 'Date grouping proposal only; no independent-session certification'),
                'caller_identity': fact(None, 'unresolved', 'No per-file individual identity in annotation table'),
                'sample_rate_hz': fact(item['sample_rate'], 'verified', 'Decoded example waveform'),
                'equipment': fact(None, 'unresolved', 'No per-example equipment crosswalk'),
                'preprocessing': fact(None, 'unresolved', 'Example selection and conversion not audited against original parent waveform'),
                'original_waveform_coordinates': fact(None, 'unresolved', 'No parent time-offset mapping'),
                'selection': fact('illustrative example archive', 'verified', 'Audio_Examples archive; not a representative sample'),
                'license': fact('CC-BY-4.0', 'verified', 'Zenodo 15017207 repository license'),
                'acoustic_support': {'verified_span_positions': item['eligible_positions'],
                                     'decoded_frames_capped_256': item['frames'],
                                     'meets_v1_128_frame_file_length': item['frames'] >= 128,
                                     'meets_successor_contract': None,
                                     'note': 'Supported spans do not imply complete v1 eligibility or a frozen successor population'},
                'overlap': {'exact_bytes_match_prior_macde_audits': item['exact_byte_overlap_with_macde_audits'],
                            'parent_session_overlap': 'unresolved', 'transcoding_resistant_overlap': 'not assessed'},
                'independence_status': 'unresolved', 'partition': None,
            })
    source_rows = defaultdict(list)
    with (ROOT / 'data/ceb/train_xenocanto.csv').open() as source:
        for row in csv.DictReader(source):
            if 'Anthus trivialis' in row['scientific_name'].split('|'):
                source_rows[row['filepath']].append(row)
    ceb = base('CEB Zenodo 21821133, tree-pipit-labelled focal subset files', audit)
    for candidate in audit['ceb_tree_pipit']['candidates']:
        rows = source_rows[candidate['filepath']]
        originals = sorted({r['xc_original_scientific_name'] for r in rows})
        original_focal_match = originals == ['Anthus trivialis']
        ceb['records'].append({
            'id': 'xeno-canto:' + candidate['xc_id'],
            'taxon': fact('Anthus trivialis', 'verified', 'CEB event annotation; not necessarily original focal species'),
            'original_focal_taxa': fact(originals, 'verified', 'xc_original_scientific_name column; blank entries remain unknown'),
            'original_focal_species_matches_target': original_focal_match,
            'source_reference': fact(candidate['source_url'], 'verified', 'CEB xc_url column; direct source page not retrieved'),
            'original_identifier': fact(candidate['xc_id'], 'verified', 'CEB immutable xc_id provenance field'),
            'derivative_path': fact(candidate['filepath'], 'verified', 'CEB train_xenocanto.csv'),
            'parent_recording': fact('XC' + candidate['xc_id'], 'candidate', 'Repository recording reference; relationship to field session unresolved'),
            'session': fact(None, 'unresolved', 'Original session/date unavailable'),
            'date': fact(None, 'unresolved', 'Not in acquired focal metadata'),
            'recordist': fact(candidate['recordist'], 'verified', 'CEB xc_recordist attribution, not independence evidence'),
            'candidate_group': fact('XC' + candidate['xc_id'], 'candidate', 'Immutable reference only; do not count as independent session'),
            'equipment': fact(None, 'unresolved', 'No recorder/microphone details in acquired focal metadata'),
            'original_waveform_coordinates': fact(None, 'unresolved', 'Annotation bounds refer to derivative file; original offset/equivalence unverified'),
            'license': fact(candidate['license'], 'verified', 'Per-file CEB license field; retain individual restrictions'),
            'audio_sha256': fact(None, 'unresolved', 'No CEB waveform downloaded'),
            'acoustic_support': fact(None, 'unresolved', 'No decoded waveform; annotation duration is not acoustic-support evidence'),
            'annotation_spans_seconds': [[float(r['start_time']), float(r['end_time'])] for r in rows],
            'overlap': {'source_id': candidate['xc_id'], 'within_candidate_duplicate_xc_id': False,
                        'existing_pipit_crosswalk': 'unavailable', 'exact_hash': 'not assessed; audio absent',
                        'parent_session_overlap': 'unresolved'},
            'independence_status': 'unresolved', 'partition': None,
        })
    marm['summary'] = {'files': len(marm['records']),
                       'metadata_dates': len({r['date']['value'] for r in marm['records']}),
                       'files_with_at_least128_frames': sum(r['acoustic_support']['meets_v1_128_frame_file_length'] for r in marm['records'])}
    ceb['summary'] = {'files': len(ceb['records']),
                      'original_focal_taxon_matches': sum(r['original_focal_species_matches_target'] for r in ceb['records']),
                      'original_focal_taxa_by_file': dict(Counter(t for r in ceb['records'] for t in r['original_focal_taxa']['value']))}
    for corpus, manifest in [('marmaudio', marm), ('ceb_tree_pipit', ceb)]:
        validate_manifest(manifest)
        path = ROOT / f'outputs/macde_{corpus}_provenance_manifest.json'
        if path.exists():
            raise RuntimeError('Refusing to overwrite provenance manifest')
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        print(corpus, json.dumps(manifest['summary']), flush=True)


if __name__ == '__main__':
    main()
