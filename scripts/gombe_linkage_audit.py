#!/usr/bin/env python3
"""Build Gombe catalogue-to-interval linkage inventory without guessing boundaries."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bioacoustic_pipeline.ingestion.gombe_linkage import riff_chunks, explicit_tape_marker, validate_link


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    source_path = ROOT / 'outputs/gombe_candidate_audit.json'
    source = json.loads(source_path.read_text())
    audio_path = ROOT / 'data/gombe/Adults_3.wav'
    expected = next(f['sha256'] for f in source['files'] if f['original_name'] == 'Adults 3.wav')
    if sha(audio_path) != expected:
        raise ValueError('Acquired waveform changed')
    with audio_path.open('rb') as stream:
        chunks = riff_chunks(stream)
    rows = []
    for record in source['catalogue']['records']:
        text = ' '.join(str(record.get(k, '')) for k in ('Individual(s) with Sound/call type', 'Context of vocalizations'))
        tape = explicit_tape_marker(text)
        row = {'catalogue_id': record['catalogue_id'],
               'macaulay_reference': 'https://macaulaylibrary.org/asset/' + record['catalogue_id'],
               'reference_access_status': 'identifier from acquired catalogue;asset not individually retrieved',
               'recordist_record_number_raw': record['Recordist record number'],
               'catalogue_date': record['parsed_date'],
               'vocalizers_raw': record['Individuals Vocalizing'],
               'documented_tape_number': tape,
               'tape_evidence': 'Explicit catalogue tape-start annotation' if tape else None,
               'candidate_wav_filename': f'Adults {tape}.wav' if tape else None,
               'sample_start': None, 'sample_end': None, 'sample_rate': None,
               'interval_verified': False, 'original_field_recording_id': None,
               'recording_event_id': None, 'session_relationship_verified': False,
               'group_id': None, 'grouping_contract': None,
               'vocal_source_mask_verified': False,
               'link_status': 'tape-start relationship documented;sample coordinates missing' if tape else 'catalogue identity only;interval and parent/session unresolved'}
        validate_link(row)
        rows.append(row)
    output = {'study': 'Gombe interval-lineage linkage inventory;independent of closed MACDE experiments',
              'code_sha256': sha(Path(__file__)),
              'contract_code_sha256': sha(ROOT / 'src/bioacoustic_pipeline/ingestion/gombe_linkage.py'),
              'source_audit_sha256': sha(source_path),
              'waveform': {'path': str(audio_path.relative_to(ROOT)), 'sha256': expected,
                           'riff_chunks': chunks,
                           'embedded_cue_chunk_present': any(c['id'] == 'cue ' for c in chunks),
                           'embedded_time_metadata_chunks': [c['id'] for c in chunks if c['id'] in ('bext', 'iXML', 'LIST')],
                           'interpretation': 'Acquired Adults3 WAV has format and audio-data chunks only;no embedded cut marker or time-metadata chunks were recovered. Other six WAVs not inspected.'},
              'records': rows,
              'summary': {'catalogue_records': len(rows),
                          'explicit_tape_start_relationships': sum(r['documented_tape_number'] is not None for r in rows),
                          'verified_sample_intervals': 0, 'verified_recording_event_links': 0,
                          'certified_independent_groups': 0},
              'evidence_limits': ['Tape-start annotations are not sample coordinates and are not propagated to neighboring catalogue rows',
                                  'A tape compilation is not a field-recording session;dates and cut numbers do not supply a session crosswalk',
                                  'No source-specific chimpanzee mask or verified announcement/edit boundaries available',
                                  'Supplied alternate acoustic counts remain unreconciled without its source checksum,code,configuration and masks'],
              'required_next_artifact': 'An authoritative cut-to-WAV/sample crosswalk, or a manually validated boundary-and-cut-number annotation set with explicit evidence, followed by original event/session relationships and a prospective grouping contract',
              'partitions_assigned': False, 'models_fitted': False, 'stage2_ready': False}
    result = ROOT / 'outputs/gombe_interval_linkage_audit.json'
    table = ROOT / 'outputs/gombe_interval_linkage_table.csv'
    if result.exists() or table.exists():
        raise RuntimeError('Refusing to overwrite linkage evidence')
    result.write_text(json.dumps(output, indent=2)+'\n')
    with table.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output['summary'], indent=2))


if __name__ == '__main__':
    main()
